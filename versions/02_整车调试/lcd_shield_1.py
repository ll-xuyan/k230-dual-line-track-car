# 【128×160 高精度版】转向标志识别
import sensor
import image
import time
import display
from pyb import UART

uart = UART(3, 115200)

# 128×160 原生分辨率适配
sensor.reset()
sensor.set_pixformat(sensor.RGB565)
sensor.set_framesize(sensor.QQVGA2)
sensor.skip_frames(time=2000)
sensor.set_auto_gain(False)
sensor.set_auto_whitebal(False)

clock = time.clock()
lcd = display.SPIDisplay()

# 优化后的颜色阈值
BLUE_BACKGROUND = (38, 54, -19, 8, -39, -10)
WHITE_ARROW = (51, 79, -24, 12, -27, 23)

# 只在上半部分找标志
SIGN_DETECT_ROI = (0, 0, 128, 80)

# 防抖计数器
left_count = 0
right_count = 0
stable_direction = "STRAIGHT"

while True:
    clock.tick()
    img = sensor.snapshot()

    current_direction = "STRAIGHT"

    # 找蓝色标志底
    blue_blobs = img.find_blobs(
        [BLUE_BACKGROUND],
        roi=SIGN_DETECT_ROI,
        pixels_threshold=80,
        area_threshold=80,
        merge=True
    )

    if blue_blobs:
        blue_sign = max(blue_blobs, key=lambda b: b.pixels())

        # 几何校验1：必须是圆形
        if blue_sign.roundness() < 0.7:
            continue

        img.draw_rectangle(blue_sign.rect(), color=(0,255,0), thickness=1)

        # 找白色箭头
        arrow_blobs = img.find_blobs(
            [WHITE_ARROW],
            roi=blue_sign.rect(),
            pixels_threshold=30,
            area_threshold=30,
            merge=True
        )

        if arrow_blobs:
            arrow = max(arrow_blobs, key=lambda b: b.pixels())

            # 几何校验2：必须是水平箭头
            if arrow.w() / arrow.h() < 1.2:
                continue

            # 面积占比校验
            arrow_area_ratio = arrow.area() / blue_sign.area()
            if arrow_area_ratio < 0.1 or arrow_area_ratio > 0.5:
                continue

            img.draw_rectangle(arrow.rect(), color=(255,0,0), thickness=1)

            # 方向判断
            arrow_center_x = arrow.cx()
            blue_center_x = blue_sign.cx()

            if arrow_center_x < blue_center_x - 3:
                current_direction = "LEFT"
                img.draw_string(2, 2, "LEFT", color=(0,255,0), scale=1)
            elif arrow_center_x > blue_center_x + 3:
                current_direction = "RIGHT"
                img.draw_string(2, 2, "RIGHT", color=(255,0,0), scale=1)

    # 防抖处理
    if current_direction == "LEFT":
        left_count += 1
        right_count = 0
        if left_count >= 3:
            stable_direction = "LEFT"
    elif current_direction == "RIGHT":
        right_count += 1
        left_count = 0
        if right_count >= 3:
            stable_direction = "RIGHT"
    else:
        left_count = 0
        right_count = 0
        stable_direction = "STRAIGHT"

    # 串口输出稳定方向
    uart.write(f"{stable_direction}\n")

    # LCD全屏显示
    lcd.write(img)

    print("稳定方向：", stable_direction, "当前方向：", current_direction, "FPS：", clock.fps())
