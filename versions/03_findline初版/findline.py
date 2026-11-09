#巡线代码
import time, os, sys
from media.sensor import *
from media.display import *
from media.media import *

# ===================== 视觉可调参数 =====================
DISPLAY_WIDTH = 640
DISPLAY_HEIGHT = 480
# 黑线LAB阈值
BLACK_THRESHOLD = ((8, 18, -6, 3, -12, 5))
# 赛道双线固定像素间距
LINE_SPACE = 377
# 检测ROI
LEFT_ROI  = (2, 400, 180, 80)
RIGHT_ROI = (460, 400, 180, 80)
# 自定义校准中心线
CALIB_CENTER = 320

# 前方横线检测参数
HORIZONTAL_ROI = (220, 20, 220, 100)
# =====================================================

SCREEN_CENTER = DISPLAY_WIDTH // 2
ROI_CENTER_Y = LEFT_ROI[1] + LEFT_ROI[3] // 2

def init_sensor():
    sensor = Sensor()
    sensor.reset()
    sensor.set_framesize(width=DISPLAY_WIDTH, height=DISPLAY_HEIGHT)
    sensor.set_pixformat(Sensor.RGB565)
    return sensor

def init_display():
    Display.init(Display.ST7701, to_ide=True)
    MediaManager.init()

# 识别ROI下半部分的竖线，避开横边干扰
def get_line_position(img, roi, threshold):
    """
    只识别ROI下半段的竖线，避免L形横边导致的中心偏移
    """
    x, y, w, h = roi
    bottom_roi = (x, y + h//2, w, h//2)
    blobs = img.find_blobs([threshold], roi=bottom_roi, pixels_threshold=10, area_threshold=10)
    if blobs:
        max_blob = max(blobs, key=lambda b: b[4])
        return max_blob[0] + max_blob[2] // 2
    return None

# 识别前方横线
def get_horizontal_line(img, roi, threshold):
    """纯视觉找横线，只识别水平黑线，过滤竖线干扰"""
    blobs = img.find_blobs([threshold], roi=roi, pixels_threshold=30, area_threshold=30)
    if blobs:
        horizontal_blobs = [b for b in blobs if (b[2] / b[3]) > 2]
        if not horizontal_blobs:
            return None
        max_blob = max(horizontal_blobs, key=lambda b: b[1])
        return max_blob[1] + max_blob[3] // 2
    return None

# 绘图函数（移除标识符相关代码）
def draw_visual_info(img, err, ld, rd, mid_x, h_line_y):
    # 绘制检测框
    img.draw_rectangle(LEFT_ROI,  (255, 0, 0), 2)
    img.draw_rectangle(RIGHT_ROI, (0, 255, 0), 2)
    img.draw_rectangle(HORIZONTAL_ROI, (255, 255, 0), 1)

    # 参考线
    img.draw_line(SCREEN_CENTER, 0, SCREEN_CENTER, DISPLAY_HEIGHT, (0, 255, 0), 1)
    img.draw_line(CALIB_CENTER, 0, CALIB_CENTER, DISPLAY_HEIGHT, (0, 255, 255), 2)

    # 状态文本
    img.draw_string_advanced(0, 20, 40, f"E: {err:.1f}" if err is not None else "E: --", (255,255,255))
    img.draw_string_advanced(0, 80, 30, f"Ld: {ld}", (255,255,0))
    img.draw_string_advanced(200, 80, 30, f"Rd: {rd}", (255,255,0))
    if mid_x is not None:
        cali_off = mid_x - CALIB_CENTER
        img.draw_string_advanced(0, 130, 30, f"校准偏差:{cali_off}", (0,255,255))

    # 横线状态
    if h_line_y is not None:
        img.draw_string_advanced(0, 180, 30, "横线: 已检测", (255, 0, 255))
        img.draw_line(0, h_line_y, DISPLAY_WIDTH, h_line_y, (255, 0, 0), 2)
    else:
        img.draw_string_advanced(0, 180, 30, "横线: 未检测", (255, 0, 255))

    # 绘制左右赛道线（黄色）
    if ld != 999:
        line_x = SCREEN_CENTER - ld
        img.draw_line(line_x, LEFT_ROI[1], line_x, LEFT_ROI[1] + LEFT_ROI[3], (255,255,0), 2)
    if rd != 999:
        line_x = SCREEN_CENTER + rd
        img.draw_line(line_x, RIGHT_ROI[1], line_x, RIGHT_ROI[1] + RIGHT_ROI[3], (255,255,0), 2)

    # 绘制虚拟中线（蓝色）
    if mid_x is not None:
        img.draw_line(mid_x, LEFT_ROI[1], mid_x, DISPLAY_HEIGHT, (0,0,255), 2)

def draw_fps(img, fps):
    img.draw_string_advanced(520, 20, 30, f"FPS: {fps:.1f}", (255,255,255))

def main():
    try:
        sensor = init_sensor()
        init_display()
        sensor.run()
        clock = time.clock()

        while True:
            clock.tick()
            img = sensor.snapshot()

            # 获取左右黑线坐标
            left_x  = get_line_position(img, LEFT_ROI, BLACK_THRESHOLD)
            right_x = get_line_position(img, RIGHT_ROI, BLACK_THRESHOLD)

            # 获取前方横线坐标
            horizontal_line_y = get_horizontal_line(img, HORIZONTAL_ROI, BLACK_THRESHOLD)

            # 初始化参数
            Ld = 999
            Rd = 999
            virtual_mid = None
            error = None

            # 计算中线
            if left_x is not None and right_x is not None:
                Ld = SCREEN_CENTER - left_x
                Rd = right_x - SCREEN_CENTER
                virtual_mid = (left_x + right_x) // 2
            elif left_x is not None:
                Ld = SCREEN_CENTER - left_x
                fit_right_x = left_x + LINE_SPACE
                virtual_mid = (left_x + fit_right_x) // 2
            elif right_x is not None:
                Rd = right_x - SCREEN_CENTER
                fit_left_x = right_x - LINE_SPACE
                virtual_mid = (fit_left_x + right_x) // 2

            # 计算误差
            if virtual_mid is not None:
                error = virtual_mid - SCREEN_CENTER

            # 绘图展示
            draw_visual_info(img, error, Ld, Rd, virtual_mid, horizontal_line_y)
            draw_fps(img, clock.fps())
            Display.show_image(img)

    except KeyboardInterrupt:
        print("视觉程序停止")
    except Exception as e:
        print("视觉报错：", e)
    finally:
        if 'sensor' in locals():
            sensor.stop()
        Display.deinit()
        MediaManager.deinit()

if __name__ == "__main__":
    main()