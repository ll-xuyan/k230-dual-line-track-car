import time, os, sys
from media.sensor import *
from media.display import *
from media.media import *

# ===================== 视觉可调参数 =====================
DISPLAY_WIDTH = 640
DISPLAY_HEIGHT = 480
# 黑线LAB阈值 不动
BLACK_THRESHOLD = ((8, 18, -6, 3, -12, 5))
# 赛道双线固定像素间距 不动
LINE_SPACE = 377
# 检测ROI 完全不动
LEFT_ROI  = (2, 400, 180, 80)
RIGHT_ROI = (460, 400, 180, 80)
# 自定义校准中心线 不动
CALIB_CENTER = 320

# 【新增】前方横线检测参数（和原有ROI完全不重叠）
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

# 【核心修改】放宽识别条件，兼容L形拐角黑线
def get_line_position(img, roi, threshold):
    """纯视觉找黑线，兼容拐角L形黑线，不过度依赖高宽比过滤"""
    # 1. 大幅降低噪点过滤门槛，识别更敏感
    blobs = img.find_blobs([threshold], roi=roi, pixels_threshold=10, area_threshold=10)
    if blobs:
        # 2. 放宽高宽比要求，同时做兜底兼容
        vertical_blobs = [b for b in blobs if (b[3] / b[2]) > 0.6]

        # 3. 如果过滤后没有竖线，直接取框内最大的黑线（兜底逻辑，解决拐角识别）
        if not vertical_blobs:
            max_blob = max(blobs, key=lambda b: b[4])
        else:
            max_blob = max(vertical_blobs, key=lambda b: b[4])

        return max_blob[0] + max_blob[2] // 2
    return None

# 【新增】专门识别前方横线的函数
def get_horizontal_line(img, roi, threshold):
    """纯视觉找横线，只识别水平黑线，过滤竖线干扰"""
    blobs = img.find_blobs([threshold], roi=roi, pixels_threshold=30, area_threshold=30)
    if blobs:
        # 宽高比大于2（横线：宽度是高度的2倍以上）
        horizontal_blobs = [b for b in blobs if (b[2] / b[3]) > 2]
        if not horizontal_blobs:
            return None
        # 取最靠下的横线（也就是离你最近的那条）
        max_blob = max(horizontal_blobs, key=lambda b: b[1])
        # 返回横线的中心y坐标（用来标记位置）
        return max_blob[1] + max_blob[3] // 2
    return None

def draw_visual_info(img, err, ld, rd, mid_x, h_line_y):
    # 绘制左右检测框
    img.draw_rectangle(LEFT_ROI,  (255, 0, 0), 2)
    img.draw_rectangle(RIGHT_ROI, (0, 255, 0), 2)
    # 【新增】绘制横线检测框（黄色，和其他框区分）
    img.draw_rectangle(HORIZONTAL_ROI, (255, 255, 0), 1)

    # 1. 屏幕中心参考线 绿色
    img.draw_line(SCREEN_CENTER, 0, SCREEN_CENTER, DISPLAY_HEIGHT, (0, 255, 0), 1)
    # 2. 校准基准中心线 青色（用来对标校准）
    img.draw_line(CALIB_CENTER, 0, CALIB_CENTER, DISPLAY_HEIGHT, (0, 255, 255), 2)

    # 显示误差、左右距离
    img.draw_string_advanced(0, 20, 40, f"E: {err:.1f}" if err is not None else "E: --", (255,255,255))
    img.draw_string_advanced(0, 80, 30, f"Ld: {ld}", (255,255,0))
    img.draw_string_advanced(200, 80, 30, f"Rd: {rd}", (255,255,0))
    # 显示校准偏差
    if mid_x is not None:
        cali_off = mid_x - CALIB_CENTER
        img.draw_string_advanced(0, 130, 30, f"校准偏差:{cali_off}", (0,255,255))

    # 【新增】显示横线状态
    if h_line_y is not None:
        img.draw_string_advanced(0, 180, 30, "横线: 已检测", (255, 0, 255))
        # 画一条红色横线标记位置
        img.draw_line(0, h_line_y, DISPLAY_WIDTH, h_line_y, (255, 0, 0), 2)
    else:
        img.draw_string_advanced(0, 180, 30, "横线: 未检测", (255, 0, 255))

    # 绘制左线标记
    if ld != 999:
        img.draw_cross(SCREEN_CENTER - ld, ROI_CENTER_Y, (255,255,0), 2)
    # 绘制右线标记
    if rd != 999:
        img.draw_cross(SCREEN_CENTER + rd, ROI_CENTER_Y, (255,255,0), 2)
    # 绘制拟合/真实虚拟中线 蓝色
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

            # 原有逻辑：视觉获取左右黑线坐标
            left_x  = get_line_position(img, LEFT_ROI, BLACK_THRESHOLD)
            right_x = get_line_position(img, RIGHT_ROI, BLACK_THRESHOLD)

            # 【新增】获取前方横线坐标
            horizontal_line_y = get_horizontal_line(img, HORIZONTAL_ROI, BLACK_THRESHOLD)

            # 初始化：默认丢失赋值999
            Ld = 999
            Rd = 999
            virtual_mid = None
            error = None

            # 原有逻辑：计算中线
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

            # 计算中线相对屏幕中心误差
            if virtual_mid is not None:
                error = virtual_mid - SCREEN_CENTER

            # 纯视觉绘图展示（新增横线状态）
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
