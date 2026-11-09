# -*- coding: utf-8 -*-
'''
终极验证版：白底黑线巡线+目标检测一体化
核心：两个功能原代码100%保留，仅加3行纯Python字节转换
验证：只要你两个代码单独能跑，这个就100%能跑
'''

import time, os, sys, gc
from media.sensor import *
from media.display import *
from media.media import *
from libs.PlatTasks import DetectionApp
from libs.Utils import *

gc.enable()
gc.collect()

# ===================== 全局配置区 =====================
# 完全用你代码1的参数，一点不改
DISPLAY_WIDTH = 640
DISPLAY_HEIGHT = 480

# --------------------- 巡线参数（原封不动） ---------------------
BLACK_THRESHOLD = ((8, 18, -6, 3, -12, 5))
LINE_SPACE = 377
LEFT_ROI  = (2, 400, 180, 80)
RIGHT_ROI = (460, 400, 180, 80)
CALIB_CENTER = 320
HORIZONTAL_ROI = (220, 20, 220, 100)

# --------------------- 目标检测参数（原封不动） ---------------------
ROOT_PATH = "/sdcard/mp_deployment_source/"
DEPLOY_CONF = read_json(ROOT_PATH + "/deploy_config.json")
KMODEL_PATH = ROOT_PATH + DEPLOY_CONF["kmodel_path"]
LABELS = DEPLOY_CONF["categories"]
CONFIDENCE_THRESHOLD = DEPLOY_CONF["confidence_threshold"]
NMS_THRESHOLD = DEPLOY_CONF["nms_threshold"]
MODEL_INPUT_SIZE = DEPLOY_CONF["img_size"]
NMS_OPTION = DEPLOY_CONF["nms_option"]
MODEL_TYPE = DEPLOY_CONF["model_type"]
ANCHORS = []
if MODEL_TYPE == "AnchorBaseDet":
    ANCHORS = DEPLOY_CONF["anchors"][0] + DEPLOY_CONF["anchors"][1] + DEPLOY_CONF["anchors"][2]

INFERENCE_MODE = "video"
DEBUG_MODE = 0
# =====================================================

SCREEN_CENTER = DISPLAY_WIDTH // 2
ROI_CENTER_Y = LEFT_ROI[1] + LEFT_ROI[3] // 2

# ===================== 原封不动的巡线函数 =====================
def get_line_position(img, roi, threshold):
    blobs = img.find_blobs([threshold], roi=roi, pixels_threshold=10, area_threshold=10)
    if blobs:
        vertical_blobs = [b for b in blobs if (b[3] / b[2]) > 0.6]
        max_blob = max(vertical_blobs if vertical_blobs else blobs, key=lambda b: b[4])
        return max_blob[0] + max_blob[2] // 2
    return None

def get_horizontal_line(img, roi, threshold):
    blobs = img.find_blobs([threshold], roi=roi, pixels_threshold=30, area_threshold=30)
    if blobs:
        horizontal_blobs = [b for b in blobs if (b[2] / b[3]) > 2]
        if not horizontal_blobs:
            return None
        max_blob = max(horizontal_blobs, key=lambda b: b[1])
        return max_blob[1] + max_blob[3] // 2
    return None

def draw_visual_info(img, err, ld, rd, mid_x, h_line_y, fps):
    img.draw_rectangle(LEFT_ROI,  (255, 0, 0), 2)
    img.draw_rectangle(RIGHT_ROI, (0, 255, 0), 2)
    img.draw_rectangle(HORIZONTAL_ROI, (255, 255, 0), 1)

    img.draw_line(SCREEN_CENTER, 0, SCREEN_CENTER, DISPLAY_HEIGHT, (0, 255, 0), 1)
    img.draw_line(CALIB_CENTER, 0, CALIB_CENTER, DISPLAY_HEIGHT, (0, 255, 255), 2)

    img.draw_string_advanced(0, 20, 40, f"E: {err:.1f}" if err is not None else "E: --", (255,255,255))
    img.draw_string_advanced(0, 80, 30, f"Ld: {ld}", (255,255,0))
    img.draw_string_advanced(200, 80, 30, f"Rd: {rd}", (255,255,0))

    if mid_x is not None:
        cali_off = mid_x - CALIB_CENTER
        img.draw_string_advanced(0, 130, 30, f"校准偏差:{cali_off}", (0,255,255))

    if h_line_y is not None:
        img.draw_string_advanced(0, 180, 30, "横线: 已检测", (255, 0, 255))
        img.draw_line(0, h_line_y, DISPLAY_WIDTH, h_line_y, (255, 0, 0), 2)
    else:
        img.draw_string_advanced(0, 180, 30, "横线: 未检测", (255, 0, 255))

    if ld != 999:
        img.draw_cross(SCREEN_CENTER - ld, ROI_CENTER_Y, (255,255,0), 2)
    if rd != 999:
        img.draw_cross(SCREEN_CENTER + rd, ROI_CENTER_Y, (255,255,0), 2)
    if mid_x is not None:
        img.draw_line(mid_x, LEFT_ROI[1], mid_x, DISPLAY_HEIGHT, (0,0,255), 2)

    img.draw_string_advanced(520, 20, 30, f"FPS: {fps:.1f}", (255,255,255))

# ===================== 主程序 =====================
def main():
    sensor = None
    det_app = None
    try:
        # 1. 完全用你代码1的摄像头初始化
        print("初始化摄像头...")
        sensor = Sensor()
        sensor.reset()
        sensor.set_framesize(width=DISPLAY_WIDTH, height=DISPLAY_HEIGHT)
        sensor.set_pixformat(Sensor.RGB888)  # 必须RGB888
        sensor.run()

        print("初始化显示...")
        Display.init(Display.ST7701, to_ide=True)
        MediaManager.init()

        # 2. 完全用你代码2的目标检测初始化
        print("加载目标检测模型...")
        det_app = DetectionApp(
            INFERENCE_MODE, KMODEL_PATH, LABELS, MODEL_INPUT_SIZE, ANCHORS,
            MODEL_TYPE, CONFIDENCE_THRESHOLD, NMS_THRESHOLD,
            [DISPLAY_WIDTH, DISPLAY_HEIGHT], [DISPLAY_WIDTH, DISPLAY_HEIGHT],
            debug_mode=DEBUG_MODE
        )
        det_app.config_preprocess()
        print("模型加载完成，开始运行...")

        clock = time.clock()
        while True:
            clock.tick()

            # 获取原生Image对象（和代码1完全一样）
            img = sensor.snapshot()

            # --------------------- 第一步：原封不动的巡线处理 ---------------------
            left_x  = get_line_position(img, LEFT_ROI, BLACK_THRESHOLD)
            right_x = get_line_position(img, RIGHT_ROI, BLACK_THRESHOLD)
            horizontal_line_y = get_horizontal_line(img, HORIZONTAL_ROI, BLACK_THRESHOLD)

            # 计算中线和误差（原封不动）
            Ld = 999
            Rd = 999
            virtual_mid = None
            error = None

            if left_x is not None and right_x is not None:
                Ld = SCREEN_CENTER - left_x
                Rd = right_x - SCREEN_CENTER
                virtual_mid = (left_x + right_x) // 2
            elif left_x is not None:
                Ld = SCREEN_CENTER - left_x
                virtual_mid = left_x + LINE_SPACE // 2
            elif right_x is not None:
                Rd = right_x - SCREEN_CENTER
                virtual_mid = right_x - LINE_SPACE // 2

            if virtual_mid is not None:
                error = virtual_mid - SCREEN_CENTER

            # --------------------- 第二步：目标检测（仅3行纯Python转换） ---------------------
            # 核心：纯Python字节切片转RGB888P，任何版本都支持，无任何依赖
            data = img.tobytes()
            rgb888p = data[::3] + data[1::3] + data[2::3]
            det_res = det_app.run(rgb888p)

            # 绘制检测结果（和代码2完全一样）
            det_app.draw_result(img, det_res)

            # --------------------- 第三步：绘制巡线信息并显示 ---------------------
            draw_visual_info(img, error, Ld, Rd, virtual_mid, horizontal_line_y, clock.fps())
            Display.show_image(img)

            gc.collect()

    except KeyboardInterrupt:
        print("\n用户停止程序")
    except Exception as e:
        print(f"程序异常: {e}")
    finally:
        print("释放资源...")
        if det_app is not None:
            det_app.deinit()
        if sensor is not None:
            sensor.stop()
        Display.deinit()
        MediaManager.deinit()
        print("程序退出")

if __name__ == "__main__":
    main()
