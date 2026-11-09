#最终成品代码
# -*- coding: utf-8 -*-
import os, gc, time

from libs.PlatTasks import DetectionApp
from libs.PipeLine import PipeLine
from libs.Utils import *

from ybUtils.YbUart import YbUart


# =====================================================
# 串口配置
# =====================================================
uart = YbUart(baudrate=115200)

FRAME_HEAD = 0x23

# True : [0x23, dir, light, err_H, err_L, checksum]
# False: [0x23, dir, light, 0, 0, checksum]
SEND_LINE_ERROR = True

# 调试打印很拖帧，正式跑车建议 False
PRINT_DEBUG = False

# 如果丢线，短时间内保持上一次误差，避免突然变 0
USE_LAST_ERROR_WHEN_LOST = True
MAX_LOST_KEEP_FRAMES = 5


# =====================================================
# 灰度巡线参数
# =====================================================
BASE_WIDTH = 640
BASE_HEIGHT = 480

# 灰度阈值：灰度值小于该值，认为是黑线
# 黑线识别不到：调大，例如 60、70、80
# 背景也被识别：调小，例如 45、40
BLACK_GRAY_THRESHOLD = 45

# 横向扫描步长，越小越准但越慢
SCAN_X_STEP = 3

# 横线检测扫描步长
SCAN_H_STEP = 3

# 每一条采样线上至少检测到多少个黑点，才认为有效
MIN_LINE_POINTS = 5

# 赛道双线固定像素间距，基于 640x480 标定
LINE_SPACE = 377

# ROI 坐标，基于 640x480 标定
LEFT_ROI_BASE = (2, 400, 180, 80)
RIGHT_ROI_BASE = (460, 400, 180, 80)
HORIZONTAL_ROI_BASE = (220, 20, 220, 100)

CALIB_CENTER_BASE = 320


# =====================================================
# E 值缩放和平滑参数
# =====================================================

# E 值缩放比例：35 / 100 = 0.35
# 原始 E = 100，处理后目标 E 约等于 35
ERROR_GAIN_NUM = 25
ERROR_GAIN_DEN = 100

# 低通滤波比例：20 / 100 = 0.20
# 越小越平滑，变化越慢
ERROR_FILTER_NUM = 20
ERROR_FILTER_DEN = 100

# 单帧最大变化量
# 越小，E 的变化率越小
ERROR_MAX_STEP = 4


# =====================================================
# 显示与绘图开关
# =====================================================
DRAW_AI_BOX = True
DRAW_LINE_INFO = True


# =====================================================
# AI 模型配置
# =====================================================
display_mode = "lt9611"

# 保留你原来的模型输入画面尺寸
rgb888p_size = [1280, 720]

root_path = "/sdcard/mp_deployment_source/"

deploy_conf = read_json(root_path + "/deploy_config.json")
kmodel_path = root_path + deploy_conf["kmodel_path"]

raw_labels = deploy_conf["categories"]
if isinstance(raw_labels, str):
    labels = raw_labels.split(",")
else:
    labels = list(raw_labels)

print("=== 模型标签列表 ===")
print(labels)
print("====================")

confidence_threshold = deploy_conf["confidence_threshold"]
nms_threshold = deploy_conf["nms_threshold"]
model_input_size = deploy_conf["img_size"]
nms_option = deploy_conf["nms_option"]
model_type = deploy_conf["model_type"]

anchors = []
if model_type == "AnchorBaseDet":
    anchors = (
        deploy_conf["anchors"][0]
        + deploy_conf["anchors"][1]
        + deploy_conf["anchors"][2]
    )

inference_mode = "video"
debug_mode = 0


# =====================================================
# 串口打包
# =====================================================
def clamp_int16(value):
    if value is None:
        value = 0

    value = int(value)

    if value > 32767:
        value = 32767

    if value < -32768:
        value = -32768

    return value


def pack_uart(dir_cmd=0, light=0, line_error=0):
    """
    6 字节串口帧：
    [0] 0x23
    [1] dir
        0: 无方向
        1: 左转
        2: 右转

    [2] light
        0: 无灯
        1: 红灯
        2: 黄灯
        3: 绿灯

    [3] line_error 高 8 位
    [4] line_error 低 8 位
    [5] checksum，前 5 字节累加低 8 位
    """

    if SEND_LINE_ERROR:
        err = clamp_int16(line_error) & 0xFFFF
        err_h = (err >> 8) & 0xFF
        err_l = err & 0xFF
    else:
        err_h = 0
        err_l = 0

    frame = [
        FRAME_HEAD,
        dir_cmd & 0xFF,
        light & 0xFF,
        err_h,
        err_l
    ]

    checksum = sum(frame) & 0xFF
    frame.append(checksum)

    return bytes(frame)


# =====================================================
# E 处理函数：缩小 + 平滑 + 限制变化率
# =====================================================
def process_error(raw_error, last_error):
    """
    raw_error:
        原始巡线误差

    last_error:
        上一帧处理后的误差

    返回：
        处理后的误差
    """

    if raw_error is None:
        return None

    raw_error = int(raw_error)
    last_error = int(last_error)

    # 1. 缩小 E 值
    target_error = raw_error * ERROR_GAIN_NUM // ERROR_GAIN_DEN

    # 2. 低通滤波
    filtered_error = last_error + (
        (target_error - last_error) * ERROR_FILTER_NUM // ERROR_FILTER_DEN
    )

    # 3. 限制单帧最大变化量
    delta = filtered_error - last_error

    if delta > ERROR_MAX_STEP:
        filtered_error = last_error + ERROR_MAX_STEP
    elif delta < -ERROR_MAX_STEP:
        filtered_error = last_error - ERROR_MAX_STEP

    return int(filtered_error)


# =====================================================
# 图像格式判断
# =====================================================
def get_img_info(img):
    """
    PipeLine.get_frame() 通常返回 RGB888P，shape 为 [3, H, W]

    layout:
        0 = CHW, [3, H, W]
        1 = HWC, [H, W, 3]
    """
    shape = img.shape

    if len(shape) == 3 and shape[0] == 3:
        cam_h = shape[1]
        cam_w = shape[2]
        layout = 0
        return cam_w, cam_h, layout

    if len(shape) == 3 and shape[2] == 3:
        cam_h = shape[0]
        cam_w = shape[1]
        layout = 1
        return cam_w, cam_h, layout

    return rgb888p_size[0], rgb888p_size[1], 0


def scale_roi(roi, sx, sy):
    x, y, w, h = roi

    return (
        int(x * sx),
        int(y * sy),
        int(w * sx),
        int(h * sy)
    )


def limit_roi(roi, cam_w, cam_h):
    x, y, w, h = roi

    if x < 0:
        x = 0
    if y < 0:
        y = 0

    if x >= cam_w:
        x = cam_w - 1
    if y >= cam_h:
        y = cam_h - 1

    if x + w > cam_w:
        w = cam_w - x
    if y + h > cam_h:
        h = cam_h - y

    if w < 1:
        w = 1
    if h < 1:
        h = 1

    return x, y, w, h


# =====================================================
# 灰度巡线核心
# =====================================================
def get_vertical_line_position_from_gray(img, roi, layout):
    """
    快速灰度巡线：
    不扫整个 ROI，只扫 ROI 下半部分的几条横向采样线。
    每条线上找最长黑色连续段，用连续段中心作为黑线位置。
    """
    x, y, w, h = roi

    scan_ys = [
        y + h * 2 // 3,
        y + h * 3 // 4,
        y + h * 7 // 8
    ]

    center_sum = 0
    weight_sum = 0

    x_end = x + w

    if layout == 0:
        for yy in scan_ys:
            in_run = False
            run_start = 0
            run_count = 0

            best_start = 0
            best_count = 0

            for xx in range(x, x_end, SCAN_X_STEP):
                r = int(img[0, yy, xx])
                g = int(img[1, yy, xx])
                b = int(img[2, yy, xx])

                # 快速灰度：约等于 0.25R + 0.5G + 0.25B
                gray = (r + (g << 1) + b) >> 2

                if gray < BLACK_GRAY_THRESHOLD:
                    if not in_run:
                        in_run = True
                        run_start = xx
                        run_count = 1
                    else:
                        run_count += 1
                else:
                    if in_run:
                        if run_count > best_count:
                            best_count = run_count
                            best_start = run_start

                    in_run = False
                    run_count = 0

            if in_run:
                if run_count > best_count:
                    best_count = run_count
                    best_start = run_start

            if best_count >= MIN_LINE_POINTS:
                center_x = best_start + (best_count * SCAN_X_STEP) // 2
                center_sum += center_x * best_count
                weight_sum += best_count

    else:
        for yy in scan_ys:
            in_run = False
            run_start = 0
            run_count = 0

            best_start = 0
            best_count = 0

            for xx in range(x, x_end, SCAN_X_STEP):
                r = int(img[yy, xx, 0])
                g = int(img[yy, xx, 1])
                b = int(img[yy, xx, 2])

                gray = (r + (g << 1) + b) >> 2

                if gray < BLACK_GRAY_THRESHOLD:
                    if not in_run:
                        in_run = True
                        run_start = xx
                        run_count = 1
                    else:
                        run_count += 1
                else:
                    if in_run:
                        if run_count > best_count:
                            best_count = run_count
                            best_start = run_start

                    in_run = False
                    run_count = 0

            if in_run:
                if run_count > best_count:
                    best_count = run_count
                    best_start = run_start

            if best_count >= MIN_LINE_POINTS:
                center_x = best_start + (best_count * SCAN_X_STEP) // 2
                center_sum += center_x * best_count
                weight_sum += best_count

    if weight_sum <= 0:
        return None

    return center_sum // weight_sum


def get_horizontal_line_from_gray(img, roi, layout):
    """
    灰度横线检测：
    每隔几行扫描一次，找黑点最多的行。
    """
    x, y, w, h = roi

    best_y = None
    best_count = 0

    x_end = x + w
    y_end = y + h

    if layout == 0:
        for yy in range(y, y_end, SCAN_H_STEP):
            black_count = 0

            for xx in range(x, x_end, SCAN_H_STEP):
                r = int(img[0, yy, xx])
                g = int(img[1, yy, xx])
                b = int(img[2, yy, xx])

                gray = (r + (g << 1) + b) >> 2

                if gray < BLACK_GRAY_THRESHOLD:
                    black_count += 1

            if black_count > best_count:
                best_count = black_count
                best_y = yy

    else:
        for yy in range(y, y_end, SCAN_H_STEP):
            black_count = 0

            for xx in range(x, x_end, SCAN_H_STEP):
                r = int(img[yy, xx, 0])
                g = int(img[yy, xx, 1])
                b = int(img[yy, xx, 2])

                gray = (r + (g << 1) + b) >> 2

                if gray < BLACK_GRAY_THRESHOLD:
                    black_count += 1

            if black_count > best_count:
                best_count = black_count
                best_y = yy

    min_black_count = max(6, (w // SCAN_H_STEP) // 3)

    if best_count >= min_black_count:
        return best_y

    return None


def get_line_state(img, cam_w, cam_h, layout):
    sx = cam_w / BASE_WIDTH
    sy = cam_h / BASE_HEIGHT

    left_roi = limit_roi(scale_roi(LEFT_ROI_BASE, sx, sy), cam_w, cam_h)
    right_roi = limit_roi(scale_roi(RIGHT_ROI_BASE, sx, sy), cam_w, cam_h)
    horizontal_roi = limit_roi(scale_roi(HORIZONTAL_ROI_BASE, sx, sy), cam_w, cam_h)

    screen_center = cam_w // 2
    line_space = int(LINE_SPACE * sx)

    left_x = get_vertical_line_position_from_gray(img, left_roi, layout)
    right_x = get_vertical_line_position_from_gray(img, right_roi, layout)
    horizontal_line_y = get_horizontal_line_from_gray(img, horizontal_roi, layout)

    Ld = 999
    Rd = 999
    virtual_mid = None
    error = None

    if left_x is not None and right_x is not None:
        Ld = int((screen_center - left_x) / sx)
        Rd = int((right_x - screen_center) / sx)
        virtual_mid = (left_x + right_x) // 2

    elif left_x is not None:
        Ld = int((screen_center - left_x) / sx)
        fit_right_x = left_x + line_space
        virtual_mid = (left_x + fit_right_x) // 2

    elif right_x is not None:
        Rd = int((right_x - screen_center) / sx)
        fit_left_x = right_x - line_space
        virtual_mid = (fit_left_x + right_x) // 2

    if virtual_mid is not None:
        # 原始 E，基于 640x480 坐标尺度
        error = int((virtual_mid - screen_center) / sx)

    return {
        "left_x": left_x,
        "right_x": right_x,
        "Ld": Ld,
        "Rd": Rd,
        "virtual_mid": virtual_mid,
        "error": error,
        "horizontal_line_y": horizontal_line_y,
        "left_roi": left_roi,
        "right_roi": right_roi,
        "horizontal_roi": horizontal_roi,
        "sx": sx,
        "sy": sy
    }


# =====================================================
# AI 识别结果解析
# =====================================================
def parse_detection_result(res, labels):
    dir_cmd = 0
    light = 0
    detected_names = []

    idx_array = res.get("idx", [])

    for label_id in idx_array:
        label_id = int(label_id)

        if 0 <= label_id < len(labels):
            obj_label = labels[label_id].strip()
            detected_names.append(obj_label)

    if "left" in detected_names:
        dir_cmd = 1
    elif "right" in detected_names:
        dir_cmd = 2

    if "red" in detected_names:
        light = 1
    elif "yellow" in detected_names:
        light = 2
    elif "green" in detected_names:
        light = 3

    return dir_cmd, light, detected_names


# =====================================================
# 绘图函数
# =====================================================
def draw_line_info(osd_img, line_state, cam_w, cam_h, display_w, display_h, fps, detected_names):
    dx = display_w / cam_w
    dy = display_h / cam_h

    def mx(v):
        return int(v * dx)

    def my(v):
        return int(v * dy)

    def mroi(roi):
        x, y, w, h = roi
        return (
            mx(x),
            my(y),
            mx(w),
            my(h)
        )

    left_roi = line_state["left_roi"]
    right_roi = line_state["right_roi"]
    horizontal_roi = line_state["horizontal_roi"]

    left_x = line_state["left_x"]
    right_x = line_state["right_x"]
    virtual_mid = line_state["virtual_mid"]
    horizontal_line_y = line_state["horizontal_line_y"]

    error = line_state["error"]
    raw_error = line_state.get("raw_error", None)

    Ld = line_state["Ld"]
    Rd = line_state["Rd"]

    screen_center = cam_w // 2
    calib_center = int(CALIB_CENTER_BASE * line_state["sx"])

    osd_img.draw_rectangle(mroi(left_roi), (255, 0, 0), 2)
    osd_img.draw_rectangle(mroi(right_roi), (0, 255, 0), 2)
    osd_img.draw_rectangle(mroi(horizontal_roi), (255, 255, 0), 1)

    osd_img.draw_line(mx(screen_center), 0, mx(screen_center), display_h, (0, 255, 0), 1)
    osd_img.draw_line(mx(calib_center), 0, mx(calib_center), display_h, (0, 255, 255), 2)

    if left_x is not None:
        osd_img.draw_line(
            mx(left_x),
            my(left_roi[1]),
            mx(left_x),
            my(left_roi[1] + left_roi[3]),
            (255, 255, 0),
            2
        )

    if right_x is not None:
        osd_img.draw_line(
            mx(right_x),
            my(right_roi[1]),
            mx(right_x),
            my(right_roi[1] + right_roi[3]),
            (255, 255, 0),
            2
        )

    if virtual_mid is not None:
        osd_img.draw_line(
            mx(virtual_mid),
            my(left_roi[1]),
            mx(virtual_mid),
            display_h,
            (0, 0, 255),
            2
        )

    if horizontal_line_y is not None:
        osd_img.draw_line(
            0,
            my(horizontal_line_y),
            display_w,
            my(horizontal_line_y),
            (255, 0, 255),
            2
        )

    osd_img.draw_string_advanced(
        0,
        20,
        28,
        "E: {}".format(error if error is not None else "--"),
        (255, 255, 255)
    )

    osd_img.draw_string_advanced(
        0,
        58,
        22,
        "RawE: {}".format(raw_error if raw_error is not None else "--"),
        (200, 200, 200)
    )

    osd_img.draw_string_advanced(
        0,
        88,
        22,
        "Ld: {}  Rd: {}".format(Ld, Rd),
        (255, 255, 0)
    )

    osd_img.draw_string_advanced(
        0,
        118,
        22,
        "FPS: {:.1f}".format(fps),
        (255, 255, 255)
    )

    if horizontal_line_y is not None:
        h_text = "H-Line: YES"
    else:
        h_text = "H-Line: NO"

    osd_img.draw_string_advanced(
        0,
        148,
        22,
        h_text,
        (255, 0, 255)
    )

    if detected_names:
        ai_text = "AI: " + ",".join(detected_names)
    else:
        ai_text = "AI: none"

    osd_img.draw_string_advanced(
        0,
        178,
        22,
        ai_text,
        (0, 255, 255)
    )


# =====================================================
# 主程序
# =====================================================
pl = None
det_app = None

try:
    pl = PipeLine(
        rgb888p_size=rgb888p_size,
        display_mode=display_mode
    )

    pl.create()

    display_size = pl.get_display_size()
    display_w = display_size[0]
    display_h = display_size[1]

    det_app = DetectionApp(
        inference_mode,
        kmodel_path,
        labels,
        model_input_size,
        anchors,
        model_type,
        confidence_threshold,
        nms_threshold,
        rgb888p_size,
        display_size,
        debug_mode=debug_mode
    )

    det_app.config_preprocess()

    clock = time.clock()

    frame_count = 0
    last_valid_error = 0
    lost_count = 0

    cam_w = rgb888p_size[0]
    cam_h = rgb888p_size[1]
    layout = 0

    while True:
        clock.tick()
        frame_count += 1

        img = pl.get_frame()

        if frame_count == 1:
            cam_w, cam_h, layout = get_img_info(img)
            print("Image size:", cam_w, cam_h, "layout:", layout)

        try:
            pl.osd_img.clear()
        except Exception:
            pass

        # -----------------------------
        # 1. AI 识别
        # -----------------------------
        res = det_app.run(img)
        dir_cmd, light, detected_names = parse_detection_result(res, labels)

        # -----------------------------
        # 2. 灰度巡线
        # -----------------------------
        line_state = get_line_state(img, cam_w, cam_h, layout)

        # 原始 E
        raw_error = line_state["error"]
        line_state["raw_error"] = raw_error

        # 缩小、平滑、限制变化率后的 E
        line_error = process_error(raw_error, last_valid_error)

        if line_error is not None:
            last_valid_error = line_error
            lost_count = 0
            send_error = line_error
            line_state["error"] = line_error

        else:
            lost_count += 1

            if USE_LAST_ERROR_WHEN_LOST and lost_count <= MAX_LOST_KEEP_FRAMES:
                send_error = last_valid_error
                line_state["error"] = last_valid_error
            else:
                send_error = 0
                line_state["error"] = 0

        # -----------------------------
        # 3. 绘制显示
        # -----------------------------
        if DRAW_AI_BOX:
            det_app.draw_result(pl.osd_img, res)

        if DRAW_LINE_INFO:
            draw_line_info(
                pl.osd_img,
                line_state,
                cam_w,
                cam_h,
                display_w,
                display_h,
                clock.fps(),
                detected_names
            )

        pl.show_image()

        # -----------------------------
        # 4. 串口发送
        # -----------------------------
        send_frame = pack_uart(
            dir_cmd=dir_cmd,
            light=light,
            line_error=send_error
        )

        uart.send(send_frame)

        if PRINT_DEBUG:
            print(
                "dir:",
                dir_cmd,
                "light:",
                light,
                "raw_error:",
                raw_error,
                "send_error:",
                send_error,
                "labels:",
                detected_names,
                "frame:",
                send_frame.hex()
            )

        if frame_count % 20 == 0:
            gc.collect()

except KeyboardInterrupt:
    print("\n程序停止")

except Exception as e:
    print("程序报错：", e)

finally:
    try:
        uart.deinit()
    except Exception:
        pass

    try:
        if det_app is not None:
            det_app.deinit()
    except Exception:
        pass

    try:
        if pl is not None:
            pl.destroy()
    except Exception:
        pass

    print("资源已释放")