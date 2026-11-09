# -*- coding: utf-8 -*-
import os, gc, time

from libs.PlatTasks import DetectionApp
from libs.PipeLine import PipeLine
from libs.Utils import *

from ybUtils.YbUart import YbUart
from ybUtils.YbBuzzer import YbBuzzer


# =====================================================
# 串口配置
# =====================================================
uart = YbUart(baudrate=115200)

# 创建蜂鸣器实例
buzzer = YbBuzzer()

FRAME_HEAD = 0x23
FRAME_LEN = 8

SEND_LINE_ERROR = True
PRINT_DEBUG = False

USE_LAST_ERROR_WHEN_LOST = True
MAX_LOST_KEEP_FRAMES = 5

UART_SEND_INTERVAL_MS = 20
TURN_CMD_DURATION_FRAMES = 10

HISTORY_BUFFER_SIZE = 10
FIT_MODE_MAX_FRAMES = 50


# =====================================================
# 障碍物蜂鸣器参数
# =====================================================
BUZZER_FREQ = 2000          # 蜂鸣器频率
BUZZER_VOLUME = 90          # 音量90%
BUZZER_DURATION = 0.1       # 每次响0.1秒
BUZZER_REFRESH_MS = 80      # 每80ms刷新一次，实现持续响


# =====================================================
# 灰度巡线参数
# =====================================================
BASE_WIDTH = 640
BASE_HEIGHT = 480

BLACK_GRAY_THRESHOLD = 50

SCAN_X_STEP = 5
SCAN_H_STEP = 8

MIN_LINE_POINTS = 5

LINE_SPACE = 377

LEFT_ROI_BASE = (2, 400, 200, 80)
RIGHT_ROI_BASE = (440, 400, 200, 80)
HORIZONTAL_ROI_BASE = (270, 20, 100, 100)

CALIB_CENTER_BASE = 320


# =====================================================
# E 值缩放和平滑参数
# =====================================================
ERROR_GAIN_NUM = 25
ERROR_GAIN_DEN = 100

ERROR_FILTER_NUM = 20
ERROR_FILTER_DEN = 100

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
def pack_uart(dir_cmd=0, light=0, obstacle=0, end_point=0, line_error=0, place=0):
    if SEND_LINE_ERROR:
        line_error = int(line_error)
        line_error = max(-128, min(127, line_error))
        line_error_byte = line_error & 0xFF
    else:
        line_error_byte = 0

    frame = [
        FRAME_HEAD,
        dir_cmd & 0xFF,
        light & 0xFF,
        obstacle & 0xFF,
        end_point & 0xFF,
        line_error_byte,
        place & 0xFF
    ]

    checksum = sum(frame) & 0xFF
    frame.append(checksum)

    return bytes(frame)


# =====================================================
# E 处理函数
# =====================================================
def process_error(raw_error, last_error):
    if raw_error is None:
        return None

    raw_error = int(raw_error)
    last_error = int(last_error)

    target_error = raw_error * ERROR_GAIN_NUM // ERROR_GAIN_DEN

    filtered_error = last_error + (
        (target_error - last_error) * ERROR_FILTER_NUM // ERROR_FILTER_DEN
    )

    delta = filtered_error - last_error

    if delta > ERROR_MAX_STEP:
        filtered_error = last_error + ERROR_MAX_STEP
    elif delta < -ERROR_MAX_STEP:
        filtered_error = last_error - ERROR_MAX_STEP

    return int(filtered_error)


# =====================================================
# 历史轨迹拟合函数
# =====================================================
def fit_virtual_error(history_buffer):
    if not history_buffer:
        return 0

    total_weight = 0
    weighted_sum = 0

    for i, error in enumerate(history_buffer):
        weight = i + 1
        weighted_sum += error * weight
        total_weight += weight

    return weighted_sum // total_weight


# =====================================================
# 图像格式判断
# =====================================================
def get_img_info(img):
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
    obstacle = 0
    end_point = 0
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

    if "roadblock" in detected_names:
        obstacle = 1

    if "destination" in detected_names:
        end_point = 1

    return dir_cmd, light, obstacle, end_point, detected_names


# =====================================================
# 绘图函数
# =====================================================
def draw_line_info(
    osd_img,
    line_state,
    cam_w,
    cam_h,
    display_w,
    display_h,
    fps,
    detected_names,
    turn_state,
    turn_remaining,
    fit_mode_active,
    fit_error,
    obstacle,
    end_point,
    place,
    buzzer_obstacle_on
):
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

    h_text = "H-Line: YES" if horizontal_line_y is not None else "H-Line: NO"

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

    if turn_state == 1:
        turn_text = "TURN: LEFT ({})".format(turn_remaining)
        turn_color = (255, 0, 0)
    elif turn_state == 2:
        turn_text = "TURN: RIGHT ({})".format(turn_remaining)
        turn_color = (0, 255, 0)
    else:
        turn_text = "TURN: STRAIGHT"
        turn_color = (255, 255, 255)

    osd_img.draw_string_advanced(
        0,
        208,
        22,
        turn_text,
        turn_color
    )

    if fit_mode_active:
        fit_text = "FIT MODE: ACTIVE (E: {})".format(fit_error)
        fit_color = (255, 165, 0)
    else:
        fit_text = "FIT MODE: INACTIVE"
        fit_color = (128, 128, 128)

    osd_img.draw_string_advanced(
        0,
        238,
        22,
        fit_text,
        fit_color
    )

    if obstacle == 1:
        obstacle_text = "OBSTACLE: YES"
        obstacle_color = (255, 0, 0)
    else:
        obstacle_text = "OBSTACLE: NO"
        obstacle_color = (128, 128, 128)

    osd_img.draw_string_advanced(
        0,
        268,
        22,
        obstacle_text,
        obstacle_color
    )

    if end_point == 1:
        end_text = "DESTINATION: YES"
        end_color = (0, 255, 0)
    else:
        end_text = "DESTINATION: NO"
        end_color = (128, 128, 128)

    osd_img.draw_string_advanced(
        0,
        298,
        22,
        end_text,
        end_color
    )

    if place == 1:
        place_text = "PLACE: LOST (999)"
        place_color = (255, 0, 0)
    else:
        place_text = "PLACE: NORMAL"
        place_color = (0, 255, 0)

    osd_img.draw_string_advanced(
        0,
        328,
        22,
        place_text,
        place_color
    )

    if buzzer_obstacle_on:
        buzzer_text = "BUZZER: ON"
        buzzer_color = (255, 0, 0)
    else:
        buzzer_text = "BUZZER: OFF"
        buzzer_color = (128, 128, 128)

    osd_img.draw_string_advanced(
        0,
        358,
        22,
        buzzer_text,
        buzzer_color
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

    last_uart_send_time = time.ticks_ms()

    latest_dir_cmd = 0
    latest_light = 0
    latest_obstacle = 0
    latest_end_point = 0
    latest_send_error = 0
    latest_place = 0

    # 障碍物蜂鸣器状态
    buzzer_obstacle_on = False
    last_buzzer_time = time.ticks_ms()

    last_Ld = 999
    last_Rd = 999
    turn_state = 0
    turn_remaining_frames = 0

    error_history = []
    fit_mode_active = False
    fit_mode_frames = 0
    current_fit_error = 0

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
        ai_dir_cmd, light, obstacle, end_point, detected_names = parse_detection_result(res, labels)

        # -----------------------------
        # 1.1 障碍物蜂鸣器控制
        # 检测到障碍物：持续响
        # 障碍物消失：关闭
        # -----------------------------
        current_buzzer_time = time.ticks_ms()

        if obstacle == 1:
            buzzer_obstacle_on = True

            if time.ticks_diff(current_buzzer_time, last_buzzer_time) >= BUZZER_REFRESH_MS:
                buzzer.on(BUZZER_FREQ, BUZZER_VOLUME, BUZZER_DURATION)
                last_buzzer_time = current_buzzer_time

        else:
            if buzzer_obstacle_on:
                buzzer_obstacle_on = False

                try:
                    buzzer.off()
                except Exception:
                    pass

        # -----------------------------
        # 2. 灰度巡线
        # -----------------------------
        line_state = get_line_state(img, cam_w, cam_h, layout)
        current_Ld = line_state["Ld"]
        current_Rd = line_state["Rd"]
        has_horizontal_line = line_state["horizontal_line_y"] is not None

        place = 1 if (current_Ld == 999 or current_Rd == 999) else 0

        raw_error = line_state["error"]
        line_state["raw_error"] = raw_error

        line_error = process_error(raw_error, last_valid_error)

        # -----------------------------
        # 3. 历史轨迹拟合逻辑
        # -----------------------------
        if line_error is not None:
            last_valid_error = line_error
            lost_count = 0
            send_error = line_error
            line_state["error"] = line_error

            error_history.append(line_error)
            if len(error_history) > HISTORY_BUFFER_SIZE:
                error_history.pop(0)

            fit_mode_active = False
            fit_mode_frames = 0

        else:
            lost_count += 1

            if (
                turn_state == 0 and
                current_Ld == 999 and
                current_Rd == 999 and
                len(error_history) >= HISTORY_BUFFER_SIZE // 2 and
                not fit_mode_active
            ):
                fit_mode_active = True
                fit_mode_frames = 0
                current_fit_error = fit_virtual_error(error_history)
                print("进入历史轨迹拟合模式，拟合误差：", current_fit_error)

            if fit_mode_active:
                fit_mode_frames += 1

                if fit_mode_frames > FIT_MODE_MAX_FRAMES:
                    fit_mode_active = False
                    send_error = 0
                    line_state["error"] = 0
                    print("拟合模式超时，退出")
                else:
                    send_error = current_fit_error
                    line_state["error"] = current_fit_error

            else:
                if USE_LAST_ERROR_WHEN_LOST and lost_count <= MAX_LOST_KEEP_FRAMES:
                    send_error = last_valid_error
                    line_state["error"] = last_valid_error
                else:
                    send_error = 0
                    line_state["error"] = 0

        # -----------------------------
        # 4. 转弯逻辑判断
        # -----------------------------
        if turn_remaining_frames > 0:
            turn_remaining_frames -= 1
            if turn_remaining_frames == 0:
                turn_state = 0
        else:
            left_edge_lost = (last_Ld != 999) and (current_Ld == 999)
            right_edge_lost = (last_Rd != 999) and (current_Rd == 999)

            if left_edge_lost or right_edge_lost:
                if ai_dir_cmd != 0:
                    turn_state = ai_dir_cmd
                    turn_remaining_frames = TURN_CMD_DURATION_FRAMES

                elif not has_horizontal_line and ai_dir_cmd != 0:
                    turn_state = ai_dir_cmd
                    turn_remaining_frames = TURN_CMD_DURATION_FRAMES

                elif has_horizontal_line:
                    if left_edge_lost:
                        turn_state = 1
                    else:
                        turn_state = 2
                    turn_remaining_frames = TURN_CMD_DURATION_FRAMES

                else:
                    turn_state = 0

        last_Ld = current_Ld
        last_Rd = current_Rd

        final_dir_cmd = turn_state

        latest_dir_cmd = final_dir_cmd
        latest_light = light
        latest_obstacle = obstacle
        latest_end_point = end_point
        latest_send_error = send_error
        latest_place = place

        # -----------------------------
        # 5. 绘制显示
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
                detected_names,
                turn_state,
                turn_remaining_frames,
                fit_mode_active,
                current_fit_error,
                obstacle,
                end_point,
                place,
                buzzer_obstacle_on
            )

        pl.show_image()

        # -----------------------------
        # 6. 定时发送串口
        # -----------------------------
        current_time = time.ticks_ms()
        if time.ticks_diff(current_time, last_uart_send_time) >= UART_SEND_INTERVAL_MS:
            send_frame = pack_uart(
                dir_cmd=latest_dir_cmd,
                light=latest_light,
                obstacle=latest_obstacle,
                end_point=latest_end_point,
                line_error=latest_send_error,
                place=latest_place
            )

            uart.send(send_frame)

            if PRINT_DEBUG:
                print(
                    "dir:",
                    latest_dir_cmd,
                    "light:",
                    latest_light,
                    "obstacle:",
                    latest_obstacle,
                    "end_point:",
                    latest_end_point,
                    "raw_error:",
                    raw_error,
                    "send_error:",
                    latest_send_error,
                    "place:",
                    latest_place,
                    "buzzer:",
                    buzzer_obstacle_on,
                    "fit_mode:",
                    fit_mode_active,
                    "labels:",
                    detected_names,
                    "frame:",
                    send_frame.hex()
                )

            last_uart_send_time = current_time

            if frame_count % 20 == 0:
                gc.collect()

except KeyboardInterrupt:
    print("\n程序停止")

except Exception as e:
    print("程序报错：", e)

finally:
    try:
        buzzer.off()
    except Exception:
        pass

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
