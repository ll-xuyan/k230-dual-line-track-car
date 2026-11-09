import time
import sensor
import image
from pyb import UART  # 或者用 from machine import UART，看你固件版本

# ================== 摄像头初始化 ==================
sensor.reset()
sensor.set_pixformat(sensor.RGB565)
sensor.set_framesize(sensor.QQVGA)       # 160*120 保持原分辨率不变
sensor.set_auto_gain(False, gain_db=12)
sensor.set_auto_whitebal(False, rgb_gain_db=(55, 55, 55))
sensor.skip_frames(time=2000)           # 跳过预热帧

# ================== 视觉参数 (全部沿用原值) ==================
black_threshold = [(0, 20, -2, 5, -12, 9)]

# 底部巡线 ROI
ROI_LEFT  = (0, 80, 100, 40)
ROI_RIGHT = (60, 80, 100, 40)

# 前瞻 ROI
ROI_FRONT_LEFT  = (0, 45, 120, 35)
ROI_FRONT_RIGHT = (40, 45, 120, 35)

LANE_HALF_WIDTH = 72          # 车道标准半宽
MIN_LANE_DISTANCE = 40        # 左右线最小有效间距

MIN_PIXELS = 30

# 串口初始化
uart = UART(3, 115200)

# ================== 状态机变量 完全不变 ==================
STRAIGHT = 0
TURNING = 1
state = STRAIGHT

# 入弯检测计数器
turn_detect_count = 0
TURN_DETECT_THRESHOLD = 6
pending_turn_dir = None

# 转弯保持变量
turn_hold_frames = 0
TURN_TIMEOUT = 15
last_turn_error = 0

# 上一帧有效误差
last_valid_error = 0

# 辅助函数：判断是否是有效的双线
def is_valid_dual_line(lcx, rcx):
    return (lcx is not None and rcx is not None and abs(rcx - lcx) >= MIN_LANE_DISTANCE)

# ================== 主循环 ==================
while True:
    img = sensor.snapshot()

    # ---- 前瞻黑线检测 ----
    front_left = img.find_blobs(black_threshold, roi=ROI_FRONT_LEFT,
                                pixels_threshold=MIN_PIXELS, merge=True)
    front_right = img.find_blobs(black_threshold, roi=ROI_FRONT_RIGHT,
                                 pixels_threshold=MIN_PIXELS, merge=True)
    front_left_ok = True if front_left else False
    front_right_ok = True if front_right else False

    # ---- 底部左右黑线检测 ----
    blobs_left = img.find_blobs(black_threshold, roi=ROI_LEFT,
                                pixels_threshold=MIN_PIXELS, merge=True)
    blobs_right = img.find_blobs(black_threshold, roi=ROI_RIGHT,
                                 pixels_threshold=MIN_PIXELS, merge=True)

    left_cx = None
    right_cx = None

    if blobs_left:
        max_blob = max(blobs_left, key=lambda b: b.area())
        left_cx = max_blob.cx()
        print("left_cx", left_cx)
        img.draw_rectangle(max_blob.rect(), color=(255, 0, 0))
        img.draw_cross(left_cx, max_blob.cy(), color=(255, 0, 0))

    if blobs_right:
        max_blob = max(blobs_right, key=lambda b: b.area())
        right_cx = max_blob.cx()
        img.draw_rectangle(max_blob.rect(), color=(0, 255, 0))
        img.draw_cross(right_cx, max_blob.cy(), color=(0, 255, 0))

    # ---- 直道/弯道 状态机核心逻辑 一字未改 ----
    error = 0
    valid_dual = is_valid_dual_line(left_cx, right_cx)

    if state == STRAIGHT:
        # 直道巡线模式
        if valid_dual:
            center_line = (left_cx + right_cx) / 2
            error = 80 - center_line
            last_valid_error = error
            turn_detect_count = 0
            pending_turn_dir = None
            img.draw_line(80, 0, 80, 120, color=(128, 128, 128))
            print("state: STRAIGHT dual")

        elif left_cx is not None and right_cx is not None:
            center_line = (left_cx + (left_cx + LANE_HALF_WIDTH*2)) / 2
            error = 80 - center_line
            last_valid_error = error
            if not front_right_ok:
                if pending_turn_dir == 'right':
                    turn_detect_count += 1
                else:
                    pending_turn_dir = 'right'
                    turn_detect_count = 1
            else:
                turn_detect_count = 0
                pending_turn_dir = None
            print("state: STRAIGHT fake dual (same line), as left_only")

        elif left_cx is not None and right_cx is None:
            center_line = (left_cx + (left_cx + LANE_HALF_WIDTH*2)) / 2
            error = 80 - center_line
            last_valid_error = error
            if not front_right_ok:
                if pending_turn_dir == 'right':
                    turn_detect_count += 1
                else:
                    pending_turn_dir = 'right'
                    turn_detect_count = 1
            else:
                turn_detect_count = 0
                pending_turn_dir = None
            print("state: STRAIGHT left_only, turn_cnt:", turn_detect_count)

        elif right_cx is not None and left_cx is None:
            center_line = ((right_cx - LANE_HALF_WIDTH*2) + right_cx) / 2
            error = 80 - center_line
            last_valid_error = error
            if not front_left_ok:
                if pending_turn_dir == 'left':
                    turn_detect_count += 1
                else:
                    pending_turn_dir = 'left'
                    turn_detect_count = 1
            else:
                turn_detect_count = 0
                pending_turn_dir = None
            print("state: STRAIGHT right_only, turn_cnt:", turn_detect_count)

        else:
            error = last_valid_error * 0.8
            print("state: STRAIGHT lost, hold weak")

        # 触发进入弯道状态
        if turn_detect_count >= TURN_DETECT_THRESHOLD and pending_turn_dir is not None:
            state = TURNING
            turn_hold_frames = 0
            if pending_turn_dir == 'right':
                last_turn_error = -50
                print(">>> ENTER TURN: RIGHT <<<")
            else:
                last_turn_error = 50
                print(">>> ENTER TURN: LEFT <<<")
            turn_detect_count = 0
            pending_turn_dir = None

        # 直道前瞻微小修正
        if valid_dual:
            if (not front_left_ok) and front_right_ok:
                error -= 2
            elif (not front_right_ok) and front_left_ok:
                error += 2

    elif state == TURNING:
        # 弯道保持行驶
        if turn_hold_frames < TURN_TIMEOUT:
            decay = max(0.3, 1.0 - (turn_hold_frames / TURN_TIMEOUT) * 0.7)
            error = last_turn_error * decay
            turn_hold_frames += 1
            print("state: TURNING, hold:", error)
        else:
            error = last_turn_error * 0.2
            print("state: TURNING, timeout residual:", error)

        # 检测出弯，切回直道
        if valid_dual:
            center_line = (left_cx + right_cx) / 2
            new_err = 80 - center_line
            if abs(new_err) < 15:
                state = STRAIGHT
                error = new_err
                last_valid_error = new_err
                turn_detect_count = 0
                pending_turn_dir = None
                print(">>> EXIT TURN, back to straight <<<")

    # 串口发送偏差值 格式不变
    uart.write("E:%.1f\n" % error)
    print("E:", error)
