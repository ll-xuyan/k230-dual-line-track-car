# 巡线识别模块
from media.media import Image

# ===================== 1280x720 巡线参数 =====================
BLACK_THRESHOLD = ((8, 18, -6, 3, -12, 5))
# 左/右检测ROI（按比例放大）
LEFT_ROI = (4, 600, 360, 160)
RIGHT_ROI = (920, 600, 360, 160)
# 横线检测ROI
HORIZONTAL_ROI = (440, 40, 440, 200)
# 双线间距/中心
LINE_SPACE = 754
SCREEN_CENTER = 1280 // 2
CALIB_CENTER = 640
DISPLAY_HEIGHT = 720


# ============================================================

# 识别黑线位置
def get_line_position(img_ndarray, roi, threshold):
    # numpy数组 转 CanMV Image对象
    h, w, c = img_ndarray.shape
    img = Image(w, h, Image.RGB888, img_ndarray.tobytes())

    x, y, w_roi, h_roi = roi
    bottom_roi = (x, y + h_roi // 2, w_roi, h_roi // 2)
    blobs = img.find_blobs([threshold], roi=bottom_roi, pixels_threshold=10, area_threshold=10)

    if blobs:
        max_blob = max(blobs, key=lambda b: b[4])
        return max_blob[0] + max_blob[2] // 2
    return None


# 识别横线
def get_horizontal_line(img_ndarray, roi, threshold):
    h, w, c = img_ndarray.shape
    img = Image(w, h, Image.RGB888, img_ndarray.tobytes())

    blobs = img.find_blobs([threshold], roi=roi, pixels_threshold=30, area_threshold=30)
    if blobs:
        horizontal_blobs = [b for b in blobs if (b[2] / b[3]) > 2]
        if horizontal_blobs:
            max_blob = max(horizontal_blobs, key=lambda b: b[1])
            return max_blob[1] + max_blob[3] // 2
    return None


# 绘制巡线信息
def draw_line(osd_img, err, ld, rd, mid_x, h_line_y):
    # 检测框
    osd_img.draw_rectangle(LEFT_ROI, (255, 0, 0), 2)
    osd_img.draw_rectangle(RIGHT_ROI, (0, 255, 0), 2)
    osd_img.draw_rectangle(HORIZONTAL_ROI, (255, 255, 0), 1)
    # 中心参考线
    osd_img.draw_line(SCREEN_CENTER, 0, SCREEN_CENTER, DISPLAY_HEIGHT, (0, 255, 0), 1)
    osd_img.draw_line(CALIB_CENTER, 0, CALIB_CENTER, DISPLAY_HEIGHT, (0, 255, 255), 2)

    # 状态文本
    osd_img.draw_string_advanced(0, 20, 40, f"E:{err:.1f}" if err else "E:--", (255, 255, 255))
    osd_img.draw_string_advanced(0, 80, 30, f"Ld:{ld}", (255, 255, 0))
    osd_img.draw_string_advanced(200, 80, 30, f"Rd:{rd}", (255, 255, 0))

    # 横线
    if h_line_y:
        osd_img.draw_string_advanced(0, 180, 30, "横线:已检测", (255, 0, 255))
        osd_img.draw_line(0, h_line_y, 1280, h_line_y, (255, 0, 0), 2)

    # 绘制赛道线
    if ld != 999:
        x = SCREEN_CENTER - ld
        osd_img.draw_line(x, LEFT_ROI[1], x, LEFT_ROI[1] + LEFT_ROI[3], (255, 255, 0), 2)
    if rd != 999:
        x = SCREEN_CENTER + rd
        osd_img.draw_line(x, RIGHT_ROI[1], x, RIGHT_ROI[1] + RIGHT_ROI[3], (255, 255, 0), 2)
    if mid_x:
        osd_img.draw_line(mid_x, LEFT_ROI[1], mid_x, DISPLAY_HEIGHT, (0, 0, 255), 2)


# 运行巡线
def run_line(img, osd_img):
    left_x = get_line_position(img, LEFT_ROI, BLACK_THRESHOLD)
    right_x = get_line_position(img, RIGHT_ROI, BLACK_THRESHOLD)
    h_line_y = get_horizontal_line(img, HORIZONTAL_ROI, BLACK_THRESHOLD)

    Ld = Rd = 999
    virtual_mid = error = None

    if left_x and right_x:
        Ld = SCREEN_CENTER - left_x
        Rd = right_x - SCREEN_CENTER
        virtual_mid = (left_x + right_x) // 2
    elif left_x:
        Ld = SCREEN_CENTER - left_x
        virtual_mid = (left_x + left_x + LINE_SPACE) // 2
    elif right_x:
        Rd = right_x - SCREEN_CENTER
        virtual_mid = (right_x - LINE_SPACE + right_x) // 2

    if virtual_mid:
        error = virtual_mid - SCREEN_CENTER

    draw_line(osd_img, error, Ld, Rd, virtual_mid, h_line_y)