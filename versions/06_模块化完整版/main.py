# 主程序：开机自启 + 调用AI + 调用巡线
import gc
from libs.PipeLine import PipeLine
# 导入两个功能模块
import detect
import line_follow

# 配置
display_mode = "lt9611"
rgb888p_size = [1280, 720]

# 初始化摄像头/显示（只初始化1次！）
pl = PipeLine(rgb888p_size=rgb888p_size, display_mode=display_mode)
pl.create()
display_size = pl.get_display_size()

# 初始化两个功能
detect.init_detect(pl, display_size)
print("✅ 巡线模块加载完成")

# 主循环
while True:
    img = pl.get_frame()
    osd_img = pl.osd_img

    # 同时运行两个功能
    detect.run_detect(img, osd_img)  # AI检测
    line_follow.run_line(img, osd_img)  # 巡线识别

    pl.show_image()
    gc.collect()

# 释放资源
pl.destroy()