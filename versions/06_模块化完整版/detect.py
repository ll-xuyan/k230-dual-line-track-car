# AI目标检测模块
import os, gc
from libs.PlatTasks import DetectionApp
from libs.Utils import *

# 全局变量
det_app = None
root_path = "/sdcard/mp_deployment_source/"


# 初始化AI模型
def init_detect(pl, display_size):
    global det_app
    deploy_conf = read_json(root_path + "/deploy_config.json")
    kmodel_path = root_path + deploy_conf["kmodel_path"]
    labels = deploy_conf["categories"]
    confidence_threshold = deploy_conf["confidence_threshold"]
    nms_threshold = deploy_conf["nms_threshold"]
    model_input_size = deploy_conf["img_size"]
    nms_option = deploy_conf["nms_option"]
    model_type = deploy_conf["model_type"]

    anchors = []
    if model_type == "AnchorBaseDet":
        anchors = deploy_conf["anchors"][0] + deploy_conf["anchors"][1] + deploy_conf["anchors"][2]

    inference_mode = "video"
    debug_mode = 0
    rgb888p_size = [1280, 720]

    det_app = DetectionApp(inference_mode, kmodel_path, labels, model_input_size,
                           anchors, model_type, confidence_threshold, nms_threshold,
                           rgb888p_size, display_size, debug_mode=debug_mode)
    det_app.config_preprocess()
    print("✅ AI模型初始化完成")


# 运行AI检测
def run_detect(img, osd_img):
    if det_app is None:
        return
    res = det_app.run(img)
    det_app.draw_result(osd_img, res)