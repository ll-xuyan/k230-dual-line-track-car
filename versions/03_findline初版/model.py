#模型代码
# -*- coding: utf-8 -*-
import os, gc
from libs.PlatTasks import DetectionApp
from libs.PipeLine import PipeLine
from libs.Utils import *

display_mode = "lt9611"

rgb888p_size = [1280, 720]

root_path = "/sdcard/mp_deployment_source/"

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

pl = PipeLine(rgb888p_size=rgb888p_size, display_mode=display_mode)
pl.create()
display_size = pl.get_display_size()

det_app = DetectionApp(inference_mode,kmodel_path,labels,model_input_size,anchors,model_type,confidence_threshold,nms_threshold,rgb888p_size,display_size,debug_mode=debug_mode)

det_app.config_preprocess()

while True:
    with ScopedTiming("total", 1):
        img = pl.get_frame()
        res = det_app.run(img)
        det_app.draw_result(pl.osd_img, res)
        pl.show_image()
        gc.collect()

det_app.deinit()
pl.destroy()