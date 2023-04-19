from nuscenes.nuscenes import NuScenes
from visual_utils import open3d_vis_utils as V

import operator
import time

import cv2
import numpy as np
import open3d as o3d
from mayavi import mlab

import matplotlib.pyplot as plt

from transformers import AutoImageProcessor, DetrForObjectDetection
import torch
from PIL import Image,ImageDraw
import requests

import os
import json


# from nuscenes_tools import radar_tool

# nusc = NuScenes(version='v1.0-mini', 
#                 dataroot='/home/charles/myDataSet/nuScenes/v1.0-mini', verbose=True)

# # 获取所有场景
# nusc.list_scenes()

# # 以第一个场景为例
# scene = nusc.scene[0]

# # 获取该场景的first token
# cur_sample_info = nusc.get('sample', scene['first_sample_token'])
# # print(cur_sample_info)

# # init visualizer
# # vis = o3d.visualization.Visualizer()
# # vis.create_window()
# # vis.get_render_option().point_size = 1.0
# # vis.get_render_option().background_color = np.zeros(3)

image_processor = AutoImageProcessor.from_pretrained("facebook/detr-resnet-50")
model = DetrForObjectDetection.from_pretrained("facebook/detr-resnet-50")
device = torch.device("cuda")
model.to(device)
model.device

dataroot='/media/charles/ShareDisk/00myDataSet/nuScenes/v1.0-test/'
root = './demo/result/v1.0-test/'
for filename in os.listdir(root):
    print(filename)
    
    f = open(root + filename, 'r')
    all_json_data = f.read()
    de_json = json.loads(all_json_data)
    
    for idx in range(len(de_json)):
        img_path_ = dataroot + de_json[idx]['CAM_FRONT']
        image = Image.open(img_path_)
        inputs = image_processor(images=image, return_tensors="pt")
        inputs.to(device)
        outputs = model(**inputs)
        target_sizes = torch.tensor([image.size[::-1]])
        results = image_processor.post_process_object_detection(outputs, threshold=0.6, target_sizes=target_sizes)[0]
        
        # draw = ImageDraw.Draw(image)
        img = cv2.imread(img_path_)
        for score, label, box in zip(results["scores"], results["labels"], results["boxes"]):
            box = [round(i, 2) for i in box.tolist()]
            
            # draw.rectangle(box, outline="#FF0000", width=2)
            img = cv2.rectangle(img, (int(box[0]), int(box[1])),
                                    (int(box[2]),int(box[3])), (255, 0,0))

        # image.show()
        cv2.imshow('xxx', img)
        cv2.waitKey(1)
    
# # fig, ax = plt.subplots(1, 1, figsize=(9, 9))
# # radar_list = ['RADAR_FRONT','RADAR_FRONT_LEFT','RADAR_FRONT_RIGHT','RADAR_BACK_LEFT','RADAR_BACK_RIGHT']
# # 连续读取该scenes下的所有frame
# while cur_sample_info['next'] != "":
#     # 可视化该token所有sensor数据
#     # nusc.render_sample_data(sensor_data_token['token'], with_anns=False)

#     # # 或者获取指定sensor数据的路径（二次开发可视化功能）
#     test_sersor = 'CAM_FRONT'
#     sensor_data_token = nusc.get('sample_data', cur_sample_info['data'][test_sersor])
#     img_path_ = nusc.get_sample_data_path(cur_sample_info['data'][test_sersor])

#     if operator.contains(test_sersor, "CAM"):
#         # img = cv2.imread(data_path)
#         # cv2.imshow("image1", img)
#         # cv2.waitKey(10)
#         image = Image.open(img_path_)
#         inputs = image_processor(images=image, return_tensors="pt")
#         inputs.to(device)
#         outputs = model(**inputs)
#         target_sizes = torch.tensor([image.size[::-1]])
#         results = image_processor.post_process_object_detection(outputs, threshold=0.6, target_sizes=target_sizes)[0]

#         # draw = ImageDraw.Draw(image)
#         img = cv2.imread(img_path_)
#         for score, label, box in zip(results["scores"], results["labels"], results["boxes"]):
#             box = [round(i, 2) for i in box.tolist()]
            
#             # draw.rectangle(box, outline="#FF0000", width=2)
#             img = cv2.rectangle(img, (int(box[0]), int(box[1])),
#                                     (int(box[2]),int(box[3])), (255, 0,0))

#         # image.show()
#         cv2.imshow('xxx', img)
#         cv2.waitKey(1)
        
        

