# Copyright (c) OpenMMLab. All rights reserved.
from argparse import ArgumentParser

from ultralytics import YOLO

from mmdet3d.apis import inference_detector, init_model
from mmdet3d.registry import VISUALIZERS

import matplotlib as mpl
import matplotlib.pyplot as plt

from nuscenes.nuscenes import NuScenes

import cv2
import open3d as o3d
import numpy as np
from visual_utils import open3d_vis_utils as V

import os
import numpy as np
import math

# ROS 
import rospy
from sensor_msgs.msg import PointCloud2, Image 
from jsk_recognition_msgs.msg import BoundingBoxArray, BoundingBox
from cv_bridge import CvBridge
import sensor_msgs.point_cloud2 as pcl2
from std_msgs.msg import Header
import geometry_msgs

# user define
from nuscenes_tools import radar_tool
'''
python demo/nuscenes_seq_demo.py \
    v1.0-mini \
    demo/data/nuscenes/v1.0-mini \
    configs/pointpillars/pointpillars_hv_secfpn_sbn-all_8xb4-2x_nus-3d.py \
    checkpoints/nuscenes/hv_pointpillars_secfpn_sbn-all_4x8_2x_nus-3d_20210826_225857-f19d00a3.pth --show --score-thr=0.6
'''

USING_ROS_RVIZ = True

radar_list = ['RADAR_FRONT', 'RADAR_FRONT_LEFT','RADAR_FRONT_RIGHT', 'RADAR_BACK_LEFT', 'RADAR_BACK_RIGHT']
img_pos_list = ['CAM_FRONT']

rospy.init_node('nuScenes', anonymous=True)
pcl_pub=rospy.Publisher('/LIDAR_TOP',PointCloud2, queue_size=1)
img_pub=rospy.Publisher('/CAM_FRONT', Image, queue_size=1)
pub_boxes = rospy.Publisher("/bounding_boxes", BoundingBoxArray, queue_size=1)
cv_bridge = CvBridge()
rate=rospy.Rate(10)
frame=0

'''
names: 
description: Briefly describe the function of your function
return {*}
'''
def parse_args():
    parser = ArgumentParser()
    parser.add_argument('nus_type', type=str, help='Point cloud file')
    parser.add_argument('path', type=str, help='Point cloud file')
    parser.add_argument('config', help='Config file')
    parser.add_argument('checkpoint', help='Checkpoint file')
    parser.add_argument(
        '--device', default='cuda:0', help='Device used for inference')
    parser.add_argument(
        '--score-thr', type=float, default=0.1, help='bbox score threshold')
    parser.add_argument(
        '--out-dir', type=str, default='demo', help='dir to save results')
    parser.add_argument(
        '--show',
        action='store_true',
        help='show online visualization results')
    args = parser.parse_args()
    return args

 # 欧拉角转四元数
'''
names: 
description: Briefly describe the function of your function
param {*} augler
return {*}
'''
def RPY2Quar(augler):
    cy=math.cos(augler[0]*0.5)  # YAW
    sy=math.sin(augler[0]*0.5)
    cp=math.cos(augler[1]*0.5) # pitch
    sp=math.sin(augler[1]*0.5)
    cr=math.cos(augler[2] * 0.5) # roll
    sr=math.sin(augler[2] * 0.5)
    
    q=geometry_msgs.msg.Pose()
    q.orientation.w= cy * cp * cr + sy * sp * sr
    q.orientation.x = cy * cp * sr - sy * sp * cr
    q.orientation.y = sy * cp * sr + cy * sp * cr
    q.orientation.z = sy * cp * cr - cy * sp * sr
    return q

'''
names: 
description: Briefly describe the function of your function
param {*} bboxes_3d
return {*}
'''
def publish_boundingBox(bboxes_3d):
    bboxes = BoundingBoxArray()
    bboxes.boxes = []
    bboxes.header.frame_id = 'map'
    for i in range(bboxes_3d.shape[0]):
        box = BoundingBox()
        center = bboxes_3d[i, 0:3]
        lwh = bboxes_3d[i, 3:6]
        # axis_angles = np.array([0, 0, bboxes_3d[i, 6] + 1e-10])
        axis_angles = np.array([bboxes_3d[i, 6] + 1e-10, 0, 0])
        q = RPY2Quar(axis_angles)
        box.header.frame_id = 'map'
        box.pose = q
        box.pose.position.x = float(center[0])
        box.pose.position.y = float(center[1])
        box.pose.position.z = float(center[2])
        # box.pose.orientation.w = q.w
        # box.pose.orientation.x = q.x
        # box.pose.orientation.y = q.y
        # box.pose.orientation.z = q.z
        box.dimensions.x = float(lwh[0])
        box.dimensions.y = float(lwh[1])
        box.dimensions.z = float(lwh[2])
        box.label = 1
        box.value = 0
        bboxes.boxes.append(box)
    return bboxes

'''
names: 
description: Briefly describe the function of your function
param {*} args
return {*}
'''
def main(args):
    model = init_model(args.config, args.checkpoint, device=args.device)

    # 加载视觉模型
    YOLO_model = YOLO("checkpoints/YOLO/yolov8n.pt")

    # init visualizer
    if USING_ROS_RVIZ is not True:
        vis = o3d.visualization.Visualizer()
        vis.create_window()
        vis.get_render_option().point_size = 2.0
        vis.get_render_option().background_color = np.zeros(3)

    # 加载nuscenes数据
    nusc = NuScenes(version=args.nus_type,
                    dataroot=args.path, verbose=True)

    plt.figure()
    fig, ax = plt.subplots(1, 1, figsize=(9, 9))

    for scene_idx in range(len(nusc.scene)):
        scene = nusc.scene[scene_idx]

        # 获取该场景的first token
        cur_sample_info = nusc.get('sample', scene['first_sample_token'])

        # 连续读取该scenes下的所有frame
        while cur_sample_info['next'] != "":

            pc_path_ = nusc.get_sample_data_path(
                cur_sample_info['data']['LIDAR_TOP'])

            # 点云模型推理
            result, data = inference_detector(model, pc_path_)
            points = data['inputs']['points']
            data_input = dict(points=points)

            pred_instances_3d = result.pred_instances_3d
            pred_instances_3d = pred_instances_3d[
                            pred_instances_3d.scores_3d > args.score_thr].to('cpu')

            bboxes_3d = pred_instances_3d.bboxes_3d.tensor.cpu().numpy()
            labels_3d = pred_instances_3d.labels_3d
            scores_3d = pred_instances_3d.scores_3d

            # 视觉模型推理
            for img_pos in img_pos_list:
                img_path_ = nusc.get_sample_data_path(
                    cur_sample_info['data'][img_pos])
                results = YOLO_model(img_path_)  # predict on an image
                img = cv2.imread(img_path_)

                # 绘制检测bbox
                box = results[0].boxes.boxes.cpu().numpy()
                box_cls = results[0].boxes.cls.cpu().numpy()
                for box_idx in range(box.shape[0]):
                    img = cv2.rectangle(img,
                                        (int(box[box_idx, 0]),
                                         int(box[box_idx, 1])),
                                        (int(box[box_idx, 2]),
                                         int(box[box_idx, 3])),
                                        (box_cls[box_idx]*10, box_cls[box_idx]*10, 255), 2)
                
                if USING_ROS_RVIZ is not True:
                    height, width = img.shape[:2]  # 图片的高度和宽度     
                    imgZoom1 = cv2.resize(img, (int(0.4*width), int(0.4*height)))
                    cv2.imshow(img_pos, imgZoom1)
                    cv2.waitKey(1)
                else:
                    img_msg = cv_bridge.cv2_to_imgmsg(img, "bgr8")
                    img_pub.publish(img_msg)

            # 毫米波雷达点云显示
            for sub_ in radar_list:
                pc, velocities = radar_tool.get_radar_point(
                    nusc, cur_sample_info['data'][sub_], ax=ax)
                if sub_ == 'RADAR_FRONT':
                    all_point = pc.points
                else:
                    all_point = np.hstack((all_point, pc.points))
                    
                    
            if USING_ROS_RVIZ is not True:
                # 点云和检测结果可视化
                V.draw_scenes(vis,
                            points=data_input['points'][:, :3],
                            points_radar=V.radar_filter(all_point[:3, :].T),
                            ref_boxes=bboxes_3d,
                            ref_scores=scores_3d,
                            ref_labels=labels_3d,
                            wait=0.02,
                            )     
            else:
                header=Header()
                header.stamp=rospy.Time.now()
                header.frame_id='map'
                pcl_pub.publish(pcl2.create_cloud_xyz32(header,points[:,:3]))
                rospy.loginfo('published')
                
                bboxes = publish_boundingBox(bboxes_3d)
                pub_boxes.publish(bboxes)
                
            # 获取下一帧
            cur_sample_info = nusc.get('sample', cur_sample_info['next'])
            

if __name__ == '__main__':
    args = parse_args()
    main(args)
