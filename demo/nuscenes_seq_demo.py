# Copyright (c) OpenMMLab. All rights reserved.
from argparse import ArgumentParser

from ultralytics import YOLO

from mmdet3d.apis import inference_detector, init_model
from mmdet3d.registry import VISUALIZERS

import matplotlib as mpl
import matplotlib.pyplot as plt

from nuscenes.nuscenes import NuScenes
import torch

import cv2
import open3d as o3d
import numpy as np
from visual_utils import open3d_vis_utils as V

import os
import math
import yaml
import time
import sys
import signal
import json

# ByteTracker
from tracker import byte_tracker as ByteTracker

# 3DMOT module
from mot_3d.mot import MOTModel
from mot_3d.frame_data import FrameData
from mot_3d.data_protos import BBox, Validity

# ROS 
import rospy
from sensor_msgs.msg import PointCloud, PointCloud2, Image
from sensor_msgs.msg import PointField
from visualization_msgs.msg import Marker, MarkerArray
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


'''
names: 
description: Briefly describe the function of your function
return {*}
'''
def quit(signum, frame):
    sys.exit()
    
signal.signal(signal.SIGINT, quit)
signal.signal(signal.SIGTERM, quit)

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
def publish_boundingBox(bboxes, type='detection', ids=None, label=None):
    ros_bboxes = BoundingBoxArray()
    ros_bboxes.boxes = []
    ros_bboxes.header.frame_id = 'nuscenes'
    
    marker_array = MarkerArray()
    marker_array.markers = []
    if type == 'detection':
        for i in range(bboxes.shape[0]):
            box = BoundingBox()
            center = bboxes[i, 0:3]
            lwh = bboxes[i, 4:7]
            # axis_angles = np.array([0, 0, bboxes[i, 6] + 1e-10])
            axis_angles = np.array([bboxes[i, 3] + 1e-10, 0, 0])
            q = RPY2Quar(axis_angles)
            box.header.frame_id = 'nuscenes'
            box.pose = q
            box.pose.position.x = float(center[0])
            box.pose.position.y = float(center[1])
            box.pose.position.z = float(center[2])
            box.dimensions.x = float(lwh[0])
            box.dimensions.y = float(lwh[1])
            box.dimensions.z = float(lwh[2])
            box.label = 1
            box.value = 0
            ros_bboxes.boxes.append(box)

    else:
        bboxes_3d = [BBox.bbox2array(bbox) for bbox in bboxes]
        bboxes_3d = np.array(bboxes_3d)
        for i in range(bboxes_3d.shape[0]):
            box = BoundingBox()
            center = bboxes_3d[i, 0:3]
            lwh = bboxes_3d[i, 4:7]
            # axis_angles = np.array([0, 0, bboxes_3d[i, 6] + 1e-10])
            axis_angles = np.array([bboxes_3d[i, 3] + 1e-10, 0, 0])
            q = RPY2Quar(axis_angles)
            box.header.frame_id = 'nuscenes'
            box.pose = q
            box.pose.position.x = (center[0])
            box.pose.position.y = (center[1])
            box.pose.position.z = (center[2])
            # box.pose.orientation.w = q.w
            # box.pose.orientation.x = q.x
            # box.pose.orientation.y = q.y
            # box.pose.orientation.z = q.z
            box.dimensions.x = (lwh[0])
            box.dimensions.y = (lwh[1])
            box.dimensions.z = (lwh[2])
            box.label = 1
            box.value = 0
            ros_bboxes.boxes.append(box)

            sub_mark = Marker()
            # sub_mark.header.stamp = rospy.Time.now()
            sub_mark.header.frame_id = 'nuscenes'
            sub_mark.id = i
            sub_mark.text = str(label[i]) + '_' + str(ids[i]) + '__' + str(i)
            sub_mark.action = Marker.ADD
            sub_mark.type = Marker.TEXT_VIEW_FACING
            # sub_mark.lifetime = rospy.Duration(0)
            
            sub_mark.pose.position.x = (center[0])
            sub_mark.pose.position.y = (center[1])
            sub_mark.pose.position.z = (center[2]) + 0.5
            
            sub_mark.color.a = 1
            sub_mark.color.r = 1
            sub_mark.color.g = 0
            sub_mark.color.b = 0

            sub_mark.scale.x = 1
            sub_mark.scale.y = 1
            sub_mark.scale.z = 1
            
            marker_array.markers.append(sub_mark)
            
    return ros_bboxes, marker_array


'''
names: 
description: Briefly describe the function of your function
param {*} points
return {*}
'''
def publish_lidar_point(points):
    header.stamp=rospy.Time.now()
    msg = PointCloud2()
    msg.header.stamp = rospy.Time().now()
    msg.header.frame_id = "nuscenes"
    msg.height = 1
    msg.width = points.shape[0]
    msg.fields = [
    PointField('x', 0, PointField.FLOAT32, 1),
    PointField('y', 4, PointField.FLOAT32, 1),
    PointField('z', 8, PointField.FLOAT32, 1)]
    msg.is_bigendian = False
    msg.point_step = 12
    msg.row_step = msg.point_step * points.shape[0]
    msg.is_dense = False
    msg.data = np.asarray(points[:,:3], np.float32).tostring()
    pcl_pub.publish(msg)
    
    
    
'''
names: run_3DMOT
description: liadr 3D-MOT
param {*} bboxes_3d
param {*} scores_3d
param {*} labels_3d
return {*}
'''
def run_3DMOT(bboxes_3d, scores_3d, labels_3d):

    bboxes_3d[:, 7] = scores_3d # 3DMOT需要使用box的score属性进行匹配关联，此处需要进行属性拼接
    bboxes_3d[:, 8] = labels_3d # 拼接labels
            
    # 构造MOT输入
    aux_info = {'is_key_frame': True}
    millis = int(round(time.time() * 1000))
    new_dets=list()
    for idx, det in enumerate(bboxes_3d):
        if det[8] < 2:
            new_dets.append(det[:8])  
    frame_data = FrameData(dets=new_dets, ego=None, time_stamp=millis, pc=None, det_types=None, aux_info=aux_info)
    
    # 跟踪算法入口
    results = pc_tracker.frame_mot(frame_data)
    
    # print(results)
    result_pred_bboxes = [trk[0] for trk in results]
    result_pred_ids = [trk[1] for trk in results]
    result_pred_states = [trk[2] for trk in results]
    result_types = [trk[3] for trk in results]
    
    result_pred_bbox, markers = publish_boundingBox(result_pred_bboxes,'track', result_pred_ids, result_types)
    pub_trace_boxes.publish(result_pred_bbox)
    pub_trace_markers.publish(markers)

'''
names: 
description: Briefly describe the function of your function
param {*} args
return {*}
'''
def main(args):
    device = torch.device("cuda:0")
    model = init_model(args.config, args.checkpoint, device=device)
    # model.to(device)

    # 加载视觉模型
    YOLO_model = YOLO("checkpoints/YOLO/yolov8n.pt")
    YOLO_model.to(device)

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

    signal.signal(signal.SIGINT, quit)
    signal.signal(signal.SIGTERM, quit)

    dataroot='/media/charles/ShareDisk/00myDataSet/nuScenes/v1.0-test/'
    root = './demo/result/v1.0-test/'
    for filename in os.listdir(root):
        print(filename)
    
        f = open(root + filename, 'r')
        all_json_data = f.read()
        de_json = json.loads(all_json_data)
        
        for idx in range(len(de_json)):
            pc_path_ = dataroot + de_json[idx]['LIDAR_TOP']

            # 点云模型推理
            result, data = inference_detector(model, pc_path_)
            points = data['inputs']['points']
            
            # 发布点云
            publish_lidar_point(points)
            data_input = dict(points=points)

            pred_instances_3d = result.pred_instances_3d
            pred_instances_3d = pred_instances_3d[
                            pred_instances_3d.scores_3d > args.score_thr].to('cpu')
            
            # 9D features = [x, y, z, l, w, h, yaw, roll, picth]
            bboxes_3d = pred_instances_3d.bboxes_3d.tensor.cpu().numpy()  
            bboxes_3d[:, [3,4,5,6]] = bboxes_3d[:, [6,3,4,5]] 
            labels_3d = pred_instances_3d.labels_3d
            scores_3d = pred_instances_3d.scores_3d
            
            #执行MOT算法
            if RUN_3DMOT_FLAG:
                run_3DMOT(pred_instances_3d.bboxes_3d.tensor.cpu().numpy() , 
                            pred_instances_3d.scores_3d, 
                            pred_instances_3d.labels_3d)

            # 视觉模型推理
            img_list = []
            for img_pos in img_pos_list:
                # img_path_ = nusc.get_sample_data_path(cur_sample_info['data'][img_pos])
                img_path_ = dataroot + de_json[idx]['CAM_FRONT']
                img = cv2.imread(img_path_)

                if RUN_CAMERA_DET:
                    results = YOLO_model(img_path_)  # predict on an image
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
                    img_list.append(img)
                
            # img_FRONT = np.hstack((np.hstack((img_list[0], img_list[1])), img_list[2]))
            # img_BACK = np.hstack((np.hstack((img_list[3], img_list[4])), img_list[5]))  
            # img_ALL = np.vstack((img_FRONT, img_BACK))
            img_msg = cv_bridge.cv2_to_imgmsg(img_list[0], "bgr8")
            img_pub.publish(img_msg)
                            
            bboxes, not_uesd = publish_boundingBox(bboxes_3d)
            pub_boxes.publish(bboxes)
                            
                
             
radar_list = ['RADAR_FRONT', 'RADAR_FRONT_LEFT','RADAR_FRONT_RIGHT', 'RADAR_BACK_LEFT', 'RADAR_BACK_RIGHT']
# img_pos_list = ['CAM_FRONT_LEFT', 'CAM_FRONT', 'CAM_FRONT_RIGHT', 'CAM_BACK_LEFT', 'CAM_BACK', 'CAM_BACK_RIGHT']
img_pos_list = ['CAM_FRONT_LEFT']

# config
USING_ROS_RVIZ = True

RUN_3DMOT_FLAG = True
RUN_CAMERA_DET = True
RADAR_VISIABLE = False

header=Header()
header.frame_id='nuscenes'
rospy.init_node('nuscenes', anonymous=True)
pcl_pub=rospy.Publisher('/LIDAR_TOP', PointCloud2, queue_size=10)
radar_pub=rospy.Publisher('/RADAR', PointCloud2, queue_size=10)
img_pub=rospy.Publisher('/CAMERA', Image, queue_size=10)
pub_boxes = rospy.Publisher("/BBOX", BoundingBoxArray, queue_size=10)
pub_trace_boxes = rospy.Publisher("/PC_TRACE", BoundingBoxArray, queue_size=10)
pub_trace_markers = rospy.Publisher("/PC_TRACE_MARK", MarkerArray, queue_size=10)
cv_bridge = CvBridge()
# rate=rospy.Rate(10)

# 跟踪器初始化
# load model configs
config_path = './demo/mot_3d/config/iou.yaml'
configs = yaml.load(open(config_path, 'r'), Loader=yaml.Loader)
pc_tracker = MOTModel(configs)


'''
names: 
description: Briefly describe the function of your function
return {*}
'''
if __name__ == '__main__':
    args = parse_args()
    main(args)
    rospy.signal_shutdown("closed!")
    rospy.spin()
