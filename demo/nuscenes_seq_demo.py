# Copyright (c) OpenMMLab. All rights reserved.
from argparse import ArgumentParser

from ultralytics import YOLO

from mmdet3d.apis import inference_detector, init_model
from mmdet3d.registry import VISUALIZERS

from nuscenes.nuscenes import NuScenes

import cv2
import open3d as o3d
import numpy as np
from visual_utils import open3d_vis_utils as V

import matplotlib as mpl
import matplotlib.pyplot as plt
# mpl.use('TkAgg')
# plt.ion()
from nuscenes_tools import radar_tool

# img_pos_list = ['CAM_BACK', 'CAM_BACK_LEFT', 'CAM_BACK_RIGHT', 'CAM_FRONT', 'CAM_FRONT_LEFT', 'CAM_FRONT_RIGHT' ]
radar_list = ['RADAR_FRONT', 'RADAR_FRONT_LEFT',
    'RADAR_FRONT_RIGHT', 'RADAR_BACK_LEFT', 'RADAR_BACK_RIGHT']
img_pos_list = ['CAM_BACK', 'CAM_FRONT']


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


def main(args):
    # TODO: Support inference of point cloud numpy file.
    # build the model from a config file and a checkpoint file
    model = init_model(args.config, args.checkpoint, device=args.device)

    # 加载视觉模型
    YOLO_model = YOLO("checkpoints/YOLO/yolov8n.pt")

    # init visualizer
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

                height, width = img.shape[:2]  # 图片的高度和宽度
                imgZoom1 = cv2.resize(img, (int(0.4*width), int(0.4*height)))
                cv2.imshow(img_pos, imgZoom1)
                cv2.waitKey(1)

            # 毫米波雷达点云显示
            for sub_ in radar_list:
                pc, velocities = radar_tool.get_radar_point(
                    nusc, cur_sample_info['data'][sub_], ax=ax)
                if sub_ == 'RADAR_FRONT':
                    all_point = pc.points
                else:
                    all_point = np.hstack((all_point, pc.points))
            
            # 点云和检测结果可视化
            V.draw_scenes(vis,
                        points=data_input['points'][:, :3],
                        points_radar=V.radar_filter(all_point[:3, :].T),
                        ref_boxes=bboxes_3d,
                        ref_scores=scores_3d,
                        ref_labels=labels_3d,
                        wait=0.02,
                        )     

            # 获取下一帧
            cur_sample_info = nusc.get('sample', cur_sample_info['next'])
            

if __name__ == '__main__':
    args = parse_args()
    main(args)
