# Copyright (c) OpenMMLab. All rights reserved.
from argparse import ArgumentParser

from ultralytics import YOLO

from mmdet3d.apis import inference_detector, init_model
from mmdet3d.registry import VISUALIZERS

from nuscenes.nuscenes import NuScenes

import cv2

def parse_args():
    parser = ArgumentParser()
    parser.add_argument('nus_type', type=str,help='Point cloud file')
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
    visualizer = VISUALIZERS.build(model.cfg.visualizer)
    visualizer.dataset_meta = model.dataset_meta

    # 加载nuscenes数据
    nusc = NuScenes(version=args.nus_type,
                    dataroot=args.path, verbose=True)
    
    
    for scene_idx in range(len(nusc.scene)):
        scene = nusc.scene[scene_idx]

        # 获取该场景的first token
        cur_sample_info = nusc.get('sample', scene['first_sample_token'])

        # 连续读取该scenes下的所有frame
        while cur_sample_info['next'] != "":
            # test_sersor = 'LIDAR_TOP'
            # sensor_data_token = nusc.get('sample_data', cur_sample_info['data'][test_sersor])
            # data_path, box_list, cam_intrinsic = nusc.get_sample_data(sensor_data_token['token'])
            # point = np.fromfile(data_path, dtype=np.float32,count=-1).reshape([-1, 5])
            
            pc_path_ = nusc.get_sample_data_path(cur_sample_info['data']['LIDAR_TOP'])
            img_path_ = nusc.get_sample_data_path(cur_sample_info['data']['CAM_FRONT'])
            
            # 点云模型推理
            result, data = inference_detector(model, pc_path_)
            points = data['inputs']['points']
            data_input = dict(points=points)
            
            # 视觉模型推理
            results = YOLO_model(img_path_)  # predict on an image
            img = cv2.imread(img_path_)

            # 绘制检测bbox
            box = results[0].boxes.boxes.cpu().numpy()
            box_cls = results[0].boxes.cls.cpu().numpy()
            for box_idx in range(box.shape[0]):
                img = cv2.rectangle(img,
                                    (int(box[box_idx, 0]),int(box[box_idx, 1])),
                                    (int(box[box_idx, 2]), int(box[box_idx, 3])), 
                                    (box_cls[box_idx]*10, box_cls[box_idx]*10, 255), 2)

            cv2.imshow('img', img)
            cv2.waitKey(100)
            
            # show the results
            visualizer.add_datasample(
                'result',
                data_input,
                data_sample=result,
                draw_gt=False,
                show=args.show,
                wait_time=0.1,
                out_file=args.out_dir,
                pred_score_thr=args.score_thr,
                vis_task='lidar_det')
            
            # 获取下一帧
            cur_sample_info = nusc.get('sample', cur_sample_info['next'])
            
    # # test a single point cloud sample
    # result, data = inference_detector(model, args.pcd)
    # points = data['inputs']['points']
    # data_input = dict(points=points)

    # # show the results
    # visualizer.add_datasample(
    #     'result',
    #     data_input,
    #     data_sample=result,
    #     draw_gt=False,
    #     show=args.show,
    #     wait_time=0,
    #     out_file=args.out_dir,
    #     pred_score_thr=args.score_thr,
    #     vis_task='lidar_det')


if __name__ == '__main__':
    args = parse_args()
    main(args)
