# Copyright (c) OpenMMLab. All rights reserved.
from argparse import ArgumentParser

from mmdet3d.apis import inference_detector, init_model
from mmdet3d.registry import VISUALIZERS

import open3d as o3d
import numpy as np
from visual_utils import open3d_vis_utils as V

def parse_args():
    parser = ArgumentParser()
    parser.add_argument('pcd', help='Point cloud file')
    parser.add_argument('config', help='Config file')
    parser.add_argument('checkpoint', help='Checkpoint file')
    parser.add_argument(
        '--device', default='cuda:0', help='Device used for inference')
    parser.add_argument(
        '--score-thr', type=float, default=0.0, help='bbox score threshold')
    parser.add_argument(
        '--out-dir', type=str, default='demo', help='dir to save results')
    parser.add_argument(
        '--show',
        action='store_true',
        help='show online visualization results')
    parser.add_argument(
        '--snapshot',
        action='store_true',
        help='whether to save online visualization results')
    args = parser.parse_args()
    return args


def main(args):
    # TODO: Support inference of point cloud numpy file.
    # build the model from a config file and a checkpoint file
    model = init_model(args.config, args.checkpoint, device=args.device)

    # # init visualizer
    # visualizer = VISUALIZERS.build(model.cfg.visualizer)
    # visualizer.dataset_meta = model.dataset_meta
    
    vis = o3d.visualization.Visualizer()
    vis.create_window()
    vis.get_render_option().point_size = 1.0
    vis.get_render_option().background_color = np.zeros(3)

    # test a single point cloud sample
    result, data = inference_detector(model, args.pcd)
    points = data['inputs']['points']
    data_input = dict(points=points)

    pred_instances_3d = result.pred_instances_3d
    pred_instances_3d = pred_instances_3d[
                    pred_instances_3d.scores_3d > args.score_thr].to('cpu')
                
    bboxes_3d = pred_instances_3d.bboxes_3d.tensor.cpu().numpy()
    labels_3d = pred_instances_3d.labels_3d
    scores_3d = pred_instances_3d.scores_3d

    V.draw_scenes(vis,
                points=data_input['points'][:, :3],
                ref_boxes=bboxes_3d,
                ref_scores=scores_3d,
                ref_labels=labels_3d
                )



if __name__ == '__main__':
    args = parse_args()
    main(args)
