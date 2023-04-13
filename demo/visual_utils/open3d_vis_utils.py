"""
Open3d visualization tool box
Written by Jihan YANG
All rights preserved from 2021 - present.
"""
import open3d
import torch
import matplotlib
import numpy as np
import time


# 定义直通滤波函数
def radar_filter(point_stack):
    x = point_stack[:, 0]
    y = point_stack[:, 1]
    
    field1 = x > -20
    field2 = x < 20
    field = np.logical_and(field1, field2)
    field1 = y > -50
    field = np.logical_and(field, field1)
    field1 = y < 50
    field = np.logical_and(field, field1)
    point_stack = point_stack[field, :]
    # print(point_stack.shape)
    return point_stack
    


def get_coor_colors(obj_labels):
    """
    Args:
        obj_labels: 1 is ground, labels > 1 indicates different instance cluster

    Returns:
        rgb: [N, 3]. color for each point.
    """
    colors = matplotlib.colors.XKCD_COLORS.values()
    max_color_num = obj_labels.max()

    color_list = list(colors)[:max_color_num+1]
    colors_rgba = [matplotlib.colors.to_rgba_array(
        color) for color in color_list]
    label_rgba = np.array(colors_rgba)[obj_labels]
    label_rgba = label_rgba.squeeze()[:, :3]

    return label_rgba


def draw_scenes(vis, points, points_radar=None, gt_boxes=None, ref_boxes=None, ref_labels=None, ref_scores=None, point_colors=None, draw_origin=True, wait=None):
    if isinstance(points, torch.Tensor):
        points = points.cpu().numpy()
    if isinstance(gt_boxes, torch.Tensor):
        gt_boxes = gt_boxes.cpu().numpy()
    if isinstance(ref_boxes, torch.Tensor):
        ref_boxes = ref_boxes.cpu().numpy()

    # draw origin
    if draw_origin:
        axis_pcd = open3d.geometry.TriangleMesh.create_coordinate_frame(
            size=1.0, origin=[0, 0, 0])
        vis.add_geometry(axis_pcd)

    pts = open3d.geometry.PointCloud()
    pts.points = open3d.utility.Vector3dVector(points[:, :3])

    vis.add_geometry(pts)
    if point_colors is None:
        pts.colors = open3d.utility.Vector3dVector(
            np.ones((points.shape[0], 3)))
    else:
        pts.colors = open3d.utility.Vector3dVector(point_colors)

    if gt_boxes is not None:
        vis = draw_box(vis, gt_boxes, (0, 0, 1))

    if ref_boxes is not None:
        vis = draw_box(vis, ref_boxes, (0, 1, 0), ref_labels, ref_scores)

    if points_radar is not None:
        pts2 = open3d.geometry.PointCloud()
        pts2.points = open3d.utility.Vector3dVector(points_radar[:, :3])
        
        vis.add_geometry(pts2)
        color_ = np.ones((points_radar.shape[0], 3))
        color_[:, 1:3] = 0.5  # RBG
        pts2.colors = open3d.utility.Vector3dVector(color_)
        
    if wait is not None:
        vis.poll_events()
        vis.update_renderer()
        vis.clear_geometries()

        time.sleep(wait)
    else:
        vis.run()


def translate_boxes_to_open3d_instance(gt_boxes):
    """
             4-------- 6
           /|         /|
          5 -------- 3 .
          | |        | |
          . 7 -------- 1
          |/         |/
          2 -------- 0
    """
    center = gt_boxes[0:3]
    lwh = gt_boxes[3:6]
    axis_angles = np.array([0, 0, gt_boxes[6] + 1e-10])
    rot = open3d.geometry.get_rotation_matrix_from_axis_angle(axis_angles)
    box3d = open3d.geometry.OrientedBoundingBox(center, rot, lwh)

    line_set = open3d.geometry.LineSet.create_from_oriented_bounding_box(box3d)

    # import ipdb; ipdb.set_trace(context=20)
    lines = np.asarray(line_set.lines)
    lines = np.concatenate([lines, np.array([[1, 4], [7, 6]])], axis=0)

    line_set.lines = open3d.utility.Vector2iVector(lines)

    return line_set, box3d


def draw_box(vis, gt_boxes, color=(0, 1, 0), ref_labels=None, score=None):
    for i in range(gt_boxes.shape[0]):
        line_set, box3d = translate_boxes_to_open3d_instance(gt_boxes[i])

        if ref_labels[i] <= 1: # CAR and TRUCK
            line_set.paint_uniform_color((0, 1, 0))
        else:
            line_set.paint_uniform_color((0, 0, 1))

        vis.add_geometry(line_set)
    return vis


def draw_radar_scenes(vis, points):
    pts = open3d.geometry.PointCloud()
    pts.points = open3d.utility.Vector3dVector(points[:, :3])
    
    vis.add_geometry(pts)
    color_ = np.ones((points.shape[0], 3))
    color_[:, 1:3] = 0.5  # RBG
    pts.colors = open3d.utility.Vector3dVector(color_)
    render_option = vis.get_render_option()	#设置点云渲染参数
    render_option.point_size = 2.0	#设置渲染点的大小
    # vis.run()