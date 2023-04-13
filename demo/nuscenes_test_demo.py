from nuscenes.nuscenes import NuScenes


import operator
import time

import cv2
import numpy as np
import open3d as o3d
from mayavi import mlab

import matplotlib.pyplot as plt

plt.ion()

from nuscenes_tools import radar_tool



nusc = NuScenes(version='v1.0-mini', 
                dataroot='/home/zdhjs-05/myGitHubCode/v1.0-mini', verbose=True)

# 获取所有场景
nusc.list_scenes()

# 以第一个场景为例
scene = nusc.scene[0]

# 获取该场景的first token
cur_sample_info = nusc.get('sample', scene['first_sample_token'])
# print(cur_sample_info)
 
fig, ax = plt.subplots(1, 1, figsize=(9, 9))
radar_list = ['RADAR_FRONT','RADAR_FRONT_LEFT','RADAR_FRONT_RIGHT','RADAR_BACK_LEFT','RADAR_BACK_RIGHT']
# 连续读取该scenes下的所有frame
while cur_sample_info['next'] != "":
    # 可视化该token所有sensor数据
    # nusc.render_sample_data(sensor_data_token['token'], with_anns=False)

    # # 或者获取指定sensor数据的路径（二次开发可视化功能）
    # test_sersor = 'CAM_FRONT'
    # sensor_data_token = nusc.get('sample_data', cur_sample_info['data'][test_sersor])
    # data_path, box_list, cam_intrinsic = nusc.get_sample_data(sensor_data_token['token'])

    # if operator.contains(test_sersor, "CAM"):
    #     img = cv2.imread(data_path)
    #     cv2.imshow("image1", img)
    #     cv2.waitKey(10)

    # 毫米波雷达点云显示
    # plt.clf()
    # ax.cla()
    for sub_ in radar_list:
        pc, velocities = radar_tool.get_radar_point(nusc, cur_sample_info['data'][sub_], ax=ax)

    plt.show()
    plt.pause(0.02)
    plt.cla()

    # 获取next frame token
    cur_sample_info = nusc.get('sample', cur_sample_info['next'])
    # print(cur_sample_info['next'])

