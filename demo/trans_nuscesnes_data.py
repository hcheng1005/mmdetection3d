from nuscenes.nuscenes import NuScenes
import json
import copy

sensor_list = ['LIDAR_TOP',
               'RADAR_FRONT', 'RADAR_FRONT_LEFT', 'RADAR_FRONT_RIGHT', 'RADAR_BACK_LEFT', 'RADAR_BACK_RIGHT', 'CAM_FRONT_LEFT', 'CAM_FRONT', 'CAM_FRONT_RIGHT',
               'CAM_BACK_LEFT', 'CAM_BACK', 'CAM_BACK_RIGHT']

nusc = NuScenes(version='v1.0-test',
                dataroot='/media/charles/ShareDisk/00myDataSet/nuScenes/v1.0-test/', verbose=True)


mytemplate = {'LIDAR_TOP': '',
            'RADAR_FRONT': '',
            'RADAR_FRONT_LEFT': '',
            'RADAR_FRONT_RIGHT': '',
            'RADAR_BACK_LEFT': '',
            'RADAR_BACK_RIGHT': '',
            'CAM_FRONT_LEFT': '',
            'CAM_FRONT': '',
            'CAM_FRONT_RIGHT': '',
            'CAM_BACK_LEFT': '',
            'CAM_BACK': '',
            'CAM_BACK_RIGHT': '',
            }

# 遍历所有场景
for scene_idx in range(len(nusc.scene)):
    sample_info_dict_list = []
    scene = nusc.scene[scene_idx]

    # 获取该场景的first token
    cur_sample_info = nusc.get('sample', scene['first_sample_token'])
    while cur_sample_info['next'] != "":
        
        sub_file_info = mytemplate
        for sub_pos in sensor_list:
            file_path = nusc.get_sample_data_path(cur_sample_info['data'][sub_pos])
            file_name_list = file_path.split('/')
            sub_file_info[sub_pos] = file_name_list[-3]+'/'+file_name_list[-2]+'/'+file_name_list[-1]

        cpy_data = copy.deepcopy(sub_file_info)
        sample_info_dict_list.append(cpy_data)

        # 获取下一帧
        cur_sample_info = nusc.get('sample', cur_sample_info['next'])

        print(cur_sample_info['token'])


        jsonData = json.dumps(sample_info_dict_list, sort_keys=False,
                            indent=4, separators=(',', ': '))

        # print(jsonData)
        f = open('./demo/result/' + scene['name'] + '.json', 'w')
        f.write(jsonData)
        f.close()
