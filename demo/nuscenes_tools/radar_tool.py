
from nuscenes.nuscenes import RadarPointCloud
from nuscenes.utils.geometry_utils import view_points, box_in_image, BoxVisibility, transform_matrix

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.axes import Axes

from pyquaternion import Quaternion


def get_radar_point(nusc, 
                    sample_data_token: str,
                    nsweeps=1, use_flat_vehicle_coordinates=True, 
                    ax: Axes = None, axes_limit: float = 100,
                    box_vis_level: BoxVisibility = BoxVisibility.ANY,
                    with_anns: bool = True):
    
    sd_record = nusc.get('sample_data', sample_data_token)
    sample_rec = nusc.get('sample', sd_record['sample_token'])
    chan = sd_record['channel']
    ref_chan = 'LIDAR_TOP'
    ref_sd_token = sample_rec['data'][ref_chan]
    ref_sd_record = nusc.get('sample_data', ref_sd_token)

    pc, times = RadarPointCloud.from_file_multisweep(nusc, sample_rec, chan, ref_chan, nsweeps=nsweeps)

    # Transform radar velocities (x is front, y is left), as these are not transformed when loading the
    # point cloud.
    radar_cs_record = nusc.get('calibrated_sensor', sd_record['calibrated_sensor_token'])
    ref_cs_record = nusc.get('calibrated_sensor', ref_sd_record['calibrated_sensor_token'])
    velocities = pc.points[8:10, :]  # Compensated velocity
    velocities = np.vstack((velocities, np.zeros(pc.points.shape[1])))
    velocities = np.dot(Quaternion(radar_cs_record['rotation']).rotation_matrix, velocities)
    velocities = np.dot(Quaternion(ref_cs_record['rotation']).rotation_matrix.T, velocities)
    velocities[2, :] = np.zeros(pc.points.shape[1])

    # By default we render the sample_data top down in the sensor frame.
    # This is slightly inaccurate when rendering the map as the sensor frame may not be perfectly upright.
    # Using use_flat_vehicle_coordinates we can render the map in the ego frame instead.
    if use_flat_vehicle_coordinates:
        # Retrieve transformation matrices for reference point cloud.
        cs_record = nusc.get('calibrated_sensor', ref_sd_record['calibrated_sensor_token'])
        pose_record = nusc.get('ego_pose', ref_sd_record['ego_pose_token'])
        ref_to_ego = transform_matrix(translation=cs_record['translation'],
                                        rotation=Quaternion(cs_record["rotation"]))

        # Compute rotation between 3D vehicle pose and "flat" vehicle pose (parallel to global z plane).
        ego_yaw = Quaternion(pose_record['rotation']).yaw_pitch_roll[0]
        rotation_vehicle_flat_from_vehicle = np.dot(
            Quaternion(scalar=np.cos(ego_yaw / 2), vector=[0, 0, np.sin(ego_yaw / 2)]).rotation_matrix,
            Quaternion(pose_record['rotation']).inverse.rotation_matrix)
        vehicle_flat_from_vehicle = np.eye(4)
        vehicle_flat_from_vehicle[:3, :3] = rotation_vehicle_flat_from_vehicle
        viewpoint = np.dot(vehicle_flat_from_vehicle, ref_to_ego)
    else:
        viewpoint = np.eye(4)
    
    points = view_points(pc.points[:3, :], viewpoint, normalize=False)
    dists = np.sqrt(np.sum(pc.points[:2, :] ** 2, axis=0))
    colors = np.minimum(1, dists / axes_limit / np.sqrt(2))
            
    point_scale = 3.0
    scatter = ax.scatter(points[0, :], points[1, :], c=colors, s=point_scale)
                        
    points_vel = view_points(pc.points[:3, :] + velocities, viewpoint, normalize=False)
    deltas_vel = points_vel - points
    deltas_vel = 6 * deltas_vel  # Arbitrary scaling
    max_delta = 20
    deltas_vel = np.clip(deltas_vel, -max_delta, max_delta)  # Arbitrary clipping
    colors_rgba = scatter.to_rgba(colors)
    for i in range(points.shape[1]):
        ax.arrow(points[0, i], points[1, i], deltas_vel[0, i], deltas_vel[1, i], color=colors_rgba[i])

    # Show ego vehicle.
    ax.plot(0, 0, 'x', color='red')
    
    # # Get boxes in lidar frame.
    # _, boxes, _ = nusc.get_sample_data(ref_sd_token, box_vis_level=box_vis_level,
    #                                         use_flat_vehicle_coordinates=use_flat_vehicle_coordinates)

    # # Show boxes.
    # if with_anns:
    #     for box in boxes:
    #         c = np.array(nusc.explorer.get_color(box.name)) / 255.0
    #         box.render(ax, view=np.eye(4), colors=(c, c, c))

    # Limit visible range.
    ax.set_xlim(-axes_limit, axes_limit)
    ax.set_ylim(-axes_limit, axes_limit)
            
    return pc, velocities