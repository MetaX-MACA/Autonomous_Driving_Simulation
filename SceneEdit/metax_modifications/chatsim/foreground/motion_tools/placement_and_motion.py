# 2025 - Modified by MetaX Integrated Circuits (Shanghai) Co., Ltd. All Rights Reserved.
import openai

import ipdb
import os
import re
import ast
import matplotlib.pyplot as plt
from chatsim.foreground.motion_tools.tools import (
    transform_node_to_lane,
    generate_vertices,
    visualize,
    find_closest_centerline,
    rot_and_trans,
    filter_forward_lane,
    inverse_rot_and_trans,
    rot_and_trans_bbox,
    rot_and_trans_point,
    filter_right_lane,
    filter_left_lane,
    visualize_placement,
    hermite_spline_once,
    hermite_spline_twice,
)
from chatsim.foreground.motion_tools.placement_iterative import vehicle_placement
from chatsim.foreground.motion_tools.check_collision import (
    check_collision_and_revise_static
)
import numpy as np
import random
# from motion_tracking import motion_tracking


def vehicle_motion(
    map_data,
    all_current_vertices,
    placement_result=[],
    high_level_action_direction=[],
    high_level_action_speed=[],
    dt=0.4,
    total_len=10,
    origin_extrinsics=[],
):
    
    if placement_result[0] is None:
        return (None, "no placement")
    
    if high_level_action_direction == "static":
        return np.array(placement_result[0:2])[None,...].repeat(total_len, axis=0)
    
    slow_speed_threshold = (1.5, 2)
    fast_speed_threshold = (10, 25)
    random_speed_threshold = (3, 25)

    current_position = placement_result
    transformed_map_data = rot_and_trans(map_data, current_position)
    transformed_all_current_vertices = rot_and_trans_bbox(
        all_current_vertices, current_position
    )
    points = origin_extrinsics[:, 0:2, 3] # 坐标变换，以车辆插入点为原点
    transformed_points = rot_and_trans_point(points, current_position)

    if high_level_action_speed == 'slow':
        v_init = random.uniform(slow_speed_threshold[0], slow_speed_threshold[1])
    elif high_level_action_speed == 'fast':
        v_init = random.randint(fast_speed_threshold[0], fast_speed_threshold[1])
    else:
        v_init = random.randint(random_speed_threshold[0], random_speed_threshold[1])

    transformed_map_data = filter_forward_lane(transformed_map_data)

    if high_level_action_direction == "turn left":
        transformed_map_data_dest = filter_left_lane(transformed_map_data)
    elif high_level_action_direction == "turn right":
        transformed_map_data_dest = filter_right_lane(transformed_map_data)

    if (high_level_action_direction == "turn left"
        or high_level_action_direction == "turn right"):
        # destination_anchor = transformed_map_data_dest["centerline"]#[::5] # 每隔 5 个元素取一次点，对轨迹进行稀疏化处理
        # sorted_destination = destination_anchor[
        #     random.randint(0, len(destination_anchor) - 1)
        # ]
        # sorted_destination_direction = sorted_destination[2:4] - sorted_destination[0:2]
        # sorted_destination = sorted_destination[:2]

        # 手动选择终点
        mid_index = int(total_len*0.9)
        sorted_destination = np.array([transformed_points[mid_index, 0], transformed_points[mid_index, 1]])
        next_destination = np.array([transformed_points[mid_index+1, 0], transformed_points[mid_index+1, 1]])
        sorted_destination_direction = next_destination - sorted_destination
        

    elif high_level_action_direction == "straight":
        sorted_destination_init = np.array([v_init * dt * total_len, 0])
        _, sorted_destination = find_closest_centerline(
            transformed_map_data, sorted_destination_init
        )

        sorted_destination_direction = sorted_destination[2:4] - sorted_destination[0:2]
        sorted_destination = (sorted_destination[0:2] + sorted_destination[2:4]) / 2


    start = np.array([0, 0])
    end = np.array([sorted_destination[0], sorted_destination[1]])
    Vs = np.array([v_init, 0])  

    Ve = (
        v_init
        * sorted_destination_direction
        / np.linalg.norm(sorted_destination_direction)
    )
    Ve = np.abs(Ve)

    # 使用 Hermite 样条生成一条从起点到终点的初步轨迹
    coordinates = hermite_spline_once(
        start,
        end,
        Vs,
        Ve,
    )

    # 计算中心线上离轨迹中点最近的点，作为新轨迹的中点，并计算中点的速度方向
    current_midpoint = coordinates[-int(len(coordinates) / 2)]
    midpoint_check_flag, closest_centerline = find_closest_centerline(
        transformed_map_data, current_midpoint
    )
    midpoint = (closest_centerline[0:2] + closest_centerline[2:4]) / 2
    midpoint_direction = closest_centerline[2:4] - closest_centerline[0:2]
    Vm = v_init * (midpoint_direction) / np.linalg.norm(midpoint_direction)
    Vm = np.abs(Vm)

    # 生成经过中点、终点的 Hermite 样条曲线，构成车辆的完整轨迹
    coordinates = hermite_spline_twice(
        start,
        end,
        midpoint,
        Vs,
        Ve,
        Vm,
    )

    generated_trajectory = np.array(coordinates[:: int(len(coordinates) / total_len)])
    # generated_trajectory = check_collision_and_revise_static(
    #     generated_trajectory, transformed_all_current_vertices
    # )

    
    if high_level_action_direction == "turn left":
        generated_trajectory = np.array(coordinates[:: int(len(coordinates) / int(total_len*0.8))])
        # 用最后一个元素填充，使轨迹点数仍然是 total_len
        num = len(generated_trajectory)
        if num < total_len:
            padding = []
            for _ in range(total_len-num):
                padding.append(generated_trajectory[-1])
            padding = np.array(padding)
            generated_trajectory = np.concatenate([generated_trajectory, padding], axis=0)

    # 检查生成的轨迹是否与其他静止车辆发生碰撞，并在必要时对轨迹进行修正
    # generated_trajectory = check_collision_and_revise_static(
    #     generated_trajectory, transformed_all_current_vertices
    # )

    # 生成的轨迹从局部坐标系转换回全局坐标系
    generated_trajectory = inverse_rot_and_trans(
        generated_trajectory, current_position
    )
    print("generated_trajectory: ", generated_trajectory)


    return generated_trajectory

