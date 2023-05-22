from __future__ import annotations
import time
import numpy as np
from scipy.spatial.transform import Rotation as R

def load_motion_data(bvh_file_path):
    """part2 辅助函数，读取bvh文件"""
    with open(bvh_file_path, 'r') as f:
        lines = f.readlines()
        for i in range(len(lines)):
            if lines[i].startswith('Frame Time'):
                break
        motion_data = []
        for line in lines[i+1:]:
            data = [float(x) for x in line.split()]
            if len(data) == 0:
                break
            motion_data.append(np.array(data).reshape(1,-1))
        motion_data = np.concatenate(motion_data, axis=0)
    return motion_data



def part1_calculate_T_pose(bvh_file_path):
    """请填写以下内容
    输入： bvh 文件路径
    输出:
        joint_name: List[str]，字符串列表，包含着所有关节的名字
        joint_parent: List[int]，整数列表，包含着所有关节的父关节的索引,根节点的父关节索引为-1
        joint_offset: np.ndarray，形状为(M, 3)的numpy数组，包含着所有关节的偏移量

    Tips:
        joint_name顺序应该和bvh一致
    """
    joint_name: list[str] = []
    joint_parent: list[int] = []
    joint_offset: list[list[float]] = []

    with open(bvh_file_path, 'r') as f:
        parent: str = ''
        lines = f.readlines()
        for i in range(len(lines)):
            if lines[i].startswith('HIERARCHY'):
                break
        for line in lines[i+1:]:
            if line.startswith('MOTION'):
                break
            line = line.strip()
            if line.startswith('JOINT') or line.startswith('ROOT'):
                infos = line.split()
                joint_name.append(infos[1])
                joint_parent.append(-1 if parent == '' else joint_name.index(parent))
            if line.startswith('End'):
                infos = line.split()
                joint_name.append(parent + '_end')
                joint_parent.append(-1 if parent == '' else joint_name.index(parent))
            elif line.startswith('{'): # 入栈
                parent = joint_name[-1]
            elif line.startswith('}'): # 出栈
                parent_index = joint_name.index(parent)
                parent = joint_name[joint_parent[parent_index]] # 找到当前父节点的父节点
            elif line.startswith('OFFSET'):
                data = [float(x) for x in line.split()[1:]]
                joint_offset.append(data)

    print(joint_name)
    print(joint_parent)

    return joint_name, joint_parent, np.array(joint_offset)


def part2_forward_kinematics(joint_name: list[str], joint_parent: list[int], joint_offset: np.ndarray, motion_data: np.ndarray, frame_id: int):
    """请填写以下内容
    输入: part1 获得的关节名字，父节点列表，偏移量列表
        motion_data: np.ndarray，形状为(N,X)的numpy数组，其中N为帧数，X为Channel数
        frame_id: int，需要返回的帧的索引
    输出:
        joint_positions: np.ndarray，形状为(M, 3)的numpy数组，包含着所有关节的全局位置
        joint_orientations: np.ndarray，形状为(M, 4)的numpy数组，包含着所有关节的全局旋转(四元数)
    Tips:
        1. joint_orientations的四元数顺序为(x, y, z, w)
        2. from_euler时注意使用大写的XYZ
    """
    joint_positions = []
    joint_orientations = []
    frame_motion: np.ndarray = motion_data[frame_id]
    root_pos = frame_motion[0: 3]
    root_orientation = R.from_euler('XYZ', frame_motion[3: 6], degrees=True).as_quat()

    joint_positions.append(root_pos)
    joint_orientations.append(root_orientation)
    channel_index = 0

    # print('motion: ', type(frame_motion[0: 3]))

    for i in range(1, len(joint_name)):
        is_end = joint_name[i].endswith('_end')
        # 由于末端节点没有channel信息，所以不能简单的用当前数组索引进行取值
        if not is_end:
            channel_index += 1

        parent_index = joint_parent[i]
        parent_orientation = R.from_quat(joint_orientations[parent_index])
        parent_pos: np.ndarray = joint_positions[parent_index]
        cur_offset: np.ndarray = joint_offset[i]
        start_index = 6 + (channel_index - 1) * 3
        # print(frame_motion[start_index: start_index + 3])
        cur_rotation = R.from_euler('XYZ', frame_motion[start_index: start_index + 3], degrees=True)
        # 当前关节的全局坐标位置
        cur_pos = parent_pos + parent_orientation.apply(cur_offset)
        # 当前关节的全局朝向（旋转）；末端节点没有自身的rotation信息，只有相对位移
        cur_orientation = parent_orientation if is_end else parent_orientation * cur_rotation
        joint_positions.append(cur_pos)
        joint_orientations.append(cur_orientation.as_quat())

    return np.array(joint_positions), np.array(joint_orientations)


def part3_retarget_func(T_pose_bvh_path: str, A_pose_bvh_path: str):
    """
    将 A-pose的bvh重定向到T-pose上
    输入: 两个bvh文件的路径
    输出: 
        motion_data: np.ndarray，形状为(N,X)的numpy数组，其中N为帧数，X为Channel数。retarget后的运动数据
    Tips:
        两个bvh的joint name顺序可能不一致哦(
        as_euler时也需要大写的XYZ
    """
    start_time = time.time()
    T_pose_joint_name, _, _ = part1_calculate_T_pose(T_pose_bvh_path)
    A_pose_joint_name, _, _ = part1_calculate_T_pose(A_pose_bvh_path)
    A_pose_motion = load_motion_data(A_pose_bvh_path)
    # T-pose关节在A-pose中的索引映射
    A_pose_joint_map: list[int] = [0]
    A_pose_joint_start: list[int] = [0]
    motion_data = []
    l_shoulder_rotation = R.from_euler('XYZ', [0, 0, 45], degrees=True).inv()
    r_shoulder_rotation = R.from_euler('XYZ', [0, 0, -45], degrees=True).inv()

    channel_index = 0
    for i in range(1, len(A_pose_joint_name)):
        A_joint_name = A_pose_joint_name[i]
        is_end = A_joint_name.endswith('_end')
        # 由于末端节点没有channel信息，所以不能简单的用当前数组索引进行取值
        if not is_end:
            channel_index += 1
        start_index = 6 + (channel_index - 1) * 3
        A_pose_joint_map.append(A_pose_joint_name.index(T_pose_joint_name[i]))
        A_pose_joint_start.append(start_index) # 记录A_pose每个关节对应的channel开始索引，方便进行关节数据映射

    for i in range(len(A_pose_motion)):
        motion_row: np.ndarray = A_pose_motion[i]
        retarget_motion_row = np.array(motion_row[0: 6])
        channel_index = 0
        for joint_i in range(1, len(A_pose_joint_name)):
            # A_pose_index = A_pose_joint_map[joint_i]
            A_joint_name = A_pose_joint_name[joint_i]
            is_end = A_joint_name.endswith('_end')
            # 由于末端节点没有channel信息，所以不能简单的用当前数组索引进行取值
            if is_end:
                continue # 末端关节没有channel数据
            channel_index += 1
            start_index = 6 + (channel_index - 1) * 3
            A_pose_data = motion_row[start_index: start_index + 3]
            A_pose_rotation = R.from_euler('XYZ', A_pose_data, degrees=True)

            # 还原A_pose变动的关节（逆变换）
            if A_joint_name == 'lShoulder':
                rotation = l_shoulder_rotation * A_pose_rotation
                retarget_motion_row = np.concatenate((retarget_motion_row, rotation.as_euler('XYZ', degrees=True)), axis=0)
            elif A_joint_name == 'rShoulder':
                rotation = r_shoulder_rotation * A_pose_rotation
                retarget_motion_row = np.concatenate((retarget_motion_row, rotation.as_euler('XYZ', degrees=True)), axis=0)
            else:
                retarget_motion_row = np.concatenate((retarget_motion_row, A_pose_data), axis=0)

        remap_motion_row = np.array(retarget_motion_row[0: 6])
        # 按照T-pose的关节顺序重新映射motion
        for joint_i in range(1, len(T_pose_joint_name)):
            T_joint_name = T_pose_joint_name[joint_i]
            is_end = T_joint_name.endswith('_end')
            if is_end:
                continue # 末端关节没有channel数据
            A_pose_index = A_pose_joint_map[joint_i]
            start_index = A_pose_joint_start[A_pose_index]
            A_pose_data = retarget_motion_row[start_index: start_index + 3]
            remap_motion_row = np.concatenate((remap_motion_row, A_pose_data), axis=0)

        motion_data.append(remap_motion_row)

    end_time = time.time()
    print("part3_retarget_func 耗时: {:.2f}秒".format(end_time - start_time))
    return np.array(motion_data)
