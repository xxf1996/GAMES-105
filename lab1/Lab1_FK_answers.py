from __future__ import annotations
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


def part3_retarget_func(T_pose_bvh_path, A_pose_bvh_path):
    """
    将 A-pose的bvh重定向到T-pose上
    输入: 两个bvh文件的路径
    输出: 
        motion_data: np.ndarray，形状为(N,X)的numpy数组，其中N为帧数，X为Channel数。retarget后的运动数据
    Tips:
        两个bvh的joint name顺序可能不一致哦(
        as_euler时也需要大写的XYZ
    """
    motion_data = None
    return motion_data
