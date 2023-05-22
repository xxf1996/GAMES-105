import numpy as np
from scipy.spatial.transform import Rotation as R
from metadata import MetaData

# 位置精度误差
LIMIT_DELTA = 0.01
# 最大迭代次数
MAX_ITER_NUM = 5000

def normalize(v: np.ndarray):
    return v / np.linalg.norm(v)

def part1_inverse_kinematics(meta_data: MetaData, joint_positions: np.ndarray, joint_orientations: np.ndarray, target_pose: np.ndarray):
    """
    完成函数，计算逆运动学
    输入: 
        meta_data: 为了方便，将一些固定信息进行了打包，见上面的meta_data类
        joint_positions: 当前的关节位置，是一个numpy数组，shape为(M, 3)，M为关节数
        joint_orientations: 当前的关节朝向，是一个numpy数组，shape为(M, 4)，M为关节数
        target_pose: 目标位置，是一个numpy数组，shape为(3,)
    输出:
        经过IK后的姿态
        joint_positions: 计算得到的关节位置，是一个numpy数组，shape为(M, 3)，M为关节数
        joint_orientations: 计算得到的关节朝向，是一个numpy数组，shape为(M, 4)，M为关节数
    """

    # TODO: 需要区分自顶向下和自底向上两个不同方向的关节链的处理（自底向上需要逆处理）
    path, path_name, path1, path2 = meta_data.get_path_from_root_to_end()
    print(path1, path2)
    end_joint_index = meta_data.joint_name.index(meta_data.end_joint)
    joint_offset = [
        np.array([0.0, 0.0, 0.0])
    ]
    path_len = len(path)

    for i in range(1, len(meta_data.joint_parent)):
        cur_pos = meta_data.joint_initial_position[i]
        parent_pos: np.ndarray = meta_data.joint_initial_position[meta_data.joint_parent[i]]
        joint_offset.append(cur_pos - parent_pos) # 基于初始的全局位置计算每个关节在父关节局部坐标系中的相对位移

    for _ in range(MAX_ITER_NUM):
        end_pos: np.ndarray = joint_positions[end_joint_index]
        target_diff = np.linalg.norm(end_pos - target_pose)
        # 判断指定末端关节和目标点的距离是否满足误差
        if (target_diff < LIMIT_DELTA):
            break
        # 索引从1开始，避免对指定root关节（位置和朝向）进行改变
        # 基于CCD方法进行优化求解
        for i in range(1, path_len):
            joint_index = path[i]
            if i == path_len - 1:
                continue
            cur_pos: np.ndarray = joint_positions[joint_index]
            end_pos: np.ndarray = joint_positions[end_joint_index]
            # 当前关节到末端关节的方向
            cur_direction = normalize(end_pos - cur_pos)
            # 当前关节到目标点的方向（基于CCD求IK的本质可知，实际上就是将关节到末端关节之间的轴旋转到该方向，即为该轴方向上的最近距离点）
            next_direction = normalize(target_pose - cur_pos)
            # 叉乘方向就是旋转轴
            rotation_axis = normalize(np.cross(cur_direction, next_direction))
            # 基于点乘和cos可以求出两个向量之间的角度
            rotation_angle: float = np.arccos(np.dot(cur_direction, next_direction))
            joint_orientations[joint_index] = R.from_rotvec(rotation_angle * rotation_axis).as_quat()

            parent_pos = cur_pos
            parent_orientation = R.from_quat(joint_orientations[joint_index])

            # NOTICE: 求出当前关节的新朝向后，需要基于FK方法更新所有后续子关节的位置（主要就是要得到最新的末端关节位置）
            for j in range(i + 1, path_len):
                joint_index = path[j]
                cur_offset: np.ndarray = joint_offset[joint_index]
                cur_pos = parent_pos + parent_orientation.apply(cur_offset)
                joint_positions[joint_index] = cur_pos
                parent_orientation = R.from_quat(joint_orientations[joint_index])
                parent_pos = cur_pos

    # NOTICE: 对其他的关节位置进行一遍FK计算，确保其他关节位置同时发生改变
    for i in range(1, len(meta_data.joint_name)):
        if i in path:
            continue
        parent_index = meta_data.joint_parent[i]
        parent_orientation = R.from_quat(joint_orientations[parent_index])
        parent_pos: np.ndarray = joint_positions[parent_index]
        cur_offset: np.ndarray = joint_offset[i]
        cur_pos = parent_pos + parent_orientation.apply(cur_offset)
        joint_positions[i] = cur_pos


    end_pos: np.ndarray = joint_positions[end_joint_index]
    target_diff = np.linalg.norm(end_pos - target_pose)

    print('target diff:', target_diff)

    return joint_positions, joint_orientations

def part2_inverse_kinematics(meta_data, joint_positions, joint_orientations, relative_x, relative_z, target_height):
    """
    输入lWrist相对于RootJoint前进方向的xz偏移，以及目标高度，IK以外的部分与bvh一致
    """
    
    return joint_positions, joint_orientations

def bonus_inverse_kinematics(meta_data, joint_positions, joint_orientations, left_target_pose, right_target_pose):
    """
    输入左手和右手的目标位置，固定左脚，完成函数，计算逆运动学
    """
    
    return joint_positions, joint_orientations