import numpy as np
from scipy.spatial.transform import Rotation as R
from metadata import MetaData

# 位置精度误差
LIMIT_DELTA = 0.01
# 最大迭代次数
MAX_ITER_NUM = 100

def normalize(v: np.ndarray):
    return v / np.linalg.norm(v)

def part1_inverse_kinematics(meta_data: MetaData, joint_positions: np.ndarray, joint_orientations: np.ndarray, target_pose: np.ndarray, skip_fk = False):
    """
    完成函数，计算逆运动学
    输入: 
        meta_data: 为了方便，将一些固定信息进行了打包，见上面的meta_data类
        joint_positions: 当前的关节位置，是一个numpy数组，shape为(M, 3)，M为关节数
        joint_orientations: 当前的关节朝向，是一个numpy数组，shape为(M, 4)，M为关节数
        target_pose: 目标位置，是一个numpy数组，shape为(3,)
        skip_fk: 是否跳过不属于IK关键链的其他关节的FK计算，跳过FK计算即相当于固定其它的关节位置和朝向
    输出:
        经过IK后的姿态
        joint_positions: 计算得到的关节位置，是一个numpy数组，shape为(M, 3)，M为关节数
        joint_orientations: 计算得到的关节朝向，是一个numpy数组，shape为(M, 4)，M为关节数
    """

    path, path_name, path1, path2 = meta_data.get_path_from_root_to_end()
    print(path, path1, path2)
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
            is_reverse = path[i + 1] in path2
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
            # if is_reverse:
            #     rotation_angle *= -1
            # 这里计算的只是当前需要进行的旋转，因此还需要应用到之前的旋转向量（朝向）上
            rotation = R.from_rotvec(rotation_angle * rotation_axis)
            if is_reverse:
                next_joint = path[i + 1]
                # 逆向的关节链实际上就是旋转的父关节的朝向，而父关节的朝向先要反向才能基于当前关节点的位置进行旋转，旋转完后就再反向变回父关节的朝向
                cur_orientation = R.from_quat(joint_orientations[next_joint]).inv()
                joint_orientations[next_joint] = (rotation * cur_orientation).inv().as_quat()
            else:
                cur_orientation = R.from_quat(joint_orientations[joint_index])
                joint_orientations[joint_index] = (rotation * cur_orientation).as_quat()

            prev_pos = cur_pos
            prev_orientation = R.from_quat(joint_orientations[joint_index])

            # NOTICE: 求出当前关节的新朝向后，需要基于FK方法更新所有后续子关节的位置（主要就是要得到最新的末端关节位置）
            for j in range(i + 1, path_len):
                joint_index = path[j]
                is_reverse = joint_index in path2
                cur_orientation = R.from_quat(joint_orientations[joint_index])
                if is_reverse:
                    child_path = path[j - 1]
                    child_offset = joint_offset[child_path]
                    # 根据FK计算公式可以倒推父关节位置
                    cur_pos = prev_pos - cur_orientation.apply(child_offset)
                    # 由于是逆向的改变，因此父关节的朝向也发生了变化（因为并不能基于FK的原方向进行自动影响）
                    if j != i + 1 and j != 0:
                        cur_orientation = rotation * cur_orientation.inv()
                        joint_orientations[joint_index] = cur_orientation.inv().as_quat()
                else:
                    cur_offset: np.ndarray = joint_offset[joint_index]
                    cur_pos = prev_pos + prev_orientation.apply(cur_offset)
                joint_positions[joint_index] = cur_pos
                # joint_orientations[joint_index] = cur_orientation.as_quat()
                prev_orientation = cur_orientation
                prev_pos = cur_pos

    if not skip_fk:
        # NOTICE: 对其他的关节位置进行一遍FK计算，确保其他关节位置同时发生改变
        for i in range(0, len(meta_data.joint_name)):
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

def part2_inverse_kinematics(meta_data: MetaData, joint_positions: np.ndarray, joint_orientations: np.ndarray, relative_x: float, relative_z: float, target_height: float):
    """
    输入lWrist相对于RootJoint前进方向的xz偏移，以及目标高度，IK以外的部分与bvh一致
    """

    root_pos: np.ndarray = joint_positions[meta_data.joint_name.index(meta_data.root_joint)]
    wrist_index = meta_data.joint_name.index("lWrist_end")
    shouler_index = meta_data.joint_name.index("lShoulder")
    target_pos = np.array([root_pos[0] + relative_x, target_height, root_pos[2] + relative_z])
    path = [wrist_index]
    parent_index = meta_data.joint_parent[wrist_index]
    joint_offset = [
        np.array([0.0, 0.0, 0.0])
    ]

    for i in range(1, len(meta_data.joint_parent)):
        cur_pos = meta_data.joint_initial_position[i]
        parent_pos: np.ndarray = meta_data.joint_initial_position[meta_data.joint_parent[i]]
        joint_offset.append(cur_pos - parent_pos) # 基于初始的全局位置计算每个关节在父关节局部坐标系中的相对位移

    while parent_index > -1 and parent_index != shouler_index:
        path.append(parent_index)
        parent_index = meta_data.joint_parent[parent_index]

    path = list(reversed(path))
    path_len = len(path)

    # 实际上就是求解lShoulder到lWrist_end这条关节链的IK（lShoulder为根关节，lWrist_end为目标关节）
    for _ in range(MAX_ITER_NUM):
        end_pos: np.ndarray = joint_positions[wrist_index]
        target_diff = np.linalg.norm(end_pos - target_pos)
        # 判断指定末端关节和目标点的距离是否满足误差
        if (target_diff < LIMIT_DELTA):
            break

        # 索引从1开始，避免对指定root关节（位置和朝向）进行改变
        # 基于CCD方法进行优化求解
        for i in range(path_len):
            joint_index = path[i]
            if i == path_len - 1:
                continue
            cur_pos: np.ndarray = joint_positions[joint_index]
            end_pos: np.ndarray = joint_positions[wrist_index]
            # 当前关节到末端关节的方向
            cur_direction = normalize(end_pos - cur_pos)
            # 当前关节到目标点的方向（基于CCD求IK的本质可知，实际上就是将关节到末端关节之间的轴旋转到该方向，即为该轴方向上的最近距离点）
            next_direction = normalize(target_pos - cur_pos)
            # 叉乘方向就是旋转轴
            rotation_axis = normalize(np.cross(cur_direction, next_direction))
            # 基于点乘和cos可以求出两个向量之间的角度
            rotation_angle: float = np.arccos(np.dot(cur_direction, next_direction))
            # 这里计算的只是当前需要进行的旋转，因此还需要应用到之前的旋转向量（朝向）上
            rotation = R.from_rotvec(rotation_angle * rotation_axis)
            cur_orientation = R.from_quat(joint_orientations[joint_index])
            try:
                joint_orientations[joint_index] = (rotation * cur_orientation).as_quat()
            except:
                joint_orientations[joint_index] = cur_orientation.as_quat()

            prev_pos = cur_pos
            prev_orientation = R.from_quat(joint_orientations[joint_index])

            # NOTICE: 求出当前关节的新朝向后，需要基于FK方法更新所有后续子关节的位置（主要就是要得到最新的末端关节位置）
            for j in range(i + 1, path_len):
                joint_index = path[j]
                cur_orientation = R.from_quat(joint_orientations[joint_index])
                cur_offset: np.ndarray = joint_offset[joint_index]
                cur_pos = prev_pos + prev_orientation.apply(cur_offset)
                joint_positions[joint_index] = cur_pos
                # joint_orientations[joint_index] = cur_orientation.as_quat()
                prev_orientation = cur_orientation
                prev_pos = cur_pos

    return joint_positions, joint_orientations

def bonus_inverse_kinematics(meta_data: MetaData, joint_positions: np.ndarray, joint_orientations, left_target_pose: np.ndarray, right_target_pose: np.ndarray,):
    """
    输入左手和右手的目标位置，固定左脚，完成函数，计算逆运动学
    """
    joint_positions, joint_orientations = part1_inverse_kinematics(meta_data, joint_positions, joint_orientations, left_target_pose)
    # NOTICE：从两个关节链重叠部分的父节点开始，构建一个新的关节链，可以避免对已经计算的部分的干扰
    meta_data.root_joint = "lowerback_torso"
    meta_data.end_joint = "rWrist_end"
    joint_positions, joint_orientations = part1_inverse_kinematics(meta_data, joint_positions, joint_orientations, right_target_pose, skip_fk=True)


    return joint_positions, joint_orientations