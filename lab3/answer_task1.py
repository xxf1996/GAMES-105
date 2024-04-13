import numpy as np
from scipy.spatial.transform import Rotation as R
from bvh_loader import BVHMotion
from physics_warpper import PhysicsInfo

def convert_rotation_to_axis_angle(rotation: np.ndarray):
    '''
    将四元数转为轴角表示，返回轴*角；来源：chatGPT
        rotation: 旋转四元数
    '''
    rotation = rotation / np.linalg.norm(rotation)
    angle = 2.0 * np.arccos(rotation[3]) # w分量（基于w = cos(θ/2)）
    sin_half_angle = np.sqrt(1.0 - rotation[3]**2)
    axis = rotation[[0, 1, 2]] / sin_half_angle if sin_half_angle > 1e-8 else np.array([1.0, 0.0, 0.0])
    return axis * angle


def part1_cal_torque(pose: np.ndarray, physics_info: PhysicsInfo, **kargs):
    '''
    输入：
        pose: (20,4)的numpy数组，表示每个关节的目标旋转(相对于父关节的)
        physics_info: PhysicsInfo类，包含了当前的物理信息，参见physics_warpper.py
        **kargs: 指定参数，可能包含kp,kd
    输出：
        global_torque: (20,3)的numpy数组，表示每个关节的全局坐标下的目标力矩，根节点力矩会被后续代码无视
    '''
    # ------一些提示代码，你可以随意修改------------#
    total_kp = kargs.get('kp', 500.0) # 需要自行调整kp和kd！ 而且也可以是一个数组，指定每个关节的kp和kd
    total_kd = kargs.get('kd', 200.0) # NOTICE: 这里kd值太小时，抖动很明显
    parent_index = physics_info.parent_index
    joint_name = physics_info.joint_name
    joint_orientation = physics_info.get_joint_orientation()
    joint_avel = physics_info.get_body_angular_velocity()
    # joint_translation = physics_info.get_joint_translation()

    global_torque = np.zeros((20,3))

    # for i in range(1, len(pose)):
    #     joint_parent = parent_index[i]
    #     parent_orientation = R.from_quat(joint_orientation[joint_parent])
    #     parent_translation = joint_translation[joint_parent]
    #     # cur_orientation = R.from_quat(joint_orientation[i])
    #     cur_translation = joint_translation[i]
    #     parent_l = parent_orientation.inv().apply((cur_translation - parent_translation))
    #     target_orientation = parent_orientation * R.from_quat(pose[joint_parent])
    #     target_translation = parent_translation + target_orientation.apply(parent_l)
    #     local_avel = parent_orientation.inv().apply(joint_avel[joint_parent])
    #     local_torque = kp * parent_orientation.inv().apply(target_translation - cur_translation) - kd * local_avel * np.linalg.norm(parent_l)
    #     global_torque_val = np.clip(parent_orientation.apply(local_torque), -100, 100)
    #     # print(local_torque, global_torque_val)
    #     global_torque[parent_index] = global_torque_val

    # joint_l = np.array([1, 0, 0])

    for i in range(0, len(pose)):
        kp = kargs.get("kp_{}".format(i), total_kp)
        kd = kargs.get("kd_{}".format(i), total_kd)
        joint_parent = parent_index[i]
        cur_orientation = R.from_quat(joint_orientation[i])
        target_quat = pose[i]
        # root关节需要特殊处理（因为没有父关节）
        if joint_parent == -1:
            angle = np.arccos(np.dot(target_quat, cur_orientation.as_quat()))
            if np.abs(angle) > np.pi * 0.5:
                target_quat = -target_quat # 基于单位四元数的特性进行负向取值
            target_rotation = R.from_quat(target_quat)
            rotation_error = convert_rotation_to_axis_angle((target_rotation.inv() * cur_orientation).as_quat())
            global_torque[i] = kp * rotation_error - kd * joint_avel[i]
            continue
        parent_orientation = R.from_quat(joint_orientation[joint_parent])
        local_rotation = parent_orientation.inv() * cur_orientation
        angle = np.arccos(np.dot(target_quat, local_rotation.as_quat()))
        # 这里需要判断四元数的夹角，因为有的目标姿势可能跟当前姿势夹角差别大，导致用力不对
        if np.abs(angle) > np.pi * 0.5:
            target_quat = -target_quat # 基于单位四元数的特性进行负向取值
        target_rotation = R.from_quat(target_quat)
        # 旋转的误差（这里并不能直接套用公式做减法，或许四元数可以直接运算？）
        rotation_error = convert_rotation_to_axis_angle((target_rotation.inv() * local_rotation).as_quat())
        # NOTICE: 这里直接基于Rotation类提供的as_rotvec()方法转为轴角表示得到的结果并不对！
        # rotation_error = (target_rotation.inv() * local_rotation).as_rotvec()
        # 角速度也需要转换到当前关节坐标系下！
        local_avel = cur_orientation.inv().apply(joint_avel[i])
        avel_error = 0 - local_avel
        local_torque = kp * rotation_error + kd * avel_error
        global_torque_val = np.clip(cur_orientation.apply(local_torque), -800, 800) # 避免关节力矩过大
        global_torque[i] = global_torque_val

    return global_torque

def part2_cal_float_base_torque(target_position: np.ndarray, pose: np.ndarray, physics_info: PhysicsInfo, **kargs):
    '''
    输入： target_position: (3,)的numpy数组，表示根节点的目标位置，其余同上
    输出： global_root_force: (3,)的numpy数组，表示根节点的全局坐标下的辅助力
          global_torque: 同上
    注意：
        1. 你需要自己计算kp和kd，并且可以通过kargs调整part1中的kp和kd
        2. global_torque[0]在track静止姿态时会被无视，但是track走路时会被加到根节点上，不然无法保持根节点朝向
    '''
    # TODO: 系数调整（太难了，完全就是炼丹）
    kp = kargs.get('root_kp', 4500.0) # 需要自行调整root的kp和kd！
    kd = kargs.get('root_kd', 50.0)
    global_torque = part1_cal_torque(pose, physics_info, kp_0 = 5000.0, kd_0 = 200.0, kp = 500.0, kd = 50.0)
    root_position, root_velocity = physics_info.get_root_pos_and_vel()
    global_root_force = kp * (target_position - root_position) - kd * root_velocity
    return global_root_force, global_torque

def part3_cal_static_standing_torque(bvh: BVHMotion, physics_info: PhysicsInfo):
    '''
    输入： bvh: BVHMotion类，包含了当前的动作信息，参见bvh_loader.py
    其余同上
    Tips:
        只track第0帧就能保持站立了
        为了保持平衡可以把目标的根节点位置适当前移，比如把根节点位置和左右脚的中点加权平均
        为了仿真稳定最好不要在Toe关节上加额外力矩
    '''
    tar_pos = bvh.joint_position[0][0]
    pose = bvh.joint_rotation[0]
    joint_name: list[str] = physics_info.joint_name
    # apply_joints = ["lHip", "lKnee", "lAnkle", "rHip", "rKnee", "rAnkle"]
    # 需要应用反馈关节力矩的关节列表
    apply_joints = ["lHip", "lKnee", "rHip", "rKnee"]
    kp = 10000.0
    kd = 500.0
    joint_positions = physics_info.get_joint_translation()
    joint_vel = physics_info.get_body_velocity()
    joint_mass = physics_info.get_body_mass()
    # joint_orientation = physics_info.get_joint_orientation()
    # 基于质量的加权平均速度
    com_vel = np.array([0.0, 0.0, 0.0])
    # 质心位置
    com = np.array([0.0, 0.0, 0.0])
    total_mass = 0.0
    for i in range(len(joint_name)):
        # cur_orientation = R.from_quat(joint_orientation[i])
        mass = joint_mass[i]
        pos = joint_positions[i]
        vel = joint_vel[i]
        com_vel += mass * vel
        com += mass * pos
        total_mass += mass

    com_vel = com_vel / total_mass
    com = com / total_mass
    # 适当前移
    tar_pos[[0, 2]] = tar_pos[[0, 2]] * 0.8 + joint_positions[9][[0, 2]] * 0.1 + joint_positions[10][[0, 2]] * 0.1
    # tar_pos[1] = 0
    root_pos = joint_positions[0]
    # root_pos[1] = 0

    torque = part1_cal_torque(pose, physics_info)
    # 雅克比虚拟反馈力
    feedback_force = kp * (tar_pos - root_pos) - kd * com_vel

    for name in apply_joints:
        # TODO: 不知道力的应用位置是质心还是根关节位置？
        joint_index = joint_name.index(name)
        torque[joint_index] += np.cross(root_pos - joint_positions[joint_index], feedback_force)

    # print(com_vel, torque[r_ankle], torque[l_ankle])

    return torque

