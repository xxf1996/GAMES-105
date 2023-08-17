from viewer import SimpleViewer
import numpy as np
from Lab1_FK_answers import *

# FIXME: 补充末端关节的映射&修复异常关节位置
mocap_joint_map = {
    "RootJoint": "Hips",
    "lHip": "LeftUpLeg",
    "lKnee": "LeftLeg",
    "lAnkle": "LeftFoot",
    "lToeJoint": "LeftToeBase",
    "pelvis_lowerback": "Spine", # 骨盆下背部
    "lTorso_Clavicle": "Spine1", # 锁骨
    "lShoulder": "LeftShoulder",
    "lElbow": "LeftForeArm", # 肘关节
    "lWrist": "LeftHandThumb", # 腕关节
    "rTorso_Clavicle": "Spine1",
    "rShoulder": "RightShoulder",
    "rElbow": "RightForeArm",
    "rWrist": "RightHandThumb",
    "torso_head": "Head",
    "rHip": "RightUpLeg",
    "rKnee": "RightLeg",
    "rAnkle": "RightFoot",
    "rToeJoint": "RightToeBase"
}
'''
基于[SFU MOCAP](https://mocap.cs.sfu.ca/)的骨骼模型进行映射
'''

def exchange_dict(a):
    '''
    交换对象的KV值
    '''
    return dict((v,k) for k,v in a.items())


def part1(viewer, bvh_file_path):
    """
    part1 读取T-pose， 完成part1_calculate_T_pose函数
    """
    joint_name, joint_parent, joint_offset = part1_calculate_T_pose(bvh_file_path)
    viewer.show_rest_pose(joint_name, joint_parent, joint_offset)
    viewer.run()


def part2_one_pose(viewer, bvh_file_path):
    """
    part2 读取一桢的pose, 完成part2_forward_kinematics函数
    """
    joint_name, joint_parent, joint_offset = part1_calculate_T_pose(bvh_file_path)
    motion_data = load_motion_data(bvh_file_path)
    joint_positions, joint_orientations = part2_forward_kinematics(joint_name, joint_parent, joint_offset, motion_data, 0)
    viewer.show_pose(joint_name, joint_positions, joint_orientations)
    viewer.run()


def part2_animation(viewer, bvh_file_path):
    """
    播放完整bvh
    正确完成part2_one_pose后，无需任何操作，直接运行即可
    """
    joint_name, joint_parent, joint_offset = part1_calculate_T_pose(bvh_file_path)
    motion_data = load_motion_data(bvh_file_path)
    frame_num = motion_data.shape[0]
    print("frame_num:", frame_num)
    class UpdateHandle:
        def __init__(self):
            self.current_frame = 0
        def update_func(self, viewer_):
            joint_positions, joint_orientations = part2_forward_kinematics(joint_name, joint_parent, joint_offset, motion_data, self.current_frame)
            viewer.show_pose(joint_name, joint_positions, joint_orientations)
            self.current_frame = (self.current_frame + 1) % frame_num
    handle = UpdateHandle()
    viewer.update_func = handle.update_func
    viewer.run()


def part3_retarget(viewer, T_pose_bvh_path, A_pose_bvh_path):
    """
    将 A-pose的bvh重定向到T-pose上
    Tips:
        我们不需要T-pose bvh的动作数据，只需要其定义的骨骼模型
    """
    # T-pose的骨骼数据
    joint_name, joint_parent, joint_offset = part1_calculate_T_pose(T_pose_bvh_path)
    # A-pose的动作数据
    retarget_motion_data = part3_retarget_func(T_pose_bvh_path, A_pose_bvh_path)

    #播放和上面完全相同
    frame_num = retarget_motion_data.shape[0]
    class UpdateHandle:
        def __init__(self):
            self.current_frame = 0
        def update_func(self, viewer_):
            joint_positions, joint_orientations = part2_forward_kinematics(joint_name, joint_parent, joint_offset, retarget_motion_data, self.current_frame)
            viewer.show_pose(joint_name, joint_positions, joint_orientations)
            self.current_frame = (self.current_frame + 1) % frame_num
    handle = UpdateHandle()
    viewer.update_func = handle.update_func
    viewer.run()


def main():
    # create a viewer
    viewer = SimpleViewer(customJointMap=mocap_joint_map, worldScale=np.array([1.0, 1.0, 1.0]) / 30.0)
    bvh_file_path = "data/0005_JumpRope001.bvh"

    # 请取消注释需要运行的代码
    # part1
    # part1(viewer, bvh_file_path)

    # part2
    # part2_one_pose(viewer, bvh_file_path)
    part2_animation(viewer, bvh_file_path)

    # part3
    # part3_retarget(viewer, "data/walk60.bvh", "data/A_pose_run.bvh")


if __name__ == "__main__":
    main()