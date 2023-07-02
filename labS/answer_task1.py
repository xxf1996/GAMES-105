import numpy as np
from scipy.spatial.transform import Rotation as R
from multiprocessing.dummy import Pool
import time


#---------------你的代码------------------#
# translation 和 orientation 都是全局的
def skinning(joint_translation: np.ndarray, joint_orientation: np.ndarray, T_pose_joint_translation: np.ndarray, T_pose_vertex_translation: np.ndarray, skinning_idx: np.ndarray, skinning_weight: np.ndarray):
    """
    skinning函数，给出一桢骨骼的位姿，计算蒙皮顶点的位置
    假设M个关节，N个蒙皮顶点，每个顶点受到最多4个关节影响
    输入：
        joint_translation: (M,3)的ndarray, 目标关节的位置
        joint_orientation: (M,4)的ndarray, 目标关节的旋转，用四元数表示
        T_pose_joint_translation: (M,3)的ndarray, T pose下关节的位置
        T_pose_vertex_translation: (N,3)的ndarray, T pose下蒙皮顶点的位置
        skinning_idx: (N,4)的ndarray, 每个顶点受到哪些关节的影响（假设最多受4个关节影响）
        skinning_weight: (N,4)的ndarray, 每个顶点受到对应关节影响的权重
    输出：
        vertex_translation: (N,3)的ndarray, 蒙皮顶点的位置
    """
    vertex_translation = T_pose_vertex_translation.copy()
    start_time = time.time()

    # def func(i: int):
    #     T_pose_pos: np.ndarray = T_pose_vertex_translation[i]
    #     x = np.zeros(T_pose_pos.shape)
    #     # r = np.zeros(joint_orientation[0].shape)

    #     for j in range(4):
    #         joint_index = skinning_idx[i, j]
    #         joint_weight = skinning_weight[i, j]
    #         joint_pos = T_pose_joint_translation[joint_index]
    #         r = R.from_quat(joint_orientation[joint_index])
    #         x += joint_weight * (r.apply(T_pose_pos - joint_pos) + joint_translation[joint_index])

    #     # r = R.from_quat(r / np.linalg.norm(r))
    #     vertex_translation[i] = x

    # with Pool(32) as p:
    #     p.map(func, range(T_pose_vertex_translation.shape[0]))
    for i in range(T_pose_vertex_translation.shape[0]):
        T_pose_pos: np.ndarray = T_pose_vertex_translation[i]
        x = np.zeros(T_pose_pos.shape)
        # r = np.zeros(joint_orientation[0].shape)
        cur_skinning_idx = skinning_idx[i]
        cur_skinning_weight = skinning_weight[i]


        for j in range(4):
            joint_index = cur_skinning_idx[j]
            joint_weight = cur_skinning_weight[j]
            joint_pos = T_pose_joint_translation[joint_index]
            # FIXME: 旋转矩阵计算量偏大，严重降低FPS
            r = R.from_quat(joint_orientation[joint_index])
            x += joint_weight * (r.apply(T_pose_pos - joint_pos) + joint_translation[joint_index])

        # r = R.from_quat(r / np.linalg.norm(r))
        vertex_translation[i] = x

    #---------------你的代码------------------#

    end_time = time.time()
    print("耗时: {:.2f}秒".format(end_time - start_time))
    print("s/iter: ", (end_time - start_time) / T_pose_vertex_translation.shape[0])

    return vertex_translation