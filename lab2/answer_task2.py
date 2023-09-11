# 以下部分均为可更改部分
from __future__ import annotations
from answer_task1 import *
from Viewer.controller import Controller
from scipy.spatial import KDTree
from typing import Optional


WALK_MAX_V: float = 2.5
WALK_MIN_V: float = 0.01
MOTION_V = [0, WALK_MIN_V, WALK_MAX_V, 5.0]
'''各个动作的速度区间，依次增大'''

def get_face_xz_from_rotation(quat: np.ndarray, motion: BVHMotion):
    (Ry, _) = motion.decompose_rotation_with_yaxis(quat)
    Ry = R.from_quat(Ry)
    face_direction = Ry.apply(np.array([0, 0, 1]))

    return face_direction[[0, 2]]

def quat_linear_interpolation(q0: np.ndarray, q1: np.ndarray, a: float):
    angle = np.arccos(np.dot(q0, q1))
    # 避免夹角过大插值出现奇异值，利用单位四元数的特性进行反向
    if np.abs(angle) > np.pi * 0.5:
        q0 = -q0
    # 简单线性插值
    q = (1.0 - a) * q0 + a * q1
    q = q / np.linalg.norm(q)

    return q


class CharacterController():
    def __init__(self, controller: Controller) -> None:
        self.motions: list[BVHMotion] = []
        self.long_motion = BVHMotion('motion_material/kinematic_motion/long_walk.bvh')
        self.long_motion_vel = np.zeros(shape=(self.long_motion.motion_length, 3))
        self.long_motion_vectors = np.zeros(shape=(self.long_motion.motion_length, 7))
        self.long_motion_translation, self.long_motion_orientation, self.long_motion_simulation_translation, self.long_motion_simulation_orientation = self.long_motion.batch_forward_kinematics()
        self.controller = controller
        self.cur_frame = 0
        self.prev_motion_type = 0
        self.simulation_arrow = None
        self.simulation_point = None
        self.long_motion_tree: Optional[KDTree] = None

        self.compute_motion_vel()
        pass

    def get_motion_type(self, v: float):
        if v < WALK_MIN_V:
            return 0
        if v > WALK_MAX_V:
            return 2
        return 1

    def get_motion_basis(self, cur_v: float, next_v: float):
        '''
        基于当前动作和下一动作之间的速度区间，基于当前速度进行插值；

        即越靠近下一动作的速度区间越接近于1
        '''
        cur_motion_type = self.get_motion_type(cur_v)
        next_motion_type = self.get_motion_type(next_v)

        min_v = MOTION_V[cur_motion_type]
        max_v = MOTION_V[cur_motion_type + 1]

        # 降速时
        if cur_motion_type > next_motion_type:
            return 1.0 - (cur_v - min_v) / (max_v - min_v)

        # 提速时
        return 1.0 - (max_v - cur_v) / (max_v - min_v)

    def load_motions(self):
        '''
        用于加载motion graph所需要的bvh motion
        '''
        self.motions.append(BVHMotion('motion_material/idle.bvh'))
        self.motions.append(BVHMotion('motion_material/walk_forward.bvh'))
        self.motions.append(BVHMotion('motion_material/run_forward.bvh'))
        self.cur_root_pos = self.motions[0].joint_position[0, 0]
        self.cur_root_rot = self.motions[0].joint_rotation[0, 0]

    def motion_graph(self,
                     desired_pos_list: np.ndarray,
                     desired_rot_list: np.ndarray,
                     desired_vel_list: np.ndarray
                     ):
        '''
        一个简单的motion graph, 动作根据速度不同分别落在站立、行走和跑步这三个动作之间
        '''
        cur_v = desired_vel_list[0]
        cur_v_norm = np.linalg.norm(cur_v)
        # 基于的下一步速度间隔越大，动作反应越灵敏
        next_v = desired_vel_list[4]
        next_v_norm = np.linalg.norm(next_v)
        next_pos_xz = desired_pos_list[1, [0, 2]]
        next_face_direction = desired_rot_list[1]
        cur_motion_type = self.get_motion_type(cur_v_norm)
        next_motion_type = self.get_motion_type(next_v_norm)

        # 当motion变更时需要从0开始播放
        if self.prev_motion_type != cur_motion_type:
            self.cur_frame = 0

        self.prev_motion_type = cur_motion_type
        k = 0.05 # 1/20
        next_pos_xz = (1.0 - k) * self.cur_root_pos[[0, 2]] + k * next_pos_xz
        next_face_direction = quat_linear_interpolation(self.cur_root_rot, next_face_direction, k)

        print("cur_motion_type: ", cur_motion_type, "; next_motion_type: ", next_motion_type)

        # motion状态没有变化时，直接播放
        if cur_motion_type == next_motion_type:
            new_motion = self.motions[cur_motion_type].raw_copy()
            new_motion.joint_position[self.cur_frame, 0, [0, 2]] = next_pos_xz
            new_motion.joint_rotation[self.cur_frame, 0] = next_face_direction
            joint_translation_fk, joint_orientation_fk, simulation_translation, simulation_orientation = new_motion.batch_forward_kinematics()
            joint_translation = joint_translation_fk[self.cur_frame]
            joint_orientation = joint_orientation_fk[self.cur_frame]
            self.update_simulation_draw(
                simulation_translation[self.cur_frame],
                simulation_orientation[self.cur_frame],
                new_motion
            )

            return joint_translation, joint_orientation

        motion_basis = self.get_motion_basis(cur_v_norm, next_v_norm)
        cur_motion = self.motions[cur_motion_type].raw_copy()
        next_motion = self.motions[next_motion_type].raw_copy()
        cur_motion.joint_position[self.cur_frame, 0, [0, 2]] = next_pos_xz
        cur_motion.joint_rotation[self.cur_frame, 0] = next_face_direction
        next_motion.joint_position[0, 0, [0, 2]] = next_pos_xz
        next_motion.joint_rotation[0, 0] = next_face_direction
        # FIXME: BVH的关节顺序可能不一致
        cur_motion.joint_position[self.cur_frame] = (1.0 - motion_basis) * cur_motion.joint_position[self.cur_frame] + motion_basis * next_motion.joint_position[0]
        for i in range(cur_motion.joint_rotation.shape[1]):
            cur_motion.joint_rotation[self.cur_frame, i] = quat_linear_interpolation(cur_motion.joint_rotation[self.cur_frame, i], next_motion.joint_rotation[0, i], motion_basis)
        joint_translation_fk, joint_orientation_fk, simulation_translation, simulation_orientation = cur_motion.batch_forward_kinematics()
        joint_translation = joint_translation_fk[self.cur_frame]
        joint_orientation = joint_orientation_fk[self.cur_frame]
        self.update_simulation_draw(
            simulation_translation[self.cur_frame],
            simulation_orientation[self.cur_frame],
            cur_motion
        )

        return joint_translation, joint_orientation

    def compute_motion_vel(self):
        for i in range(self.long_motion.motion_length):
            cur_pos = self.long_motion_simulation_translation[i]
            prev_pos = self.long_motion_simulation_translation[i - 1]
            self.long_motion_vel[i] = (cur_pos - prev_pos) / self.long_motion.frame_time
            # NOTICE: 提前计算好需要匹配的向量，减少耗时
            self.long_motion_vectors[i] = np.concatenate((self.long_motion_simulation_orientation[i], self.long_motion_vel[i]))
        self.long_motion_tree = KDTree(self.long_motion_vectors)

    def update_simulation_draw(self, translation: np.ndarray, orientation: np.ndarray, motion: BVHMotion):
        # NOTICE: 这一步就是想旋转方向投影到xz平面上，类似于求face direction
        face_direction = R.from_quat(orientation).apply(np.array([0, 0, 1]))
        face_xz = face_direction[[0, 2]]
        # print("face xz:", face_xz, orientation)
        if self.simulation_arrow == None:
            self.simulation_arrow = self.controller.viewer.create_arrow(
                translation,
                face_xz,
                width=0.05,
                length=0.5
            )
        else:
            self.simulation_arrow.setPos(*translation)
            self.simulation_arrow.setQuat(self.controller.viewer.get_quat_from_forward_xz(
                face_xz
            ))

    def update_state_test(self,
                     desired_pos_list: np.ndarray,
                     desired_rot_list: np.ndarray,
                     desired_vel_list: np.ndarray,
                     desired_avel_list: np.ndarray,
                     current_gait
                     ):
        '''
        此接口会被用于获取新的期望状态
        Input: 平滑过的手柄输入,包含了现在(第0帧)和未来20,40,60,80,100帧的期望状态,以及一个额外输入的步态
        简单起见你可以先忽略步态输入,它是用来控制走路还是跑步的
            desired_pos_list: 期望位置, 6x3的矩阵, 每一行对应0，20，40...帧的期望位置(水平)， 期望位置可以用来拟合根节点位置也可以是质心位置或其他
            desired_rot_list: 期望旋转, 6x4的矩阵, 每一行对应0，20，40...帧的期望旋转(水平), 期望旋转可以用来拟合根节点旋转也可以是其他
            desired_vel_list: 期望速度, 6x3的矩阵, 每一行对应0，20，40...帧的期望速度(水平), 期望速度可以用来拟合根节点速度也可以是其他
            desired_avel_list: 期望角速度, 6x3的矩阵, 每一行对应0，20，40...帧的期望角速度(水平), 期望角速度可以用来拟合根节点角速度也可以是其他

        Output: 同作业一,输出下一帧的关节名字,关节位置,关节旋转
            joint_name: List[str], 代表了所有关节的名字
            joint_translation: np.ndarray，形状为(M, 3)的numpy数组，包含着所有关节的全局位置
            joint_orientation: np.ndarray，形状为(M, 4)的numpy数组，包含着所有关节的全局旋转(四元数)
        Tips:
            输出三者顺序需要对应
            controller 本身有一个move_speed属性,是形状(3,)的ndarray,
            分别对应着面朝向移动速度,侧向移动速度和向后移动速度.目前根据LAFAN的统计数据设为(1.75,1.5,1.25)
            如果和你的角色动作速度对不上,你可以在init或这里对属性进行修改
        '''
        # 一个简单的例子，输出第i帧的状态
        joint_name = self.motions[0].joint_name
        next_pos_xz = desired_pos_list[1, [0, 2]]
        next_v = desired_vel_list[1]
        next_v_norm = np.linalg.norm(next_v)
        next_frame = self.cur_frame + 1
        basis = int(np.round(next_v_norm * 0.6))
        if next_v_norm < 0.001:
            next_frame = 20
            # basis = 10
        else:
            next_frame = self.cur_frame + basis
        k = 0.05 # 1/20
        next_frame = (next_frame + 1) % self.motions[0].motion_length
        # print("next_v: ", next_v)
        # print("current_gait: ", current_gait)
        next_face_direction = desired_rot_list[1]
        next_pos_xz = (1.0 - k) * self.cur_root_pos[[0, 2]] + k * next_pos_xz
        next_face_direction = quat_linear_interpolation(self.cur_root_rot, next_face_direction, k)
        new_motion = self.motions[0].raw_copy()
        # new_motion.translation_and_rotation(self.cur_frame, next_pos_xz, next_face_xz)
        new_motion.joint_position[self.cur_frame, 0, [0, 2]] = next_pos_xz
        new_motion.joint_rotation[self.cur_frame, 0] = next_face_direction
        new_motion.joint_position[next_frame, 0, [0, 2]] = next_pos_xz
        new_motion.joint_rotation[next_frame, 0] = next_face_direction
        new_motion.joint_position[self.cur_frame] = new_motion.joint_position[self.cur_frame] * (1.0 - k) + k * new_motion.joint_position[next_frame]
        for i in range(new_motion.joint_rotation.shape[1]):
            new_motion.joint_rotation[self.cur_frame, i] = quat_linear_interpolation(new_motion.joint_rotation[self.cur_frame, i], new_motion.joint_rotation[next_frame, i], k)

        joint_translation_fk, joint_orientation_fk = new_motion.batch_forward_kinematics()
        joint_translation = joint_translation_fk[self.cur_frame]
        joint_orientation = joint_orientation_fk[self.cur_frame]

        self.cur_root_pos = joint_translation[0]
        self.cur_root_rot = joint_orientation[0]
        self.cur_frame = next_frame

        return joint_name, joint_translation, joint_orientation

    def motion_matching(
            self,
            desired_pos_list: np.ndarray,
            desired_rot_list: np.ndarray,
            desired_vel_list: np.ndarray,
            desired_avel_list: np.ndarray,
            current_gait
    ):
        # TODO: motion matching匹配最近邻动作的细节处理，比如long motion需不需要做朝向对齐；
        self.long_motion
        next_frame = self.cur_frame
        min_dist = 1e10
        cur_simulation_v = np.concatenate((desired_rot_list[0], desired_vel_list[0]))
        next_20_simulation_v = np.concatenate((desired_rot_list[1], desired_vel_list[1]))
        next_40_simulation_v = np.concatenate((desired_rot_list[2], desired_vel_list[2]))
        # TODO: 需要用kd tree进行一下遍历的加速
        min_dist, next_frame = self.long_motion_tree.query(cur_simulation_v)
        # for i in range(self.long_motion.motion_length):
        #     if i == self.cur_frame:
        #         continue
        #     # TODO: 对simulation bone和期望位置和速度进行匹配（或许还要加上角速度？），基于欧氏距离
        #     next_20 = (i + 20) % self.long_motion.motion_length
        #     next_40 = (i + 40) % self.long_motion.motion_length
        #     cur_object_v = self.long_motion_vectors[i]
        #     next_20_object_v = self.long_motion_vectors[next_20]
        #     next_40_object_v = self.long_motion_vectors[next_40]
        #     cur_dist = np.linalg.norm(cur_simulation_v - cur_object_v) + np.linalg.norm(next_20_simulation_v - next_20_object_v) + np.linalg.norm(next_40_simulation_v - next_40_object_v)
        #     if cur_dist < min_dist:
        #         min_dist = cur_dist
        #         next_frame = i

        if np.abs(next_frame - self.cur_frame) < 5:
            self.cur_frame = (self.cur_frame + 1) % self.long_motion.motion_length
        else:
            self.cur_frame = next_frame

        joint_translation: np.ndarray = self.long_motion_translation[self.cur_frame].copy()
        joint_orientation: np.ndarray = self.long_motion_orientation[self.cur_frame]
        simulation_translation: np.ndarray = self.long_motion_simulation_translation[self.cur_frame]
        offset_translation = desired_pos_list[0] - simulation_translation

        for i in range(joint_translation.shape[0]):
            joint_translation[i] += offset_translation

        self.cur_root_pos = desired_pos_list[0]
        self.cur_root_rot = joint_orientation[0]

        return self.long_motion.joint_name, joint_translation, joint_orientation




    def update_state(self,
                     desired_pos_list: np.ndarray,
                     desired_rot_list: np.ndarray,
                     desired_vel_list: np.ndarray,
                     desired_avel_list: np.ndarray,
                     current_gait
                     ):
        '''
        此接口会被用于获取新的期望状态
        Input: 平滑过的手柄输入,包含了现在(第0帧)和未来20,40,60,80,100帧的期望状态,以及一个额外输入的步态
        简单起见你可以先忽略步态输入,它是用来控制走路还是跑步的
            desired_pos_list: 期望位置, 6x3的矩阵, 每一行对应0，20，40...帧的期望位置(水平)， 期望位置可以用来拟合根节点位置也可以是质心位置或其他
            desired_rot_list: 期望旋转, 6x4的矩阵, 每一行对应0，20，40...帧的期望旋转(水平), 期望旋转可以用来拟合根节点旋转也可以是其他
            desired_vel_list: 期望速度, 6x3的矩阵, 每一行对应0，20，40...帧的期望速度(水平), 期望速度可以用来拟合根节点速度也可以是其他
            desired_avel_list: 期望角速度, 6x3的矩阵, 每一行对应0，20，40...帧的期望角速度(水平), 期望角速度可以用来拟合根节点角速度也可以是其他
        
        Output: 同作业一,输出下一帧的关节名字,关节位置,关节旋转
            joint_name: List[str], 代表了所有关节的名字
            joint_translation: np.ndarray，形状为(M, 3)的numpy数组，包含着所有关节的全局位置
            joint_orientation: np.ndarray，形状为(M, 4)的numpy数组，包含着所有关节的全局旋转(四元数)
        Tips:
            输出三者顺序需要对应
            controller 本身有一个move_speed属性,是形状(3,)的ndarray,
            分别对应着面朝向移动速度,侧向移动速度和向后移动速度.目前根据LAFAN的统计数据设为(1.75,1.5,1.25)
            如果和你的角色动作速度对不上,你可以在init或这里对属性进行修改
        '''
        # 一个简单的例子，输出第i帧的状态
        (joint_translation, joint_orientation) = self.motion_graph(desired_pos_list, desired_rot_list, desired_vel_list)
        joint_name = self.motions[self.prev_motion_type].joint_name

        self.cur_root_pos = joint_translation[0]
        self.cur_root_rot = joint_orientation[0]
        self.cur_frame = (self.cur_frame + 1) % self.motions[self.prev_motion_type].motion_length

        return joint_name, joint_translation, joint_orientation


    def sync_controller_and_character(self, controller: Controller, character_state: tuple[list[str], np.ndarray, np.ndarray]):
        '''
        这一部分用于同步你的角色和手柄的状态
        更新后很有可能会出现手柄和角色的位置不一致，这里可以用于修正
        让手柄位置服从你的角色? 让角色位置服从手柄? 或者插值折中一下?
        需要你进行取舍
        Input: 手柄对象，角色状态
        手柄对象我们提供了set_pos和set_rot接口,输入分别是3维向量和四元数,会提取水平分量来设置手柄的位置和旋转
        角色状态实际上是一个tuple, (joint_name, joint_translation, joint_orientation),为你在update_state中返回的三个值
        你可以更新他们,并返回一个新的角色状态
        '''
        
        # 一个简单的例子，将手柄的位置与角色对齐
        controller.set_pos(self.cur_root_pos)
        controller.set_rot(self.cur_root_rot)

        return character_state
    # 你的其他代码,state matchine, motion matching, learning, etc.