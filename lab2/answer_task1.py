import numpy as np
import copy
from scipy.spatial.transform import Rotation as R
from scipy.spatial.transform import Slerp
# ------------- lab1里的代码 -------------#
def load_meta_data(bvh_path):
    with open(bvh_path, 'r') as f:
        channels = []
        joints = []
        joint_parents = []
        joint_offsets = []
        end_sites = []

        parent_stack = [None]
        for line in f:
            if 'ROOT' in line or 'JOINT' in line:
                joints.append(line.split()[-1])
                joint_parents.append(parent_stack[-1])
                channels.append('')
                joint_offsets.append([0, 0, 0])

            elif 'End Site' in line:
                end_sites.append(len(joints))
                joints.append(parent_stack[-1] + '_end')
                joint_parents.append(parent_stack[-1])
                channels.append('')
                joint_offsets.append([0, 0, 0])

            elif '{' in line:
                parent_stack.append(joints[-1])

            elif '}' in line:
                parent_stack.pop()

            elif 'OFFSET' in line:
                joint_offsets[-1] = np.array([float(x) for x in line.split()[-3:]]).reshape(1,3)

            elif 'CHANNELS' in line:
                trans_order = []
                rot_order = []
                for token in line.split():
                    if 'position' in token:
                        trans_order.append(token[0])

                    if 'rotation' in token:
                        rot_order.append(token[0])

                channels[-1] = ''.join(trans_order)+ ''.join(rot_order)

            elif 'Frame Time:' in line:
                break
        
    joint_parents = [-1]+ [joints.index(i) for i in joint_parents[1:]]
    channels = [len(i) for i in channels]
    return joints, joint_parents, channels, joint_offsets

def load_motion_data(bvh_path):
    with open(bvh_path, 'r') as f:
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

# ------------- 实现一个简易的BVH对象，进行数据处理 -------------#

'''
注释里统一N表示帧数，M表示关节数
position, rotation表示局部平移和旋转
translation, orientation表示全局平移和旋转
'''

class BVHMotion():
    def __init__(self, bvh_file_name = None) -> None:
        
        # 一些 meta data
        self.joint_name = []
        self.joint_channel = []
        self.joint_parent = []
        self.hip_index = 0
        self.spine_index = 0
        
        # 一些local数据, 对应bvh里的channel, XYZposition和 XYZrotation
        #! 这里我们把没有XYZ position的joint的position设置为offset, 从而进行统一
        self.joint_position = None # (N,M,3) 的ndarray, 局部平移
        self.joint_rotation = None # (N,M,4)的ndarray, 用四元数表示的局部旋转
        
        if bvh_file_name is not None:
            self.load_motion(bvh_file_name)
        pass
    
    #------------------- 一些辅助函数 ------------------- #
    def load_motion(self, bvh_file_path):
        '''
            读取bvh文件，初始化元数据和局部数据
        '''
        self.joint_name, self.joint_parent, self.joint_channel, joint_offset = \
            load_meta_data(bvh_file_path)
        
        motion_data = load_motion_data(bvh_file_path)

        # 把motion_data里的数据分配到joint_position和joint_rotation里
        self.joint_position = np.zeros((motion_data.shape[0], len(self.joint_name), 3))
        self.joint_rotation = np.zeros((motion_data.shape[0], len(self.joint_name), 4))
        self.joint_rotation[:,:,3] = 1.0 # 四元数的w分量默认为1
        
        cur_channel = 0
        for i in range(len(self.joint_name)):
            if self.joint_channel[i] == 0:
                self.joint_position[:,i,:] = joint_offset[i].reshape(1,3)
                continue   
            elif self.joint_channel[i] == 3:
                self.joint_position[:,i,:] = joint_offset[i].reshape(1,3)
                rotation = motion_data[:, cur_channel:cur_channel+3]
            elif self.joint_channel[i] == 6:
                self.joint_position[:, i, :] = motion_data[:, cur_channel:cur_channel+3]
                rotation = motion_data[:, cur_channel+3:cur_channel+6]
            self.joint_rotation[:, i, :] = R.from_euler('XYZ', rotation,degrees=True).as_quat()
            cur_channel += self.joint_channel[i]

        # NOTICE: 这里的骨骼有左右两个髋关节，感觉用其父关节进行代替比较好？
        self.hip_index = self.joint_name.index("lHip")
        self.spine_index = self.joint_name.index("lowerback_torso") # 上脊柱关节

        return

    def batch_forward_kinematics(self, joint_position = None, joint_rotation = None):
        '''
        利用自身的metadata进行批量前向运动学
        joint_position: (N,M,3)的ndarray, 局部平移
        joint_rotation: (N,M,4)的ndarray, 用四元数表示的局部旋转
        '''
        if joint_position is None:
            joint_position = self.joint_position
        if joint_rotation is None:
            joint_rotation = self.joint_rotation
        
        joint_translation = np.zeros_like(joint_position)
        joint_orientation = np.zeros_like(joint_rotation)
        joint_orientation[:,:,3] = 1.0 # 四元数的w分量默认为1
        simulation_translation = np.zeros(shape=(joint_position.shape[0], 3))
        simulation_orientation = np.zeros(shape=(joint_position.shape[0], 4))

        # 一个小hack是root joint的parent是-1, 对应最后一个关节
        # 计算根节点时最后一个关节还未被计算，刚好是0偏移和单位朝向
        
        for i in range(len(self.joint_name)):
            pi = self.joint_parent[i]
            parent_orientation = R.from_quat(joint_orientation[:,pi,:])
            translation = joint_translation[:, pi, :] + \
                parent_orientation.apply(joint_position[:, i, :])
            joint_translation[:, i, :] = translation
            orientation = (parent_orientation * R.from_quat(joint_rotation[:, i, :])).as_quat()
            joint_orientation[:, i, :] = orientation

        for i in range(len(simulation_translation)):
            translation, orientation = self.get_simulation_bone(joint_translation[i], joint_orientation[i])
            simulation_translation[i] = translation
            simulation_orientation[i] = orientation

        return joint_translation, joint_orientation, simulation_translation, simulation_orientation

    def get_simulation_bone(self, joint_translation: np.ndarray, joint_orientation: np.ndarray):
        '''
        基于[代码与数据驱动的位移](https://theorangeduck.com/page/code-vs-data-driven-displacement)构造的某一帧下当前骨架对应的仿真骨骼信息
        '''
        projected_translation = np.array([joint_translation[self.spine_index, 0], 0.01, joint_translation[self.spine_index, 2]])
        # Ry, _ = self.decompose_rotation_with_yaxis(joint_orientation)

        return projected_translation, joint_orientation[self.hip_index]


    def adjust_joint_name(self, target_joint_name):
        '''
        调整关节顺序为target_joint_name
        '''
        idx = [self.joint_name.index(joint_name) for joint_name in target_joint_name]
        idx_inv = [target_joint_name.index(joint_name) for joint_name in self.joint_name]
        self.joint_name = [self.joint_name[i] for i in idx]
        self.joint_parent = [idx_inv[self.joint_parent[i]] for i in idx]
        self.joint_parent[0] = -1
        self.joint_channel = [self.joint_channel[i] for i in idx]
        self.joint_position = self.joint_position[:,idx,:]
        self.joint_rotation = self.joint_rotation[:,idx,:]
        pass
    
    def raw_copy(self):
        '''
        返回一个拷贝
        '''
        return copy.deepcopy(self)
    
    @property
    def motion_length(self):
        return self.joint_position.shape[0]

    
    def sub_sequence(self, start, end):
        '''
        返回一个子序列
        start: 开始帧
        end: 结束帧
        '''
        res = self.raw_copy()
        res.joint_position = res.joint_position[start:end,:,:]
        res.joint_rotation = res.joint_rotation[start:end,:,:]
        return res
    
    def append(self, other):
        '''
        在末尾添加另一个动作
        '''
        other = other.raw_copy()
        other.adjust_joint_name(self.joint_name)
        self.joint_position = np.concatenate((self.joint_position, other.joint_position), axis=0)
        self.joint_rotation = np.concatenate((self.joint_rotation, other.joint_rotation), axis=0)
        pass
    
    #--------------------- 你的任务 -------------------- #
    
    def decompose_rotation_with_yaxis(self, rotation: np.ndarray):
        '''
        输入: rotation 形状为(4,)的ndarray, 四元数旋转
        输出: Ry, Rxz，分别为绕y轴的旋转和转轴在xz平面的旋转，并满足R = Ry * Rxz
        '''
        Ry = np.zeros_like(rotation)
        Rxz = np.zeros_like(rotation)

        r = R.from_quat(rotation)
        angle = r.as_euler("yzx")
        ry = R.from_rotvec(np.array([0, 1, 0]) * angle[1])
        Ry = ry.as_quat()
        rxz = ry.inv() * r
        Rxz = rxz.as_quat()

        return Ry, Rxz

    # part 1
    def translation_and_rotation(self, frame_num: int, target_translation_xz: np.ndarray, target_facing_direction_xz: np.ndarray):
        '''
        计算出新的joint_position和joint_rotation
        使第frame_num帧的根节点平移为target_translation_xz, 水平面朝向为target_facing_direction_xz
        frame_num: int
        target_translation_xz: (2,)的ndarray
        target_faceing_direction_xz: (2,)的ndarray，表示水平朝向。你可以理解为原本的z轴被旋转到这个方向。
        Tips:
            主要是调整root节点的joint_position和joint_rotation
            frame_num可能是负数，遵循python的索引规则
            你需要完成并使用decompose_rotation_with_yaxis
            输入的target_facing_direction_xz的norm不一定是1
        '''
        
        res = self.raw_copy() # 拷贝一份，不要修改原始数据
        
        # 比如说，你可以这样调整第frame_num帧的根节点平移
        offset = target_translation_xz - res.joint_position[frame_num, 0, [0,2]]
        res.joint_position[:, 0, [0,2]] += offset
        # root_R = R.from_quat(res.joint_rotation[frame_num, 0])
        target_facing_direction_xz = target_facing_direction_xz / np.linalg.norm(target_facing_direction_xz)
        [x, z] = target_facing_direction_xz
        # rotation_angle = np.arctan(-x / z)
        rotation_angle = np.arctan2(x, z) # TODO: 这里x轴和z轴的关系难道不应该取负值吗？
        # print("rotation_angle: ", rotation_angle)
        if rotation_angle == 0.0:
            return res

        delta_R = R.from_rotvec(np.array([0, 1, 0]) * rotation_angle)
        init_pos = res.joint_position[frame_num, 0, [0, 1, 2]]
        for i in range(0, res.joint_rotation.shape[0]):
            origin_R = R.from_quat(res.joint_rotation[i, 0])
            origin_pos = res.joint_position[i, 0, [0, 1, 2]]
            offset_pos = delta_R.apply(origin_pos - init_pos)
            res.joint_rotation[i, 0] = (delta_R * origin_R).as_quat()
            # NOTICE: 由于旋转后之前的平移矢量也发生了旋转，因此需要重新计算平移矢量！
            res.joint_position[i, 0, [0,2]] = np.array([offset_pos[0] + init_pos[0], offset_pos[2] + init_pos[2]])

        return res

# part2
def blend_two_motions(bvh_motion1: BVHMotion, bvh_motion2: BVHMotion, alpha: np.ndarray):
    '''
    blend两个bvh动作
    假设两个动作的帧数分别为n1, n2
    alpha: 0~1之间的浮点数组，形状为(n3,)
    返回的动作应该有n3帧，第i帧由(1-alpha[i]) * bvh_motion1[j] + alpha[i] * bvh_motion2[k]得到
    i均匀地遍历0~n3-1的同时，j和k应该均匀地遍历0~n1-1和0~n2-1
    '''
    
    res = bvh_motion1.raw_copy()
    res.joint_position = np.zeros((len(alpha), res.joint_position.shape[1], res.joint_position.shape[2]))
    res.joint_rotation = np.zeros((len(alpha), res.joint_rotation.shape[1], res.joint_rotation.shape[2]))
    res.joint_rotation[...,3] = 1.0

    motion1_len = float(bvh_motion1.motion_length - 1)
    motion2_len = float(bvh_motion2.motion_length - 1)
    motion_len = len(alpha)
    for i in range(motion_len):
        ratio = float(i) / float(motion_len - 1)
        motion1_index = int(motion1_len * ratio)
        motion2_index = int(motion2_len * ratio)
        a = alpha[i]
        res.joint_position[i] = bvh_motion1.joint_position[motion1_index] * (1.0 - a) + bvh_motion2.joint_position[motion2_index] * a
        # res.joint_rotation[i] = bvh_motion1.joint_rotation[motion1_index] * (1.0 - a) + bvh_motion2.joint_rotation[motion2_index] * a
        # for joint_i in range(bvh_motion1.joint_rotation.shape[1]):
        #     res.joint_rotation[i, joint_i] = res.joint_rotation[i, joint_i] / np.linalg.norm(res.joint_rotation[i, joint_i]) # 转化为单位四元数
        for joint_i in range(bvh_motion1.joint_rotation.shape[1]):
            q0 = bvh_motion1.joint_rotation[motion1_index, joint_i]
            q1 = bvh_motion2.joint_rotation[motion2_index, joint_i]
            angle = np.arccos(np.dot(q0, q1))
            # 避免夹角过大插值出现奇异值，利用单位四元数的特性进行反向
            if np.abs(angle) > np.pi * 0.5:
                q0 = -q0
                # angle = np.arccos(np.dot(q0, q1))
            # 基于slerp对四元数进行插值
            # q = np.sin((1.0 - a) * angle) / np.sin(angle) * q0 + np.sin(a * angle) / np.sin(angle) * q1
            # 简单线性插值
            q = (1.0 - a) * q0 + a * q1
            q = q / np.linalg.norm(q)
            # print(q0, q1, q)
            res.joint_rotation[i, joint_i] = q

    return res

# part3
def build_loop_motion(bvh_motion):
    '''
    将bvh动作变为循环动作
    由于比较复杂,作为福利,不用自己实现
    (当然你也可以自己实现试一下)
    推荐阅读 https://theorangeduck.com/
    Creating Looping Animations from Motion Capture
    '''
    res = bvh_motion.raw_copy()
    
    from smooth_utils import build_loop_motion
    return build_loop_motion(res)

# part4
def concatenate_two_motions(bvh_motion1: BVHMotion, bvh_motion2: BVHMotion, mix_frame1: int, mix_time: int):
    '''
    将两个bvh动作平滑地连接起来，mix_time表示用于混合的帧数
    混合开始时间是第一个动作的第mix_frame1帧
    虽然某些混合方法可能不需要mix_time，但是为了保证接口一致，我们还是保留这个参数
    Tips:
        你可能需要用到BVHMotion.sub_sequence 和 BVHMotion.append
    '''
    res = bvh_motion1.raw_copy()

    # 计算出插值开始帧根关节的位置和旋转
    start_postion = bvh_motion1.joint_position[mix_frame1, 0, [0, 1, 2]]
    (start_Ry, _) = bvh_motion1.decompose_rotation_with_yaxis(bvh_motion1.joint_rotation[mix_frame1, 0])
    Ry = R.from_quat(start_Ry)
    start_face = Ry.apply(np.array([0, 0, 1]))
    # 对第二个动作进行位移和朝向的调整
    bvh_motion2_transformed = bvh_motion2.translation_and_rotation(0, start_postion[[0, 2]], start_face[[0, 2]])
    # 硬拼接
    res.joint_position = np.concatenate([res.joint_position[:mix_frame1], bvh_motion2_transformed.joint_position], axis=0)
    res.joint_rotation = np.concatenate([res.joint_rotation[:mix_frame1], bvh_motion2_transformed.joint_rotation], axis=0)

    # 插值段内进行线性插值
    for i in range(mix_time):
        ratio = float(i) / float(mix_time - 1) # [0, 1]
        motion1_index = mix_frame1 + i
        motion2_index = i
        res.joint_position[motion1_index] = bvh_motion1.joint_position[motion1_index] * (1.0 - ratio) + bvh_motion2_transformed.joint_position[motion2_index] * ratio
        for joint_i in range(bvh_motion1.joint_rotation.shape[1]):
            q0 = bvh_motion1.joint_rotation[motion1_index, joint_i]
            q1 = bvh_motion2_transformed.joint_rotation[motion2_index, joint_i]
            angle = np.arccos(np.dot(q0, q1))
            # 避免夹角过大插值出现奇异值，利用单位四元数的特性进行反向
            if np.abs(angle) > np.pi * 0.5:
                q0 = -q0
            # 简单线性插值
            q = (1.0 - ratio) * q0 + ratio * q1
            q = q / np.linalg.norm(q)
            # print(q0, q1, q)
            res.joint_rotation[motion1_index, joint_i] = q



    return res

