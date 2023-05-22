from __future__ import annotations
import numpy as np

class MetaData:
    def __init__(self, joint_name: list[str], joint_parent: list[int], joint_initial_position: np.ndarray, root_joint: str, end_joint: str):
        """
        一些固定信息，其中joint_initial_position是T-pose下的关节位置，可以用于计算关节相互的offset
        root_joint是固定节点的索引，并不是RootJoint节点
        """
        self.joint_name = joint_name
        self.joint_parent = joint_parent
        self.joint_initial_position = joint_initial_position
        self.root_joint = root_joint
        self.end_joint = end_joint

    def get_path_from_root_to_end(self):
        """
        辅助函数，返回从root节点到end节点的路径
        
        输出：
            path: 各个关节的索引
            path_name: 各个关节的名字
        Note: 
            如果root_joint在脚，而end_joint在手，那么此路径会路过RootJoint节点。
            在这个例子下path2返回从脚到根节点的路径，path1返回从根节点到手的路径。
            你可能会需要这两个输出。
        """
        
        # 从end节点开始，一直往上找，直到找到腰部节点
        path1: list[int] = [self.joint_name.index(self.end_joint)]
        while self.joint_parent[path1[-1]] != -1:
            path1.append(self.joint_parent[path1[-1]])
            
        # 从root节点开始，一直往上找，直到找到腰部节点
        path2: list[int] = [self.joint_name.index(self.root_joint)]
        while self.joint_parent[path2[-1]] != -1:
            path2.append(self.joint_parent[path2[-1]])
        
        # 合并路径，消去重复的节点
        while path1 and path2 and path2[-1] == path1[-1]:
            path1.pop()
            a = path2.pop()
            
        path2.append(a)
        path = path2 + list(reversed(path1))
        path_name: list[str] = [self.joint_name[i] for i in path]
        return path, path_name, path1, path2