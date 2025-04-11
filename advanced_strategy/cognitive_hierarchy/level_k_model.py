import numpy as np
from typing import Dict, List, Tuple, Any, Optional
from .base_model import BaseModel

class LevelKModel(BaseModel):
    """
    层级思考模型 (Level-k)
    
    模型假设:
    - 0级玩家随机选择策略
    - k级玩家假设所有其他玩家都是k-1级，并做出最佳响应
    """
    
    def __init__(self, num_strategies: int, level: int, payoff_matrix: np.ndarray = None):
        """
        初始化层级思考模型
        
        参数:
            num_strategies: 策略数量
            level: 思考层级，>=0
            payoff_matrix: 收益矩阵，形状为 (num_strategies, num_strategies)
        """
        super().__init__(num_strategies, payoff_matrix)
        self.level = level
        
    def get_level_0_distribution(self) -> np.ndarray:
        """
        获取0级玩家的策略分布（通常是均匀分布）
        
        返回:
            策略分布概率向量
        """
        return np.ones(self.num_strategies) / self.num_strategies
    
    def get_level_k_distribution(self, k: int) -> np.ndarray:
        """
        递归计算k级玩家的策略分布
        
        参数:
            k: 思考层级
            
        返回:
            策略分布概率向量
        """
        if k == 0:
            return self.get_level_0_distribution()
        
        # k级玩家假设对手是k-1级
        opponent_distribution = self.get_level_k_distribution(k - 1)
        
        # 计算对k-1级玩家的最佳响应
        best_resp = self.best_response(opponent_distribution)
        
        # 返回确定性策略分布
        distribution = np.zeros(self.num_strategies)
        distribution[best_resp] = 1.0
        return distribution
        
    def calculate_strategy_distribution(self, opponent_distribution: np.ndarray = None) -> np.ndarray:
        """
        计算当前层级的策略分布
        
        参数:
            opponent_distribution: 在层级思考模型中不使用，仅为接口一致性
            
        返回:
            策略分布概率向量
        """
        return self.get_level_k_distribution(self.level) 