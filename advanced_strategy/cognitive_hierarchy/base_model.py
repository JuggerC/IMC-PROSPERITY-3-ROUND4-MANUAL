import numpy as np
from typing import Dict, List, Tuple, Any, Optional

class BaseModel:
    """
    基础认知模型，为所有认知层次模型提供基础功能
    """
    
    def __init__(self, num_strategies: int, payoff_matrix: np.ndarray = None):
        """
        初始化基础模型
        
        参数:
            num_strategies: 策略数量
            payoff_matrix: 收益矩阵，形状为 (num_strategies, num_strategies)
        """
        self.num_strategies = num_strategies
        self.payoff_matrix = payoff_matrix
        
    def set_payoff_matrix(self, payoff_matrix: np.ndarray) -> None:
        """
        设置收益矩阵
        
        参数:
            payoff_matrix: 收益矩阵
        """
        if payoff_matrix.shape != (self.num_strategies, self.num_strategies):
            raise ValueError(f"收益矩阵形状应为 ({self.num_strategies}, {self.num_strategies})，"
                            f"但得到的是 {payoff_matrix.shape}")
        self.payoff_matrix = payoff_matrix
        
    def calculate_expected_payoff(self, strategy: int, opponent_distribution: np.ndarray) -> float:
        """
        计算给定对手策略分布下的预期收益
        
        参数:
            strategy: 玩家策略索引
            opponent_distribution: 对手策略分布概率向量
            
        返回:
            预期收益
        """
        if self.payoff_matrix is None:
            raise ValueError("收益矩阵未设置")
            
        return np.dot(self.payoff_matrix[strategy], opponent_distribution)
    
    def best_response(self, opponent_distribution: np.ndarray) -> int:
        """
        计算对给定对手策略分布的最佳响应
        
        参数:
            opponent_distribution: 对手策略分布概率向量
            
        返回:
            最佳响应策略的索引
        """
        expected_payoffs = np.array([
            self.calculate_expected_payoff(i, opponent_distribution)
            for i in range(self.num_strategies)
        ])
        
        return np.argmax(expected_payoffs)
    
    def calculate_strategy_distribution(self, opponent_distribution: np.ndarray) -> np.ndarray:
        """
        计算对给定对手策略分布的策略分布
        子类应重写此方法以实现特定的认知模型
        
        参数:
            opponent_distribution: 对手策略分布概率向量
            
        返回:
            策略分布概率向量
        """
        # 基础实现：最佳响应（确定性策略）
        best_resp = self.best_response(opponent_distribution)
        distribution = np.zeros(self.num_strategies)
        distribution[best_resp] = 1.0
        return distribution 