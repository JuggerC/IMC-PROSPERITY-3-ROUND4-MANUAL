import numpy as np
from typing import Dict, List, Tuple, Any, Optional
from .base_model import BaseModel

class BoundedRationalityModel(BaseModel):
    """
    有限理性模型
    
    模型假设:
    - 玩家有认知成本，无法精确计算最佳响应
    - 玩家有概率做出非最优决策，这个概率随策略收益差距而变化
    """
    
    def __init__(self, num_strategies: int, rationality: float, noise: float, payoff_matrix: np.ndarray = None):
        """
        初始化有限理性模型
        
        参数:
            num_strategies: 策略数量
            rationality: 理性程度参数，值越大表示越理性
            noise: 噪声参数，决定非最优决策的概率程度
            payoff_matrix: 收益矩阵，形状为 (num_strategies, num_strategies)
        """
        super().__init__(num_strategies, payoff_matrix)
        self.rationality = rationality
        self.noise = noise
        
    def set_rationality(self, rationality: float) -> None:
        """
        设置理性程度参数
        
        参数:
            rationality: 理性程度参数
        """
        self.rationality = rationality
        
    def set_noise(self, noise: float) -> None:
        """
        设置噪声参数
        
        参数:
            noise: 噪声参数
        """
        self.noise = noise
        
    def calculate_strategy_distribution(self, opponent_distribution: np.ndarray) -> np.ndarray:
        """
        计算有限理性下的策略分布
        
        参数:
            opponent_distribution: 对手策略分布概率向量
            
        返回:
            策略分布概率向量
        """
        # 计算每个策略的预期收益
        expected_payoffs = np.array([
            self.calculate_expected_payoff(i, opponent_distribution)
            for i in range(self.num_strategies)
        ])
        
        # 找出最优策略的预期收益
        max_payoff = np.max(expected_payoffs)
        
        # 计算每个策略相对于最优策略的收益差距
        payoff_gaps = max_payoff - expected_payoffs
        
        # 应用有限理性模型
        # 对差距应用理性系数和噪声参数
        selection_probabilities = np.exp(-self.rationality * payoff_gaps + self.noise * np.random.normal(size=self.num_strategies))
        
        # 归一化为概率分布
        return selection_probabilities / np.sum(selection_probabilities)
    
    def calibrate_from_data(self, strategy_choices: np.ndarray, opponent_distributions: np.ndarray) -> Tuple[float, float]:
        """
        从观察数据中校准模型参数
        
        参数:
            strategy_choices: 玩家的策略选择记录，形状为(样本数,)
            opponent_distributions: 对应的对手策略分布记录，形状为(样本数, num_strategies)
            
        返回:
            (rationality, noise): 校准后的参数
        """
        # 简单的校准算法，在实际应用中可以使用更复杂的方法（如最大似然估计）
        
        # 计算观察到的策略分布
        observed_distribution = np.zeros(self.num_strategies)
        for choice in strategy_choices:
            observed_distribution[choice] += 1
        observed_distribution = observed_distribution / np.sum(observed_distribution)
        
        # 获取平均对手分布
        avg_opponent_distribution = np.mean(opponent_distributions, axis=0)
        
        # 使用网格搜索寻找最佳参数
        best_rationality = self.rationality
        best_noise = self.noise
        best_error = float('inf')
        
        for r in np.linspace(0.1, 5.0, 20):
            for n in np.linspace(0.0, 2.0, 20):
                self.rationality = r
                self.noise = n
                
                # 计算预测分布
                predicted = self.calculate_strategy_distribution(avg_opponent_distribution)
                
                # 计算与观察分布的差距
                error = np.sum((predicted - observed_distribution) ** 2)
                
                if error < best_error:
                    best_error = error
                    best_rationality = r
                    best_noise = n
        
        # 设置为最佳参数
        self.rationality = best_rationality
        self.noise = best_noise
        
        return best_rationality, best_noise 