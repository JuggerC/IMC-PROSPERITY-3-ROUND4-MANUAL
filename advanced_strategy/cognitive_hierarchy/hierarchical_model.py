import math
import numpy as np
from typing import Dict, List, Tuple, Any, Optional
from .base_model import BaseModel
from .level_k_model import LevelKModel

class HierarchicalModel(BaseModel):
    """
    分层认知模型 - 整合多个层级的玩家
    
    模型假设:
    - 人口中有不同思考层级的玩家
    - 每个层级的玩家比例由分布决定
    """
    
    def __init__(self, num_strategies: int, max_level: int, level_distribution: np.ndarray = None, 
                payoff_matrix: np.ndarray = None):
        """
        初始化分层认知模型
        
        参数:
            num_strategies: 策略数量
            max_level: 最大思考层级
            level_distribution: 各层级玩家的分布，长度为max_level+1
                              如果为None，则使用泊松分布
            payoff_matrix: 收益矩阵，形状为 (num_strategies, num_strategies)
        """
        super().__init__(num_strategies, payoff_matrix)
        self.max_level = max_level
        
        # 初始化层级分布
        if level_distribution is None:
            # 默认使用参数为1.5的截断泊松分布
            self.level_distribution = self._poisson_distribution(max_level, 1.5)
        else:
            # 确保分布长度正确
            if len(level_distribution) != max_level + 1:
                raise ValueError(f"层级分布长度应为 {max_level + 1}，但得到的是 {len(level_distribution)}")
                
            # 归一化分布
            self.level_distribution = level_distribution / np.sum(level_distribution)
            
        # 创建各层级的模型
        self.level_models = [
            LevelKModel(num_strategies, level, payoff_matrix)
            for level in range(max_level + 1)
        ]
        
    def _poisson_distribution(self, max_level: int, tau: float) -> np.ndarray:
        """
        生成截断的泊松分布
        
        参数:
            max_level: 最大层级
            tau: 泊松分布参数
            
        返回:
            归一化的泊松分布概率向量
        """
        dist = np.zeros(max_level + 1)
        for k in range(max_level + 1):
            dist[k] = np.exp(-tau) * (tau ** k) / math.factorial(k)
            
        # 归一化
        return dist / np.sum(dist)
        
    def set_level_distribution(self, level_distribution: np.ndarray) -> None:
        """
        设置层级分布
        
        参数:
            level_distribution: 各层级玩家的分布
        """
        if len(level_distribution) != self.max_level + 1:
            raise ValueError(f"层级分布长度应为 {self.max_level + 1}，但得到的是 {len(level_distribution)}")
            
        self.level_distribution = level_distribution / np.sum(level_distribution)
        
    def set_tau(self, tau: float) -> None:
        """
        使用泊松参数tau设置层级分布
        
        参数:
            tau: 泊松分布参数
        """
        self.level_distribution = self._poisson_distribution(self.max_level, tau)
        
    def set_payoff_matrix(self, payoff_matrix: np.ndarray) -> None:
        """
        设置收益矩阵
        
        参数:
            payoff_matrix: 收益矩阵
        """
        super().set_payoff_matrix(payoff_matrix)
        
        # 更新所有层级模型的收益矩阵
        for model in self.level_models:
            model.set_payoff_matrix(payoff_matrix)
            
    def calculate_level_distributions(self) -> List[np.ndarray]:
        """
        计算各层级玩家的策略分布
        
        返回:
            各层级策略分布的列表
        """
        return [model.calculate_strategy_distribution() for model in self.level_models]
        
    def calculate_strategy_distribution(self, opponent_distribution: np.ndarray = None) -> np.ndarray:
        """
        计算综合策略分布
        
        参数:
            opponent_distribution: 在分层模型中不使用，仅为接口一致性
            
        返回:
            综合策略分布概率向量
        """
        level_distributions = self.calculate_level_distributions()
        
        # 加权平均各层级的策略分布
        combined_distribution = np.zeros(self.num_strategies)
        for level, dist in enumerate(level_distributions):
            combined_distribution += self.level_distribution[level] * dist
            
        return combined_distribution
    
    def estimate_population_distribution(self, observed_strategies: np.ndarray) -> np.ndarray:
        """
        从观察到的策略中估计人口中的层级分布
        
        参数:
            observed_strategies: 观察到的策略选择，形状为 (样本数,)
            
        返回:
            估计的层级分布
        """
        # 计算观察到的策略分布
        observed_dist = np.zeros(self.num_strategies)
        for strategy in observed_strategies:
            observed_dist[strategy] += 1
        observed_dist = observed_dist / np.sum(observed_dist)
        
        # 计算各层级的策略分布
        level_dists = self.calculate_level_distributions()
        
        # 使用简单的最小二乘法估计层级分布
        # 构建线性系统 Ax = b，其中 A 是各层级策略分布，x 是层级分布，b 是观察到的策略分布
        A = np.column_stack(level_dists)
        b = observed_dist
        
        # 求解有约束的最小二乘问题：min ||Ax - b||^2 s.t. x >= 0, sum(x) = 1
        from scipy.optimize import minimize
        
        def objective(x):
            return np.sum((A @ x - b) ** 2)
            
        constraints = [
            {'type': 'eq', 'fun': lambda x: np.sum(x) - 1}  # sum(x) = 1
        ]
        
        bounds = [(0, 1) for _ in range(self.max_level + 1)]  # 0 <= x[i] <= 1
        
        result = minimize(objective, self.level_distribution, method='SLSQP', 
                        bounds=bounds, constraints=constraints)
                        
        if result.success:
            return result.x
        else:
            # 如果优化失败，保持原有分布
            return self.level_distribution 

    def get_best_strategy_for_level(self, level: int) -> int:
        """
        获取指定层级玩家的最优策略
        
        参数:
            level: 层级
            
        返回:
            最优策略索引
        """
        if not 0 <= level <= self.max_level:
            raise ValueError(f"层级必须在0到{self.max_level}之间")
            
        # 获取该层级的策略分布
        strategy_dist = self.level_models[level].calculate_strategy_distribution()
        
        # 返回概率最高的策略
        return np.argmax(strategy_dist) 