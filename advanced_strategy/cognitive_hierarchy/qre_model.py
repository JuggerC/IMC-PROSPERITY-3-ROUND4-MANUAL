import numpy as np
from typing import Dict, List, Tuple, Any, Optional
import scipy.optimize as optimize
from .base_model import BaseModel

class QREModel(BaseModel):
    """
    量化反应均衡模型 (Quantal Response Equilibrium)
    
    模型假设:
    - 玩家选择策略的概率与该策略的预期收益成正比
    - 玩家的理性程度由lambda参数控制
    """
    
    def __init__(self, num_strategies: int, rationality: float, payoff_matrix: np.ndarray = None):
        """
        初始化QRE模型
        
        参数:
            num_strategies: 策略数量
            rationality: 理性程度参数lambda，值越大表示越理性
                       lambda=0表示完全随机，lambda=inf表示完全理性
            payoff_matrix: 收益矩阵，形状为 (num_strategies, num_strategies)
        """
        super().__init__(num_strategies, payoff_matrix)
        self.rationality = rationality
        self._equilibrium = None
        
    def set_rationality(self, rationality: float) -> None:
        """
        设置理性程度参数
        
        参数:
            rationality: 理性程度参数lambda
        """
        self.rationality = rationality
        # 清除缓存的均衡结果
        self._equilibrium = None
        
    def _response_function(self, opponent_distribution: np.ndarray) -> np.ndarray:
        """
        计算给定对手策略分布下的量化反应函数
        
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
        
        # 应用逻辑响应函数
        # 防止数值溢出
        exp_values = np.exp(self.rationality * expected_payoffs)
        return exp_values / np.sum(exp_values)
    
    def _fixed_point_objective(self, p: np.ndarray) -> np.ndarray:
        """
        寻找固定点的目标函数
        
        参数:
            p: 策略分布概率向量
            
        返回:
            与固定点的差距
        """
        # 确保概率分布有效
        p_normalized = p / np.sum(p)
        response = self._response_function(p_normalized)
        return response - p_normalized
        
    def compute_equilibrium(self) -> np.ndarray:
        """
        计算量化反应均衡
        
        返回:
            均衡策略分布概率向量
        """
        if self._equilibrium is not None:
            return self._equilibrium
            
        # 初始猜测: 均匀分布
        initial_guess = np.ones(self.num_strategies) / self.num_strategies
        
        # 寻找固定点
        result = optimize.root(self._fixed_point_objective, initial_guess, 
                            method='anderson', options={'disp': False})
        
        if result.success:
            # 确保是有效的概率分布
            equilibrium = result.x
            equilibrium = equilibrium / np.sum(equilibrium)
            self._equilibrium = equilibrium
            return equilibrium
        else:
            # 如果优化失败，返回逻辑响应
            uniform = np.ones(self.num_strategies) / self.num_strategies
            self._equilibrium = self._response_function(uniform)
            return self._equilibrium
        
    def calculate_strategy_distribution(self, opponent_distribution: np.ndarray = None) -> np.ndarray:
        """
        计算策略分布
        在QRE中，我们计算均衡点，而不是直接响应特定的对手分布
        
        参数:
            opponent_distribution: 在QRE中不使用，仅为接口一致性
            
        返回:
            均衡策略分布概率向量
        """
        return self.compute_equilibrium() 