import numpy as np
from typing import Dict, List, Tuple, Any, Optional, Callable
import pandas as pd

class SocialNormsModel:
    """
    社会规范模型
    
    模拟群体中社会规范的形成、传播和影响，
    包括规范压力、惩罚机制和规范演化。
    """
    
    def __init__(self, num_strategies: int = 3, population_size: int = 1000):
        """
        初始化社会规范模型
        
        参数:
            num_strategies: 可选策略的数量
            population_size: 人口规模
        """
        self.num_strategies = num_strategies
        self.population_size = population_size
        
        # 各策略的规范强度 (0-1)，表示社会规范对该策略的支持程度
        self.norm_strengths = np.ones(num_strategies) / num_strategies
        
        # 规范服从倾向 (0-1)
        self.norm_compliance = 0.5
        
        # 规范惩罚系数 (0-1)
        self.norm_punishment = 0.3
        
        # 规范变化速率 (0-1)
        self.norm_evolution_rate = 0.1
        
        # 策略分布历史
        self.strategy_history = []
        
        # 规范强度历史
        self.norm_history = []
        
        # 记录当前规范
        self._record_norms()
        
    def set_norm_strengths(self, strengths: np.ndarray) -> None:
        """
        设置策略规范强度
        
        参数:
            strengths: 规范强度数组，长度必须为num_strategies
        """
        if len(strengths) != self.num_strategies:
            raise ValueError(f"规范强度数组长度必须为{self.num_strategies}")
            
        # 确保规范强度和为1
        self.norm_strengths = strengths / np.sum(strengths)
        self._record_norms()
        
    def set_norm_parameters(self, compliance: float, punishment: float, 
                          evolution_rate: float) -> None:
        """
        设置规范参数
        
        参数:
            compliance: 规范服从倾向 (0-1)
            punishment: 规范惩罚系数 (0-1)
            evolution_rate: 规范变化速率 (0-1)
        """
        self.norm_compliance = max(0.0, min(1.0, compliance))
        self.norm_punishment = max(0.0, min(1.0, punishment))
        self.norm_evolution_rate = max(0.0, min(1.0, evolution_rate))
        
    def _record_norms(self) -> None:
        """记录当前规范状态"""
        self.norm_history.append(self.norm_strengths.copy())
        
    def apply_norm_pressure(self, strategy_distribution: np.ndarray) -> np.ndarray:
        """
        应用规范压力到策略分布
        
        参数:
            strategy_distribution: 原始策略分布
            
        返回:
            调整后的策略分布
        """
        # 确保输入是有效的概率分布
        if not np.isclose(np.sum(strategy_distribution), 1.0):
            strategy_distribution = strategy_distribution / np.sum(strategy_distribution)
            
        # 计算规范压力
        norm_pressure = self.norm_compliance * self.norm_strengths
        
        # 应用规范压力，调整策略分布
        adjusted_distribution = (1 - self.norm_compliance) * strategy_distribution + norm_pressure
        
        # 确保结果是有效的概率分布
        return adjusted_distribution / np.sum(adjusted_distribution)
        
    def calculate_norm_penalty(self, strategy: int) -> float:
        """
        计算违反规范的惩罚
        
        参数:
            strategy: 策略索引
            
        返回:
            规范惩罚值
        """
        # 计算策略与规范的偏差
        norm_deviation = 1.0 - self.norm_strengths[strategy]
        
        # 惩罚与偏差和惩罚系数成正比
        return norm_deviation * self.norm_punishment
        
    def apply_norm_penalties(self, payoffs: np.ndarray) -> np.ndarray:
        """
        应用规范惩罚到策略收益
        
        参数:
            payoffs: 各策略的收益
            
        返回:
            调整后的收益
        """
        # 计算每个策略的规范惩罚
        penalties = np.array([self.calculate_norm_penalty(i) for i in range(self.num_strategies)])
        
        # 应用惩罚到收益
        adjusted_payoffs = payoffs - penalties
        
        return adjusted_payoffs
        
    def update_norms(self, strategy_distribution: np.ndarray, iterations: int = 1) -> np.ndarray:
        """
        更新社会规范强度
        
        参数:
            strategy_distribution: 当前策略分布
            iterations: 更新迭代次数
            
        返回:
            更新后的规范强度
        """
        for _ in range(iterations):
            # 规范向流行策略转变
            # 新规范 = 当前规范 + 演化率 * (策略分布 - 当前规范)
            norm_delta = self.norm_evolution_rate * (strategy_distribution - self.norm_strengths)
            self.norm_strengths += norm_delta
            
            # 确保规范强度为有效概率分布
            self.norm_strengths = np.maximum(0, self.norm_strengths)
            self.norm_strengths = self.norm_strengths / np.sum(self.norm_strengths)
            
            # 记录规范历史
            self._record_norms()
            
        return self.norm_strengths
        
    def simulate_norm_evolution(self, initial_distribution: np.ndarray, 
                              iterations: int = 100) -> Tuple[List[np.ndarray], List[np.ndarray]]:
        """
        模拟社会规范和策略分布的共同演化
        
        参数:
            initial_distribution: 初始策略分布
            iterations: 模拟迭代次数
            
        返回:
            (策略分布历史, 规范强度历史)
        """
        # 重置历史记录
        self.strategy_history = []
        self.norm_history = []
        
        # 当前策略分布
        strategy_dist = initial_distribution.copy()
        
        for _ in range(iterations):
            # 记录当前策略分布
            self.strategy_history.append(strategy_dist.copy())
            
            # 应用规范压力
            adjusted_dist = self.apply_norm_pressure(strategy_dist)
            
            # 更新规范
            self.update_norms(adjusted_dist)
            
            # 更新策略分布
            strategy_dist = adjusted_dist
            
        return self.strategy_history, self.norm_history
        
    def get_current_norms(self) -> np.ndarray:
        """
        获取当前规范强度
        
        返回:
            规范强度数组
        """
        return self.norm_strengths
        
    def get_norm_history(self) -> List[np.ndarray]:
        """
        获取规范强度历史
        
        返回:
            规范强度历史列表
        """
        return self.norm_history
        
    def get_dominant_norm(self) -> int:
        """
        获取主导规范
        
        返回:
            主导规范的策略索引
        """
        return np.argmax(self.norm_strengths)
        
    def get_norm_diversity(self) -> float:
        """
        计算规范多样性（使用熵）
        
        返回:
            规范多样性指数
        """
        # 确保没有0值（防止log(0)）
        adjusted_strengths = self.norm_strengths + 1e-10
        adjusted_strengths = adjusted_strengths / np.sum(adjusted_strengths)
        
        # 计算熵
        entropy = -np.sum(adjusted_strengths * np.log(adjusted_strengths))
        
        # 归一化熵（0:单一规范, 1:完全多样）
        max_entropy = np.log(self.num_strategies)
        return entropy / max_entropy if max_entropy > 0 else 0.0
        
    def get_norm_stability(self) -> float:
        """
        计算规范稳定性
        
        返回:
            规范稳定性指数（0-1）
        """
        if len(self.norm_history) < 2:
            return 1.0  # 只有一个时间点，假设完全稳定
            
        # 计算最近10个时间点或所有可用点的规范变化
        history_window = min(10, len(self.norm_history))
        recent_norms = self.norm_history[-history_window:]
        
        # 计算每个时间步的平均变化
        changes = []
        for i in range(1, len(recent_norms)):
            change = np.sum(np.abs(recent_norms[i] - recent_norms[i-1]))
            changes.append(change)
            
        avg_change = np.mean(changes) if changes else 0.0
        
        # 转换为稳定性指数（变化越小越稳定）
        stability = 1.0 - min(1.0, avg_change * 5.0)  # 缩放因子5.0使指标更敏感
        
        return stability
        
    def to_dataframe(self) -> pd.DataFrame:
        """
        将模型数据转换为DataFrame
        
        返回:
            包含规范历史的DataFrame
        """
        if not self.norm_history:
            return pd.DataFrame()
            
        data = []
        for i, norms in enumerate(self.norm_history):
            row = {'timestep': i}
            for j, strength in enumerate(norms):
                row[f'strategy_{j}_norm'] = strength
                
            if i < len(self.strategy_history):
                for j, freq in enumerate(self.strategy_history[i]):
                    row[f'strategy_{j}_freq'] = freq
                    
            data.append(row)
            
        return pd.DataFrame(data) 