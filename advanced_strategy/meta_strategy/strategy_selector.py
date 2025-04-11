import numpy as np
from typing import Dict, List, Tuple, Any, Optional, Callable
import pandas as pd

class StrategySelector:
    """
    策略选择器
    
    基于多种因素评估和选择最优策略，包括:
    - 收益最大化
    - 风险管理
    - 稳健性
    - 时间序列分析
    """
    
    def __init__(self, num_strategies: int = 3):
        """
        初始化策略选择器
        
        参数:
            num_strategies: 可选策略的数量
        """
        self.num_strategies = num_strategies
        
        # 各种评估指标的权重
        self.evaluation_weights = {
            'expected_payoff': 0.5,  # 期望收益权重
            'risk': 0.2,             # 风险权重
            'robustness': 0.2,       # 稳健性权重
            'trend': 0.1             # 趋势权重
        }
        
        # 风险厌恶系数 (0-1)
        self.risk_aversion = 0.5
        
        # 历史记录
        self.evaluation_history = []
        
    def set_evaluation_weights(self, weights: Dict[str, float]) -> None:
        """
        设置评估指标权重
        
        参数:
            weights: 权重字典，键为指标名称，值为权重
        """
        # 确保所有必需的键都存在
        required_keys = {'expected_payoff', 'risk', 'robustness', 'trend'}
        if not required_keys.issubset(weights.keys()):
            missing = required_keys - weights.keys()
            raise ValueError(f"缺少必需的权重键: {missing}")
            
        # 归一化权重
        total = sum(weights.values())
        self.evaluation_weights = {k: v / total for k, v in weights.items()}
        
    def set_risk_aversion(self, risk_aversion: float) -> None:
        """
        设置风险厌恶系数
        
        参数:
            risk_aversion: 风险厌恶系数 (0-1)
        """
        self.risk_aversion = max(0.0, min(1.0, risk_aversion))
        
    def evaluate_expected_payoff(self, payoff_matrix: np.ndarray, 
                               opponent_distribution: np.ndarray) -> np.ndarray:
        """
        评估各策略的期望收益
        
        参数:
            payoff_matrix: 收益矩阵
            opponent_distribution: 对手策略分布
            
        返回:
            各策略的期望收益数组
        """
        return np.dot(payoff_matrix, opponent_distribution)
        
    def evaluate_risk(self, payoff_matrix: np.ndarray, 
                    opponent_distribution: np.ndarray) -> np.ndarray:
        """
        评估各策略的风险
        
        参数:
            payoff_matrix: 收益矩阵
            opponent_distribution: 对手策略分布
            
        返回:
            各策略的风险评分数组 (越低越好)
        """
        # 计算期望收益
        expected_payoffs = self.evaluate_expected_payoff(payoff_matrix, opponent_distribution)
        
        # 计算每个策略的方差作为风险度量
        risks = np.zeros(self.num_strategies)
        
        for i in range(self.num_strategies):
            # 计算策略i的收益偏差平方
            squared_deviations = [(payoff_matrix[i, j] - expected_payoffs[i])**2 
                                for j in range(self.num_strategies)]
            
            # 加权平均偏差平方
            variance = np.sum(opponent_distribution * squared_deviations)
            risks[i] = variance
            
        # 标准化风险评分（越低越好）
        if np.max(risks) > 0:
            risks = risks / np.max(risks)
            
        return risks
        
    def evaluate_robustness(self, payoff_matrix: np.ndarray) -> np.ndarray:
        """
        评估各策略的稳健性（对抗最差情况）
        
        参数:
            payoff_matrix: 收益矩阵
            
        返回:
            各策略的稳健性评分数组 (越高越好)
        """
        # 计算每个策略的最坏情况收益
        worst_case_payoffs = np.min(payoff_matrix, axis=1)
        
        # 标准化（转换为0-1范围，越高越好）
        min_payoff = np.min(worst_case_payoffs)
        max_payoff = np.max(worst_case_payoffs)
        
        if max_payoff > min_payoff:
            robustness = (worst_case_payoffs - min_payoff) / (max_payoff - min_payoff)
        else:
            robustness = np.ones(self.num_strategies) / self.num_strategies
            
        return robustness
        
    def evaluate_trend(self, historical_distributions: List[np.ndarray]) -> np.ndarray:
        """
        评估策略分布趋势
        
        参数:
            historical_distributions: 历史策略分布列表
            
        返回:
            各策略的趋势评分数组 (越高越好)
        """
        if len(historical_distributions) < 2:
            # 没有足够的历史数据，返回均等分布
            return np.ones(self.num_strategies) / self.num_strategies
            
        # 获取最近几个时间步的分布
        window_size = min(5, len(historical_distributions))
        recent_distributions = historical_distributions[-window_size:]
        
        # 计算每个策略的线性趋势斜率
        trends = np.zeros(self.num_strategies)
        
        for i in range(self.num_strategies):
            # 提取策略i的历史频率
            strategy_freqs = [dist[i] for dist in recent_distributions]
            
            # 时间索引
            time_indices = np.arange(len(strategy_freqs))
            
            # 计算线性回归斜率
            if len(strategy_freqs) > 1:
                cov = np.cov(time_indices, strategy_freqs)
                if cov[0, 0] > 0:
                    trends[i] = cov[0, 1] / cov[0, 0]
            
        # 标准化趋势（转换为0-1范围）
        min_trend = np.min(trends)
        max_trend = np.max(trends)
        
        if max_trend > min_trend:
            normalized_trends = (trends - min_trend) / (max_trend - min_trend)
        else:
            normalized_trends = np.ones(self.num_strategies) / self.num_strategies
            
        return normalized_trends
        
    def select_strategy(self, payoff_matrix: np.ndarray, 
                      opponent_distribution: np.ndarray,
                      historical_distributions: Optional[List[np.ndarray]] = None) -> Tuple[int, Dict[str, np.ndarray]]:
        """
        选择最优策略
        
        参数:
            payoff_matrix: 收益矩阵
            opponent_distribution: 对手策略分布
            historical_distributions: 历史策略分布列表，可选
            
        返回:
            (最优策略索引, 评估指标字典)
        """
        # 计算各评估指标
        expected_payoffs = self.evaluate_expected_payoff(payoff_matrix, opponent_distribution)
        risks = self.evaluate_risk(payoff_matrix, opponent_distribution)
        robustness = self.evaluate_robustness(payoff_matrix)
        
        # 计算趋势（如果有历史数据）
        if historical_distributions:
            trends = self.evaluate_trend(historical_distributions)
        else:
            trends = np.ones(self.num_strategies) / self.num_strategies
            
        # 标准化期望收益（转换为0-1范围）
        min_payoff = np.min(expected_payoffs)
        max_payoff = np.max(expected_payoffs)
        
        if max_payoff > min_payoff:
            normalized_payoffs = (expected_payoffs - min_payoff) / (max_payoff - min_payoff)
        else:
            normalized_payoffs = np.ones(self.num_strategies) / self.num_strategies
            
        # 反转风险评分（转换为越高越好）
        risk_scores = 1.0 - risks
        
        # 根据评估权重计算总评分
        total_scores = (
            self.evaluation_weights['expected_payoff'] * normalized_payoffs + 
            self.evaluation_weights['risk'] * risk_scores * (1 - self.risk_aversion) +
            self.evaluation_weights['robustness'] * robustness * self.risk_aversion +
            self.evaluation_weights['trend'] * trends
        )
        
        # 选择得分最高的策略
        best_strategy = np.argmax(total_scores)
        
        # 记录评估结果
        evaluation_result = {
            'expected_payoff': expected_payoffs,
            'risk': risks,
            'robustness': robustness,
            'trend': trends,
            'total_score': total_scores
        }
        
        self.evaluation_history.append(evaluation_result)
        
        return best_strategy, evaluation_result
        
    def get_mixed_strategy(self, payoff_matrix: np.ndarray,
                         opponent_distribution: np.ndarray,
                         historical_distributions: Optional[List[np.ndarray]] = None,
                         temperature: float = 1.0) -> np.ndarray:
        """
        获取混合策略分布
        
        参数:
            payoff_matrix: 收益矩阵
            opponent_distribution: 对手策略分布
            historical_distributions: 历史策略分布列表，可选
            temperature: 温度参数，控制分布的平滑程度
            
        返回:
            混合策略概率分布
        """
        # 获取评估结果
        _, evaluation_result = self.select_strategy(
            payoff_matrix, opponent_distribution, historical_distributions
        )
        
        # 使用softmax将总评分转换为概率分布
        scores = evaluation_result['total_score']
        exp_scores = np.exp(scores / temperature)
        probabilities = exp_scores / np.sum(exp_scores)
        
        return probabilities
        
    def get_strategy_report(self, strategy_names: Optional[List[str]] = None) -> pd.DataFrame:
        """
        获取策略评估报告
        
        参数:
            strategy_names: 策略名称列表，可选
            
        返回:
            评估报告DataFrame
        """
        if not self.evaluation_history:
            return pd.DataFrame()
            
        # 使用最新的评估结果
        latest = self.evaluation_history[-1]
        
        # 准备数据
        data = []
        
        for i in range(self.num_strategies):
            strategy_name = f"策略{i+1}" if strategy_names is None else strategy_names[i]
            
            data.append({
                '策略': strategy_name,
                '期望收益': latest['expected_payoff'][i],
                '风险': latest['risk'][i],
                '稳健性': latest['robustness'][i],
                '趋势': latest['trend'][i],
                '总评分': latest['total_score'][i]
            })
            
        # 创建DataFrame
        df = pd.DataFrame(data)
        
        # 按总评分降序排序
        return df.sort_values('总评分', ascending=False).reset_index(drop=True)
        
    def get_evaluation_history(self) -> List[Dict[str, np.ndarray]]:
        """
        获取评估历史
        
        返回:
            评估历史列表
        """
        return self.evaluation_history 