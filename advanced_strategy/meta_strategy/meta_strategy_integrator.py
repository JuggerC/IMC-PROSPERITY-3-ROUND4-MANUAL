import numpy as np
from typing import Dict, List, Tuple, Any, Optional, Callable
import pandas as pd
from .strategy_selector import StrategySelector
from .opponent_modeler import OpponentModeler
from ..cognitive_hierarchy.hierarchical_model import HierarchicalModel
from ..behavioral_economics.integrated_behavioral_model import IntegratedBehavioralModel
from ..social_dynamics.social_dynamics_integrator import SocialDynamicsIntegrator

class MetaStrategyIntegrator:
    """
    元策略整合器
    
    整合认知层次、行为经济学和社会动态模块，
    提供全面的策略决策框架。
    """
    
    def __init__(self, num_strategies: int = 3, payoff_matrix: np.ndarray = None):
        """
        初始化元策略整合器
        
        参数:
            num_strategies: 策略数量
            payoff_matrix: 收益矩阵
        """
        self.num_strategies = num_strategies
        self.payoff_matrix = payoff_matrix
        
        # 初始化子模块
        self.strategy_selector = StrategySelector(num_strategies)
        self.opponent_modeler = OpponentModeler(num_strategies, payoff_matrix)
        self.cognitive_model = HierarchicalModel(num_strategies, 3, None, payoff_matrix)
        self.behavioral_model = IntegratedBehavioralModel(num_strategies)
        self.social_model = SocialDynamicsIntegrator(num_strategies, 100, 'small-world')
        
        # 模块整合权重
        self.integration_weights = {
            'cognitive': 0.3,     # 认知层次权重
            'behavioral': 0.3,    # 行为经济学权重
            'social': 0.2,        # 社会动态权重
            'history': 0.2        # 历史数据权重
        }
        
        # 记录历史
        self.strategy_history = []
        self.opponent_history = []
        self.payoff_history = []
        
        # 当前状态
        self.current_strategy = None
        self.current_distribution = None
        
    def set_payoff_matrix(self, payoff_matrix: np.ndarray) -> None:
        """
        设置收益矩阵
        
        参数:
            payoff_matrix: 收益矩阵
        """
        self.payoff_matrix = payoff_matrix
        
        # 更新子模块
        self.opponent_modeler.set_payoff_matrix(payoff_matrix)
        self.cognitive_model.set_payoff_matrix(payoff_matrix)
        
    def set_integration_weights(self, weights: Dict[str, float]) -> None:
        """
        设置模块整合权重
        
        参数:
            weights: 权重字典，键为模块名称，值为权重
        """
        # 确保所有必需的键都存在
        required_keys = {'cognitive', 'behavioral', 'social', 'history'}
        if not required_keys.issubset(weights.keys()):
            missing = required_keys - weights.keys()
            raise ValueError(f"缺少必需的权重键: {missing}")
            
        # 归一化权重
        total = sum(weights.values())
        self.integration_weights = {k: v / total for k, v in weights.items()}
        
    def update_history(self, own_strategy: int, opponent_strategy: int, payoff: float) -> None:
        """
        更新历史记录
        
        参数:
            own_strategy: 我方选择的策略索引
            opponent_strategy: 对手选择的策略索引
            payoff: 获得的收益
        """
        self.strategy_history.append(own_strategy)
        self.opponent_history.append(opponent_strategy)
        self.payoff_history.append(payoff)
        
        # 更新对手建模器的历史
        self.opponent_modeler.update_history(opponent_strategy)
        
        # 评估预测准确度
        if len(self.opponent_modeler.prediction_history) > 0:
            self.opponent_modeler.evaluate_prediction_accuracy(opponent_strategy)
            
        # 自适应调整对手模型权重
        self.opponent_modeler.adapt_model_weights()
        
        # 更新行为模型
        event_type = 'win' if payoff > 0 else 'lose' if payoff < 0 else 'tie'
        self.behavioral_model.process_event(own_strategy, payoff, event_type)
        
    def get_empirical_strategy_distribution(self) -> np.ndarray:
        """
        获取历史策略经验分布
        
        返回:
            策略经验分布
        """
        if not self.strategy_history:
            return np.ones(self.num_strategies) / self.num_strategies
            
        # 计算历史选择的频率
        counts = np.zeros(self.num_strategies)
        for strategy in self.strategy_history:
            counts[strategy] += 1
            
        # 转换为概率分布
        distribution = counts / np.sum(counts)
        
        return distribution
        
    def predict_opponent_distribution(self) -> np.ndarray:
        """
        预测对手策略分布
        
        返回:
            预测的对手策略分布
        """
        # 使用对手建模器预测
        return self.opponent_modeler.predict_opponent_distribution(self.current_distribution)
        
    def calculate_optimal_strategy(self) -> Tuple[int, np.ndarray]:
        """
        计算最优策略
        
        返回:
            (最优策略索引, 策略分布)
        """
        if self.payoff_matrix is None:
            raise ValueError("收益矩阵未设置")
            
        # 预测对手分布
        opponent_distribution = self.predict_opponent_distribution()
        
        # 获取各模块的策略分布
        strategy_distributions = {}
        
        # 1. 认知层次模型分布
        strategy_distributions['cognitive'] = self.cognitive_model.calculate_strategy_distribution()
        
        # 2. 行为经济学模型分布
        # 首先计算期望收益
        expected_payoffs = np.zeros(self.num_strategies)
        for i in range(self.num_strategies):
            for j in range(self.num_strategies):
                expected_payoffs[i] += opponent_distribution[j] * self.payoff_matrix[i, j]
                
        strategy_distributions['behavioral'] = self.behavioral_model.predict_strategy(expected_payoffs)
        
        # 3. 社会动态模型分布
        # 使用当前预测的分布作为初始分布
        self.social_model.set_initial_strategies(opponent_distribution)
        strategy_distributions['social'] = self.social_model.get_current_distribution()
        
        # 4. 基于历史数据的经验分布
        strategy_distributions['history'] = self.get_empirical_strategy_distribution()
        
        # 整合各模块的分布
        integrated_distribution = np.zeros(self.num_strategies)
        for module, weight in self.integration_weights.items():
            integrated_distribution += weight * strategy_distributions[module]
            
        # 确保是有效的概率分布
        integrated_distribution = integrated_distribution / np.sum(integrated_distribution)
        
        # 使用策略选择器选择最优策略
        best_strategy, _ = self.strategy_selector.select_strategy(
            self.payoff_matrix, 
            opponent_distribution,
            [integrated_distribution] if self.current_distribution is None else [self.current_distribution, integrated_distribution]
        )
        
        # 更新当前状态
        self.current_strategy = best_strategy
        self.current_distribution = integrated_distribution
        
        return best_strategy, integrated_distribution
        
    def get_mixed_strategy(self, temperature: float = 1.0) -> np.ndarray:
        """
        获取混合策略分布
        
        参数:
            temperature: 温度参数，控制分布的平滑程度
            
        返回:
            混合策略概率分布
        """
        # 预测对手分布
        opponent_distribution = self.predict_opponent_distribution()
        
        # 使用策略选择器获取混合策略
        return self.strategy_selector.get_mixed_strategy(
            self.payoff_matrix, 
            opponent_distribution,
            None,
            temperature
        )
        
    def select_strategy(self, use_pure_strategy: bool = True, temperature: float = 1.0) -> int:
        """
        选择策略
        
        参数:
            use_pure_strategy: 是否使用纯策略
            temperature: 温度参数
            
        返回:
            选择的策略索引
        """
        if use_pure_strategy:
            # 使用纯策略
            strategy, _ = self.calculate_optimal_strategy()
            return strategy
        else:
            # 使用混合策略
            distribution = self.get_mixed_strategy(temperature)
            return np.random.choice(self.num_strategies, p=distribution)
            
    def simulate_game(self, num_rounds: int = 10, 
                     opponent_model: Callable[[List[int]], int] = None) -> pd.DataFrame:
        """
        模拟博弈过程
        
        参数:
            num_rounds: 模拟轮数
            opponent_model: 对手模型函数，接收我方历史选择，返回对手策略
            
        返回:
            模拟结果DataFrame
        """
        if self.payoff_matrix is None:
            raise ValueError("收益矩阵未设置")
            
        if opponent_model is None:
            # 默认对手模型：随机策略
            opponent_model = lambda history: np.random.randint(0, self.num_strategies)
            
        # 重置历史
        self.strategy_history = []
        self.opponent_history = []
        self.payoff_history = []
        
        # 模拟轮次
        results = []
        
        for round_num in range(num_rounds):
            # 选择策略
            my_strategy = self.select_strategy()
            
            # 获取对手策略
            opponent_strategy = opponent_model(self.strategy_history)
            
            # 计算收益
            payoff = self.payoff_matrix[my_strategy, opponent_strategy]
            
            # 更新历史
            self.update_history(my_strategy, opponent_strategy, payoff)
            
            # 记录结果
            results.append({
                'round': round_num + 1,
                'my_strategy': my_strategy,
                'opponent_strategy': opponent_strategy,
                'payoff': payoff
            })
            
        return pd.DataFrame(results)
        
    def get_strategy_report(self, strategy_names: Optional[List[str]] = None) -> pd.DataFrame:
        """
        获取策略评估报告
        
        参数:
            strategy_names: 策略名称列表，可选
            
        返回:
            评估报告DataFrame
        """
        return self.strategy_selector.get_strategy_report(strategy_names)
        
    def get_opponent_model_report(self) -> pd.DataFrame:
        """
        获取对手建模报告
        
        返回:
            对手建模报告DataFrame
        """
        return self.opponent_modeler.to_dataframe()
        
    def get_behavioral_profile(self) -> Dict[str, Any]:
        """
        获取行为模型概况
        
        返回:
            行为模型概况字典
        """
        return self.behavioral_model.get_behavioral_profile()
        
    def get_social_dynamics_stats(self) -> Dict[str, Any]:
        """
        获取社会动态统计信息
        
        返回:
            社会动态统计信息字典
        """
        return self.social_model.get_social_dynamics_stats()
        
    def get_integration_weights(self) -> Dict[str, float]:
        """
        获取模块整合权重
        
        返回:
            模块整合权重字典
        """
        return self.integration_weights
        
    def get_performance_summary(self) -> Dict[str, Any]:
        """
        获取性能概要
        
        返回:
            性能概要字典
        """
        if not self.payoff_history:
            return {'avg_payoff': 0.0, 'total_payoff': 0.0, 'num_rounds': 0}
            
        return {
            'avg_payoff': np.mean(self.payoff_history),
            'total_payoff': np.sum(self.payoff_history),
            'num_rounds': len(self.payoff_history),
            'strategy_distribution': self.get_empirical_strategy_distribution().tolist(),
            'opponent_distribution': self.opponent_modeler.get_empirical_distribution().tolist(),
            'prediction_accuracy': self.opponent_modeler.get_average_accuracy()
        }
        
    def to_dataframe(self) -> pd.DataFrame:
        """
        将模型数据转换为DataFrame
        
        返回:
            包含历史数据的DataFrame
        """
        if not self.strategy_history:
            return pd.DataFrame()
            
        data = []
        
        for i in range(len(self.strategy_history)):
            row = {
                'round': i + 1,
                'my_strategy': self.strategy_history[i],
                'opponent_strategy': self.opponent_history[i],
                'payoff': self.payoff_history[i]
            }
            
            # 添加累计收益
            row['cumulative_payoff'] = sum(self.payoff_history[:i+1])
            
            # 添加移动平均收益
            window_size = min(5, i+1)
            row['moving_avg_payoff'] = np.mean(self.payoff_history[i+1-window_size:i+1])
            
            data.append(row)
            
        return pd.DataFrame(data) 