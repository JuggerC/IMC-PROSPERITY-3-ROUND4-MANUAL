import numpy as np
from typing import Dict, List, Tuple, Any, Optional, Callable
from ..cognitive_hierarchy.base_model import BaseModel
from ..cognitive_hierarchy.level_k_model import LevelKModel
from ..cognitive_hierarchy.qre_model import QREModel
from ..cognitive_hierarchy.bounded_rationality_model import BoundedRationalityModel
from ..behavioral_economics.integrated_behavioral_model import IntegratedBehavioralModel
import pandas as pd

class OpponentModeler:
    """
    对手建模器
    
    使用多种认知和行为模型预测对手的策略分布。
    结合历史数据和不同模型输出，提供对对手策略分布的精确预测。
    """
    
    def __init__(self, num_strategies: int = 3, payoff_matrix: np.ndarray = None):
        """
        初始化对手建模器
        
        参数:
            num_strategies: 策略数量
            payoff_matrix: 收益矩阵
        """
        self.num_strategies = num_strategies
        self.payoff_matrix = payoff_matrix
        
        # 初始化各种模型
        self.models = {
            'level_0': LevelKModel(num_strategies, 0, payoff_matrix),
            'level_1': LevelKModel(num_strategies, 1, payoff_matrix),
            'level_2': LevelKModel(num_strategies, 2, payoff_matrix),
            'qre': QREModel(num_strategies, 1.0, payoff_matrix),
            'bounded_rationality': BoundedRationalityModel(num_strategies, 1.0, 0.1, payoff_matrix),
            'behavioral': IntegratedBehavioralModel(num_strategies)
        }
        
        # 模型权重
        self.model_weights = {
            'level_0': 0.1,
            'level_1': 0.2,
            'level_2': 0.15,
            'qre': 0.2,
            'bounded_rationality': 0.15,
            'behavioral': 0.2
        }
        
        # 历史数据
        self.opponent_history = []
        
        # 历史预测
        self.prediction_history = []
        
        # 预测精度
        self.accuracy_history = []
        
    def set_payoff_matrix(self, payoff_matrix: np.ndarray) -> None:
        """
        设置收益矩阵
        
        参数:
            payoff_matrix: 收益矩阵
        """
        self.payoff_matrix = payoff_matrix
        
        # 更新各个模型的收益矩阵
        for model_name, model in self.models.items():
            if isinstance(model, BaseModel):
                model.set_payoff_matrix(payoff_matrix)
        
    def set_model_weights(self, weights: Dict[str, float]) -> None:
        """
        设置模型权重
        
        参数:
            weights: 权重字典，键为模型名称，值为权重
        """
        # 确保所有必需的键都存在
        if not set(self.models.keys()).issubset(weights.keys()):
            missing = set(self.models.keys()) - weights.keys()
            raise ValueError(f"缺少必需的权重键: {missing}")
            
        # 归一化权重
        total = sum(weights.values())
        self.model_weights = {k: v / total for k, v in weights.items()}
        
    def update_history(self, opponent_strategy: int) -> None:
        """
        更新对手历史
        
        参数:
            opponent_strategy: 对手选择的策略索引
        """
        self.opponent_history.append(opponent_strategy)
        
    def get_empirical_distribution(self) -> np.ndarray:
        """
        根据历史数据获取经验分布
        
        返回:
            经验策略分布
        """
        if not self.opponent_history:
            # 没有历史数据，返回均匀分布
            return np.ones(self.num_strategies) / self.num_strategies
            
        # 计算历史选择的频率
        counts = np.zeros(self.num_strategies)
        for strategy in self.opponent_history:
            counts[strategy] += 1
            
        # 转换为概率分布
        distribution = counts / np.sum(counts)
        
        return distribution
        
    def get_recent_empirical_distribution(self, window_size: int = 10) -> np.ndarray:
        """
        获取最近若干轮的经验分布
        
        参数:
            window_size: 使用的历史数据窗口大小
            
        返回:
            近期经验策略分布
        """
        if not self.opponent_history:
            return np.ones(self.num_strategies) / self.num_strategies
            
        # 获取最近的历史
        recent_history = self.opponent_history[-window_size:]
        
        # 计算频率
        counts = np.zeros(self.num_strategies)
        for strategy in recent_history:
            counts[strategy] += 1
            
        # 转换为概率分布
        distribution = counts / np.sum(counts)
        
        return distribution
        
    def predict_with_model(self, model_name: str, 
                         own_strategy_distribution: Optional[np.ndarray] = None) -> np.ndarray:
        """
        使用特定模型预测对手策略分布
        
        参数:
            model_name: 模型名称
            own_strategy_distribution: 我方策略分布
            
        返回:
            预测的对手策略分布
        """
        if model_name not in self.models:
            raise ValueError(f"未知的模型名称: {model_name}")
            
        model = self.models[model_name]
        
        # 确保收益矩阵已设置
        if not hasattr(model, 'payoff_matrix') or model.payoff_matrix is None:
            raise ValueError(f"模型 {model_name} 的收益矩阵未设置")
            
        # 根据模型类型获取预测
        if isinstance(model, BaseModel):
            # 认知层次模型
            prediction = model.calculate_strategy_distribution(own_strategy_distribution)
        elif isinstance(model, IntegratedBehavioralModel):
            # 行为经济学模型
            if own_strategy_distribution is not None and hasattr(model, 'predict_strategy'):
                # 如果有predict_strategy方法且给定我方分布
                expected_payoffs = np.zeros(self.num_strategies)
                
                # 计算每个策略的期望收益
                for i in range(self.num_strategies):
                    for j in range(self.num_strategies):
                        expected_payoffs[i] += own_strategy_distribution[j] * self.payoff_matrix[i, j]
                
                prediction = model.predict_strategy(expected_payoffs)
            else:
                # 默认使用均匀分布
                prediction = np.ones(self.num_strategies) / self.num_strategies
        else:
            # 未知模型类型
            prediction = np.ones(self.num_strategies) / self.num_strategies
            
        return prediction
        
    def predict_opponent_distribution(self, 
                                   own_strategy_distribution: Optional[np.ndarray] = None) -> np.ndarray:
        """
        综合多个模型预测对手策略分布
        
        参数:
            own_strategy_distribution: 我方策略分布，可选
            
        返回:
            预测的对手策略分布
        """
        # 获取各模型的预测
        model_predictions = {}
        
        for model_name in self.models:
            model_predictions[model_name] = self.predict_with_model(model_name, own_strategy_distribution)
            
        # 获取经验分布
        empirical_distribution = self.get_empirical_distribution()
        
        # 根据历史数据量调整经验分布的权重
        empirical_weight = min(0.5, len(self.opponent_history) / 20)  # 最多0.5
        model_weight = 1.0 - empirical_weight
        
        # 加权平均模型预测
        weighted_prediction = np.zeros(self.num_strategies)
        
        for model_name, prediction in model_predictions.items():
            weighted_prediction += self.model_weights[model_name] * prediction * model_weight
            
        # 合并经验分布
        final_prediction = weighted_prediction + empirical_weight * empirical_distribution
        
        # 确保是有效的概率分布
        final_prediction = final_prediction / np.sum(final_prediction)
        
        # 记录预测
        self.prediction_history.append(final_prediction)
        
        return final_prediction
        
    def evaluate_prediction_accuracy(self, actual_strategy: int) -> float:
        """
        评估最近预测的准确性
        
        参数:
            actual_strategy: 对手实际选择的策略
            
        返回:
            预测准确度 (0-1)
        """
        if not self.prediction_history:
            return 0.0
            
        # 获取最近的预测
        latest_prediction = self.prediction_history[-1]
        
        # 计算准确度（使用预测的概率）
        accuracy = latest_prediction[actual_strategy]
        
        # 记录准确度
        self.accuracy_history.append(accuracy)
        
        return accuracy
        
    def get_average_accuracy(self, window_size: int = 10) -> float:
        """
        获取一段时间内的平均预测准确度
        
        参数:
            window_size: 时间窗口大小
            
        返回:
            平均准确度
        """
        if not self.accuracy_history:
            return 0.0
            
        # 获取最近的准确度历史
        recent_accuracy = self.accuracy_history[-window_size:]
        
        return np.mean(recent_accuracy)
        
    def adapt_model_weights(self) -> None:
        """
        基于预测准确度自适应调整模型权重
        """
        if len(self.opponent_history) < 5 or len(self.prediction_history) < 5:
            # 没有足够的历史数据
            return
            
        # 获取最近的对手策略
        recent_strategy = self.opponent_history[-1]
        
        # 计算每个模型的准确度
        model_accuracy = {}
        
        for model_name in self.models:
            # 使用模型预测上一轮对手的策略
            prediction = self.predict_with_model(model_name)
            
            # 准确度为预测概率
            accuracy = prediction[recent_strategy]
            model_accuracy[model_name] = accuracy
            
        # 根据准确度调整权重
        total_accuracy = sum(model_accuracy.values())
        
        if total_accuracy > 0:
            # 根据准确度比例调整权重
            new_weights = {model_name: acc / total_accuracy 
                         for model_name, acc in model_accuracy.items()}
            
            # 平滑调整（避免剧烈波动）
            for model_name in self.models:
                self.model_weights[model_name] = (
                    0.8 * self.model_weights[model_name] + 
                    0.2 * new_weights[model_name]
                )
                
            # 归一化
            total = sum(self.model_weights.values())
            self.model_weights = {k: v / total for k, v in self.model_weights.items()}
        
    def get_prediction_history(self) -> List[np.ndarray]:
        """
        获取预测历史
        
        返回:
            预测历史列表
        """
        return self.prediction_history
        
    def get_opponent_history(self) -> List[int]:
        """
        获取对手历史
        
        返回:
            对手历史列表
        """
        return self.opponent_history
        
    def get_model_weights(self) -> Dict[str, float]:
        """
        获取当前模型权重
        
        返回:
            模型权重字典
        """
        return self.model_weights
        
    def to_dataframe(self) -> pd.DataFrame:
        """
        将对手建模数据转换为DataFrame
        
        返回:
            包含历史数据的DataFrame
        """
        if not self.opponent_history or not self.prediction_history:
            return pd.DataFrame()
            
        data = []
        
        # 确保预测和历史长度匹配
        history_len = min(len(self.opponent_history), len(self.prediction_history))
        
        for i in range(history_len):
            row = {
                'round': i + 1,
                'actual_strategy': self.opponent_history[i]
            }
            
            # 添加预测概率
            for j in range(self.num_strategies):
                row[f'pred_prob_strategy_{j}'] = self.prediction_history[i][j]
                
            # 添加准确度（如果有）
            if i < len(self.accuracy_history):
                row['accuracy'] = self.accuracy_history[i]
                
            data.append(row)
            
        return pd.DataFrame(data) 