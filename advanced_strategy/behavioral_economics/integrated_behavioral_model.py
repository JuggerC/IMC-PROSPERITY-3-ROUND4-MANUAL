import numpy as np
from typing import Dict, List, Tuple, Any, Optional, Callable
from .prospect_theory import ProspectTheoryModel
from .emotion_engine import EmotionEngine, EmotionType
from .behavioral_biases import BehavioralBiasesModel, BiasType
import pandas as pd

class IntegratedBehavioralModel:
    """
    整合行为经济学模型
    
    将前景理论、情感引擎和行为偏差模型整合在一起，
    提供对玩家行为的全面建模。
    """
    
    def __init__(self, num_strategies: int = 3):
        """
        初始化整合行为模型
        
        参数:
            num_strategies: 策略数量
        """
        self.num_strategies = num_strategies
        
        # 初始化各子模型
        self.prospect_model = ProspectTheoryModel()
        self.emotion_engine = EmotionEngine()
        self.bias_model = BehavioralBiasesModel()
        
        # 历史记录
        self.history = []
        
        # 模型权重
        self.model_weights = {
            'prospect_theory': 0.4,
            'emotion': 0.3,
            'bias': 0.3
        }
        
    def set_model_weights(self, prospect_weight: float, emotion_weight: float, bias_weight: float) -> None:
        """
        设置各子模型的权重
        
        参数:
            prospect_weight: 前景理论模型权重
            emotion_weight: 情感引擎权重
            bias_weight: 偏差模型权重
        """
        # 确保权重和为1
        total = prospect_weight + emotion_weight + bias_weight
        self.model_weights = {
            'prospect_theory': prospect_weight / total,
            'emotion': emotion_weight / total,
            'bias': bias_weight / total
        }
        
    def process_event(self, choice: int, outcome: float, event_type: str) -> None:
        """
        处理一个事件，更新所有子模型
        
        参数:
            choice: 选择的策略索引
            outcome: 结果收益
            event_type: 事件类型，例如'win', 'lose'
        """
        # 更新情感引擎
        self.emotion_engine.update_emotion(event_type)
        
        # 更新行为偏差模型的历史
        self.bias_model.update_history(choice, outcome)
        
        # 记录历史
        self.history.append({
            'choice': choice,
            'outcome': outcome,
            'event_type': event_type,
            'emotion': self.emotion_engine.emotion_state.get_dominant_emotion()[0]
        })
        
    def adjust_rationality(self, base_rationality: float) -> float:
        """
        调整理性参数
        
        参数:
            base_rationality: 基础理性参数
            
        返回:
            调整后的理性参数
        """
        return self.emotion_engine.adjust_rationality(base_rationality)
        
    def adjust_risk_aversion(self, base_risk_aversion: float) -> float:
        """
        调整风险厌恶参数
        
        参数:
            base_risk_aversion: 基础风险厌恶参数
            
        返回:
            调整后的风险厌恶参数
        """
        return self.emotion_engine.adjust_risk_preference(base_risk_aversion)
        
    def transform_payoff_matrix(self, payoff_matrix: np.ndarray) -> np.ndarray:
        """
        基于前景理论转换收益矩阵
        
        参数:
            payoff_matrix: 原始收益矩阵
            
        返回:
            转换后的收益矩阵
        """
        return self.prospect_model.adjust_payoff_matrix(payoff_matrix)
        
    def adjust_strategy_distribution(self, base_distribution: np.ndarray, 
                                   current_choice: int = None) -> np.ndarray:
        """
        调整策略分布
        
        参数:
            base_distribution: 基础策略分布
            current_choice: 当前选择，用于现状偏好
            
        返回:
            调整后的策略分布
        """
        # 准备偏差模型的参数
        bias_kwargs = {}
        if current_choice is not None:
            bias_kwargs['current_choice'] = current_choice
            
        # 提取历史值
        if self.history:
            historical_values = []
            for event in self.history[-5:]:  # 使用最近5个事件
                values = np.zeros(self.num_strategies)
                values[event['choice']] = event['outcome']
                historical_values.append(values)
                
            bias_kwargs['historical_values'] = historical_values
            
        # 应用行为偏差
        adjusted_distribution = self.bias_model.apply_all_biases(base_distribution, **bias_kwargs)
        
        return adjusted_distribution
        
    def predict_strategy(self, expected_payoffs: np.ndarray, 
                       base_rationality: float = 1.0,
                       current_choice: int = None) -> np.ndarray:
        """
        预测策略选择概率
        
        参数:
            expected_payoffs: 各策略的期望收益
            base_rationality: 基础理性参数
            current_choice: 当前选择
            
        返回:
            策略选择概率分布
        """
        # 调整理性参数
        adjusted_rationality = self.adjust_rationality(base_rationality)
        
        # 基于logit模型计算基础分布
        exp_values = np.exp(adjusted_rationality * expected_payoffs)
        base_distribution = exp_values / np.sum(exp_values)
        
        # 应用行为偏差
        adjusted_distribution = self.adjust_strategy_distribution(base_distribution, current_choice)
        
        return adjusted_distribution
        
    def calibrate_from_data(self, choices: List[int], outcomes: List[float], 
                           events: List[str]) -> Dict[str, Any]:
        """
        从历史数据校准模型
        
        参数:
            choices: 历史选择
            outcomes: 历史结果
            events: 历史事件类型
            
        返回:
            校准后的参数字典
        """
        # 校准前景理论模型
        # 为简化，假设每个选择对应一个结果和概率
        prospect_data = []
        choice_indices = []
        
        for i in range(len(choices) - 1):
            # 建立简单的选项集合
            options = []
            for s in range(self.num_strategies):
                if s == choices[i]:
                    # 实际选择的选项
                    options.append([(outcomes[i], 1.0)])
                else:
                    # 未选择的选项（假设收益为0）
                    options.append([(0.0, 1.0)])
                    
            prospect_data.append(options)
            choice_indices.append(choices[i])
            
        if prospect_data:
            prospect_params = self.prospect_model.calibrate_from_data(choice_indices, prospect_data)
        else:
            prospect_params = {}
            
        # 处理历史事件，更新模型状态
        for choice, outcome, event in zip(choices, outcomes, events):
            self.process_event(choice, outcome, event)
            
        # 返回校准结果
        return {
            'prospect_theory': prospect_params,
            'emotion_state': self.emotion_engine.emotion_state.get_dominant_emotion()[0].name,
            'bias_intensities': {
                bias.name: self.bias_model.get_bias_intensity(bias)
                for bias in BiasType
            }
        }
        
    def get_behavioral_profile(self) -> Dict[str, Any]:
        """
        获取当前行为模型的概况
        
        返回:
            行为模型概况字典
        """
        # 获取主导情感
        dominant_emotion, intensity = self.emotion_engine.emotion_state.get_dominant_emotion()
        
        # 获取最显著的行为偏差
        significant_biases = {
            bias.name: intensity
            for bias in BiasType
            if self.bias_model.get_bias_intensity(bias) > 0.3  # 只包括强度大于0.3的偏差
        }
        
        # 前景理论参数
        prospect_params = {
            'reference_point': self.prospect_model.reference_point,
            'loss_aversion': self.prospect_model.lambda_loss,
            'risk_behavior': 'risk_seeking' if self.prospect_model.alpha > self.prospect_model.beta else 'risk_averse'
        }
        
        return {
            'dominant_emotion': {
                'type': dominant_emotion.name,
                'intensity': intensity
            },
            'significant_biases': significant_biases,
            'prospect_theory': prospect_params,
            'model_weights': self.model_weights
        }
        
    def get_history_dataframe(self) -> pd.DataFrame:
        """
        获取历史数据的DataFrame
        
        返回:
            历史数据DataFrame
        """
        return pd.DataFrame(self.history)
    
    def integrate_factors(self, factor_pairs: List[Tuple[np.ndarray, float]]) -> np.ndarray:
        """
        整合多个因子，根据权重加权平均
        
        参数:
            factor_pairs: (因子数组, 权重)元组的列表
            
        返回:
            整合后的权重数组
        """
        # 初始化结果数组
        if not factor_pairs:
            return np.array([])
            
        result = np.zeros_like(factor_pairs[0][0])
        total_weight = 0.0
        
        # 加权求和
        for factor, weight in factor_pairs:
            result += factor * weight
            total_weight += weight
            
        # 归一化
        if total_weight > 0:
            result = result / total_weight
            
        # 确保结果是概率分布
        if np.sum(result) > 0:
            result = result / np.sum(result)
            
        return result 