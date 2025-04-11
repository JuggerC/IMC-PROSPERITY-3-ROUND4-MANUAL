import numpy as np
from typing import Dict, List, Tuple, Any, Optional, Callable
import random
from enum import Enum, auto

class BiasType(Enum):
    """行为偏差类型枚举"""
    ANCHORING = auto()          # 锚定效应
    AVAILABILITY = auto()       # 可得性偏差
    CONFIRMATION = auto()       # 确认偏差
    GAMBLER_FALLACY = auto()    # 赌徒谬误
    HOT_HAND_FALLACY = auto()   # 热手谬误
    OVERCONFIDENCE = auto()     # 过度自信
    LOSS_AVERSION = auto()      # 损失厌恶
    RECENCY_BIAS = auto()       # 近因偏差
    FRAMING_EFFECT = auto()     # 框架效应
    STATUS_QUO_BIAS = auto()    # 现状偏好
    SUNK_COST_FALLACY = auto()  # 沉没成本谬误

class BehavioralBiasesModel:
    """
    行为偏差建模器，模拟多种认知偏差对决策的影响
    """
    
    def __init__(self):
        """初始化行为偏差模型"""
        # 每种偏差的初始强度
        self.bias_intensities = {bias: 0.0 for bias in BiasType}
        
        # 历史选择和结果
        self.choice_history = []
        self.outcome_history = []
        
        # 锚点值
        self.anchor_value = None
        
        # 累计投资（用于沉没成本）
        self.cumulative_investment = 0.0
        
    def set_bias_intensity(self, bias_type: BiasType, intensity: float) -> None:
        """
        设置特定偏差的强度
        
        参数:
            bias_type: 偏差类型
            intensity: 偏差强度，0.0-1.0
        """
        self.bias_intensities[bias_type] = max(0.0, min(1.0, intensity))
        
    def get_bias_intensity(self, bias_type: BiasType) -> float:
        """
        获取特定偏差的强度
        
        参数:
            bias_type: 偏差类型
            
        返回:
            偏差强度，0.0-1.0
        """
        return self.bias_intensities[bias_type]
        
    def update_history(self, choice: int, outcome: float) -> None:
        """
        更新历史记录
        
        参数:
            choice: 选择的策略索引
            outcome: 结果收益
        """
        self.choice_history.append(choice)
        self.outcome_history.append(outcome)
        
        # 更新锚点值（如果未设置）
        if self.anchor_value is None:
            self.anchor_value = outcome
            
        # 更新累计投资
        self.cumulative_investment += abs(outcome)
        
    def set_anchor(self, value: float) -> None:
        """
        设置锚点值
        
        参数:
            value: 锚点值
        """
        self.anchor_value = value
        
    def apply_anchoring_bias(self, values: np.ndarray) -> np.ndarray:
        """
        应用锚定效应
        
        参数:
            values: 原始值数组
            
        返回:
            调整后的值数组
        """
        if self.anchor_value is None or self.bias_intensities[BiasType.ANCHORING] == 0:
            return values
            
        # 锚定效应: 向锚点值靠近
        intensity = self.bias_intensities[BiasType.ANCHORING]
        anchored_values = (1 - intensity) * values + intensity * self.anchor_value
        
        return anchored_values
        
    def apply_gambler_fallacy(self, probabilities: np.ndarray) -> np.ndarray:
        """
        应用赌徒谬误
        
        参数:
            probabilities: 原始概率数组
            
        返回:
            调整后的概率数组
        """
        if not self.choice_history or self.bias_intensities[BiasType.GAMBLER_FALLACY] == 0:
            return probabilities
            
        # 赌徒谬误: 连续出现同一结果后，倾向于选择不同的选项
        adjusted_probs = probabilities.copy()
        
        # 检查最近的选择
        recent_choices = self.choice_history[-3:]  # 看最近3次选择
        
        if len(recent_choices) >= 3 and len(set(recent_choices)) == 1:
            # 如果最近三次选择相同
            repeated_choice = recent_choices[0]
            
            # 降低重复选择的概率，增加其他选择的概率
            intensity = self.bias_intensities[BiasType.GAMBLER_FALLACY]
            reduction = adjusted_probs[repeated_choice] * intensity
            adjusted_probs[repeated_choice] -= reduction
            
            # 将减少的概率分配给其他选项
            other_indices = [i for i in range(len(adjusted_probs)) if i != repeated_choice]
            if other_indices:
                for idx in other_indices:
                    adjusted_probs[idx] += reduction / len(other_indices)
                    
        # 确保概率和为1
        return adjusted_probs / np.sum(adjusted_probs)
        
    def apply_hot_hand_fallacy(self, probabilities: np.ndarray) -> np.ndarray:
        """
        应用热手谬误
        
        参数:
            probabilities: 原始概率数组
            
        返回:
            调整后的概率数组
        """
        if len(self.choice_history) < 2 or len(self.outcome_history) < 2 or self.bias_intensities[BiasType.HOT_HAND_FALLACY] == 0:
            return probabilities
            
        # 热手谬误: 最近取得好结果的选择更可能被重复选择
        adjusted_probs = probabilities.copy()
        
        # 检查最近的选择和结果
        recent_choices = self.choice_history[-3:]
        recent_outcomes = self.outcome_history[-3:]
        
        # 找出最近带来正收益的选择
        positive_choices = [choice for choice, outcome in zip(recent_choices, recent_outcomes) if outcome > 0]
        
        if positive_choices:
            # 统计每个选择出现的频率
            choice_counts = {}
            for choice in positive_choices:
                if choice in choice_counts:
                    choice_counts[choice] += 1
                else:
                    choice_counts[choice] = 1
                    
            # 增加带来正收益选择的概率
            intensity = self.bias_intensities[BiasType.HOT_HAND_FALLACY]
            total_boost = 0.0
            
            for choice, count in choice_counts.items():
                # 增强与计数和强度成比例
                boost = adjusted_probs[choice] * intensity * (count / len(positive_choices))
                adjusted_probs[choice] += boost
                total_boost += boost
                
            # 从其他选项中减去总增强量
            other_indices = [i for i in range(len(adjusted_probs)) if i not in choice_counts]
            if other_indices and total_boost > 0:
                for idx in other_indices:
                    adjusted_probs[idx] = max(0, adjusted_probs[idx] - (total_boost * adjusted_probs[idx] / sum(adjusted_probs[i] for i in other_indices)))
                    
        # 确保概率和为1
        return adjusted_probs / np.sum(adjusted_probs)
        
    def apply_recency_bias(self, values: np.ndarray, historical_values: List[np.ndarray]) -> np.ndarray:
        """
        应用近因偏差
        
        参数:
            values: 当前值数组
            historical_values: 历史值数组列表
            
        返回:
            调整后的值数组
        """
        if not historical_values or self.bias_intensities[BiasType.RECENCY_BIAS] == 0:
            return values
            
        # 近因偏差: 最近的事件影响更大
        intensity = self.bias_intensities[BiasType.RECENCY_BIAS]
        
        # 用最近的值替换部分当前值
        most_recent = historical_values[-1]
        adjusted_values = (1 - intensity) * values + intensity * most_recent
        
        return adjusted_values
        
    def apply_framing_effect(self, gains: np.ndarray, losses: np.ndarray, frame_type: str = 'gain') -> Tuple[np.ndarray, np.ndarray]:
        """
        应用框架效应
        
        参数:
            gains: 收益数组
            losses: 损失数组
            frame_type: 框架类型，'gain'或'loss'
            
        返回:
            (调整后的收益, 调整后的损失)
        """
        if self.bias_intensities[BiasType.FRAMING_EFFECT] == 0:
            return gains, losses
            
        # 框架效应: 在收益框架下风险厌恶，在损失框架下风险寻求
        intensity = self.bias_intensities[BiasType.FRAMING_EFFECT]
        
        if frame_type == 'gain':
            # 收益框架: 高估确定收益，低估风险收益
            certain_gains = gains * (1 + intensity * 0.2)
            risky_losses = losses * (1 + intensity * 0.1)
        else:
            # 损失框架: 低估确定损失，高估风险可能的收益
            certain_gains = gains * (1 - intensity * 0.1)
            risky_losses = losses * (1 - intensity * 0.2)
            
        return certain_gains, risky_losses
        
    def apply_sunk_cost_fallacy(self, expected_values: np.ndarray, investment_per_option: np.ndarray) -> np.ndarray:
        """
        应用沉没成本谬误
        
        参数:
            expected_values: 期望值数组
            investment_per_option: 每个选项的已投资成本
            
        返回:
            调整后的期望值数组
        """
        if self.bias_intensities[BiasType.SUNK_COST_FALLACY] == 0:
            return expected_values
            
        # 沉没成本谬误: 已投入资源多的选项会获得额外价值
        intensity = self.bias_intensities[BiasType.SUNK_COST_FALLACY]
        
        # 计算每个选项的相对投资比例
        if np.sum(investment_per_option) > 0:
            relative_investment = investment_per_option / np.sum(investment_per_option)
        else:
            return expected_values
            
        # 基于投资增强期望值
        adjustment = intensity * relative_investment * np.mean(expected_values)
        adjusted_values = expected_values + adjustment
        
        return adjusted_values
        
    def apply_status_quo_bias(self, values: np.ndarray, current_choice: int) -> np.ndarray:
        """
        应用现状偏好
        
        参数:
            values: 选项的值数组
            current_choice: 当前选择的索引
            
        返回:
            调整后的值数组
        """
        if self.bias_intensities[BiasType.STATUS_QUO_BIAS] == 0:
            return values
            
        # 现状偏好: 增加当前选择的吸引力
        intensity = self.bias_intensities[BiasType.STATUS_QUO_BIAS]
        
        adjusted_values = values.copy()
        if 0 <= current_choice < len(adjusted_values):
            # 增加当前选择的值
            adjusted_values[current_choice] *= (1 + intensity * 0.3)
            
        return adjusted_values
        
    def apply_all_biases(self, original_values: np.ndarray, **kwargs) -> np.ndarray:
        """
        应用所有相关偏差
        
        参数:
            original_values: 原始值数组
            **kwargs: 应用各种偏差所需的其他参数
            
        返回:
            经所有偏差调整后的值数组
        """
        values = original_values.copy()
        
        # 应用锚定效应
        if self.bias_intensities[BiasType.ANCHORING] > 0:
            values = self.apply_anchoring_bias(values)
            
        # 应用赌徒谬误
        if self.bias_intensities[BiasType.GAMBLER_FALLACY] > 0 and 'probabilities' in kwargs:
            values = self.apply_gambler_fallacy(values if 'probabilities' not in kwargs else kwargs['probabilities'])
            
        # 应用热手谬误
        if self.bias_intensities[BiasType.HOT_HAND_FALLACY] > 0 and 'probabilities' in kwargs:
            values = self.apply_hot_hand_fallacy(values if 'probabilities' not in kwargs else kwargs['probabilities'])
            
        # 应用近因偏差
        if self.bias_intensities[BiasType.RECENCY_BIAS] > 0 and 'historical_values' in kwargs:
            values = self.apply_recency_bias(values, kwargs['historical_values'])
            
        # 应用框架效应
        if self.bias_intensities[BiasType.FRAMING_EFFECT] > 0 and 'gains' in kwargs and 'losses' in kwargs:
            gains, losses = self.apply_framing_effect(
                kwargs['gains'], kwargs['losses'], 
                kwargs.get('frame_type', 'gain')
            )
            # 这里需要根据具体情况调整返回值
            
        # 应用沉没成本谬误
        if self.bias_intensities[BiasType.SUNK_COST_FALLACY] > 0 and 'investment_per_option' in kwargs:
            values = self.apply_sunk_cost_fallacy(values, kwargs['investment_per_option'])
            
        # 应用现状偏好
        if self.bias_intensities[BiasType.STATUS_QUO_BIAS] > 0 and 'current_choice' in kwargs:
            values = self.apply_status_quo_bias(values, kwargs['current_choice'])
            
        return values 

    def apply_biases(self, values: np.ndarray, bias_weights: Dict[str, float]) -> np.ndarray:
        """
        应用多种偏见对值数组的综合影响
        
        参数:
            values: 原始值数组
            bias_weights: 各种偏见的权重字典，键为偏见名称，值为权重
            
        返回:
            应用偏见后的权重数组
        """
        # 复制原始值
        biased_values = values.copy()
        
        # 应用锚定效应
        if "anchoring" in bias_weights and bias_weights["anchoring"] > 0:
            # 使用平均值作为锚
            self.set_anchor(np.mean(values))
            biased_values = self.apply_anchoring_bias(biased_values)
            
        # 应用代表性启发式（近似为强化高价值宝箱）
        if "representativeness" in bias_weights and bias_weights["representativeness"] > 0:
            # 简单实现：增强最高价值项的权重
            max_idx = np.argmax(biased_values)
            boost = np.max(biased_values) * bias_weights["representativeness"] * 0.2
            biased_values[max_idx] += boost
            
        # 应用可得性偏见（近似为强化容易记忆的选项）
        if "availability" in bias_weights and bias_weights["availability"] > 0:
            # 简单实现：假设高/低极值更容易记忆
            sorted_indices = np.argsort(biased_values)
            high_idx = sorted_indices[-1]
            low_idx = sorted_indices[0]
            
            biased_values[high_idx] += biased_values[high_idx] * bias_weights["availability"] * 0.15
            biased_values[low_idx] -= biased_values[low_idx] * bias_weights["availability"] * 0.1
            
        # 确保值非负
        biased_values = np.maximum(biased_values, 0)
        
        # 归一化
        if np.sum(biased_values) > 0:
            biased_values = biased_values / np.sum(biased_values)
            
        return biased_values 