import numpy as np
from typing import Dict, List, Tuple, Any, Optional, Callable
import pandas as pd
from enum import Enum

class EmotionType(Enum):
    """情感类型枚举"""
    JOY = "joy"             # 喜悦
    ANGER = "anger"         # 愤怒
    FEAR = "fear"           # 恐惧
    DISAPPOINTMENT = "disappointment"  # 失望
    SATISFACTION = "satisfaction"      # 满足
    REGRET = "regret"       # 后悔
    RELIEF = "relief"       # 宽慰
    NEUTRAL = "neutral"     # 中性
    
class EmotionIntensity(Enum):
    """情感强度枚举"""
    NONE = 0.0        # 无
    VERY_WEAK = 0.2   # 非常弱
    WEAK = 0.4        # 弱
    MODERATE = 0.6    # 中等
    STRONG = 0.8      # 强
    VERY_STRONG = 1.0 # 非常强

class EmotionState:
    """情感状态类，表示玩家的情感状态"""
    
    def __init__(self):
        """初始化情感状态"""
        self.emotions = {emotion: 0.0 for emotion in EmotionType}
        self.emotions[EmotionType.NEUTRAL] = 1.0  # 默认为中性
        
    def set_emotion(self, emotion_type: EmotionType, intensity: float) -> None:
        """
        设置特定情感的强度
        
        参数:
            emotion_type: 情感类型
            intensity: 情感强度，0.0-1.0
        """
        # 确保强度在有效范围内
        intensity = max(0.0, min(1.0, intensity))
        
        # 重置中性情感
        if emotion_type != EmotionType.NEUTRAL and intensity > 0:
            self.emotions[EmotionType.NEUTRAL] = 0.0
            
        self.emotions[emotion_type] = intensity
        
        # 更新中性情感
        self._normalize_emotions()
        
    def get_emotion_intensity(self, emotion_type: EmotionType) -> float:
        """
        获取特定情感的强度
        
        参数:
            emotion_type: 情感类型
            
        返回:
            情感强度，0.0-1.0
        """
        return self.emotions[emotion_type]
    
    def get_dominant_emotion(self) -> Tuple[EmotionType, float]:
        """
        获取主导情感及其强度
        
        返回:
            (情感类型, 强度)
        """
        dominant_emotion = max(self.emotions.items(), key=lambda x: x[1])
        return dominant_emotion
    
    def blend_emotions(self, other_state: 'EmotionState', weight: float = 0.5) -> None:
        """
        与另一个情感状态混合
        
        参数:
            other_state: 其他情感状态
            weight: 其他状态的权重，0.0-1.0
        """
        for emotion in EmotionType:
            self.emotions[emotion] = (1 - weight) * self.emotions[emotion] + weight * other_state.emotions[emotion]
            
        self._normalize_emotions()
        
    def decay(self, decay_rate: float = 0.1) -> None:
        """
        情感衰减
        
        参数:
            decay_rate: 衰减率，0.0-1.0
        """
        # 所有非中性情感衰减
        for emotion in EmotionType:
            if emotion != EmotionType.NEUTRAL:
                self.emotions[emotion] *= (1 - decay_rate)
                
        # 增加中性情感
        self.emotions[EmotionType.NEUTRAL] = min(1.0, self.emotions[EmotionType.NEUTRAL] + decay_rate)
        
        self._normalize_emotions()
        
    def _normalize_emotions(self) -> None:
        """归一化情感强度，确保总和为1"""
        total = sum(self.emotions.values())
        if total > 0:
            for emotion in self.emotions:
                self.emotions[emotion] /= total

class EmotionEngine:
    """
    情感引擎，用于建模情感对决策的影响
    """
    
    def __init__(self):
        """初始化情感引擎"""
        self.emotion_state = EmotionState()
        
        # 默认情感调整因子
        self.adjustment_factors = {
            EmotionType.JOY: {'risk_aversion': -0.2, 'rationality': -0.1},
            EmotionType.ANGER: {'risk_aversion': -0.3, 'rationality': -0.4},
            EmotionType.FEAR: {'risk_aversion': 0.4, 'rationality': -0.2},
            EmotionType.DISAPPOINTMENT: {'risk_aversion': 0.2, 'rationality': -0.1},
            EmotionType.SATISFACTION: {'risk_aversion': -0.1, 'rationality': 0.2},
            EmotionType.REGRET: {'risk_aversion': 0.3, 'rationality': 0.1},
            EmotionType.RELIEF: {'risk_aversion': 0.1, 'rationality': 0.1},
            EmotionType.NEUTRAL: {'risk_aversion': 0.0, 'rationality': 0.0}
        }
        
        # 情感转换矩阵
        self.emotion_transition = {
            # 当前情感 -> {事件类型 -> 新情感类型}
            EmotionType.NEUTRAL: {
                'win': EmotionType.JOY,
                'lose': EmotionType.DISAPPOINTMENT,
                'near_win': EmotionType.DISAPPOINTMENT,
                'near_lose': EmotionType.RELIEF
            },
            EmotionType.JOY: {
                'win': EmotionType.JOY,
                'lose': EmotionType.REGRET,
                'near_win': EmotionType.DISAPPOINTMENT,
                'near_lose': EmotionType.FEAR
            },
            # ... 可以扩展更多情感转换规则
        }
        
        # 情感历史记录
        self.emotion_history = []
        
    def update_emotion(self, event_type: str, intensity: float = 0.6) -> None:
        """
        基于事件更新情感状态
        
        参数:
            event_type: 事件类型，如'win', 'lose', 'near_win', 'near_lose'
            intensity: 情感变化的强度
        """
        # 获取当前主导情感
        current_emotion, _ = self.emotion_state.get_dominant_emotion()
        
        # 查找情感转换规则
        if current_emotion in self.emotion_transition and event_type in self.emotion_transition[current_emotion]:
            new_emotion = self.emotion_transition[current_emotion][event_type]
            self.emotion_state.set_emotion(new_emotion, intensity)
        
        # 记录情感历史
        self.emotion_history.append((current_emotion, event_type, self.emotion_state.get_dominant_emotion()[0]))
        
    def adjust_risk_preference(self, base_risk_aversion: float) -> float:
        """
        基于情感状态调整风险偏好
        
        参数:
            base_risk_aversion: 基础风险厌恶系数
            
        返回:
            调整后的风险厌恶系数
        """
        adjustment = 0.0
        
        # 加权求和各情感的调整
        for emotion, intensity in self.emotion_state.emotions.items():
            if intensity > 0 and emotion in self.adjustment_factors:
                adjustment += intensity * self.adjustment_factors[emotion]['risk_aversion']
                
        # 确保风险厌恶系数在合理范围内
        return max(0.0, min(1.0, base_risk_aversion + adjustment))
        
    def adjust_rationality(self, base_rationality: float) -> float:
        """
        基于情感状态调整理性程度
        
        参数:
            base_rationality: 基础理性程度
            
        返回:
            调整后的理性程度
        """
        adjustment = 0.0
        
        # 加权求和各情感的调整
        for emotion, intensity in self.emotion_state.emotions.items():
            if intensity > 0 and emotion in self.adjustment_factors:
                adjustment += intensity * self.adjustment_factors[emotion]['rationality']
                
        # 确保理性程度在合理范围内
        return max(0.0, min(5.0, base_rationality + adjustment))
        
    def set_adjustment_factor(self, emotion: EmotionType, factor_name: str, value: float) -> None:
        """
        设置情感调整因子
        
        参数:
            emotion: 情感类型
            factor_name: 因子名称，如'risk_aversion', 'rationality'
            value: 调整值
        """
        if emotion in self.adjustment_factors and factor_name in self.adjustment_factors[emotion]:
            self.adjustment_factors[emotion][factor_name] = value
            
    def predict_emotion_impact(self, history: List[str]) -> Dict[str, float]:
        """
        预测情感对未来决策的潜在影响
        
        参数:
            history: 事件历史列表
            
        返回:
            各参数的预测调整值
        """
        # 创建临时情感状态
        temp_state = EmotionState()
        
        # 模拟事件序列
        for event in history:
            # 简化版的情感更新
            if event == 'win':
                temp_state.set_emotion(EmotionType.JOY, 0.7)
            elif event == 'lose':
                temp_state.set_emotion(EmotionType.DISAPPOINTMENT, 0.7)
            # ... 其他事件类型
            
            # 应用情感衰减
            temp_state.decay(0.1)
            
        # 计算预测的参数调整
        risk_adjustment = 0.0
        rationality_adjustment = 0.0
        
        for emotion, intensity in temp_state.emotions.items():
            if emotion in self.adjustment_factors:
                risk_adjustment += intensity * self.adjustment_factors[emotion]['risk_aversion']
                rationality_adjustment += intensity * self.adjustment_factors[emotion]['rationality']
                
        return {
            'risk_aversion': risk_adjustment,
            'rationality': rationality_adjustment
        }
        
    def get_emotion_history_dataframe(self) -> pd.DataFrame:
        """
        获取情感历史数据框
        
        返回:
            情感历史的DataFrame
        """
        return pd.DataFrame(self.emotion_history, 
                           columns=['previous_emotion', 'event', 'resulting_emotion'])

    def calculate_emotion_influence(self, payoffs: np.ndarray, emotion_state: str = "neutral") -> np.ndarray:
        """
        计算情绪对payoff值的影响
        
        参数:
            payoffs: 收益数组
            emotion_state: 情绪状态描述
            
        返回:
            情绪调整后的权重数组
        """
        # 复制原始值
        emotion_weights = payoffs.copy()
        
        # 根据情绪状态应用不同的调整
        if emotion_state == "joy":
            # 喜悦状态下更倾向于风险
            emotion_weights = self._adjust_for_risk_seeking(emotion_weights, 0.2)
        elif emotion_state == "fear":
            # 恐惧状态下更厌恶风险
            emotion_weights = self._adjust_for_risk_aversion(emotion_weights, 0.3)
        elif emotion_state == "anger":
            # 愤怒状态下更倾向于风险，但不理性
            emotion_weights = self._adjust_for_risk_seeking(emotion_weights, 0.3)
            # 随机扰动
            emotion_weights += np.random.normal(0, 0.05, size=len(emotion_weights))
        elif emotion_state == "neutral":
            # 中性状态基本不调整
            pass
        
        # 确保权重非负
        emotion_weights = np.maximum(emotion_weights, 0)
        
        # 归一化
        if np.sum(emotion_weights) > 0:
            emotion_weights = emotion_weights / np.sum(emotion_weights)
            
        return emotion_weights
        
    def _adjust_for_risk_seeking(self, values: np.ndarray, intensity: float) -> np.ndarray:
        """
        调整为更风险寻求的权重
        
        参数:
            values: 原始值数组
            intensity: 调整强度
            
        返回:
            调整后的数组
        """
        # 风险寻求倾向于增加高风险高收益选项的权重
        sorted_indices = np.argsort(values)
        
        # 增加最高值的权重
        adjusted_values = values.copy()
        for i in range(1, min(3, len(values)) + 1):
            if len(sorted_indices) >= i:
                idx = sorted_indices[-i]
                adjusted_values[idx] *= (1 + intensity * (4 - i) / 3)
                
        return adjusted_values
        
    def _adjust_for_risk_aversion(self, values: np.ndarray, intensity: float) -> np.ndarray:
        """
        调整为更风险厌恶的权重
        
        参数:
            values: 原始值数组
            intensity: 调整强度
            
        返回:
            调整后的数组
        """
        # 风险厌恶倾向于减少收益变化大的选项的权重
        variance = np.var(values)
        if variance == 0:
            return values.copy()
            
        # 计算标准化的值
        norm_values = (values - np.mean(values)) / np.std(values)
        
        # 风险厌恶调整：惩罚极端值
        adjusted_values = values.copy()
        for i in range(len(values)):
            # 极端值受到惩罚
            penalty = intensity * abs(norm_values[i]) * 0.1
            adjusted_values[i] *= (1 - penalty)
            
        return adjusted_values 