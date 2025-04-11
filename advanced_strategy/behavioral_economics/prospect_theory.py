import numpy as np
from typing import Dict, List, Tuple, Any, Optional, Callable

class ProspectTheoryModel:
    """
    前景理论模型
    
    实现Kahneman和Tversky的前景理论，用于建模玩家风险偏好。
    包括价值函数和权重函数的实现，以及根据前景理论调整收益的方法。
    """
    
    def __init__(self, 
                 reference_point: float = 0.0,
                 alpha: float = 0.88,  # 收益区域的值函数指数
                 beta: float = 0.88,   # 损失区域的值函数指数
                 lambda_loss: float = 2.25,  # 损失厌恶参数
                 gamma: float = 0.61,  # 权重函数参数
                 delta: float = 0.69   # 权重函数参数
                ):
        """
        初始化前景理论模型
        
        参数:
            reference_point: 参考点，用于定义收益和损失
            alpha: 值函数收益部分的曲率参数，0 < alpha < 1
            beta: 值函数损失部分的曲率参数，0 < beta < 1
            lambda_loss: 损失厌恶参数，lambda > 1表示损失厌恶
            gamma: 权重函数参数，用于积极结果
            delta: 权重函数参数，用于消极结果
        """
        self.reference_point = reference_point
        self.alpha = alpha
        self.beta = beta
        self.lambda_loss = lambda_loss
        self.gamma = gamma
        self.delta = delta
        
    def value_function(self, x: float) -> float:
        """
        前景理论的值函数
        
        参数:
            x: 相对于参考点的收益
            
        返回:
            主观价值
        """
        # 相对于参考点的偏差
        dx = x - self.reference_point
        
        if dx >= 0:
            # 收益区域
            return dx ** self.alpha
        else:
            # 损失区域，注意负号
            return -self.lambda_loss * ((-dx) ** self.beta)
            
    def weight_function_positive(self, p: float) -> float:
        """
        积极结果的权重函数
        
        参数:
            p: 概率
            
        返回:
            决策权重
        """
        if p == 0:
            return 0
            
        return p ** self.gamma / ((p ** self.gamma + (1 - p) ** self.gamma) ** (1 / self.gamma))
        
    def weight_function_negative(self, p: float) -> float:
        """
        消极结果的权重函数
        
        参数:
            p: 概率
            
        返回:
            决策权重
        """
        if p == 0:
            return 0
            
        return p ** self.delta / ((p ** self.delta + (1 - p) ** self.delta) ** (1 / self.delta))
        
    def calculate_prospect_value(self, outcomes: np.ndarray, probabilities: np.ndarray) -> float:
        """
        计算前景价值
        
        参数:
            outcomes: 可能结果的数组
            probabilities: 对应结果的概率数组
            
        返回:
            前景总价值
        """
        if len(outcomes) != len(probabilities):
            raise ValueError("结果和概率数组长度必须相同")
            
        # 确保概率和为1
        if not np.isclose(np.sum(probabilities), 1.0):
            probabilities = probabilities / np.sum(probabilities)
            
        # 分离积极和消极结果
        positive_mask = outcomes >= self.reference_point
        negative_mask = ~positive_mask
        
        # 计算积极结果的前景价值
        positive_value = 0
        if np.any(positive_mask):
            pos_outcomes = outcomes[positive_mask]
            pos_probs = probabilities[positive_mask]
            
            # 应用值函数和权重函数
            for outcome, prob in zip(pos_outcomes, pos_probs):
                positive_value += self.weight_function_positive(prob) * self.value_function(outcome)
                
        # 计算消极结果的前景价值
        negative_value = 0
        if np.any(negative_mask):
            neg_outcomes = outcomes[negative_mask]
            neg_probs = probabilities[negative_mask]
            
            # 应用值函数和权重函数
            for outcome, prob in zip(neg_outcomes, neg_probs):
                negative_value += self.weight_function_negative(prob) * self.value_function(outcome)
                
        # 总前景价值
        return positive_value + negative_value
        
    def adjust_payoff_matrix(self, payoff_matrix: np.ndarray) -> np.ndarray:
        """
        使用前景理论调整收益矩阵
        
        参数:
            payoff_matrix: 原始收益矩阵
            
        返回:
            调整后的收益矩阵
        """
        adjusted_matrix = np.zeros_like(payoff_matrix)
        
        # 对每个元素应用值函数
        for i in range(payoff_matrix.shape[0]):
            for j in range(payoff_matrix.shape[1]):
                adjusted_matrix[i, j] = self.value_function(payoff_matrix[i, j])
                
        return adjusted_matrix
        
    def set_parameters(self, 
                       reference_point: Optional[float] = None,
                       alpha: Optional[float] = None,
                       beta: Optional[float] = None,
                       lambda_loss: Optional[float] = None,
                       gamma: Optional[float] = None,
                       delta: Optional[float] = None) -> None:
        """
        设置模型参数
        
        参数:
            reference_point: 参考点
            alpha: 值函数收益部分的曲率参数
            beta: 值函数损失部分的曲率参数
            lambda_loss: 损失厌恶参数
            gamma: 权重函数参数，用于积极结果
            delta: 权重函数参数，用于消极结果
        """
        if reference_point is not None:
            self.reference_point = reference_point
        if alpha is not None:
            self.alpha = alpha
        if beta is not None:
            self.beta = beta
        if lambda_loss is not None:
            self.lambda_loss = lambda_loss
        if gamma is not None:
            self.gamma = gamma
        if delta is not None:
            self.delta = delta
            
    def calibrate_from_data(self, choices: List[int], options: List[List[Tuple[float, float]]]) -> Dict[str, float]:
        """
        从实验数据中校准模型参数
        
        参数:
            choices: 选择的索引列表
            options: 选项列表，每个选项是一个(结果, 概率)元组的列表
            
        返回:
            校准后的参数字典
        """
        from scipy.optimize import minimize
        
        # 定义优化目标
        def objective(params):
            # 展开参数
            alpha, beta, lambda_loss, gamma, delta = params
            
            # 更新模型参数
            self.alpha = alpha
            self.beta = beta
            self.lambda_loss = lambda_loss
            self.gamma = gamma
            self.delta = delta
            
            # 计算预测正确的概率对数和
            log_likelihood = 0
            
            for choice_idx, option_set in zip(choices, options):
                # 计算每个选项的前景价值
                values = []
                for option in option_set:
                    outcomes = np.array([o[0] for o in option])
                    probs = np.array([o[1] for o in option])
                    values.append(self.calculate_prospect_value(outcomes, probs))
                
                # 使用softmax计算选择概率
                values = np.array(values)
                exp_values = np.exp(values)
                probs = exp_values / np.sum(exp_values)
                
                # 添加到对数似然
                if 0 <= choice_idx < len(probs):
                    log_likelihood -= np.log(probs[choice_idx] + 1e-10)  # 避免log(0)
            
            return log_likelihood
        
        # 参数边界
        bounds = [
            (0.01, 0.99),  # alpha
            (0.01, 0.99),  # beta
            (1.0, 5.0),    # lambda_loss
            (0.3, 0.9),    # gamma
            (0.3, 0.9)     # delta
        ]
        
        # 初始猜测
        initial_guess = [self.alpha, self.beta, self.lambda_loss, self.gamma, self.delta]
        
        # 优化
        result = minimize(objective, initial_guess, method='L-BFGS-B', bounds=bounds)
        
        if result.success:
            # 更新参数
            alpha, beta, lambda_loss, gamma, delta = result.x
            self.set_parameters(alpha=alpha, beta=beta, lambda_loss=lambda_loss, 
                              gamma=gamma, delta=delta)
            
            return {
                'alpha': alpha,
                'beta': beta,
                'lambda_loss': lambda_loss,
                'gamma': gamma,
                'delta': delta
            }
        else:
            # 优化失败，返回当前参数
            return {
                'alpha': self.alpha,
                'beta': self.beta,
                'lambda_loss': self.lambda_loss,
                'gamma': self.gamma,
                'delta': self.delta
            }

    def calculate_prospect_weights(self, payoffs: np.ndarray, 
                                  reference_point: float,
                                  loss_aversion: float = 2.25,
                                  value_curve: float = 0.88) -> np.ndarray:
        """
        计算基于前景理论的权重
        
        参数:
            payoffs: 收益数组
            reference_point: 参考点
            loss_aversion: 损失厌恶系数
            value_curve: 价值函数曲率
            
        返回:
            调整后的前景权重数组
        """
        # 临时设置参数
        original_ref = self.reference_point
        original_alpha = self.alpha
        original_beta = self.beta
        original_lambda = self.lambda_loss
        
        # 设置新参数
        self.reference_point = reference_point
        self.alpha = value_curve
        self.beta = value_curve
        self.lambda_loss = loss_aversion
        
        # 计算前景理论权重
        weights = np.zeros_like(payoffs, dtype=float)
        
        for i, payoff in enumerate(payoffs):
            # 应用值函数
            weights[i] = self.value_function(payoff)
            
        # 确保权重为非负数
        weights = weights - np.min(weights)
        
        # 归一化
        if np.sum(weights) > 0:
            weights = weights / np.sum(weights)
            
        # 恢复原始参数
        self.reference_point = original_ref
        self.alpha = original_alpha
        self.beta = original_beta
        self.lambda_loss = original_lambda
        
        return weights 