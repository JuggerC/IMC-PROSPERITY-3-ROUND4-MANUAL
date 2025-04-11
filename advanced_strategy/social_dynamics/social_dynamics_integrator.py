import numpy as np
from typing import Dict, List, Tuple, Any, Optional, Callable
import pandas as pd
from .network_influence import NetworkInfluenceModel
from .social_norms import SocialNormsModel

class SocialDynamicsIntegrator:
    """
    社会动态整合模型
    
    整合网络影响和社会规范模型，提供综合的社会动态模拟。
    """
    
    def __init__(self, num_strategies: int = 3, num_agents: int = 100,
               network_type: str = 'small-world'):
        """
        初始化社会动态整合模型
        
        参数:
            num_strategies: 可选策略的数量
            num_agents: 网络中的智能体数量
            network_type: 网络类型
        """
        self.num_strategies = num_strategies
        self.num_agents = num_agents
        
        # 初始化子模型
        self.network_model = NetworkInfluenceModel(
            num_agents=num_agents,
            num_strategies=num_strategies,
            network_type=network_type
        )
        
        self.norm_model = SocialNormsModel(
            num_strategies=num_strategies,
            population_size=num_agents
        )
        
        # 模型整合权重
        self.integration_weights = {
            'network': 0.6,  # 网络影响权重
            'norms': 0.4     # 社会规范权重
        }
        
        # 历史记录
        self.integrated_strategy_history = []
        
        # 记录初始策略分布
        self._record_distribution()
        
    def set_integration_weights(self, network_weight: float, norm_weight: float) -> None:
        """
        设置整合权重
        
        参数:
            network_weight: 网络影响权重
            norm_weight: 社会规范权重
        """
        total = network_weight + norm_weight
        self.integration_weights = {
            'network': network_weight / total,
            'norms': norm_weight / total
        }
        
    def _record_distribution(self) -> None:
        """记录当前策略分布"""
        network_dist = self.network_model.get_strategy_distribution()
        self.integrated_strategy_history.append(network_dist)
        
    def set_initial_strategies(self, strategy_distribution: np.ndarray) -> None:
        """
        设置初始策略分布
        
        参数:
            strategy_distribution: 策略分布概率向量
        """
        # 生成代理策略数组
        strategies = np.zeros(self.num_agents, dtype=int)
        
        # 按照分布概率分配策略
        cutoffs = np.cumsum(strategy_distribution)
        random_vals = np.random.random(self.num_agents)
        
        for i in range(self.num_agents):
            for j in range(self.num_strategies):
                if random_vals[i] <= cutoffs[j]:
                    strategies[i] = j
                    break
        
        # 设置网络模型初始策略
        self.network_model.set_initial_strategies(strategies)
        
        # 设置规范模型初始规范强度（匹配策略分布）
        self.norm_model.set_norm_strengths(strategy_distribution)
        
        # 记录初始分布
        self._record_distribution()
        
    def update(self, iterations: int = 1) -> np.ndarray:
        """
        更新社会动态，整合网络影响和社会规范
        
        参数:
            iterations: 更新迭代次数
            
        返回:
            更新后的策略分布
        """
        for _ in range(iterations):
            # 第1步: 获取当前网络策略分布
            network_distribution = self.network_model.get_strategy_distribution()
            
            # 第2步: 应用社会规范压力
            norm_adjusted_distribution = self.norm_model.apply_norm_pressure(network_distribution)
            
            # 第3步: 整合网络和规范的影响
            integrated_distribution = (
                self.integration_weights['network'] * network_distribution +
                self.integration_weights['norms'] * norm_adjusted_distribution
            )
            
            # 确保是有效的概率分布
            integrated_distribution = integrated_distribution / np.sum(integrated_distribution)
            
            # 第4步: 更新网络模型
            # 首先需要将概率分布转换为智能体策略
            agent_strategies = self.network_model.agent_strategies.copy()
            
            # 对每个智能体，有一定概率根据整合分布更新策略
            update_prob = 0.2  # 更新概率
            for agent in range(self.num_agents):
                if np.random.random() < update_prob:
                    agent_strategies[agent] = np.random.choice(
                        self.num_strategies, p=integrated_distribution
                    )
            
            # 设置更新后的策略
            self.network_model.set_initial_strategies(agent_strategies)
            
            # 第5步: 更新网络中的影响传播
            self.network_model.update(1)
            
            # 第6步: 更新社会规范
            self.norm_model.update_norms(integrated_distribution, 1)
            
            # 记录更新后的分布
            self._record_distribution()
            
        # 返回最终分布
        return self.get_current_distribution()
        
    def get_current_distribution(self) -> np.ndarray:
        """
        获取当前策略分布
        
        返回:
            策略分布数组
        """
        return self.network_model.get_strategy_distribution()
        
    def set_opinion_leaders_strategy(self, strategy: int) -> None:
        """
        设置观点领袖的策略，研究其对整体分布的影响
        
        参数:
            strategy: 策略索引
        """
        # 设置网络模型中领袖的策略
        self.network_model.set_opinion_leaders_strategy(strategy)
        
        # 记录变化
        self._record_distribution()
        
    def set_dominant_norm(self, strategy: int, strength: float = 0.7) -> None:
        """
        设置主导社会规范
        
        参数:
            strategy: 规范支持的策略索引
            strength: 规范强度 (0-1)
        """
        # 创建新的规范强度分布
        norm_strengths = np.ones(self.num_strategies) * ((1 - strength) / (self.num_strategies - 1))
        norm_strengths[strategy] = strength
        
        # 设置规范强度
        self.norm_model.set_norm_strengths(norm_strengths)
        
        # 记录变化
        self._record_distribution()
        
    def simulate_intervention(self, intervention_type: str, target_strategy: int,
                           strength: float, duration: int = 10) -> List[np.ndarray]:
        """
        模拟干预措施对策略分布的影响
        
        参数:
            intervention_type: 干预类型，'opinion_leaders'或'social_norms'
            target_strategy: 目标策略索引
            strength: 干预强度
            duration: 干预后的模拟时长
            
        返回:
            干预后的策略分布历史
        """
        # 记录初始分布
        initial_distribution = self.get_current_distribution()
        
        # 实施干预
        if intervention_type == 'opinion_leaders':
            # 通过观点领袖影响
            self.network_model.set_opinion_leaders_strategy(target_strategy)
        elif intervention_type == 'social_norms':
            # 通过社会规范影响
            self.set_dominant_norm(target_strategy, strength)
        else:
            raise ValueError(f"未知的干预类型: {intervention_type}")
            
        # 模拟干预后的演化
        history = [self.get_current_distribution()]
        for _ in range(duration):
            self.update(1)
            history.append(self.get_current_distribution())
            
        return history
        
    def get_community_strategy_distributions(self) -> Dict[int, np.ndarray]:
        """
        获取各社区的策略分布
        
        返回:
            社区策略分布字典，键为社区ID，值为策略分布
        """
        # 获取社区结构
        communities = self.network_model.get_community_structure()
        
        # 计算每个社区的策略分布
        community_distributions = {}
        
        for comm_id, members in communities.items():
            # 收集社区成员的策略
            community_strategies = [self.network_model.agent_strategies[m] for m in members]
            
            # 计算分布
            distribution = np.zeros(self.num_strategies)
            for strategy in community_strategies:
                distribution[strategy] += 1
                
            # 归一化
            distribution = distribution / len(members) if members else np.zeros(self.num_strategies)
            
            community_distributions[comm_id] = distribution
            
        return community_distributions
        
    def get_strategy_history(self) -> List[np.ndarray]:
        """
        获取策略分布历史
        
        返回:
            策略分布历史列表
        """
        return self.integrated_strategy_history
        
    def get_social_dynamics_stats(self) -> Dict[str, Any]:
        """
        获取社会动态统计信息
        
        返回:
            统计信息字典
        """
        # 获取当前分布
        current_distribution = self.get_current_distribution()
        
        # 计算策略熵（多样性）
        distribution = current_distribution + 1e-10  # 避免log(0)
        distribution = distribution / np.sum(distribution)
        strategy_entropy = -np.sum(distribution * np.log(distribution))
        max_entropy = np.log(self.num_strategies)
        normalized_entropy = strategy_entropy / max_entropy if max_entropy > 0 else 0
        
        # 获取主导策略
        dominant_strategy = np.argmax(current_distribution)
        dominant_proportion = current_distribution[dominant_strategy]
        
        # 获取规范信息
        norm_strengths = self.norm_model.get_current_norms()
        norm_dominant = np.argmax(norm_strengths)
        norm_strength = norm_strengths[norm_dominant]
        norm_diversity = self.norm_model.get_norm_diversity()
        norm_stability = self.norm_model.get_norm_stability()
        
        # 获取网络信息
        network_stats = self.network_model.get_network_stats()
        
        # 整合所有统计信息
        return {
            'strategy_distribution': current_distribution.tolist(),
            'dominant_strategy': int(dominant_strategy),
            'dominant_proportion': float(dominant_proportion),
            'strategy_diversity': float(normalized_entropy),
            'norm_strengths': norm_strengths.tolist(),
            'dominant_norm': int(norm_dominant),
            'norm_strength': float(norm_strength),
            'norm_diversity': float(norm_diversity),
            'norm_stability': float(norm_stability),
            'network_stats': network_stats,
            'integration_weights': self.integration_weights
        }
        
    def to_dataframe(self) -> pd.DataFrame:
        """
        将模型数据转换为DataFrame
        
        返回:
            包含社会动态历史的DataFrame
        """
        if not self.integrated_strategy_history:
            return pd.DataFrame()
            
        data = []
        
        for i, distribution in enumerate(self.integrated_strategy_history):
            row = {'timestep': i}
            
            # 添加策略分布
            for j, freq in enumerate(distribution):
                row[f'strategy_{j}_freq'] = freq
                
            # 如果有对应的规范历史
            if i < len(self.norm_model.norm_history):
                norms = self.norm_model.norm_history[i]
                for j, strength in enumerate(norms):
                    row[f'norm_{j}_strength'] = strength
                    
            data.append(row)
            
        return pd.DataFrame(data) 