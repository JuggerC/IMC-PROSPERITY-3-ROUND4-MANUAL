import numpy as np
from typing import Dict, List, Tuple, Any, Optional, Callable
import networkx as nx
import pandas as pd

class NetworkInfluenceModel:
    """
    网络影响模型
    
    模拟社交网络中策略选择的影响传播，
    包括社交学习、群体从众和观点领袖的影响。
    """
    
    def __init__(self, num_agents: int = 100, num_strategies: int = 3, 
                network_type: str = 'small-world'):
        """
        初始化网络影响模型
        
        参数:
            num_agents: 网络中的智能体数量
            num_strategies: 可选策略的数量
            network_type: 网络类型，可选：'random', 'small-world', 'scale-free'
        """
        self.num_agents = num_agents
        self.num_strategies = num_strategies
        
        # 初始化网络
        self.network = self._create_network(network_type)
        
        # 每个智能体的当前策略选择
        self.agent_strategies = np.random.randint(0, num_strategies, size=num_agents)
        
        # 每个智能体的影响力
        self.influence_weights = np.ones(num_agents)
        
        # 社交学习率
        self.social_learning_rate = 0.3
        
        # 从众倾向
        self.conformity_tendency = 0.2
        
        # 观点领袖影响力增强因子
        self.leader_influence_factor = 2.0
        
        # 定义观点领袖
        self._identify_opinion_leaders()
        
        # 历史策略分布
        self.strategy_history = []
        
        # 记录当前的策略分布
        self._record_strategy_distribution()
        
    def _create_network(self, network_type: str) -> nx.Graph:
        """
        创建社交网络
        
        参数:
            network_type: 网络类型
            
        返回:
            NetworkX图对象
        """
        if network_type == 'random':
            # Erdos-Renyi随机图，连接概率0.1
            return nx.erdos_renyi_graph(self.num_agents, 0.1)
        elif network_type == 'small-world':
            # Watts-Strogatz小世界网络，每个节点4个近邻，重连概率0.1
            return nx.watts_strogatz_graph(self.num_agents, 4, 0.1)
        elif network_type == 'scale-free':
            # Barabasi-Albert无标度网络，每个新节点2个连接
            return nx.barabasi_albert_graph(self.num_agents, 2)
        else:
            # 默认使用小世界网络
            return nx.watts_strogatz_graph(self.num_agents, 4, 0.1)
            
    def _identify_opinion_leaders(self) -> None:
        """识别网络中的观点领袖"""
        # 使用节点的度中心性作为影响力的基础
        centrality = nx.degree_centrality(self.network)
        
        # 找出中心性最高的前10%的节点作为观点领袖
        centrality_values = np.array(list(centrality.values()))
        threshold = np.percentile(centrality_values, 90)
        
        # 增强观点领袖的影响力
        for i, value in enumerate(centrality_values):
            self.influence_weights[i] = 1.0 + value * 5.0  # 基础权重加上中心性的5倍
            
            # 标记观点领袖
            if value >= threshold:
                self.influence_weights[i] *= self.leader_influence_factor
                
    def set_learning_parameters(self, social_learning_rate: float, 
                              conformity_tendency: float,
                              leader_influence_factor: float) -> None:
        """
        设置社交学习参数
        
        参数:
            social_learning_rate: 社交学习率 (0-1)
            conformity_tendency: 从众倾向 (0-1)
            leader_influence_factor: 领袖影响因子 (>1)
        """
        self.social_learning_rate = max(0.0, min(1.0, social_learning_rate))
        self.conformity_tendency = max(0.0, min(1.0, conformity_tendency))
        self.leader_influence_factor = max(1.0, leader_influence_factor)
        
        # 重新计算观点领袖的影响力
        self._identify_opinion_leaders()
        
    def set_initial_strategies(self, strategies: np.ndarray) -> None:
        """
        设置初始策略分布
        
        参数:
            strategies: 初始策略数组，长度必须为num_agents
        """
        if len(strategies) != self.num_agents:
            raise ValueError(f"策略数组长度必须为{self.num_agents}")
            
        self.agent_strategies = strategies.copy()
        self._record_strategy_distribution()
        
    def set_opinion_leaders_strategy(self, strategy: int) -> None:
        """
        设置所有观点领袖的策略
        
        参数:
            strategy: 要设置的策略索引
        """
        if not 0 <= strategy < self.num_strategies:
            raise ValueError(f"策略索引必须在0到{self.num_strategies-1}之间")
            
        # 寻找影响力高于阈值的观点领袖
        leader_threshold = 1.0 * self.leader_influence_factor
        leaders = np.where(self.influence_weights >= leader_threshold)[0]
        
        # 设置他们的策略
        for leader in leaders:
            self.agent_strategies[leader] = strategy
            
        self._record_strategy_distribution()
        
    def _record_strategy_distribution(self) -> None:
        """记录当前的策略分布"""
        distribution = np.zeros(self.num_strategies)
        for strategy in self.agent_strategies:
            distribution[strategy] += 1
        distribution = distribution / self.num_agents
        
        self.strategy_history.append(distribution)
        
    def update(self, iterations: int = 1) -> np.ndarray:
        """
        更新网络，模拟社交影响的传播
        
        参数:
            iterations: 更新迭代次数
            
        返回:
            更新后的策略分布
        """
        for _ in range(iterations):
            # 创建新的策略数组
            new_strategies = self.agent_strategies.copy()
            
            # 对每个智能体
            for agent in range(self.num_agents):
                # 有一定概率进行社交学习
                if np.random.random() < self.social_learning_rate:
                    # 获取邻居节点
                    neighbors = list(self.network.neighbors(agent))
                    
                    if neighbors:
                        # 收集邻居的策略和影响力
                        neighbor_strategies = [self.agent_strategies[n] for n in neighbors]
                        neighbor_influences = [self.influence_weights[n] for n in neighbors]
                        
                        # 计算邻居策略的加权分布
                        strategy_weights = np.zeros(self.num_strategies)
                        for s, w in zip(neighbor_strategies, neighbor_influences):
                            strategy_weights[s] += w
                            
                        # 考虑从众倾向 - 增加常见策略的权重
                        if self.conformity_tendency > 0:
                            strategy_counts = np.zeros(self.num_strategies)
                            for s in neighbor_strategies:
                                strategy_counts[s] += 1
                                
                            # 对权重应用从众因子
                            conformity_boost = strategy_counts / len(neighbors) * self.conformity_tendency
                            strategy_weights = strategy_weights * (1 + conformity_boost)
                            
                        # 基于加权概率选择新策略
                        if np.sum(strategy_weights) > 0:
                            probs = strategy_weights / np.sum(strategy_weights)
                            new_strategies[agent] = np.random.choice(self.num_strategies, p=probs)
            
            # 更新策略
            self.agent_strategies = new_strategies
            
            # 记录策略分布
            self._record_strategy_distribution()
            
        # 返回当前策略分布
        return self.get_strategy_distribution()
        
    def get_strategy_distribution(self) -> np.ndarray:
        """
        获取当前策略分布
        
        返回:
            策略分布数组
        """
        distribution = np.zeros(self.num_strategies)
        for strategy in self.agent_strategies:
            distribution[strategy] += 1
        return distribution / self.num_agents
        
    def get_strategy_history(self) -> List[np.ndarray]:
        """
        获取策略分布历史
        
        返回:
            策略分布历史列表
        """
        return self.strategy_history
        
    def get_community_structure(self) -> Dict[int, List[int]]:
        """
        获取网络的社区结构
        
        返回:
            社区结构字典，键为社区ID，值为包含的节点列表
        """
        # 使用Louvain算法找出社区
        communities = nx.community.louvain_communities(self.network)
        
        # 转换为字典格式
        community_dict = {}
        for i, community in enumerate(communities):
            community_dict[i] = list(community)
            
        return community_dict
        
    def get_opinion_leaders(self) -> List[int]:
        """
        获取观点领袖列表
        
        返回:
            观点领袖的索引列表
        """
        leader_threshold = 1.0 * self.leader_influence_factor
        return list(np.where(self.influence_weights >= leader_threshold)[0])
        
    def get_network_stats(self) -> Dict[str, Any]:
        """
        获取网络统计信息
        
        返回:
            网络统计信息字典
        """
        return {
            'num_nodes': self.network.number_of_nodes(),
            'num_edges': self.network.number_of_edges(),
            'avg_degree': np.mean([d for _, d in self.network.degree()]),
            'density': nx.density(self.network),
            'clustering_coefficient': nx.average_clustering(self.network),
            'num_communities': len(self.get_community_structure()),
            'num_opinion_leaders': len(self.get_opinion_leaders()),
            'avg_path_length': nx.average_shortest_path_length(self.network)
                if nx.is_connected(self.network) else float('inf')
        }
        
    def to_dataframe(self) -> pd.DataFrame:
        """
        将模型数据转换为DataFrame
        
        返回:
            包含节点信息的DataFrame
        """
        data = []
        for i in range(self.num_agents):
            neighbors = list(self.network.neighbors(i))
            data.append({
                'agent_id': i,
                'strategy': self.agent_strategies[i],
                'influence_weight': self.influence_weights[i],
                'is_opinion_leader': self.influence_weights[i] >= 1.0 * self.leader_influence_factor,
                'degree': self.network.degree(i),
                'num_neighbors': len(neighbors)
            })
            
        return pd.DataFrame(data) 