import numpy as np
import pandas as pd
from typing import Dict, List, Tuple, Any, Optional
import matplotlib.pyplot as plt
import os

# 导入高级策略模块
from .meta_strategy.meta_strategy_integrator import MetaStrategyIntegrator
from .meta_strategy.strategy_selector import StrategySelector
from .meta_strategy.opponent_modeler import OpponentModeler

from .cognitive_hierarchy.hierarchical_model import HierarchicalModel
from .cognitive_hierarchy.level_k_model import LevelKModel

from .behavioral_economics.integrated_behavioral_model import IntegratedBehavioralModel
from .behavioral_economics.prospect_theory import ProspectTheoryModel
from .behavioral_economics.behavioral_biases import BehavioralBiasesModel
from .behavioral_economics.emotion_engine import EmotionEngine

from .social_dynamics.social_dynamics_integrator import SocialDynamicsIntegrator
from .social_dynamics.network_influence import NetworkInfluenceModel
from .social_dynamics.social_norms import SocialNormsModel

from .visualization.strategy_visualizer import StrategyVisualizer


class TreasureStrategyAnalyzer:
    """
    宝箱策略分析器
    
    使用高级策略模块分析宝箱选择问题，整合认知层次、行为经济学、
    社会动态和元策略方法，为宝箱选择提供全面的决策支持。
    """
    
    def __init__(self, treasures: List, 
                num_players: int = 4130,
                rational_pct: float = 0.45,
                heuristic_pct: float = 0.35,
                random_pct: float = 0.2,
                second_box_pct: float = 0.15,
                second_box_cost: int = 50000,
                third_box_pct: float = 0.05,
                third_box_cost: int = 100000,
                previous_selection: Dict[str, float] = None):
        """
        初始化宝箱策略分析器
        
        参数:
            treasures: 宝箱列表
            num_players: 玩家数量
            rational_pct: 理性玩家占比
            heuristic_pct: 启发式玩家占比
            random_pct: 随机玩家占比
            second_box_pct: 选择第二个宝箱的玩家占比
            second_box_cost: 选择第二个宝箱的成本
            third_box_pct: 选择第三个宝箱的玩家占比
            third_box_cost: 选择第三个宝箱的成本
            previous_selection: 上一轮的箱子选择分布，ID到百分比的映射
        """
        self.treasures = treasures
        self.num_treasures = len(treasures)
        self.num_players = num_players
        self.rational_pct = rational_pct
        self.heuristic_pct = heuristic_pct
        self.random_pct = random_pct
        self.second_box_pct = second_box_pct
        self.second_box_cost = second_box_cost
        self.third_box_pct = third_box_pct
        self.third_box_cost = third_box_cost
        self.previous_selection = previous_selection or {}
        
        # 策略空间为选择宝箱1-n
        self.strategies = list(range(self.num_treasures))
        
        # ID到索引的映射
        self.treasures_dict = {t.id: t for t in self.treasures}
        self.id_to_index = {t.id: i for i, t in enumerate(self.treasures)}
        self.index_to_id = {i: t.id for i, t in enumerate(self.treasures)}
        
        # 计算各箱子的预估选择分布
        self.estimated_selection = self._estimate_selection_distribution()
        
        # 初始化收益矩阵
        self.payoff_matrix = self._create_payoff_matrix()
        
        # 初始化各种模型
        self._init_models()
        
        # 分析结果
        self.results = {}
    
    def _estimate_selection_distribution(self) -> Dict[str, float]:
        """
        估计各箱子的选择分布
        
        根据上一轮的数据和箱子的基础效用，估计玩家对各箱子的选择分布
        """
        estimated_selection = {}
        has_previous_data = len(self.previous_selection) > 0
        
        # 首先基于基础效用计算初始分布
        total_utility = sum(t.base_utility for t in self.treasures)
        for treasure in self.treasures:
            if has_previous_data and treasure.id in self.previous_selection:
                # 如果有上一轮数据，使用它但进行一些调整
                # 玩家会更谨慎，不会那么集中在热门选择上
                prev_pct = self.previous_selection[treasure.id]
                if prev_pct > 10:
                    # 热门选择可能会降低一些
                    estimated_selection[treasure.id] = 0.6 * prev_pct
                else:
                    # 小众选择可能会增加一些
                    estimated_selection[treasure.id] = 1.2 * prev_pct
            else:
                # 如果没有上一轮数据，使用基础效用计算
                # 但也要考虑玩家更加谨慎的因素
                # 对于高效用的新箱子，预计会有更多玩家选择
                utility_ratio = treasure.base_utility / total_utility
                estimated_selection[treasure.id] = utility_ratio * 100 * 0.8  # 转为百分比且考虑谨慎因素
        
        # 归一化确保总和为100%
        total_estimated = sum(estimated_selection.values())
        if total_estimated > 0:
            for box_id in estimated_selection:
                estimated_selection[box_id] = (estimated_selection[box_id] / total_estimated) * 100
        
        return estimated_selection
        
    def _init_models(self):
        """初始化各种高级策略模型"""
        # 元策略模型
        self.meta_strategy = MetaStrategyIntegrator(
            num_strategies=self.num_treasures,
            payoff_matrix=self.payoff_matrix
        )
        
        # 策略选择器
        self.strategy_selector = StrategySelector(self.num_treasures)
        
        # 对手建模器
        self.opponent_modeler = OpponentModeler(
            num_strategies=self.num_treasures,
            payoff_matrix=self.payoff_matrix
        )
        
        # 认知层次模型
        self.cognitive_model = HierarchicalModel(
            num_strategies=self.num_treasures,
            max_level=3,
            level_distribution=None,
            payoff_matrix=self.payoff_matrix
        )
        
        # Level-K模型
        self.level_k_model = LevelKModel(
            num_strategies=self.num_treasures,
            level=2,
            payoff_matrix=self.payoff_matrix
        )
        
        # 行为经济学模型
        self.behavioral_model = IntegratedBehavioralModel(self.num_treasures)
        self.prospect_model = ProspectTheoryModel(self.num_treasures)
        self.biases_model = BehavioralBiasesModel()
        self.emotion_engine = EmotionEngine()
        
        # 社会动态模型
        self.social_model = SocialDynamicsIntegrator(
            num_strategies=self.num_treasures,
            num_agents=min(1000, self.num_players),
            network_type='small-world'
        )
        
        # 可视化器
        self.visualizer = StrategyVisualizer(self.meta_strategy)
    
    def _create_payoff_matrix(self) -> np.ndarray:
        """
        创建收益矩阵
        
        矩阵中的entry (i,j)表示当玩家选择宝箱i而其他人选择分布j时的收益
        
        返回:
            收益矩阵
        """
        # 创建空的收益矩阵
        payoff_matrix = np.zeros((self.num_treasures, self.num_treasures))
        
        # 对于每种可能的策略分布计算收益
        for i in range(self.num_treasures):
            for j in range(self.num_treasures):
                # i是自己选的宝箱，j是大多数其他玩家选的宝箱
                
                # 创建玩家分布：大部分人选j，少数选其他
                distribution = np.ones(self.num_treasures) * 0.01  # 基础分布
                distribution[j] = 0.9  # 大部分人选j
                distribution = distribution / distribution.sum()  # 归一化
                
                # 计算在这种分布下选i的收益
                treasure_i = self.treasures[i]
                selection_pct = distribution[i] * 100  # 转换为百分比
                payoff = treasure_i.calculate_profit(selection_pct)
                
                payoff_matrix[i, j] = payoff
        
        return payoff_matrix
    
    def analyze_with_cognitive_hierarchy(self) -> Dict[str, Any]:
        """
        使用认知层次模型分析宝箱选择
        
        返回:
            认知层次分析结果
        """
        # 设置合理的层次分布：多数玩家是0级和1级，少数是高级
        level_distribution = np.array([0.4, 0.4, 0.15, 0.05])
        self.cognitive_model.set_level_distribution(level_distribution)
        
        # 计算每个层次的最优策略
        level_strategies = {}
        for level in range(4):
            level_strategies[f"Level-{level}"] = self.cognitive_model.get_best_strategy_for_level(level)
        
        # 计算整体策略分布
        overall_distribution = self.cognitive_model.calculate_strategy_distribution()
        
        # 找出最优策略
        best_strategy = np.argmax(overall_distribution)
        
        # 返回结果
        return {
            "level_strategies": level_strategies,
            "overall_distribution": overall_distribution,
            "best_strategy": best_strategy,
            "best_treasure": self.treasures[best_strategy]
        }
    
    def analyze_with_behavioral_economics(self) -> Dict[str, Any]:
        """
        使用行为经济学模型分析宝箱选择
        
        返回:
            行为经济学分析结果
        """
        # 计算基础收益期望
        base_payoffs = np.mean(self.payoff_matrix, axis=1)
        
        # 应用前景理论
        # 设置参数：损失厌恶系数=2.25，价值函数曲率=0.88
        prospect_weights = self.prospect_model.calculate_prospect_weights(
            base_payoffs, 
            reference_point=np.mean(base_payoffs),
            loss_aversion=2.25,
            value_curve=0.88
        )
        
        # 应用行为偏见
        # 考虑锚定效应和代表性启发式
        bias_weights = self.biases_model.apply_biases(
            base_payoffs,
            {"anchoring": 0.3, "representativeness": 0.4, "availability": 0.3}
        )
        
        # 应用情绪影响
        emotion_weights = self.emotion_engine.calculate_emotion_influence(
            base_payoffs,
            emotion_state="neutral"
        )
        
        # 整合所有影响
        final_weights = self.behavioral_model.integrate_factors([
            (prospect_weights, 0.4),
            (bias_weights, 0.4),
            (emotion_weights, 0.2)
        ])
        
        # 找出最优策略
        best_strategy = np.argmax(final_weights)
        
        # 返回结果
        return {
            "prospect_weights": prospect_weights,
            "bias_weights": bias_weights,
            "emotion_weights": emotion_weights,
            "final_weights": final_weights,
            "best_strategy": best_strategy,
            "best_treasure": self.treasures[best_strategy]
        }
    
    def analyze_with_social_dynamics(self, num_iterations: int = 50) -> Dict[str, Any]:
        """
        使用社会动态模型分析宝箱选择
        
        参数:
            num_iterations: 模拟迭代次数
            
        返回:
            社会动态分析结果
        """
        # 设置初始分布：基于宝箱效用
        utilities = np.array([t.base_utility for t in self.treasures])
        initial_distribution = utilities / utilities.sum()
        
        # 设置网络影响参数
        self.social_model.network_model.set_learning_parameters(
            social_learning_rate=0.3,
            conformity_tendency=0.7,
            leader_influence_factor=2.0
        )
        
        # 设置社会规范参数 - 修正属性名为norm_model
        self.social_model.norm_model.set_norm_strengths(
            initial_distribution
        )
        
        # 运行社会动态模拟 - 使用update方法代替simulate_evolution
        evolution_data = []
        self.social_model.set_initial_strategies(initial_distribution)
        
        # 记录初始分布
        current_distribution = self.social_model.get_current_distribution()
        evolution_data.append(current_distribution)
        
        # 迭代更新
        for _ in range(num_iterations):
            current_distribution = self.social_model.update(1)
            evolution_data.append(current_distribution)
        
        # 获取最终分布
        final_distribution = evolution_data[-1]
        
        # 找出最优策略
        best_strategy = np.argmax(final_distribution)
        
        # 返回结果
        return {
            "initial_distribution": initial_distribution,
            "evolution_data": evolution_data,
            "final_distribution": final_distribution,
            "best_strategy": best_strategy,
            "best_treasure": self.treasures[best_strategy]
        }
    
    def analyze_with_meta_strategy(self) -> Dict[str, Any]:
        """
        使用元策略方法分析宝箱选择
        
        返回:
            元策略分析结果
        """
        # 预测对手分布
        # 基于前面的分析结果，使用加权平均
        opponent_distribution = np.zeros(self.num_treasures)
        total_weight = 0.0
        
        # 如果已有认知层次分析结果，加入分布
        if "cognitive" in self.results and "overall_distribution" in self.results["cognitive"]:
            weight = 0.3
            opponent_distribution += weight * self.results["cognitive"]["overall_distribution"]
            total_weight += weight
            
        # 如果已有行为经济学分析结果，加入分布
        if "behavioral" in self.results and "final_weights" in self.results["behavioral"]:
            weight = 0.3
            opponent_distribution += weight * self.results["behavioral"]["final_weights"]
            total_weight += weight
            
        # 如果已有社会动态分析结果，加入分布
        if "social" in self.results and "final_distribution" in self.results["social"]:
            weight = 0.4
            opponent_distribution += weight * self.results["social"]["final_distribution"]
            total_weight += weight
            
        # 如果还没有其他分析或总权重为0，使用均匀分布
        if total_weight == 0 or np.sum(opponent_distribution) == 0:
            opponent_distribution = np.ones(self.num_treasures) / self.num_treasures
        else:
            # 归一化
            opponent_distribution = opponent_distribution / np.sum(opponent_distribution)
        
        # 计算最优回应策略，传入收益矩阵和对手分布
        best_response_and_metrics = self.strategy_selector.select_strategy(
            self.payoff_matrix,
            opponent_distribution
        )
        
        # 从返回的元组中提取最优策略
        best_response = best_response_and_metrics[0]
        
        # 计算混合策略
        mixed_strategy = self.strategy_selector.get_mixed_strategy(
            self.payoff_matrix,
            opponent_distribution,
            temperature=1.0  # 使用默认温度
        )
        
        # 返回结果
        return {
            "opponent_distribution": opponent_distribution,
            "best_response": best_response,  # 直接使用策略索引
            "best_treasure": self.treasures[best_response],
            "mixed_strategy": mixed_strategy
        }
    
    def run_complete_analysis(self) -> Dict[str, Any]:
        """
        运行完整分析
        
        返回:
            完整分析结果
        """
        # 运行各种分析
        self.results["cognitive"] = self.analyze_with_cognitive_hierarchy()
        self.results["behavioral"] = self.analyze_with_behavioral_economics()
        self.results["social"] = self.analyze_with_social_dynamics()
        
        # 元策略分析需要其他分析的结果
        self.results["meta"] = self.analyze_with_meta_strategy()
        
        # 整合结果
        self.results["integrated"] = self.integrate_results()
        
        return self.results
    
    def integrate_results(self) -> Dict[str, Any]:
        """
        整合所有模型的结果，提供综合分析
        
        返回:
            综合分析结果
        """
        # 收集各模型的最佳策略
        model_best_strategies = {
            "cognitive": self.results.get("cognitive_hierarchy", {}).get("best_strategy"),
            "behavioral": self.results.get("behavioral_economics", {}).get("best_strategy"),
            "social": self.results.get("social_dynamics", {}).get("best_strategy"),
            "meta": self.results.get("meta_strategy", {}).get("best_response")
        }
        
        # 筛选出有效的结果
        model_best_strategies = {k: v for k, v in model_best_strategies.items() if v is not None}
        
        # 计算每个策略被多少个模型选为最佳
        strategy_count = {}
        for model, strategy in model_best_strategies.items():
            if strategy not in strategy_count:
                strategy_count[strategy] = []
            strategy_count[strategy].append(model)
        
        # 找出被最多模型选择的策略
        max_count = 0
        best_strategy = None
        for strategy, models in strategy_count.items():
            if len(models) > max_count:
                max_count = len(models)
                best_strategy = strategy
        
        # 计算每个箱子的预期收益
        expected_profits = {}
        for i, treasure in enumerate(self.treasures):
            box_id = treasure.id
            # 使用估计的选择分布计算收益
            selection_pct = self.estimated_selection.get(box_id, 1.0)  # 默认1%
            profit = treasure.calculate_profit(selection_pct)
            expected_profits[i] = profit
            # 也保存到索引映射
            expected_profits[box_id] = profit
        
        # 计算最佳单选箱子
        best_single_index = max(range(self.num_treasures), key=lambda i: expected_profits[i])
        best_single_profit = expected_profits[best_single_index]
        
        # 计算最佳双选箱子组合
        best_pair_indices = None
        best_pair_profit = 0
        
        for i in range(self.num_treasures):
            for j in range(i+1, self.num_treasures):
                pair_profit = expected_profits[i] + expected_profits[j] - self.second_box_cost
                if pair_profit > best_pair_profit:
                    best_pair_profit = pair_profit
                    best_pair_indices = (i, j)
        
        # 计算最佳三选箱子组合
        best_triple_indices = None
        best_triple_profit = 0
        
        for i in range(self.num_treasures):
            for j in range(i+1, self.num_treasures):
                for k in range(j+1, self.num_treasures):
                    triple_profit = expected_profits[i] + expected_profits[j] + expected_profits[k] - self.second_box_cost - self.third_box_cost
                    if triple_profit > best_triple_profit:
                        best_triple_profit = triple_profit
                        best_triple_indices = (i, j, k)
        
        # 整合结果
        return {
            "best_strategy": best_strategy,
            "best_treasure": self.treasures[best_strategy] if best_strategy is not None else None,
            "best_pair_treasures": (self.treasures[best_pair_indices[0]], self.treasures[best_pair_indices[1]]) if best_pair_indices else None,
            "best_triple_treasures": (self.treasures[best_triple_indices[0]], self.treasures[best_triple_indices[1]], self.treasures[best_triple_indices[2]]) if best_triple_indices else None,
            "model_agreement": strategy_count,
            "best_single_profit": best_single_profit,
            "best_pair_profit": best_pair_profit,
            "best_triple_profit": best_triple_profit,
            "expected_profits": expected_profits
        }
    
    def generate_report(self, output_dir: str) -> str:
        """
        生成分析报告
        
        参数:
            output_dir: 输出目录
            
        返回:
            报告文件路径
        """
        report_path = os.path.join(output_dir, "advanced_strategy_report.md")
        
        with open(report_path, 'w') as f:
            f.write("# 宝箱选择高级策略分析报告\n\n")
            
            f.write("## 分析概要\n\n")
            f.write(f"- 分析的宝箱数量: {self.num_treasures}\n")
            f.write(f"- 玩家数量: {self.num_players}\n")
            f.write(f"- 理性玩家比例: {self.rational_pct:.2f}\n")
            f.write(f"- 启发式玩家比例: {self.heuristic_pct:.2f}\n")
            f.write(f"- 随机玩家比例: {self.random_pct:.2f}\n")
            f.write(f"- 选择第二宝箱比例: {self.second_box_pct:.2f}\n")
            f.write(f"- 第二宝箱成本: {self.second_box_cost}\n")
            f.write(f"- 选择第三宝箱比例: {self.third_box_pct:.2f}\n")
            f.write(f"- 第三宝箱成本: {self.third_box_cost}\n\n")
            
            # 认知层次分析结果
            cognitive_results = self.results.get("cognitive_hierarchy", {})
            if cognitive_results:
                f.write("## 认知层次分析\n\n")
                level_strategies = cognitive_results.get("level_strategies", {})
                for level, strategy in level_strategies.items():
                    treasure = self.treasures[strategy]
                    f.write(f"- {level}思考玩家最优选择: 宝箱{treasure.id} (乘数={treasure.multiplier}, 居民={treasure.inhabitants})\n")
                
                best_strategy = cognitive_results.get("best_strategy")
                if best_strategy is not None:
                    best_treasure = self.treasures[best_strategy]
                    f.write(f"\n认知层次模型的整体最优选择: 宝箱{best_treasure.id} (乘数={best_treasure.multiplier}, 居民={best_treasure.inhabitants})\n\n")
            
            # 行为经济学分析结果
            behavioral_results = self.results.get("behavioral_economics", {})
            if behavioral_results:
                f.write("## 行为经济学分析\n\n")
                best_strategy = behavioral_results.get("best_strategy")
                if best_strategy is not None:
                    best_treasure = self.treasures[best_strategy]
                    f.write(f"考虑前景理论、行为偏见和情绪因素后的最优选择: 宝箱{best_treasure.id} (乘数={best_treasure.multiplier}, 居民={best_treasure.inhabitants})\n\n")
            
            # 社会动态分析结果
            social_results = self.results.get("social_dynamics", {})
            if social_results:
                f.write("## 社会动态分析\n\n")
                best_strategy = social_results.get("best_strategy")
                if best_strategy is not None:
                    best_treasure = self.treasures[best_strategy]
                    f.write(f"考虑社会影响和规范演化后的最优选择: 宝箱{best_treasure.id} (乘数={best_treasure.multiplier}, 居民={best_treasure.inhabitants})\n\n")
            
            # 元策略分析结果
            meta_results = self.results.get("meta_strategy", {})
            if meta_results:
                f.write("## 元策略分析\n\n")
                best_response = meta_results.get("best_response")
                if best_response is not None:
                    best_treasure = self.treasures[best_response]
                    f.write(f"基于对手建模和最优回应的最佳选择: 宝箱{best_treasure.id} (乘数={best_treasure.multiplier}, 居民={best_treasure.inhabitants})\n\n")
            
            # 综合分析结果
            integrated_results = self.integrate_results()
            f.write("## 综合分析结果\n\n")
            
            # 最佳单选策略
            f.write("### 最佳单选策略\n\n")
            best_treasure = integrated_results.get("best_treasure")
            if best_treasure:
                f.write(f"综合所有模型的最优单选: 宝箱{best_treasure.id} (乘数={best_treasure.multiplier}, 居民={best_treasure.inhabitants})\n")
                f.write(f"预期收益: {integrated_results['best_single_profit']:.2f}\n\n")
            
            # 最佳双选策略
            f.write("### 最佳双选策略\n\n")
            best_pair = integrated_results.get("best_pair_treasures")
            if best_pair:
                f.write(f"综合所有模型的最优双选: 宝箱{best_pair[0].id}和宝箱{best_pair[1].id}\n")
                f.write(f"- 宝箱{best_pair[0].id}: 乘数={best_pair[0].multiplier}, 居民={best_pair[0].inhabitants}\n")
                f.write(f"- 宝箱{best_pair[1].id}: 乘数={best_pair[1].multiplier}, 居民={best_pair[1].inhabitants}\n")
                f.write(f"预期净收益: {integrated_results['best_pair_profit']:.2f}\n\n")
            
            # 最佳三选策略
            f.write("### 最佳三选策略\n\n")
            best_triple = integrated_results.get("best_triple_treasures")
            if best_triple:
                f.write(f"综合所有模型的最优三选: 宝箱{best_triple[0].id}、宝箱{best_triple[1].id}和宝箱{best_triple[2].id}\n")
                f.write(f"- 宝箱{best_triple[0].id}: 乘数={best_triple[0].multiplier}, 居民={best_triple[0].inhabitants}\n")
                f.write(f"- 宝箱{best_triple[1].id}: 乘数={best_triple[1].multiplier}, 居民={best_triple[1].inhabitants}\n")
                f.write(f"- 宝箱{best_triple[2].id}: 乘数={best_triple[2].multiplier}, 居民={best_triple[2].inhabitants}\n")
                f.write(f"预期净收益: {integrated_results['best_triple_profit']:.2f}\n\n")
            
            # 模型一致性
            f.write("### 模型一致性\n\n")
            model_agreement = integrated_results.get("model_agreement", {})
            for strategy, models in model_agreement.items():
                treasure = self.treasures[strategy]
                f.write(f"宝箱{treasure.id} (乘数={treasure.multiplier}, 居民={treasure.inhabitants})被以下模型选为最佳: {', '.join(models)}\n")
            
            # 总结建议
            f.write("\n## 总结建议\n\n")
            
            # 找出收益最高的策略
            best_profits = {
                "单选": integrated_results.get("best_single_profit", 0),
                "双选": integrated_results.get("best_pair_profit", 0),
                "三选": integrated_results.get("best_triple_profit", 0)
            }
            
            best_strategy_type = max(best_profits.items(), key=lambda x: x[1])[0]
            
            if best_strategy_type == "单选":
                f.write(f"建议选择单一宝箱策略: 宝箱{best_treasure.id} (乘数={best_treasure.multiplier}, 居民={best_treasure.inhabitants})\n\n")
                f.write(f"预期收益: {integrated_results['best_single_profit']:.2f}\n\n")
            elif best_strategy_type == "双选":
                f.write(f"建议选择双宝箱策略: 宝箱{best_pair[0].id}和宝箱{best_pair[1].id}\n\n")
                f.write(f"预期净收益: {integrated_results['best_pair_profit']:.2f}\n\n")
            else:
                f.write(f"建议选择三宝箱策略: 宝箱{best_triple[0].id}、宝箱{best_triple[1].id}和宝箱{best_triple[2].id}\n\n")
                f.write(f"预期净收益: {integrated_results['best_triple_profit']:.2f}\n\n")
            
            # 理由说明
            f.write("理由: 综合考虑了各宝箱的基础属性、玩家行为模式和预期收益后，上述策略在当前条件下提供最高的预期收益。")
        
        return report_path 