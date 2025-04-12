#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
IMC Prosperity 宝箱选择优化模型 - 反向博弈思维分析
"""

import numpy as np
import matplotlib.pyplot as plt
from simulator import Treasure, TreasureOptimizer
from scenario_analysis import ScenarioAnalyzer, MonteCarloSimulator
import pandas as pd
from typing import Dict, List, Tuple

class ReverseGameAnalyzer:
    """反向博弈分析器"""
    
    def __init__(self, treasures: List[Treasure], base_distribution: Dict[int, float]):
        """
        初始化反向博弈分析器
        
        参数:
            treasures: 宝箱列表
            base_distribution: 基准选择分布
        """
        self.treasures = {t.id: t for t in treasures}
        self.base_distribution = base_distribution
    
    def analyze_undervalued_boxes(self, profits: Dict[int, float], 
                                 popularity_threshold: float = 0.1) -> List[int]:
        """
        分析被低估的宝箱
        
        参数:
            profits: 各宝箱的预期收益
            popularity_threshold: 受欢迎度阈值
            
        返回:
            被低估的宝箱ID列表
        """
        # 计算收益排名
        profit_ranking = {box_id: rank for rank, (box_id, _) in 
                         enumerate(sorted(profits.items(), key=lambda x: x[1], reverse=True))}
        
        # 计算受欢迎度排名
        popularity_ranking = {box_id: rank for rank, (box_id, _) in 
                            enumerate(sorted(self.base_distribution.items(), 
                                           key=lambda x: x[1], reverse=True))}
        
        # 找出收益排名高但受欢迎度排名低的宝箱
        undervalued = []
        for box_id in self.treasures:
            profit_rank = profit_ranking.get(box_id, len(self.treasures))
            popularity_rank = popularity_ranking.get(box_id, len(self.treasures))
            popularity = self.base_distribution.get(box_id, 0)
            
            # 收益排名靠前但受欢迎度低于阈值的宝箱被视为低估
            if profit_rank < len(self.treasures) / 2 and popularity < popularity_threshold:
                undervalued.append(box_id)
        
        return undervalued
    
    def analyze_reverse_strategy(self, profits: Dict[int, float]) -> List[Tuple[int, float]]:
        """
        分析反主流策略
        
        参数:
            profits: 各宝箱的预期收益
            
        返回:
            (宝箱ID, 反向收益)的列表
        """
        # 找出最受欢迎的宝箱（前3个）
        popular_boxes = [box_id for box_id, _ in 
                        sorted(self.base_distribution.items(), 
                              key=lambda x: x[1], reverse=True)[:3]]
        
        print(f"最受欢迎的宝箱: {popular_boxes}")
        
        # 模拟如果更多玩家选择这些宝箱的情况
        reverse_profits = []
        
        for box_id in self.treasures:
            # 如果是热门宝箱，增加选择百分比
            if box_id in popular_boxes:
                continue
            
            # 创建一个新的分布，其中热门宝箱的选择比例增加50%
            modified_distribution = self.base_distribution.copy()
            
            # 增加热门宝箱的选择比例
            for popular_id in popular_boxes:
                modified_distribution[popular_id] *= 1.5
            
            # 重新归一化
            total = sum(modified_distribution.values())
            for box_id_inner in modified_distribution:
                modified_distribution[box_id_inner] /= total
            
            # 计算在新分布下的收益
            treasure = self.treasures[box_id]
            modified_profit = treasure.calculate_profit(modified_distribution[box_id] * 100)
            
            # 计算收益变化
            original_profit = profits[box_id]
            profit_change = (modified_profit - original_profit) / original_profit * 100
            
            reverse_profits.append((box_id, modified_profit, profit_change))
        
        # 按收益降序排序
        reverse_profits.sort(key=lambda x: x[1], reverse=True)
        
        return reverse_profits
    
    def analyze_differentiation_value(self, profits: Dict[int, float]) -> Dict[int, float]:
        """
        分析差异化价值
        
        参数:
            profits: 各宝箱的预期收益
            
        返回:
            各宝箱的差异化价值
        """
        # 计算每个宝箱的差异化价值
        differentiation_values = {}
        
        for box_id in self.treasures:
            # 差异化价值 = 收益 × (1 - 选择比例)
            # 这反映了宝箱的收益和稀缺性
            value = profits[box_id] * (1 - self.base_distribution.get(box_id, 0))
            differentiation_values[box_id] = value
        
        # 按差异化价值降序排序
        sorted_values = sorted(differentiation_values.items(), key=lambda x: x[1], reverse=True)
        
        print("\n宝箱差异化价值分析:")
        for box_id, value in sorted_values:
            print(f"宝箱{box_id}: 差异化价值={value:.2f}, " + 
                 f"收益={profits[box_id]:.2f}, 选择比例={self.base_distribution.get(box_id, 0)*100:.2f}%")
        
        return differentiation_values
    
    def analyze_meta_game(self, profits: Dict[int, float], 
                         player_types: List[str] = ['理性', '启发式', '反主流'],
                         type_weights: List[float] = [0.2, 0.5, 0.3]) -> Dict[int, float]:
        """
        分析元游戏（考虑不同类型玩家的决策）
        
        参数:
            profits: 各宝箱的预期收益
            player_types: 玩家类型列表
            type_weights: 玩家类型权重
            
        返回:
            考虑元游戏后的修正收益
        """
        if len(player_types) != len(type_weights):
            raise ValueError("玩家类型和权重列表长度必须匹配")
        
        # 计算理性玩家的选择分布
        rational_prefs = {box_id: profit for box_id, profit in profits.items()}
        total_rational = sum(rational_prefs.values())
        for box_id in rational_prefs:
            rational_prefs[box_id] /= total_rational
        
        # 计算启发式玩家的选择分布（基于乘数）
        heuristic_prefs = {box_id: self.treasures[box_id].multiplier for box_id in self.treasures}
        total_heuristic = sum(heuristic_prefs.values())
        for box_id in heuristic_prefs:
            heuristic_prefs[box_id] /= total_heuristic
        
        # 计算反主流玩家的选择分布（倾向于选择不受欢迎的宝箱）
        reverse_prefs = {box_id: 1 - self.base_distribution.get(box_id, 0) for box_id in self.treasures}
        total_reverse = sum(reverse_prefs.values())
        for box_id in reverse_prefs:
            reverse_prefs[box_id] /= total_reverse
        
        # 整合各类型玩家的选择分布
        meta_distribution = {}
        for box_id in self.treasures:
            meta_distribution[box_id] = (
                type_weights[0] * rational_prefs.get(box_id, 0) +
                type_weights[1] * heuristic_prefs.get(box_id, 0) +
                type_weights[2] * reverse_prefs.get(box_id, 0)
            )
        
        # 重新计算收益
        meta_profits = {}
        for box_id, treasure in self.treasures.items():
            meta_profits[box_id] = treasure.calculate_profit(meta_distribution[box_id] * 100)
        
        print("\n元游戏分析 (考虑不同类型玩家):")
        print(f"玩家类型权重: 理性={type_weights[0]}, 启发式={type_weights[1]}, 反主流={type_weights[2]}")
        
        print("\n预期分布:")
        for box_id, percentage in sorted(meta_distribution.items()):
            print(f"宝箱{box_id}: {percentage*100:.2f}%")
        
        print("\n预期收益:")
        for box_id, profit in sorted(meta_profits.items(), key=lambda x: x[1], reverse=True):
            print(f"宝箱{box_id}: {profit:.2f}")
        
        return meta_profits


class OptimalStrategyFinder:
    """最优策略查找器"""
    
    def __init__(self, treasures: List[Treasure], second_box_cost: int = 50000):
        """
        初始化最优策略查找器
        
        参数:
            treasures: 宝箱列表
            second_box_cost: 选择第二个宝箱的成本
        """
        self.treasures = {t.id: t for t in treasures}
        self.second_box_cost = second_box_cost
        self.profit_scenarios = {}
    
    def add_profit_scenario(self, name: str, profits: Dict[int, float], weight: float = 1.0):
        """
        添加一个收益情景
        
        参数:
            name: 情景名称
            profits: 各宝箱的预期收益
            weight: 情景权重
        """
        self.profit_scenarios[name] = {
            'profits': profits,
            'weight': weight
        }
    
    def find_optimal_strategy(self, risk_aversion: float = 0.5) -> Dict:
        """
        查找最优策略
        
        参数:
            risk_aversion: 风险规避系数 (0-1)
            
        返回:
            最优策略信息
        """
        if not self.profit_scenarios:
            raise ValueError("请先添加收益情景")
        
        # 计算各宝箱在不同情景下的加权平均收益和标准差
        combined_profits = {}
        profit_series = {box_id: [] for box_id in self.treasures}
        
        total_weight = sum(scenario['weight'] for scenario in self.profit_scenarios.values())
        
        for scenario_name, scenario_data in self.profit_scenarios.items():
            profits = scenario_data['profits']
            weight = scenario_data['weight'] / total_weight
            
            for box_id in self.treasures:
                profit = profits.get(box_id, 0)
                profit_series[box_id].append(profit)
                
                if box_id not in combined_profits:
                    combined_profits[box_id] = 0
                combined_profits[box_id] += profit * weight
        
        # 计算风险（标准差）
        profit_risk = {}
        for box_id, profits in profit_series.items():
            profit_risk[box_id] = np.std(profits)
        
        # 计算风险调整后的收益
        risk_adjusted_profits = {}
        for box_id in self.treasures:
            # 风险调整后的收益 = 收益 - 风险规避系数 × 标准差
            risk_adjusted_profits[box_id] = combined_profits[box_id] - risk_aversion * profit_risk[box_id]
        
        # 单选最优策略
        best_single = max(risk_adjusted_profits.items(), key=lambda x: x[1])
        
        # 双选最优策略
        best_pair = None
        best_pair_profit = 0
        
        for i in self.treasures:
            for j in self.treasures:
                if i >= j:
                    continue
                
                pair_profit = risk_adjusted_profits[i] + risk_adjusted_profits[j] - self.second_box_cost
                
                if pair_profit > best_pair_profit:
                    best_pair_profit = pair_profit
                    best_pair = (i, j)
        
        # 所有宝箱的排名
        ranked_boxes = sorted(risk_adjusted_profits.items(), key=lambda x: x[1], reverse=True)
        
        # 混合风险策略（选择一个低风险和一个高收益的宝箱）
        low_risk_boxes = sorted([(box_id, risk) for box_id, risk in profit_risk.items()], 
                              key=lambda x: x[1])
        high_profit_boxes = sorted([(box_id, profit) for box_id, profit in combined_profits.items()], 
                                 key=lambda x: x[1], reverse=True)
        
        mixed_strategy = None
        mixed_strategy_profit = 0
        
        for low_risk_box_id, _ in low_risk_boxes[:3]:  # 前3个低风险宝箱
            for high_profit_box_id, _ in high_profit_boxes[:3]:  # 前3个高收益宝箱
                if low_risk_box_id != high_profit_box_id:
                    pair_profit = (risk_adjusted_profits[low_risk_box_id] + 
                                  risk_adjusted_profits[high_profit_box_id] - 
                                  self.second_box_cost)
                    
                    if pair_profit > mixed_strategy_profit:
                        mixed_strategy_profit = pair_profit
                        mixed_strategy = (low_risk_box_id, high_profit_box_id)
        
        # 最小后悔策略
        min_regret_box = None
        min_regret = float('inf')
        
        for box_id in self.treasures:
            # 计算在各情景下选择该宝箱而不是最优宝箱的后悔值
            max_regret = 0
            
            for scenario_name, scenario_data in self.profit_scenarios.items():
                profits = scenario_data['profits']
                best_profit_in_scenario = max(profits.values())
                regret = best_profit_in_scenario - profits.get(box_id, 0)
                max_regret = max(max_regret, regret)
            
            if max_regret < min_regret:
                min_regret = max_regret
                min_regret_box = box_id
        
        # 整理结果
        results = {
            'best_single': {
                'box_id': best_single[0],
                'profit': best_single[1],
                'original_profit': combined_profits[best_single[0]],
                'risk': profit_risk[best_single[0]],
                'multiplier': self.treasures[best_single[0]].multiplier,
                'inhabitants': self.treasures[best_single[0]].inhabitants
            },
            'best_pair': {
                'box_ids': best_pair,
                'profit': best_pair_profit,
                'box1_profit': combined_profits[best_pair[0]],
                'box2_profit': combined_profits[best_pair[1]],
                'box1_risk': profit_risk[best_pair[0]],
                'box2_risk': profit_risk[best_pair[1]]
            },
            'mixed_strategy': {
                'box_ids': mixed_strategy,
                'profit': mixed_strategy_profit,
                'low_risk_box': {'id': mixed_strategy[0], 'risk': profit_risk[mixed_strategy[0]]},
                'high_profit_box': {'id': mixed_strategy[1], 'profit': combined_profits[mixed_strategy[1]]}
            },
            'min_regret_strategy': {
                'box_id': min_regret_box,
                'max_regret': min_regret,
                'profit': combined_profits[min_regret_box],
                'risk': profit_risk[min_regret_box]
            },
            'ranked_boxes': ranked_boxes
        }
        
        return results
    
    def print_strategy_recommendation(self, results: Dict, verbose: bool = True):
        """
        打印策略推荐
        
        参数:
            results: 查找最优策略的结果
            verbose: 是否打印详细信息
        """
        best_single = results['best_single']
        best_pair = results['best_pair']
        mixed_strategy = results['mixed_strategy']
        min_regret_strategy = results['min_regret_strategy']
        
        print("\n=============== 策略推荐 ===============")
        
        print("\n1. 单选最优策略:")
        print(f"   选择宝箱{best_single['box_id']}")
        print(f"   预期收益: {best_single['original_profit']:.2f}")
        print(f"   风险(标准差): {best_single['risk']:.2f}")
        print(f"   风险调整收益: {best_single['profit']:.2f}")
        print(f"   宝箱参数: 乘数={best_single['multiplier']}, 居民={best_single['inhabitants']}")
        
        print("\n2. 双选最优策略:")
        print(f"   选择宝箱{best_pair['box_ids'][0]}和宝箱{best_pair['box_ids'][1]}")
        print(f"   预期净收益: {best_pair['profit']:.2f} (减去成本{self.second_box_cost})")
        print(f"   宝箱{best_pair['box_ids'][0]}收益: {best_pair['box1_profit']:.2f}, 风险: {best_pair['box1_risk']:.2f}")
        print(f"   宝箱{best_pair['box_ids'][1]}收益: {best_pair['box2_profit']:.2f}, 风险: {best_pair['box2_risk']:.2f}")
        
        print("\n3. 混合风险策略:")
        print(f"   选择宝箱{mixed_strategy['box_ids'][0]}(低风险)和宝箱{mixed_strategy['box_ids'][1]}(高收益)")
        print(f"   预期净收益: {mixed_strategy['profit']:.2f} (减去成本{self.second_box_cost})")
        print(f"   低风险宝箱{mixed_strategy['low_risk_box']['id']}: 风险={mixed_strategy['low_risk_box']['risk']:.2f}")
        print(f"   高收益宝箱{mixed_strategy['high_profit_box']['id']}: 收益={mixed_strategy['high_profit_box']['profit']:.2f}")
        
        print("\n4. 最小后悔策略:")
        print(f"   选择宝箱{min_regret_strategy['box_id']}")
        print(f"   最大后悔值: {min_regret_strategy['max_regret']:.2f}")
        print(f"   预期收益: {min_regret_strategy['profit']:.2f}")
        print(f"   风险: {min_regret_strategy['risk']:.2f}")
        
        if verbose:
            print("\n所有宝箱排名(风险调整后):")
            for rank, (box_id, profit) in enumerate(results['ranked_boxes']):
                print(f"   {rank+1}. 宝箱{box_id}: 收益={profit:.2f}")
        
        # 最终建议
        print("\n=============== 最终建议 ===============")
        
        best_single_profit = best_single['original_profit']
        best_pair_profit = best_pair['box1_profit'] + best_pair['box2_profit'] - self.second_box_cost
        
        if best_pair_profit > best_single_profit:
            if best_pair_profit > mixed_strategy['profit']:
                print(f"推荐选择宝箱{best_pair['box_ids'][0]}和宝箱{best_pair['box_ids'][1]}")
                print(f"预期净收益: {best_pair_profit:.2f}")
                print("理由: 这对宝箱组合提供最高的风险调整后收益，超过单选策略和其他对组合。")
            else:
                print(f"推荐选择宝箱{mixed_strategy['box_ids'][0]}和宝箱{mixed_strategy['box_ids'][1]}")
                print(f"预期净收益: {mixed_strategy['profit']:.2f}")
                print("理由: 这个混合风险策略在保持较高收益的同时降低了风险。")
        else:
            print(f"推荐选择宝箱{best_single['box_id']}")
            print(f"预期收益: {best_single_profit:.2f}")
            print("理由: 单选策略的收益超过了任何双选策略(考虑第二个宝箱的成本)。")
        
        # 情境建议
        print("\n=============== 针对不同情境的建议 ===============")
        print("1. 如果您偏好最大化期望收益:")
        if best_pair_profit > best_single_profit:
            print(f"   选择宝箱{best_pair['box_ids'][0]}和宝箱{best_pair['box_ids'][1]}")
        else:
            print(f"   选择宝箱{best_single['box_id']}")
        
        print("\n2. 如果您偏好风险最小化:")
        print(f"   选择宝箱{mixed_strategy['low_risk_box']['id']}")
        
        print("\n3. 如果您偏好平衡风险和收益:")
        print(f"   选择宝箱{mixed_strategy['box_ids'][0]}和宝箱{mixed_strategy['box_ids'][1]}")
        
        print("\n4. 如果您担心做出错误决策:")
        print(f"   选择宝箱{min_regret_strategy['box_id']}")
        print("   这是'最小最大后悔'策略，确保在最坏情况下的后悔最小化。")


def main():
    """主函数"""
    # 创建宝箱列表
    treasures = [
        Treasure(1, 80, 6),
        Treasure(2, 37, 3),
        Treasure(3, 10, 1),
        Treasure(4, 31, 2),
        Treasure(5, 17, 1),
        Treasure(6, 90, 10),
        Treasure(7, 50, 4),
        Treasure(8, 20, 2),
        Treasure(9, 73, 4),
        Treasure(10, 89, 8)
    ]
    
    # 创建基准优化器
    print("运行基准模型...")
    base_optimizer = TreasureOptimizer(
        treasures=treasures,
        rational_pct=0.2,
        heuristic_pct=0.5,
        second_box_pct=0.1
    )
    
    base_distribution, base_profits = base_optimizer.run_iteration()
    
    # 创建情景分析器
    print("\n=== 多情景分析 ===")
    scenario_analyzer = ScenarioAnalyzer(treasures)
    
    # 添加不同情景
    scenario_analyzer.add_scenario("基准", 0.2, 0.5, 0.1)
    scenario_analyzer.add_scenario("高理性", 0.4, 0.4, 0.1)
    scenario_analyzer.add_scenario("低理性", 0.1, 0.6, 0.1)
    scenario_analyzer.add_scenario("极端启发式", 0.05, 0.85, 0.1)
    
    # 运行情景分析
    scenario_analyzer.run_all_scenarios()
    
    # 获取所有情景的收益
    scenario_profits = {name: result['profits'] for name, result in scenario_analyzer.results.items()}
    
    # 创建反向博弈分析器
    print("\n=== 反向博弈分析 ===")
    reverse_analyzer = ReverseGameAnalyzer(treasures, base_distribution)
    
    # 分析被低估的宝箱
    undervalued_boxes = reverse_analyzer.analyze_undervalued_boxes(base_profits)
    print(f"\n被低估的宝箱: {undervalued_boxes}")
    
    # 分析反主流策略
    reverse_profits = reverse_analyzer.analyze_reverse_strategy(base_profits)
    print("\n反主流策略分析:")
    for box_id, profit, change in reverse_profits:
        print(f"宝箱{box_id}: 修正收益={profit:.2f}, 变化={change:.2f}%")
    
    # 分析差异化价值
    differentiation_values = reverse_analyzer.analyze_differentiation_value(base_profits)
    
    # 分析元游戏
    meta_profits = reverse_analyzer.analyze_meta_game(base_profits)
    
    # 创建最优策略查找器
    print("\n=== 最优策略查找 ===")
    strategy_finder = OptimalStrategyFinder(treasures)
    
    # 添加不同情景的收益
    strategy_finder.add_profit_scenario("基准", base_profits, 0.4)
    for name, profits in scenario_profits.items():
        if name != "基准":
            strategy_finder.add_profit_scenario(name, profits, 0.15)
    
    strategy_finder.add_profit_scenario("反主流", {box_id: profit for box_id, profit, _ in reverse_profits}, 0.1)
    strategy_finder.add_profit_scenario("元游戏", meta_profits, 0.2)
    
    # 查找最优策略
    results = strategy_finder.find_optimal_strategy(risk_aversion=0.5)
    
    # 打印策略推荐
    strategy_finder.print_strategy_recommendation(results)


if __name__ == "__main__":
    main() 