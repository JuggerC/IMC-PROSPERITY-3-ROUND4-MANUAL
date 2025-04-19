#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
IMC Prosperity 宝箱选择优化模型
"""

import numpy as np
import matplotlib.pyplot as plt
from collections import defaultdict
import pandas as pd
from typing import Dict, List, Tuple, Union, Any

class Treasure:
    """宝箱类，表示一个可选择的宝箱"""
    
    def __init__(self, box_id: Union[int, str], multiplier: int, inhabitants: int):
        """
        初始化宝箱
        
        参数:
            box_id: 宝箱编号(可以是整数或字符串，如"A1", "B3"等)
            multiplier: 宝藏乘数
            inhabitants: 预设的居民数量
        """
        self.id = box_id
        self.multiplier = multiplier
        self.inhabitants = inhabitants
        self.base_treasure = 10000  # 基础宝藏值
        
        # 计算基础指标
        self.m_h_ratio = multiplier / inhabitants  # 乘数与居民比
        self.base_utility = multiplier / (inhabitants + 1)  # 基础效用
        
        # 动态数据
        self.selection_percentage = 0.0  # 选择该宝箱的玩家百分比
        self.expected_profit = 0.0  # 预期收益
        
    def calculate_profit(self, selection_percentage: float) -> float:
        """
        计算给定选择百分比下的宝箱收益
        
        参数:
            selection_percentage: 选择该宝箱的玩家百分比 (0-100)
            
        返回:
            计算出的收益
        """
        total_treasure = self.base_treasure * self.multiplier
        profit = total_treasure / (self.inhabitants + selection_percentage)
        return profit
    
    def __str__(self) -> str:
        return f"宝箱{self.id}: 乘数={self.multiplier}, 居民={self.inhabitants}, M/H比={self.m_h_ratio:.2f}"
    
    def __repr__(self) -> str:
        return self.__str__()


class TreasureOptimizer:
    """宝箱选择优化器"""
    
    def __init__(self, treasures: List[Treasure], num_players: int = 4130, 
                 rational_pct: float = 0.45, heuristic_pct: float = 0.35,
                 second_box_pct: float = 0.15, second_box_cost: int = 50000,
                 third_box_pct: float = 0.05, third_box_cost: int = 100000,
                 second_choice_rational_factor: float = 1.0,
                 third_choice_rational_factor: float = 1.0,
                 previous_selection: Dict[str, float] = None):
        """
        初始化优化器
        
        参数:
            treasures: 宝箱列表
            num_players: 玩家总数
            rational_pct: 理性玩家比例
            heuristic_pct: 启发式玩家比例
            second_box_pct: 选择第二个宝箱的玩家比例
            second_box_cost: 选择第二个宝箱的成本
            third_box_pct: 选择第三个宝箱的玩家比例
            third_box_cost: 选择第三个宝箱的成本
            second_choice_rational_factor: 第二次选择的理性调整因子，大于1表示更理性，小于1表示更依赖启发式
            third_choice_rational_factor: 第三次选择的理性调整因子
            previous_selection: 上一轮的箱子选择分布，ID到百分比的映射
        """
        self.treasures = treasures
        self.num_players = num_players
        self.rational_pct = rational_pct
        self.heuristic_pct = heuristic_pct
        self.random_pct = 1.0 - rational_pct - heuristic_pct
        self.second_box_pct = second_box_pct
        self.second_box_cost = second_box_cost
        self.third_box_pct = third_box_pct
        self.third_box_cost = third_box_cost
        self.second_choice_rational_factor = second_choice_rational_factor
        self.third_choice_rational_factor = third_choice_rational_factor
        self.previous_selection = previous_selection or {}
        
        # 计算总选择次数
        self.total_selections = num_players * (1 + second_box_pct + third_box_pct)
        
        # ID到索引的映射
        self.id_to_index = {t.id: i for i, t in enumerate(self.treasures)}
        self.index_to_id = {i: t.id for i, t in enumerate(self.treasures)}
        
        # 初始分布
        self._calculate_initial_distribution()
        
    def _calculate_initial_distribution(self):
        """计算初始选择分布"""
        # 计算总效用
        total_utility = sum(t.base_utility for t in self.treasures)
        
        # 如果有上一轮的选择分布，使用它来调整初始分布
        has_previous_data = len(self.previous_selection) > 0
        
        # 计算初始分布
        for treasure in self.treasures:
            if has_previous_data and treasure.id in self.previous_selection:
                # 如果有上一轮数据，使用它但进行一些调整
                # 玩家会更谨慎，不会那么集中在热门选择上
                prev_pct = self.previous_selection[treasure.id]
                if prev_pct > 10:
                    # 热门选择可能会降低一些
                    treasure.initial_probability = 0.6 * prev_pct / 100
                else:
                    # 小众选择可能会增加一些
                    treasure.initial_probability = 1.2 * prev_pct / 100
            else:
                # 如果没有上一轮数据，使用基础效用计算
                treasure.initial_probability = treasure.base_utility / total_utility
            
        # 归一化
        total_prob = sum(t.initial_probability for t in self.treasures)
        for treasure in self.treasures:
            treasure.initial_probability /= total_prob
            
        # 打印初始分布
        print("\n预估选择分布:")
        for treasure in self.treasures:
            print(f"宝箱{treasure.id}: {treasure.initial_probability:.4f} " + 
                  f"({treasure.initial_probability * 100:.2f}%)")
    
    def calculate_expected_profits(self, first_distribution: Dict[Union[int, str], float], 
                                 second_distributions: Dict[Union[int, str], Dict[Union[int, str], float]] = None,
                                 third_distributions: Dict[Union[int, str], Dict[Union[int, str], Dict[Union[int, str], float]]] = None) -> Dict[Union[int, str], float]:
        """
        计算给定分布下的预期收益
        
        参数:
            first_distribution: 宝箱ID到第一选择概率的映射
            second_distributions: 第一选择宝箱ID到第二选择分布的映射，如果为None则不考虑第二选择
            third_distributions: 第一选择宝箱ID到第二选择宝箱ID到第三选择分布的映射，如果为None则不考虑第三选择
            
        返回:
            宝箱ID到预期收益的映射
        """
        profits = {}
        
        if second_distributions is None and third_distributions is None:
            # 如果没有第二和第三选择分布，只考虑第一选择
            for treasure in self.treasures:
                box_id = treasure.id
                percentage = first_distribution.get(box_id, 0) * 100  # 转换为百分比
                profit = treasure.calculate_profit(percentage)
                profits[box_id] = profit
        else:
            # 考虑第一选择、第二选择和第三选择
            
            # 计算每个宝箱被选择的总概率
            total_choice = {box_id: first_distribution.get(box_id, 0) for box_id in self.id_to_index.keys()}
            
            # 添加第二选择
            if second_distributions is not None and self.second_box_pct > 0:
                for first_box_id, first_prob in first_distribution.items():
                    if first_box_id in second_distributions:
                        for second_box_id, second_prob in second_distributions[first_box_id].items():
                            total_choice[second_box_id] += first_prob * second_prob * self.second_box_pct
            
            # 添加第三选择
            if third_distributions is not None and self.third_box_pct > 0:
                for first_box_id, first_prob in first_distribution.items():
                    if first_box_id in second_distributions:
                        for second_box_id, second_prob in second_distributions[first_box_id].items():
                            if second_box_id in third_distributions.get(first_box_id, {}):
                                for third_box_id, third_prob in third_distributions[first_box_id][second_box_id].items():
                                    total_choice[third_box_id] += first_prob * second_prob * third_prob * self.third_box_pct
            
            # 计算每个宝箱的总选择百分比和收益
            for treasure in self.treasures:
                box_id = treasure.id
                total_percentage = total_choice.get(box_id, 0) * 100  # 转换为百分比
                profit = treasure.calculate_profit(total_percentage)
                profits[box_id] = profit
        
        return profits
    
    def run_iteration(self, max_iterations: int = 10, convergence_threshold: float = 0.01) -> Tuple[Dict[Union[int, str], float], Dict[Union[int, str], float]]:
        """
        运行迭代计算，直到收敛或达到最大迭代次数
        
        参数:
            max_iterations: 最大迭代次数
            convergence_threshold: 收敛阈值
            
        返回:
            (最终第一选择分布, 最终收益)的元组
        """
        # 初始第一选择分布
        first_distribution = {t.id: t.initial_probability for t in self.treasures}
        second_distributions = None
        third_distributions = None
        
        for iteration in range(max_iterations):
            # 如果有第二选择，计算第二选择分布
            if self.second_box_pct > 0:
                # 使用当前利润计算第二选择分布
                current_profits = self.calculate_expected_profits(first_distribution)
                second_distributions = self.calculate_second_choice_distribution(first_distribution, current_profits)
                
                # 如果有第三选择，计算第三选择分布
                if self.third_box_pct > 0:
                    third_distributions = self.calculate_third_choice_distribution(first_distribution, second_distributions, current_profits)
            
            # 考虑第一、第二和第三选择计算收益
            profits = self.calculate_expected_profits(first_distribution, second_distributions, third_distributions)
            
            # 基于收益更新理性玩家的选择（仅针对第一选择）
            total_profit = sum(profits.values())
            rational_distribution = {
                box_id: profit / total_profit for box_id, profit in profits.items()
            }
            
            # 启发式玩家分布（基于初始分布）
            heuristic_distribution = {
                t.id: t.initial_probability for t in self.treasures
            }
            
            # 随机玩家分布（均匀）
            random_distribution = {
                t.id: 1.0 / len(self.treasures) for t in self.treasures
            }
            
            # 计算新的第一选择综合分布
            new_first_distribution = {}
            for box_id in first_distribution.keys():
                new_first_distribution[box_id] = (
                    self.rational_pct * rational_distribution.get(box_id, 0) +
                    self.heuristic_pct * heuristic_distribution.get(box_id, 0) +
                    self.random_pct * random_distribution.get(box_id, 0)
                )
            
            # 检查收敛
            max_change = max(abs(new_first_distribution[box_id] - first_distribution[box_id]) 
                             for box_id in first_distribution.keys())
            
            # 更新分布
            first_distribution = new_first_distribution
            
            print(f"迭代 {iteration+1}: 最大变化 = {max_change:.6f}")
            if max_change < convergence_threshold:
                print(f"已收敛，在第 {iteration+1} 次迭代")
                break
        
        # 再次计算最终分布下的第二选择分布和收益
        if self.second_box_pct > 0:
            final_profits = self.calculate_expected_profits(first_distribution)
            second_distributions = self.calculate_second_choice_distribution(first_distribution, final_profits)
            
            if self.third_box_pct > 0:
                third_distributions = self.calculate_third_choice_distribution(first_distribution, second_distributions, final_profits)
                
            final_profits = self.calculate_expected_profits(first_distribution, second_distributions, third_distributions)
        else:
            final_profits = self.calculate_expected_profits(first_distribution)
        
        return first_distribution, final_profits
    
    def analyze_optimal_strategy(self, profits: Dict[Union[int, str], float]) -> Tuple[Union[int, str], Tuple[Union[int, str], Union[int, str]], Tuple[Union[int, str], Union[int, str], Union[int, str]], float, float, float]:
        """
        分析最优策略
        
        参数:
            profits: 各宝箱的预期收益
            
        返回:
            (最佳单选宝箱, 最佳双选宝箱组合, 最佳三选宝箱组合, 单选收益, 双选净收益, 三选净收益)
        """
        # 最佳单选宝箱
        best_single = max(profits.items(), key=lambda x: x[1])
        
        # 最佳双选宝箱组合
        best_pair = None
        best_pair_profit = 0
        
        for i in range(len(self.treasures)):
            for j in range(i+1, len(self.treasures)):
                box1_id = self.treasures[i].id
                box2_id = self.treasures[j].id
                pair_profit = profits[box1_id] + profits[box2_id] - self.second_box_cost
                
                if pair_profit > best_pair_profit:
                    best_pair = (box1_id, box2_id)
                    best_pair_profit = pair_profit
        
        # 最佳三选宝箱组合
        best_triple = None
        best_triple_profit = 0
        
        for i in range(len(self.treasures)):
            for j in range(i+1, len(self.treasures)):
                for k in range(j+1, len(self.treasures)):
                    box1_id = self.treasures[i].id
                    box2_id = self.treasures[j].id
                    box3_id = self.treasures[k].id
                    triple_profit = profits[box1_id] + profits[box2_id] + profits[box3_id] - self.second_box_cost - self.third_box_cost
                    
                    if triple_profit > best_triple_profit:
                        best_triple = (box1_id, box2_id, box3_id)
                        best_triple_profit = triple_profit
        
        return best_single[0], best_pair, best_triple, best_single[1], best_pair_profit, best_triple_profit
    
    def calculate_second_choice_distribution(self, first_choice_distribution: Dict[Union[int, str], float], profits: Dict[Union[int, str], float]) -> Dict[Union[int, str], Dict[Union[int, str], float]]:
        """
        计算第二选择分布
        
        参数:
            first_choice_distribution: 第一选择分布
            profits: 各宝箱的预期收益
            
        返回:
            第一选择宝箱ID到第二选择分布的映射
        """
        # 初始化结果
        second_choice_distribution = {}
        
        # 对每个可能的第一选择
        for first_box_id in first_choice_distribution.keys():
            # 第二选择不能是第一选择
            available_boxes = [t.id for t in self.treasures if t.id != first_box_id]
            
            # 计算第二选择的效用
            utilities = {box_id: profits[box_id] for box_id in available_boxes}
            
            # 计算合理的第二选择概率分布
            if self.second_choice_rational_factor >= 1.0:
                # 更理性的选择 - 基于收益
                total_utility = sum(utilities.values())
                second_choice_distribution[first_box_id] = {
                    box_id: utility / total_utility for box_id, utility in utilities.items()
                }
            else:
                # 更随机的选择
                rational_part = {}
                total_utility = sum(utilities.values())
                for box_id, utility in utilities.items():
                    rational_part[box_id] = utility / total_utility
                
                # 随机部分是均匀分布
                random_part = {box_id: 1.0 / len(available_boxes) for box_id in available_boxes}
                
                # 组合
                second_choice_distribution[first_box_id] = {
                    box_id: self.second_choice_rational_factor * rational_part[box_id] + 
                           (1 - self.second_choice_rational_factor) * random_part[box_id]
                    for box_id in available_boxes
                }
        
        return second_choice_distribution
    
    def calculate_third_choice_distribution(self, first_choice_distribution: Dict[Union[int, str], float], 
                                         second_choice_distribution: Dict[Union[int, str], Dict[Union[int, str], float]],
                                         profits: Dict[Union[int, str], float]) -> Dict[Union[int, str], Dict[Union[int, str], Dict[Union[int, str], float]]]:
        """
        计算第三选择分布
        
        参数:
            first_choice_distribution: 第一选择分布
            second_choice_distribution: 第二选择分布
            profits: 各宝箱的预期收益
            
        返回:
            嵌套字典：第一选择ID -> 第二选择ID -> 第三选择ID -> 概率
        """
        # 初始化结果
        third_choice_distribution = {}
        
        # 对每个可能的第一选择和第二选择组合
        for first_box_id in first_choice_distribution.keys():
            third_choice_distribution[first_box_id] = {}
            
            if first_box_id in second_choice_distribution:
                for second_box_id in second_choice_distribution[first_box_id].keys():
                    # 第三选择不能是第一或第二选择
                    available_boxes = [t.id for t in self.treasures if t.id != first_box_id and t.id != second_box_id]
                    
                    # 计算第三选择的效用
                    utilities = {box_id: profits[box_id] for box_id in available_boxes}
                    
                    # 计算合理的第三选择概率分布
                    if self.third_choice_rational_factor >= 1.0:
                        # 更理性的选择 - 基于收益
                        total_utility = sum(utilities.values())
                        if third_choice_distribution[first_box_id].get(second_box_id) is None:
                            third_choice_distribution[first_box_id][second_box_id] = {}
                        for box_id, utility in utilities.items():
                            third_choice_distribution[first_box_id][second_box_id][box_id] = utility / total_utility
                    else:
                        # 更随机的选择
                        rational_part = {}
                        total_utility = sum(utilities.values())
                        for box_id, utility in utilities.items():
                            rational_part[box_id] = utility / total_utility
                        
                        # 随机部分是均匀分布
                        random_part = {box_id: 1.0 / len(available_boxes) for box_id in available_boxes}
                        
                        # 组合
                        if third_choice_distribution[first_box_id].get(second_box_id) is None:
                            third_choice_distribution[first_box_id][second_box_id] = {}
                        for box_id in available_boxes:
                            third_choice_distribution[first_box_id][second_box_id][box_id] = (
                                self.third_choice_rational_factor * rational_part[box_id] + 
                                (1 - self.third_choice_rational_factor) * random_part[box_id]
                            )
        
        return third_choice_distribution


def main():
    """主函数"""
    # 创建宝箱列表 (新一轮的20个箱子)
    treasures = [
        # A 组箱子
        Treasure("A1", 80, 6),
        Treasure("A2", 50, 4),
        Treasure("A3", 83, 7),
        Treasure("A4", 31, 2),
        Treasure("A5", 60, 4),
        # B 组箱子
        Treasure("B1", 89, 8),
        Treasure("B2", 10, 1),
        Treasure("B3", 37, 3),
        Treasure("B4", 70, 4),
        Treasure("B5", 90, 10),
        # C 组箱子
        Treasure("C1", 17, 1),
        Treasure("C2", 40, 3),
        Treasure("C3", 73, 4),
        Treasure("C4", 100, 15),
        Treasure("C5", 20, 2),
        # D 组箱子
        Treasure("D1", 41, 3),
        Treasure("D2", 79, 5),
        Treasure("D3", 23, 2),
        Treasure("D4", 47, 3),
        Treasure("D5", 30, 2)
    ]
    
    # 设置上一轮的箱子选择分布
    previous_selection = {
        "A1": 18.178,  # 80x, 6居民
        "A2": 8.516,   # 50x, 4居民
        "A4": 6.987,   # 31x, 2居民
        "B1": 15.184,  # 89x, 8居民
        "B2": 0.998,   # 10x, 1居民
        "B3": 5.118,   # 37x, 3居民
        "B5": 11.807,  # 90x, 10居民
        "C1": 7.539,   # 17x, 1居民
        "C3": 24.060,  # 73x, 4居民
        "C5": 1.614    # 20x, 2居民
    }
    
    # 创建优化器
    optimizer = TreasureOptimizer(
        treasures=treasures,
        num_players=4130,
        rational_pct=0.45,
        heuristic_pct=0.35,
        second_box_pct=0.15,
        second_box_cost=50000,
        third_box_pct=0.05,
        third_box_cost=100000,
        previous_selection=previous_selection
    )
    
    # 打印宝箱信息
    print("\n宝箱基本信息")
    print("-" * 70)
    print(f"{'宝箱ID':<10}{'乘数':<10}{'居民':<10}{'乘数/居民':<15}{'基础效用':<15}")
    print("-" * 70)
    for t in treasures:
        print(f"{t.id:<10}{t.multiplier:<10}{t.inhabitants:<10}{t.m_h_ratio:<15.2f}{t.base_utility:<15.2f}")
    print("-" * 70)
    
    # 运行迭代
    first_distribution, profits = optimizer.run_iteration(max_iterations=15)
    
    # 分析最优策略
    best_single, best_pair, best_triple, single_profit, pair_profit, triple_profit = optimizer.analyze_optimal_strategy(profits)
    
    # 打印结果
    print("\n最优策略分析结果:")
    print("-" * 70)
    
    print(f"\n最佳单选策略: 宝箱 {best_single}")
    for t in treasures:
        if t.id == best_single:
            print(f"- 乘数={t.multiplier}, 居民={t.inhabitants}")
            break
    print(f"预期收益: {single_profit:.2f}")
    
    print(f"\n最佳双选策略: 宝箱 {best_pair[0]} 和 宝箱 {best_pair[1]}")
    for pair_id in best_pair:
        for t in treasures:
            if t.id == pair_id:
                print(f"- 宝箱 {t.id}: 乘数={t.multiplier}, 居民={t.inhabitants}")
                break
    print(f"预期净收益(减去成本): {pair_profit:.2f}")
    
    print(f"\n最佳三选策略: 宝箱 {best_triple[0]}、宝箱 {best_triple[1]} 和 宝箱 {best_triple[2]}")
    for triple_id in best_triple:
        for t in treasures:
            if t.id == triple_id:
                print(f"- 宝箱 {t.id}: 乘数={t.multiplier}, 居民={t.inhabitants}")
                break
    print(f"预期净收益(减去成本): {triple_profit:.2f}")
    
    # 找出最优策略
    best_strategy = "single"
    best_profit = single_profit
    
    if pair_profit > best_profit:
        best_strategy = "pair"
        best_profit = pair_profit
    
    if triple_profit > best_profit:
        best_strategy = "triple"
        best_profit = triple_profit
    
    print("\n综合推荐:")
    if best_strategy == "single":
        print(f"推荐单箱子策略: 选择宝箱 {best_single}，预期收益 {single_profit:.2f}")
    elif best_strategy == "pair":
        print(f"推荐双箱子策略: 选择宝箱 {best_pair[0]} 和 宝箱 {best_pair[1]}，预期净收益 {pair_profit:.2f}")
    else:
        print(f"推荐三箱子策略: 选择宝箱 {best_triple[0]}、宝箱 {best_triple[1]} 和 宝箱 {best_triple[2]}，预期净收益 {triple_profit:.2f}")
    
    return {
        "optimizer": optimizer,
        "first_distribution": first_distribution,
        "profits": profits,
        "best_single": best_single,
        "best_pair": best_pair,
        "best_triple": best_triple,
        "single_profit": single_profit,
        "pair_profit": pair_profit,
        "triple_profit": triple_profit,
        "recommendation": best_strategy
    }

if __name__ == "__main__":
    main()
