#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
IMC Prosperity 宝箱选择优化模型
"""

import numpy as np
import matplotlib.pyplot as plt
from collections import defaultdict
import pandas as pd
from typing import Dict, List, Tuple

class Treasure:
    """宝箱类，表示一个可选择的宝箱"""
    
    def __init__(self, box_id: int, multiplier: int, inhabitants: int):
        """
        初始化宝箱
        
        参数:
            box_id: 宝箱编号
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
    
    def __init__(self, treasures: List[Treasure], num_players: int = 10000, 
                 rational_pct: float = 0.2, heuristic_pct: float = 0.5,
                 second_box_pct: float = 0.1, second_box_cost: int = 50000,
                 second_choice_rational_factor: float = 1.0):
        """
        初始化优化器
        
        参数:
            treasures: 宝箱列表
            num_players: 玩家总数
            rational_pct: 理性玩家比例
            heuristic_pct: 启发式玩家比例
            second_box_pct: 选择第二个宝箱的玩家比例
            second_box_cost: 选择第二个宝箱的成本
            second_choice_rational_factor: 第二次选择的理性调整因子，大于1表示更理性，小于1表示更依赖启发式
        """
        self.treasures = treasures
        self.num_players = num_players
        self.rational_pct = rational_pct
        self.heuristic_pct = heuristic_pct
        self.random_pct = 1.0 - rational_pct - heuristic_pct
        self.second_box_pct = second_box_pct
        self.second_box_cost = second_box_cost
        self.second_choice_rational_factor = second_choice_rational_factor
        
        # 计算总选择次数
        self.total_selections = num_players * (1 + second_box_pct)
        
        # 初始分布
        self._calculate_initial_distribution()
        
    def _calculate_initial_distribution(self):
        """计算初始选择分布"""
        # 计算总效用
        total_utility = sum(t.base_utility for t in self.treasures)
        
        # 计算初始分布
        for treasure in self.treasures:
            treasure.initial_probability = treasure.base_utility / total_utility
            
        # 打印初始分布
        print("初始选择分布:")
        for treasure in self.treasures:
            print(f"宝箱{treasure.id}: {treasure.initial_probability:.4f} " + 
                  f"({treasure.initial_probability * 100:.2f}%)")
    
    def calculate_expected_profits(self, first_distribution: Dict[int, float], 
                                 second_distributions: Dict[int, Dict[int, float]] = None) -> Dict[int, float]:
        """
        计算给定分布下的预期收益
        
        参数:
            first_distribution: 宝箱ID到第一选择概率的映射
            second_distributions: 第一选择宝箱ID到第二选择分布的映射，如果为None则不考虑第二选择
            
        返回:
            宝箱ID到预期收益的映射
        """
        profits = {}
        
        if second_distributions is None or self.second_box_pct == 0:
            # 如果没有第二选择分布或第二选择比例为0，只考虑第一选择
            for treasure in self.treasures:
                box_id = treasure.id
                percentage = first_distribution.get(box_id, 0) * 100  # 转换为百分比
                profit = treasure.calculate_profit(percentage)
                profits[box_id] = profit
        else:
            # 考虑第一选择和第二选择
            
            # 计算每个宝箱被选为第二选择的总概率
            second_choice_total = {box_id: 0.0 for box_id in [t.id for t in self.treasures]}
            
            for first_box_id, first_prob in first_distribution.items():
                if first_box_id in second_distributions:
                    for second_box_id, second_prob in second_distributions[first_box_id].items():
                        second_choice_total[second_box_id] += first_prob * second_prob
            
            # 计算每个宝箱的总选择百分比
            for treasure in self.treasures:
                box_id = treasure.id
                first_choice_pct = first_distribution.get(box_id, 0)
                second_choice_pct = second_choice_total.get(box_id, 0) * self.second_box_pct
                
                # 总选择百分比
                total_percentage = (first_choice_pct + second_choice_pct) * 100
                
                # 计算收益
                profit = treasure.calculate_profit(total_percentage)
                profits[box_id] = profit
        
        return profits
    
    def run_iteration(self, max_iterations: int = 10, convergence_threshold: float = 0.01) -> Tuple[Dict[int, float], Dict[int, float]]:
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
        
        for iteration in range(max_iterations):
            # 如果有第二选择，计算第二选择分布
            if self.second_box_pct > 0:
                # 使用当前利润计算第二选择分布
                current_profits = self.calculate_expected_profits(first_distribution)
                second_distributions = self.calculate_second_choice_distribution(first_distribution, current_profits)
            
            # 考虑第一和第二选择计算收益
            profits = self.calculate_expected_profits(first_distribution, second_distributions)
            
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
            final_profits = self.calculate_expected_profits(first_distribution, second_distributions)
        else:
            final_profits = self.calculate_expected_profits(first_distribution)
        
        return first_distribution, final_profits
    
    def analyze_optimal_strategy(self, profits: Dict[int, float]) -> Tuple[int, Tuple[int, int], float, float]:
        """
        分析最优策略
        
        参数:
            profits: 各宝箱的预期收益
            
        返回:
            (最佳单选宝箱, 最佳双选宝箱组合, 单选收益, 双选净收益)
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
                    best_pair_profit = pair_profit
                    best_pair = (box1_id, box2_id)
        
        return best_single[0], best_pair, best_single[1], best_pair_profit

    def calculate_second_choice_distribution(self, first_choice_distribution: Dict[int, float], profits: Dict[int, float]) -> Dict[int, Dict[int, float]]:
        """
        计算给定第一选择分布下的第二选择分布
        
        参数:
            first_choice_distribution: 第一选择的分布
            profits: 各宝箱的预期收益
            
        返回:
            字典，键为第一选择的宝箱ID，值为相应的第二选择分布
        """
        # 为每个可能的第一选择计算第二选择的分布
        second_choice_distributions = {}
        
        # 调整理性、启发式和随机玩家的比例
        adjusted_rational_pct = min(1.0, self.rational_pct * self.second_choice_rational_factor)
        remaining_pct = 1.0 - adjusted_rational_pct
        
        if self.heuristic_pct + self.random_pct > 0:
            # 保持启发式和随机玩家的相对比例
            ratio = self.heuristic_pct / (self.heuristic_pct + self.random_pct)
            adjusted_heuristic_pct = remaining_pct * ratio
            adjusted_random_pct = remaining_pct * (1 - ratio)
        else:
            adjusted_heuristic_pct = 0
            adjusted_random_pct = remaining_pct
        
        # 获取总收益用于归一化
        total_profit = sum(profits.values())
        
        # 对每个可能的第一选择
        for first_box_id in first_choice_distribution.keys():
            # 初始化每个第一选择对应的第二选择分布
            second_distribution = {}
            
            # 基于收益的理性分布（排除已选宝箱）
            rational_distribution = {}
            excluded_profit_sum = total_profit - profits.get(first_box_id, 0)
            
            if excluded_profit_sum > 0:
                for box_id, profit in profits.items():
                    if box_id != first_box_id:  # 排除第一选择
                        rational_distribution[box_id] = profit / excluded_profit_sum
            
            # 基于初始分布的启发式分布（排除已选宝箱）
            heuristic_distribution = {}
            excluded_init_sum = sum(t.initial_probability for t in self.treasures if t.id != first_box_id)
            
            if excluded_init_sum > 0:
                for treasure in self.treasures:
                    if treasure.id != first_box_id:  # 排除第一选择
                        heuristic_distribution[treasure.id] = treasure.initial_probability / excluded_init_sum
            
            # 均匀分布的随机分布（排除已选宝箱）
            random_distribution = {}
            excluded_count = len(self.treasures) - 1  # 排除第一选择
            
            if excluded_count > 0:
                for treasure in self.treasures:
                    if treasure.id != first_box_id:  # 排除第一选择
                        random_distribution[treasure.id] = 1.0 / excluded_count
            
            # 合并三种分布
            for box_id in [t.id for t in self.treasures if t.id != first_box_id]:
                second_distribution[box_id] = (
                    adjusted_rational_pct * rational_distribution.get(box_id, 0) +
                    adjusted_heuristic_pct * heuristic_distribution.get(box_id, 0) +
                    adjusted_random_pct * random_distribution.get(box_id, 0)
                )
            
            # 保存这个第一选择对应的第二选择分布
            second_choice_distributions[first_box_id] = second_distribution
        
        return second_choice_distributions


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
    
    # 打印宝箱基本信息
    print("宝箱基本信息:")
    for treasure in treasures:
        print(f"宝箱{treasure.id}: 乘数={treasure.multiplier}, 居民={treasure.inhabitants}, " + 
              f"M/H比={treasure.m_h_ratio:.2f}, 基础效用={treasure.base_utility:.2f}")
    
    # 创建优化器
    optimizer = TreasureOptimizer(
        treasures=treasures,
        num_players=10000,
        rational_pct=0.35,
        heuristic_pct=0.45,
        second_box_pct=0.05,
        second_box_cost=50000,
        second_choice_rational_factor=0.7
    )
    
    # 运行迭代
    print("\n开始迭代优化...")
    final_distribution, final_profits = optimizer.run_iteration(max_iterations=10)
    
    # 打印最终分布和收益
    print("\n最终选择分布:")
    for box_id, prob in sorted(final_distribution.items()):
        print(f"宝箱{box_id}: {prob:.4f} ({prob * 100:.2f}%)")
    
    print("\n预期收益:")
    for box_id, profit in sorted(final_profits.items()):
        print(f"宝箱{box_id}: {profit:.2f}")
    
    # 分析最优策略
    best_single, best_pair, single_profit, pair_profit = optimizer.analyze_optimal_strategy(final_profits)
    
    print("\n最优策略:")
    print(f"单选策略: 选择宝箱{best_single}, 预期收益: {single_profit:.2f}")
    print(f"双选策略: 选择宝箱{best_pair[0]}和宝箱{best_pair[1]}, 预期净收益: {pair_profit:.2f} " + 
          f"(减去成本{optimizer.second_box_cost})")
    
    # 策略建议
    if pair_profit > single_profit:
        print(f"\n推荐策略: 选择宝箱{best_pair[0]}和宝箱{best_pair[1]}, 预期净收益: {pair_profit:.2f}")
    else:
        print(f"\n推荐策略: 选择宝箱{best_single}, 预期收益: {single_profit:.2f}")


if __name__ == "__main__":
    main()
