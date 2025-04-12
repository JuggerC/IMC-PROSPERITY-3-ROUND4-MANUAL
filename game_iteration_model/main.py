#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
IMC Prosperity 宝箱选择优化模型 - 主分析脚本
"""

import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from typing import Dict, List, Tuple
import time
import os
import sys
import argparse

# 添加当前目录到模块搜索路径
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from simulator import Treasure, TreasureOptimizer
from scenario_analysis import ScenarioAnalyzer, MonteCarloSimulator
from reverse_game_theory import ReverseGameAnalyzer, OptimalStrategyFinder

# 导入高级策略分析模块
import advanced_treasure_analysis


def create_output_dir():
    """创建输出目录"""
    output_dir = os.path.join(os.path.dirname(__file__), "output")
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
    return output_dir


def main():
    """主函数"""
    # 解析命令行参数
    parser = argparse.ArgumentParser(description="IMC Prosperity 宝箱选择优化分析")
    parser.add_argument("--with-advanced", action="store_true",
                      help="是否使用高级策略分析")
    parser.add_argument("--rational", type=float, default=0.35,
                      help="高级分析: 理性玩家比例 (默认: 0.35)")
    parser.add_argument("--heuristic", type=float, default=0.45,
                      help="高级分析: 启发式玩家比例 (默认: 0.45)")
    parser.add_argument("--second-box-pct", type=float, default=0.05,
                      help="高级分析: 选择第二个宝箱的玩家比例 (默认: 0.05)")
    args = parser.parse_args()
    
    start_time = time.time()
    
    print("="*80)
    print("IMC Prosperity 宝箱选择优化分析")
    print("="*80)
    
    # 创建输出目录
    output_dir = create_output_dir()
    
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
    print("\n1. 宝箱基本信息")
    print("-"*50)
    for treasure in treasures:
        print(f"宝箱{treasure.id}: 乘数={treasure.multiplier}, 居民={treasure.inhabitants}, " + 
              f"M/H比={treasure.m_h_ratio:.2f}, 基础效用={treasure.base_utility:.2f}")
    
    # 1. 基准模型分析
    print("\n\n2. 基准模型分析")
    print("-"*50)
    base_optimizer = TreasureOptimizer(
        treasures=treasures,
        rational_pct=0.2,
        heuristic_pct=0.5,
        second_box_pct=0.1
    )
    
    base_distribution, base_profits = base_optimizer.run_iteration()
    
    # 打印基准分布和收益
    print("\n基准模型下的选择分布:")
    for box_id, prob in sorted(base_distribution.items()):
        print(f"宝箱{box_id}: {prob:.4f} ({prob*100:.2f}%)")
    
    print("\n基准模型下的预期收益:")
    for box_id, profit in sorted(base_profits.items(), key=lambda x: x[1], reverse=True):
        print(f"宝箱{box_id}: {profit:.2f}")
    
    # 分析最优策略
    best_single, best_pair, single_profit, pair_profit = base_optimizer.analyze_optimal_strategy(base_profits)
    
    print("\n基准模型下的最优策略:")
    print(f"单选策略: 选择宝箱{best_single}, 预期收益: {single_profit:.2f}")
    print(f"双选策略: 选择宝箱{best_pair[0]}和宝箱{best_pair[1]}, 预期净收益: {pair_profit:.2f}")
    
    # 2. 多情景分析
    print("\n\n3. 多情景分析")
    print("-"*50)
    scenario_analyzer = ScenarioAnalyzer(treasures)
    
    # 添加不同情景
    scenario_analyzer.add_scenario("基准", treasures, 0.2, 0.5, 0.1)
    scenario_analyzer.add_scenario("高理性", treasures, 0.4, 0.4, 0.1)
    scenario_analyzer.add_scenario("低理性", treasures, 0.1, 0.6, 0.1)
    scenario_analyzer.add_scenario("极端启发式", treasures, 0.05, 0.85, 0.1)
    
    # 运行情景分析
    scenario_analyzer.run_all_scenarios()
    
    # 比较情景
    profit_pivot, stability = scenario_analyzer.compare_scenarios()
    
    # 绘制情景比较图
    scenario_analyzer.plot_scenario_comparison()
    
    # 获取所有情景的收益
    scenario_profits = {name: result['profits'] for name, result in scenario_analyzer.results.items()}
    
    # 3. 蒙特卡洛模拟
    print("\n\n4. 蒙特卡洛模拟")
    print("-"*50)
    mc_simulator = MonteCarloSimulator(
        treasures=treasures,
        base_distribution=base_distribution,
        num_simulations=1000,
        distribution_variance=0.2
    )
    
    # 运行模拟
    mc_stats = mc_simulator.run_simulation()
    
    # 分析结果
    mc_results = mc_simulator.analyze_results()
    
    # 绘制结果图
    mc_simulator.plot_results()
    
    # 4. 反向博弈分析
    print("\n\n5. 反向博弈分析")
    print("-"*50)
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
    
    # 5. 最优策略查找
    print("\n\n6. 最优策略查找")
    print("-"*50)
    strategy_finder = OptimalStrategyFinder(treasures)
    
    # 添加不同情景的收益
    strategy_finder.add_profit_scenario("基准", base_profits, 0.3)
    for name, profits in scenario_profits.items():
        if name != "基准":
            strategy_finder.add_profit_scenario(name, profits, 0.1)
    
    # 添加蒙特卡洛模拟结果
    mc_profits = {box_id: stats['mean'] for box_id, stats in mc_stats.items()}
    strategy_finder.add_profit_scenario("蒙特卡洛", mc_profits, 0.2)
    
    # 添加反主流和元游戏
    reverse_profit_dict = {box_id: profit for box_id, profit, _ in reverse_profits}
    strategy_finder.add_profit_scenario("反主流", reverse_profit_dict, 0.1)
    strategy_finder.add_profit_scenario("元游戏", meta_profits, 0.2)
    
    # 查找最优策略
    results = strategy_finder.find_optimal_strategy(risk_aversion=0.5)
    
    # 打印策略推荐
    strategy_finder.print_strategy_recommendation(results)
    
    # 6. 总结报告
    print("\n\n7. 总结报告")
    print("-"*50)
    print("\n基于多方面分析，我们的最终宝箱选择建议是:")
    
    # 基于各种分析结果综合得出最终建议
    best_single_boxes = []
    best_single_boxes.append((results['best_single']['box_id'], "综合情景加权"))
    
    for name, result in scenario_analyzer.results.items():
        best_single_boxes.append((result['best_single'], name))
    
    best_pair_boxes = []
    best_pair_boxes.append((results['best_pair']['box_ids'], "综合情景加权"))
    
    for name, result in scenario_analyzer.results.items():
        best_pair_boxes.append((result['best_pair'], name))
    
    # 统计每个宝箱被选为最佳的次数
    single_counts = {}
    for box_id, source in best_single_boxes:
        if box_id not in single_counts:
            single_counts[box_id] = []
        single_counts[box_id].append(source)
    
    # 打印排名前三的最佳单选宝箱
    print("\n最佳单选宝箱排名:")
    for rank, (box_id, sources) in enumerate(sorted(single_counts.items(), 
                                                   key=lambda x: len(x[1]), reverse=True)):
        if rank < 3:
            print(f"{rank+1}. 宝箱{box_id}: 被选为最佳 {len(sources)} 次")
            print(f"   来源: {', '.join(sources)}")
            print(f"   乘数: {treasures[box_id-1].multiplier}, 居民: {treasures[box_id-1].inhabitants}")
    
    # 统计宝箱对被选为最佳的次数
    pair_counts = {}
    for pair, source in best_pair_boxes:
        pair_key = tuple(sorted(pair))
        if pair_key not in pair_counts:
            pair_counts[pair_key] = []
        pair_counts[pair_key].append(source)
    
    # 打印排名前三的最佳宝箱对
    print("\n最佳宝箱对排名:")
    for rank, (pair, sources) in enumerate(sorted(pair_counts.items(), 
                                                key=lambda x: len(x[1]), reverse=True)):
        if rank < 3:
            print(f"{rank+1}. 宝箱{pair[0]}和宝箱{pair[1]}: 被选为最佳 {len(sources)} 次")
            print(f"   来源: {', '.join(sources)}")
            print(f"   宝箱{pair[0]}: 乘数={treasures[pair[0]-1].multiplier}, " + 
                 f"居民={treasures[pair[0]-1].inhabitants}")
            print(f"   宝箱{pair[1]}: 乘数={treasures[pair[1]-1].multiplier}, " + 
                 f"居民={treasures[pair[1]-1].inhabitants}")
    
    # 最终建议
    print("\n最终建议:")
    
    if results['best_pair']['profit'] > results['best_single']['profit']:
        print(f"选择宝箱{results['best_pair']['box_ids'][0]}和宝箱{results['best_pair']['box_ids'][1]}")
        print(f"预期净收益: {results['best_pair']['profit']:.2f}")
        print("理由: 该组合在考虑多种情景和风险因素后，提供最高的风险调整收益。")
    else:
        print(f"选择宝箱{results['best_single']['box_id']}")
        print(f"预期收益: {results['best_single']['profit']:.2f}")
        print("理由: 单选策略提供了最佳的风险调整收益，超过了双选策略的净收益。")
    
    # 备选策略
    print("\n备选策略:")
    print(f"1. 低风险策略: 选择宝箱{results['mixed_strategy']['low_risk_box']['id']}")
    print(f"2. 混合风险策略: 选择宝箱{results['mixed_strategy']['box_ids'][0]}和宝箱{results['mixed_strategy']['box_ids'][1]}")
    print(f"3. 最小后悔策略: 选择宝箱{results['min_regret_strategy']['box_id']}")
    
    # 7. 高级策略分析（如果启用）
    if args.with_advanced:
        print("\n\n8. 高级策略分析")
        print("-"*50)
        print("运行高级策略模块分析...")
        
        # 设置高级分析的输出目录
        advanced_output_dir = os.path.join(os.path.dirname(output_dir), "advanced_analysis")
        
        # 运行高级策略分析
        advanced_results = advanced_treasure_analysis.run_advanced_analysis(
            output_dir=advanced_output_dir,
            rational_pct=args.rational,
            heuristic_pct=args.heuristic,
            second_box_pct=args.second_box_pct,
            second_box_cost=50000
        )
        
        # 整合高级分析结果到最终建议
        print("\n高级策略分析结论整合:")
        print("-"*50)
        if advanced_results["recommendation_type"] == "单选策略":
            print(f"高级策略分析推荐: {advanced_results['recommendation']}")
            print(f"与基本分析推荐比较: {'相同' if results['best_single']['box_id'] == int(advanced_results['recommendation'].split()[0].replace('宝箱', '')) else '不同'}")
            print("\n整合后的最终建议:")
            print(f"选择 {advanced_results['recommendation']}")
        else:
            pair = advanced_results['recommendation'].split("和")
            box1 = int(pair[0].replace("宝箱", ""))
            box2 = int(pair[1].replace("宝箱", ""))
            
            basic_pair = set(results['best_pair']['box_ids'])
            advanced_pair = {box1, box2}
            
            print(f"高级策略分析推荐: {advanced_results['recommendation']}")
            print(f"与基本分析推荐比较: {'相同' if basic_pair == advanced_pair else '不同'}")
            print("\n整合后的最终建议:")
            print(f"选择 {advanced_results['recommendation']}")
    
    # 打印运行时间
    end_time = time.time()
    print(f"\n分析完成，总运行时间: {end_time - start_time:.2f} 秒")


if __name__ == "__main__":
    main() 