#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
IMC Prosperity 宝箱选择的高级策略分析 (三箱子版本)
"""

import os
import sys
import argparse
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from typing import List, Dict, Any
from pathlib import Path
import math
import seaborn as sns
import json
import random
import time
import matplotlib
matplotlib.rcParams['font.sans-serif'] = ['SimHei']  # 用来正常显示中文标签
matplotlib.rcParams['axes.unicode_minus'] = False  # 用来正常显示负号

# 添加当前目录到模块搜索路径
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

# 导入宝箱类和高级策略分析器
from models.simulator import Treasure
from advanced_strategy.treasure_strategy_analyzer import TreasureStrategyAnalyzer
import shutil

def create_output_dir():
    """创建输出目录"""
    output_dir = os.path.join(os.path.dirname(__file__), "output", "advanced_analysis")
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
    return output_dir

def create_treasure_list() -> List[Treasure]:
    """创建宝箱列表 (新一轮的20个箱子)"""
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
    return treasures

def print_treasure_info(treasures: List[Treasure]):
    """打印宝箱基本信息"""
    print("\n宝箱基本信息")
    print("-" * 70)
    print(f"{'宝箱ID':<10}{'乘数':<10}{'居民':<10}{'乘数/居民':<15}{'基础效用':<15}")
    print("-" * 70)
    for t in treasures:
        print(f"{t.id:<10}{t.multiplier:<10}{t.inhabitants:<10}{t.m_h_ratio:<15.2f}{t.base_utility:<15.2f}")
    print("-" * 70)

def run_advanced_analysis(output_dir: str, 
                         num_players: int,
                         rational_pct: float, 
                         heuristic_pct: float, 
                         second_box_pct: float,
                         second_box_cost: int,
                         third_box_pct: float,
                         third_box_cost: int):
    """
    运行高级策略分析
    
    参数:
        output_dir: 输出目录
        num_players: 玩家总数
        rational_pct: 理性玩家比例
        heuristic_pct: 启发式玩家比例
        second_box_pct: 选择第二个宝箱的玩家比例
        second_box_cost: 选择第二个宝箱的成本
        third_box_pct: 选择第三个宝箱的玩家比例
        third_box_cost: 选择第三个宝箱的成本
    """
    # 创建宝箱列表
    treasures = create_treasure_list()
    
    # 打印宝箱信息
    print_treasure_info(treasures)
    
    # 计算随机玩家比例
    random_pct = 1.0 - rational_pct - heuristic_pct
    
    print(f"\n开始高级策略分析...")
    print(f"参数设置:")
    print(f"- 玩家总数: {num_players}")
    print(f"- 理性玩家比例: {rational_pct:.2f}")
    print(f"- 启发式玩家比例: {heuristic_pct:.2f}")
    print(f"- 随机玩家比例: {random_pct:.2f}")
    print(f"- 选择第二宝箱比例: {second_box_pct:.2f}")
    print(f"- 第二宝箱成本: {second_box_cost}")
    print(f"- 选择第三宝箱比例: {third_box_pct:.2f}")
    print(f"- 第三宝箱成本: {third_box_cost}")
    
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
    
    # 创建分析器
    analyzer = TreasureStrategyAnalyzer(
        treasures=treasures,
        num_players=num_players,
        rational_pct=rational_pct,
        heuristic_pct=heuristic_pct,
        random_pct=random_pct,
        second_box_pct=second_box_pct,
        second_box_cost=second_box_cost,
        third_box_pct=third_box_pct,
        third_box_cost=third_box_cost,
        previous_selection=previous_selection
    )
    
    # 运行认知层次分析
    print("\n正在进行认知层次分析...")
    cognitive_results = analyzer.analyze_with_cognitive_hierarchy()
    best_treasure = cognitive_results["best_treasure"]
    print(f"认知层次模型最优选择: 宝箱{best_treasure.id} (乘数={best_treasure.multiplier}, 居民={best_treasure.inhabitants})")
    
    # 运行行为经济学分析
    print("\n正在进行行为经济学分析...")
    behavioral_results = analyzer.analyze_with_behavioral_economics()
    best_treasure = behavioral_results["best_treasure"]
    print(f"行为经济学模型最优选择: 宝箱{best_treasure.id} (乘数={best_treasure.multiplier}, 居民={best_treasure.inhabitants})")
    
    # 运行社会动态分析
    print("\n正在进行社会动态分析...")
    social_results = analyzer.analyze_with_social_dynamics(num_iterations=50)
    best_treasure = social_results["best_treasure"]
    print(f"社会动态模型最优选择: 宝箱{best_treasure.id} (乘数={best_treasure.multiplier}, 居民={best_treasure.inhabitants})")
    
    # 运行元策略分析
    print("\n正在进行元策略分析...")
    meta_results = analyzer.analyze_with_meta_strategy()
    best_treasure = meta_results["best_treasure"]
    print(f"元策略模型最优选择: 宝箱{best_treasure.id} (乘数={best_treasure.multiplier}, 居民={best_treasure.inhabitants})")
    
    # 整合结果
    print("\n正在整合分析结果...")
    integrated_results = analyzer.integrate_results()
    best_treasure = integrated_results.get("best_treasure")
    best_pair = integrated_results.get("best_pair_treasures")
    best_triple = integrated_results.get("best_triple_treasures")
    
    print("\n综合分析结果:")
    print("-" * 70)
    
    if best_treasure:
        print(f"最优单选策略: 宝箱{best_treasure.id} (乘数={best_treasure.multiplier}, 居民={best_treasure.inhabitants})")
        print(f"预期收益: {integrated_results['best_single_profit']:.2f}")
    else:
        print("无法确定最优单选策略")
    
    if best_pair:
        print(f"\n最优双选策略: 宝箱{best_pair[0].id}和宝箱{best_pair[1].id}")
        print(f"- 宝箱{best_pair[0].id}: 乘数={best_pair[0].multiplier}, 居民={best_pair[0].inhabitants}")
        print(f"- 宝箱{best_pair[1].id}: 乘数={best_pair[1].multiplier}, 居民={best_pair[1].inhabitants}")
        print(f"预期净收益(减去成本): {integrated_results['best_pair_profit']:.2f}")
    else:
        print("\n无法确定最优双选策略")
    
    if best_triple:
        print(f"\n最优三选策略: 宝箱{best_triple[0].id}、宝箱{best_triple[1].id}和宝箱{best_triple[2].id}")
        print(f"- 宝箱{best_triple[0].id}: 乘数={best_triple[0].multiplier}, 居民={best_triple[0].inhabitants}")
        print(f"- 宝箱{best_triple[1].id}: 乘数={best_triple[1].multiplier}, 居民={best_triple[1].inhabitants}")
        print(f"- 宝箱{best_triple[2].id}: 乘数={best_triple[2].multiplier}, 居民={best_triple[2].inhabitants}")
        print(f"预期净收益(减去成本): {integrated_results['best_triple_profit']:.2f}")
    else:
        print("\n无法确定最优三选策略")
    
    # 模型一致性
    print("\n模型一致性:")
    for strategy, models in integrated_results["model_agreement"].items():
        treasure = treasures[strategy]
        print(f"宝箱{treasure.id} (乘数={treasure.multiplier}, 居民={treasure.inhabitants})被以下模型选为最佳: {', '.join(models)}")
    
    # 生成可视化内容
    print("\n生成可视化内容...")
    
    # 创建宝箱名称列表，用于可视化
    treasure_names = [f"Box {t.id}" for t in treasures]
    
    # 1. 创建认知层次模型的策略分布可视化
    cognitive_distribution = cognitive_results["overall_distribution"]
    analyzer.visualizer.plot_strategy_distribution(
        cognitive_distribution, 
        title="认知层次模型分布", 
        strategy_names=treasure_names
    ).savefig(os.path.join(output_dir, "cognitive_distribution.png"), dpi=300)
    
    # 2. 创建行为经济学分析可视化
    behavioral_weights = behavioral_results["final_weights"]
    analyzer.visualizer.plot_strategy_distribution(
        behavioral_weights,
        title="行为经济学模型权重",
        strategy_names=treasure_names
    ).savefig(os.path.join(output_dir, "behavioral_weights.png"), dpi=300)
    
    # 3. 创建社会动态分析可视化
    social_distribution = social_results["final_distribution"]
    analyzer.visualizer.plot_strategy_distribution(
        social_distribution,
        title="社会动态模型最终分布",
        strategy_names=treasure_names
    ).savefig(os.path.join(output_dir, "social_distribution.png"), dpi=300)
    
    # 4. 创建收益矩阵热图
    analyzer.visualizer.plot_payoff_matrix(
        analyzer.payoff_matrix,
        strategy_names=treasure_names
    ).savefig(os.path.join(output_dir, "payoff_matrix.png"), dpi=300)
    
    # 5. 创建社会动态演化图
    # 准备社会动态演化数据
    evolution_data = np.array(social_results["evolution_data"])
    plt.figure(figsize=(12, 8))
    for i in range(len(treasures)):
        plt.plot(range(len(evolution_data)), evolution_data[:, i], 
                 label=f"Box {treasures[i].id}")
    plt.xlabel("迭代次数")
    plt.ylabel("选择概率")
    plt.title("社会动态演化过程")
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, "social_dynamics_evolution.png"), dpi=300)
    plt.close()
    
    # 6. 创建整合结果可视化 - 各模型最佳策略比较
    model_names = ["认知层次", "行为经济学", "社会动态", "元策略"]
    best_strategies = [
        cognitive_results["best_strategy"],
        behavioral_results["best_strategy"],
        social_results["best_strategy"],
        meta_results["best_response"]
    ]
    
    plt.figure(figsize=(10, 6))
    for i, (model, strategy) in enumerate(zip(model_names, best_strategies)):
        plt.bar(i, treasures[strategy].base_utility, label=f"宝箱 {treasures[strategy].id}")
        plt.text(i, 0.2, f"宝箱 {treasures[strategy].id}", ha='center')
    
    plt.xticks(range(len(model_names)), model_names, rotation=45)
    plt.ylabel("基础效用")
    plt.title("各模型最佳策略比较")
    plt.grid(True, alpha=0.3, axis='y')
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, "model_comparison.png"), dpi=300)
    plt.close()
    
    # 7. 创建预期收益比较图
    plt.figure(figsize=(10, 6))
    strategies = ["单选策略", "双选策略", "三选策略"]
    profits = [
        integrated_results["best_single_profit"],
        integrated_results["best_pair_profit"],
        integrated_results["best_triple_profit"]
    ]
    plt.bar(strategies, profits)
    for i, p in enumerate(profits):
        plt.text(i, p/2, f"{p:.0f}", ha='center', fontsize=12)
    plt.ylabel("预期净收益")
    plt.title("不同策略预期收益比较")
    plt.grid(True, alpha=0.3, axis='y')
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, "strategy_profit_comparison.png"), dpi=300)
    plt.close()
    
    # 8. 创建预估选择分布图
    strategy_names = [t.id for t in treasures]
    estimated_selection = analyzer.estimated_selection
    plt.figure(figsize=(12, 6))
    plt.bar(strategy_names, [estimated_selection.get(name, 0) for name in strategy_names])
    plt.xticks(rotation=45)
    plt.ylabel("预估选择百分比")
    plt.title("宝箱预估选择分布")
    plt.grid(True, alpha=0.3, axis='y')
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, "estimated_selection.png"), dpi=300)
    plt.close()
    
    # 生成和保存报告
    print("\n生成分析报告...")
    report = analyzer.generate_report(output_dir)
    
    # 更新报告，添加图表引用
    report_path = os.path.join(output_dir, "advanced_strategy_report.md")
    with open(report_path, 'a') as f:
        f.write("\n\n## 可视化图表\n\n")
        f.write("### 认知层次模型分布\n")
        f.write("![认知层次模型分布](cognitive_distribution.png)\n\n")
        f.write("### 行为经济学模型权重\n")
        f.write("![行为经济学模型权重](behavioral_weights.png)\n\n")
        f.write("### 社会动态模型最终分布\n")
        f.write("![社会动态模型最终分布](social_distribution.png)\n\n")
        f.write("### 收益矩阵热图\n")
        f.write("![收益矩阵热图](payoff_matrix.png)\n\n")
        f.write("### 社会动态演化过程\n")
        f.write("![社会动态演化过程](social_dynamics_evolution.png)\n\n")
        f.write("### 各模型最佳策略比较\n")
        f.write("![各模型最佳策略比较](model_comparison.png)\n\n")
        f.write("### 不同策略预期收益比较\n")
        f.write("![不同策略预期收益比较](strategy_profit_comparison.png)\n\n")
        f.write("### 宝箱预估选择分布\n")
        f.write("![宝箱预估选择分布](estimated_selection.png)\n\n")
    
    print(f"\n分析完成! 报告和可视化图表已保存到 {output_dir}")
    
    # 返回建议
    strategy_profits = {
        "single": integrated_results["best_single_profit"],
        "pair": integrated_results["best_pair_profit"],
        "triple": integrated_results["best_triple_profit"]
    }
    
    best_strategy_type = max(strategy_profits.items(), key=lambda x: x[1])[0]
    
    if best_strategy_type == "single":
        recommendation = f"宝箱{best_treasure.id} (乘数={best_treasure.multiplier}, 居民={best_treasure.inhabitants})"
        recommendation_type = "单选策略"
        recommendation_profit = integrated_results["best_single_profit"]
    elif best_strategy_type == "pair":
        recommendation = f"宝箱{best_pair[0].id}和宝箱{best_pair[1].id}"
        recommendation_type = "双选策略"
        recommendation_profit = integrated_results["best_pair_profit"]
    else:
        recommendation = f"宝箱{best_triple[0].id}、宝箱{best_triple[1].id}和宝箱{best_triple[2].id}"
        recommendation_type = "三选策略"
        recommendation_profit = integrated_results["best_triple_profit"]
    
    print(f"\n最终建议: {recommendation_type} - {recommendation}")
    print(f"预期净收益: {recommendation_profit:.2f}")
    
    return {
        "analyzer": analyzer,
        "results": analyzer.results,
        "recommendation": recommendation,
        "recommendation_type": recommendation_type,
        "recommendation_profit": recommendation_profit
    }

def main():
    """主函数"""
    parser = argparse.ArgumentParser(description='高级宝箱策略分析')
    parser.add_argument('--rational', type=float, default=0.45, help='理性玩家比例')
    parser.add_argument('--heuristic', type=float, default=0.35, help='启发式玩家比例')
    parser.add_argument('--second-box-pct', type=float, default=0.15, help='选择第二个宝箱的玩家比例')
    parser.add_argument('--second-box-cost', type=int, default=50000, help='选择第二个宝箱的成本')
    parser.add_argument('--third-box-pct', type=float, default=0.05, help='选择第三个宝箱的玩家比例')
    parser.add_argument('--third-box-cost', type=int, default=100000, help='选择第三个宝箱的成本')
    parser.add_argument('--num-players', type=int, default=4130, help='玩家总数')
    parser.add_argument('--output-dir', type=str, default=None, help='输出目录')
    
    args = parser.parse_args()
    
    # 创建输出目录
    output_dir = args.output_dir if args.output_dir else create_output_dir()
    
    # 运行高级策略分析
    results = run_advanced_analysis(
        output_dir=output_dir,
        num_players=args.num_players,
        rational_pct=args.rational,
        heuristic_pct=args.heuristic,
        second_box_pct=args.second_box_pct,
        second_box_cost=args.second_box_cost,
        third_box_pct=args.third_box_pct,
        third_box_cost=args.third_box_cost
    )
    
    # 返回结果
    return results

if __name__ == "__main__":
    main() 