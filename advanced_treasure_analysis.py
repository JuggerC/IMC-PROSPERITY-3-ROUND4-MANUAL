#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
IMC Prosperity Advanced Strategy Analysis for Treasure Selection (Three-Box Version)
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
# 使用支持中文显示的字体，针对MacOS系统
if sys.platform == 'darwin':  # macOS
    matplotlib.rcParams['font.sans-serif'] = ['Heiti TC', 'Arial', 'Helvetica', 'sans-serif']
else:  # Windows和Linux
    matplotlib.rcParams['font.sans-serif'] = ['SimHei', 'Microsoft YaHei', 'Arial', 'sans-serif']
matplotlib.rcParams['axes.unicode_minus'] = True

# Add current directory to module search path
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

# Import Treasure class and advanced strategy analyzer
from models.simulator import Treasure
from advanced_strategy.treasure_strategy_analyzer import TreasureStrategyAnalyzer
import shutil

def create_output_dir(output_dir=None):
    """Create output directory"""
    if output_dir is None:
        output_dir = os.path.join(os.path.dirname(__file__), "output", "advanced_analysis")
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
    return output_dir

def create_treasure_list() -> List[Treasure]:
    """Create treasure list (new round with 20 boxes)"""
    treasures = [
        # Group A boxes
        Treasure("A1", 80, 6),
        Treasure("A2", 50, 4),
        Treasure("A3", 83, 7),
        Treasure("A4", 31, 2),
        Treasure("A5", 60, 4),
        # Group B boxes
        Treasure("B1", 89, 8),
        Treasure("B2", 10, 1),
        Treasure("B3", 37, 3),
        Treasure("B4", 70, 4),
        Treasure("B5", 90, 10),
        # Group C boxes
        Treasure("C1", 17, 1),
        Treasure("C2", 40, 3),
        Treasure("C3", 73, 4),
        Treasure("C4", 100, 15),
        Treasure("C5", 20, 2),
        # Group D boxes
        Treasure("D1", 41, 3),
        Treasure("D2", 79, 5),
        Treasure("D3", 23, 2),
        Treasure("D4", 47, 3),
        Treasure("D5", 30, 2)
    ]
    return treasures

def print_treasure_info(treasures: List[Treasure]):
    """Print basic treasure information"""
    print("\nTreasure Basic Information")
    print("-" * 70)
    print(f"{'Box ID':<10}{'Multiplier':<10}{'Residents':<10}{'M/R Ratio':<15}{'Base Utility':<15}")
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
    Run advanced strategy analysis
    
    Parameters:
        output_dir: Output directory
        num_players: Total number of players
        rational_pct: Proportion of rational players
        heuristic_pct: Proportion of heuristic players
        second_box_pct: Proportion of players choosing a second box
        second_box_cost: Cost of choosing a second box
        third_box_pct: Proportion of players choosing a third box
        third_box_cost: Cost of choosing a third box
    """
    # Create treasure list
    treasures = create_treasure_list()
    
    # Print treasure information
    print_treasure_info(treasures)
    
    # Calculate random player proportion
    random_pct = 1.0 - rational_pct - heuristic_pct
    
    print(f"\nStarting advanced strategy analysis...")
    print(f"Parameter settings:")
    print(f"- Total players: {num_players}")
    print(f"- Rational player proportion: {rational_pct:.2f}")
    print(f"- Heuristic player proportion: {heuristic_pct:.2f}")
    print(f"- Random player proportion: {random_pct:.2f}")
    print(f"- Second box selection proportion: {second_box_pct:.2f}")
    print(f"- Second box cost: {second_box_cost}")
    print(f"- Third box selection proportion: {third_box_pct:.2f}")
    print(f"- Third box cost: {third_box_cost}")
    
    # Set previous round box selection distribution
    previous_selection = {
        "A1": 18.178,  # 80x, 6 residents
        "A2": 8.516,   # 50x, 4 residents
        "A4": 6.987,   # 31x, 2 residents
        "B1": 15.184,  # 89x, 8 residents
        "B2": 0.998,   # 10x, 1 resident
        "B3": 5.118,   # 37x, 3 residents
        "B5": 11.807,  # 90x, 10 residents
        "C1": 7.539,   # 17x, 1 resident
        "C3": 24.060,  # 73x, 4 residents
        "C5": 1.614    # 20x, 2 residents
    }
    
    # Create analyzer
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
    
    # Run cognitive hierarchy analysis
    print("\nPerforming cognitive hierarchy analysis...")
    cognitive_results = analyzer.analyze_with_cognitive_hierarchy()
    best_treasure = cognitive_results["best_treasure"]
    print(f"Cognitive hierarchy model optimal choice: Box {best_treasure.id} (Multiplier={best_treasure.multiplier}, Residents={best_treasure.inhabitants})")
    
    # Run behavioral economics analysis
    print("\nPerforming behavioral economics analysis...")
    behavioral_results = analyzer.analyze_with_behavioral_economics()
    best_treasure = behavioral_results["best_treasure"]
    print(f"Behavioral economics model optimal choice: Box {best_treasure.id} (Multiplier={best_treasure.multiplier}, Residents={best_treasure.inhabitants})")
    
    # Run social dynamics analysis
    print("\nPerforming social dynamics analysis...")
    social_results = analyzer.analyze_with_social_dynamics(num_iterations=50)
    best_treasure = social_results["best_treasure"]
    print(f"Social dynamics model optimal choice: Box {best_treasure.id} (Multiplier={best_treasure.multiplier}, Residents={best_treasure.inhabitants})")
    
    # Run meta-strategy analysis
    print("\nPerforming meta-strategy analysis...")
    meta_results = analyzer.analyze_with_meta_strategy()
    best_treasure = meta_results["best_treasure"]
    print(f"Meta-strategy model optimal choice: Box {best_treasure.id} (Multiplier={best_treasure.multiplier}, Residents={best_treasure.inhabitants})")
    
    # Integrate results
    print("\nIntegrating analysis results...")
    integrated_results = analyzer.integrate_results()
    best_treasure = integrated_results.get("best_treasure")
    best_pair = integrated_results.get("best_pair_treasures")
    best_triple = integrated_results.get("best_triple_treasures")
    
    print("\nIntegrated Analysis Results:")
    print("-" * 70)
    
    if best_treasure:
        print(f"Optimal single-box strategy: Box {best_treasure.id} (Multiplier={best_treasure.multiplier}, Residents={best_treasure.inhabitants})")
        print(f"Expected profit: {integrated_results['best_single_profit']:.2f}")
    else:
        print("Unable to determine optimal single-box strategy")
    
    if best_pair:
        print(f"\nOptimal two-box strategy: Box {best_pair[0].id} and Box {best_pair[1].id}")
        print(f"- Box {best_pair[0].id}: Multiplier={best_pair[0].multiplier}, Residents={best_pair[0].inhabitants}")
        print(f"- Box {best_pair[1].id}: Multiplier={best_pair[1].multiplier}, Residents={best_pair[1].inhabitants}")
        print(f"Expected net profit (after costs): {integrated_results['best_pair_profit']:.2f}")
    else:
        print("\nUnable to determine optimal two-box strategy")
    
    if best_triple:
        print(f"\nOptimal three-box strategy: Box {best_triple[0].id}, Box {best_triple[1].id} and Box {best_triple[2].id}")
        print(f"- Box {best_triple[0].id}: Multiplier={best_triple[0].multiplier}, Residents={best_triple[0].inhabitants}")
        print(f"- Box {best_triple[1].id}: Multiplier={best_triple[1].multiplier}, Residents={best_triple[1].inhabitants}")
        print(f"- Box {best_triple[2].id}: Multiplier={best_triple[2].multiplier}, Residents={best_triple[2].inhabitants}")
        print(f"Expected net profit (after costs): {integrated_results['best_triple_profit']:.2f}")
    else:
        print("\nUnable to determine optimal three-box strategy")
    
    print("\nModel consistency:")
    
    # Generate visualizations
    print("\nGenerating visualizations...")
    
    # Create treasure name list for visualization
    treasure_names = [f"{t.id}" for t in treasures]
    
    # 1. Create cognitive hierarchy model strategy distribution visualization
    cognitive_distribution = cognitive_results["overall_distribution"]
    analyzer.visualizer.plot_strategy_distribution(
        cognitive_distribution, 
        title="Cognitive Hierarchy Model Strategy Distribution", 
        strategy_names=treasure_names
    ).savefig(os.path.join(output_dir, "cognitive_distribution.png"), dpi=300)
    
    # 2. Create behavioral economics analysis visualization
    behavioral_weights = behavioral_results["final_weights"]
    analyzer.visualizer.plot_strategy_distribution(
        behavioral_weights,
        title="Behavioral Economics Model Strategy Weights",
        strategy_names=treasure_names
    ).savefig(os.path.join(output_dir, "behavioral_weights.png"), dpi=300)
    
    # 3. Create social dynamics analysis visualization
    social_distribution = social_results["final_distribution"]
    analyzer.visualizer.plot_strategy_distribution(
        social_distribution,
        title="Social Dynamics Model Final Distribution",
        strategy_names=treasure_names
    ).savefig(os.path.join(output_dir, "social_distribution.png"), dpi=300)
    
    # 4. Create social dynamics evolution plot
    plt.figure(figsize=(14, 7))
    evolution_data = social_results["evolution_data"]
    
    # Transpose for plotting
    evolution_matrix = np.array([dist for dist in evolution_data]).T
    
    for i in range(len(treasures)):
        plt.plot(range(len(evolution_data)), evolution_matrix[i], label=f"{treasures[i].id}")
    
    plt.xlabel("Iteration")
    plt.ylabel("Strategy Selection Probability")
    plt.title("Social Dynamics Strategy Evolution")
    plt.grid(True, alpha=0.3)
    plt.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, "social_dynamics_evolution.png"), dpi=300)
    plt.close()
    
    # 5. Create payoff matrix heatmap
    plt.figure(figsize=(12, 10))
    sns.heatmap(analyzer.payoff_matrix, annot=False, fmt=".0f", 
              xticklabels=[t.id for t in treasures],
              yticklabels=[t.id for t in treasures], 
              cmap="YlOrRd")
    plt.xlabel("Other Players' Strategy")
    plt.ylabel("My Strategy")
    plt.title("Payoff Matrix Heatmap")
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, "payoff_matrix.png"), dpi=300)
    plt.close()
    
    # 6. Create integrated results visualization - Best strategy comparison among models
    model_names = ["Cognitive Hierarchy", "Behavioral Economics", "Social Dynamics", "Meta-Strategy"]
    best_strategies = [
        cognitive_results["best_strategy"],
        behavioral_results["best_strategy"],
        social_results["best_strategy"],
        meta_results["best_response"]
    ]
    
    plt.figure(figsize=(10, 6))
    for i, (model, strategy) in enumerate(zip(model_names, best_strategies)):
        plt.bar(i, treasures[strategy].base_utility, label=f"{treasures[strategy].id}")
        plt.text(i, 0.2, f"{treasures[strategy].id}", ha='center')
    
    plt.xticks(range(len(model_names)), model_names, rotation=45)
    plt.ylabel("Base Utility")
    plt.title("Best Strategy Comparison Among Models")
    plt.grid(True, alpha=0.3, axis='y')
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, "model_comparison.png"), dpi=300)
    plt.close()
    
    # 7. Create expected profit comparison graph
    plt.figure(figsize=(10, 6))
    strategies = ["Single-Box Strategy", "Two-Box Strategy", "Three-Box Strategy"]
    profits = [
        integrated_results["best_single_profit"],
        integrated_results["best_pair_profit"],
        integrated_results["best_triple_profit"]
    ]
    plt.bar(strategies, profits)
    for i, p in enumerate(profits):
        plt.text(i, p/2, f"{p:.0f}", ha='center', fontsize=12)
    plt.ylabel("Expected Net Profit")
    plt.title("Expected Net Profit Comparison Among Different Strategies")
    plt.grid(True, alpha=0.3, axis='y')
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, "strategy_profit_comparison.png"), dpi=300)
    plt.close()
    
    # 8. Create estimated selection distribution graph
    strategy_names = [t.id for t in treasures]
    estimated_selection = analyzer.estimated_selection
    plt.figure(figsize=(12, 6))
    plt.bar(strategy_names, [estimated_selection.get(name, 0) for name in strategy_names])
    plt.xticks(rotation=45)
    plt.ylabel("Estimated Selection Percentage")
    plt.title("Treasure Estimated Selection Distribution")
    plt.grid(True, alpha=0.3, axis='y')
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, "estimated_selection.png"), dpi=300)
    plt.close()
    
    # Generate and save report
    print("\nGenerating analysis report...")
    report = ["# 宝箱选择高级策略分析报告\n"]
    
    # Analysis summary
    report.append("## 分析概要\n")
    report.append(f"- 分析的宝箱数量: {len(treasures)}\n")
    report.append(f"- 玩家数量: {analyzer.num_players}\n")
    report.append(f"- 理性玩家比例: {analyzer.rational_pct:.2f}\n")
    report.append(f"- 启发式玩家比例: {analyzer.heuristic_pct:.2f}\n")
    report.append(f"- 随机玩家比例: {analyzer.random_pct:.2f}\n")
    report.append(f"- 选择第二宝箱比例: {analyzer.second_box_pct:.2f}\n")
    report.append(f"- 第二宝箱成本: {analyzer.second_box_cost}\n")
    report.append(f"- 选择第三宝箱比例: {analyzer.third_box_pct:.2f}\n")
    report.append(f"- 第三宝箱成本: {analyzer.third_box_cost}\n")
    
    # Integrated analysis results
    report.append("\n## 综合分析结果\n")
    
    # Single-box strategy
    report.append("\n### 最佳单选策略\n")
    best_treasure = integrated_results.get("best_treasure")
    if best_treasure:
        report.append(f"综合所有模型的最优单选: 宝箱{best_treasure.id}\n")
        report.append(f"- 乘数: {best_treasure.multiplier}\n")
        report.append(f"- 居民: {best_treasure.inhabitants}\n")
        report.append(f"预期收益: {integrated_results['best_single_profit']:.2f}\n")
    else:
        report.append("无法确定最优单选策略。所有模型的分析结果不一致或无法产生有效的单选策略推荐。\n")
    
    # Two-box strategy
    report.append("\n### 最佳双选策略\n")
    best_pair = integrated_results.get("best_pair_treasures")
    if best_pair:
        report.append(f"综合所有模型的最优双选: 宝箱{best_pair[0].id}和宝箱{best_pair[1].id}\n")
        report.append(f"- 宝箱{best_pair[0].id}: 乘数={best_pair[0].multiplier}, 居民={best_pair[0].inhabitants}\n")
        report.append(f"- 宝箱{best_pair[1].id}: 乘数={best_pair[1].multiplier}, 居民={best_pair[1].inhabitants}\n")
        report.append(f"预期净收益: {integrated_results['best_pair_profit']:.2f}\n")
    else:
        report.append("无法确定最优双选策略。\n")
    
    # Three-box strategy
    report.append("\n### 最佳三选策略\n")
    best_triple = integrated_results.get("best_triple_treasures")
    if best_triple:
        report.append(f"综合所有模型的最优三选: 宝箱{best_triple[0].id}、宝箱{best_triple[1].id}和宝箱{best_triple[2].id}\n")
        report.append(f"- 宝箱{best_triple[0].id}: 乘数={best_triple[0].multiplier}, 居民={best_triple[0].inhabitants}\n")
        report.append(f"- 宝箱{best_triple[1].id}: 乘数={best_triple[1].multiplier}, 居民={best_triple[1].inhabitants}\n")
        report.append(f"- 宝箱{best_triple[2].id}: 乘数={best_triple[2].multiplier}, 居民={best_triple[2].inhabitants}\n")
        report.append(f"预期净收益: {integrated_results['best_triple_profit']:.2f}\n")
    else:
        report.append("无法确定最优三选策略。\n")
    
    # Model consistency
    report.append("\n### 模型一致性\n")
    model_recommendations = {
        "认知层次模型": cognitive_results["best_treasure"].id if "best_treasure" in cognitive_results else "无推荐",
        "行为经济学模型": behavioral_results["best_treasure"].id if "best_treasure" in behavioral_results else "无推荐",
        "社会动态模型": social_results["best_treasure"].id if "best_treasure" in social_results else "无推荐",
        "元策略模型": meta_results["best_treasure"].id if "best_treasure" in meta_results else "无推荐"
    }
    
    for model, rec in model_recommendations.items():
        report.append(f"- {model}: 宝箱{rec}\n")
    
    # Calculate consistency
    recommendations = list(model_recommendations.values())
    if "无推荐" in recommendations:
        consistency = "部分模型无法提供推荐"
    elif len(set(recommendations)) == 1:
        consistency = "完全一致 (100%)"
    else:
        most_common = max(set(recommendations), key=recommendations.count)
        consistency_pct = recommendations.count(most_common) / len(recommendations) * 100
        consistency = f"部分一致 ({consistency_pct:.0f}%)"
    
    report.append(f"\n模型一致性评估: {consistency}\n")
    
    # Final recommendation
    report.append("\n## 总结建议\n")
    
    strategy_profits = {
        "single": integrated_results.get("best_single_profit", float("-inf")),
        "pair": integrated_results.get("best_pair_profit", float("-inf")),
        "triple": integrated_results.get("best_triple_profit", float("-inf"))
    }
    
    best_strategy_type = max(strategy_profits.items(), key=lambda x: x[1])[0]
    
    if best_strategy_type == "single" and best_treasure:
        report.append(f"建议选择单宝箱策略: 宝箱{best_treasure.id}\n\n")
        report.append(f"预期收益: {integrated_results['best_single_profit']:.2f}\n\n")
    elif best_strategy_type == "pair" and best_pair:
        report.append(f"建议选择双宝箱策略: 宝箱{best_pair[0].id}和宝箱{best_pair[1].id}\n\n")
        report.append(f"预期净收益: {integrated_results['best_pair_profit']:.2f}\n\n")
    elif best_strategy_type == "triple" and best_triple:
        report.append(f"建议选择三宝箱策略: 宝箱{best_triple[0].id}、宝箱{best_triple[1].id}和宝箱{best_triple[2].id}\n\n")
        report.append(f"预期净收益: {integrated_results['best_triple_profit']:.2f}\n\n")
    else:
        report.append("无法提供明确的策略建议，请参考各模型分析结果进行决策。\n\n")
    
    report.append("理由: 综合考虑了各宝箱的基础属性、玩家行为模式和预期收益后，上述策略在当前条件下提供最高的预期收益。\n")
    
    # Save report
    with open(os.path.join(output_dir, "advanced_strategy_report.md"), "w") as f:
        f.write("".join(report))
    
    # Update report, add chart references
    report_path = os.path.join(output_dir, "advanced_strategy_report.md")
    with open(report_path, 'a') as f:
        f.write("\n\n## 可视化图表\n\n")
        f.write("### 认知层次模型分布\n")
        f.write("![Cognitive Hierarchy Model Strategy Distribution](cognitive_distribution.png)\n\n")
        f.write("### 行为经济学模型权重\n")
        f.write("![Behavioral Economics Model Strategy Weights](behavioral_weights.png)\n\n")
        f.write("### 社会动态模型最终分布\n")
        f.write("![Social Dynamics Model Final Distribution](social_distribution.png)\n\n")
        f.write("### 收益矩阵热图\n")
        f.write("![Payoff Matrix Heatmap](payoff_matrix.png)\n\n")
        f.write("### 社会动态演化过程\n")
        f.write("![Social Dynamics Strategy Evolution](social_dynamics_evolution.png)\n\n")
        f.write("### 各模型最佳策略比较\n")
        f.write("![Best Strategy Comparison Among Models](model_comparison.png)\n\n")
        f.write("### 不同策略预期收益比较\n")
        f.write("![Expected Net Profit Comparison Among Different Strategies](strategy_profit_comparison.png)\n\n")
        f.write("### 宝箱预估选择分布\n")
        f.write("![Treasure Estimated Selection Distribution](estimated_selection.png)\n\n")
    
    print(f"\nAnalysis completed! Report and visualization charts have been saved to {output_dir}")
    
    # Return recommendation
    strategy_profits = {
        "single": integrated_results["best_single_profit"],
        "pair": integrated_results["best_pair_profit"],
        "triple": integrated_results["best_triple_profit"]
    }
    
    best_strategy_type = max(strategy_profits.items(), key=lambda x: x[1])[0]
    
    if best_strategy_type == "single":
        if best_treasure:
            recommendation = f"Box {best_treasure.id} (Multiplier={best_treasure.multiplier}, Residents={best_treasure.inhabitants})"
        else:
            recommendation = "No optimal single box identified"
        recommendation_type = "Single-Box Strategy"
        recommendation_profit = integrated_results["best_single_profit"]
    elif best_strategy_type == "pair":
        recommendation = f"Box {best_pair[0].id} and Box {best_pair[1].id}"
        recommendation_type = "Two-Box Strategy"
        recommendation_profit = integrated_results["best_pair_profit"]
    else:
        recommendation = f"Box {best_triple[0].id}, Box {best_triple[1].id} and Box {best_triple[2].id}"
        recommendation_type = "Three-Box Strategy"
        recommendation_profit = integrated_results["best_triple_profit"]
    
    print(f"\nFinal Recommendation: {recommendation_type} - {recommendation}")
    print(f"Expected Net Profit: {recommendation_profit:.2f}")
    
    return {
        "analyzer": analyzer,
        "results": analyzer.results,
        "recommendation": recommendation,
        "recommendation_type": recommendation_type,
        "recommendation_profit": recommendation_profit
    }

def main():
    """Main function"""
    parser = argparse.ArgumentParser(description='Advanced Treasure Strategy Analysis')
    parser.add_argument('--rational', type=float, default=0.45, help='Proportion of rational players')
    parser.add_argument('--heuristic', type=float, default=0.35, help='Proportion of heuristic players')
    parser.add_argument('--second-box-pct', type=float, default=0.15, help='Proportion of players choosing a second box')
    parser.add_argument('--second-box-cost', type=int, default=50000, help='Cost of choosing a second box')
    parser.add_argument('--third-box-pct', type=float, default=0.05, help='Proportion of players choosing a third box')
    parser.add_argument('--third-box-cost', type=int, default=100000, help='Cost of choosing a third box')
    parser.add_argument('--num-players', type=int, default=4130, help='Total number of players')
    parser.add_argument('--output-dir', type=str, default='output/advanced_analysis', help='Output directory for analysis results')
    
    args = parser.parse_args()
    
    # Create output directory if it doesn't exist
    create_output_dir(args.output_dir)
    
    # Run advanced analysis
    result = run_advanced_analysis(
        output_dir=args.output_dir,
        num_players=args.num_players,
        rational_pct=args.rational,
        heuristic_pct=args.heuristic,
        second_box_pct=args.second_box_pct,
        second_box_cost=args.second_box_cost,
        third_box_pct=args.third_box_pct,
        third_box_cost=args.third_box_cost
    )
    
    return result

if __name__ == "__main__":
    main() 