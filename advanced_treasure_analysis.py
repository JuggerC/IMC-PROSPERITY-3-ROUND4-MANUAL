#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
IMC Prosperity 宝箱选择的高级策略分析
"""

import os
import sys
import argparse
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from typing import List, Dict, Any

# 添加当前目录到模块搜索路径
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

# 导入宝箱类和高级策略分析器
from game_iteration_model.simulator import Treasure
from advanced_strategy.treasure_strategy_analyzer import TreasureStrategyAnalyzer

def create_output_dir():
    """创建输出目录"""
    output_dir = os.path.join(os.path.dirname(__file__), "output", "advanced_analysis")
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
    return output_dir

def create_treasure_list() -> List[Treasure]:
    """创建宝箱列表"""
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
    return treasures

def print_treasure_info(treasures: List[Treasure]):
    """打印宝箱基本信息"""
    print("\n宝箱基本信息")
    print("-" * 60)
    print(f"{'宝箱ID':<10}{'乘数':<10}{'居民':<10}{'乘数/居民':<15}{'基础效用':<15}")
    print("-" * 60)
    for t in treasures:
        print(f"{t.id:<10}{t.multiplier:<10}{t.inhabitants:<10}{t.m_h_ratio:<15.2f}{t.base_utility:<15.2f}")
    print("-" * 60)

def run_advanced_analysis(output_dir: str, rational_pct: float, 
                        heuristic_pct: float, second_box_pct: float,
                        second_box_cost: int):
    """
    运行高级策略分析
    
    参数:
        output_dir: 输出目录
        rational_pct: 理性玩家比例
        heuristic_pct: 启发式玩家比例
        second_box_pct: 选择第二个宝箱的玩家比例
        second_box_cost: 选择第二个宝箱的成本
    """
    # 创建宝箱列表
    treasures = create_treasure_list()
    
    # 打印宝箱信息
    print_treasure_info(treasures)
    
    # 计算随机玩家比例
    random_pct = 1.0 - rational_pct - heuristic_pct
    
    print(f"\n开始高级策略分析...")
    print(f"参数设置:")
    print(f"- 理性玩家比例: {rational_pct:.2f}")
    print(f"- 启发式玩家比例: {heuristic_pct:.2f}")
    print(f"- 随机玩家比例: {random_pct:.2f}")
    print(f"- 选择第二宝箱比例: {second_box_pct:.2f}")
    print(f"- a第二宝箱成本: {second_box_cost}")
    
    # 创建分析器
    analyzer = TreasureStrategyAnalyzer(
        treasures=treasures,
        rational_pct=rational_pct,
        heuristic_pct=heuristic_pct,
        random_pct=random_pct,
        second_box_pct=second_box_pct,
        second_box_cost=second_box_cost
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
    best_treasure = integrated_results["best_treasure"]
    best_pair = integrated_results["best_pair_treasures"]
    
    print("\n综合分析结果:")
    print("-" * 60)
    print(f"最优单选策略: 宝箱{best_treasure.id} (乘数={best_treasure.multiplier}, 居民={best_treasure.inhabitants})")
    print(f"最优双选策略: 宝箱{best_pair[0].id}和宝箱{best_pair[1].id}")
    print(f"- 宝箱{best_pair[0].id}: 乘数={best_pair[0].multiplier}, 居民={best_pair[0].inhabitants}")
    print(f"- 宝箱{best_pair[1].id}: 乘数={best_pair[1].multiplier}, 居民={best_pair[1].inhabitants}")
    
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
        title="Cognitive Hierarchy Model Distribution", 
        strategy_names=treasure_names
    ).savefig(os.path.join(output_dir, "cognitive_distribution.png"), dpi=300)
    
    # 2. 创建行为经济学分析可视化
    behavioral_weights = behavioral_results["final_weights"]
    analyzer.visualizer.plot_strategy_distribution(
        behavioral_weights,
        title="Behavioral Economics Model Weights",
        strategy_names=treasure_names
    ).savefig(os.path.join(output_dir, "behavioral_weights.png"), dpi=300)
    
    # 3. 创建社会动态分析可视化
    social_distribution = social_results["final_distribution"]
    analyzer.visualizer.plot_strategy_distribution(
        social_distribution,
        title="Social Dynamics Model Final Distribution",
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
    plt.xlabel("Iterations")
    plt.ylabel("Selection Probability")
    plt.title("Social Dynamics Evolution Process")
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, "social_dynamics_evolution.png"), dpi=300)
    plt.close()
    
    # 6. 创建整合结果可视化 - 各模型最佳策略比较
    model_names = ["Cognitive", "Behavioral", "Social", "Meta"]
    best_strategies = [
        cognitive_results["best_strategy"],
        behavioral_results["best_strategy"],
        social_results["best_strategy"],
        meta_results["best_response"]
    ]
    
    plt.figure(figsize=(10, 6))
    for i, (model, strategy) in enumerate(zip(model_names, best_strategies)):
        plt.bar(i, treasures[strategy].base_utility, label=f"Box {treasures[strategy].id}")
        plt.text(i, 0.2, f"Box {treasures[strategy].id}", ha='center')
    
    plt.xticks(range(len(model_names)), model_names, rotation=45)
    plt.ylabel("Base Utility")
    plt.title("Best Strategy Comparison Across Models")
    plt.grid(True, alpha=0.3, axis='y')
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, "model_comparison.png"), dpi=300)
    plt.close()
    
    # 生成和保存报告
    print("\n生成分析报告...")
    report = analyzer.generate_report(output_dir)
    
    # 更新报告，添加图表引用
    report_path = os.path.join(output_dir, "advanced_strategy_report.md")
    with open(report_path, 'a') as f:
        f.write("\n\n## Visualization Charts\n\n")
        f.write("### Cognitive Hierarchy Model Distribution\n")
        f.write("![Cognitive Hierarchy Model Distribution](cognitive_distribution.png)\n\n")
        f.write("### Behavioral Economics Model Weights\n")
        f.write("![Behavioral Economics Model Weights](behavioral_weights.png)\n\n")
        f.write("### Social Dynamics Model Final Distribution\n")
        f.write("![Social Dynamics Model Final Distribution](social_distribution.png)\n\n")
        f.write("### Payoff Matrix Heatmap\n")
        f.write("![Payoff Matrix Heatmap](payoff_matrix.png)\n\n")
        f.write("### Social Dynamics Evolution Process\n")
        f.write("![Social Dynamics Evolution Process](social_dynamics_evolution.png)\n\n")
        f.write("### Best Strategy Comparison Across Models\n")
        f.write("![Best Strategy Comparison Across Models](model_comparison.png)\n\n")
    
    print(f"\n分析完成! 报告和可视化图表已保存到 {output_dir}")
    
    # 返回建议
    if len(integrated_results["model_agreement"].get(integrated_results["best_strategy"], [])) >= 3:
        recommendation = f"宝箱{best_treasure.id} (乘数={best_treasure.multiplier}, 居民={best_treasure.inhabitants})"
        recommendation_type = "单选策略"
    else:
        recommendation = f"宝箱{best_pair[0].id}和宝箱{best_pair[1].id}"
        recommendation_type = "双选策略"
    
    print(f"\n最终建议: {recommendation_type} - {recommendation}")
    
    return {
        "analyzer": analyzer,
        "results": analyzer.results,
        "recommendation": recommendation,
        "recommendation_type": recommendation_type
    }

def main():
    """主函数"""
    # 创建命令行参数解析器
    parser = argparse.ArgumentParser(description="IMC Prosperity 宝箱选择高级策略分析")
    
    # 添加命令行参数
    parser.add_argument("--rational", type=float, default=0.35,
                      help="理性玩家比例 (默认: 0.35)")
    parser.add_argument("--heuristic", type=float, default=0.45,
                      help="启发式玩家比例 (默认: 0.45)")
    parser.add_argument("--second-box-pct", type=float, default=0.05,
                      help="选择第二个宝箱的玩家比例 (默认: 0.05)")
    parser.add_argument("--second-box-cost", type=int, default=50000,
                      help="选择第二个宝箱的成本 (默认: 50000)")
    parser.add_argument("--output-dir", type=str, default=None,
                      help="输出目录 (默认: output/advanced_analysis)")
    
    # 解析命令行参数
    args = parser.parse_args()
    
    # 验证比例参数
    if args.rational < 0 or args.rational > 1:
        print("错误: 理性玩家比例必须在0到1之间")
        return 1
        
    if args.heuristic < 0 or args.heuristic > 1:
        print("错误: 启发式玩家比例必须在0到1之间")
        return 1
        
    if args.rational + args.heuristic > 1:
        print("错误: 理性玩家比例和启发式玩家比例之和不能超过1")
        return 1
        
    if args.second_box_pct < 0 or args.second_box_pct > 1:
        print("错误: 选择第二个宝箱的玩家比例必须在0到1之间")
        return 1
    
    # 设置输出目录
    output_dir = args.output_dir if args.output_dir else create_output_dir()
    
    # 运行分析
    run_advanced_analysis(
        output_dir=output_dir,
        rational_pct=args.rational,
        heuristic_pct=args.heuristic,
        second_box_pct=args.second_box_pct,
        second_box_cost=args.second_box_cost
    )
    
    return 0

if __name__ == "__main__":
    sys.exit(main()) 