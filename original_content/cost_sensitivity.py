#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
成本敏感性分析模块
分析第二宝箱成本变化对最优策略的影响
"""

import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from typing import List, Dict, Tuple, Optional
from simulator import Treasure, TreasureOptimizer

class CostSensitivityAnalyzer:
    """成本敏感性分析器"""
    
    def __init__(self, treasures: List[Treasure], 
                 output_dir: str = "output",
                 rational_pct: float = 0.35, 
                 heuristic_pct: float = 0.45,
                 second_box_pct: float = 0.05,
                 second_choice_rational_factor: float = 0.7):
        """
        初始化成本敏感性分析器
        
        参数:
            treasures: 宝箱列表
            output_dir: 输出目录
            rational_pct: 理性玩家比例
            heuristic_pct: 启发式玩家比例
            second_box_pct: 选择第二个宝箱的玩家比例
            second_choice_rational_factor: 第二选择理性因子
        """
        self.treasures = treasures
        self.rational_pct = rational_pct
        self.heuristic_pct = heuristic_pct
        self.second_box_pct = second_box_pct
        self.second_choice_rational_factor = second_choice_rational_factor
        
        # 创建输出目录
        self.output_dir = output_dir
        os.makedirs(output_dir, exist_ok=True)
    
    def analyze_cost_sensitivity(self, 
                               base_cost: int = 50000, 
                               cost_range: Tuple[int, int] = (30000, 70000), 
                               step: int = 5000) -> Dict:
        """
        分析第二宝箱成本对最优策略的影响
        
        参数:
            base_cost: 基准成本
            cost_range: 成本范围元组 (最小值, 最大值)
            step: 成本步长
            
        返回:
            包含分析结果的字典
        """
        results = []
        costs = range(cost_range[0], cost_range[1]+1, step)
        
        print(f"开始成本敏感性分析 (成本范围: {cost_range[0]}-{cost_range[1]}, 步长: {step})...")
        
        for cost in costs:
            # 创建优化器并设置成本
            optimizer = TreasureOptimizer(
                treasures=self.treasures,
                rational_pct=self.rational_pct,
                heuristic_pct=self.heuristic_pct,
                second_box_pct=self.second_box_pct,
                second_box_cost=cost,
                second_choice_rational_factor=self.second_choice_rational_factor
            )
            
            # 运行迭代
            distribution, profits = optimizer.run_iteration(print_progress=False)
            
            # 分析最优策略
            best_single, best_pair, single_profit, pair_profit = optimizer.analyze_optimal_strategy(profits)
            
            # 收集结果
            results.append({
                "cost": cost,
                "best_single": best_single,
                "best_pair": best_pair,
                "single_profit": single_profit,
                "pair_profit": pair_profit,
                "profit_diff": pair_profit - single_profit
            })
            
            print(f"成本 {cost}: 最佳单选=宝箱{best_single}(收益:{single_profit:.2f}), " +
                  f"最佳双选=宝箱{best_pair[0]}+宝箱{best_pair[1]}(净收益:{pair_profit:.2f}), " +
                  f"差值:{pair_profit - single_profit:.2f}")
        
        # 成本平衡点分析 - 找到双选策略开始优于单选的成本阈值
        break_even = None
        for r in results:
            if r["profit_diff"] > 0:
                break_even = r["cost"]
                break
        
        if break_even:
            print(f"\n找到成本平衡点: {break_even}")
            print(f"当第二宝箱成本低于{break_even}时，双选策略开始优于单选策略")
        else:
            print("\n未找到成本平衡点 - 在测试的成本范围内，单选策略始终优于双选策略")
        
        # 返回结果和平衡点
        return {
            "results": results,
            "break_even": break_even
        }
    
    def visualize_results(self, results: List[Dict]) -> None:
        """
        可视化分析结果
        
        参数:
            results: 分析结果列表
        """
        costs = [r["cost"] for r in results]
        single_profits = [r["single_profit"] for r in results]
        pair_profits = [r["pair_profit"] for r in results]
        profit_diffs = [r["profit_diff"] for r in results]
        
        # 创建一个新的图形
        plt.figure(figsize=(12, 8))
        
        # 第一个子图: 单选与双选收益对比
        plt.subplot(2, 1, 1)
        plt.plot(costs, single_profits, 'b-', marker='o', label='单选最佳策略')
        plt.plot(costs, pair_profits, 'r-', marker='x', label='双选最佳策略')
        plt.axhline(y=0, color='black', linestyle='-', alpha=0.3)
        
        # 填充单选更好的区域
        better_single_x = []
        better_single_y1 = []
        better_single_y2 = []
        
        for i, cost in enumerate(costs):
            if single_profits[i] > pair_profits[i]:
                better_single_x.append(cost)
                better_single_y1.append(single_profits[i])
                better_single_y2.append(pair_profits[i])
        
        if better_single_x:
            plt.fill_between(better_single_x, better_single_y1, better_single_y2, 
                            color='blue', alpha=0.2, label='单选更优区域')
        
        # 填充双选更好的区域
        better_pair_x = []
        better_pair_y1 = []
        better_pair_y2 = []
        
        for i, cost in enumerate(costs):
            if pair_profits[i] > single_profits[i]:
                better_pair_x.append(cost)
                better_pair_y1.append(pair_profits[i])
                better_pair_y2.append(single_profits[i])
        
        if better_pair_x:
            plt.fill_between(better_pair_x, better_pair_y1, better_pair_y2, 
                            color='red', alpha=0.2, label='双选更优区域')
        
        plt.grid(True, linestyle='--', alpha=0.7)
        plt.xlabel('第二宝箱成本')
        plt.ylabel('收益')
        plt.title('不同成本下的单选与双选策略收益对比')
        plt.legend()
        
        # 第二个子图: 收益差值
        plt.subplot(2, 1, 2)
        plt.bar(costs, profit_diffs, color=['green' if diff > 0 else 'red' for diff in profit_diffs])
        plt.axhline(y=0, color='black', linestyle='-')
        
        # 添加标签
        for i, diff in enumerate(profit_diffs):
            plt.text(costs[i], diff + (1000 if diff > 0 else -1000), 
                    f'{diff:.0f}', ha='center', va='center', fontsize=9)
        
        plt.grid(True, linestyle='--', alpha=0.7)
        plt.xlabel('第二宝箱成本')
        plt.ylabel('双选收益 - 单选收益')
        plt.title('不同成本下的双选策略相对收益')
        
        plt.tight_layout()
        plt.savefig(os.path.join(self.output_dir, 'cost_sensitivity_analysis.png'))
        print(f"图表已保存到 {os.path.join(self.output_dir, 'cost_sensitivity_analysis.png')}")
        
        # 显示图表
        plt.close()
    
    def generate_report(self, analysis_results: Dict) -> None:
        """
        生成分析报告
        
        参数:
            analysis_results: 分析结果
        """
        results = analysis_results["results"]
        break_even = analysis_results["break_even"]
        
        # 创建DataFrame
        df = pd.DataFrame(results)
        
        # 添加是否双选更优的列
        df['双选更优'] = df['profit_diff'] > 0
        
        # 保存CSV报告
        csv_path = os.path.join(self.output_dir, 'cost_sensitivity_report.csv')
        df.to_csv(csv_path, index=False)
        print(f"详细报告已保存到 {csv_path}")
        
        # 生成文本报告
        report_path = os.path.join(self.output_dir, 'cost_sensitivity_report.txt')
        with open(report_path, 'w', encoding='utf-8') as f:
            f.write("# 成本敏感性分析报告\n\n")
            f.write(f"分析日期: {pd.Timestamp.now().strftime('%Y-%m-%d %H:%M:%S')}\n\n")
            
            f.write("## 参数设置\n\n")
            f.write(f"- 理性玩家比例: {self.rational_pct}\n")
            f.write(f"- 启发式玩家比例: {self.heuristic_pct}\n")
            f.write(f"- 随机玩家比例: {1 - self.rational_pct - self.heuristic_pct}\n")
            f.write(f"- 第二宝箱选择比例: {self.second_box_pct}\n")
            f.write(f"- 第二选择理性因子: {self.second_choice_rational_factor}\n\n")
            
            f.write("## 关键发现\n\n")
            
            if break_even:
                f.write(f"- 成本平衡点: {break_even}\n")
                f.write(f"- 当第二宝箱成本低于{break_even}时，双选策略开始优于单选策略\n\n")
            else:
                f.write("- 未找到成本平衡点 - 在测试的成本范围内，单选策略始终优于双选策略\n\n")
            
            # 最优成本设置
            best_dual_result = sorted(results, key=lambda x: x['profit_diff'], reverse=True)[0]
            worst_dual_result = sorted(results, key=lambda x: x['profit_diff'])[0]
            
            f.write(f"- 双选策略最有利的成本: {best_dual_result['cost']}\n")
            f.write(f"  - 此时双选收益: {best_dual_result['pair_profit']:.2f}\n")
            f.write(f"  - 相对于单选的差值: {best_dual_result['profit_diff']:.2f}\n\n")
            
            f.write(f"- 双选策略最不利的成本: {worst_dual_result['cost']}\n")
            f.write(f"  - 此时双选收益: {worst_dual_result['pair_profit']:.2f}\n")
            f.write(f"  - 相对于单选的差值: {worst_dual_result['profit_diff']:.2f}\n\n")
            
            # 添加单选策略稳定性分析
            single_choices = [r['best_single'] for r in results]
            most_common_single = max(set(single_choices), key=single_choices.count)
            
            f.write(f"- 单选策略稳定性: 在所有测试成本下，最常见的最优单选是宝箱{most_common_single}\n")
            f.write(f"  - 在{len(results)}个测试成本中出现{single_choices.count(most_common_single)}次\n\n")
            
            # 添加双选策略稳定性分析
            pair_choices = [f"{r['best_pair'][0]}-{r['best_pair'][1]}" for r in results]
            most_common_pair = max(set(pair_choices), key=pair_choices.count)
            box1, box2 = most_common_pair.split('-')
            
            f.write(f"- 双选策略稳定性: 在所有测试成本下，最常见的最优双选是宝箱{box1}+宝箱{box2}\n")
            f.write(f"  - 在{len(results)}个测试成本中出现{pair_choices.count(most_common_pair)}次\n\n")
            
            f.write("## 详细结果\n\n")
            
            # 添加表格标题
            f.write("| 成本 | 最佳单选 | 单选收益 | 最佳双选 | 双选净收益 | 收益差值 | 双选更优 |\n")
            f.write("|------|----------|----------|----------|------------|----------|----------|\n")
            
            # 添加表格内容
            for r in results:
                is_better = "是" if r["profit_diff"] > 0 else "否"
                f.write(f"| {r['cost']} | 宝箱{r['best_single']} | {r['single_profit']:.2f} | " +
                        f"宝箱{r['best_pair'][0]}+宝箱{r['best_pair'][1]} | {r['pair_profit']:.2f} | " +
                        f"{r['profit_diff']:.2f} | {is_better} |\n")
            
            f.write("\n## 结论\n\n")
            
            if break_even:
                f.write(f"根据分析，当第二宝箱成本低于{break_even}时，双选策略开始优于单选策略。")
                f.write(f"在当前成本为{50000}的情况下，单选策略更优。")
                f.write(f"如果未来比赛规则调整，使第二宝箱成本降至{break_even}以下，应当考虑双选策略。\n")
            else:
                f.write("在分析的整个成本范围内，单选策略始终优于双选策略。")
                f.write("即使将成本降低到最低测试值（30,000），双选策略仍无法超过单选策略。")
                f.write("建议坚持单选策略，除非第二宝箱成本有显著降低。\n")
        
        print(f"分析报告已保存到 {report_path}")
    
    def run_full_analysis(self, 
                         base_cost: int = 50000, 
                         cost_range: Tuple[int, int] = (30000, 70000), 
                         step: int = 5000) -> Dict:
        """
        运行完整的成本敏感性分析
        
        参数:
            base_cost: 基准成本
            cost_range: 成本范围元组 (最小值, 最大值)
            step: 成本步长
            
        返回:
            包含分析结果的字典
        """
        # 运行分析
        analysis_results = self.analyze_cost_sensitivity(base_cost, cost_range, step)
        
        # 可视化结果
        self.visualize_results(analysis_results["results"])
        
        # 生成报告
        self.generate_report(analysis_results)
        
        return analysis_results


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
    
    # 创建成本敏感性分析器
    analyzer = CostSensitivityAnalyzer(
        treasures=treasures,
        rational_pct=0.35,
        heuristic_pct=0.45,
        second_box_pct=0.05,
        second_choice_rational_factor=0.7
    )
    
    # 运行完整分析
    # 测试成本范围从10,000到70,000，步长为5,000
    analyzer.run_full_analysis(
        base_cost=50000,
        cost_range=(10000, 70000),
        step=5000
    )


if __name__ == "__main__":
    main() 