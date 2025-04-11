#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
参数敏感性分析模块
分析不同参数组合对最优策略的影响
"""

import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from typing import List, Dict, Tuple, Optional, Union, Any
from simulator import Treasure, TreasureOptimizer
import itertools

class SensitivityAnalyzer:
    """参数敏感性分析器"""
    
    def __init__(self, treasures: List[Treasure], 
                 output_dir: str = "output"):
        """
        初始化参数敏感性分析器
        
        参数:
            treasures: 宝箱列表
            output_dir: 输出目录
        """
        self.treasures = treasures
        
        # 创建输出目录
        self.output_dir = output_dir
        os.makedirs(output_dir, exist_ok=True)
    
    def create_parameter_grid(self, 
                            rational_range: Tuple[float, float, int] = (0.2, 0.5, 4),
                            heuristic_range: Tuple[float, float, int] = (0.3, 0.6, 4),
                            second_box_range: Tuple[float, float, int] = (0.01, 0.1, 3),
                            second_choice_rational_range: Tuple[float, float, int] = (0.5, 1.0, 3)) -> List[Dict]:
        """
        创建参数网格
        
        参数:
            rational_range: 理性玩家比例范围(最小值, 最大值, 步数)
            heuristic_range: 启发式玩家比例范围(最小值, 最大值, 步数)
            second_box_range: 第二宝箱选择比例范围(最小值, 最大值, 步数)
            second_choice_rational_range: 第二选择理性因子范围(最小值, 最大值, 步数)
            
        返回:
            参数组合列表
        """
        # 生成每个参数的等距点
        rational_values = np.linspace(rational_range[0], rational_range[1], rational_range[2])
        heuristic_values = np.linspace(heuristic_range[0], heuristic_range[1], heuristic_range[2])
        second_box_values = np.linspace(second_box_range[0], second_box_range[1], second_box_range[2])
        second_choice_values = np.linspace(second_choice_rational_range[0], second_choice_rational_range[1], second_choice_rational_range[2])
        
        # 创建参数组合
        param_grid = []
        
        for rational in rational_values:
            for heuristic in heuristic_values:
                # 确保理性玩家和启发式玩家比例之和不超过1
                if rational + heuristic > 0.95:
                    continue
                    
                for second_box in second_box_values:
                    for second_choice in second_choice_values:
                        param_grid.append({
                            "rational_pct": rational,
                            "heuristic_pct": heuristic,
                            "second_box_pct": second_box,
                            "second_choice_rational_factor": second_choice
                        })
        
        return param_grid
    
    def run_sensitivity_analysis(self, parameter_grid: List[Dict], max_combinations: int = 100) -> Dict:
        """
        运行参数敏感性分析
        
        参数:
            parameter_grid: 参数组合列表
            max_combinations: 最大组合数(防止计算量过大)
            
        返回:
            分析结果字典
        """
        # 限制组合数量
        if len(parameter_grid) > max_combinations:
            print(f"警告: 参数组合数({len(parameter_grid)})超过最大限制({max_combinations})。随机选择{max_combinations}个组合进行分析。")
            np.random.seed(42)  # 设置随机种子以保证结果可重复
            parameter_grid = np.random.choice(parameter_grid, max_combinations, replace=False).tolist()
        
        results = []
        
        print(f"开始参数敏感性分析，共{len(parameter_grid)}个参数组合...")
        
        for i, params in enumerate(parameter_grid):
            print(f"分析参数组合 {i+1}/{len(parameter_grid)}: {params}")
            
            # 创建优化器
            optimizer = TreasureOptimizer(
                treasures=self.treasures,
                **params
            )
            
            # 运行迭代
            distribution, profits = optimizer.run_iteration(print_progress=False)
            
            # 分析最优策略
            best_single, best_pair, single_profit, pair_profit = optimizer.analyze_optimal_strategy(profits)
            
            # 收集结果
            result = {
                **params,  # 包含所有输入参数
                "best_single": best_single,
                "best_pair": best_pair,
                "single_profit": single_profit,
                "pair_profit": pair_profit,
                "profit_diff": pair_profit - single_profit,
                "distribution": {i: distribution[i] for i in range(1, len(self.treasures) + 1)}
            }
            
            results.append(result)
        
        return {
            "results": results,
            "parameter_grid": parameter_grid
        }
    
    def visualize_parameter_heatmap(self, sensitivity_results: Dict, target_param1: str, target_param2: str, 
                                 metric: str = "single_profit", output_filename: str = None) -> None:
        """
        可视化两个参数之间的关系热图
        
        参数:
            sensitivity_results: 敏感性分析结果
            target_param1: 第一个目标参数名称
            target_param2: 第二个目标参数名称
            metric: 要分析的指标
            output_filename: 输出文件名
        """
        results = sensitivity_results["results"]
        
        # 提取所有参数的唯一值
        param1_values = sorted(list(set(result[target_param1] for result in results)))
        param2_values = sorted(list(set(result[target_param2] for result in results)))
        
        # 创建热图数据矩阵
        matrix = np.zeros((len(param1_values), len(param2_values)))
        
        # 填充矩阵
        for result in results:
            i = param1_values.index(result[target_param1])
            j = param2_values.index(result[target_param2])
            matrix[i, j] = result[metric]
        
        # 创建DataFrame
        df = pd.DataFrame(matrix, index=param1_values, columns=param2_values)
        
        # 创建热图
        plt.figure(figsize=(12, 10))
        
        # 中文参数名称映射
        param_names = {
            "rational_pct": "理性玩家比例",
            "heuristic_pct": "启发式玩家比例",
            "second_box_pct": "第二宝箱选择比例",
            "second_choice_rational_factor": "第二选择理性因子"
        }
        
        # 中文指标名称映射
        metric_names = {
            "single_profit": "单选策略收益",
            "pair_profit": "双选策略收益",
            "profit_diff": "双选与单选收益差值",
            "best_single": "最优单选宝箱"
        }
        
        # 设置热图样式
        cmap = "YlGnBu" if metric != "best_single" else "tab10"
        
        # 绘制热图
        heatmap = sns.heatmap(df, annot=True, fmt=".2f" if metric != "best_single" else ".0f", 
                             cmap=cmap, linewidths=0.5, cbar_kws={"label": metric_names.get(metric, metric)})
        
        plt.xlabel(param_names.get(target_param2, target_param2))
        plt.ylabel(param_names.get(target_param1, target_param1))
        plt.title(f"{param_names.get(target_param1, target_param1)}与{param_names.get(target_param2, target_param2)}对{metric_names.get(metric, metric)}的影响")
        
        # 保存图像
        if output_filename is None:
            output_filename = f"param_heatmap_{target_param1}_{target_param2}_{metric}.png"
            
        plt.tight_layout()
        plt.savefig(os.path.join(self.output_dir, output_filename))
        plt.close()
        
        print(f"参数热图已保存到 {os.path.join(self.output_dir, output_filename)}")
    
    def visualize_parameter_importance(self, sensitivity_results: Dict, metric: str = "single_profit", 
                                    output_filename: str = "parameter_importance.png") -> Dict:
        """
        可视化参数重要性
        
        参数:
            sensitivity_results: 敏感性分析结果
            metric: 要分析的指标
            output_filename: 输出文件名
            
        返回:
            参数重要性字典
        """
        results = sensitivity_results["results"]
        
        # 将结果转换为DataFrame
        df = pd.DataFrame(results)
        
        # 计算每个参数的重要性
        # 使用方差作为重要性度量
        importance = {}
        
        # 要分析的参数
        params = ["rational_pct", "heuristic_pct", "second_box_pct", "second_choice_rational_factor"]
        param_names = {
            "rational_pct": "理性玩家比例",
            "heuristic_pct": "启发式玩家比例",
            "second_box_pct": "第二宝箱选择比例",
            "second_choice_rational_factor": "第二选择理性因子"
        }
        
        # 中文指标名称映射
        metric_names = {
            "single_profit": "单选策略收益",
            "pair_profit": "双选策略收益",
            "profit_diff": "双选与单选收益差值",
            "best_single": "最优单选宝箱"
        }
        
        for param in params:
            # 对于每个唯一的参数值，计算目标指标的平均值
            param_values = df[param].unique()
            metric_means = [df[df[param] == val][metric].mean() for val in param_values]
            
            # 计算这些平均值的标准差作为重要性度量
            importance[param] = np.std(metric_means)
        
        # 归一化重要性
        total_importance = sum(importance.values())
        normalized_importance = {k: v / total_importance for k, v in importance.items()}
        
        # 按重要性排序
        sorted_importance = sorted(normalized_importance.items(), key=lambda x: x[1], reverse=True)
        
        # 可视化
        plt.figure(figsize=(10, 6))
        
        # 创建条形图
        bars = plt.bar(
            [param_names.get(param, param) for param, _ in sorted_importance],
            [imp for _, imp in sorted_importance],
            color="skyblue"
        )
        
        # 添加数值标签
        for bar in bars:
            height = bar.get_height()
            plt.text(bar.get_x() + bar.get_width() / 2., height + 0.01,
                     f'{height:.2%}', ha='center', va='bottom')
        
        plt.ylim(0, max([imp for _, imp in sorted_importance]) * 1.2)
        plt.ylabel("相对重要性")
        plt.title(f"参数对{metric_names.get(metric, metric)}的相对重要性")
        plt.grid(axis='y', linestyle='--', alpha=0.7)
        
        # 保存图像
        plt.tight_layout()
        plt.savefig(os.path.join(self.output_dir, output_filename))
        plt.close()
        
        print(f"参数重要性图已保存到 {os.path.join(self.output_dir, output_filename)}")
        
        return normalized_importance
    
    def analyze_optimal_parameters(self, sensitivity_results: Dict, metric: str = "single_profit") -> Dict:
        """
        分析最优参数组合
        
        参数:
            sensitivity_results: 敏感性分析结果
            metric: 要优化的指标
            
        返回:
            最优参数分析结果
        """
        results = sensitivity_results["results"]
        
        # 将结果转换为DataFrame
        df = pd.DataFrame(results)
        
        # 按指标排序
        df_sorted = df.sort_values(by=metric, ascending=False)
        
        # 获取最优组合
        best_combination = df_sorted.iloc[0].to_dict()
        
        # 计算每个参数的最优区间
        # 取指标值排名前10%的结果
        top_percent = 0.1
        top_n = max(int(len(df) * top_percent), 1)
        top_df = df_sorted.head(top_n)
        
        # 计算每个参数的区间
        param_ranges = {}
        
        for param in ["rational_pct", "heuristic_pct", "second_box_pct", "second_choice_rational_factor"]:
            param_values = top_df[param].values
            param_ranges[param] = {
                "min": np.min(param_values),
                "max": np.max(param_values),
                "mean": np.mean(param_values),
                "median": np.median(param_values)
            }
        
        return {
            "best_combination": best_combination,
            "param_ranges": param_ranges,
            "top_combinations": top_df.to_dict(orient="records")
        }
    
    def generate_sensitivity_report(self, sensitivity_results: Dict, importance_results: Dict, 
                                  optimal_params: Dict, output_filename: str = "sensitivity_report.txt") -> None:
        """
        生成敏感性分析报告
        
        参数:
            sensitivity_results: 敏感性分析结果
            importance_results: 参数重要性结果
            optimal_params: 最优参数分析结果
            output_filename: 输出文件名
        """
        report_path = os.path.join(self.output_dir, output_filename)
        
        with open(report_path, 'w', encoding='utf-8') as f:
            f.write("# 参数敏感性分析报告\n\n")
            f.write(f"分析日期: {pd.Timestamp.now().strftime('%Y-%m-%d %H:%M:%S')}\n\n")
            
            # 参数重要性
            f.write("## 参数重要性\n\n")
            
            # 参数中文名映射
            param_names = {
                "rational_pct": "理性玩家比例",
                "heuristic_pct": "启发式玩家比例",
                "second_box_pct": "第二宝箱选择比例",
                "second_choice_rational_factor": "第二选择理性因子"
            }
            
            # 排序后的重要性
            sorted_importance = sorted(importance_results.items(), key=lambda x: x[1], reverse=True)
            
            for param, importance in sorted_importance:
                f.write(f"- {param_names.get(param, param)}: {importance:.2%}\n")
            
            f.write("\n## 最优参数组合\n\n")
            
            best = optimal_params["best_combination"]
            
            f.write("### 最佳单一组合\n\n")
            f.write(f"- 理性玩家比例: {best['rational_pct']:.2f}\n")
            f.write(f"- 启发式玩家比例: {best['heuristic_pct']:.2f}\n")
            f.write(f"- 随机玩家比例: {1 - best['rational_pct'] - best['heuristic_pct']:.2f}\n")
            f.write(f"- 第二宝箱选择比例: {best['second_box_pct']:.2f}\n")
            f.write(f"- 第二选择理性因子: {best['second_choice_rational_factor']:.2f}\n\n")
            
            f.write(f"最佳单选策略: 宝箱{best['best_single']} (收益: {best['single_profit']:.2f})\n")
            f.write(f"最佳双选策略: 宝箱{best['best_pair'][0]}+宝箱{best['best_pair'][1]} (收益: {best['pair_profit']:.2f})\n")
            f.write(f"双选与单选收益差值: {best['profit_diff']:.2f}\n\n")
            
            f.write("### 最优参数区间\n\n")
            
            ranges = optimal_params["param_ranges"]
            
            f.write("| 参数 | 最小值 | 最大值 | 平均值 | 中位数 |\n")
            f.write("|------|--------|--------|--------|--------|\n")
            
            for param, range_data in ranges.items():
                f.write(f"| {param_names.get(param, param)} | {range_data['min']:.2f} | {range_data['max']:.2f} | {range_data['mean']:.2f} | {range_data['median']:.2f} |\n")
            
            f.write("\n## 参数敏感性阈值\n\n")
            
            # 计算参数的敏感性阈值
            # 对于每个参数，找出它在什么值时会导致最优宝箱改变
            results_df = pd.DataFrame(sensitivity_results["results"])
            params = ["rational_pct", "heuristic_pct", "second_box_pct", "second_choice_rational_factor"]
            
            for param in params:
                # 按参数值排序
                sorted_df = results_df.sort_values(by=param)
                
                # 查找最优宝箱变化的点
                thresholds = []
                prev_best = None
                
                for i, row in sorted_df.iterrows():
                    current_best = row["best_single"]
                    if prev_best is not None and current_best != prev_best:
                        thresholds.append((row[param], prev_best, current_best))
                    prev_best = current_best
                
                # 写入阈值
                if thresholds:
                    f.write(f"### {param_names.get(param, param)}的敏感性阈值\n\n")
                    
                    for threshold, old_best, new_best in thresholds:
                        f.write(f"- 当{param_names.get(param, param)}超过{threshold:.2f}时，最优宝箱从宝箱{old_best}变为宝箱{new_best}\n")
                    
                    f.write("\n")
            
            f.write("## 结论与建议\n\n")
            
            # 根据重要性排名得出结论
            most_important = sorted_importance[0][0]
            second_important = sorted_importance[1][0]
            
            f.write(f"根据敏感性分析，{param_names.get(most_important, most_important)}是影响最优策略的最重要参数，其次是{param_names.get(second_important, second_important)}。\n\n")
            
            # 提出建议
            f.write("### 参数设置建议\n\n")
            
            f.write(f"1. {param_names.get(most_important, most_important)}应设置在{ranges[most_important]['min']:.2f}到{ranges[most_important]['max']:.2f}之间，建议值为{ranges[most_important]['median']:.2f}。\n")
            f.write(f"2. {param_names.get(second_important, second_important)}应设置在{ranges[second_important]['min']:.2f}到{ranges[second_important]['max']:.2f}之间，建议值为{ranges[second_important]['median']:.2f}。\n")
            
            # 建议的最优策略
            best_single_counts = results_df["best_single"].value_counts()
            most_common_best = best_single_counts.index[0]
            occurrence_pct = best_single_counts[most_common_best] / len(results_df) * 100
            
            f.write(f"\n在{len(sensitivity_results['results'])}个参数组合中，宝箱{most_common_best}作为最优选择出现了{occurrence_pct:.1f}%的情况，是最稳健的选择。\n")
            
            # 对未来的建议
            f.write("\n### 进一步分析建议\n\n")
            f.write(f"1. 对{param_names.get(most_important, most_important)}进行更细致的分析，特别是在{ranges[most_important]['min']:.2f}到{ranges[most_important]['max']:.2f}范围内。\n")
            f.write("2. 考虑进一步研究不同参数组合之间的交互效应。\n")
            f.write("3. 可以使用更高级的优化算法(如贝叶斯优化)来寻找最优参数组合。\n")
        
        print(f"敏感性分析报告已保存到 {report_path}")
    
    def run_full_analysis(self, max_combinations: int = 100) -> Dict:
        """
        运行完整的敏感性分析
        
        参数:
            max_combinations: 最大参数组合数
            
        返回:
            分析结果字典
        """
        print(f"开始运行参数敏感性分析 (最大组合数: {max_combinations})...")
        
        # 创建参数网格
        parameter_grid = self.create_parameter_grid(
            rational_range=(0.2, 0.5, 4),
            heuristic_range=(0.3, 0.6, 4),
            second_box_range=(0.01, 0.1, 3),
            second_choice_rational_range=(0.5, 1.0, 3)
        )
        
        # 运行敏感性分析
        sensitivity_results = self.run_sensitivity_analysis(parameter_grid, max_combinations)
        
        # 可视化参数重要性
        importance_results = self.visualize_parameter_importance(sensitivity_results, metric="single_profit")
        
        # 分析最优参数
        optimal_params = self.analyze_optimal_parameters(sensitivity_results, metric="single_profit")
        
        # 生成热图
        self.visualize_parameter_heatmap(sensitivity_results, "rational_pct", "heuristic_pct", metric="single_profit")
        self.visualize_parameter_heatmap(sensitivity_results, "second_box_pct", "second_choice_rational_factor", metric="single_profit")
        self.visualize_parameter_heatmap(sensitivity_results, "rational_pct", "second_box_pct", metric="single_profit")
        
        # 生成报告
        self.generate_sensitivity_report(sensitivity_results, importance_results, optimal_params)
        
        return {
            "sensitivity_results": sensitivity_results,
            "importance_results": importance_results,
            "optimal_params": optimal_params
        }


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
    
    # 创建敏感性分析器
    analyzer = SensitivityAnalyzer(
        treasures=treasures,
        output_dir="output/sensitivity"
    )
    
    # 运行完整分析
    analyzer.run_full_analysis(max_combinations=50)  # 限制组合数以加快分析


if __name__ == "__main__":
    main() 