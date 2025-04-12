#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
多目标优化框架
综合考虑收益、风险和稳定性的多目标优化
"""

import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from typing import List, Dict, Tuple, Optional, Union, Any
from simulator import Treasure, TreasureOptimizer
from risk_analysis import RiskAnalyzer

class MultiObjectiveOptimizer:
    """多目标优化器"""
    
    def __init__(self, treasures: List[Treasure], 
                 output_dir: str = "output/multi_objective",
                 rational_pct: float = 0.35, 
                 heuristic_pct: float = 0.45,
                 second_box_pct: float = 0.05,
                 second_box_cost: int = 50000,
                 second_choice_rational_factor: float = 0.7):
        """
        初始化多目标优化器
        
        参数:
            treasures: 宝箱列表
            output_dir: 输出目录
            rational_pct: 理性玩家比例
            heuristic_pct: 启发式玩家比例
            second_box_pct: 选择第二个宝箱的玩家比例
            second_box_cost: 选择第二个宝箱的成本
            second_choice_rational_factor: 第二选择理性因子
        """
        self.treasures = treasures
        self.rational_pct = rational_pct
        self.heuristic_pct = heuristic_pct
        self.second_box_pct = second_box_pct
        self.second_box_cost = second_box_cost
        self.second_choice_rational_factor = second_choice_rational_factor
        
        # 创建输出目录
        self.output_dir = output_dir
        os.makedirs(output_dir, exist_ok=True)
        
        # 创建优化器和风险分析器
        self.optimizer = TreasureOptimizer(
            treasures=self.treasures,
            rational_pct=self.rational_pct,
            heuristic_pct=self.heuristic_pct,
            second_box_pct=self.second_box_pct,
            second_box_cost=self.second_box_cost,
            second_choice_rational_factor=self.second_choice_rational_factor
        )
        
        self.risk_analyzer = RiskAnalyzer(
            treasures=self.treasures,
            output_dir=os.path.join(output_dir, "risk"),
            rational_pct=self.rational_pct,
            heuristic_pct=self.heuristic_pct,
            second_box_pct=self.second_box_pct,
            second_choice_rational_factor=self.second_choice_rational_factor
        )
    
    def run_optimization(self, num_simulations: int = 1000, user_weights: Dict = None) -> Dict:
        """
        运行多目标优化
        
        参数:
            num_simulations: 模拟次数
            user_weights: 用户对不同目标的权重
                {
                    "profit": 0.6,       # 收益权重
                    "risk": 0.3,         # 风险权重
                    "stability": 0.1     # 稳定性权重
                }
                
        返回:
            优化结果字典
        """
        # 设置默认权重
        if user_weights is None:
            user_weights = {
                "profit": 0.6,
                "risk": 0.3,
                "stability": 0.1
            }
        
        # 归一化权重
        total_weight = sum(user_weights.values())
        normalized_weights = {k: v / total_weight for k, v in user_weights.items()}
        
        print(f"开始运行多目标优化 (权重: 收益={normalized_weights['profit']:.2f}, " +
              f"风险={normalized_weights['risk']:.2f}, 稳定性={normalized_weights['stability']:.2f})...")
        
        # 1. 计算收益
        distribution, profits = self.optimizer.run_iteration(print_progress=False)
        
        # 2. 计算风险指标
        risk_measures = self.risk_analyzer.analyze_all_boxes(num_simulations)
        
        # 3. 计算稳定性指标
        stability_scores = self.calculate_stability_scores(num_simulations)
        
        # 4. 多目标评分
        multi_objective_scores = self.calculate_multi_objective_scores(
            profits=profits,
            risk_measures=risk_measures,
            stability_scores=stability_scores,
            weights=normalized_weights
        )
        
        # 5. 找出帕累托最优解
        pareto_optimal = self.find_pareto_optimal_solutions(
            profits=profits,
            risk_measures=risk_measures,
            stability_scores=stability_scores
        )
        
        # 6. 找出综合得分最高的宝箱
        best_box = max(multi_objective_scores.items(), key=lambda x: x[1])[0]
        
        # 收集结果
        result = {
            "distribution": distribution,
            "profits": profits,
            "risk_measures": risk_measures,
            "stability_scores": stability_scores,
            "multi_objective_scores": multi_objective_scores,
            "pareto_optimal": pareto_optimal,
            "best_box": best_box,
            "weights": normalized_weights
        }
        
        return result
    
    def calculate_stability_scores(self, num_simulations: int = 100) -> Dict[int, float]:
        """
        计算宝箱选择的稳定性评分
        
        参数:
            num_simulations: 模拟次数
            
        返回:
            宝箱稳定性评分字典
        """
        print(f"计算稳定性评分 (模拟次数: {num_simulations})...")
        
        # 跟踪每个宝箱在不同参数设置下成为最优选择的频率
        optimal_counts = {box_id: 0 for box_id in range(1, len(self.treasures) + 1)}
        
        # 参数扰动范围
        perturbation_range = 0.05
        
        for _ in range(num_simulations):
            # 随机扰动参数
            rational_perturb = self.rational_pct * (1 + np.random.uniform(-perturbation_range, perturbation_range))
            heuristic_perturb = self.heuristic_pct * (1 + np.random.uniform(-perturbation_range, perturbation_range))
            
            # 确保比例之和不超过1
            total = rational_perturb + heuristic_perturb
            if total > 0.95:
                ratio = 0.95 / total
                rational_perturb *= ratio
                heuristic_perturb *= ratio
            
            second_box_perturb = self.second_box_pct * (1 + np.random.uniform(-perturbation_range, perturbation_range))
            second_box_perturb = max(0.01, min(0.2, second_box_perturb))
            
            rational_factor_perturb = self.second_choice_rational_factor * (1 + np.random.uniform(-perturbation_range, perturbation_range))
            rational_factor_perturb = max(0.1, min(1.0, rational_factor_perturb))
            
            # 创建扰动的优化器
            perturbed_optimizer = TreasureOptimizer(
                treasures=self.treasures,
                rational_pct=rational_perturb,
                heuristic_pct=heuristic_perturb,
                second_box_pct=second_box_perturb,
                second_box_cost=self.second_box_cost,
                second_choice_rational_factor=rational_factor_perturb
            )
            
            # 运行一次迭代
            _, profits = perturbed_optimizer.run_iteration(print_progress=False)
            
            # 找出最优宝箱
            best_single, _, _, _ = perturbed_optimizer.analyze_optimal_strategy(profits)
            
            # 累计计数
            optimal_counts[best_single] += 1
        
        # 计算稳定性评分 (出现频率)
        stability_scores = {box_id: count / num_simulations for box_id, count in optimal_counts.items()}
        
        return stability_scores
    
    def calculate_multi_objective_scores(self, profits: Dict, risk_measures: Dict, 
                                       stability_scores: Dict, weights: Dict) -> Dict[int, float]:
        """
        计算多目标评分
        
        参数:
            profits: 收益字典
            risk_measures: 风险指标字典
            stability_scores: 稳定性评分字典
            weights: 权重字典
            
        返回:
            多目标评分字典
        """
        print("计算多目标评分...")
        
        # 提取所有宝箱的ID
        box_ids = list(profits.keys())
        
        # 构建评分组件
        profit_scores = {box_id: profits[box_id] for box_id in box_ids}
        risk_scores = {box_id: risk_measures[box_id]["sharpe_ratio"] for box_id in box_ids}
        
        # 归一化评分
        def normalize(scores):
            min_score = min(scores.values())
            max_score = max(scores.values())
            return {k: (v - min_score) / (max_score - min_score) if max_score > min_score else 0.5 
                   for k, v in scores.items()}
        
        normalized_profit_scores = normalize(profit_scores)
        normalized_risk_scores = normalize(risk_scores)
        normalized_stability_scores = normalize(stability_scores)
        
        # 计算加权综合评分
        multi_objective_scores = {}
        
        for box_id in box_ids:
            score = (
                weights["profit"] * normalized_profit_scores[box_id] +
                weights["risk"] * normalized_risk_scores[box_id] +
                weights["stability"] * normalized_stability_scores[box_id]
            )
            multi_objective_scores[box_id] = score
        
        return multi_objective_scores
    
    def find_pareto_optimal_solutions(self, profits: Dict, risk_measures: Dict, 
                                    stability_scores: Dict) -> List[int]:
        """
        找出帕累托最优解
        
        参数:
            profits: 收益字典
            risk_measures: 风险指标字典
            stability_scores: 稳定性评分字典
            
        返回:
            帕累托最优宝箱ID列表
        """
        print("寻找帕累托最优解...")
        
        # 提取所有宝箱的ID
        box_ids = list(profits.keys())
        
        # 构建评分组件
        profit_scores = {box_id: profits[box_id] for box_id in box_ids}
        risk_scores = {box_id: risk_measures[box_id]["sharpe_ratio"] for box_id in box_ids}
        
        # 帕累托判定: 一个解如果在所有维度上都不比另一个解差，且至少在一个维度上更好，则支配另一个解
        pareto_optimal = []
        
        for box_id in box_ids:
            is_dominated = False
            
            for other_id in box_ids:
                if box_id == other_id:
                    continue
                
                # 检查other_id是否在所有维度上都不比box_id差
                if (profit_scores[other_id] >= profit_scores[box_id] and
                    risk_scores[other_id] >= risk_scores[box_id] and
                    stability_scores[other_id] >= stability_scores[box_id]):
                    
                    # 检查other_id是否至少在一个维度上严格更好
                    if (profit_scores[other_id] > profit_scores[box_id] or
                        risk_scores[other_id] > risk_scores[box_id] or
                        stability_scores[other_id] > stability_scores[box_id]):
                        
                        is_dominated = True
                        break
            
            if not is_dominated:
                pareto_optimal.append(box_id)
        
        return pareto_optimal
    
    def visualize_pareto_front(self, optimization_result: Dict, output_filename: str = "pareto_front.png") -> None:
        """
        可视化帕累托前沿
        
        参数:
            optimization_result: 优化结果
            output_filename: 输出文件名
        """
        profits = optimization_result["profits"]
        risk_measures = optimization_result["risk_measures"]
        pareto_optimal = optimization_result["pareto_optimal"]
        
        # 提取所有宝箱的ID
        box_ids = list(profits.keys())
        
        # 提取收益和风险数据
        profit_values = [profits[box_id] for box_id in box_ids]
        risk_values = [risk_measures[box_id]["sharpe_ratio"] for box_id in box_ids]
        
        # 创建图像
        plt.figure(figsize=(12, 8))
        
        # 绘制所有点
        non_pareto = [box_id for box_id in box_ids if box_id not in pareto_optimal]
        plt.scatter([profits[box_id] for box_id in non_pareto], 
                   [risk_measures[box_id]["sharpe_ratio"] for box_id in non_pareto], 
                   color='blue', alpha=0.5, label='非帕累托最优')
        
        # 绘制帕累托前沿点
        plt.scatter([profits[box_id] for box_id in pareto_optimal], 
                   [risk_measures[box_id]["sharpe_ratio"] for box_id in pareto_optimal], 
                   color='red', s=100, label='帕累托最优')
        
        # 标记每个宝箱
        for box_id in box_ids:
            plt.annotate(f'{box_id}', (profits[box_id], risk_measures[box_id]["sharpe_ratio"]),
                        xytext=(5, 5), textcoords='offset points',
                        fontsize=12, fontweight='bold' if box_id in pareto_optimal else 'normal')
        
        # 添加标题和标签
        plt.xlabel('收益')
        plt.ylabel('夏普比率')
        plt.title('收益-风险帕累托前沿')
        plt.grid(True, linestyle='--', alpha=0.7)
        plt.legend()
        
        # 保存图像
        plt.tight_layout()
        plt.savefig(os.path.join(self.output_dir, output_filename))
        plt.close()
        
        print(f"帕累托前沿图已保存到 {os.path.join(self.output_dir, output_filename)}")
    
    def visualize_3d_objective_space(self, optimization_result: Dict, output_filename: str = "3d_objective_space.png") -> None:
        """
        可视化三维目标空间
        
        参数:
            optimization_result: 优化结果
            output_filename: 输出文件名
        """
        profits = optimization_result["profits"]
        risk_measures = optimization_result["risk_measures"]
        stability_scores = optimization_result["stability_scores"]
        pareto_optimal = optimization_result["pareto_optimal"]
        
        # 提取所有宝箱的ID
        box_ids = list(profits.keys())
        
        # 创建3D图像
        fig = plt.figure(figsize=(12, 10))
        ax = fig.add_subplot(111, projection='3d')
        
        # 提取数据
        profit_values = [profits[box_id] for box_id in box_ids]
        risk_values = [risk_measures[box_id]["sharpe_ratio"] for box_id in box_ids]
        stability_values = [stability_scores[box_id] for box_id in box_ids]
        
        # 绘制非帕累托最优点
        non_pareto = [box_id for box_id in box_ids if box_id not in pareto_optimal]
        ax.scatter([profits[box_id] for box_id in non_pareto], 
                  [risk_measures[box_id]["sharpe_ratio"] for box_id in non_pareto],
                  [stability_scores[box_id] for box_id in non_pareto],
                  color='blue', alpha=0.5, label='非帕累托最优')
        
        # 绘制帕累托最优点
        ax.scatter([profits[box_id] for box_id in pareto_optimal], 
                  [risk_measures[box_id]["sharpe_ratio"] for box_id in pareto_optimal],
                  [stability_scores[box_id] for box_id in pareto_optimal],
                  color='red', s=100, label='帕累托最优')
        
        # 标记每个宝箱
        for box_id in box_ids:
            ax.text(profits[box_id], risk_measures[box_id]["sharpe_ratio"], stability_scores[box_id],
                   f'{box_id}', fontsize=10, fontweight='bold' if box_id in pareto_optimal else 'normal')
        
        # 添加标题和标签
        ax.set_xlabel('收益')
        ax.set_ylabel('夏普比率')
        ax.set_zlabel('稳定性')
        ax.set_title('三维目标空间')
        plt.legend()
        
        # 保存图像
        plt.tight_layout()
        plt.savefig(os.path.join(self.output_dir, output_filename))
        plt.close()
        
        print(f"三维目标空间图已保存到 {os.path.join(self.output_dir, output_filename)}")
    
    def visualize_radar_chart(self, optimization_result: Dict, output_filename: str = "radar_chart.png") -> None:
        """
        可视化雷达图
        
        参数:
            optimization_result: 优化结果
            output_filename: 输出文件名
        """
        profits = optimization_result["profits"]
        risk_measures = optimization_result["risk_measures"]
        stability_scores = optimization_result["stability_scores"]
        pareto_optimal = optimization_result["pareto_optimal"]
        best_box = optimization_result["best_box"]
        
        # 只为帕累托最优点和最佳点创建雷达图
        boxes_to_plot = list(set(pareto_optimal + [best_box]))
        
        # 归一化数据
        normalized_profits = {}
        normalized_risk = {}
        normalized_stability = {}
        
        max_profit = max(profits.values())
        min_profit = min(profits.values())
        profit_range = max_profit - min_profit
        
        max_risk = max([risk_measures[box_id]["sharpe_ratio"] for box_id in boxes_to_plot])
        min_risk = min([risk_measures[box_id]["sharpe_ratio"] for box_id in boxes_to_plot])
        risk_range = max_risk - min_risk
        
        max_stability = max([stability_scores[box_id] for box_id in boxes_to_plot])
        min_stability = min([stability_scores[box_id] for box_id in boxes_to_plot])
        stability_range = max_stability - min_stability
        
        for box_id in boxes_to_plot:
            if profit_range > 0:
                normalized_profits[box_id] = (profits[box_id] - min_profit) / profit_range
            else:
                normalized_profits[box_id] = 0.5
                
            if risk_range > 0:
                normalized_risk[box_id] = (risk_measures[box_id]["sharpe_ratio"] - min_risk) / risk_range
            else:
                normalized_risk[box_id] = 0.5
                
            if stability_range > 0:
                normalized_stability[box_id] = (stability_scores[box_id] - min_stability) / stability_range
            else:
                normalized_stability[box_id] = 0.5
        
        # 添加更多指标
        categories = ['收益', '夏普比率', '稳定性', '下行偏差', '负收益概率']
        
        # 创建图像
        fig, ax = plt.subplots(figsize=(10, 10), subplot_kw=dict(polar=True))
        
        # 设置角度
        angles = np.linspace(0, 2*np.pi, len(categories), endpoint=False).tolist()
        angles += angles[:1]  # 闭合图形
        
        # 设置刻度标签
        ax.set_xticks(angles[:-1])
        ax.set_xticklabels(categories)
        
        # 绘制每个宝箱的雷达图
        for box_id in boxes_to_plot:
            # 为其他指标进行归一化
            downside_deviation = 1 - (risk_measures[box_id]["downside_deviation"] - min([risk_measures[bid]["downside_deviation"] for bid in boxes_to_plot])) / (max([risk_measures[bid]["downside_deviation"] for bid in boxes_to_plot]) - min([risk_measures[bid]["downside_deviation"] for bid in boxes_to_plot]) + 1e-10)
            neg_prob = 1 - (risk_measures[box_id]["prob_negative"] - min([risk_measures[bid]["prob_negative"] for bid in boxes_to_plot])) / (max([risk_measures[bid]["prob_negative"] for bid in boxes_to_plot]) - min([risk_measures[bid]["prob_negative"] for bid in boxes_to_plot]) + 1e-10)
            
            # 收集指标数据
            values = [normalized_profits[box_id], 
                     normalized_risk[box_id], 
                     normalized_stability[box_id], 
                     downside_deviation, 
                     neg_prob]
            values += values[:1]  # 闭合图形
            
            # 绘制雷达图
            ax.plot(angles, values, linewidth=2, label=f'宝箱{box_id}')
            ax.fill(angles, values, alpha=0.1)
        
        # 突出显示最佳宝箱
        best_values = [normalized_profits[best_box], 
                      normalized_risk[best_box], 
                      normalized_stability[best_box],
                      1 - (risk_measures[best_box]["downside_deviation"] - min([risk_measures[bid]["downside_deviation"] for bid in boxes_to_plot])) / (max([risk_measures[bid]["downside_deviation"] for bid in boxes_to_plot]) - min([risk_measures[bid]["downside_deviation"] for bid in boxes_to_plot]) + 1e-10),
                      1 - (risk_measures[best_box]["prob_negative"] - min([risk_measures[bid]["prob_negative"] for bid in boxes_to_plot])) / (max([risk_measures[bid]["prob_negative"] for bid in boxes_to_plot]) - min([risk_measures[bid]["prob_negative"] for bid in boxes_to_plot]) + 1e-10)]
        best_values += best_values[:1]  # 闭合图形
        
        ax.plot(angles, best_values, linewidth=3, color='red', label=f'最佳: 宝箱{best_box}')
        ax.fill(angles, best_values, alpha=0.2, color='red')
        
        # 添加图例
        plt.legend(loc='upper right', bbox_to_anchor=(0.1, 0.1))
        
        # 设置标题
        plt.title('多目标评估雷达图', size=15, y=1.1)
        
        # 保存图像
        plt.tight_layout()
        plt.savefig(os.path.join(self.output_dir, output_filename))
        plt.close()
        
        print(f"雷达图已保存到 {os.path.join(self.output_dir, output_filename)}")
    
    def generate_multi_objective_report(self, optimization_result: Dict, output_filename: str = "multi_objective_report.txt") -> None:
        """
        生成多目标优化报告
        
        参数:
            optimization_result: 优化结果
            output_filename: 输出文件名
        """
        profits = optimization_result["profits"]
        risk_measures = optimization_result["risk_measures"]
        stability_scores = optimization_result["stability_scores"]
        multi_objective_scores = optimization_result["multi_objective_scores"]
        pareto_optimal = optimization_result["pareto_optimal"]
        best_box = optimization_result["best_box"]
        weights = optimization_result["weights"]
        
        report_path = os.path.join(self.output_dir, output_filename)
        
        with open(report_path, 'w', encoding='utf-8') as f:
            f.write("# 多目标优化分析报告\n\n")
            f.write(f"分析日期: {pd.Timestamp.now().strftime('%Y-%m-%d %H:%M:%S')}\n\n")
            
            f.write("## 优化参数\n\n")
            f.write(f"- 理性玩家比例: {self.rational_pct}\n")
            f.write(f"- 启发式玩家比例: {self.heuristic_pct}\n")
            f.write(f"- 随机玩家比例: {1 - self.rational_pct - self.heuristic_pct}\n")
            f.write(f"- 第二宝箱选择比例: {self.second_box_pct}\n")
            f.write(f"- 第二宝箱成本: {self.second_box_cost}\n")
            f.write(f"- 第二选择理性因子: {self.second_choice_rational_factor}\n\n")
            
            f.write("## 目标权重\n\n")
            f.write(f"- 收益权重: {weights['profit']:.2f}\n")
            f.write(f"- 风险权重: {weights['risk']:.2f}\n")
            f.write(f"- 稳定性权重: {weights['stability']:.2f}\n\n")
            
            f.write("## 多目标评分排名\n\n")
            
            # 按多目标评分排序
            sorted_scores = sorted(multi_objective_scores.items(), key=lambda x: x[1], reverse=True)
            
            f.write("| 排名 | 宝箱ID | 综合评分 | 收益 | 夏普比率 | 稳定性 | 帕累托最优 |\n")
            f.write("|------|-------|----------|------|----------|--------|------------|\n")
            
            for rank, (box_id, score) in enumerate(sorted_scores, 1):
                is_pareto = "是" if box_id in pareto_optimal else "否"
                f.write(f"| {rank} | 宝箱{box_id} | {score:.4f} | {profits[box_id]:.2f} | " +
                       f"{risk_measures[box_id]['sharpe_ratio']:.2f} | {stability_scores[box_id]:.2f} | {is_pareto} |\n")
            
            f.write("\n## 帕累托最优解分析\n\n")
            
            f.write(f"共找到{len(pareto_optimal)}个帕累托最优解: " + 
                   ", ".join([f"宝箱{box_id}" for box_id in pareto_optimal]) + "\n\n")
            
            f.write("### 帕累托最优解详细比较\n\n")
            
            f.write("| 宝箱ID | 收益 | 夏普比率 | 稳定性 | 综合评分 | 排名 |\n")
            f.write("|--------|------|----------|--------|----------|------|\n")
            
            for box_id in pareto_optimal:
                rank = [b[0] for b in sorted_scores].index(box_id) + 1
                f.write(f"| 宝箱{box_id} | {profits[box_id]:.2f} | {risk_measures[box_id]['sharpe_ratio']:.2f} | " +
                       f"{stability_scores[box_id]:.2f} | {multi_objective_scores[box_id]:.4f} | {rank} |\n")
            
            f.write("\n## 最优选择分析\n\n")
            
            f.write(f"### 最佳综合选择: 宝箱{best_box}\n\n")
            
            # 详细分析最佳选择
            f.write(f"- 收益: {profits[best_box]:.2f}\n")
            f.write(f"- 夏普比率: {risk_measures[best_box]['sharpe_ratio']:.2f}\n")
            f.write(f"- 索提诺比率: {risk_measures[best_box]['sortino_ratio']:.2f}\n")
            f.write(f"- 下行偏差: {risk_measures[best_box]['downside_deviation']:.2f}\n")
            f.write(f"- 负收益概率: {risk_measures[best_box]['prob_negative']:.2%}\n")
            f.write(f"- 稳定性评分: {stability_scores[best_box]:.2f}\n")
            f.write(f"- 综合评分: {multi_objective_scores[best_box]:.4f}\n\n")
            
            # 针对不同风险偏好的建议
            f.write("## 不同风险偏好的建议\n\n")
            
            # 保守型投资者 (低风险优先)
            conservative_weights = {"profit": 0.3, "risk": 0.6, "stability": 0.1}
            conservative_scores = self.calculate_multi_objective_scores(
                profits=profits,
                risk_measures=risk_measures,
                stability_scores=stability_scores,
                weights=conservative_weights
            )
            conservative_best = max(conservative_scores.items(), key=lambda x: x[1])[0]
            
            f.write("### 保守型投资者 (低风险优先)\n\n")
            f.write(f"建议选择: 宝箱{conservative_best}\n")
            f.write(f"- 收益: {profits[conservative_best]:.2f}\n")
            f.write(f"- 夏普比率: {risk_measures[conservative_best]['sharpe_ratio']:.2f}\n")
            f.write(f"- 负收益概率: {risk_measures[conservative_best]['prob_negative']:.2%}\n")
            f.write(f"- 稳定性评分: {stability_scores[conservative_best]:.2f}\n\n")
            
            # 平衡型投资者 (平衡风险与收益)
            balanced_weights = {"profit": 0.5, "risk": 0.4, "stability": 0.1}
            balanced_scores = self.calculate_multi_objective_scores(
                profits=profits,
                risk_measures=risk_measures,
                stability_scores=stability_scores,
                weights=balanced_weights
            )
            balanced_best = max(balanced_scores.items(), key=lambda x: x[1])[0]
            
            f.write("### 平衡型投资者 (平衡风险与收益)\n\n")
            f.write(f"建议选择: 宝箱{balanced_best}\n")
            f.write(f"- 收益: {profits[balanced_best]:.2f}\n")
            f.write(f"- 夏普比率: {risk_measures[balanced_best]['sharpe_ratio']:.2f}\n")
            f.write(f"- 负收益概率: {risk_measures[balanced_best]['prob_negative']:.2%}\n")
            f.write(f"- 稳定性评分: {stability_scores[balanced_best]:.2f}\n\n")
            
            # 进取型投资者 (高收益优先)
            aggressive_weights = {"profit": 0.7, "risk": 0.2, "stability": 0.1}
            aggressive_scores = self.calculate_multi_objective_scores(
                profits=profits,
                risk_measures=risk_measures,
                stability_scores=stability_scores,
                weights=aggressive_weights
            )
            aggressive_best = max(aggressive_scores.items(), key=lambda x: x[1])[0]
            
            f.write("### 进取型投资者 (高收益优先)\n\n")
            f.write(f"建议选择: 宝箱{aggressive_best}\n")
            f.write(f"- 收益: {profits[aggressive_best]:.2f}\n")
            f.write(f"- 夏普比率: {risk_measures[aggressive_best]['sharpe_ratio']:.2f}\n")
            f.write(f"- 负收益概率: {risk_measures[aggressive_best]['prob_negative']:.2%}\n")
            f.write(f"- 稳定性评分: {stability_scores[aggressive_best]:.2f}\n\n")
            
            # 结论
            f.write("## 结论与建议\n\n")
            
            if best_box == conservative_best == balanced_best == aggressive_best:
                f.write(f"综合分析表明，宝箱{best_box}在各种风险偏好下都是最优选择，" +
                       f"显示出其卓越的综合表现和稳健性。\n\n")
            else:
                f.write("不同风险偏好下的最优选择各不相同，表明在选择宝箱时应根据个人风险偏好进行决策。\n\n")
            
            # 对比单一目标和多目标
            single_objective_best = max(profits.items(), key=lambda x: x[1])[0]
            
            if best_box == single_objective_best:
                f.write(f"多目标优化结果与单纯追求最高收益的策略一致，都选择宝箱{best_box}。" +
                       f"这表明宝箱{best_box}不仅收益最高，在风险和稳定性方面也表现良好。\n")
            else:
                f.write(f"多目标优化结果(宝箱{best_box})与单纯追求最高收益的策略(宝箱{single_objective_best})不同。" +
                       f"虽然宝箱{single_objective_best}收益略高({profits[single_objective_best]:.2f} vs {profits[best_box]:.2f})，" +
                       f"但宝箱{best_box}在风险控制和稳定性方面表现更佳，整体更为平衡。\n")
        
        print(f"多目标优化报告已保存到 {report_path}")
    
    def run_full_analysis(self, num_simulations: int = 500, user_weights: Dict = None) -> Dict:
        """
        运行完整的多目标优化分析
        
        参数:
            num_simulations: 模拟次数
            user_weights: 用户权重
            
        返回:
            分析结果字典
        """
        print(f"开始运行多目标优化分析 (模拟次数: {num_simulations})...")
        
        # 运行优化
        optimization_result = self.run_optimization(num_simulations, user_weights)
        
        # 可视化帕累托前沿
        self.visualize_pareto_front(optimization_result)
        
        # 可视化三维目标空间
        self.visualize_3d_objective_space(optimization_result)
        
        # 可视化雷达图
        self.visualize_radar_chart(optimization_result)
        
        # 生成报告
        self.generate_multi_objective_report(optimization_result)
        
        return optimization_result


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
    
    # 创建多目标优化器
    optimizer = MultiObjectiveOptimizer(
        treasures=treasures,
        output_dir="output/multi_objective",
        rational_pct=0.35,
        heuristic_pct=0.45,
        second_box_pct=0.05,
        second_box_cost=50000,
        second_choice_rational_factor=0.7
    )
    
    # 运行完整分析
    # 设置用户权重
    user_weights = {
        "profit": 0.6,    # 60%权重给收益
        "risk": 0.3,      # 30%权重给风险
        "stability": 0.1  # 10%权重给稳定性
    }
    
    optimizer.run_full_analysis(
        num_simulations=200,  # 为了快速测试，使用较少的模拟次数
        user_weights=user_weights
    )


if __name__ == "__main__":
    main() 