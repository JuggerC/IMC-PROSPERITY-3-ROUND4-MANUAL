#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
多维度风险分析模块
分析不同宝箱选择的风险特性
"""

import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from typing import List, Dict, Tuple, Optional, Union
from simulator import Treasure, TreasureOptimizer

class RiskAnalyzer:
    """多维度风险分析器"""
    
    def __init__(self, treasures: List[Treasure], 
                 output_dir: str = "output",
                 rational_pct: float = 0.35, 
                 heuristic_pct: float = 0.45,
                 second_box_pct: float = 0.05,
                 second_choice_rational_factor: float = 0.7,
                 risk_free_rate: float = 0.01,
                 confidence_level: float = 0.95):
        """
        初始化风险分析器
        
        参数:
            treasures: 宝箱列表
            output_dir: 输出目录
            rational_pct: 理性玩家比例
            heuristic_pct: 启发式玩家比例
            second_box_pct: 选择第二个宝箱的玩家比例
            second_choice_rational_factor: 第二选择理性因子
            risk_free_rate: 无风险利率(用于计算夏普比率)
            confidence_level: 风险计算的置信水平
        """
        self.treasures = treasures
        self.rational_pct = rational_pct
        self.heuristic_pct = heuristic_pct
        self.second_box_pct = second_box_pct
        self.second_choice_rational_factor = second_choice_rational_factor
        self.risk_free_rate = risk_free_rate
        self.confidence_level = confidence_level
        
        # 创建输出目录
        self.output_dir = output_dir
        os.makedirs(output_dir, exist_ok=True)
        
        # 初始化TreasureOptimizer
        self.optimizer = TreasureOptimizer(
            treasures=self.treasures,
            rational_pct=self.rational_pct,
            heuristic_pct=self.heuristic_pct,
            second_box_pct=self.second_box_pct,
            second_choice_rational_factor=self.second_choice_rational_factor
        )
    
    def calculate_standard_deviation(self, profits: List[float]) -> float:
        """
        计算收益的标准差
        
        参数:
            profits: 收益列表
            
        返回:
            标准差
        """
        return np.std(profits)
    
    def calculate_sharpe_ratio(self, profits: List[float]) -> float:
        """
        计算夏普比率(Sharpe Ratio)
        夏普比率 = (平均收益 - 无风险利率) / 收益标准差
        
        参数:
            profits: 收益列表
            
        返回:
            夏普比率
        """
        mean_profit = np.mean(profits)
        std_profit = np.std(profits)
        
        # 防止除以零
        if std_profit == 0:
            return float('inf') if mean_profit > self.risk_free_rate else float('-inf')
            
        return (mean_profit - self.risk_free_rate) / std_profit
    
    def calculate_downside_deviation(self, profits: List[float], target_return: float = 0) -> float:
        """
        计算下行偏差(Downside Deviation)
        只考虑低于目标收益的部分
        
        参数:
            profits: 收益列表
            target_return: 目标收益(默认为0)
            
        返回:
            下行偏差
        """
        # 计算低于目标收益的差值
        downside_returns = [min(0, r - target_return) for r in profits]
        
        # 计算下行差值的均方根
        return np.sqrt(np.mean(np.square(downside_returns)))
    
    def calculate_sortino_ratio(self, profits: List[float], target_return: float = 0) -> float:
        """
        计算索提诺比率(Sortino Ratio)
        索提诺比率 = (平均收益 - 无风险利率) / 下行偏差
        
        参数:
            profits: 收益列表
            target_return: 目标收益(默认为0)
            
        返回:
            索提诺比率
        """
        mean_profit = np.mean(profits)
        downside_dev = self.calculate_downside_deviation(profits, target_return)
        
        # 防止除以零
        if downside_dev == 0:
            return float('inf') if mean_profit > self.risk_free_rate else float('-inf')
            
        return (mean_profit - self.risk_free_rate) / downside_dev
    
    def calculate_max_drawdown(self, profits: List[float]) -> float:
        """
        计算最大回撤(Maximum Drawdown)
        最大回撤衡量从高点到后续低点的最大损失百分比
        
        参数:
            profits: 收益列表
            
        返回:
            最大回撤(百分比)
        """
        # 计算累积收益
        cumulative = np.cumsum(profits)
        
        # 计算当前累积收益的最大值
        max_so_far = np.maximum.accumulate(cumulative)
        
        # 计算回撤
        drawdowns = (max_so_far - cumulative) / np.maximum(max_so_far, 1)  # 防止除以零
        
        # 返回最大回撤
        return np.max(drawdowns) if len(drawdowns) > 0 else 0
    
    def calculate_var(self, profits: List[float], confidence_level: float = None) -> float:
        """
        计算风险价值(Value at Risk, VaR)
        在给定置信水平下，可能的最大损失
        
        参数:
            profits: 收益列表
            confidence_level: 置信水平(默认使用类初始化时设置的值)
            
        返回:
            风险价值
        """
        if confidence_level is None:
            confidence_level = self.confidence_level
            
        # 排序收益
        sorted_profits = np.sort(profits)
        
        # 计算分位数位置
        index = int(np.floor((1 - confidence_level) * len(sorted_profits)))
        
        # 返回风险价值
        return -sorted_profits[index]
    
    def calculate_cvar(self, profits: List[float], confidence_level: float = None) -> float:
        """
        计算条件风险价值(Conditional Value at Risk, CVaR)
        超过VaR的平均损失
        
        参数:
            profits: 收益列表
            confidence_level: 置信水平(默认使用类初始化时设置的值)
            
        返回:
            条件风险价值
        """
        if confidence_level is None:
            confidence_level = self.confidence_level
            
        # 计算VaR
        var = self.calculate_var(profits, confidence_level)
        
        # 找出超过VaR的所有损失
        tail_losses = [p for p in profits if p <= -var]
        
        # 如果没有超过VaR的损失，返回VaR
        if not tail_losses:
            return var
            
        # 返回平均尾部损失的负值
        return -np.mean(tail_losses)
    
    def calculate_risk_measures(self, box_id: int, num_simulations: int = 1000) -> Dict:
        """
        计算指定宝箱的所有风险指标
        
        参数:
            box_id: 宝箱ID
            num_simulations: 模拟次数
            
        返回:
            包含风险指标的字典
        """
        print(f"计算宝箱{box_id}的风险指标...")
        
        # 运行多次模拟以获取收益分布
        profits = []
        
        for _ in range(num_simulations):
            # 运行一次迭代
            distribution, iteration_profits = self.optimizer.run_iteration(print_progress=False)
            
            # 收集收益
            profits.append(iteration_profits[box_id])
        
        # 计算各项风险指标
        mean_profit = np.mean(profits)
        std_dev = self.calculate_standard_deviation(profits)
        sharpe = self.calculate_sharpe_ratio(profits)
        downside_dev = self.calculate_downside_deviation(profits)
        sortino = self.calculate_sortino_ratio(profits)
        max_dd = self.calculate_max_drawdown(profits)
        var_95 = self.calculate_var(profits, 0.95)
        cvar_95 = self.calculate_cvar(profits, 0.95)
        
        # 计算收益负值的概率
        prob_negative = len([p for p in profits if p < 0]) / len(profits)
        
        # 计算百分位数
        percentiles = {
            "p10": np.percentile(profits, 10),
            "p25": np.percentile(profits, 25),
            "p50": np.percentile(profits, 50),  # 中位数
            "p75": np.percentile(profits, 75),
            "p90": np.percentile(profits, 90)
        }
        
        # 返回所有风险指标
        return {
            "box_id": box_id,
            "mean_profit": mean_profit,
            "std_dev": std_dev,
            "sharpe_ratio": sharpe,
            "downside_deviation": downside_dev,
            "sortino_ratio": sortino,
            "max_drawdown": max_dd,
            "var_95": var_95,
            "cvar_95": cvar_95,
            "prob_negative": prob_negative,
            "percentiles": percentiles,
            "simulation_profits": profits  # 保存原始模拟数据
        }
    
    def analyze_all_boxes(self, num_simulations: int = 1000) -> Dict[int, Dict]:
        """
        分析所有宝箱的风险
        
        参数:
            num_simulations: 每个宝箱的模拟次数
            
        返回:
            按宝箱ID索引的风险指标字典
        """
        risk_measures = {}
        
        for treasure in self.treasures:
            box_id = treasure.box_id
            risk_measures[box_id] = self.calculate_risk_measures(box_id, num_simulations)
            
        return risk_measures
        
    def calculate_risk_score(self, risk_measures: Dict, 
                           risk_weights: Dict = None, 
                           risk_preference: float = 0.5) -> Dict[int, float]:
        """
        计算每个宝箱的风险评分
        风险偏好参数: 0表示极度风险厌恶, 1表示极度风险追求
        
        参数:
            risk_measures: 风险指标字典
            risk_weights: 各风险指标的权重字典
            risk_preference: 风险偏好参数(0-1)
            
        返回:
            宝箱风险评分字典
        """
        if risk_weights is None:
            # 默认风险指标权重
            risk_weights = {
                "mean_profit": 0.25,           # 平均收益
                "sharpe_ratio": 0.20,          # 夏普比率
                "sortino_ratio": 0.15,         # 索提诺比率
                "prob_negative": -0.15,        # 负收益概率(负权重)
                "std_dev": -0.10,              # 标准差(负权重)
                "max_drawdown": -0.05,         # 最大回撤(负权重)
                "var_95": -0.05,               # VaR(负权重)
                "cvar_95": -0.05,              # CVaR(负权重)
            }
        
        # 归一化函数
        def normalize(values, is_higher_better=True):
            min_val = min(values)
            max_val = max(values)
            if min_val == max_val:
                return [0.5] * len(values)
            
            if is_higher_better:
                return [(v - min_val) / (max_val - min_val) for v in values]
            else:
                return [(max_val - v) / (max_val - min_val) for v in values]
        
        # 收集每个指标的值
        metric_values = {metric: [] for metric in risk_weights.keys()}
        box_ids = []
        
        for box_id, measures in risk_measures.items():
            box_ids.append(box_id)
            for metric in risk_weights.keys():
                metric_values[metric].append(measures[metric])
        
        # 归一化每个指标
        normalized_values = {}
        for metric, values in metric_values.items():
            # 对于负权重的指标，越低越好
            is_higher_better = risk_weights[metric] > 0
            normalized_values[metric] = normalize(values, is_higher_better)
        
        # 计算每个宝箱的评分
        scores = {}
        for i, box_id in enumerate(box_ids):
            # 风险调整的评分
            risk_components = 0
            return_components = 0
            
            for metric, norm_values in normalized_values.items():
                weight = abs(risk_weights[metric])
                value = norm_values[i]
                
                # 区分收益和风险指标
                if risk_weights[metric] > 0:  # 收益类指标
                    return_components += value * weight
                else:  # 风险类指标
                    risk_components += value * weight
            
            # 总评分 = 风险偏好 * 收益评分 + (1-风险偏好) * 风险评分
            total_score = (risk_preference * return_components + 
                           (1 - risk_preference) * risk_components)
            
            scores[box_id] = total_score
        
        return scores
        
    def visualize_risk_heatmap(self, risk_measures: Dict, output_filename: str = "risk_heatmap.png") -> None:
        """
        创建风险指标热图
        
        参数:
            risk_measures: 风险指标字典
            output_filename: 输出文件名
        """
        # 提取需要可视化的指标
        metrics = [
            "mean_profit", "std_dev", "sharpe_ratio", "sortino_ratio", 
            "max_drawdown", "var_95", "cvar_95", "prob_negative"
        ]
        
        # 为热图准备数据
        data = []
        box_ids = []
        
        for box_id, measures in risk_measures.items():
            box_ids.append(f"宝箱{box_id}")
            row = [measures[metric] for metric in metrics]
            data.append(row)
        
        # 创建数据框
        df = pd.DataFrame(data, index=box_ids, columns=metrics)
        
        # 归一化数据以便更好地可视化
        for col in df.columns:
            if col in ["mean_profit", "sharpe_ratio", "sortino_ratio"]:
                # 这些指标越高越好
                df[col] = (df[col] - df[col].min()) / (df[col].max() - df[col].min())
            else:
                # 这些指标越低越好
                df[col] = 1 - (df[col] - df[col].min()) / (df[col].max() - df[col].min())
        
        # 设置更友好的中文列名
        column_mapping = {
            "mean_profit": "平均收益",
            "std_dev": "波动性",
            "sharpe_ratio": "夏普比率",
            "sortino_ratio": "索提诺比率",
            "max_drawdown": "最大回撤",
            "var_95": "风险价值",
            "cvar_95": "条件风险价值",
            "prob_negative": "负收益概率"
        }
        df.columns = [column_mapping[col] for col in df.columns]
        
        # 创建热图
        plt.figure(figsize=(12, 8))
        
        # 使用自定义配色方案
        cmap = sns.diverging_palette(10, 133, as_cmap=True)
        
        # 绘制热图
        sns.heatmap(df, annot=True, cmap=cmap, fmt=".2f", linewidths=0.5, cbar_kws={"label": "归一化分数"})
        
        plt.title("宝箱风险指标热图", fontsize=16, pad=20)
        plt.xlabel("风险指标", fontsize=12)
        plt.ylabel("宝箱", fontsize=12)
        
        # 保存图像
        plt.tight_layout()
        plt.savefig(os.path.join(self.output_dir, output_filename))
        plt.close()
        
        print(f"风险热图已保存到 {os.path.join(self.output_dir, output_filename)}")
    
    def visualize_profit_distributions(self, risk_measures: Dict, output_filename: str = "profit_distributions.png") -> None:
        """
        可视化每个宝箱的收益分布
        
        参数:
            risk_measures: 风险指标字典
            output_filename: 输出文件名
        """
        # 确定绘图的行列数
        n_boxes = len(risk_measures)
        n_cols = min(3, n_boxes)
        n_rows = (n_boxes + n_cols - 1) // n_cols
        
        # 创建图像
        fig, axes = plt.subplots(n_rows, n_cols, figsize=(15, n_rows * 4))
        
        # 如果只有一行或一列，确保axes是二维的
        if n_rows == 1:
            axes = np.array([axes])
        if n_cols == 1:
            axes = axes.reshape(-1, 1)
        
        # 按照宝箱ID排序
        sorted_box_ids = sorted(risk_measures.keys())
        
        for i, box_id in enumerate(sorted_box_ids):
            row, col = i // n_cols, i % n_cols
            ax = axes[row, col]
            
            # 获取该宝箱的收益数据
            profits = risk_measures[box_id]["simulation_profits"]
            mean_profit = risk_measures[box_id]["mean_profit"]
            var_95 = risk_measures[box_id]["var_95"]
            
            # 绘制直方图
            sns.histplot(profits, kde=True, ax=ax, bins=30, color='skyblue')
            
            # 添加均值线
            ax.axvline(mean_profit, color='g', linestyle='--', label=f'平均: {mean_profit:.2f}')
            
            # 添加VaR线
            ax.axvline(-var_95, color='r', linestyle='--', label=f'95% VaR: {var_95:.2f}')
            
            # 添加零线
            ax.axvline(0, color='k', linestyle='-', alpha=0.3)
            
            # 设置标题和标签
            ax.set_title(f'宝箱{box_id}收益分布')
            ax.set_xlabel('收益')
            ax.set_ylabel('频率')
            ax.legend()
            
            # 添加统计信息
            stats_text = (f"平均值: {mean_profit:.2f}\n"
                          f"标准差: {risk_measures[box_id]['std_dev']:.2f}\n"
                          f"夏普比率: {risk_measures[box_id]['sharpe_ratio']:.2f}\n"
                          f"负收益概率: {risk_measures[box_id]['prob_negative']:.2%}")
            
            ax.text(0.95, 0.95, stats_text, transform=ax.transAxes, 
                    verticalalignment='top', horizontalalignment='right',
                    bbox=dict(boxstyle='round', facecolor='white', alpha=0.7))
        
        # 处理多余的子图
        for i in range(len(sorted_box_ids), n_rows * n_cols):
            row, col = i // n_cols, i % n_cols
            fig.delaxes(axes[row, col])
        
        # 调整布局并保存
        plt.tight_layout()
        plt.savefig(os.path.join(self.output_dir, output_filename))
        plt.close()
        
        print(f"收益分布图已保存到 {os.path.join(self.output_dir, output_filename)}")
    
    def visualize_risk_return_plot(self, risk_measures: Dict, 
                                output_filename: str = "risk_return_plot.png", 
                                x_metric: str = "std_dev", 
                                y_metric: str = "mean_profit") -> None:
        """
        创建风险-收益散点图
        
        参数:
            risk_measures: 风险指标字典
            output_filename: 输出文件名
            x_metric: X轴指标(默认为标准差)
            y_metric: Y轴指标(默认为平均收益)
        """
        # 准备数据
        x_values = []
        y_values = []
        box_ids = []
        colors = []
        sizes = []
        
        # 设置指标的展示名称
        metric_names = {
            "mean_profit": "平均收益",
            "std_dev": "标准差",
            "sharpe_ratio": "夏普比率",
            "sortino_ratio": "索提诺比率",
            "max_drawdown": "最大回撤",
            "var_95": "95% VaR",
            "cvar_95": "95% CVaR",
            "prob_negative": "负收益概率"
        }
        
        for box_id, measures in risk_measures.items():
            x_values.append(measures[x_metric])
            y_values.append(measures[y_metric])
            box_ids.append(box_id)
            
            # 使用夏普比率决定颜色
            sharpe = measures["sharpe_ratio"]
            colors.append(sharpe)
            
            # 使用收益绝对值决定大小
            sizes.append(abs(measures["mean_profit"]) * 100)
        
        plt.figure(figsize=(10, 8))
        
        # 创建散点图
        scatter = plt.scatter(x_values, y_values, c=colors, s=sizes, alpha=0.7, 
                             cmap='coolwarm', edgecolors='black', linewidths=1)
        
        # 添加宝箱标签
        for i, box_id in enumerate(box_ids):
            plt.annotate(f'{box_id}', (x_values[i], y_values[i]),
                        xytext=(5, 5), textcoords='offset points',
                        fontsize=12, fontweight='bold')
        
        # 添加色条
        cbar = plt.colorbar(scatter)
        cbar.set_label('夏普比率', fontsize=12)
        
        # 设置轴标签和标题
        plt.xlabel(metric_names.get(x_metric, x_metric), fontsize=12)
        plt.ylabel(metric_names.get(y_metric, y_metric), fontsize=12)
        plt.title('宝箱风险-收益分析', fontsize=16)
        
        # 添加网格线
        plt.grid(True, linestyle='--', alpha=0.7)
        
        # 保存图像
        plt.tight_layout()
        plt.savefig(os.path.join(self.output_dir, output_filename))
        plt.close()
        
        print(f"风险-收益图已保存到 {os.path.join(self.output_dir, output_filename)}")
    
    def generate_risk_report(self, risk_measures: Dict, risk_scores: Dict, output_filename: str = "risk_analysis_report.txt") -> None:
        """
        生成风险分析报告
        
        参数:
            risk_measures: 风险指标字典
            risk_scores: 风险评分字典
            output_filename: 输出文件名
        """
        # 创建报告文件
        report_path = os.path.join(self.output_dir, output_filename)
        
        with open(report_path, 'w', encoding='utf-8') as f:
            f.write("# 宝箱风险分析报告\n\n")
            f.write(f"分析日期: {pd.Timestamp.now().strftime('%Y-%m-%d %H:%M:%S')}\n\n")
            
            f.write("## 分析参数\n\n")
            f.write(f"- 理性玩家比例: {self.rational_pct}\n")
            f.write(f"- 启发式玩家比例: {self.heuristic_pct}\n")
            f.write(f"- 随机玩家比例: {1 - self.rational_pct - self.heuristic_pct}\n")
            f.write(f"- 第二宝箱选择比例: {self.second_box_pct}\n")
            f.write(f"- 第二选择理性因子: {self.second_choice_rational_factor}\n")
            f.write(f"- 无风险利率: {self.risk_free_rate}\n")
            f.write(f"- 风险计算置信水平: {self.confidence_level}\n\n")
            
            f.write("## 风险评分排名\n\n")
            
            # 按风险评分排序宝箱
            sorted_boxes = sorted(risk_scores.items(), key=lambda x: x[1], reverse=True)
            
            # 写入表格标题
            f.write("| 排名 | 宝箱ID | 风险评分 | 平均收益 | 标准差 | 夏普比率 | 负收益概率 | 95% VaR |\n")
            f.write("|------|-------|----------|----------|--------|----------|------------|--------|\n")
            
            # 写入表格内容
            for rank, (box_id, score) in enumerate(sorted_boxes, 1):
                measures = risk_measures[box_id]
                f.write(f"| {rank} | 宝箱{box_id} | {score:.4f} | {measures['mean_profit']:.2f} | " +
                        f"{measures['std_dev']:.2f} | {measures['sharpe_ratio']:.2f} | " +
                        f"{measures['prob_negative']:.2%} | {measures['var_95']:.2f} |\n")
            
            f.write("\n## 宝箱详细风险指标\n\n")
            
            # 对每个宝箱写入详细风险指标
            for box_id, measures in risk_measures.items():
                f.write(f"### 宝箱{box_id}\n\n")
                
                f.write("#### 收益指标\n\n")
                f.write(f"- 平均收益: {measures['mean_profit']:.2f}\n")
                f.write(f"- 收益中位数: {measures['percentiles']['p50']:.2f}\n")
                f.write(f"- 收益范围: [{measures['percentiles']['p10']:.2f}, {measures['percentiles']['p90']:.2f}] (10%-90%分位数)\n\n")
                
                f.write("#### 风险指标\n\n")
                f.write(f"- 标准差: {measures['std_dev']:.2f}\n")
                f.write(f"- 下行偏差: {measures['downside_deviation']:.2f}\n")
                f.write(f"- 最大回撤: {measures['max_drawdown']:.2%}\n")
                f.write(f"- 95% VaR: {measures['var_95']:.2f}\n")
                f.write(f"- 95% CVaR: {measures['cvar_95']:.2f}\n")
                f.write(f"- 负收益概率: {measures['prob_negative']:.2%}\n\n")
                
                f.write("#### 风险调整后收益\n\n")
                f.write(f"- 夏普比率: {measures['sharpe_ratio']:.2f}\n")
                f.write(f"- 索提诺比率: {measures['sortino_ratio']:.2f}\n\n")
                
                f.write("#### 综合评分\n\n")
                f.write(f"- 风险评分: {risk_scores[box_id]:.4f}\n")
                f.write(f"- 评分排名: {[b[0] for b in sorted_boxes].index(box_id) + 1}/{len(sorted_boxes)}\n\n")
            
            f.write("## 风险分析结论\n\n")
            
            # 获取最佳宝箱
            best_box = sorted_boxes[0][0]
            best_measures = risk_measures[best_box]
            
            f.write(f"基于综合风险评分，**宝箱{best_box}**是最佳选择，其综合评分为{sorted_boxes[0][1]:.4f}。\n\n")
            
            # 根据平均收益排序
            by_profit = sorted(risk_measures.items(), key=lambda x: x[1]['mean_profit'], reverse=True)
            f.write(f"从纯收益角度，**宝箱{by_profit[0][0]}**提供最高平均收益({by_profit[0][1]['mean_profit']:.2f})。\n")
            
            # 根据夏普比率排序
            by_sharpe = sorted(risk_measures.items(), key=lambda x: x[1]['sharpe_ratio'], reverse=True)
            f.write(f"从风险调整后收益角度，**宝箱{by_sharpe[0][0]}**提供最高夏普比率({by_sharpe[0][1]['sharpe_ratio']:.2f})。\n")
            
            # 根据负收益概率排序
            by_neg_prob = sorted(risk_measures.items(), key=lambda x: x[1]['prob_negative'])
            f.write(f"**宝箱{by_neg_prob[0][0]}**具有最低的负收益概率({by_neg_prob[0][1]['prob_negative']:.2%})。\n\n")
            
            # 为不同风险偏好提供建议
            f.write("### 不同风险偏好的建议\n\n")
            
            # 保守型投资者
            conservative = sorted(risk_measures.items(), key=lambda x: x[1]['sharpe_ratio'] * (1 - x[1]['prob_negative']), reverse=True)
            f.write(f"- **保守型投资者**应选择**宝箱{conservative[0][0]}**，其在保持较高收益的同时最小化风险。\n")
            
            # 平衡型投资者
            balanced = sorted_boxes[0]  # 使用综合评分
            f.write(f"- **平衡型投资者**应选择**宝箱{balanced[0]}**，其提供最佳的风险-收益平衡。\n")
            
            # 进取型投资者
            aggressive = sorted(risk_measures.items(), key=lambda x: x[1]['mean_profit'] * x[1]['sortino_ratio'], reverse=True)
            f.write(f"- **进取型投资者**应选择**宝箱{aggressive[0][0]}**，其在可接受的下行风险条件下最大化收益。\n\n")
            
            f.write("### 风险多样化建议\n\n")
            
            # 计算相关性矩阵来找出分散风险的组合
            correlation_data = {}
            box_ids = list(risk_measures.keys())
            
            for i, box1 in enumerate(box_ids):
                profits1 = risk_measures[box1]["simulation_profits"]
                
                for box2 in box_ids[i+1:]:
                    profits2 = risk_measures[box2]["simulation_profits"]
                    
                    # 计算相关系数
                    corr = np.corrcoef(profits1, profits2)[0, 1]
                    
                    # 存储相关系数
                    if box1 not in correlation_data:
                        correlation_data[box1] = {}
                    correlation_data[box1][box2] = corr
            
            # 寻找低相关性的宝箱对
            low_corr_pairs = []
            
            for box1, correlations in correlation_data.items():
                for box2, corr in correlations.items():
                    if corr < 0.5:  # 可以调整这个阈值
                        low_corr_pairs.append((box1, box2, corr))
            
            low_corr_pairs.sort(key=lambda x: x[2])
            
            if low_corr_pairs:
                f.write("以下宝箱对具有较低的收益相关性，可以考虑组合以降低整体风险：\n\n")
                
                for box1, box2, corr in low_corr_pairs[:3]:  # 显示前3个低相关性对
                    f.write(f"- 宝箱{box1} + 宝箱{box2} (相关系数: {corr:.2f})\n")
            else:
                f.write("分析未发现明显的低相关性宝箱对，建议仍然专注于单一最优宝箱选择。\n")
        
        print(f"风险分析报告已保存到 {report_path}")
    
    def run_full_analysis(self, num_simulations: int = 1000, risk_preference: float = 0.5) -> Dict:
        """
        运行完整的风险分析
        
        参数:
            num_simulations: 每个宝箱的模拟次数
            risk_preference: 风险偏好参数(0-1)
            
        返回:
            分析结果字典
        """
        print(f"开始运行全面风险分析 (模拟次数: {num_simulations}, 风险偏好: {risk_preference})...")
        
        # 分析所有宝箱的风险
        risk_measures = self.analyze_all_boxes(num_simulations)
        
        # 计算风险评分
        risk_scores = self.calculate_risk_score(risk_measures, risk_preference=risk_preference)
        
        # 可视化风险热图
        self.visualize_risk_heatmap(risk_measures)
        
        # 可视化收益分布
        self.visualize_profit_distributions(risk_measures)
        
        # 可视化风险-收益图
        self.visualize_risk_return_plot(risk_measures)
        
        # 生成风险报告
        self.generate_risk_report(risk_measures, risk_scores)
        
        # 返回分析结果
        return {
            "risk_measures": risk_measures,
            "risk_scores": risk_scores
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
    
    # 创建风险分析器
    analyzer = RiskAnalyzer(
        treasures=treasures,
        rational_pct=0.35,
        heuristic_pct=0.45,
        second_box_pct=0.05,
        second_choice_rational_factor=0.7,
        risk_free_rate=0.01,
        confidence_level=0.95
    )
    
    # 运行完整分析
    analyzer.run_full_analysis(
        num_simulations=500,  # 为了快速测试，使用较少的模拟次数
        risk_preference=0.5   # 平衡的风险偏好
    )


if __name__ == "__main__":
    main() 