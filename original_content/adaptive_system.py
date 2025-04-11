#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
实时自适应决策系统
根据实时反馈调整策略和参数
"""

import os
import random
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from typing import List, Dict, Tuple, Optional, Union, Any
from simulator import Treasure, TreasureOptimizer

class AdaptiveDecisionSystem:
    """实时自适应决策系统"""
    
    def __init__(self, treasures: List[Treasure], 
                 output_dir: str = "output/adaptive",
                 initial_rational_pct: float = 0.35, 
                 initial_heuristic_pct: float = 0.45,
                 initial_second_box_pct: float = 0.05,
                 initial_second_box_cost: int = 50000,
                 initial_second_choice_rational_factor: float = 0.7,
                 adaptation_rate: float = 0.1,
                 exploration_probability: float = 0.2):
        """
        初始化自适应决策系统
        
        参数:
            treasures: 宝箱列表
            output_dir: 输出目录
            initial_rational_pct: 初始理性玩家比例
            initial_heuristic_pct: 初始启发式玩家比例
            initial_second_box_pct: 初始第二宝箱选择比例
            initial_second_box_cost: 初始第二宝箱成本
            initial_second_choice_rational_factor: 初始第二选择理性因子
            adaptation_rate: 适应率 (参数调整速度)
            exploration_probability: 探索概率 (尝试新策略的概率)
        """
        self.treasures = treasures
        self.output_dir = output_dir
        self.initial_rational_pct = initial_rational_pct
        self.initial_heuristic_pct = initial_heuristic_pct
        self.initial_second_box_pct = initial_second_box_pct
        self.initial_second_box_cost = initial_second_box_cost
        self.initial_second_choice_rational_factor = initial_second_choice_rational_factor
        self.adaptation_rate = adaptation_rate
        self.exploration_probability = exploration_probability
        
        # 创建输出目录
        os.makedirs(output_dir, exist_ok=True)
        
        # 当前参数
        self.current_parameters = {
            "rational_pct": initial_rational_pct,
            "heuristic_pct": initial_heuristic_pct,
            "second_box_pct": initial_second_box_pct,
            "second_box_cost": initial_second_box_cost,
            "second_choice_rational_factor": initial_second_choice_rational_factor
        }
        
        # 当前策略
        self.current_strategy = None
        
        # 反馈历史
        self.feedback_history = []
        
        # 参数历史
        self.parameter_history = [self.current_parameters.copy()]
        
        # 策略历史
        self.strategy_history = []
        
        # 收益历史
        self.profit_history = []
        
        # 绩效评估指标
        self.performance_metrics = {
            "cumulative_reward": 0,
            "average_reward": 0,
            "reward_variance": 0,
            "best_reward": float('-inf'),
            "worst_reward": float('inf')
        }
    
    def initialize_strategy(self) -> int:
        """
        初始化策略
        
        返回:
            初始宝箱选择
        """
        # 创建优化器
        optimizer = TreasureOptimizer(
            treasures=self.treasures,
            rational_pct=self.current_parameters["rational_pct"],
            heuristic_pct=self.current_parameters["heuristic_pct"],
            second_box_pct=self.current_parameters["second_box_pct"],
            second_box_cost=self.current_parameters["second_box_cost"],
            second_choice_rational_factor=self.current_parameters["second_choice_rational_factor"]
        )
        
        # 运行迭代
        distribution, profits = optimizer.run_iteration(print_progress=False)
        
        # 找出最优宝箱
        best_single, best_pair, single_profit, pair_profit = optimizer.analyze_optimal_strategy(profits)
        
        # 选择单选策略作为初始策略
        initial_strategy = best_single
        
        # 记录初始策略
        self.current_strategy = initial_strategy
        self.strategy_history.append(initial_strategy)
        
        # 记录初始收益预期
        self.profit_history.append(profits)
        
        return initial_strategy
    
    def generate_feedback(self, strategy: int, round_num: int, is_real_feedback: bool = False, 
                        real_reward: float = None) -> Dict:
        """
        生成或接收策略反馈
        
        参数:
            strategy: 当前策略(宝箱ID)
            round_num: 轮次
            is_real_feedback: 是否为真实反馈
            real_reward: 真实收益 (当is_real_feedback为True时使用)
            
        返回:
            反馈信息字典
        """
        # 获取当前期望收益
        optimizer = TreasureOptimizer(
            treasures=self.treasures,
            rational_pct=self.current_parameters["rational_pct"],
            heuristic_pct=self.current_parameters["heuristic_pct"],
            second_box_pct=self.current_parameters["second_box_pct"],
            second_box_cost=self.current_parameters["second_box_cost"],
            second_choice_rational_factor=self.current_parameters["second_choice_rational_factor"]
        )
        
        _, profits = optimizer.run_iteration(print_progress=False)
        expected_reward = profits[strategy]
        
        if is_real_feedback and real_reward is not None:
            # 使用真实反馈
            actual_reward = real_reward
        else:
            # 生成模拟反馈: 在期望收益附近随机波动
            variation = expected_reward * 0.2  # 20%的波动范围
            actual_reward = expected_reward + random.uniform(-variation, variation)
        
        # 计算反馈比例 (实际/期望)
        reward_ratio = actual_reward / expected_reward if expected_reward != 0 else 1.0
        
        # 创建反馈信息
        feedback = {
            "round": round_num,
            "strategy": strategy,
            "expected_reward": expected_reward,
            "actual_reward": actual_reward,
            "reward_ratio": reward_ratio,
            "parameters": self.current_parameters.copy()
        }
        
        # 更新绩效指标
        self.performance_metrics["cumulative_reward"] += actual_reward
        self.performance_metrics["average_reward"] = (
            self.performance_metrics["cumulative_reward"] / (round_num + 1)
        )
        
        # 更新最佳和最差收益
        if actual_reward > self.performance_metrics["best_reward"]:
            self.performance_metrics["best_reward"] = actual_reward
            
        if actual_reward < self.performance_metrics["worst_reward"]:
            self.performance_metrics["worst_reward"] = actual_reward
        
        # 计算收益方差
        if round_num > 0:
            rewards = [f["actual_reward"] for f in self.feedback_history] + [actual_reward]
            self.performance_metrics["reward_variance"] = np.var(rewards)
        
        # 记录反馈
        self.feedback_history.append(feedback)
        
        return feedback
    
    def update_parameters(self, feedback: Dict) -> Dict:
        """
        根据反馈更新参数
        
        参数:
            feedback: 反馈信息
            
        返回:
            更新后的参数
        """
        # 获取反馈信息
        reward_ratio = feedback["reward_ratio"]
        strategy = feedback["strategy"]
        
        # 创建新参数
        new_params = self.current_parameters.copy()
        
        # 根据反馈调整参数
        # 如果实际收益显著低于预期 (反馈比例 < 0.9)
        if reward_ratio < 0.9:
            # 调整理性玩家比例 (假设理性玩家模型可能不准确)
            rational_adjustment = -self.adaptation_rate * (1 - reward_ratio)
            new_params["rational_pct"] = max(0.2, min(0.6, 
                new_params["rational_pct"] + rational_adjustment
            ))
            
            # 相应调整启发式玩家比例，保持总比例为1
            new_params["heuristic_pct"] = max(0.2, min(0.7, 
                1 - new_params["rational_pct"] - 0.1  # 保留10%随机玩家
            ))
            
            # 如果当前宝箱是热门宝箱，可能需要减少第二宝箱选择比例
            treasure_value = next(t.box_value for t in self.treasures if t.box_id == strategy)
            average_value = np.mean([t.box_value for t in self.treasures])
            
            if treasure_value > average_value:
                # 热门宝箱，减少第二宝箱选择
                new_params["second_box_pct"] = max(0.01, 
                    new_params["second_box_pct"] - self.adaptation_rate * 0.5
                )
                
            # 调整第二选择理性因子
            new_params["second_choice_rational_factor"] = max(0.3, min(1.0,
                new_params["second_choice_rational_factor"] - self.adaptation_rate * (1 - reward_ratio)
            ))
                
        # 如果实际收益显著高于预期 (反馈比例 > 1.1)
        elif reward_ratio > 1.1:
            # 当前参数设置似乎低估了实际收益，可能需要增加理性玩家比例
            rational_adjustment = self.adaptation_rate * (reward_ratio - 1)
            new_params["rational_pct"] = max(0.2, min(0.6, 
                new_params["rational_pct"] + rational_adjustment
            ))
            
            # 相应调整启发式玩家比例
            new_params["heuristic_pct"] = max(0.2, min(0.7, 
                1 - new_params["rational_pct"] - 0.1
            ))
            
            # 如果当前宝箱是冷门宝箱，可能需要增加第二宝箱选择比例
            treasure_value = next(t.box_value for t in self.treasures if t.box_id == strategy)
            average_value = np.mean([t.box_value for t in self.treasures])
            
            if treasure_value < average_value:
                # 冷门宝箱，增加第二宝箱选择
                new_params["second_box_pct"] = min(0.2, 
                    new_params["second_box_pct"] + self.adaptation_rate * 0.5
                )
                
            # 调整第二选择理性因子
            new_params["second_choice_rational_factor"] = max(0.3, min(1.0,
                new_params["second_choice_rational_factor"] + self.adaptation_rate * (reward_ratio - 1)
            ))
        
        # 更新当前参数
        self.current_parameters = new_params
        
        # 记录参数历史
        self.parameter_history.append(new_params.copy())
        
        return new_params
    
    def update_strategy(self, feedback: Dict) -> int:
        """
        根据反馈更新策略
        
        参数:
            feedback: 反馈信息
            
        返回:
            更新后的策略
        """
        # 获取当前参数下的最优策略
        optimizer = TreasureOptimizer(
            treasures=self.treasures,
            rational_pct=self.current_parameters["rational_pct"],
            heuristic_pct=self.current_parameters["heuristic_pct"],
            second_box_pct=self.current_parameters["second_box_pct"],
            second_box_cost=self.current_parameters["second_box_cost"],
            second_choice_rational_factor=self.current_parameters["second_choice_rational_factor"]
        )
        
        distribution, profits = optimizer.run_iteration(print_progress=False)
        best_single, best_pair, single_profit, pair_profit = optimizer.analyze_optimal_strategy(profits)
        
        # 计算每个宝箱的性能历史
        box_performance = {}
        
        for box_id in range(1, len(self.treasures) + 1):
            # 找出所有使用这个宝箱的反馈
            relevant_feedback = [f for f in self.feedback_history if f["strategy"] == box_id]
            
            if relevant_feedback:
                # 计算平均实际/预期比例
                avg_ratio = np.mean([f["reward_ratio"] for f in relevant_feedback])
                # 计算最近趋势 (如果有足够的历史数据)
                if len(relevant_feedback) >= 3:
                    recent_trend = np.mean([f["reward_ratio"] for f in relevant_feedback[-3:]])
                else:
                    recent_trend = avg_ratio
                    
                # 综合考虑历史表现和当前预期收益
                adjusted_profit = profits[box_id] * (0.7 + 0.3 * recent_trend)
                
                box_performance[box_id] = {
                    "count": len(relevant_feedback),
                    "avg_ratio": avg_ratio,
                    "recent_trend": recent_trend,
                    "expected_profit": profits[box_id],
                    "adjusted_profit": adjusted_profit
                }
            else:
                # 没有历史数据，使用当前预期
                box_performance[box_id] = {
                    "count": 0,
                    "avg_ratio": 1.0,
                    "recent_trend": 1.0,
                    "expected_profit": profits[box_id],
                    "adjusted_profit": profits[box_id]
                }
        
        # 找出调整后收益最高的宝箱
        best_adjusted = max(box_performance.items(), key=lambda x: x[1]["adjusted_profit"])[0]
        
        # 决策: 利用 vs 探索
        if random.random() < self.exploration_probability:
            # 探索: 随机选择一个非当前最优的宝箱，但也不是最差的
            # 按调整后收益排序
            sorted_boxes = sorted(box_performance.items(), key=lambda x: x[1]["adjusted_profit"], reverse=True)
            
            # 排除最好和最差的选项
            candidates = [box_id for box_id, _ in sorted_boxes[1:-1]]
            
            if candidates:
                # 随机选择一个候选宝箱
                new_strategy = random.choice(candidates)
            else:
                # 如果候选集为空，退回到最优选择
                new_strategy = best_adjusted
        else:
            # 利用: 选择调整后收益最高的宝箱
            new_strategy = best_adjusted
        
        # 更新当前策略
        self.current_strategy = new_strategy
        
        # 记录策略历史
        self.strategy_history.append(new_strategy)
        
        # 记录当前收益预期
        self.profit_history.append(profits)
        
        return new_strategy
    
    def run_adaptive_decision_cycle(self, num_rounds: int = 5, 
                                  real_feedback: List[Dict] = None) -> Dict:
        """
        运行自适应决策循环
        
        参数:
            num_rounds: 循环轮数
            real_feedback: 真实反馈数据列表 (可选)
            
        返回:
            决策循环结果
        """
        print(f"开始运行自适应决策循环 (轮数: {num_rounds})...")
        
        # 初始化策略
        initial_strategy = self.initialize_strategy()
        print(f"初始策略: 宝箱{initial_strategy}")
        
        # 运行决策循环
        for round_num in range(num_rounds):
            print(f"\n--- 第{round_num+1}轮 ---")
            strategy = self.current_strategy
            print(f"当前策略: 宝箱{strategy}")
            
            # 生成或接收反馈
            is_real = real_feedback is not None and round_num < len(real_feedback)
            real_reward = real_feedback[round_num]["reward"] if is_real else None
            
            feedback = self.generate_feedback(
                strategy=strategy, 
                round_num=round_num,
                is_real_feedback=is_real,
                real_reward=real_reward
            )
            
            print(f"反馈: 预期={feedback['expected_reward']:.2f}, " +
                 f"实际={feedback['actual_reward']:.2f}, " +
                 f"比例={feedback['reward_ratio']:.2f}")
            
            # 如果不是最后一轮，更新参数和策略
            if round_num < num_rounds - 1:
                # 更新参数
                new_params = self.update_parameters(feedback)
                
                param_changes = [
                    f"理性:{new_params['rational_pct']:.2f}",
                    f"启发:{new_params['heuristic_pct']:.2f}",
                    f"第二宝箱:{new_params['second_box_pct']:.2f}",
                    f"理性因子:{new_params['second_choice_rational_factor']:.2f}"
                ]
                print(f"参数更新: " + ", ".join(param_changes))
                
                # 更新策略
                new_strategy = self.update_strategy(feedback)
                print(f"策略更新: 宝箱{strategy} -> 宝箱{new_strategy}")
        
        # 收集结果
        result = {
            "feedback_history": self.feedback_history,
            "parameter_history": self.parameter_history,
            "strategy_history": self.strategy_history,
            "profit_history": self.profit_history,
            "final_strategy": self.current_strategy,
            "final_parameters": self.current_parameters,
            "performance_metrics": self.performance_metrics
        }
        
        return result
    
    def visualize_strategy_evolution(self, decision_result: Dict, output_filename: str = "strategy_evolution.png") -> None:
        """
        可视化策略演变
        
        参数:
            decision_result: 决策循环结果
            output_filename: 输出文件名
        """
        strategy_history = decision_result["strategy_history"]
        feedback_history = decision_result["feedback_history"]
        rounds = list(range(len(strategy_history)))
        
        # 提取每轮的实际和预期收益
        actual_rewards = [f["actual_reward"] for f in feedback_history]
        expected_rewards = [f["expected_reward"] for f in feedback_history]
        
        # 创建图像
        plt.figure(figsize=(12, 8))
        
        # 主图: 策略演变
        ax1 = plt.subplot(2, 1, 1)
        ax1.plot(rounds, strategy_history, 'bo-', linewidth=2)
        
        # 设置Y轴为宝箱ID
        ax1.set_yticks(range(1, len(self.treasures) + 1))
        ax1.set_yticklabels([f'宝箱{i}' for i in range(1, len(self.treasures) + 1)])
        
        # 添加网格和标签
        ax1.grid(True, linestyle='--', alpha=0.7)
        ax1.set_xlabel('轮次')
        ax1.set_ylabel('选择的宝箱')
        ax1.set_title('策略演变')
        
        # 次图: 收益比较
        ax2 = plt.subplot(2, 1, 2)
        ax2.plot(rounds, expected_rewards, 'g-', marker='o', label='预期收益')
        ax2.plot(rounds, actual_rewards, 'r-', marker='x', label='实际收益')
        
        # 添加网格和标签
        ax2.grid(True, linestyle='--', alpha=0.7)
        ax2.set_xlabel('轮次')
        ax2.set_ylabel('收益')
        ax2.set_title('预期与实际收益比较')
        ax2.legend()
        
        # 保存图像
        plt.tight_layout()
        plt.savefig(os.path.join(self.output_dir, output_filename))
        plt.close()
        
        print(f"策略演变图已保存到 {os.path.join(self.output_dir, output_filename)}")
    
    def visualize_parameter_adaptation(self, decision_result: Dict, output_filename: str = "parameter_adaptation.png") -> None:
        """
        可视化参数适应
        
        参数:
            decision_result: 决策循环结果
            output_filename: 输出文件名
        """
        parameter_history = decision_result["parameter_history"]
        rounds = list(range(len(parameter_history)))
        
        # 提取参数历史
        rational_pct = [p["rational_pct"] for p in parameter_history]
        heuristic_pct = [p["heuristic_pct"] for p in parameter_history]
        second_box_pct = [p["second_box_pct"] for p in parameter_history]
        rational_factor = [p["second_choice_rational_factor"] for p in parameter_history]
        
        # 创建图像
        plt.figure(figsize=(12, 10))
        
        # 主图: 玩家类型比例
        ax1 = plt.subplot(2, 1, 1)
        ax1.plot(rounds, rational_pct, 'b-', marker='o', label='理性玩家比例')
        ax1.plot(rounds, heuristic_pct, 'g-', marker='s', label='启发式玩家比例')
        ax1.plot(rounds, [1 - r - h for r, h in zip(rational_pct, heuristic_pct)], 
                'r-', marker='^', label='随机玩家比例')
        
        # 添加网格和标签
        ax1.grid(True, linestyle='--', alpha=0.7)
        ax1.set_xlabel('轮次')
        ax1.set_ylabel('比例')
        ax1.set_title('玩家类型比例适应')
        ax1.legend()
        
        # 次图: 其他参数
        ax2 = plt.subplot(2, 1, 2)
        ax2.plot(rounds, second_box_pct, 'c-', marker='o', label='第二宝箱选择比例')
        ax2.plot(rounds, rational_factor, 'm-', marker='s', label='第二选择理性因子')
        
        # 添加网格和标签
        ax2.grid(True, linestyle='--', alpha=0.7)
        ax2.set_xlabel('轮次')
        ax2.set_ylabel('参数值')
        ax2.set_title('策略参数适应')
        ax2.legend()
        
        # 保存图像
        plt.tight_layout()
        plt.savefig(os.path.join(self.output_dir, output_filename))
        plt.close()
        
        print(f"参数适应图已保存到 {os.path.join(self.output_dir, output_filename)}")
    
    def visualize_feedback_analysis(self, decision_result: Dict, output_filename: str = "feedback_analysis.png") -> None:
        """
        可视化反馈分析
        
        参数:
            decision_result: 决策循环结果
            output_filename: 输出文件名
        """
        feedback_history = decision_result["feedback_history"]
        strategy_history = decision_result["strategy_history"]
        rounds = list(range(len(feedback_history)))
        
        # 提取反馈比例
        reward_ratios = [f["reward_ratio"] for f in feedback_history]
        
        # 创建图像
        plt.figure(figsize=(12, 8))
        
        # 主图: 反馈比例
        ax = plt.subplot(1, 1, 1)
        
        # 使用不同颜色表示不同的宝箱策略
        unique_strategies = list(set(strategy_history))
        colors = plt.cm.tab10(np.linspace(0, 1, len(unique_strategies)))
        strategy_to_color = {strategy: color for strategy, color in zip(unique_strategies, colors)}
        
        # 绘制水平线表示1.0 (完美预测)
        ax.axhline(y=1.0, color='k', linestyle='--', alpha=0.7)
        
        # 绘制反馈比例点，按策略上色
        for i, ratio in enumerate(reward_ratios):
            strategy = strategy_history[i]
            ax.scatter(i, ratio, color=strategy_to_color[strategy], s=100)
            
            # 添加宝箱ID标签
            ax.annotate(f'{strategy}', (i, ratio), 
                       xytext=(0, 5), textcoords='offset points',
                       ha='center', fontsize=10)
        
        # 绘制线条连接点
        ax.plot(rounds, reward_ratios, 'k-', alpha=0.3)
        
        # 添加阈值区域
        ax.axhspan(0.9, 1.1, alpha=0.2, color='g', label='正常波动区间 (±10%)')
        
        # 添加图例
        legend_elements = [plt.Line2D([0], [0], marker='o', color='w', 
                                     markerfacecolor=strategy_to_color[s], 
                                     label=f'宝箱{s}', markersize=10)
                          for s in unique_strategies]
        legend_elements.append(plt.Rectangle((0, 0), 1, 1, fc='g', alpha=0.2, label='正常波动区间'))
        
        ax.legend(handles=legend_elements, loc='best')
        
        # 添加网格和标签
        ax.grid(True, linestyle='--', alpha=0.7)
        ax.set_xlabel('轮次')
        ax.set_ylabel('实际/预期收益比例')
        ax.set_title('反馈分析')
        
        # 保存图像
        plt.tight_layout()
        plt.savefig(os.path.join(self.output_dir, output_filename))
        plt.close()
        
        print(f"反馈分析图已保存到 {os.path.join(self.output_dir, output_filename)}")
    
    def visualize_performance_comparison(self, decision_result: Dict, output_filename: str = "performance_comparison.png") -> None:
        """
        可视化性能比较
        
        参数:
            decision_result: 决策循环结果
            output_filename: 输出文件名
        """
        strategy_history = decision_result["strategy_history"]
        feedback_history = decision_result["feedback_history"]
        profit_history = decision_result["profit_history"]
        
        # 收集每个宝箱在各轮的预期收益
        box_expected_profits = {}
        
        for round_num, profits in enumerate(profit_history):
            for box_id, profit in profits.items():
                if box_id not in box_expected_profits:
                    box_expected_profits[box_id] = []
                
                # 确保每个宝箱在每轮都有数据
                while len(box_expected_profits[box_id]) < round_num:
                    box_expected_profits[box_id].append(None)
                
                box_expected_profits[box_id].append(profit)
        
        # 确保所有宝箱列表长度一致
        max_rounds = len(profit_history)
        for box_id in box_expected_profits:
            while len(box_expected_profits[box_id]) < max_rounds:
                box_expected_profits[box_id].append(None)
        
        # 获取实际选择的宝箱的实际收益
        chosen_actual_profits = [f["actual_reward"] for f in feedback_history]
        
        # 创建图像
        plt.figure(figsize=(12, 8))
        
        # 绘制每个宝箱的预期收益
        rounds = list(range(max_rounds))
        
        for box_id, profits in box_expected_profits.items():
            plt.plot(rounds, profits, '--', alpha=0.5, label=f'宝箱{box_id}预期')
        
        # 绘制实际选择的宝箱及其实际收益
        for i, (strategy, actual) in enumerate(zip(strategy_history, chosen_actual_profits)):
            plt.scatter(i, actual, s=100, 
                       color=plt.cm.tab10(strategy / len(self.treasures)), 
                       edgecolors='k', zorder=10)
            
            # 添加宝箱ID标签
            plt.annotate(f'{strategy}', (i, actual), 
                        xytext=(0, 5), textcoords='offset points',
                        ha='center', fontsize=10, fontweight='bold')
        
        # 连接实际收益点
        plt.plot(range(len(chosen_actual_profits)), chosen_actual_profits, 
                'k-', linewidth=2, label='实际选择收益')
        
        # 添加网格和标签
        plt.grid(True, linestyle='--', alpha=0.7)
        plt.xlabel('轮次')
        plt.ylabel('收益')
        plt.title('各宝箱预期收益与实际收益比较')
        
        # 自定义图例 (只显示实际收益线)
        plt.legend([plt.Line2D([0], [0], color='k', linewidth=2)],
                  ['实际选择收益'],
                  loc='best')
        
        # 保存图像
        plt.tight_layout()
        plt.savefig(os.path.join(self.output_dir, output_filename))
        plt.close()
        
        print(f"性能比较图已保存到 {os.path.join(self.output_dir, output_filename)}")
    
    def generate_adaptive_report(self, decision_result: Dict, output_filename: str = "adaptive_decision_report.txt") -> None:
        """
        生成自适应决策报告
        
        参数:
            decision_result: 决策循环结果
            output_filename: 输出文件名
        """
        feedback_history = decision_result["feedback_history"]
        parameter_history = decision_result["parameter_history"]
        strategy_history = decision_result["strategy_history"]
        final_strategy = decision_result["final_strategy"]
        final_parameters = decision_result["final_parameters"]
        performance_metrics = decision_result["performance_metrics"]
        
        report_path = os.path.join(self.output_dir, output_filename)
        
        with open(report_path, 'w', encoding='utf-8') as f:
            f.write("# 自适应决策系统分析报告\n\n")
            f.write(f"分析日期: {pd.Timestamp.now().strftime('%Y-%m-%d %H:%M:%S')}\n\n")
            
            f.write("## 初始参数\n\n")
            f.write(f"- 初始理性玩家比例: {self.initial_rational_pct}\n")
            f.write(f"- 初始启发式玩家比例: {self.initial_heuristic_pct}\n")
            f.write(f"- 初始随机玩家比例: {1 - self.initial_rational_pct - self.initial_heuristic_pct}\n")
            f.write(f"- 初始第二宝箱选择比例: {self.initial_second_box_pct}\n")
            f.write(f"- 初始第二宝箱成本: {self.initial_second_box_cost}\n")
            f.write(f"- 初始第二选择理性因子: {self.initial_second_choice_rational_factor}\n")
            f.write(f"- 适应率: {self.adaptation_rate}\n")
            f.write(f"- 探索概率: {self.exploration_probability}\n\n")
            
            f.write("## 最终参数\n\n")
            f.write(f"- 最终理性玩家比例: {final_parameters['rational_pct']:.2f}\n")
            f.write(f"- 最终启发式玩家比例: {final_parameters['heuristic_pct']:.2f}\n")
            f.write(f"- 最终随机玩家比例: {1 - final_parameters['rational_pct'] - final_parameters['heuristic_pct']:.2f}\n")
            f.write(f"- 最终第二宝箱选择比例: {final_parameters['second_box_pct']:.2f}\n")
            f.write(f"- 最终第二宝箱成本: {final_parameters['second_box_cost']}\n")
            f.write(f"- 最终第二选择理性因子: {final_parameters['second_choice_rational_factor']:.2f}\n\n")
            
            f.write("## 参数演变\n\n")
            
            # 计算参数变化
            initial_params = parameter_history[0]
            final_params = parameter_history[-1]
            
            f.write("| 参数 | 初始值 | 最终值 | 变化 | 变化率(%) |\n")
            f.write("|------|--------|--------|------|----------|\n")
            
            params_to_report = [
                ("理性玩家比例", "rational_pct"),
                ("启发式玩家比例", "heuristic_pct"),
                ("第二宝箱选择比例", "second_box_pct"),
                ("第二选择理性因子", "second_choice_rational_factor")
            ]
            
            for name, key in params_to_report:
                initial = initial_params[key]
                final = final_params[key]
                change = final - initial
                change_pct = (change / initial) * 100 if initial != 0 else float('inf')
                
                f.write(f"| {name} | {initial:.3f} | {final:.3f} | {change:+.3f} | {change_pct:+.1f}% |\n")
            
            f.write("\n## 策略演变\n\n")
            
            # 统计策略使用频率
            strategy_counts = {}
            for strategy in strategy_history:
                if strategy not in strategy_counts:
                    strategy_counts[strategy] = 0
                strategy_counts[strategy] += 1
            
            sorted_strategies = sorted(strategy_counts.items(), key=lambda x: x[1], reverse=True)
            
            f.write("| 宝箱ID | 使用次数 | 使用频率 |\n")
            f.write("|--------|----------|----------|\n")
            
            for strategy, count in sorted_strategies:
                frequency = count / len(strategy_history)
                f.write(f"| 宝箱{strategy} | {count} | {frequency:.1%} |\n")
            
            f.write("\n## 决策历史\n\n")
            
            f.write("| 轮次 | 策略 | 预期收益 | 实际收益 | 比例 | 理性比例 | 启发式比例 | 第二宝箱比例 |\n")
            f.write("|------|------|----------|----------|------|----------|------------|------------|\n")
            
            for i, (feedback, params) in enumerate(zip(feedback_history, parameter_history)):
                f.write(f"| {i+1} | 宝箱{feedback['strategy']} | {feedback['expected_reward']:.2f} | " +
                       f"{feedback['actual_reward']:.2f} | {feedback['reward_ratio']:.2f} | " +
                       f"{params['rational_pct']:.2f} | {params['heuristic_pct']:.2f} | " +
                       f"{params['second_box_pct']:.2f} |\n")
            
            f.write("\n## 性能指标\n\n")
            
            f.write(f"- 累计收益: {performance_metrics['cumulative_reward']:.2f}\n")
            f.write(f"- 平均收益: {performance_metrics['average_reward']:.2f}\n")
            f.write(f"- 收益方差: {performance_metrics['reward_variance']:.2f}\n")
            f.write(f"- 最佳收益: {performance_metrics['best_reward']:.2f}\n")
            f.write(f"- 最差收益: {performance_metrics['worst_reward']:.2f}\n\n")
            
            # 计算最后三轮和前三轮的平均收益比较
            if len(feedback_history) >= 6:
                first_three_avg = sum(f["actual_reward"] for f in feedback_history[:3]) / 3
                last_three_avg = sum(f["actual_reward"] for f in feedback_history[-3:]) / 3
                improvement = ((last_three_avg / first_three_avg) - 1) * 100 if first_three_avg > 0 else float('inf')
                
                f.write(f"- 前三轮平均收益: {first_three_avg:.2f}\n")
                f.write(f"- 后三轮平均收益: {last_three_avg:.2f}\n")
                f.write(f"- 收益提升: {improvement:+.1f}%\n\n")
            
            f.write("## 最终建议\n\n")
            
            f.write(f"根据自适应决策系统的分析，最终推荐选择**宝箱{final_strategy}**。\n\n")
            
            # 添加参数建议
            f.write("### 参数调整建议\n\n")
            
            # 检查参数变化趋势
            rational_trend = final_parameters["rational_pct"] - initial_params["rational_pct"]
            heuristic_trend = final_parameters["heuristic_pct"] - initial_params["heuristic_pct"]
            second_box_trend = final_parameters["second_box_pct"] - initial_params["second_box_pct"]
            rational_factor_trend = final_parameters["second_choice_rational_factor"] - initial_params["second_choice_rational_factor"]
            
            # 对各参数给出建议
            if abs(rational_trend) > 0.05:
                direction = "增加" if rational_trend > 0 else "减少"
                f.write(f"- 建议{direction}理性玩家比例：自适应过程中，理性玩家比例{direction}了{abs(rational_trend):.2f}，" +
                       f"表明模型可能{direction}估计了理性玩家的影响。\n")
            
            if abs(heuristic_trend) > 0.05:
                direction = "增加" if heuristic_trend > 0 else "减少"
                f.write(f"- 建议{direction}启发式玩家比例：自适应过程中，启发式玩家比例{direction}了{abs(heuristic_trend):.2f}，" +
                       f"表明模型可能{direction}估计了启发式行为的影响。\n")
            
            if abs(second_box_trend) > 0.02:
                direction = "增加" if second_box_trend > 0 else "减少"
                f.write(f"- 建议{direction}第二宝箱选择比例：自适应过程中，该参数{direction}了{abs(second_box_trend):.2f}，" +
                       f"表明玩家选择第二宝箱的倾向可能比初始估计{'更强' if second_box_trend > 0 else '更弱'}。\n")
            
            if abs(rational_factor_trend) > 0.1:
                direction = "增加" if rational_factor_trend > 0 else "减少"
                f.write(f"- 建议{direction}第二选择理性因子：自适应过程中，该参数{direction}了{abs(rational_factor_trend):.2f}，" +
                       f"表明玩家在第二选择时的理性程度可能比初始估计{'更高' if rational_factor_trend > 0 else '更低'}。\n")
            
            # 总结
            f.write("\n### 总体结论\n\n")
            
            # 评估自适应系统的表现
            if len(feedback_history) >= 4:
                last_half_avg_ratio = np.mean([f["reward_ratio"] for f in feedback_history[len(feedback_history)//2:]])
                
                if abs(last_half_avg_ratio - 1.0) < 0.1:
                    f.write("自适应系统表现良好，后半段的实际/预期收益比接近1.0，说明模型调整后能准确预测实际收益。")
                elif last_half_avg_ratio > 1.1:
                    f.write("自适应系统仍然低估了实际收益，建议在未来分析中继续调整参数，尤其是提高理性玩家的比例。")
                else:  # last_half_avg_ratio < 0.9
                    f.write("自适应系统仍然高估了实际收益，建议在未来分析中继续调整参数，可能需要降低理性玩家比例或提高启发式行为的权重。")
            else:
                f.write("决策轮数较少，尚未能充分评估自适应系统的性能。建议在未来进行更多轮次的实验以获取更可靠的结论。")
            
            f.write("\n\n最重要的是，通过实时自适应调整，我们能够根据实际比赛中的反馈不断优化模型和策略，提高决策的准确性和稳健性。")
        
        print(f"自适应决策报告已保存到 {report_path}")
    
    def run_full_analysis(self, num_rounds: int = 5, real_feedback: List[Dict] = None) -> Dict:
        """
        运行完整的自适应决策分析
        
        参数:
            num_rounds: 循环轮数
            real_feedback: 真实反馈数据列表 (可选)
            
        返回:
            分析结果字典
        """
        print(f"开始运行自适应决策分析 (轮数: {num_rounds})...")
        
        # 运行自适应决策循环
        decision_result = self.run_adaptive_decision_cycle(num_rounds, real_feedback)
        
        # 可视化策略演变
        self.visualize_strategy_evolution(decision_result)
        
        # 可视化参数适应
        self.visualize_parameter_adaptation(decision_result)
        
        # 可视化反馈分析
        self.visualize_feedback_analysis(decision_result)
        
        # 可视化性能比较
        self.visualize_performance_comparison(decision_result)
        
        # 生成报告
        self.generate_adaptive_report(decision_result)
        
        return decision_result


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
    
    # 创建自适应决策系统
    adaptive_system = AdaptiveDecisionSystem(
        treasures=treasures,
        output_dir="output/adaptive",
        initial_rational_pct=0.35,
        initial_heuristic_pct=0.45,
        initial_second_box_pct=0.05,
        initial_second_box_cost=50000,
        initial_second_choice_rational_factor=0.7,
        adaptation_rate=0.1,
        exploration_probability=0.2
    )
    
    # 模拟一些真实反馈 (可选)
    # 在实际应用中，这些反馈会来自真实比赛
    real_feedback = [
        {"reward": 12500},   # 第1轮反馈
        {"reward": 13200},   # 第2轮反馈
        {"reward": 11800},   # 第3轮反馈
        {"reward": 14500},   # 第4轮反馈
        {"reward": 15000}    # 第5轮反馈
    ]
    
    # 运行完整分析 (使用模拟反馈)
    # adaptive_system.run_full_analysis(num_rounds=5, real_feedback=real_feedback)
    
    # 运行完整分析 (使用系统生成的反馈)
    adaptive_system.run_full_analysis(num_rounds=5)


if __name__ == "__main__":
    main() 