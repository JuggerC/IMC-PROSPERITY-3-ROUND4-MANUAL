#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
动态博弈模型
模拟玩家在多轮博弈中的策略适应行为
"""

import os
import random
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from typing import List, Dict, Tuple, Optional, Union, Any
from simulator import Treasure, TreasureOptimizer

class DynamicGameSimulator:
    """动态博弈模拟器"""
    
    def __init__(self, treasures: List[Treasure], 
                 output_dir: str = "output/dynamic_game",
                 initial_rational_pct: float = 0.35, 
                 initial_heuristic_pct: float = 0.45,
                 initial_second_box_pct: float = 0.05,
                 initial_second_choice_rational_factor: float = 0.7,
                 learning_rate: float = 0.05,
                 random_exploration: float = 0.1):
        """
        初始化动态博弈模拟器
        
        参数:
            treasures: 宝箱列表
            output_dir: 输出目录
            initial_rational_pct: 初始理性玩家比例
            initial_heuristic_pct: 初始启发式玩家比例
            initial_second_box_pct: 初始第二宝箱选择比例
            initial_second_choice_rational_factor: 初始第二选择理性因子
            learning_rate: 学习率
            random_exploration: 随机探索概率
        """
        self.treasures = treasures
        self.initial_rational_pct = initial_rational_pct
        self.initial_heuristic_pct = initial_heuristic_pct
        self.initial_second_box_pct = initial_second_box_pct
        self.initial_second_choice_rational_factor = initial_second_choice_rational_factor
        self.learning_rate = learning_rate
        self.random_exploration = random_exploration
        
        # 创建输出目录
        self.output_dir = output_dir
        os.makedirs(output_dir, exist_ok=True)
        
        # 初始化多类型玩家群体
        self.player_groups = {
            "rational": {
                "population": initial_rational_pct,
                "adaptation_speed": 0.1,  # 理性玩家适应较快
                "current_strategy": None,  # 当前策略
                "strategy_history": []     # 策略历史
            },
            "heuristic": {
                "population": initial_heuristic_pct,
                "adaptation_speed": 0.05,  # 启发式玩家适应较慢
                "current_strategy": None,
                "strategy_history": []
            },
            "random": {
                "population": 1 - initial_rational_pct - initial_heuristic_pct,
                "adaptation_speed": 0.01,  # 随机玩家几乎不适应
                "current_strategy": None,
                "strategy_history": []
            }
        }
        
        # 宝箱选择历史
        self.box_selection_history = []
        
        # 参数历史
        self.parameter_history = []
        
        # 收益历史
        self.profit_history = []
    
    def initialize_strategies(self):
        """初始化各类玩家的初始策略"""
        # 创建优化器
        optimizer = TreasureOptimizer(
            treasures=self.treasures,
            rational_pct=self.initial_rational_pct,
            heuristic_pct=self.initial_heuristic_pct,
            second_box_pct=self.initial_second_box_pct,
            second_choice_rational_factor=self.initial_second_choice_rational_factor
        )
        
        # 运行迭代以获取初始分布
        distribution, _ = optimizer.run_iteration(print_progress=False)
        
        # 设置初始策略
        
        # 理性玩家选择最优宝箱
        _, profits = optimizer.run_iteration(print_progress=False)
        best_single, _, _, _ = optimizer.analyze_optimal_strategy(profits)
        self.player_groups["rational"]["current_strategy"] = best_single
        self.player_groups["rational"]["strategy_history"].append(best_single)
        
        # 启发式玩家选择最高价值宝箱
        highest_value_box = max(self.treasures, key=lambda t: t.box_value).box_id
        self.player_groups["heuristic"]["current_strategy"] = highest_value_box
        self.player_groups["heuristic"]["strategy_history"].append(highest_value_box)
        
        # 随机玩家随机选择
        random_box = random.randint(1, len(self.treasures))
        self.player_groups["random"]["current_strategy"] = random_box
        self.player_groups["random"]["strategy_history"].append(random_box)
        
        # 记录初始选择分布
        self.box_selection_history.append(distribution)
        
        # 记录初始参数
        self.parameter_history.append({
            "rational_pct": self.initial_rational_pct,
            "heuristic_pct": self.initial_heuristic_pct,
            "second_box_pct": self.initial_second_box_pct,
            "second_choice_rational_factor": self.initial_second_choice_rational_factor
        })
        
        # 记录初始收益
        self.profit_history.append(profits)
    
    def update_player_strategies(self, round_results: Dict):
        """
        根据上一轮结果更新玩家策略
        
        参数:
            round_results: 上一轮的结果
        """
        distribution = round_results["distribution"]
        profits = round_results["profits"]
        
        # 更新理性玩家策略
        # 理性玩家总是选择收益最高的宝箱
        best_box = max(profits.items(), key=lambda x: x[1])[0]
        self.player_groups["rational"]["current_strategy"] = best_box
        self.player_groups["rational"]["strategy_history"].append(best_box)
        
        # 更新启发式玩家策略
        # 启发式玩家以一定概率模仿上一轮选择率最高的宝箱
        most_chosen_box = max(distribution.items(), key=lambda x: x[1])[0]
        
        if random.random() < self.player_groups["heuristic"]["adaptation_speed"]:
            self.player_groups["heuristic"]["current_strategy"] = most_chosen_box
        # 否则保持原有策略
        
        self.player_groups["heuristic"]["strategy_history"].append(
            self.player_groups["heuristic"]["current_strategy"]
        )
        
        # 更新随机玩家策略
        # 随机玩家以较小概率模仿上一轮选择率最高的宝箱，否则完全随机
        if random.random() < self.player_groups["random"]["adaptation_speed"]:
            self.player_groups["random"]["current_strategy"] = most_chosen_box
        else:
            self.player_groups["random"]["current_strategy"] = random.randint(1, len(self.treasures))
            
        self.player_groups["random"]["strategy_history"].append(
            self.player_groups["random"]["current_strategy"]
        )
    
    def update_parameters(self, round_results: Dict):
        """
        更新模拟参数
        
        参数:
            round_results: 上一轮的结果
        """
        # 获取上一轮参数
        prev_params = self.parameter_history[-1]
        
        # 创建新参数
        new_params = prev_params.copy()
        
        # 理性玩家比例可能随时间增加(玩家学习博弈规则)
        rational_adjustment = self.learning_rate * random.uniform(-0.5, 1.0)
        new_params["rational_pct"] = max(0.1, min(0.6, prev_params["rational_pct"] + rational_adjustment))
        
        # 启发式玩家比例相应调整
        # 保持总比例为1
        new_params["heuristic_pct"] = max(0.1, min(0.7, 1 - new_params["rational_pct"] - 0.2))
        
        # 第二宝箱选择比例的变化
        # 如果双选策略收益好，增加第二宝箱选择比例
        _, best_pair, single_profit, pair_profit = TreasureOptimizer.analyze_optimal_strategy(
            round_results["profits"], self.treasures
        )
        
        if pair_profit > single_profit:
            # 双选策略更好，增加第二宝箱选择比例
            second_box_adjustment = self.learning_rate * random.uniform(0, 1.0)
        else:
            # 单选策略更好，减少第二宝箱选择比例
            second_box_adjustment = -self.learning_rate * random.uniform(0, 1.0)
            
        new_params["second_box_pct"] = max(0.01, min(0.2, prev_params["second_box_pct"] + second_box_adjustment))
        
        # 第二选择理性因子的变化
        # 随机小幅调整
        rational_factor_adjustment = self.learning_rate * random.uniform(-1.0, 1.0)
        new_params["second_choice_rational_factor"] = max(0.3, min(1.0, 
            prev_params["second_choice_rational_factor"] + rational_factor_adjustment
        ))
        
        # 记录新参数
        self.parameter_history.append(new_params)
        
        return new_params
    
    def run_game_round(self, round_num: int) -> Dict:
        """
        运行一轮博弈
        
        参数:
            round_num: 轮次编号
            
        返回:
            包含本轮结果的字典
        """
        # 获取当前参数
        current_params = self.parameter_history[-1]
        
        # 创建优化器
        optimizer = TreasureOptimizer(
            treasures=self.treasures,
            rational_pct=current_params["rational_pct"],
            heuristic_pct=current_params["heuristic_pct"],
            second_box_pct=current_params["second_box_pct"],
            second_choice_rational_factor=current_params["second_choice_rational_factor"]
        )
        
        # 运行迭代
        distribution, profits = optimizer.run_iteration(print_progress=False)
        
        # 分析最优策略
        best_single, best_pair, single_profit, pair_profit = optimizer.analyze_optimal_strategy(profits)
        
        # 收集结果
        round_results = {
            "round": round_num,
            "parameters": current_params,
            "distribution": distribution,
            "profits": profits,
            "best_single": best_single,
            "best_pair": best_pair,
            "single_profit": single_profit,
            "pair_profit": pair_profit,
            "player_strategies": {
                player_type: group["current_strategy"] 
                for player_type, group in self.player_groups.items()
            }
        }
        
        # 记录选择分布
        self.box_selection_history.append(distribution)
        
        # 记录收益
        self.profit_history.append(profits)
        
        print(f"轮次 {round_num}: 最佳单选=宝箱{best_single}(收益:{single_profit:.2f}), " +
              f"最佳双选=宝箱{best_pair[0]}+宝箱{best_pair[1]}(净收益:{pair_profit:.2f})")
        
        return round_results
    
    def simulate_dynamic_game(self, num_rounds: int = 10) -> Dict:
        """
        模拟多轮动态博弈
        
        参数:
            num_rounds: 博弈轮数
            
        返回:
            模拟结果字典
        """
        print(f"开始模拟{num_rounds}轮动态博弈...")
        
        # 初始化策略
        self.initialize_strategies()
        
        rounds_data = []
        
        # 模拟多轮博弈
        for round_num in range(1, num_rounds + 1):
            print(f"\n--- 第{round_num}轮博弈 ---")
            
            # 运行一轮博弈
            round_results = self.run_game_round(round_num)
            rounds_data.append(round_results)
            
            # 如果不是最后一轮，更新参数和策略
            if round_num < num_rounds:
                # 更新玩家策略
                self.update_player_strategies(round_results)
                
                # 更新参数
                self.update_parameters(round_results)
        
        # 返回模拟结果
        return {
            "rounds_data": rounds_data,
            "parameter_history": self.parameter_history,
            "box_selection_history": self.box_selection_history,
            "profit_history": self.profit_history,
            "player_groups": self.player_groups
        }
    
    def visualize_parameter_evolution(self, simulation_results: Dict, output_filename: str = "parameter_evolution.png") -> None:
        """
        可视化参数随时间的演变
        
        参数:
            simulation_results: 模拟结果
            output_filename: 输出文件名
        """
        parameter_history = simulation_results["parameter_history"]
        
        # 提取参数历史
        rounds = list(range(len(parameter_history)))
        rational_pct = [p["rational_pct"] for p in parameter_history]
        heuristic_pct = [p["heuristic_pct"] for p in parameter_history]
        random_pct = [1 - p["rational_pct"] - p["heuristic_pct"] for p in parameter_history]
        second_box_pct = [p["second_box_pct"] for p in parameter_history]
        rational_factor = [p["second_choice_rational_factor"] for p in parameter_history]
        
        # 创建图像
        plt.figure(figsize=(14, 10))
        
        # 绘制玩家类型比例
        plt.subplot(2, 1, 1)
        plt.plot(rounds, rational_pct, 'b-', marker='o', label='理性玩家')
        plt.plot(rounds, heuristic_pct, 'g-', marker='s', label='启发式玩家')
        plt.plot(rounds, random_pct, 'r-', marker='^', label='随机玩家')
        
        plt.xlabel('轮次')
        plt.ylabel('玩家比例')
        plt.title('玩家类型比例随时间的变化')
        plt.grid(True, linestyle='--', alpha=0.7)
        plt.legend()
        
        # 绘制其他参数
        plt.subplot(2, 1, 2)
        plt.plot(rounds, second_box_pct, 'c-', marker='o', label='第二宝箱选择比例')
        plt.plot(rounds, rational_factor, 'm-', marker='s', label='第二选择理性因子')
        
        plt.xlabel('轮次')
        plt.ylabel('参数值')
        plt.title('模型参数随时间的变化')
        plt.grid(True, linestyle='--', alpha=0.7)
        plt.legend()
        
        # 保存图像
        plt.tight_layout()
        plt.savefig(os.path.join(self.output_dir, output_filename))
        plt.close()
        
        print(f"参数演变图已保存到 {os.path.join(self.output_dir, output_filename)}")
    
    def visualize_box_selection(self, simulation_results: Dict, output_filename: str = "box_selection_evolution.png") -> None:
        """
        可视化宝箱选择分布随时间的变化
        
        参数:
            simulation_results: 模拟结果
            output_filename: 输出文件名
        """
        box_selection_history = simulation_results["box_selection_history"]
        num_boxes = len(self.treasures)
        
        # 创建图像
        plt.figure(figsize=(14, 10))
        
        # 为每个宝箱创建一条线
        for box_id in range(1, num_boxes + 1):
            selection_pct = [distribution[box_id] for distribution in box_selection_history]
            plt.plot(range(len(box_selection_history)), selection_pct, marker='o', label=f'宝箱{box_id}')
        
        plt.xlabel('轮次')
        plt.ylabel('选择比例')
        plt.title('宝箱选择分布随时间的变化')
        plt.grid(True, linestyle='--', alpha=0.7)
        plt.legend()
        
        # 保存图像
        plt.tight_layout()
        plt.savefig(os.path.join(self.output_dir, output_filename))
        plt.close()
        
        print(f"宝箱选择演变图已保存到 {os.path.join(self.output_dir, output_filename)}")
    
    def visualize_profit_evolution(self, simulation_results: Dict, output_filename: str = "profit_evolution.png") -> None:
        """
        可视化宝箱收益随时间的变化
        
        参数:
            simulation_results: 模拟结果
            output_filename: 输出文件名
        """
        rounds_data = simulation_results["rounds_data"]
        profit_history = simulation_results["profit_history"][1:]  # 跳过初始轮
        
        # 创建图像
        plt.figure(figsize=(14, 12))
        
        # 绘制每轮的最优宝箱收益
        rounds = list(range(1, len(rounds_data) + 1))
        single_profits = [r["single_profit"] for r in rounds_data]
        pair_profits = [r["pair_profit"] for r in rounds_data]
        
        plt.subplot(2, 1, 1)
        plt.plot(rounds, single_profits, 'b-', marker='o', label='最佳单选收益')
        plt.plot(rounds, pair_profits, 'g-', marker='s', label='最佳双选收益')
        
        # 添加最佳选择的标签
        for i, r in enumerate(rounds_data):
            plt.annotate(f"{r['best_single']}", (rounds[i], single_profits[i]),
                        xytext=(5, 5), textcoords='offset points')
            plt.annotate(f"{r['best_pair'][0]}-{r['best_pair'][1]}", (rounds[i], pair_profits[i]),
                        xytext=(5, 5), textcoords='offset points')
        
        plt.xlabel('轮次')
        plt.ylabel('收益')
        plt.title('最优策略收益随时间的变化')
        plt.grid(True, linestyle='--', alpha=0.7)
        plt.legend()
        
        # 绘制所有宝箱的收益变化
        plt.subplot(2, 1, 2)
        
        num_boxes = len(self.treasures)
        for box_id in range(1, num_boxes + 1):
            profits = [profit[box_id] for profit in profit_history]
            plt.plot(rounds, profits, marker='o', label=f'宝箱{box_id}')
        
        plt.xlabel('轮次')
        plt.ylabel('收益')
        plt.title('各宝箱收益随时间的变化')
        plt.grid(True, linestyle='--', alpha=0.7)
        plt.legend()
        
        # 保存图像
        plt.tight_layout()
        plt.savefig(os.path.join(self.output_dir, output_filename))
        plt.close()
        
        print(f"收益演变图已保存到 {os.path.join(self.output_dir, output_filename)}")
    
    def visualize_player_strategies(self, simulation_results: Dict, output_filename: str = "player_strategies.png") -> None:
        """
        可视化不同类型玩家的策略选择
        
        参数:
            simulation_results: 模拟结果
            output_filename: 输出文件名
        """
        player_groups = simulation_results["player_groups"]
        
        # 创建图像
        plt.figure(figsize=(14, 8))
        
        # 为每类玩家绘制策略变化
        player_types = ["rational", "heuristic", "random"]
        colors = ['b', 'g', 'r']
        markers = ['o', 's', '^']
        
        for i, player_type in enumerate(player_types):
            strategies = player_groups[player_type]["strategy_history"]
            rounds = list(range(len(strategies)))
            
            plt.plot(rounds, strategies, color=colors[i], marker=markers[i], label=f'{player_type}玩家')
        
        # 设置y轴只显示整数(宝箱ID)
        plt.yticks(range(1, len(self.treasures) + 1))
        
        plt.xlabel('轮次')
        plt.ylabel('选择的宝箱')
        plt.title('不同类型玩家的策略选择随时间的变化')
        plt.grid(True, linestyle='--', alpha=0.7)
        plt.legend()
        
        # 保存图像
        plt.tight_layout()
        plt.savefig(os.path.join(self.output_dir, output_filename))
        plt.close()
        
        print(f"玩家策略图已保存到 {os.path.join(self.output_dir, output_filename)}")
    
    def analyze_strategy_stability(self, simulation_results: Dict) -> Dict:
        """
        分析策略稳定性
        
        参数:
            simulation_results: 模拟结果
            
        返回:
            策略稳定性分析结果
        """
        rounds_data = simulation_results["rounds_data"]
        
        # 分析最佳单选策略的稳定性
        best_singles = [r["best_single"] for r in rounds_data]
        single_count = {}
        
        for box in best_singles:
            if box not in single_count:
                single_count[box] = 0
            single_count[box] += 1
        
        # 计算最常见的最佳单选
        most_common_single = max(single_count.items(), key=lambda x: x[1])
        
        # 连续稳定轮数
        current_streak = 1
        max_streak = 1
        max_streak_box = best_singles[0]
        
        for i in range(1, len(best_singles)):
            if best_singles[i] == best_singles[i-1]:
                current_streak += 1
                if current_streak > max_streak:
                    max_streak = current_streak
                    max_streak_box = best_singles[i]
            else:
                current_streak = 1
        
        # 分析参数稳定性
        parameter_history = simulation_results["parameter_history"]
        
        param_changes = {
            "rational_pct": [],
            "heuristic_pct": [],
            "second_box_pct": [],
            "second_choice_rational_factor": []
        }
        
        for i in range(1, len(parameter_history)):
            for param in param_changes.keys():
                change = abs(parameter_history[i][param] - parameter_history[i-1][param])
                param_changes[param].append(change)
        
        # 计算平均变化率
        avg_changes = {param: np.mean(changes) for param, changes in param_changes.items()}
        
        # 最后3轮的收敛状态
        last_rounds = min(3, len(rounds_data))
        last_best_singles = best_singles[-last_rounds:]
        is_converged = len(set(last_best_singles)) == 1
        
        return {
            "most_common_single": most_common_single,
            "max_streak": (max_streak_box, max_streak),
            "avg_parameter_changes": avg_changes,
            "is_converged": is_converged,
            "convergence_box": last_best_singles[0] if is_converged else None
        }
    
    def generate_dynamic_game_report(self, simulation_results: Dict, stability_analysis: Dict, 
                                   output_filename: str = "dynamic_game_report.txt") -> None:
        """
        生成动态博弈模拟报告
        
        参数:
            simulation_results: 模拟结果
            stability_analysis: 策略稳定性分析结果
            output_filename: 输出文件名
        """
        rounds_data = simulation_results["rounds_data"]
        parameter_history = simulation_results["parameter_history"]
        
        report_path = os.path.join(self.output_dir, output_filename)
        
        with open(report_path, 'w', encoding='utf-8') as f:
            f.write("# 动态博弈模拟分析报告\n\n")
            f.write(f"分析日期: {pd.Timestamp.now().strftime('%Y-%m-%d %H:%M:%S')}\n\n")
            
            f.write("## 初始参数\n\n")
            f.write(f"- 初始理性玩家比例: {self.initial_rational_pct}\n")
            f.write(f"- 初始启发式玩家比例: {self.initial_heuristic_pct}\n")
            f.write(f"- 初始随机玩家比例: {1 - self.initial_rational_pct - self.initial_heuristic_pct}\n")
            f.write(f"- 初始第二宝箱选择比例: {self.initial_second_box_pct}\n")
            f.write(f"- 初始第二选择理性因子: {self.initial_second_choice_rational_factor}\n")
            f.write(f"- 学习率: {self.learning_rate}\n")
            f.write(f"- 随机探索概率: {self.random_exploration}\n\n")
            
            f.write("## 最终参数\n\n")
            final_params = parameter_history[-1]
            f.write(f"- 最终理性玩家比例: {final_params['rational_pct']:.2f}\n")
            f.write(f"- 最终启发式玩家比例: {final_params['heuristic_pct']:.2f}\n")
            f.write(f"- 最终随机玩家比例: {1 - final_params['rational_pct'] - final_params['heuristic_pct']:.2f}\n")
            f.write(f"- 最终第二宝箱选择比例: {final_params['second_box_pct']:.2f}\n")
            f.write(f"- 最终第二选择理性因子: {final_params['second_choice_rational_factor']:.2f}\n\n")
            
            f.write("## 参数变化\n\n")
            avg_changes = stability_analysis["avg_parameter_changes"]
            f.write(f"- 理性玩家比例平均变化率: {avg_changes['rational_pct']:.4f}\n")
            f.write(f"- 启发式玩家比例平均变化率: {avg_changes['heuristic_pct']:.4f}\n")
            f.write(f"- 第二宝箱选择比例平均变化率: {avg_changes['second_box_pct']:.4f}\n")
            f.write(f"- 第二选择理性因子平均变化率: {avg_changes['second_choice_rational_factor']:.4f}\n\n")
            
            f.write("## 策略稳定性分析\n\n")
            most_common_box, occurrence = stability_analysis["most_common_single"]
            total_rounds = len(rounds_data)
            f.write(f"- 最常见的最优单选策略: 宝箱{most_common_box} (在{total_rounds}轮中出现{occurrence}次, {occurrence/total_rounds:.1%})\n")
            
            max_streak_box, max_streak = stability_analysis["max_streak"]
            f.write(f"- 最长连续稳定轮数: {max_streak}轮 (宝箱{max_streak_box})\n")
            
            if stability_analysis["is_converged"]:
                f.write(f"- 策略已收敛到: 宝箱{stability_analysis['convergence_box']}\n\n")
            else:
                f.write("- 策略未收敛，仍在波动\n\n")
            
            f.write("## 各轮次详细结果\n\n")
            f.write("| 轮次 | 最优单选 | 单选收益 | 最优双选 | 双选收益 | 理性比例 | 启发式比例 | 第二宝箱选择比例 |\n")
            f.write("|------|----------|----------|----------|----------|----------|------------|------------------|\n")
            
            for i, round_data in enumerate(rounds_data):
                params = parameter_history[i+1] if i+1 < len(parameter_history) else parameter_history[-1]
                
                f.write(f"| {round_data['round']} | 宝箱{round_data['best_single']} | {round_data['single_profit']:.2f} | " +
                       f"宝箱{round_data['best_pair'][0]}-{round_data['best_pair'][1]} | {round_data['pair_profit']:.2f} | " +
                       f"{params['rational_pct']:.2f} | {params['heuristic_pct']:.2f} | {params['second_box_pct']:.2f} |\n")
            
            f.write("\n## 玩家策略演变\n\n")
            player_groups = simulation_results["player_groups"]
            
            for player_type, data in player_groups.items():
                strategies = data["strategy_history"]
                
                f.write(f"### {player_type}玩家\n\n")
                f.write(f"- 人口比例: {data['population']:.2f}\n")
                f.write(f"- 适应速度: {data['adaptation_speed']:.2f}\n")
                f.write("- 策略变化: ")
                
                for i, strategy in enumerate(strategies):
                    if i > 0:
                        f.write(" → ")
                    f.write(f"宝箱{strategy}")
                    
                f.write("\n\n")
            
            f.write("## 结论\n\n")
            
            # 基于收敛状态给出不同的结论
            if stability_analysis["is_converged"]:
                f.write(f"模拟结果表明，在给定参数下，博弈最终收敛到宝箱{stability_analysis['convergence_box']}作为最优选择。")
                f.write(f"这表明策略{stability_analysis['convergence_box']}是在当前参数集下的纳什均衡。\n\n")
                
                # 检查是否与初始预测一致
                initial_best = rounds_data[0]["best_single"]
                if initial_best == stability_analysis['convergence_box']:
                    f.write("最终收敛结果与初始预测一致，表明初始模型参数设置合理且稳定。\n")
                else:
                    f.write(f"最终收敛结果(宝箱{stability_analysis['convergence_box']})与初始预测(宝箱{initial_best})不同，")
                    f.write("这表明玩家行为的演变对最优策略有显著影响。\n")
            else:
                f.write("模拟结果表明，在给定参数下，博弈未能收敛到单一最优策略。")
                f.write("这种周期性或混沌行为可能表明存在混合策略均衡，或者玩家群体存在复杂的战略互动模式。\n\n")
                
                # 提供进一步的建议
                f.write("建议增加模拟轮数，或调整学习率和适应速度参数，以观察是否能在更长时间内收敛。\n")
            
            # 参数演变建议
            f.write("\n### 参数演变建议\n\n")
            
            # 分析理性玩家比例变化趋势
            first_rational = parameter_history[0]["rational_pct"]
            last_rational = parameter_history[-1]["rational_pct"]
            
            if last_rational > first_rational:
                f.write("- 理性玩家比例呈现增长趋势，建议在未来模型中提高理性玩家的初始比例。\n")
            else:
                f.write("- 理性玩家比例呈现下降趋势，这可能表明在当前环境中，完全理性策略不一定是最有效的。\n")
            
            # 分析第二宝箱选择比例变化趋势
            first_second_box = parameter_history[0]["second_box_pct"]
            last_second_box = parameter_history[-1]["second_box_pct"]
            
            if last_second_box > first_second_box:
                f.write("- 第二宝箱选择比例呈现增长趋势，这可能表明双选策略在某些情况下具有优势。\n")
            else:
                f.write("- 第二宝箱选择比例呈现下降趋势，这强化了单选策略在当前成本结构下的优势地位。\n")
        
        print(f"动态博弈模拟报告已保存到 {report_path}")
    
    def run_full_analysis(self, num_rounds: int = 10) -> Dict:
        """
        运行完整的动态博弈分析
        
        参数:
            num_rounds: 博弈轮数
            
        返回:
            分析结果字典
        """
        print(f"开始运行动态博弈分析 (轮数: {num_rounds})...")
        
        # 模拟动态博弈
        simulation_results = self.simulate_dynamic_game(num_rounds)
        
        # 可视化参数演变
        self.visualize_parameter_evolution(simulation_results)
        
        # 可视化宝箱选择
        self.visualize_box_selection(simulation_results)
        
        # 可视化收益演变
        self.visualize_profit_evolution(simulation_results)
        
        # 可视化玩家策略
        self.visualize_player_strategies(simulation_results)
        
        # 分析策略稳定性
        stability_analysis = self.analyze_strategy_stability(simulation_results)
        
        # 生成报告
        self.generate_dynamic_game_report(simulation_results, stability_analysis)
        
        return {
            "simulation_results": simulation_results,
            "stability_analysis": stability_analysis
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
    
    # 创建动态博弈模拟器
    simulator = DynamicGameSimulator(
        treasures=treasures,
        output_dir="output/dynamic_game",
        initial_rational_pct=0.35,
        initial_heuristic_pct=0.45,
        initial_second_box_pct=0.05,
        initial_second_choice_rational_factor=0.7,
        learning_rate=0.05,
        random_exploration=0.1
    )
    
    # 运行完整分析
    simulator.run_full_analysis(num_rounds=10)


if __name__ == "__main__":
    main() 