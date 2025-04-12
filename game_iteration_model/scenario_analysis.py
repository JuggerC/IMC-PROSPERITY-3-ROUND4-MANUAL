#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
IMC Prosperity 宝箱选择优化模型 - 多情景分析和蒙特卡洛模拟
"""

import numpy as np
import matplotlib.pyplot as plt
from simulator import Treasure, TreasureOptimizer
import pandas as pd
from typing import Dict, List, Tuple
import os

class ScenarioAnalyzer:
    """多情景分析器"""
    
    def __init__(self, treasures: List[Treasure]):
        """
        初始化情景分析器
        
        参数:
            treasures: 宝箱列表
        """
        self.treasures = treasures
        self.scenarios = {}
        self.results = {}
        self.output_dir = self._create_output_dir()
    
    def _create_output_dir(self):
        """创建输出目录"""
        output_dir = os.path.join(os.path.dirname(__file__), "output")
        if not os.path.exists(output_dir):
            os.makedirs(output_dir)
        return output_dir
    
    def add_scenario(self, name: str, treasures: List[Treasure], 
                     rational_pct: float = 0.2, heuristic_pct: float = 0.5, 
                     second_box_pct: float = 0.1, second_box_cost: int = 50000,
                     second_choice_rational_factor: float = 1.0):
        """
        添加情景
        
        参数:
            name: 情景名称
            treasures: 宝箱列表
            rational_pct: 理性玩家比例
            heuristic_pct: 启发式玩家比例
            second_box_pct: 选择第二个宝箱的玩家比例
            second_box_cost: 选择第二个宝箱的成本
            second_choice_rational_factor: 第二次选择的理性调整因子
        """
        self.scenarios[name] = {
            'treasures': treasures,
            'rational_pct': rational_pct,
            'heuristic_pct': heuristic_pct,
            'second_box_pct': second_box_pct,
            'second_box_cost': second_box_cost,
            'second_choice_rational_factor': second_choice_rational_factor
        }
    
    def run_all_scenarios(self, max_iterations: int = 10, convergence_threshold: float = 0.01):
        """运行所有情景"""
        for name, scenario in self.scenarios.items():
            print(f"\n运行情景: {name}")
            
            # 创建优化器
            optimizer = TreasureOptimizer(
                treasures=scenario['treasures'],
                rational_pct=scenario['rational_pct'],
                heuristic_pct=scenario['heuristic_pct'],
                second_box_pct=scenario['second_box_pct'],
                second_box_cost=scenario['second_box_cost'],
                second_choice_rational_factor=scenario['second_choice_rational_factor']
            )
            
            # 运行迭代
            distribution, profits = optimizer.run_iteration(
                max_iterations=max_iterations,
                convergence_threshold=convergence_threshold
            )
            
            # 分析最优策略
            best_single, best_pair, best_single_profit, best_pair_profit = optimizer.analyze_optimal_strategy(profits)
            
            # 保存结果
            self.results[name] = {
                'distribution': distribution,
                'profits': profits,
                'best_single': best_single,
                'best_pair': best_pair,
                'best_single_profit': best_single_profit,
                'best_pair_profit': best_pair_profit
            }
    
    def compare_scenarios(self):
        """比较所有情景的结果"""
        if not self.results:
            print("没有情景结果可比较，请先运行run_all_scenarios()")
            return
        
        # 创建结果数据框
        data = []
        for scenario_name, result in self.results.items():
            for box_id, profit in result['profits'].items():
                data.append({
                    'Scenario': scenario_name,
                    'Box ID': box_id,
                    'Profit': profit,
                    'Distribution': result['distribution'][box_id] * 100  # 转换为百分比
                })
        
        df = pd.DataFrame(data)
        
        # 透视表展示每个情景下各宝箱的收益
        profit_pivot = df.pivot(index='Box ID', columns='Scenario', values='Profit')
        print("\n各情景下的宝箱收益对比:")
        print(profit_pivot)
        
        # 最佳选择比较
        print("\n各情景下的最佳选择:")
        for name, result in self.results.items():
            print(f"{name}:")
            print(f"  单选最佳: 宝箱{result['best_single']}, 收益: {result['best_single_profit']:.2f}")
            print(f"  双选最佳: 宝箱{result['best_pair'][0]}和宝箱{result['best_pair'][1]}, " + 
                  f"净收益: {result['best_pair_profit']:.2f}")
        
        # 计算各宝箱在不同情景下的收益方差，评估稳定性
        stability = profit_pivot.var(axis=1).sort_values()
        print("\n各宝箱收益稳定性评估 (方差，越低越稳定):")
        for box_id, variance in stability.items():
            print(f"宝箱{box_id}: {variance:.2f}")
        
        return profit_pivot, stability
    
    def plot_scenario_comparison(self):
        """绘制情景比较图"""
        if not self.results:
            print("没有情景结果可绘图，请先运行run_all_scenarios()")
            return
        
        profit_pivot, _ = self.compare_scenarios()
        
        # 创建输出目录
        output_dir = os.path.join(os.path.dirname(__file__), "output")
        if not os.path.exists(output_dir):
            os.makedirs(output_dir)
        
        plt.figure(figsize=(12, 8))
        profit_pivot.plot(kind='bar')
        plt.title('Expected Returns of Different Treasures Under Various Scenarios')
        plt.xlabel('Treasure ID')
        plt.ylabel('Expected Return')
        plt.xticks(rotation=0)
        plt.grid(axis='y', linestyle='--', alpha=0.7)
        plt.legend(title='Scenarios')
        plt.tight_layout()
        plt.savefig(os.path.join(output_dir, 'scenario_comparison.png'))
        plt.close()
        
        # 热力图
        plt.figure(figsize=(12, 8))
        plt.imshow(profit_pivot, cmap='viridis')
        plt.colorbar(label='Expected Return')
        plt.title('Heatmap of Expected Returns Under Various Scenarios')
        plt.xlabel('Scenarios')
        plt.ylabel('Treasure ID')
        plt.xticks(range(len(profit_pivot.columns)), profit_pivot.columns, rotation=45)
        plt.yticks(range(len(profit_pivot.index)), profit_pivot.index)
        
        # 在热力图上标注数值
        for i in range(len(profit_pivot.index)):
            for j in range(len(profit_pivot.columns)):
                plt.text(j, i, f'{profit_pivot.iloc[i, j]:.0f}',
                        ha='center', va='center', color='white')
        
        plt.tight_layout()
        plt.savefig(os.path.join(output_dir, 'scenario_heatmap.png'))
        plt.close()
    
    def analyze_second_box_impact(self, second_box_pcts=None, scenarios=None, rational_factors=None):
        """
        分析选择第二个宝箱比例和理性因子对收益的影响
        
        参数:
            second_box_pcts: 要测试的second_box_pct值列表，如果为None则使用默认值
            scenarios: 要分析的情景列表，每个情景是一个字典，包含rational_pct和heuristic_pct
            rational_factors: 要测试的second_choice_rational_factor值列表，如果为None则使用默认值
        """
        if second_box_pcts is None:
            second_box_pcts = [0.0, 0.05, 0.1, 0.2, 0.3, 0.5]
        
        if rational_factors is None:
            rational_factors = [0.5, 1.0, 1.5]
        
        if scenarios is None:
            scenarios = [
                {"name": "Baseline", "rational_pct": 0.2, "heuristic_pct": 0.5},
                {"name": "High Rational", "rational_pct": 0.4, "heuristic_pct": 0.4},
                {"name": "Low Rational", "rational_pct": 0.1, "heuristic_pct": 0.6}
            ]
        
        # 创建输出目录
        output_dir = os.path.join(os.path.dirname(__file__), "output")
        if not os.path.exists(output_dir):
            os.makedirs(output_dir)
        
        # 所有结果
        all_results = []
        
        # 为每个情景运行分析
        for scenario in scenarios:
            scenario_name = scenario["name"]
            rational_pct = scenario["rational_pct"]
            heuristic_pct = scenario["heuristic_pct"]
            
            print(f"\n分析情景 {scenario_name}:")
            
            # 为每个second_box_pct值和rational_factor值运行模拟
            for pct in second_box_pcts:
                for factor in rational_factors:
                    print(f"  选择第二个宝箱比例: {pct:.2f}, 理性因子: {factor:.1f}")
                    
                    optimizer = TreasureOptimizer(
                        treasures=self.treasures,
                        rational_pct=rational_pct,
                        heuristic_pct=heuristic_pct,
                        second_box_pct=pct,
                        second_choice_rational_factor=factor
                    )
                    
                    distribution, profits = optimizer.run_iteration(max_iterations=10)
                    best_single, best_pair, best_single_profit, best_pair_profit = optimizer.analyze_optimal_strategy(profits)
                    
                    all_results.append({
                        'scenario': scenario_name,
                        'rational_pct': rational_pct,
                        'heuristic_pct': heuristic_pct,
                        'second_box_pct': pct,
                        'second_choice_rational_factor': factor,
                        'best_single': best_single,
                        'best_pair': best_pair,
                        'single_profit': best_single_profit,
                        'pair_profit': best_pair_profit,
                        'diff': best_pair_profit - best_single_profit  # 双选相对于单选的净收益差
                    })
        
        # 创建数据框并打印结果
        df = pd.DataFrame(all_results)
        print("\n选择第二个宝箱比例对收益的影响 (所有情景):")
        pd.set_option('display.max_rows', None)
        print(df[['scenario', 'second_box_pct', 'second_choice_rational_factor', 'best_single', 'single_profit', 'best_pair', 'pair_profit', 'diff']])
        pd.reset_option('display.max_rows')
        
        # 为每个情景和理性因子组合绘制一张图表
        for scenario_name in df['scenario'].unique():
            for factor in rational_factors:
                scenario_factor_df = df[(df['scenario'] == scenario_name) & (df['second_choice_rational_factor'] == factor)]
                
                plt.figure(figsize=(10, 6))
                plt.plot(scenario_factor_df['second_box_pct'], scenario_factor_df['single_profit'], 'b-o', label='Best Single Box')
                plt.plot(scenario_factor_df['second_box_pct'], scenario_factor_df['pair_profit'], 'r-o', label='Best Pair (net)')
                plt.plot(scenario_factor_df['second_box_pct'], scenario_factor_df['diff'], 'g--', label='Difference (pair - single)')
                plt.axhline(y=0, color='gray', linestyle='--')
                
                plt.title(f'Impact of Second Box Percentage - {scenario_name} (Factor: {factor})')
                plt.xlabel('Second Box Selection Percentage')
                plt.ylabel('Profit')
                plt.grid(True, linestyle='--', alpha=0.7)
                plt.legend()
                plt.tight_layout()
                plt.savefig(os.path.join(output_dir, f'second_box_impact_{scenario_name}_factor_{factor}.png'))
                plt.close()
        
        # 为每个情景和固定的second_box_pct绘制理性因子影响图
        for scenario_name in df['scenario'].unique():
            for pct in second_box_pcts[1:]:  # 跳过0.0，因为当第二选择比例为0时理性因子无影响
                pct_df = df[(df['scenario'] == scenario_name) & (df['second_box_pct'] == pct)]
                if not pct_df.empty:
                    plt.figure(figsize=(10, 6))
                    plt.plot(pct_df['second_choice_rational_factor'], pct_df['single_profit'], 'b-o', label='Best Single Box')
                    plt.plot(pct_df['second_choice_rational_factor'], pct_df['pair_profit'], 'r-o', label='Best Pair (net)')
                    plt.plot(pct_df['second_choice_rational_factor'], pct_df['diff'], 'g--', label='Difference (pair - single)')
                    plt.axhline(y=0, color='gray', linestyle='--')
                    
                    plt.title(f'Impact of Rational Factor - {scenario_name} (Second Box Pct: {pct:.2f})')
                    plt.xlabel('Second Choice Rational Factor')
                    plt.ylabel('Profit')
                    plt.grid(True, linestyle='--', alpha=0.7)
                    plt.legend()
                    plt.tight_layout()
                    plt.savefig(os.path.join(output_dir, f'rational_factor_impact_{scenario_name}_pct_{pct:.2f}.png'))
                    plt.close()
        
        # 热力图展示理性因子与第二选择比例的交互作用
        for scenario_name in df['scenario'].unique():
            scenario_df = df[df['scenario'] == scenario_name]
            
            # 为单选收益创建透视表
            single_pivot = scenario_df.pivot_table(
                values='single_profit', 
                index='second_choice_rational_factor',
                columns='second_box_pct'
            )
            
            plt.figure(figsize=(12, 8))
            plt.imshow(single_pivot, cmap='viridis', aspect='auto', interpolation='nearest')
            plt.colorbar(label='Single Box Profit')
            plt.title(f'Interaction of Rational Factor and Second Box Pct - {scenario_name} (Single Box)')
            plt.xlabel('Second Box Percentage')
            plt.ylabel('Rational Factor')
            plt.xticks(range(len(single_pivot.columns)), [f'{pct:.2f}' for pct in single_pivot.columns])
            plt.yticks(range(len(single_pivot.index)), [f'{factor:.1f}' for factor in single_pivot.index])
            
            # 在热力图上标注数值
            for i in range(len(single_pivot.index)):
                for j in range(len(single_pivot.columns)):
                    plt.text(j, i, f'{single_pivot.iloc[i, j]:.0f}',
                            ha='center', va='center', color='white')
            
            plt.tight_layout()
            plt.savefig(os.path.join(output_dir, f'heatmap_single_{scenario_name}.png'))
            plt.close()
            
            # 为双选收益创建透视表
            pair_pivot = scenario_df.pivot_table(
                values='pair_profit', 
                index='second_choice_rational_factor',
                columns='second_box_pct'
            )
            
            plt.figure(figsize=(12, 8))
            plt.imshow(pair_pivot, cmap='viridis', aspect='auto', interpolation='nearest')
            plt.colorbar(label='Pair Profit')
            plt.title(f'Interaction of Rational Factor and Second Box Pct - {scenario_name} (Pair)')
            plt.xlabel('Second Box Percentage')
            plt.ylabel('Rational Factor')
            plt.xticks(range(len(pair_pivot.columns)), [f'{pct:.2f}' for pct in pair_pivot.columns])
            plt.yticks(range(len(pair_pivot.index)), [f'{factor:.1f}' for factor in pair_pivot.index])
            
            # 在热力图上标注数值
            for i in range(len(pair_pivot.index)):
                for j in range(len(pair_pivot.columns)):
                    plt.text(j, i, f'{pair_pivot.iloc[i, j]:.0f}',
                            ha='center', va='center', color='white')
            
            plt.tight_layout()
            plt.savefig(os.path.join(output_dir, f'heatmap_pair_{scenario_name}.png'))
            plt.close()
        
        # 返回结果
        return df


class MonteCarloSimulator:
    """蒙特卡洛模拟器"""
    
    def __init__(self, treasures: List[Treasure], base_distribution: Dict[int, float],
                 num_players: int = 10000, num_simulations: int = 1000,
                 distribution_variance: float = 0.1):
        """
        初始化蒙特卡洛模拟器
        
        参数:
            treasures: 宝箱列表
            base_distribution: 基准选择分布
            num_players: 模拟的玩家数量
            num_simulations: 模拟次数
            distribution_variance: 分布方差，用于生成随机扰动
        """
        self.treasures = {t.id: t for t in treasures}
        self.base_distribution = base_distribution
        self.num_players = num_players
        self.num_simulations = num_simulations
        self.distribution_variance = distribution_variance
        self.results = {}
        self.output_dir = self._create_output_dir()
    
    def _create_output_dir(self):
        """创建输出目录"""
        output_dir = os.path.join(os.path.dirname(__file__), "output")
        if not os.path.exists(output_dir):
            os.makedirs(output_dir)
        return output_dir
    
    def run_simulation(self):
        """运行蒙特卡洛模拟"""
        print(f"\n开始蒙特卡洛模拟 ({self.num_simulations}次)...")
        
        # 为每个宝箱准备结果存储
        profits = {box_id: [] for box_id in self.treasures.keys()}
        
        for sim in range(self.num_simulations):
            if sim % 100 == 0 and sim > 0:
                print(f"已完成 {sim} 次模拟...")
            
            # 生成带随机扰动的分布
            distribution = self._generate_random_distribution()
            
            # 计算各宝箱的收益
            for box_id, percentage in distribution.items():
                treasure = self.treasures[box_id]
                profit = treasure.calculate_profit(percentage * 100)
                profits[box_id].append(profit)
        
        # 计算统计数据
        stats = {}
        for box_id, profit_list in profits.items():
            stats[box_id] = {
                'mean': np.mean(profit_list),
                'std': np.std(profit_list),
                'min': np.min(profit_list),
                'max': np.max(profit_list),
                'sharpe': np.mean(profit_list) / np.std(profit_list) if np.std(profit_list) > 0 else 0,
                'profits': profit_list
            }
        
        self.results = stats
        return stats
    
    def _generate_random_distribution(self) -> Dict[int, float]:
        """
        生成带随机扰动的分布
        
        返回:
            带随机扰动的分布
        """
        # 生成随机扰动
        distribution = {}
        for box_id, base_prob in self.base_distribution.items():
            # 添加随机扰动，确保概率不为负
            random_prob = max(0, base_prob + np.random.normal(0, self.distribution_variance * base_prob))
            distribution[box_id] = random_prob
        
        # 归一化
        total = sum(distribution.values())
        for box_id in distribution:
            distribution[box_id] /= total
        
        return distribution
    
    def analyze_results(self):
        """分析模拟结果"""
        if not self.results:
            print("没有模拟结果可分析，请先运行run_simulation()")
            return
        
        # 整理统计数据
        data = []
        for box_id, stats in self.results.items():
            data.append({
                'Box ID': box_id,
                'Mean Profit': stats['mean'],
                'Std Dev': stats['std'],
                'Min Profit': stats['min'],
                'Max Profit': stats['max'],
                'Sharpe Ratio': stats['sharpe']
            })
        
        df = pd.DataFrame(data).sort_values('Mean Profit', ascending=False)
        
        print("\n蒙特卡洛模拟结果统计:")
        print(df)
        
        # 找出最佳单选和双选策略
        df_sorted = df.sort_values('Mean Profit', ascending=False)
        best_single = df_sorted.iloc[0]
        
        print("\n基于蒙特卡洛模拟的最佳策略:")
        print(f"单选最佳: 宝箱{int(best_single['Box ID'])}, " + 
              f"平均收益: {best_single['Mean Profit']:.2f}, " + 
              f"标准差: {best_single['Std Dev']:.2f}, " + 
              f"夏普比率: {best_single['Sharpe Ratio']:.2f}")
        
        # 分析双选策略
        second_box_cost = 50000
        best_pair = None
        best_pair_profit = 0
        best_pair_sharpe = 0
        
        for i in range(len(df_sorted)):
            for j in range(i+1, len(df_sorted)):
                box1 = df_sorted.iloc[i]
                box2 = df_sorted.iloc[j]
                pair_profit = box1['Mean Profit'] + box2['Mean Profit'] - second_box_cost
                
                # 计算组合夏普比率（假设两个宝箱的收益是独立的）
                pair_std = np.sqrt(box1['Std Dev']**2 + box2['Std Dev']**2)
                pair_sharpe = pair_profit / pair_std if pair_std > 0 else 0
                
                if pair_profit > best_pair_profit:
                    best_pair_profit = pair_profit
                    best_pair_sharpe = pair_sharpe
                    best_pair = (int(box1['Box ID']), int(box2['Box ID']))
        
        print(f"双选最佳: 宝箱{best_pair[0]}和宝箱{best_pair[1]}, " + 
              f"平均净收益: {best_pair_profit:.2f}, " + 
              f"夏普比率: {best_pair_sharpe:.2f}")
        
        # 最终推荐
        if best_pair_profit > best_single['Mean Profit']:
            print(f"\n最终推荐: 选择宝箱{best_pair[0]}和宝箱{best_pair[1]}, 预期净收益: {best_pair_profit:.2f}")
        else:
            print(f"\n最终推荐: 选择宝箱{int(best_single['Box ID'])}, 预期收益: {best_single['Mean Profit']:.2f}")
        
        return df
    
    def plot_results(self):
        """绘制模拟结果图表"""
        if not self.results:
            print("没有模拟结果可绘图，请先运行run_simulation()")
            return
        
        # 创建输出目录
        output_dir = os.path.join(os.path.dirname(__file__), "output")
        if not os.path.exists(output_dir):
            os.makedirs(output_dir)
        
        # 收益分布直方图
        plt.figure(figsize=(12, 8))
        for box_id, stats in self.results.items():
            plt.hist(stats['profits'], bins=30, alpha=0.3, label=f'Treasure {box_id}')
        
        plt.title('Return Distribution of Treasures')
        plt.xlabel('Return')
        plt.ylabel('Frequency')
        plt.legend()
        plt.grid(linestyle='--', alpha=0.7)
        plt.tight_layout()
        plt.savefig(os.path.join(output_dir, 'profit_distribution.png'))
        plt.close()
        
        # 风险收益散点图
        plt.figure(figsize=(10, 6))
        for box_id, stats in self.results.items():
            plt.scatter(stats['std'], stats['mean'], s=100, label=f'Treasure {box_id}')
            plt.annotate(f'{box_id}', (stats['std'], stats['mean']), 
                        fontsize=12, ha='center')
        
        plt.title('Risk-Return Analysis')
        plt.xlabel('Risk (Standard Deviation)')
        plt.ylabel('Return (Mean)')
        plt.grid(linestyle='--', alpha=0.7)
        plt.tight_layout()
        plt.savefig(os.path.join(output_dir, 'risk_return.png'))
        plt.close()


def main():
    """主函数"""
    # 创建宝箱列表
    treasures = [
        Treasure(box_id=1, multiplier=80, inhabitants=6),
        Treasure(box_id=2, multiplier=37, inhabitants=3),
        Treasure(box_id=3, multiplier=10, inhabitants=1),
        Treasure(box_id=4, multiplier=31, inhabitants=2),
        Treasure(box_id=5, multiplier=17, inhabitants=1),
        Treasure(box_id=6, multiplier=90, inhabitants=10),
        Treasure(box_id=7, multiplier=50, inhabitants=4),
        Treasure(box_id=8, multiplier=20, inhabitants=2),
        Treasure(box_id=9, multiplier=73, inhabitants=4),
        Treasure(box_id=10, multiplier=89, inhabitants=8)
    ]
    
    # 创建基准优化器
    print("运行基准模型...")
    base_optimizer = TreasureOptimizer(
        treasures=treasures,
        rational_pct=0.35,
        heuristic_pct=0.45,
        second_box_pct=0.05,
        second_choice_rational_factor=0.7
    )
    
    base_distribution, base_profits = base_optimizer.run_iteration()
    
    # 创建情景分析器
    print("\n=== 多情景分析 ===")
    scenario_analyzer = ScenarioAnalyzer(treasures)
    
    # 添加不同情景 - 使用更合理的参数设置
    scenario_analyzer.add_scenario(
        name="基本修正情景",
        treasures=treasures, 
        rational_pct=0.35, 
        heuristic_pct=0.45, 
        second_box_pct=0.05,
        second_choice_rational_factor=0.7
    )
    
    scenario_analyzer.add_scenario(
        name="高理性情景",
        treasures=treasures, 
        rational_pct=0.50, 
        heuristic_pct=0.40, 
        second_box_pct=0.03,
        second_choice_rational_factor=0.8
    )
    
    scenario_analyzer.add_scenario(
        name="高启发式情景",
        treasures=treasures, 
        rational_pct=0.25, 
        heuristic_pct=0.60, 
        second_box_pct=0.07,
        second_choice_rational_factor=0.6
    )
    
    # 对比原始参数的情景
    scenario_analyzer.add_scenario(
        name="原始参数情景",
        treasures=treasures, 
        rational_pct=0.2, 
        heuristic_pct=0.5, 
        second_box_pct=0.1,
        second_choice_rational_factor=1.0
    )
    
    # 测试不同second_choice_rational_factor的情景
    scenario_analyzer.add_scenario(
        name="第二选择完全随机",
        treasures=treasures, 
        rational_pct=0.35, 
        heuristic_pct=0.45, 
        second_box_pct=0.05,
        second_choice_rational_factor=0.0
    )
    
    scenario_analyzer.add_scenario(
        name="第二选择高度理性",
        treasures=treasures, 
        rational_pct=0.35, 
        heuristic_pct=0.45, 
        second_box_pct=0.05,
        second_choice_rational_factor=1.5
    )
    
    # 双选比例变化
    scenario_analyzer.add_scenario(
        name="无第二选择",
        treasures=treasures, 
        rational_pct=0.35, 
        heuristic_pct=0.45, 
        second_box_pct=0.0,
        second_choice_rational_factor=0.7
    )
    
    # 运行情景分析
    scenario_analyzer.run_all_scenarios()
    
    # 比较情景
    scenario_analyzer.compare_scenarios()
    
    # 绘制情景比较图
    scenario_analyzer.plot_scenario_comparison()
    
    # 分析第二个宝箱比例和理性因子对收益的影响
    second_box_pcts = [0.0, 0.03, 0.05, 0.07, 0.1]
    rational_factors = [0.0, 0.5, 0.7, 1.0, 1.5]
    
    # 只使用一个基准情景来减少组合数量
    base_scenario = [{"name": "基准情景", "rational_pct": 0.35, "heuristic_pct": 0.45}]
    
    scenario_analyzer.analyze_second_box_impact(
        second_box_pcts=second_box_pcts,
        scenarios=base_scenario,
        rational_factors=rational_factors
    )
    
    # 创建蒙特卡洛模拟器
    print("\n=== 蒙特卡洛模拟 ===")
    mc_simulator = MonteCarloSimulator(
        treasures=treasures,
        base_distribution=base_distribution,
        num_simulations=1000,
        distribution_variance=0.2
    )
    
    # 运行模拟
    mc_simulator.run_simulation()
    
    # 分析结果
    mc_simulator.analyze_results()
    
    # 绘制结果图
    mc_simulator.plot_results()


if __name__ == "__main__":
    main() 