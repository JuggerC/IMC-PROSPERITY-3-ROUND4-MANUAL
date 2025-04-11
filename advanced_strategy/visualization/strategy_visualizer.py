import numpy as np
from typing import Dict, List, Tuple, Any, Optional
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from matplotlib.figure import Figure
import networkx as nx
from ..meta_strategy.meta_strategy_integrator import MetaStrategyIntegrator

class StrategyVisualizer:
    """
    策略可视化器
    
    提供丰富的可视化功能，包括:
    - 策略分布可视化
    - 对手建模结果可视化
    - 社会网络可视化
    - 性能指标可视化
    - 策略演化可视化
    """
    
    def __init__(self, meta_strategy: Optional[MetaStrategyIntegrator] = None, 
               figsize: Tuple[int, int] = (10, 6)):
        """
        初始化策略可视化器
        
        参数:
            meta_strategy: 元策略整合器实例，可选
            figsize: 图形尺寸
        """
        self.meta_strategy = meta_strategy
        self.figsize = figsize
        
        # 设置风格
        sns.set(style="whitegrid")
        
    def set_meta_strategy(self, meta_strategy: MetaStrategyIntegrator) -> None:
        """
        设置元策略整合器
        
        参数:
            meta_strategy: 元策略整合器实例
        """
        self.meta_strategy = meta_strategy
        
    def plot_strategy_distribution(self, distribution: np.ndarray, 
                               title: str = "Strategy Distribution", 
                               strategy_names: Optional[List[str]] = None) -> Figure:
        """
        绘制策略分布
        
        参数:
            distribution: 策略分布数组
            title: 图表标题
            strategy_names: 策略名称列表，可选
            
        返回:
            matplotlib Figure对象
        """
        # 创建图形
        fig, ax = plt.subplots(figsize=self.figsize)
        
        # 准备数据
        num_strategies = len(distribution)
        strategy_labels = strategy_names if strategy_names else [f"Strategy {i+1}" for i in range(num_strategies)]
        
        # 绘制条形图
        bars = ax.bar(strategy_labels, distribution, color=sns.color_palette("muted"))
        
        # 添加数值标签
        for bar in bars:
            height = bar.get_height()
            ax.text(bar.get_x() + bar.get_width()/2., height,
                  f'{height:.2f}',
                  ha='center', va='bottom')
        
        # 设置标题和轴标签
        ax.set_title(title, fontsize=14)
        ax.set_xlabel("Strategy", fontsize=12)
        ax.set_ylabel("Probability", fontsize=12)
        
        # 设置y轴范围
        ax.set_ylim(0, 1.0)
        
        # 添加网格线
        ax.grid(axis='y', linestyle='--', alpha=0.7)
        
        # 调整布局
        plt.tight_layout()
        
        return fig
        
    def plot_payoff_matrix(self, payoff_matrix: np.ndarray, 
                        strategy_names: Optional[List[str]] = None) -> Figure:
        """
        绘制收益矩阵热图
        
        参数:
            payoff_matrix: 收益矩阵
            strategy_names: 策略名称列表，可选
            
        返回:
            matplotlib Figure对象
        """
        # 创建图形
        fig, ax = plt.subplots(figsize=self.figsize)
        
        # 准备数据
        num_strategies = payoff_matrix.shape[0]
        strategy_labels = strategy_names if strategy_names else [f"Strategy {i+1}" for i in range(num_strategies)]
        
        # 绘制热图
        sns.heatmap(payoff_matrix, annot=True, fmt=".2f", cmap="YlGnBu",
                   xticklabels=strategy_labels, yticklabels=strategy_labels,
                   ax=ax)
        
        # 设置标题和轴标签
        ax.set_title("Payoff Matrix", fontsize=14)
        ax.set_xlabel("Opponent Strategy", fontsize=12)
        ax.set_ylabel("My Strategy", fontsize=12)
        
        # 调整布局
        plt.tight_layout()
        
        return fig
        
    def plot_strategy_history(self, strategy_history: List[int], payoff_history: List[float],
                          strategy_names: Optional[List[str]] = None) -> Figure:
        """
        绘制策略选择历史和收益
        
        参数:
            strategy_history: 策略选择历史
            payoff_history: 收益历史
            strategy_names: 策略名称列表，可选
            
        返回:
            matplotlib Figure对象
        """
        # 创建图形
        fig, (ax1, ax2) = plt.subplots(2, 1, figsize=self.figsize, sharex=True)
        
        # 准备数据
        num_strategies = max(strategy_history) + 1 if strategy_history else 0
        strategy_labels = strategy_names if strategy_names else [f"Strategy {i+1}" for i in range(num_strategies)]
        rounds = list(range(1, len(strategy_history) + 1))
        
        # 绘制策略选择历史
        ax1.plot(rounds, [strategy_labels[s] for s in strategy_history], 'o-', color='blue')
        ax1.set_title("Strategy Selection History", fontsize=14)
        ax1.set_ylabel("Selected Strategy", fontsize=12)
        ax1.set_ylim(-0.5, len(strategy_labels) - 0.5)
        ax1.set_yticks(range(len(strategy_labels)))
        ax1.set_yticklabels(strategy_labels)
        ax1.grid(True)
        
        # 绘制收益历史
        ax2.plot(rounds, payoff_history, 'o-', color='green')
        
        # 绘制累计收益
        cumulative_payoff = np.cumsum(payoff_history)
        ax2_twin = ax2.twinx()
        ax2_twin.plot(rounds, cumulative_payoff, '--', color='red', alpha=0.7)
        ax2_twin.set_ylabel("Cumulative Payoff", fontsize=12, color='red')
        
        # 设置标题和轴标签
        ax2.set_title("Round Payoff", fontsize=14)
        ax2.set_xlabel("Round", fontsize=12)
        ax2.set_ylabel("Payoff", fontsize=12)
        ax2.grid(True)
        
        # 调整布局
        plt.tight_layout()
        
        return fig
        
    def plot_opponent_modeling(self, prediction_history: List[np.ndarray],
                           actual_history: List[int],
                           strategy_names: Optional[List[str]] = None) -> Figure:
        """
        绘制对手建模结果
        
        参数:
            prediction_history: 预测概率历史
            actual_history: 实际选择历史
            strategy_names: 策略名称列表，可选
            
        返回:
            matplotlib Figure对象
        """
        # 验证输入数据
        if not prediction_history or not actual_history:
            raise ValueError("Prediction history and actual history cannot be empty")
            
        # 创建图形
        fig, (ax1, ax2) = plt.subplots(2, 1, figsize=self.figsize)
        
        # 准备数据
        num_strategies = len(prediction_history[0])
        strategy_labels = strategy_names if strategy_names else [f"Strategy {i+1}" for i in range(num_strategies)]
        rounds = list(range(1, len(prediction_history) + 1))
        
        # 提取每个策略的预测概率
        strategy_probs = {
            label: [pred[i] for pred in prediction_history]
            for i, label in enumerate(strategy_labels)
        }
        
        # 绘制预测概率
        for label, probs in strategy_probs.items():
            ax1.plot(rounds, probs, '-', label=label)
            
        # 高亮实际选择
        for i, actual in enumerate(actual_history):
            if i < len(prediction_history):
                ax1.plot(i+1, prediction_history[i][actual], 'o', markersize=8, 
                       color='red', alpha=0.7)
        
        ax1.set_title("Opponent Strategy Prediction Probability", fontsize=14)
        ax1.set_ylabel("Prediction Probability", fontsize=12)
        ax1.set_ylim(0, 1.0)
        ax1.legend(loc='best')
        ax1.grid(True)
        
        # 绘制预测准确度
        # 准确度定义为: 预测的概率值在实际选择上的值
        accuracy = [pred[actual] for pred, actual in zip(prediction_history, actual_history)]
        ax2.plot(rounds, accuracy, 'o-', color='purple')
        
        # 计算移动平均准确度
        window_size = 3
        if len(accuracy) >= window_size:
            moving_avg = []
            for i in range(len(accuracy)):
                window = accuracy[max(0, i+1-window_size):i+1]
                moving_avg.append(sum(window) / len(window))
            ax2.plot(rounds, moving_avg, '--', color='blue', alpha=0.7, 
                   label=f"{window_size}轮移动平均")
            ax2.legend(loc='best')
        
        # 设置标题和轴标签
        ax2.set_title("Prediction Accuracy", fontsize=14)
        ax2.set_xlabel("Round", fontsize=12)
        ax2.set_ylabel("Accuracy", fontsize=12)
        ax2.set_ylim(0, 1.0)
        ax2.grid(True)
        
        # 调整布局
        plt.tight_layout()
        
        return fig
        
    def plot_model_weights(self, model_weights: Dict[str, float]) -> Figure:
        """
        绘制模型权重
        
        参数:
            model_weights: 模型权重字典
            
        返回:
            matplotlib Figure对象
        """
        # 创建图形
        fig, ax = plt.subplots(figsize=self.figsize)
        
        # 准备数据
        models = list(model_weights.keys())
        weights = list(model_weights.values())
        
        # 绘制饼图
        wedges, texts, autotexts = ax.pie(weights, labels=None, autopct='%1.1f%%',
                                     startangle=90, shadow=False)
        
        # 自定义外观
        plt.setp(autotexts, size=10, weight="bold")
        
        # 添加图例
        ax.legend(wedges, models, title="Model", loc="center left", bbox_to_anchor=(1, 0, 0.5, 1))
        
        # 设置标题
        ax.set_title("Model Weight Distribution", fontsize=14)
        
        # 等比例显示
        ax.axis('equal')
        
        # 调整布局
        plt.tight_layout()
        
        return fig
        
    def plot_social_network(self, network: nx.Graph, 
                        node_strategies: Optional[List[int]] = None,
                        strategy_names: Optional[List[str]] = None) -> Figure:
        """
        绘制社会网络
        
        参数:
            network: NetworkX图对象
            node_strategies: 节点策略列表，可选
            strategy_names: 策略名称列表，可选
            
        返回:
            matplotlib Figure对象
        """
        # 创建图形
        fig, ax = plt.subplots(figsize=self.figsize)
        
        # 确定节点颜色
        if node_strategies:
            num_strategies = max(node_strategies) + 1
            strategy_labels = strategy_names if strategy_names else [f"Strategy {i+1}" for i in range(num_strategies)]
            
            # 创建颜色映射
            color_map = plt.cm.get_cmap('tab10', num_strategies)
            node_colors = [color_map(s) for s in node_strategies]
            
            # 创建图例句柄
            legend_handles = [plt.Line2D([0], [0], marker='o', color='w', 
                                      markerfacecolor=color_map(i), markersize=10, 
                                      label=strategy_labels[i])
                            for i in range(num_strategies)]
        else:
            node_colors = 'skyblue'
            legend_handles = []
        
        # 绘制网络
        pos = nx.spring_layout(network)
        nx.draw_networkx_nodes(network, pos, node_color=node_colors, node_size=200, alpha=0.8, ax=ax)
        nx.draw_networkx_edges(network, pos, alpha=0.3, ax=ax)
        
        # 添加图例
        if legend_handles:
            ax.legend(handles=legend_handles, title="Strategy", loc="upper right")
        
        # 设置标题
        ax.set_title("Social Network", fontsize=14)
        
        # 移除坐标轴
        ax.axis('off')
        
        # 调整布局
        plt.tight_layout()
        
        return fig
        
    def plot_behavioral_profile(self, behavioral_profile: Dict[str, Any]) -> Figure:
        """
        绘制行为模型概况
        
        参数:
            behavioral_profile: 行为模型概况字典
            
        返回:
            matplotlib Figure对象
        """
        # 创建图形
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=self.figsize)
        
        # 绘制主导情感
        emotion_data = behavioral_profile.get('dominant_emotion', {})
        if emotion_data:
            emotion_type = emotion_data.get('type', 'NEUTRAL')
            intensity = emotion_data.get('intensity', 0.0)
            
            # 绘制情感强度仪表盘
            ax1.pie([intensity, 1-intensity], colors=['red', 'lightgray'], startangle=90, 
                  wedgeprops=dict(width=0.3))
            ax1.text(0, 0, f"{emotion_type}\n{intensity:.2f}", ha='center', va='center', 
                   fontsize=12, fontweight='bold')
            ax1.set_title("Dominant Emotion", fontsize=14)
            ax1.axis('equal')
        
        # 绘制显著行为偏差
        biases = behavioral_profile.get('significant_biases', {})
        if biases:
            bias_names = list(biases.keys())
            bias_intensities = list(biases.values())
            
            # 绘制水平条形图
            bars = ax2.barh(bias_names, bias_intensities, color='green', alpha=0.7)
            
            # 添加数值标签
            for bar in bars:
                width = bar.get_width()
                ax2.text(width + 0.01, bar.get_y() + bar.get_height()/2.,
                      f'{width:.2f}',
                      ha='left', va='center')
            
            ax2.set_title("Significant Behavioral Biases", fontsize=14)
            ax2.set_xlabel("Intensity", fontsize=12)
            ax2.set_xlim(0, 1.0)
            ax2.grid(axis='x', linestyle='--', alpha=0.7)
        
        # 调整布局
        plt.tight_layout()
        
        return fig
        
    def plot_performance_summary(self, performance_summary: Dict[str, Any]) -> Figure:
        """
        绘制性能概要
        
        参数:
            performance_summary: 性能概要字典
            
        返回:
            matplotlib Figure对象
        """
        # 创建图形
        fig, ax = plt.subplots(figsize=self.figsize)
        
        # 提取关键指标
        metrics = {
            'Average Payoff': performance_summary.get('avg_payoff', 0.0),
            'Prediction Accuracy': performance_summary.get('prediction_accuracy', 0.0),
            'Total Payoff/Round': performance_summary.get('total_payoff', 0.0) / 
                        max(1, performance_summary.get('num_rounds', 1))
        }
        
        # 绘制雷达图
        # 准备雷达图角度
        categories = list(metrics.keys())
        N = len(categories)
        angles = np.linspace(0, 2*np.pi, N, endpoint=False).tolist()
        angles += angles[:1]  # 闭合图形
        
        # 准备数据
        values = list(metrics.values())
        values += values[:1]  # 闭合数据
        
        # 绘制雷达图
        ax.plot(angles, values, 'o-', linewidth=2)
        ax.fill(angles, values, alpha=0.25)
        
        # 设置雷达图的角度、标签
        ax.set_theta_offset(np.pi / 2)
        ax.set_theta_direction(-1)
        ax.set_thetagrids(np.degrees(angles[:-1]), categories)
        
        # 设置y轴范围
        ax.set_ylim(0, 1)
        
        # 添加标题
        ax.set_title("Performance Summary", fontsize=14)
        
        # 调整布局
        plt.tight_layout()
        
        return fig
        
    def visualize_meta_strategy(self) -> Dict[str, Figure]:
        """
        可视化元策略整合器的各种输出
        
        返回:
            图形字典，键为图形名称，值为matplotlib Figure对象
        """
        if self.meta_strategy is None:
            raise ValueError("未设置元策略整合器")
            
        # 收集数据
        figures = {}
        
        # 1. 绘制策略分布
        if hasattr(self.meta_strategy, 'current_distribution') and self.meta_strategy.current_distribution is not None:
            figures['strategy_distribution'] = self.plot_strategy_distribution(
                self.meta_strategy.current_distribution,
                title="Current Strategy Distribution"
            )
        
        # 2. 绘制收益矩阵
        if hasattr(self.meta_strategy, 'payoff_matrix') and self.meta_strategy.payoff_matrix is not None:
            figures['payoff_matrix'] = self.plot_payoff_matrix(
                self.meta_strategy.payoff_matrix
            )
        
        # 3. 绘制策略历史
        if (hasattr(self.meta_strategy, 'strategy_history') and self.meta_strategy.strategy_history and
            hasattr(self.meta_strategy, 'payoff_history') and self.meta_strategy.payoff_history):
            figures['strategy_history'] = self.plot_strategy_history(
                self.meta_strategy.strategy_history,
                self.meta_strategy.payoff_history
            )
        
        # 4. 绘制对手建模结果
        if hasattr(self.meta_strategy, 'opponent_modeler'):
            opponent_modeler = self.meta_strategy.opponent_modeler
            if (opponent_modeler.prediction_history and opponent_modeler.opponent_history):
                figures['opponent_modeling'] = self.plot_opponent_modeling(
                    opponent_modeler.prediction_history,
                    opponent_modeler.opponent_history
                )
        
        # 5. 绘制模型权重
        if hasattr(self.meta_strategy, 'integration_weights'):
            figures['model_weights'] = self.plot_model_weights(
                self.meta_strategy.integration_weights
            )
        
        # 6. 绘制行为模型概况
        if hasattr(self.meta_strategy, 'get_behavioral_profile'):
            behavioral_profile = self.meta_strategy.get_behavioral_profile()
            figures['behavioral_profile'] = self.plot_behavioral_profile(
                behavioral_profile
            )
        
        # 7. 绘制性能概要
        if hasattr(self.meta_strategy, 'get_performance_summary'):
            performance_summary = self.meta_strategy.get_performance_summary()
            figures['performance_summary'] = self.plot_performance_summary(
                performance_summary
            )
        
        return figures
        
    def save_figures(self, figures: Dict[str, Figure], output_dir: str,
                  file_format: str = 'png', dpi: int = 300) -> None:
        """
        保存图形到文件
        
        参数:
            figures: 图形字典，键为图形名称，值为matplotlib Figure对象
            output_dir: 输出目录
            file_format: 文件格式，如'png', 'pdf'等
            dpi: 图像分辨率
        """
        import os
        
        # 确保输出目录存在
        os.makedirs(output_dir, exist_ok=True)
        
        # 保存每个图形
        for name, fig in figures.items():
            filepath = os.path.join(output_dir, f"{name}.{file_format}")
            fig.savefig(filepath, format=file_format, dpi=dpi)
            
        print(f"保存了 {len(figures)} 个图形到 {output_dir}")
        
    def close_figures(self, figures: Dict[str, Figure]) -> None:
        """
        关闭图形对象
        
        参数:
            figures: 图形字典
        """
        for fig in figures.values():
            plt.close(fig) 