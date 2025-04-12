# IMC Prosperity 宝箱选择优化与高级策略分析

本项目通过综合各种分析方法，为IMC Prosperity比赛中的宝箱选择问题提供最优决策支持。项目包含从博弈迭代模型到高级策略分析的完整演进路径，旨在探索最优宝箱选择策略。

## 项目概述

本项目研究IMC Prosperity中的宝箱选择问题，通过两种主要模型提供策略建议：
1. **博弈迭代模型**：基于纳什均衡和简单玩家分类的迭代收敛分析
2. **高级策略分析框架**：整合认知层次、行为经济学、社会动态和元策略的复合模型

两种模型互为补充，前者提供理论最优解，后者提供更稳健的实际决策建议。

## 项目文档导航

### 核心文档
- [模型比较分析](docs/model_comparison.md) - **新增！** 详细比较博弈迭代模型和高级策略模型的差异
- [模型演化路径](docs/model_evolution.md) - **新增！** 从博弈迭代模型到高级模型的发展历程
- [完整项目总结](comprehensive_project_summary.md) - 详细的系统设计、实现和结果分析
- [博弈迭代分析总结](game_iteration_model/SUMMARY.md) - 基础模型分析结果
- [高级策略分析报告](output/advanced_analysis/advanced_strategy_report.md) - 高级策略分析的输出报告

### 系统生成的可视化图表

- [认知层次模型策略分布](output/advanced_analysis/cognitive_distribution.png)
- [行为经济学模型权重](output/advanced_analysis/behavioral_weights.png)
- [社会动态模型最终分布](output/advanced_analysis/social_distribution.png)
- [收益矩阵热图](output/advanced_analysis/payoff_matrix.png)
- [社会动态演化过程](output/advanced_analysis/social_dynamics_evolution.png)
- [各模型最佳策略比较](output/advanced_analysis/model_comparison.png)

## 项目整体架构

```
      +------------------+
      |    用户界面       |
      | (命令行参数/报告) |
      +--------+---------+
               |
+-----------------------------+
|        分析整合层           |
|   (高级策略与博弈迭代整合)  |
+-----------------------------+
        /            \
+----------------+  +------------------+
|  博弈迭代模块   |  |  高级策略分析模块  |
| (博弈论模型)    |  | (多模型综合框架)   |
+-------+--------+  +----------+-------+
        |                      |
+----------------+  +------------------+
|    基础模型     |  |    认知层次分析   |
|                |  |    行为经济学     |
|                |  |    社会动态       |
|                |  |    元策略分析     |
+----------------+  +------------------+
```

## 项目结构

```
round2manual/
├── docs/                      # 项目文档
│   ├── model_comparison.md    # 模型比较分析
│   └── model_evolution.md     # 模型演化路径
├── advanced_strategy/         # 高级策略模块
│   ├── cognitive_hierarchy/   # 认知层次模型
│   ├── behavioral_economics/  # 行为经济学模型
│   ├── social_dynamics/       # 社会动态模型
│   ├── meta_strategy/         # 元策略模型
│   ├── visualization/         # 可视化工具
│   └── treasure_strategy_analyzer.py  # 宝箱策略分析器
├── game_iteration_model/      # 博弈迭代分析模块
│   ├── simulator.py           # 宝箱模拟器
│   ├── scenario_analysis.py   # 情景分析
│   ├── reverse_game_theory.py # 反向博弈分析
│   └── main.py                # 主分析脚本
├── advanced_treasure_analysis.py  # 高级策略分析主脚本
├── comprehensive_project_summary.md # 完整项目总结
└── output/                    # 分析结果输出
```

## 模型发展演化

本项目模型从简单到复杂经历了五个发展阶段：

```
基础博弈迭代模型 → 认知层次分析 → 行为经济学整合 → 社会动态模拟 → 元策略分析融合
```

每个阶段都引入了更丰富的理论框架，使模型能够更准确地模拟真实的人类决策过程。详见[模型演化路径](docs/model_evolution.md)文档。

## 两种模型的异同

| 对比维度 | 博弈迭代模型 | 高级策略模型 |
|---------|----------|-------------|
| 理论基础 | 博弈论、纳什均衡 | 认知层次理论、前景理论、社会动态、元策略分析 |
| 推荐宝箱 | 宝箱10 | 宝箱9 |
| 优势领域 | 理论最优收益 | 稳健性、多模型一致性 |
| 计算复杂度 | 中等 | 高 |

两种模型得出不同推荐的原因主要在于不同的决策标准和对人类行为的不同假设。详见[模型比较分析](docs/model_comparison.md)文档。

## 功能特点

### 博弈迭代分析功能

1. **基准模型**：建模不同类型玩家的宝箱选择行为
2. **多情景分析**：模拟不同玩家行为分布下的结果
3. **蒙特卡洛模拟**：评估不确定性和风险
4. **反向博弈分析**：分析主流选择和差异化价值
5. **敏感性分析**：测试参数变化对结果的影响

### 高级策略分析功能

1. **认知层次分析**：模拟不同思考层次的玩家行为
2. **行为经济学分析**：考虑前景理论、行为偏见和情绪因素
3. **社会动态分析**：模拟社交网络影响和社会规范演化
4. **元策略分析**：预测对手分布并计算最优反应策略
5. **综合分析**：整合多种模型的结果，提供全面的决策建议

## 如何使用

### 博弈迭代分析

运行博弈迭代分析：

```bash
cd round2manual
python game_iteration_model/main.py
```

### 高级策略分析

单独运行高级策略分析：

```bash
cd round2manual
python advanced_treasure_analysis.py
```

支持的命令行参数：
- `--rational`: 理性玩家比例 (默认: 0.35)
- `--heuristic`: 启发式玩家比例 (默认: 0.45)
- `--second-box-pct`: 选择第二个宝箱的玩家比例 (默认: 0.05)
- `--second-box-cost`: 选择第二个宝箱的成本 (默认: 50000)
- `--output-dir`: 输出目录 (默认: output/advanced_analysis)

例如：

```bash
python advanced_treasure_analysis.py --rational 0.4 --heuristic 0.4 --second-box-pct 0.1
```

### 整合分析

运行整合了高级策略分析的完整分析：

```bash
cd round2manual
python game_iteration_model/main.py --with-advanced
```

整合分析也支持高级策略分析的参数设置：

```bash
python game_iteration_model/main.py --with-advanced --rational 0.4 --heuristic 0.4
```

## 分析报告

运行分析后，可以查看以下文件获取结果：

- `game_iteration_model/output/`: 博弈迭代分析结果和图表
- `output/advanced_analysis/advanced_strategy_report.md`: 高级策略分析报告
- `game_iteration_model/SUMMARY.md`: 综合分析总结

## 开发团队

本项目由高级决策建模团队开发，专注于优化复杂博弈环境中的决策过程。

## 依赖库

```
matplotlib
numpy
pandas
networkx
seaborn
```

安装依赖：

```bash
pip install matplotlib numpy pandas networkx seaborn
``` 