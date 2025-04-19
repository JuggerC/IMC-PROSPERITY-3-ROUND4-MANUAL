# IMC Prosperity 第四轮高级策略分析系统

本项目为IMC Prosperity第四轮比赛开发的高级策略分析系统，通过整合多种决策理论模型，为宝箱选择问题提供最优决策支持。系统基于认知层次理论、行为经济学、社会动态分析和元策略分析，能够预测玩家行为并给出稳健的策略建议。

## 项目概述

本项目采用多维度分析方法，通过以下关键功能帮助玩家在充满不确定性和策略互动的环境中做出最优决策：

- **多模型集成分析**：整合认知层次理论、行为经济学、社会动态和元策略分析
- **精确的玩家行为建模**：考虑不同思考层次、行为偏见和社会影响
- **全面的策略评估**：评估单选、双选和三选策略的预期收益
- **稳健的决策推荐**：基于模型一致性提供可靠建议

## 主要特性

1. **认知层次分析**：模拟不同思考深度的玩家行为
2. **行为经济学分析**：考虑损失厌恶、风险偏好和行为偏见
3. **社会动态模拟**：预测群体行为演化和社会影响
4. **元策略优化**：整合多种模型结果，提供稳健推荐
5. **可视化决策支持**：生成清晰的分析图表和报告

## 项目结构

```
IMC-PROSPERITY-3-ROUND4-MANUAL/
├── advanced_strategy/         # 高级策略模块
│   ├── cognitive_hierarchy/   # 认知层次模型
│   ├── behavioral_economics/  # 行为经济学模型
│   ├── social_dynamics/       # 社会动态模型
│   ├── meta_strategy/         # 元策略模型
│   ├── visualization/         # 可视化工具
│   └── treasure_strategy_analyzer.py  # 宝箱策略分析器
├── models/                    # 基础模型
│   └── simulator.py           # 宝箱模拟器类
├── docs/                      # 项目文档
│   ├── model_comparison.md    # 模型比较分析
│   └── model_evolution.md     # 模型演化路径
├── output/                    # 分析结果输出
│   └── advanced_analysis/     # 高级策略分析结果
├── advanced_treasure_analysis.py  # 高级策略分析主脚本
└── comprehensive_project_summary.md # 完整项目总结
```

## 系统架构

```
      +------------------+
      |    用户界面       |
      | (命令行参数/报告) |
      +--------+---------+
               |
      +--------------------+
      |     分析整合层      |
      | (多模型结果整合)    |
      +--------------------+
               |
      +--------------------+
      |  高级策略分析模块   |
      | (多模型综合框架)    |
      +----------+---------+
               |
  +-------------+------------+------------+-----------+
  |             |            |            |           |
认知层次分析  行为经济学分析  社会动态分析  元策略分析  可视化模块
```

## 高级策略模型组件

| 分析模块 | 主要功能 | 理论基础 |
|---------|---------|---------|
| 认知层次分析 | 建模不同思考层级的玩家行为 | 认知层次理论、层级思考模型 |
| 行为经济学分析 | 考虑人类行为偏差和心理因素 | 前景理论、行为经济学 |
| 社会动态分析 | 模拟玩家互动和群体行为演化 | 社会网络理论、进化博弈论 |
| 元策略分析 | 计算对手策略分布下的最优回应 | 博弈论、贝叶斯均衡 |

## 如何使用

运行高级策略分析：

```bash
python advanced_treasure_analysis.py
```

支持的命令行参数：
- `--rational`: 理性玩家比例 (默认: 0.45)
- `--heuristic`: 启发式玩家比例 (默认: 0.35)
- `--second-box-pct`: 选择第二个宝箱的玩家比例 (默认: 0.15)
- `--second-box-cost`: 选择第二个宝箱的成本 (默认: 50000)
- `--third-box-pct`: 选择第三个宝箱的玩家比例 (默认: 0.05)
- `--third-box-cost`: 选择第三个宝箱的成本 (默认: 100000)
- `--num-players`: 玩家总数 (默认: 4130)
- `--output-dir`: 输出目录 (默认: output/advanced_analysis)

示例：

```bash
python advanced_treasure_analysis.py --rational 0.5 --heuristic 0.3 --second-box-pct 0.1
```

## 输出结果

运行分析后，系统将生成以下输出：

1. **分析报告**：`output/advanced_analysis/advanced_strategy_report.md`
2. **可视化图表**：
   - 认知层次分布图 (`cognitive_distribution.png`)
   - 行为经济学权重图 (`behavioral_weights.png`)
   - 社会动态演化图 (`social_dynamics_evolution.png`)
   - 模型比较图 (`model_comparison.png`)
   - 策略收益比较图 (`strategy_profit_comparison.png`)
   - 选择分布估计图 (`estimated_selection.png`)

### 示例可视化图表

#### 策略收益比较图
![策略收益比较图](https://raw.githubusercontent.com/JuggerC/IMC-PROSPERITY-3-ROUND4-MANUAL/main/output/advanced_analysis/strategy_profit_comparison.png)

#### 认知层次分布图
![认知层次分布图](https://raw.githubusercontent.com/JuggerC/IMC-PROSPERITY-3-ROUND4-MANUAL/main/output/advanced_analysis/cognitive_distribution.png)

#### 社会动态演化图
![社会动态演化图](https://raw.githubusercontent.com/JuggerC/IMC-PROSPERITY-3-ROUND4-MANUAL/main/output/advanced_analysis/social_dynamics_evolution.png)

## 决策建议

系统根据多模型分析提供三类策略建议：
1. **单选策略**：最优单个宝箱选择
2. **双选策略**：考虑成本后的最优两个宝箱组合
3. **三选策略**：考虑成本后的最优三个宝箱组合

每种策略都附带预期净收益估计，帮助决策者权衡不同选择。

## 依赖库

```
matplotlib==3.7.1
numpy==1.24.3
pandas==2.0.1
seaborn==0.12.2
networkx==3.1
```

安装依赖：

```bash
pip install matplotlib numpy pandas seaborn networkx
```

## 项目特色

- **理论融合**：整合多种决策理论，提供全面分析
- **可扩展性**：模块化设计，易于添加新的分析方法
- **易用性**：简单的命令行接口，丰富的可视化输出
- **稳健性**：多模型集成提高推荐可靠性

## 开发团队

本项目由高级决策建模团队开发，专注于优化复杂博弈环境中的策略决策。 