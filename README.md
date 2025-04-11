# IMC Prosperity 宝箱选择优化与高级策略分析

本项目通过综合各种分析方法，为IMC Prosperity比赛中的宝箱选择问题提供最优决策支持。

## 项目文档导航

- [完整项目总结](comprehensive_project_summary.md) - 详细的系统设计、实现和结果分析
- [原始分析总结](original_content/SUMMARY.md) - 基础模型分析结果
- [高级策略分析报告](output/advanced_analysis/advanced_strategy_report.md) - 高级策略分析的输出报告

## 系统生成的可视化图表

- [认知层次模型策略分布](output/advanced_analysis/cognitive_distribution.png)
- [行为经济学模型权重](output/advanced_analysis/behavioral_weights.png)
- [社会动态模型最终分布](output/advanced_analysis/social_distribution.png)
- [收益矩阵热图](output/advanced_analysis/payoff_matrix.png)
- [社会动态演化过程](output/advanced_analysis/social_dynamics_evolution.png)
- [各模型最佳策略比较](output/advanced_analysis/model_comparison.png)

## 项目结构

```
round2manual/
├── advanced_strategy/         # 高级策略模块
│   ├── cognitive_hierarchy/   # 认知层次模型
│   ├── behavioral_economics/  # 行为经济学模型
│   ├── social_dynamics/       # 社会动态模型
│   ├── meta_strategy/         # 元策略模型
│   ├── visualization/         # 可视化工具
│   └── treasure_strategy_analyzer.py  # 宝箱策略分析器
├── original_content/          # 原始分析模块
│   ├── simulator.py           # 宝箱模拟器
│   ├── scenario_analysis.py   # 情景分析
│   ├── reverse_game_theory.py # 反向博弈分析
│   └── main.py                # 主分析脚本
├── advanced_treasure_analysis.py  # 高级策略分析主脚本
├── comprehensive_project_summary.md # 完整项目总结
└── output/                    # 分析结果输出
```

## 功能特点

### 基础分析功能

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

### 基础分析

运行原始的基准分析：

```bash
cd round2manual
python original_content/main.py
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
python original_content/main.py --with-advanced
```

整合分析也支持高级策略分析的参数设置：

```bash
python original_content/main.py --with-advanced --rational 0.4 --heuristic 0.4
```

## 分析报告

运行分析后，可以查看以下文件获取结果：

- `original_content/output/`: 基础分析结果和图表
- `output/advanced_analysis/advanced_strategy_report.md`: 高级策略分析报告
- `original_content/SUMMARY.md`: 综合分析总结

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