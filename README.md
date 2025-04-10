# MCP 基准测试

这是一个用于评估大型语言模型（LLM）多通道感知（Multi-Channel Perception，MCP）能力的基准测试框架。该框架允许用户评估不同模型使用工具的能力，并对比不同模型在各种复杂度任务上的表现。

## 项目结构

```shell
mcp-benchmark/
│
├── config/
│   ├── __init__.py       # 配置模块初始化
│   ├── models.py         # 模型配置
│   └── tools.py          # MCP工具定义
│
├── data/
│   ├── __init__.py       # 数据模块初始化
│   ├── test_cases.py     # 测试用例定义
│   └── datasets.py       # 数据集加载器
│
├── models/
│   ├── __init__.py       # 模型模块初始化
│   ├── base.py           # 模型基类
│   ├── qwen.py           # Qwen模型客户端
│   └── deepseek.py       # Deepseek模型客户端
│
├── evaluation/
│   ├── __init__.py       # 评估模块初始化
│   ├── metrics.py        # 评估指标
│   └── visualize.py      # 可视化结果
│
├── utils/
│   ├── __init__.py       # 工具模块初始化
│   ├── logger.py         # 日志工具
│   └── helpers.py        # 辅助函数
│
├── main.py               # 主程序入口
├── requirements.txt      # 项目依赖
└── README.md             # 项目说明
```

## 安装步骤

### 前提条件

- Python 3.8 或更高版本
- pip（Python 包管理工具）

### 安装依赖

1. 克隆本项目到本地：

```bash
git clone https://github.com/botwu/mcp-benchmarktest
cd mcp-benchmark
```

2. 安装依赖包：

```bash
pip install -r requirements.txt
```

## 使用方法

### 基本使用

运行以下命令来执行基准测试：

```bash
python main.py
```

默认情况下，这将使用 Qwen 模型运行所有难度级别的测试用例。

### 命令行参数

可以通过命令行参数自定义基准测试的行为：

- `--model`：指定要评估的模型（可选值：`qwen`, `deepseek`, `all`；默认：`qwen`）
- `--output_dir`：结果输出目录（默认：`results`）
- `--api_key`：API密钥（默认：`None`）
- `--difficulty`：测试用例难度（可选值：`easy`, `medium`, `hard`, `all`；默认：`all`）

例如：

```bash
# 评估所有模型
python main.py --model all

# 仅评估 Deepseek 模型的困难测试用例
python main.py --model deepseek --difficulty hard

# 指定 API 密钥和自定义输出目录
python main.py --api_key your_api_key --output_dir custom_results
```

## 添加新模型

要添加新的模型，请按照以下步骤操作：

1. 在 `config/models.py` 文件中添加新模型的配置：

```python
MODEL_CONFIGS = {
    "qwen": { ... },
    "deepseek": { ... },
    "your_model": {
        "model_id": "your-model-id",
        "api_base": "https://api.example.com/v1",
        "max_tokens": 2048,
        "temperature": 0.7,
    }
}
```

2. 在 `models` 目录中创建新的模型客户端文件（例如 `your_model.py`）：

```python
from .base import BaseModel

class YourModel(BaseModel):
    # 实现必要的方法
    ...
```

3. 更新 `main.py` 中的代码以支持新模型：

```python
def run_evaluation(model_name, test_cases, api_key=None):
    # ...
    if model_name == "qwen":
        model = QwenModel(config)
    elif model_name == "deepseek":
        model = DeepseekModel(config)
    elif model_name == "your_model":
        model = YourModel(config)
    # ...
```

## 添加新测试用例

要添加新的测试用例，请编辑 `data/test_cases.py` 文件：

```python
TEST_CASES = [
    # 现有测试用例...
    {
        "id": "your_test_case_id",
        "description": "测试用例描述",
        "query": "测试查询内容",
        "expected_intent": "查询",  # 预期的意图类别
        "expected_tools": ["tool1", "tool2"],
        "expected_tool_calls": [  # 包含参数重要性的预期工具调用
            {
                "name": "tool1",
                "parameters": {
                    "param1": "value1",
                    "param2": "value2"
                },
                "param_importance": {  # 参数重要性设置
                    "param1": 3.0,     # 高重要性
                    "param2": 1.0      # 标准重要性
                }
            }
        ],
        "difficulty": "medium"  # 可选：easy, medium, hard
    }
]
```

## 添加新工具

要添加新的工具，请编辑 `config/tools.py` 文件：

```python
TOOLS = [
    # 现有工具...
    {
        "name": "your_tool_name",
        "description": "工具描述",
        "parameters": {
            "param1": "参数1描述",
            "param2": "参数2描述"
        }
    }
]
```

## 结果说明

评估完成后，结果将保存在指定的输出目录中（默认为 `results/{timestamp}/`）：

- `{model}_results.json`：每个模型的原始评估结果
- `metrics.json`：评估指标汇总
- 如果评估了多个模型，还会生成一系列比较图表：
  - `总体得分_comparison.png`：模型总体得分比较
  - `工具使用准确率_comparison.png`：工具使用的准确率比较
  - `difficulty_breakdown.png`：不同难度级别下的表现比较
  - `capability_radar.png`：能力雷达图，展示各模型在意图识别、工具匹配和参数提取三个维度的能力
  - `intent_similarity_heatmap.png`：意图语义相似度热力图
  - `intent_confusion_matrix.png`：意图识别混淆矩阵
  - `parameter_importance_accuracy.png`：参数重要性与提取准确率关系图

## 评估指标

该框架使用以下指标来评估模型的性能：

### 关键能力评估

- **意图识别能力**：
  - **标准评估**：意图分类准确率和F1值（针对不平衡类别）
  - **语义相似度评估**：考虑意图间语义相似性的评估，使用精确匹配率、部分匹配率和语义加权准确率

- **工具匹配能力**：
  - **精确匹配率**：是否使用了完全正确的工具集
  - **部分匹配率**：部分正确工具的匹配程度

- **参数提取能力**：
  - **标准评估**：参数提取的精确匹配率和部分匹配率
  - **参数重要性评估**：考虑参数重要性权重的匹配率，区分必选参数和可选参数

### 综合评分

总体得分由以下权重组成：
- **意图识别**: 40%（包括准确率和F1值）
- **工具匹配**: 30%（包括精确匹配和部分匹配）
- **参数提取**: 30%（包括精确匹配和重要性加权匹配）

## 日志

日志文件将保存在 `logs` 目录中，命名格式为 `{logger_name}_{timestamp}.log`。

## 可视化功能

本框架提供了丰富的可视化功能，帮助您更直观地分析模型表现：

1. **模型比较图**：不同模型在各指标上的得分对比
2. **能力雷达图**：多维度展示各模型在不同能力上的表现
3. **难度分解图**：显示模型在不同难度级别测试用例上的表现
4. **意图语义相似度热力图**：展示不同意图之间的语义相关程度
5. **意图识别混淆矩阵**：显示模型在意图识别任务中的错误模式
6. **参数重要性分析图**：展示参数重要性与提取准确率的关系

所有图表支持中文显示，便于分析理解。

## 许可证

[添加您的许可证信息]