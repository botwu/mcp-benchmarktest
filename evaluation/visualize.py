# 可视化结果
import matplotlib.pyplot as plt
import numpy as np
import matplotlib as mpl

# 设置中文字体
import matplotlib.font_manager as fm
# 查找系统中的中文字体，例如
font_path = '/System/Library/Fonts/STHeiti Light.ttc'  # 路径需要根据系统调整
font_prop = fm.FontProperties(fname=font_path)
plt.rcParams['font.family'] = font_prop.get_name()

def plot_model_comparison(model_names, scores, metric_name):
    """
    绘制模型比较图表
    
    Args:
        model_names: 模型名称列表
        scores: 对应模型的分数列表
        metric_name: 指标名称
    """
    plt.figure(figsize=(10, 6))
    plt.bar(model_names, scores, color='skyblue')
    plt.xlabel('模型')
    plt.ylabel(f'{metric_name}分数')
    plt.title(f'不同模型的{metric_name}比较')
    plt.ylim(0, 1)
    
    # 添加具体数值标签
    for i, score in enumerate(scores):
        plt.text(i, score + 0.02, f'{score:.2f}', ha='center')
        
    plt.tight_layout()
    plt.savefig(f'{metric_name}_comparison.png')
    
def plot_difficulty_breakdown(difficulty_scores, model_names):
    """
    绘制不同难度级别的得分分解图
    
    Args:
        difficulty_scores: 每个难度级别每个模型的得分字典
        model_names: 模型名称列表
    """
    difficulties = ["easy", "medium", "hard"]
    width = 0.25
    x = np.arange(len(difficulties))
    
    plt.figure(figsize=(12, 7))
    
    for i, model in enumerate(model_names):
        scores = [difficulty_scores[diff][model] for diff in difficulties]
        plt.bar(x + i*width, scores, width, label=model)
    
    plt.xlabel('难度级别')
    plt.ylabel('得分')
    plt.title('不同难度级别的模型表现')
    plt.xticks(x + width, difficulties)
    plt.ylim(0, 1)
    plt.legend()
    
    plt.tight_layout()
    plt.savefig('difficulty_breakdown.png')

def plot_capability_radar(model_names, metrics_data):
    """
    绘制能力雷达图
    
    Args:
        model_names: 模型名称列表
        metrics_data: 每个模型的评估指标数据
    """
    categories = ['意图识别', '工具匹配', '参数提取']
    
    fig = plt.figure(figsize=(10, 8))
    ax = fig.add_subplot(111, polar=True)
    
    angles = np.linspace(0, 2*np.pi, len(categories), endpoint=False).tolist()
    angles += angles[:1]
    
    for i, model in enumerate(model_names):
        values = [
            metrics_data[model].get("intent", {}).get("accuracy", 0),
            metrics_data[model].get("tool", {}).get("exact_match", 0),
            metrics_data[model].get("parameter", {}).get("exact_match", 0)
        ]
        values += values[:1]
        
        ax.plot(angles, values, linewidth=2, label=model)
        ax.fill(angles, values, alpha=0.1)
    
    plt.xticks(angles[:-1], categories)
    plt.yticks([0.2, 0.4, 0.6, 0.8, 1.0], ['0.2', '0.4', '0.6', '0.8', '1.0'], color='gray')
    plt.ylim(0, 1)
    
    plt.legend(loc='upper right')
    plt.savefig('capability_radar.png')

def plot_intent_similarity_heatmap(predictions, test_cases, similarity_threshold=0.7):
    """
    绘制意图语义相似度热力图
    
    Args:
        predictions: 模型预测结果列表
        test_cases: 测试用例列表
        similarity_threshold: 相似度阈值，用于高亮显示
    """
    # 收集所有意图
    all_intents = set()
    for pred, case in zip(predictions, test_cases):
        all_intents.add(case.get("expected_intent"))
        all_intents.add(pred.get("detected_intent"))
    
    all_intents = sorted(list(all_intents))
    n_intents = len(all_intents)
    
    # 创建相似度矩阵
    similarity_matrix = np.zeros((n_intents, n_intents))
    
    # 计算语义相似度
    from sklearn.feature_extraction.text import TfidfVectorizer
    from sklearn.metrics.pairwise import cosine_similarity
    
    vectorizer = TfidfVectorizer()
    vectors = vectorizer.fit_transform(all_intents)
    
    for i in range(n_intents):
        for j in range(n_intents):
            if i == j:
                similarity_matrix[i][j] = 1.0
            else:
                similarity_matrix[i][j] = cosine_similarity(
                    vectors[i:i+1], vectors[j:j+1]
                )[0][0]
    
    # 绘制热力图 (使用matplotlib替代seaborn)
    plt.figure(figsize=(12, 10))
    
    # 创建一个掩码，只显示下三角部分
    mask = np.zeros_like(similarity_matrix, dtype=bool)
    mask[np.triu_indices_from(mask, k=1)] = True
    
    # 使用matplotlib绘制热力图
    plt.imshow(similarity_matrix, cmap='YlGnBu', interpolation='nearest')
    plt.colorbar(label='相似度')
    
    # 添加文本显示具体相似度值
    for i in range(n_intents):
        for j in range(n_intents):
            if not mask[i, j]:  # 只在下三角和对角线显示
                plt.text(j, i, f"{similarity_matrix[i, j]:.2f}",
                        ha="center", va="center", color="black" if similarity_matrix[i, j] < 0.5 else "white")
    
    plt.xticks(range(n_intents), all_intents, rotation=45, ha='right')
    plt.yticks(range(n_intents), all_intents)
    plt.title("意图语义相似度矩阵")
    plt.tight_layout()
    plt.savefig("intent_similarity_heatmap.png")
    
    # 计算混淆矩阵
    confusion = np.zeros((n_intents, n_intents))
    intent_to_idx = {intent: i for i, intent in enumerate(all_intents)}
    
    for pred, case in zip(predictions, test_cases):
        expected = case.get("expected_intent")
        detected = pred.get("detected_intent")
        if expected in intent_to_idx and detected in intent_to_idx:
            confusion[intent_to_idx[expected]][intent_to_idx[detected]] += 1
    
    # 绘制混淆矩阵 (使用matplotlib替代seaborn)
    plt.figure(figsize=(12, 10))
    plt.imshow(confusion, cmap='Blues', interpolation='nearest')
    plt.colorbar(label='频率')
    
    # 添加具体数值标签
    for i in range(n_intents):
        for j in range(n_intents):
            plt.text(j, i, f"{int(confusion[i, j])}",
                    ha="center", va="center", color="black" if confusion[i, j] < np.max(confusion)/2 else "white")
    
    plt.xlabel('预测意图')
    plt.ylabel('实际意图')
    plt.xticks(range(n_intents), all_intents, rotation=45, ha='right')
    plt.yticks(range(n_intents), all_intents)
    plt.title("意图识别混淆矩阵")
    plt.tight_layout()
    plt.savefig("intent_confusion_matrix.png")

def plot_parameter_importance(metrics_by_model, test_cases):
    """
    可视化参数重要性与提取准确率的关系
    
    Args:
        metrics_by_model: 每个模型的参数提取评估结果字典
        test_cases: 测试用例列表
    """
    # 收集所有参数及其重要性
    param_importance = {}
    param_accuracy = {model: {} for model in metrics_by_model.keys()}
    
    # 遍历测试用例，收集参数重要性数据
    for case in test_cases:
        for tool_call in case.get("expected_tool_calls", []):
            tool_name = tool_call["name"]
            importance = tool_call.get("param_importance", {})
            
            for param, weight in importance.items():
                key = f"{tool_name}.{param}"
                if key not in param_importance:
                    param_importance[key] = weight
                    for model in metrics_by_model:
                        param_accuracy[model][key] = []
    
    # 计算每个参数在每个模型上的准确率
    for model, metrics in metrics_by_model.items():
        predictions = metrics.get("predictions", [])
        for pred, case in zip(predictions, test_cases):
            pred_tools = pred.get("tool_calls", [])
            expected_tools = case.get("expected_tool_calls", [])
            
            for exp_tool in expected_tools:
                exp_name = exp_tool["name"]
                exp_params = exp_tool.get("parameters", {})
                
                for pred_tool in pred_tools:
                    if pred_tool["name"] == exp_name:
                        pred_params = pred_tool.get("parameters", {})
                        
                        for param_name, exp_value in exp_params.items():
                            key = f"{exp_name}.{param_name}"
                            if key in param_importance:
                                is_correct = (
                                    param_name in pred_params and 
                                    pred_params[param_name] == exp_value
                                )
                                param_accuracy[model][key].append(1.0 if is_correct else 0.0)
    
    # 计算平均准确率
    for model in param_accuracy:
        for param in param_accuracy[model]:
            values = param_accuracy[model][param]
            param_accuracy[model][param] = sum(values) / len(values) if values else 0
    
    # 绘制参数重要性与准确率的关系图
    plt.figure(figsize=(14, 8))
    
    # 按重要性排序
    sorted_params = sorted(
        param_importance.keys(), 
        key=lambda x: param_importance[x], 
        reverse=True
    )[:15]  # 取前15个参数，避免图表过于拥挤
    
    x = np.arange(len(sorted_params))
    width = 0.8 / len(metrics_by_model)
    
    for i, model in enumerate(metrics_by_model):
        accuracy = [param_accuracy[model].get(param, 0) for param in sorted_params]
        plt.bar(x + i*width, accuracy, width, label=model)
    
    plt.xlabel('参数（按重要性排序）')
    plt.ylabel('提取准确率')
    plt.title('不同模型在关键参数上的提取能力')
    plt.xticks(x + width/2, [p.split(".")[-1] for p in sorted_params], rotation=45, ha="right")
    plt.ylim(0, 1)
    plt.legend()
    
    # 添加重要性标记
    for i, param in enumerate(sorted_params):
        plt.text(
            i, 
            -0.05, 
            f"重要性: {param_importance[param]:.1f}", 
            ha='center', 
            rotation=45,
            fontsize=8
        )
    
    plt.tight_layout()
    plt.savefig('parameter_importance_accuracy.png')