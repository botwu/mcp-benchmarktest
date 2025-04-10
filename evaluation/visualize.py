# 可视化结果
import matplotlib.pyplot as plt
import numpy as np

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