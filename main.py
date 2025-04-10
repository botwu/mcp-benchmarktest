# 主程序入口
import argparse
import os
from config.models import get_model_config
from config.tools import TOOLS
from data.test_cases import TEST_CASES
from models.qwen import QwenModel
from models.deepseek import DeepseekModel
from evaluation.metrics import (
    calculate_tool_usage_accuracy,
    calculate_response_relevance,
    calculate_response_correctness,
    calculate_overall_score
)
from evaluation.visualize import (
    plot_model_comparison,
    plot_difficulty_breakdown
)
from utils.logger import setup_logger
from utils.helpers import save_json, ensure_dir, get_timestamp

def parse_args():
    """解析命令行参数"""
    parser = argparse.ArgumentParser(description="MCP基准测试")
    parser.add_argument("--model", type=str, default="qwen", 
                        choices=["qwen", "deepseek", "all"],
                        help="要评估的模型")
    parser.add_argument("--output_dir", type=str, default="results",
                        help="结果输出目录")
    parser.add_argument("--api_key", type=str, default=None,
                        help="API密钥")
    parser.add_argument("--difficulty", type=str, default="all",
                        choices=["easy", "medium", "hard", "all"],
                        help="测试用例难度")
    return parser.parse_args()

def run_evaluation(model_name, test_cases, api_key=None):
    """运行模型评估"""
    logger = setup_logger("evaluation")
    logger.info(f"开始评估模型: {model_name}")
    
    # 获取模型配置
    config = get_model_config(model_name)
    if api_key:
        config["api_key"] = api_key
    
    # 初始化模型
    if model_name == "qwen":
        model = QwenModel(config)
    elif model_name == "deepseek":
        model = DeepseekModel(config)
    else:
        logger.error(f"不支持的模型: {model_name}")
        return None
    
    # 评估结果
    results = []
    
    # 对每个测试用例进行评估
    for case in test_cases:
        logger.info(f"评估测试用例: {case['id']}")
        
        # 生成回复
        response = model.generate_with_tools(case["query"], TOOLS)
        
        # 记录结果
        result = {
            "case_id": case["id"],
            "query": case["query"],
            "difficulty": case["difficulty"],
            "response": response,
            "expected_tools": case["expected_tools"]
        }
        results.append(result)
        
    return results

def main():
    """主函数"""
    args = parse_args()
    logger = setup_logger("main")
    logger.info("开始MCP基准测试")
    
    # 确保输出目录存在
    timestamp = get_timestamp()
    output_dir = os.path.join(args.output_dir, timestamp)
    ensure_dir(output_dir)
    
    # 筛选测试用例
    if args.difficulty != "all":
        filtered_cases = [case for case in TEST_CASES if case["difficulty"] == args.difficulty]
    else:
        filtered_cases = TEST_CASES
    
    # 确定要评估的模型
    models_to_evaluate = ["qwen", "deepseek"] if args.model == "all" else [args.model]
    
    all_results = {}
    for model_name in models_to_evaluate:
        logger.info(f"评估模型: {model_name}")
        results = run_evaluation(model_name, filtered_cases, args.api_key)
        all_results[model_name] = results
        
        # 保存原始结果
        save_json(results, os.path.join(output_dir, f"{model_name}_results.json"))
    
    # 计算评估指标
    metrics = {}
    for model_name, results in all_results.items():
        # 计算工具使用准确率
        tool_accuracy = calculate_tool_usage_accuracy(
            [r["response"] for r in results],
            [r for r in filtered_cases]
        )
        
        # 计算回复相关性分数
        relevance = calculate_response_relevance(
            [r["response"] for r in results],
            filtered_cases
        )
        
        # 计算回复正确性分数
        correctness = calculate_response_correctness(
            [r["response"] for r in results],
            filtered_cases
        )
        
        # 计算总体得分
        overall = calculate_overall_score({
            "tool_accuracy": tool_accuracy,
            "relevance": relevance,
            "correctness": correctness
        })
        
        metrics[model_name] = {
            "tool_accuracy": tool_accuracy,
            "relevance": relevance,
            "correctness": correctness,
            "overall": overall
        }
    
    # 保存评估指标
    save_json(metrics, os.path.join(output_dir, "metrics.json"))
    
    # 如果评估了多个模型，则生成比较图表
    if len(models_to_evaluate) > 1:
        model_names = list(metrics.keys())
        
        # 绘制总体得分比较图
        overall_scores = [metrics[model]["overall"] for model in model_names]
        plot_model_comparison(model_names, overall_scores, "总体得分")
        
        # 绘制工具使用准确率比较图
        tool_accuracy_scores = [metrics[model]["tool_accuracy"] for model in model_names]
        plot_model_comparison(model_names, tool_accuracy_scores, "工具使用准确率")
        
        # 绘制不同难度级别的分解图
        # 这里简化了，实际应用中需要计算每个难度级别的得分
        difficulty_scores = {
            "easy": {model: 0.9 for model in model_names},
            "medium": {model: 0.75 for model in model_names},
            "hard": {model: 0.6 for model in model_names}
        }
        plot_difficulty_breakdown(difficulty_scores, model_names)
    
    logger.info(f"评估完成，结果已保存到 {output_dir}")

if __name__ == "__main__":
    main()