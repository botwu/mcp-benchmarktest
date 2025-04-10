# 评估指标

def calculate_tool_usage_accuracy(predictions, test_cases):
    """
    计算工具使用的准确率
    
    Args:
        predictions: 模型预测结果列表
        test_cases: 测试用例列表
        
    Returns:
        float: 准确率分数
    """
    correct = 0
    total = len(predictions)
    
    for pred, case in zip(predictions, test_cases):
        expected_tools = set(case["expected_tools"])
        used_tools = set([tool_call["name"] for tool_call in pred.get("tool_calls", [])])
        
        if expected_tools == used_tools:
            correct += 1
            
    return correct / total if total > 0 else 0.0

def calculate_response_relevance(responses, test_cases):
    """
    计算回复相关性分数
    
    Args:
        responses: 模型回复列表
        test_cases: 测试用例列表
        
    Returns:
        float: 相关性分数
    """
    if not responses or not test_cases:
        return 0.0
    
    relevance_scores = []
    
    for response, test_case in zip(responses, test_cases):
        # 获取回复文本和查询
        response_text = response.get("text", "")
        query = test_case.get("query", "")
        
        # 相关性评估方法1：关键词匹配
        # 从查询中提取关键词
        keywords = extract_keywords(query)
        matched_keywords = sum(1 for keyword in keywords if keyword.lower() in response_text.lower())
        keyword_score = matched_keywords / len(keywords) if keywords else 0
        
        # 相关性评估方法2：计算语义相似度
        # 为简单起见，这里使用TF-IDF和余弦相似度
        semantic_score = calculate_semantic_similarity(query, response_text)
        
        # 综合得分 (可以调整权重)
        relevance_score = 0.4 * keyword_score + 0.6 * semantic_score
        relevance_scores.append(relevance_score)
    
    # 返回平均相关性分数
    return sum(relevance_scores) / len(relevance_scores)

def extract_keywords(text):
    """
    从文本中提取关键词
    
    Args:
        text: 输入文本
        
    Returns:
        list: 关键词列表
    """
    # 简单实现：移除常见停用词，分割文本
    import re
    
    # 简单的中文停用词列表
    stopwords = set(['的', '了', '和', '是', '在', '我', '有', '这', '个', '们',
                    '中', '到', '一', '为', '从', '以', '与', '及', '上', '下'])
    
    # 对于中文，按字符分割
    if re.search(r'[\u4e00-\u9fff]', text):
        words = list(text)
    else:
        # 对于英文，按空格分割
        words = text.split()
    
    # 移除停用词和标点符号
    keywords = [word for word in words 
                if word.lower() not in stopwords
                and re.match(r'[a-zA-Z\u4e00-\u9fff]+', word)]
    
    return keywords

def calculate_semantic_similarity(text1, text2):
    """
    计算两段文本的语义相似度
    
    Args:
        text1: 第一段文本
        text2: 第二段文本
        
    Returns:
        float: 相似度分数 (0-1)
    """
    # 使用TF-IDF和余弦相似度计算相似度
    from sklearn.feature_extraction.text import TfidfVectorizer
    from sklearn.metrics.pairwise import cosine_similarity
    import numpy as np
    
    # 处理空文本情况
    if not text1 or not text2:
        return 0.0
        
    try:
        # 创建TF-IDF向量化器
        vectorizer = TfidfVectorizer()
        # 转换文本为TF-IDF向量
        tfidf_matrix = vectorizer.fit_transform([text1, text2])
        # 计算余弦相似度
        similarity = cosine_similarity(tfidf_matrix[0:1], tfidf_matrix[1:2])[0][0]
        
        return float(similarity)
    except Exception as e:
        print(f"计算语义相似度时发生错误: {str(e)}")
        return 0.0

def calculate_intent_recognition(predictions, test_cases):
    """
    评估意图识别能力
    
    Args:
        predictions: 模型预测结果列表
        test_cases: 测试用例列表
        
    Returns:
        dict: 包含准确率和F1值的字典
    """
    # 根据用例分类统计正确识别的意图
    correct = 0
    true_positives = {}
    false_positives = {}
    false_negatives = {}
    
    for pred, case in zip(predictions, test_cases):
        expected_intent = case.get("expected_intent")
        detected_intent = pred.get("detected_intent")
        
        if expected_intent == detected_intent:
            correct += 1
            true_positives[expected_intent] = true_positives.get(expected_intent, 0) + 1
        else:
            false_positives[detected_intent] = false_positives.get(detected_intent, 0) + 1
            false_negatives[expected_intent] = false_negatives.get(expected_intent, 0) + 1
    
    # 计算F1值
    f1_scores = []
    for intent in set(true_positives.keys()) | set(false_positives.keys()) | set(false_negatives.keys()):
        tp = true_positives.get(intent, 0)
        fp = false_positives.get(intent, 0)
        fn = false_negatives.get(intent, 0)
        
        precision = tp / (tp + fp) if (tp + fp) > 0 else 0
        recall = tp / (tp + fn) if (tp + fn) > 0 else 0
        f1 = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0
        f1_scores.append(f1)
    
    return {
        "accuracy": correct / len(predictions) if predictions else 0,
        "f1": sum(f1_scores) / len(f1_scores) if f1_scores else 0
    }

def calculate_intent_recognition_with_similarity(predictions, test_cases, similarity_threshold=0.7):
    """
    评估意图识别能力（加入语义相似度）
    
    Args:
        predictions: 模型预测结果列表
        test_cases: 测试用例列表
        similarity_threshold: 语义相似度阈值，超过此值视为相似
        
    Returns:
        dict: 包含准确率、F1值和相似度评分的字典
    """
    # 基本统计
    exact_match = 0
    similar_match = 0
    all_intents = set()
    intent_sim_scores = []
    
    # 加载预训练词向量模型或调用语义模型
    # 这里使用示例函数，实际应使用如Word2Vec、BERT等模型计算相似度
    def calculate_semantic_similarity(intent1, intent2):
        # 实际实现中，可以使用如下方法：
        # 1. 预训练词向量的余弦相似度
        # 2. BERT/RoBERTa等模型输出的句向量相似度
        # 3. 调用第三方API如OpenAI嵌入模型计算相似度
        
        # 示例实现（替换为实际模型）
        from sklearn.feature_extraction.text import TfidfVectorizer
        from sklearn.metrics.pairwise import cosine_similarity
        import numpy as np
        
        if intent1 == intent2:
            return 1.0
            
        vectorizer = TfidfVectorizer()
        vectors = vectorizer.fit_transform([intent1, intent2])
        return cosine_similarity(vectors)[0, 1]
    
    # 评估每个预测
    for pred, case in zip(predictions, test_cases):
        expected_intent = case.get("expected_intent")
        detected_intent = pred.get("detected_intent")
        all_intents.add(expected_intent)
        
        # 计算语义相似度
        similarity = calculate_semantic_similarity(expected_intent, detected_intent)
        intent_sim_scores.append(similarity)
        
        # 匹配类型判断
        if expected_intent == detected_intent:
            exact_match += 1
        elif similarity >= similarity_threshold:
            similar_match += 1
    
    # 计算加权准确率，考虑相似度
    weighted_accuracy = sum(intent_sim_scores) / len(predictions) if predictions else 0
    
    # 计算标准指标
    exact_accuracy = exact_match / len(predictions) if predictions else 0
    partial_accuracy = (exact_match + similar_match) / len(predictions) if predictions else 0
    
    return {
        "exact_accuracy": exact_accuracy,
        "partial_accuracy": partial_accuracy,
        "semantic_accuracy": weighted_accuracy,
        "avg_similarity": sum(intent_sim_scores) / len(intent_sim_scores) if intent_sim_scores else 0
    }

def calculate_tool_matching(predictions, test_cases):
    """
    评估工具匹配能力
    
    Args:
        predictions: 模型预测结果列表
        test_cases: 测试用例列表
        
    Returns:
        float: 工具匹配正确率
    """
    correct = 0
    partial_scores = []
    
    for pred, case in zip(predictions, test_cases):
        expected_tools = case.get("expected_tools", [])
        used_tools = [tool_call["name"] for tool_call in pred.get("tool_calls", [])]
        
        # 完全匹配检查
        if set(expected_tools) == set(used_tools) and len(expected_tools) == len(used_tools):
            correct += 1
            partial_scores.append(1.0)
        else:
            # 部分匹配评分
            correct_tools = set(expected_tools).intersection(set(used_tools))
            partial_score = len(correct_tools) / max(len(expected_tools), len(used_tools))
            partial_scores.append(partial_score)
    
    return {
        "exact_match": correct / len(predictions) if predictions else 0,
        "partial_match": sum(partial_scores) / len(partial_scores) if partial_scores else 0
    }

def calculate_parameter_extraction(predictions, test_cases):
    """
    评估参数提取能力
    
    Args:
        predictions: 模型预测结果列表
        test_cases: 测试用例列表
        
    Returns:
        dict: 包含精确匹配率和部分匹配率的字典
    """
    exact_matches = 0
    partial_scores = []
    
    for pred, case in zip(predictions, test_cases):
        pred_tools = pred.get("tool_calls", [])
        expected_tools = case.get("expected_tool_calls", [])
        
        tools_match_scores = []
        
        # 匹配每个工具的参数
        for exp_tool in expected_tools:
            for pred_tool in pred_tools:
                if exp_tool["name"] == pred_tool["name"]:
                    exp_params = exp_tool.get("parameters", {})
                    pred_params = pred_tool.get("parameters", {})
                    
                    # 检查所有参数
                    correct_params = 0
                    total_params = len(exp_params)
                    
                    for param_name, exp_value in exp_params.items():
                        if param_name in pred_params and pred_params[param_name] == exp_value:
                            correct_params += 1
                    
                    param_score = correct_params / total_params if total_params > 0 else 0
                    tools_match_scores.append(param_score)
                    
                    if param_score == 1.0:
                        exact_matches += 1
                    
                    break
        
        if tools_match_scores:
            partial_scores.append(sum(tools_match_scores) / len(tools_match_scores))
    
    return {
        "exact_match": exact_matches / len(predictions) if predictions else 0,
        "partial_match": sum(partial_scores) / len(partial_scores) if partial_scores else 0
    }

def calculate_parameter_extraction_weighted(predictions, test_cases):
    """
    评估参数提取能力（区分必选和可选参数）
    
    Args:
        predictions: 模型预测结果列表
        test_cases: 测试用例列表（需包含参数重要性标记）
        
    Returns:
        dict: 包含精确匹配率和加权匹配率的字典
    """
    exact_matches = 0
    weighted_scores = []
    
    for pred, case in zip(predictions, test_cases):
        pred_tools = pred.get("tool_calls", [])
        expected_tools = case.get("expected_tool_calls", [])
        
        tools_match_scores = []
        
        # 匹配每个工具的参数
        for exp_tool in expected_tools:
            for pred_tool in pred_tools:
                if exp_tool["name"] == pred_tool["name"]:
                    exp_params = exp_tool.get("parameters", {})
                    pred_params = pred_tool.get("parameters", {})
                    
                    # 参数重要性定义（如果未在测试用例中定义，则使用默认值）
                    param_importance = exp_tool.get("param_importance", {})
                    
                    # 为未指定重要性的参数设置默认值
                    for param in exp_params:
                        if param not in param_importance:
                            param_importance[param] = 1.0  # 默认权重
                    
                    # 计算重要性总和用于归一化
                    total_importance = sum(param_importance.values())
                    
                    # 计算加权分数
                    weighted_score = 0
                    all_params_correct = True
                    
                    for param_name, exp_value in exp_params.items():
                        param_weight = param_importance.get(param_name, 1.0) / total_importance
                        
                        # 检查参数是否存在且正确
                        if param_name in pred_params:
                            # 字符串和基本类型的精确匹配
                            if isinstance(exp_value, (str, int, float, bool)) and pred_params[param_name] == exp_value:
                                weighted_score += param_weight
                            # 列表类型的包含关系评估
                            elif isinstance(exp_value, list):
                                pred_value = pred_params[param_name]
                                if isinstance(pred_value, list):
                                    overlap = set(exp_value).intersection(set(pred_value))
                                    match_ratio = len(overlap) / max(len(exp_value), len(pred_value))
                                    weighted_score += param_weight * match_ratio
                                else:
                                    all_params_correct = False
                            else:
                                all_params_correct = False
                        else:
                            all_params_correct = False
                    
                    tools_match_scores.append(weighted_score)
                    
                    # 记录精确匹配
                    if all_params_correct and len(exp_params) == len(pred_params):
                        exact_matches += 1
                    
                    break
        
        if tools_match_scores:
            weighted_scores.append(sum(tools_match_scores) / len(tools_match_scores))
    
    return {
        "exact_match": exact_matches / len(predictions) if predictions else 0,
        "weighted_match": sum(weighted_scores) / len(weighted_scores) if weighted_scores else 0
    }

def calculate_improved_overall_score(metrics):
    """
    计算改进后的综合评分
    
    Args:
        metrics: 包含各项指标的字典
        
    Returns:
        float: 综合评分
    """
    weights = {
        "intent": 0.4,  # 意图识别权重
        "tool": 0.3,    # 工具匹配权重
        "parameter": 0.3  # 参数提取权重
    }
    
    # 各项指标计算
    intent_score = metrics.get("intent", {}).get("accuracy", 0) * 0.7 + metrics.get("intent", {}).get("f1", 0) * 0.3
    tool_score = metrics.get("tool", {}).get("exact_match", 0) * 0.7 + metrics.get("tool", {}).get("partial_match", 0) * 0.3
    param_score = metrics.get("parameter", {}).get("exact_match", 0) * 0.7 + metrics.get("parameter", {}).get("partial_match", 0) * 0.3
    
    # 综合评分计算
    final_score = (
        intent_score * weights["intent"] +
        tool_score * weights["tool"] +
        param_score * weights["parameter"]
    )
    
    return final_score

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

def calculate_overall_score(metrics):
    """
    计算总体得分
    
    Args:
        metrics: 包含各项指标的字典
        
    Returns:
        float: 总体得分
    """
    # 加权平均，可根据需要调整权重
    weights = {
        "tool_accuracy": 0.5,
        "relevance": 0.3,
        "correctness": 0.2
    }
    
    score = 0.0
    for metric, weight in weights.items():
        if metric in metrics:
            score += metrics[metric] * weight
    
    return score

def calculate_response_correctness(responses, test_cases):
    """
    计算回复正确性分数
    
    Args:
        responses: 模型回复列表
        test_cases: 测试用例列表
        
    Returns:
        float: 正确性分数
    """
    if not responses or not test_cases:
        return 0.0
    
    correctness_scores = []
    
    for response, test_case in zip(responses, test_cases):
        # 获取回复和测试用例信息
        tool_calls = response.get("tool_calls", [])
        expected_tools = test_case.get("expected_tools", [])
        
        # 检查1：工具使用正确性
        used_tools = [tc["name"] for tc in tool_calls]
        tool_match_score = len(set(used_tools).intersection(set(expected_tools))) / max(len(expected_tools), 1)
        
        # 检查2：参数提取正确性
        # 这里需要具体测试用例中的参数期望值，简化版检查参数是否存在
        param_score = 0.0
        if "expected_parameters" in test_case and tool_calls:
            expected_params = test_case.get("expected_parameters", {})
            actual_params = {}
            for tc in tool_calls:
                actual_params.update(tc.get("arguments", {}))
            
            # 检查参数匹配度
            if expected_params:
                matched_params = sum(1 for k in expected_params if k in actual_params)
                param_score = matched_params / len(expected_params)
        else:
            # 无参数期望时，给予满分
            param_score = 1.0
            
        # 检查3：回复文本质量
        # 这里简化为文本长度检查和关键信息提取检查
        text_quality = min(1.0, len(response.get("text", "")) / 100)
        
        # 综合评分 (可调整权重)
        correctness_score = 0.5 * tool_match_score + 0.3 * param_score + 0.2 * text_quality
        correctness_scores.append(correctness_score)
    
    # 返回平均正确性分数
    return sum(correctness_scores) / len(correctness_scores)