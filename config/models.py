# 模型配置

# 定义支持的模型配置
MODEL_CONFIGS = {
    "qwen": {
        "model_id": "atom",
        "api_base": "http://172.17.120.202:31380/remote/llmops/MWH-REMOTE-SERVICE-cqudbmnua89gepmu8nrg/openai/v1/chat/completions",
        "token": "Bearer apikey-1pN-DInciyWezCVOV0aJZrzIVcte4zRPmGCs1IwLOLLk",
        "max_tokens": 2048,
        "temperature": 0,
    },
    "deepseek": {
        "model_id": "atom",
        "api_base": "http://llmops.wuya-ai.com:31380/seldon/llmops-assets/service-5a1e01da-a3c4-43f9-bfb0-08e43e238d91/8000/v1/chat/completions",
        "token": "Bearer apikey-YpUN8i4Q27LTCUQ_TKc3q7hsUvJEq2hnnXMGPrvf8ez5",
        "max_tokens": 2048,
        "temperature": 0,
    }
}

def get_model_config(model_name):
    """
    获取指定模型的配置
    
    Args:
        model_name: 模型名称
        
    Returns:
        dict: 模型配置
    """
    return MODEL_CONFIGS.get(model_name, {})