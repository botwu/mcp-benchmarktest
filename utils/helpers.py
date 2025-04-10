# 辅助函数
import json
import os
import time

def save_json(data, file_path, indent=2):
    """
    将数据保存为JSON文件
    
    Args:
        data: 要保存的数据
        file_path: 文件路径
        indent: JSON缩进空格数
    """
    directory = os.path.dirname(file_path)
    if directory and not os.path.exists(directory):
        os.makedirs(directory)
        
    with open(file_path, 'w', encoding='utf-8') as f:
        json.dump(data, f, ensure_ascii=False, indent=indent)
        
def load_json(file_path):
    """
    从JSON文件加载数据
    
    Args:
        file_path: 文件路径
        
    Returns:
        加载的数据
    """
    with open(file_path, 'r', encoding='utf-8') as f:
        return json.load(f)
        
def get_timestamp():
    """
    获取当前时间戳字符串
    
    Returns:
        str: 时间戳字符串
    """
    return time.strftime("%Y%m%d_%H%M%S")
    
def ensure_dir(directory):
    """
    确保目录存在，如不存在则创建
    
    Args:
        directory: 目录路径
    """
    if not os.path.exists(directory):
        os.makedirs(directory)