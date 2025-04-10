# 日志工具
import logging
import os
from datetime import datetime

def setup_logger(name, log_dir="logs", level=logging.INFO):
    """
    配置日志记录器
    
    Args:
        name: 日志记录器名称
        log_dir: 日志存储目录
        level: 日志级别
        
    Returns:
        logging.Logger: 配置好的日志记录器
    """
    # 确保日志目录存在
    if not os.path.exists(log_dir):
        os.makedirs(log_dir)
        
    # 创建日志记录器
    logger = logging.getLogger(name)
    logger.setLevel(level)
    
    # 避免重复处理程序
    if logger.handlers:
        return logger
        
    # 创建文件处理程序
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    file_handler = logging.FileHandler(
        os.path.join(log_dir, f"{name}_{timestamp}.log")
    )
    
    # 创建控制台处理程序
    console_handler = logging.StreamHandler()
    
    # 创建格式化器
    formatter = logging.Formatter(
        '%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    )
    
    # 设置处理程序格式
    file_handler.setFormatter(formatter)
    console_handler.setFormatter(formatter)
    
    # 添加处理程序
    logger.addHandler(file_handler)
    logger.addHandler(console_handler)
    
    return logger