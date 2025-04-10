# 数据集加载器
import os
import json

class DatasetLoader:
    """数据集加载器类"""
    
    def __init__(self, data_dir="datasets"):
        """
        初始化数据集加载器
        
        Args:
            data_dir: 数据集所在目录
        """
        self.data_dir = data_dir
        
    def load_dataset(self, dataset_name):
        """
        加载指定的数据集
        
        Args:
            dataset_name: 数据集名称
            
        Returns:
            list: 数据集内容
        """
        file_path = os.path.join(self.data_dir, f"{dataset_name}.json")
        try:
            with open(file_path, 'r', encoding='utf-8') as f:
                return json.load(f)
        except FileNotFoundError:
            print(f"数据集 {dataset_name} 不存在")
            return []
            
    def get_available_datasets(self):
        """
        获取所有可用的数据集
        
        Returns:
            list: 可用数据集名称列表
        """
        if not os.path.exists(self.data_dir):
            return []
            
        return [f[:-5] for f in os.listdir(self.data_dir) 
                if f.endswith('.json')]