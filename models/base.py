# 模型基类
import abc

class BaseModel(abc.ABC):
    """模型接口基类"""
    
    def __init__(self, model_config):
        """
        初始化模型
        
        Args:
            model_config: 模型配置
        """
        self.model_config = model_config
        
    @abc.abstractmethod
    def generate(self, prompt, max_tokens=None, temperature=None):
        """
        生成文本回复
        
        Args:
            prompt: 输入提示
            max_tokens: 最大生成token数
            temperature: 采样温度
            
        Returns:
            str: 生成的文本
        """
        pass
        
    @abc.abstractmethod
    def generate_with_tools(self, prompt, tools, max_tokens=None, temperature=None):
        """
        使用工具生成回复
        
        Args:
            prompt: 输入提示
            tools: 可用工具列表
            max_tokens: 最大生成token数
            temperature: 采样温度
            
        Returns:
            dict: 包含回复和工具调用的结果
        """
        pass