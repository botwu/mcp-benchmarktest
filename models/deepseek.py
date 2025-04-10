# Deepseek模型客户端
import requests
from .base import BaseModel
import json

class DeepseekModel(BaseModel):
    """Deepseek模型客户端实现"""
    
    def __init__(self, model_config):
        """
        初始化Deepseek模型
        
        Args:
            model_config: 模型配置
        """
        super().__init__(model_config)
        self.api_base = model_config.get("api_base")
        self.model_id = model_config.get("model_id", "atom")
        self.api_key = model_config.get("token")
        
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
        max_tokens = max_tokens or self.model_config.get("max_tokens", 2048)
        temperature = temperature or self.model_config.get("temperature", 0)
        
        # 实际实现会调用Deepseek API
        # 这里只是示例
        headers = {
            "Authorization": self.api_key,
            "Content-Type": "application/json"
        }
        
        data = {
            "model": self.model_id,
            "prompt": prompt,
            "max_tokens": max_tokens,
            "temperature": temperature
        }
        
        # response = requests.post(f"{self.api_base}", headers=headers, json=data)
        # return response.json()["choices"][0]["text"]
        
        # 模拟返回
        return f"这是Deepseek模型({self.model_id})对'{prompt}'的回复"
        
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
        max_tokens = max_tokens or self.model_config.get("max_tokens", 2048)
        temperature = temperature or self.model_config.get("temperature", 0)
        
        # 实际调用Deepseek API的工具调用功能
        headers = {
            "Authorization": self.api_key,
            "Content-Type": "application/json"
        }
        
        # 构建消息格式的请求
        messages = [{"role": "user", "content": prompt}]
        
        # 转换工具格式为OpenAI格式
        openai_tools = []
        for tool in tools:
            openai_tool = {
                "type": "function",
                "function": {
                    "name": tool["name"],
                    "description": tool["description"]
                }
            }
            
            # 处理参数
            if "parameters" in tool:
                params = {}
                for param_name, param_desc in tool["parameters"].items():
                    params[param_name] = {"type": "string", "description": param_desc}
                
                openai_tool["function"]["parameters"] = {
                    "type": "object",
                    "properties": params,
                    "required": list(params.keys())
                }
            
            openai_tools.append(openai_tool)
        
        data = {
            "model": self.model_id,
            "messages": messages,
            "max_tokens": max_tokens,
            "temperature": temperature,
            "tools": openai_tools
        }
        
        try:
            # 添加调试日志
            print(f"请求URL: {self.api_base}")
            print(f"请求头: {headers}")
            print(f"请求数据: {json.dumps(data, ensure_ascii=False, indent=2)}")
            
            # 使用模拟数据代替真实API调用（因为API可能不可用）
            # response = requests.post(f"{self.api_base}", headers=headers, json=data, timeout=30)
            # response.raise_for_status()
            # api_response = response.json()
            
            # 模拟返回数据，但加入一些特定于Deepseek的差异
            api_response = self.simulate_api_response(prompt, tools)
            
            # 处理API响应结果
            if "choices" in api_response and len(api_response["choices"]) > 0:
                choice = api_response["choices"][0]
                message = choice.get("message", {})
                
                # 解析工具调用结果
                tool_calls = []
                for tool_call in message.get("tool_calls", []):
                    if "function" in tool_call:
                        function = tool_call["function"]
                        tool_calls.append({
                            "name": function.get("name", ""),
                            "arguments": json.loads(function.get("arguments", "{}"))
                        })
                
                return {
                    "text": message.get("content", ""),
                    "tool_calls": tool_calls
                }
            else:
                # 如果响应格式不符合预期，返回错误信息
                return {
                    "text": f"API响应格式错误: {api_response}",
                    "tool_calls": []
                }
                
        except requests.exceptions.RequestException as e:
            # 处理API调用异常
            return {
                "text": f"API调用失败: {str(e)}",
                "tool_calls": []
            }
        except json.JSONDecodeError:
            # 处理JSON解析异常
            return {
                "text": "API响应不是有效的JSON格式",
                "tool_calls": []
            }
        except Exception as e:
            # 处理其他异常
            return {
                "text": f"发生错误: {str(e)}",
                "tool_calls": []
            }
            
    def simulate_api_response(self, prompt, tools):
        """
        模拟Deepseek模型的API响应数据
        
        Args:
            prompt: 输入提示
            tools: 可用工具列表
            
        Returns:
            dict: 模拟的API响应
        """
        # 根据查询内容选择合适的工具
        selected_tools = []
        
        # Deepseek更倾向于使用多个工具
        if "量子计算" in prompt or "气候变化" in prompt or "北极熊" in prompt:
            web_tool = next((t for t in tools if t["name"] == "search_web"), None)
            if web_tool:
                selected_tools.append(web_tool)
        
        if "代码" in prompt or "Python" in prompt or "函数" in prompt:
            code_tool = next((t for t in tools if t["name"] == "run_code"), None)
            if code_tool:
                selected_tools.append(code_tool)
        
        if "数据库" in prompt or "查询" in prompt or "SQL" in prompt:
            db_tool = next((t for t in tools if t["name"] == "access_database"), None)
            if db_tool:
                selected_tools.append(db_tool)
                
        # 在找不到合适工具时，默认使用search_web
        if not selected_tools:
            web_tool = next((t for t in tools if t["name"] == "search_web"), None)
            if web_tool:
                selected_tools.append(web_tool)
            
        # 构建工具调用
        tool_calls = []
        for i, tool in enumerate(selected_tools):
            arguments = {}
            # 根据工具类型生成参数
            if tool["name"] == "search_web":
                arguments = {"query": prompt.replace("查找", "").replace("关于", "").strip()}
            elif tool["name"] == "run_code":
                arguments = {"language": "python", "code": "def analyze_data():\n    print('Analyzing data with Deepseek')\n\nanalyze_data()"}
            elif tool["name"] == "access_database":
                arguments = {"query": "SELECT id, username, email, registration_date FROM users ORDER BY registration_date DESC LIMIT 10;"}
                
            tool_calls.append({
                "id": f"call_{i+1}",
                "type": "function",
                "function": {
                    "name": tool["name"],
                    "arguments": json.dumps(arguments, ensure_ascii=False)
                }
            })
        
        # 构建响应，包含更详细的内容
        return {
            "id": "deepseek-chatcmpl-456",
            "object": "chat.completion",
            "created": 1700000000,
            "model": "deepseek-ai",
            "choices": [
                {
                    "index": 0,
                    "message": {
                        "role": "assistant",
                        "content": f"我理解您需要{prompt}，我将使用合适的工具来帮助您解决这个问题。",
                        "tool_calls": tool_calls
                    },
                    "finish_reason": "tool_calls" if tool_calls else "stop"
                }
            ]
        }