# MCP工具定义

# 定义可用的MCP工具
TOOLS = [
    {
        "name": "search_web",
        "description": "搜索网络获取信息",
        "parameters": {
            "query": "要搜索的内容"
        }
    },
    {
        "name": "run_code",
        "description": "执行代码片段",
        "parameters": {
            "language": "编程语言",
            "code": "要执行的代码"
        }
    },
    {
        "name": "access_database",
        "description": "查询数据库",
        "parameters": {
            "query": "SQL查询语句"
        }
    }
]

def get_tool_by_name(tool_name):
    """
    通过名称获取工具定义
    
    Args:
        tool_name: 工具名称
        
    Returns:
        dict: 工具定义
    """
    for tool in TOOLS:
        if tool["name"] == tool_name:
            return tool
    return None