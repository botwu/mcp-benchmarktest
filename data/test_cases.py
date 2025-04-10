# 测试用例定义

# 定义基准测试用例
TEST_CASES = [
    {
        "id": "web_search_1",
        "description": "基本网络搜索能力测试",
        "query": "查找关于量子计算最新进展的信息",
        "expected_tools": ["search_web"],
        "expected_parameters": {
            "query": "量子计算最新进展"
        },
        "expected_response": "应该返回量子计算领域的最新研究和突破",
        "difficulty": "easy"
    },
    {
        "id": "code_execution_1",
        "description": "基本代码执行能力测试",
        "query": "编写并运行一个计算斐波那契数列的Python函数",
        "expected_tools": ["run_code"],
        "expected_parameters": {
            "language": "python",
            "code": "fibonacci"
        },
        "expected_response": "应该展示一个计算斐波那契数列的Python函数及其执行结果",
        "difficulty": "medium"
    },
    {
        "id": "multi_tool_1",
        "description": "多工具协作测试",
        "query": "查找最新的Python库用于数据分析，并展示一个简单的使用示例",
        "expected_tools": ["search_web", "run_code"],
        "expected_parameters": {
            "query": "Python数据分析库",
            "language": "python",
            "code": "数据分析"
        },
        "expected_response": "应该先搜索最新的Python数据分析库，然后提供一个使用该库的代码示例",
        "difficulty": "hard"
    },
    {
        "id": "database_query_1",
        "description": "数据库查询能力测试",
        "query": "查询用户表中最近注册的10个用户信息",
        "expected_tools": ["access_database"],
        "expected_parameters": {
            "query": "SELECT * FROM users ORDER BY registration_date DESC LIMIT 10"
        },
        "expected_response": "应该提供一个SQL查询语句来获取最近注册的用户信息",
        "difficulty": "medium"
    },
    {
        "id": "complex_search_1",
        "description": "复杂搜索能力测试",
        "query": "查找关于气候变化对北极熊影响的最新研究",
        "expected_tools": ["search_web"],
        "expected_parameters": {
            "query": "气候变化 北极熊 最新研究"
        },
        "expected_response": "应该提供关于气候变化如何影响北极熊生存的最新科学研究",
        "difficulty": "medium"
    }
]

def get_test_case(case_id):
    """
    获取指定ID的测试用例
    
    Args:
        case_id: 测试用例ID
        
    Returns:
        dict: 测试用例数据
    """
    for case in TEST_CASES:
        if case["id"] == case_id:
            return case
    return None

def get_test_cases_by_difficulty(difficulty):
    """
    获取指定难度的所有测试用例
    
    Args:
        difficulty: 难度级别 (easy, medium, hard)
        
    Returns:
        list: 测试用例列表
    """
    return [case for case in TEST_CASES if case["difficulty"] == difficulty]