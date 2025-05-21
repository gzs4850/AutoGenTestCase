# testcase_generator.py
import re
from configparser import ConfigParser
from pathlib import Path
from typing import List, Any
from autogen_ext.models.openai import OpenAIChatCompletionClient
from autogen_agentchat.conditions import TextMentionTermination
from autogen_agentchat.teams import RoundRobinGroupChat
from autogen_agentchat.agents import AssistantAgent
from llms import *

# 配置路径处理
main_path = Path(__file__).parent
config_path = main_path / "config.ini"

# 初始化配置
conf = ConfigParser()
conf.read(config_path)


# 将markdown格式的数据转换成list
def markdown2list(raw_data:list):
    test_cases = []

    for item in raw_data:
        # 分割单元格并去除空白
        cells = [cell.strip() for cell in item.strip('|').split('|')]
        cells = list(filter(None, cells))  # 移除空字符串

        # 跳过分隔行（所有单元格由短横线组成）
        if all(cell.startswith('---') for cell in cells if cell):
            continue

        # 跳过表头行（首列为"用例ID"）
        if cells and cells[0] == "用例ID":
            continue

        # 仅保留数据行（首列格式为"DM_xxx"）
        if cells and cells[0].startswith("DM_"):
            test_cases.append(cells)
    # 输出转换后的列表
    # print(f"test_cases:{test_cases}")

    return test_cases


import pandas as pd
import re
from io import StringIO


# 将markdown格式的数据转换成pandas的dataframe
def markdown2dataFrame(raw_data):
    # 过滤分隔线行（包含3个以上短横线）
    filtered = [line for line in raw_data if not re.search(r'[-]{3,}', line)]

    # 转换为类文件对象
    table_str = '\n'.join(filtered)

    # 读取数据
    df = pd.read_csv(
        StringIO(table_str),
        sep=r'\s*\|\s*',  # 修正后的正则表达式
        engine='python',
        header=0,
        dtype='string',
        quotechar='"',
        na_filter=False
    )

    # 清洗数据
    df = df.dropna(axis=1, how='all')  # 删除空列
    df.columns = df.columns.str.strip()  # 清理列名
    df = df.apply(lambda x: x.str.strip())  # 清理内容

    # 处理重复用例ID（保留最后一条）
    df = df.drop_duplicates(subset=['用例ID'], keep='last')  # [1,3](@ref)

    # 增强版数字提取与转换
    df['排序键'] = (
        df['用例ID'].str.extract(r'(\d+)', expand=False).fillna('0').pipe(pd.to_numeric, errors='coerce').fillna(
            0).astype(int))

    # 按ID数字排序并删除辅助列
    df = df.sort_values('排序键').drop('排序键', axis=1)

    # 处理HTML换行符
    df[['测试步骤', '预期结果']] = df[['测试步骤', '预期结果']].apply(
        lambda x: x.str.replace('<br>', '\n')
    )

    return df.reset_index(drop=True)


#     # 生成最终 DataFrame
# testcase_df = process_testcases(data)
#
# # 验证结果
# print(testcase_df.head(3))
# print("\n数据类型:")
# print(testcase_df.dtypes)


def format_testcases(raw_output: str) -> list[list[Any]]:
    """格式化测试用例输出"""
    # print(f"格式化前raw_output：{raw_output}")
    markdown_cases = re.findall(r'(\|.+\|)', raw_output, re.IGNORECASE)
    print(f"markdown_cases：{markdown_cases}")

    test_cases = markdown2dataFrame(markdown_cases)
    print(f"test_cases:{test_cases}")

    # test_cases = list(dict.fromkeys(markdown_cases))
    # print(f"test_cases：{test_cases}")

    # test_cases = []
    #
    # for item in markdown_cases:
    #     # 分割单元格并去除空白
    #     cells = [cell.strip() for cell in item.strip('|').split('|')]
    #     cells = list(filter(None, cells))  # 移除空字符串
    #
    #     # 跳过分隔行（所有单元格由短横线组成）
    #     if all(cell.startswith('---') for cell in cells if cell):
    #         continue
    #
    #     # 跳过表头行（首列为"用例ID"）
    #     if cells and cells[0] == "用例ID":
    #         continue
    #
    #     # 仅保留数据行（首列格式为"DM_xxx"）
    #     if cells and cells[0].startswith("DM_"):
    #         test_cases.append(cells)
    # # 输出转换后的列表
    # # print(f"test_cases:{test_cases}")
    # test_cases = markdown2list(markdown_cases)

    return test_cases


async def generate_testcases(user_input: str) -> List[str]:
    """生成测试用例核心函数"""

    # 读取系统消息模板
    def read_system_message(filename):
        with open(main_path / filename, "r", encoding="utf-8") as f:
            return f.read()

    # 初始化模型客户端
    deepseek_client = OpenAIChatCompletionClient(
        model=conf['deepseek']['model'],
        base_url=conf['deepseek']['base_url'],
        api_key=conf['deepseek']['api_key'],
        model_info=model_deepseek_info,
    )

    qwen_client = OpenAIChatCompletionClient(
        model=conf['qwen']['model'],
        base_url=conf['qwen']['base_url'],
        api_key=conf['qwen']['api_key'],
        model_info=model_qwen_info,
    )

    # 创建代理
    system_writer = read_system_message("TESTCASE_WRITER_SYSTEM_MESSAGE.txt")
    system_reader = read_system_message("TESTCASE_READER_SYSTEM_MESSAGE.txt")

    testcase_writer = AssistantAgent(
        name="testcase_writer",
        model_client=deepseek_client,
        system_message=system_writer,
    )

    testcase_reader = AssistantAgent(
        name="critic",
        model_client=qwen_client,
        system_message=system_reader,
        model_client_stream=True,
    )

    # 配置团队协作
    text_termination = TextMentionTermination("APPROVE")
    team = RoundRobinGroupChat(
        participants=[testcase_writer, testcase_reader],
        termination_condition=text_termination,
        max_turns=10
    )

    full_response = ""
    is_continue = True

    async for chunk in team.run_stream(task=user_input):
        content = ""
        if chunk:
            # 处理不同类型的chunk
            if hasattr(chunk, 'content') and hasattr(chunk, 'type'):
                if chunk.type != 'ModelClientStreamingChunkEvent':
                    content = chunk.content
            elif isinstance(chunk, str):
                content = chunk
            else:
                content = str(chunk)
            # 将新内容添加到完整响应中
            if is_continue and content != "" and not content.startswith("TaskResult"):
                full_response += '\n\n' + content

            # APPROVE结束退出
            if content.find("APPROVE") > 0:
                is_continue = False

    return format_testcases(full_response)


# fastapi_app.py
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from testcase_generator import generate_testcases
import asyncio

app = FastAPI()

class TestCaseRequest(BaseModel):
    requirement: str

@app.post("/generate-testcases")
async def generate_testcases_api(request: TestCaseRequest):
    try:
        cases = await generate_testcases(request.requirement)
        return {"testcases": cases}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)