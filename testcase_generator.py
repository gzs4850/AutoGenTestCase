# testcase_generator.py
import re
from configparser import ConfigParser
from pathlib import Path
import os
from typing import List

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


def format_testcases(raw_output: str) -> List[str]:
    """格式化测试用例输出"""
    print(f"格式化前raw_output：{raw_output}")
    filter_cases = re.findall(r'(\|.+\|)', raw_output, re.IGNORECASE)
    print(f"filter_cases：{filter_cases}")
    deduplicate_case = list(dict.fromkeys(filter_cases))
    print(f"去重后deduplicate_case：{deduplicate_case}")
    cases = [item for item in deduplicate_case if '**' not in item]
    print(f"去星后cases：{cases}")

    structured_cases = []
    for line in cases[2:-1]:  # 跳过表头和分隔线
        columns = [col.strip() for col in line.split("|") if col]
        case_dict = {
            "用例ID": columns[0],
            "测试目标": columns[1],
            "前置条件": columns[2],
            "测试步骤": columns[3],
            "预期结果": columns[4],
            "优先级": columns[5],
            "测试类型": columns[6],
            "关联需求": columns[7]
        }
        structured_cases.append(case_dict)

    return structured_cases


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

    # 运行生成流程
    # full_response = ""
    # async for chunk in team.run_stream(task=user_input):
    #     content = ""
    #     if chunk:
    #         if hasattr(chunk, 'content') and hasattr(chunk, 'type'):
    #             if chunk.type != 'ModelClientStreamingChunkEvent':
    #                 content = chunk.content
    #         elif isinstance(chunk, str):
    #             content = chunk
    #         else:
    #             content = str(chunk)
    #
    #         if content.find("APPROVE") > 0:
    #             break
    #         full_response += '\n\n' + content
    #
    # return format_testcases(full_response)

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