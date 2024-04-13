from langchain.agents import ZeroShotAgent, Tool, AgentExecutor
from langchain.chat_models import ChatOpenAI
from langchain.tools import StructuredTool
from langchain.agents import initialize_agent, AgentType
import os
from agents.tools import CreatorFileTool, PythonFileExecutorTool
from agents.prompts.task_creator_prompt import PREFIX, FORMAT_INSTRUCTIONS, SUFFIX


def init_creator(model, log_path, model_file_path):
    file_tools = CreatorFileTool(log_path=log_path, model_file_path=model_file_path)

    tools = [
        Tool(
            name="LoadHistoricalTrainingLogs",
            func=file_tools.load_log,
            description="This tool is designed for easily loading and reviewing model training logs. It automatically "
                        "accesses records of loss and accuracy metrics from different hyper-parameter settings."
                        "The file path is preset, so no additional input is required.",
        ),
    ]

    llm = ChatOpenAI(temperature=0, model=model)

    agent_executor = initialize_agent(
        tools=tools,
        llm=llm,
        agent=AgentType.ZERO_SHOT_REACT_DESCRIPTION,  # .STRUCTURED_CHAT_ZERO_SHOT_REACT_DESCRIPTION,
        verbose=True,
        handle_parsing_errors=True,
        # return_intermediate_steps=True,
        agent_kwargs={
            "prefix": PREFIX,
            "suffix": SUFFIX,
            "format_instructions": FORMAT_INSTRUCTIONS,
        }
    )
    return agent_executor

