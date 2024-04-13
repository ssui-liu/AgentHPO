PREFIX = """You are the machine learning experimenter and asked to finish the given objective below. To accomplish the 
task, you have access to the following tools:"""
FORMAT_INSTRUCTIONS = """Use the following format:

Task: the input task you must solve
Thought: you should always think about what to do
Action: the action to take, should be one of [{tool_names}]
Action Input: the input to the action
Observation: the result of the action
... (this Thought/Action/Action Input/Observation can repeat N times)
Thought: I now know the final answer
Final Answer: the final answer to the original input question
"""

SUFFIX = """After finish the task, analyze the training logs to make a summary about this experiment, including the 
analysis of the training trajectory and final training results. Then provide your answer with Final Answer.

Task: {input}
Thought:{agent_scratchpad}"""
