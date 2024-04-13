import os
from langchain.chat_models import ChatOpenAI
from agents.llm_chain import TaskCreationChain
from agents.task_executor import init_executor
from agents.task_creator import init_creator
from agents.controller import Controller
from typing import Dict, List, Optional, Any

from langchain.callbacks import get_openai_callback

os.environ["OPENAI_API_KEY"] = "sk-"

verbose = True
# If None, will keep ongoing forever
max_iterations: Optional[int] = 10
# model = "gpt-3.5-turbo-1104"
model = "gpt-4-1104-preview"
llm = ChatOpenAI(temperature=1, model=model)

load_config_path = "environments/gcn/node_classification/configs_changed.json"
write_config_path = "environments/gcn/node_classification/configs_changed.json"
main_file_path = "environments/gcn/node_classification/main2.py"
log_path_refresh = "environments/gcn/node_classification/refresh.log"

log_path_append = "environments/gcn/node_classification/append.log"
model_file_path = "environments/gcn/node_classification/model.py"

agent_executor = init_executor(model=model, load_config_path=load_config_path, write_config_path=write_config_path,
                               log_path=log_path_refresh, main_file_path=main_file_path)
agent_creator = init_creator(model=model, log_path=log_path_append, model_file_path=model_file_path)

task_creation_chain = TaskCreationChain.from_llm(llm, verbose=verbose)

controller = Controller(task_creation_chain=task_creation_chain, task_update_chain=agent_creator,
                        agent_executor=agent_executor, log_path=log_path_append, max_iterations=max_iterations)

objective = "Tuning the hyperparameters of a neural network to maximize the accuracy."
task_info = """
Dataset: Cora dataset, which consists of 2708 scientific publications classified into one of seven classes. 
Task type: Node Classification
Model: GCN
Metrics: Accuracy, Loss, Memory Usage
"""

hyperparameter_info = """
  num_layers: [1, 5]
  learning_rate: [1e-6, 1e-1]
  optimizer: [adam, sgd]
  num_epochs: [1, 200]
  hidden_size: [8, 16, 32, 64]
  activation: [relu, elu, silu]
  weight_decay: [1e-6, 1e-1]
  dropout: [0, 0.5]
"""
input_dict = {'objective': objective, 'task_info': task_info, 'hyperparameter_info': hyperparameter_info}

with get_openai_callback() as cb:
    try:
        result = controller.run(input_dict)
        print(result)
    except Exception as e:
        print(e)
        print("Error raised at experiment: ", controller.exp_id)
    print(f"Total Tokens: {cb.total_tokens}")
    print(f"Prompt Tokens: {cb.prompt_tokens}")
    print(f"Completion Tokens: {cb.completion_tokens}")
    print(f"Total Cost (USD): ${cb.total_cost}")

