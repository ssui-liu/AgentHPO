from collections import deque
from typing import Dict, List, Optional, Any


class Controller(object):

    def __init__(self, task_creation_chain, task_update_chain,
                 agent_executor, log_path, max_iterations: int = None):
        self.task_list = deque()
        self.task_id_counter = 1
        self.max_iterations = max_iterations
        self.task_creation_chain = task_creation_chain
        self.task_update_chain = task_update_chain
        self.agent_executor = agent_executor
        self.tool_names = agent_executor.agent.allowed_tools
        self.log_path = log_path
        self.exp_id = 0
        self.completed_tasks = []

    def update_log(self, task_info):
        with open(self.log_path, 'a', encoding="utf-8") as f:
            f.write(task_info)
        f.close()

    def read_log(self):
        with open(self.log_path, 'r', encoding="utf-8") as f:
            return f.read()

    def add_task(self, task: Dict):
        self.task_list.append(task)

    def execute_task(self, task: str) -> str:
        """Execute a task."""
        return self.agent_executor.run(input=task)

    def initialize_tasks(self, objective, task_info, hyperparameter_info) -> List[Dict]:
        """Initialize tasks."""

        response = self.task_creation_chain.run(objective=objective, task_info=task_info,
                                                hyperparameter_info=hyperparameter_info)

        return [{"task_name": response}]

    def update_tasks(self, objective, task_info, hyperparameter_info) -> List[Dict]:
        """Get the next task."""
        complete_tasks = "\n".join(self.completed_tasks)
        response = self.task_update_chain.run(input=objective, task_info=task_info,
                                              hyperparameter_info=hyperparameter_info,
                                              complete_tasks=complete_tasks
                                              )

        return [{"task_name": response}]

    def print_task_list(self):
        print("\033[95m\033[1m" + "\n*****TASK LIST*****\n" + "\033[0m\033[0m")
        for t in self.task_list:
            print(str(t["task_id"]) + ": " + t["task_name"])

    def print_next_task(self, task: Dict):
        print("\033[92m\033[1m" + "\n*****NEXT TASK*****\n" + "\033[0m\033[0m")
        print(str(task["task_id"]) + ": " + task["task_name"])

    def print_task_result(self, result: str):
        print("\033[93m\033[1m" + "\n*****TASK RESULT*****\n" + "\033[0m\033[0m")
        print(result)

    def run(self, inputs: Dict[str, Any]) -> Dict[str, Any]:
        """Run the agent."""
        objective = inputs['objective']
        task_info = inputs['task_info']
        hyperparameter_info = inputs['hyperparameter_info']

        initial_tasks = self.initialize_tasks(objective=objective, task_info=task_info,
                                              hyperparameter_info=hyperparameter_info,
                                              )

        for new_task in initial_tasks:
            self.task_id_counter += 1
            new_task.update({"task_id": self.task_id_counter})
            self.add_task(new_task)
        num_iters = 0

        while True:
            if self.task_list:
                self.print_task_list()

                # Step 1: Pull the first task
                task = self.task_list.popleft()
                self.print_next_task(task)
                self.exp_id += 1
                prefix = "*" * 10 + "Experiment {}".format(self.exp_id) + "*" * 10 + "\n"
                task_str = prefix + task["task_name"] + "\n"

                self.update_log(task_str)
                self.completed_tasks.append(task["task_name"])

                # Step 2: Execute the task
                result = self.execute_task(task=task['task_name'])
                self.print_task_result(result)
                self.update_log(result + "\n")

                new_tasks = self.update_tasks(
                    objective=objective, task_info=task_info,
                    hyperparameter_info=hyperparameter_info
                )
                for new_task in new_tasks:
                    self.task_id_counter += 1
                    new_task.update({"task_id": self.task_id_counter})

                self.task_list = deque(new_tasks)

            num_iters += 1
            if self.max_iterations is not None and num_iters == self.max_iterations:
                print("\033[91m\033[1m" + "\n*****TASK ENDING*****\n" + "\033[0m\033[0m")
                break
        return {}
