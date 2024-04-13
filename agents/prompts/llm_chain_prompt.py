template = """
As an AI expert specializing in task creation for machine learning, your ultimate goal is to 
accomplish {objective}.

Key experimental details are as follows:

{task_info}

Additionally, here are the hyperparameters available for optimization, along with their possible values or range:

{hyperparameter_info}

With this information, devise an initial set of hyper-parameter settings that you believe will enable the model to 
reach the stated objective. Keep in mind that the task executor can only modify these hyper-parameters and run the 
training script. Therefore, suggest a set of hyper-parameter settings that, in your opinion, are most likely to be 
effective. Present your recommendation clearly and succinctly.
"""