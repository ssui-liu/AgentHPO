from langchain import LLMChain, OpenAI, PromptTemplate
from langchain.llms import BaseLLM
from agents.prompts.llm_chain_prompt import template


class TaskCreationChain(LLMChain):
    """Chain to generates tasks."""

    @classmethod
    def from_llm(cls, llm: BaseLLM, verbose: bool = True) -> LLMChain:
        """Get the response parser."""
        task_creation_template = (
            template
        )
        prompt = PromptTemplate(
            template=task_creation_template,
            input_variables=["objective", "task_info", "hyperparameter_info", "tool_names"],
        )
        return cls(prompt=prompt, llm=llm, verbose=verbose)



