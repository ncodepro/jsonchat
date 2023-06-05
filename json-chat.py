# Import necessary libraries and functions
import os
import yaml

from langchain.agents import create_json_agent, AgentExecutor
from langchain.agents.agent_toolkits import JsonToolkit
from langchain.chains import LLMChain
from langchain.llms.openai import OpenAI
from langchain.requests import TextRequestsWrapper
from langchain.tools.json.tool import JsonSpec

# Define a function to create an agent executor
def create_agent_executor(yaml_file, temperature=0, max_value_length=4000, verbose=True):
    """
    Create an agent executor using a provided YAML file.
    
    Args:
        yaml_file (str): The path to the YAML file.
        temperature (float): Temperature parameter for the OpenAI model.
        max_value_length (int): The maximum value length for the JSON specification.
        verbose (bool): Whether to print verbose output.

    Returns:
        AgentExecutor: The created agent executor.
    """
    # Load the YAML file
    with open(yaml_file) as f:
        data = yaml.load(f, Loader=yaml.FullLoader)

    # Create a JSON specification
    json_spec = JsonSpec(dict_=data, max_value_length=max_value_length)

    # Create a JSON toolkit
    json_toolkit = JsonToolkit(spec=json_spec)

    # Create and return the agent executor
    return create_json_agent(
        llm=OpenAI(temperature=temperature),
        toolkit=json_toolkit,
        verbose=verbose
    )

# Use the function to create an agent executor
json_agent_executor = create_agent_executor("openai_openapi.yml")
