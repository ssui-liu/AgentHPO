import ast
import json
import os
import subprocess
from typing import Union, Dict

import yaml


def load_data_from_file(file_path: str) -> Union[Dict, str]:
    """
    Load data from a file.

    Parameters:
    file_path (str): The file path where data should be loaded from.

    Returns:
    Union[Dict, str]: The data loaded from the file.
    """

    _, extension = os.path.splitext(file_path)
    file_format = extension.lower()[1:]  # remove the leading dot

    try:
        with open(file_path, 'r', encoding="utf-8") as f:
            if file_format == 'json':
                return json.load(f)
            elif file_format in ['yaml', 'yml']:
                return yaml.safe_load(f)
            else:
                return f.read()  # return as text for unknown file formats
    except Exception as e:
        raise Exception(f"Error loading file: {str(e)}")


def write_data_to_file(file_path: str, data: Union[Dict, str]):
    """
    Writes data to a file in either JSON or YAML format.

    Parameters:
    path (str): The file path where data should be written.
    data (dict): The data to write to the file.

    Returns:
    None
    """

    _, extension = os.path.splitext(file_path)
    file_format = extension.lower()[1:]  # remove the leading dot
    if type(data) is str:
        data = ast.literal_eval(data)
    with open(file_path, 'w') as file:
        if file_format == 'json':
            json.dump(data, file, indent=4)
        elif file_format == 'yaml' or file_format == 'yml':
            yaml.safe_dump(data, file, default_flow_style=False)
        else:
            raise ValueError(f"Unsupported file format: {file_format}")


class FileTool(object):
    def __init__(self, load_config_path, write_config_path, log_path):
        self.load_config_path = load_config_path
        self.write_config_path = write_config_path
        self.log_path = log_path

    def load_config(self, path, *args, **kwargs):
        documents = load_data_from_file(self.load_config_path)
        return documents

    def load_log(self, path, *args, **kwargs):
        documents = load_data_from_file(self.log_path)
        return documents

    def write_log(self, documents):
        write_data_to_file(self.log_path, documents)

    def write_config(self, documents):
        write_data_to_file(self.write_config_path, documents)


class CreatorFileTool(object):
    def __init__(self, log_path, model_file_path):
        self.model_file_path = model_file_path
        self.log_path = log_path

    def load_model(self, path, *args, **kwargs):
        documents = load_data_from_file(self.model_file_path)
        return documents

    def load_log(self, path, *args, **kwargs):
        documents = load_data_from_file(self.log_path)
        return documents


class PythonFileExecutorTool(object):
    def __init__(self, file_path):
        self.file_path = file_path

    def execute(self, path):
        script_directory = os.path.dirname(os.path.abspath(self.file_path))
        process = subprocess.Popen(
            ["python", self.file_path],
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            text=True,
            cwd=script_directory  # Set the working directory
        )
        stdout, stderr = process.communicate()
        print(stderr)
        # return stderr
