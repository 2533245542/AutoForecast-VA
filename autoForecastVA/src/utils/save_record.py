from os import path
import pandas as pd

import hashlib
from pathlib import Path

class SaveRecord():
    @staticmethod
    def convert_dictionary_into_parameter_to_value_dataframe(dictionary_to_convert):
        list_of_keys = []
        list_of_values = []
        for key, value in dictionary_to_convert.items():
            list_of_keys.append(key)
            list_of_values.append(value)
        converted_dataframe = pd.DataFrame({'variable': list_of_keys, 'value': list_of_values})
        return converted_dataframe

    @staticmethod
    def create_file_and_write_string(file_path, string_to_write):
        '''This will create the parent directory of the file if the parent directory does not exists'''
        parent_path = Path(file_path).parent
        parent_path.mkdir(parents=True, exist_ok=True)  # will not overwrite the parent folder if the parent folder already exists
        with open(file=file_path, mode='w') as f:
            f.write(string_to_write)

    @staticmethod
    def append_string_to_file(file_path, string_to_append):
        with open(file=file_path, mode='a') as f:
            f.write(string_to_append)


