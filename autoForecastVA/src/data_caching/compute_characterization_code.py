from os import path
import pandas as pd
import copy

import hashlib
from pathlib import Path
import inspect

from autoForecastVA.src.utils.save_record import SaveRecord


class ComputeCharacterizationCodeAndMd5Code():
    '''contains functions for caluclating characterization code for each scenario '''

    '''hash methods'''
    @staticmethod
    def hash_to_MD5(string):
        return hashlib.md5(string.encode('utf-8')).hexdigest()


    '''Basic methods'''
    @staticmethod
    def list_of_simple_data_get_characterization_code(list_of_simple_data, seperator='_'):
        return(seperator.join([str(data) for data in list_of_simple_data]))

    @staticmethod
    def dictionary_get_characterization_code(dictionary_to_process, seperator='_'):
        key_code = seperator.join([str(key) for key in dictionary_to_process.keys()])
        value_code = seperator.join([str(value) for value in dictionary_to_process.values()])
        return(key_code + value_code)

    @staticmethod
    def date_get_characterization_code(date):
        return(str(date.strftime("%Y%m%d%H%M%S")))

    @staticmethod
    def hyper_parameter_value_combination_get_characterization_code(hyper_parameter_value_combination, seperator='_'):  # simple data means single (or less than 5) element data
        return(seperator.join([key + ':' + str(hyper_parameter_value_combination[key]) for key in hyper_parameter_value_combination.keys()]))

    @staticmethod
    def dataframe_get_characterization_code(dataframe, seperator='_'):  # simple data means single (or less than 5) element data
        dataframe_shape = dataframe.shape
        dataframe_mean = dataframe.mean().tolist()
        dataframe_std = dataframe.std().tolist()

        return(seperator.join([str(data) for data in list(dataframe_shape) + dataframe_mean + dataframe_std]))

    '''Customized functions
    Need to hash to MD5 otherwise the file name is too long and has system error
    '''
    @staticmethod
    def prepare_rolling_forecast_data_get_MD5_code(not_normalized_combined_train_DataFrame, number_of_days_to_predict_ahead, number_of_test_days_in_a_day_level_DataFrame, number_of_rolling_forecast_days, windowing_option, input_length, lstm_is_many_to_many, do_normalization, seperator='_'):
        dataframe_code = ComputeCharacterizationCodeAndMd5Code.dataframe_get_characterization_code(not_normalized_combined_train_DataFrame)
        list_of_simple_data_code = ComputeCharacterizationCodeAndMd5Code.list_of_simple_data_get_characterization_code([number_of_days_to_predict_ahead, number_of_test_days_in_a_day_level_DataFrame, number_of_rolling_forecast_days, windowing_option, input_length, lstm_is_many_to_many, do_normalization])
        return ComputeCharacterizationCodeAndMd5Code.hash_to_MD5(seperator.join(['rolling_forecast_day', list_of_simple_data_code, dataframe_code]))

    @staticmethod
    def train_input_output_tensor_list_get_MD5_code(day, not_normalized_combined_train_DataFrame, number_of_days_to_predict_ahead, number_of_test_days_in_a_day_level_DataFrame, number_of_rolling_forecast_days, windowing_option, input_length, lstm_is_many_to_many, do_normalization, number_of_output_variables, seperator='_'):
        day_code = ComputeCharacterizationCodeAndMd5Code.date_get_characterization_code(day)
        dataframe_code = ComputeCharacterizationCodeAndMd5Code.dataframe_get_characterization_code(not_normalized_combined_train_DataFrame)
        list_of_simple_data_code = ComputeCharacterizationCodeAndMd5Code.list_of_simple_data_get_characterization_code([number_of_days_to_predict_ahead, number_of_test_days_in_a_day_level_DataFrame, number_of_rolling_forecast_days, windowing_option, input_length, lstm_is_many_to_many, do_normalization, number_of_output_variables])
        return ComputeCharacterizationCodeAndMd5Code.hash_to_MD5(seperator.join(['train_input_output_tensor_list', day_code, list_of_simple_data_code, dataframe_code]))

    @staticmethod
    def test_input_output_tensor_list_get_MD5_code(day, not_normalized_combined_train_DataFrame, number_of_days_to_predict_ahead, number_of_test_days_in_a_day_level_DataFrame, number_of_rolling_forecast_days, windowing_option, input_length, lstm_is_many_to_many, do_normalization, number_of_output_variables, seperator='_'):
        day_code = ComputeCharacterizationCodeAndMd5Code.date_get_characterization_code(day)
        dataframe_code = ComputeCharacterizationCodeAndMd5Code.dataframe_get_characterization_code(not_normalized_combined_train_DataFrame)
        list_of_simple_data_code = ComputeCharacterizationCodeAndMd5Code.list_of_simple_data_get_characterization_code([number_of_days_to_predict_ahead, number_of_test_days_in_a_day_level_DataFrame, number_of_rolling_forecast_days, windowing_option, input_length, lstm_is_many_to_many, do_normalization, number_of_output_variables])
        return ComputeCharacterizationCodeAndMd5Code.hash_to_MD5(seperator.join(['test_input_output_tensor_list', day_code, list_of_simple_data_code, dataframe_code]))

    @staticmethod
    def get_class_MD5_code(target_class=None, parameter_to_modified_value_dictionary=None, store=False, md5_to_parameter_to_value_mapping_file_path='data/asdf', seperator='_'):
        parameter_to_default_value_dictionary = ComputeCharacterizationCodeAndMd5Code.get_parameter_to_default_value_dictionary_for_a_class(target_class) # get the default parameters for pipelineInterval and do some changes
        parameter_to_default_and_modified_value_dictionary = copy.deepcopy(parameter_to_default_value_dictionary)
        parameter_to_default_and_modified_value_dictionary.update(parameter_to_modified_value_dictionary)
        dictionary_code = ComputeCharacterizationCodeAndMd5Code.dictionary_get_characterization_code(parameter_to_default_and_modified_value_dictionary)
        class_md5_code = ComputeCharacterizationCodeAndMd5Code.hash_to_MD5(seperator.join([target_class.__name__, dictionary_code]))
        if store:
            _ = ComputeCharacterizationCodeAndMd5Code.store_mapping_from_MD5_code_to_parameter_to_value(md5_code=class_md5_code, parameter_to_value_dictionary=parameter_to_default_and_modified_value_dictionary, mapping_record_file_path=md5_to_parameter_to_value_mapping_file_path)

        return class_md5_code

    @staticmethod
    def store_mapping_from_MD5_code_to_parameter_to_value(md5_code='the_defalt_code', parameter_to_value_dictionary=None, mapping_record_file_path='default_file_path'):
        # make dataframe
        md5_to_parameter_to_value_dataframe = SaveRecord.convert_dictionary_into_parameter_to_value_dataframe(parameter_to_value_dictionary)
        md5_to_parameter_to_value_dataframe.insert(loc=0, column='md5', value=md5_code)
        string_dataframe_to_append = md5_to_parameter_to_value_dataframe.to_csv(header=False)  # the to_csv does not add new line at the end of the string

        # check for file existence
        if path.exists(mapping_record_file_path):
            pass
        else:
            SaveRecord.create_file_and_write_string(file_path=mapping_record_file_path, string_to_write=',md5,variable,value\n')

        # append to file
        SaveRecord.append_string_to_file(file_path=mapping_record_file_path, string_to_append=string_dataframe_to_append)

        return md5_to_parameter_to_value_dataframe

    '''Miscellanous functions'''
    @staticmethod
    def get_default_cached_file_name(characterization_code):
        return('data/'+characterization_code)

    @staticmethod
    def get_parameter_to_default_value_dictionary_for_a_class(target_class):
        parameter_to_default_value_dictionary = {}  # contains modifed and default values for paraemtesr
        for parameter in list(inspect.signature(target_class).parameters.values()):
            parameter_to_default_value_dictionary[parameter.name] = parameter.default
            # print(parameter.name, parameter.default)
        return parameter_to_default_value_dictionary
