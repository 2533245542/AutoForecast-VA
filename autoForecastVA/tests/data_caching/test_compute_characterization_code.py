import unittest
import os
import pandas as pd
import torch
from torch import nn

from autoForecastVA.src.data_caching.compute_characterization_code import ComputeCharacterizationCodeAndMd5Code
from autoForecastVA.src.pipelines.pipeline_ablation_studies import PipeLineTuningInterval


class TestComputeCharacterizationCode(unittest.TestCase):
    def test_get_characterization_code(self):
        '''We ensure the get characterization code functions are working correctly'''
        torch_seed = 0
        hyper_parameter_value_combination = {'dropout_rate': 0.0, 'learning_rate': 0.001, 'number_of_hidden_dimensions': 5, 'number_of_training_epochs': 5}
        number_of_rolling_forecast_days = 1
        number_of_test_days_in_a_day_level_DataFrame = 9

        speed_up_inference = False
        input_length = 3
        data_windowing_option = 2
        lstm_is_many_to_many = False
        do_normalization = True

        number_of_days_to_predict_ahead = 1

        total_data_size = 10 + 10 + input_length - 1 + number_of_days_to_predict_ahead

        loss_function = nn.MSELoss()
        torch.manual_seed(torch_seed)

        DataFrame = pd.read_csv('../components/hyper_parameter_tuning/jena_climate_2009_2016_autoforecastva.csv')
        date = pd.date_range(start='2020-01-01',
                             end=pd.to_datetime('2020-01-01') + pd.Timedelta(len(DataFrame) - 1, unit='d'), name='date')
        DataFrame.index = date
        DataFrame = DataFrame.iloc[:total_data_size, 1:]
        DataFrame = DataFrame.iloc[:, [1,2]]
        DataFrame = DataFrame.rename(columns={'T (degC)': 'case'})
        DataFrame.insert(0, 'clinic', '111')

        list_of_simple_data = [number_of_rolling_forecast_days, number_of_test_days_in_a_day_level_DataFrame, speed_up_inference, input_length, data_windowing_option, lstm_is_many_to_many, do_normalization, number_of_days_to_predict_ahead]

        self.assertEqual(ComputeCharacterizationCodeAndMd5Code.hash_to_MD5('aasdfas8df12873f7823f877802'), '07e0e4ca2e20b725833f27706c145c4b')
        self.assertEqual(ComputeCharacterizationCodeAndMd5Code.list_of_simple_data_get_characterization_code(list_of_simple_data=list_of_simple_data), '1_9_False_3_2_False_True_1')
        self.assertEqual(ComputeCharacterizationCodeAndMd5Code.hyper_parameter_value_combination_get_characterization_code(hyper_parameter_value_combination=hyper_parameter_value_combination), 'dropout_rate:0.0_learning_rate:0.001_number_of_hidden_dimensions:5_number_of_training_epochs:5')
        self.assertEqual(ComputeCharacterizationCodeAndMd5Code.dataframe_get_characterization_code(dataframe=DataFrame), '23_3_4.830917874396136e+66_-8.51608695652174_3.207826086956522_0.4226081263517042_0.10799648654803248')
        self.assertEqual(ComputeCharacterizationCodeAndMd5Code.date_get_characterization_code(DataFrame.index[0]), str(DataFrame.index[0].strftime("%Y%m%d%H%M%S")))
        self.assertEqual(ComputeCharacterizationCodeAndMd5Code.prepare_rolling_forecast_data_get_MD5_code(not_normalized_combined_train_DataFrame=DataFrame, number_of_days_to_predict_ahead=number_of_days_to_predict_ahead, number_of_test_days_in_a_day_level_DataFrame=number_of_test_days_in_a_day_level_DataFrame, number_of_rolling_forecast_days=number_of_rolling_forecast_days, windowing_option=data_windowing_option, input_length=input_length, lstm_is_many_to_many=lstm_is_many_to_many, do_normalization=do_normalization), '3a51fe7b6624c19585c5d6c9cd522bc4')
        self.assertEqual(ComputeCharacterizationCodeAndMd5Code.train_input_output_tensor_list_get_MD5_code(day=date[0], not_normalized_combined_train_DataFrame=DataFrame, number_of_days_to_predict_ahead=number_of_days_to_predict_ahead, number_of_test_days_in_a_day_level_DataFrame=number_of_test_days_in_a_day_level_DataFrame, number_of_rolling_forecast_days=number_of_rolling_forecast_days, windowing_option=data_windowing_option, input_length=input_length, lstm_is_many_to_many=lstm_is_many_to_many, do_normalization=do_normalization, number_of_output_variables=3), '761f586d8333dac7bff8d3b48b895b4f')
        self.assertEqual(ComputeCharacterizationCodeAndMd5Code.train_input_output_tensor_list_get_MD5_code(day=date[0], not_normalized_combined_train_DataFrame=DataFrame, number_of_days_to_predict_ahead=number_of_days_to_predict_ahead, number_of_test_days_in_a_day_level_DataFrame=number_of_test_days_in_a_day_level_DataFrame, number_of_rolling_forecast_days=number_of_rolling_forecast_days, windowing_option=data_windowing_option, input_length=input_length, lstm_is_many_to_many=lstm_is_many_to_many, do_normalization=do_normalization, number_of_output_variables=1), '87d8ffe2819e27c6691d7888bce512d5')

        '''for storing mapping'''
        md5_code = 'LK87A098JFIOG13HASD'
        file_path = 'data/file_for_testing_store_mapping_between_MD5_code_and_parameters.csv'
        parameter_to_default_value_dictionary_for_pipeline_interval = ComputeCharacterizationCodeAndMd5Code.get_parameter_to_default_value_dictionary_for_a_class(PipeLineTuningInterval)  # get the default parameters for pipelineInterval and do some changes

        md5_to_parameter_to_value_dataframe = ComputeCharacterizationCodeAndMd5Code.store_mapping_from_MD5_code_to_parameter_to_value(md5_code=md5_code, parameter_to_value_dictionary=parameter_to_default_value_dictionary_for_pipeline_interval, mapping_record_file_path=file_path)
        self.assertEqual(md5_to_parameter_to_value_dataframe.md5.unique().tolist()[0], md5_code)
        self.assertEqual(md5_to_parameter_to_value_dataframe.iloc[29,2], 1)
        self.assertEqual(md5_to_parameter_to_value_dataframe.iloc[12,2], 10)
        self.assertEqual(md5_to_parameter_to_value_dataframe.iloc[42,2], False)   # lazy=False
        self.assertTupleEqual(md5_to_parameter_to_value_dataframe.shape, (43, 3))
        self.assertTrue(os.path.exists(file_path))
        self.assertEqual(os.stat(file_path).st_size, 2300)
        self.assertEqual(os.remove(file_path), None) # file size

        '''for computing characterization code'''
        parameter_to_modified_value_dictionary = {'max_number_of_cluster': 10, 'number_of_days_to_predict_ahead': 1, 'lazy': True}
        store = True
        file_path = 'data/file_for_testing_get_pipeline_interval_MD5_code.csv'

        pipeLineTuningInterval_md5_code = ComputeCharacterizationCodeAndMd5Code.get_class_MD5_code(target_class=PipeLineTuningInterval, parameter_to_modified_value_dictionary=parameter_to_modified_value_dictionary, store=store, md5_to_parameter_to_value_mapping_file_path=file_path) # self.assertEqual(pipeLineTuningInterval_md5_code, )
        self.assertEqual(pipeLineTuningInterval_md5_code, 'e90d90190da293303c8675366813d8ea')
        self.assertTrue(os.path.exists(file_path))
        self.assertEqual(os.stat(file_path).st_size, 2858)
        self.assertEqual(os.remove(file_path), None)  # file size
