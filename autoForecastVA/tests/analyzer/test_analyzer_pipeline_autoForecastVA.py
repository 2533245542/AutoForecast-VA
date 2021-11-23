import unittest
import seaborn as sns
import pandas as pd
import os
import ConfigSpace as CS
import ConfigSpace.hyperparameters as CSH
from torch import nn
import matplotlib.pylab as plt

from autoForecastVA.src.analyzer.analyzer_pipeline_autoForecastVA import AnalyzerPipeLineAutoForecastVA
from autoForecastVA.src.components.hyper_parameter_tuning.hyper_parameter_tuning_loop import HyperParameterTuningLoop
from autoForecastVA.src.components.hyper_parameter_tuning.hyper_parameter_tuning_worker import LSTMWorker
from autoForecastVA.src.data_caching.variable_load_and_save import VariableSaverAndLoader
from autoForecastVA.src.pipelines.pipeline_autoForecastVA import PipeLineAutoForecastVA


class TestAnalyzerPipeLineAutoForecastVA(unittest.TestCase):
    def test_adaption_to_analyzer(self):
        '''
        first create a pipeline the same as the pipeline toy test
        then create an analyzer from pipeline
        get dictionary from pipeline and compare

        '''

        '''create the pipeline in toy test'''
        medical_center_subset = ['402', '405', '436', '437', '438', '442', '459', '460', '463', '501']  # default to be None
        time_period_subset = ['2020-3-1', '2020-4-1']  # default to be None

        dataset_path = '../../data/coviddata07292020.csv'
        number_of_days_for_data_averaging = 2

        max_number_of_cluster = 3

        number_of_days_to_predict_ahead = 1
        number_of_test_days_in_a_day_level_DataFrame = 1

        p_value_threshold = 0.05  # for feature selection

        data_windowing_option = 2
        input_length = 3
        lstm_is_many_to_many = False

        number_of_rolling_forecast_days = 2

        hyper_parameter_tuning_number_of_test_days_in_DataFrame = 1  # usually equal to 1
        hyper_parameter_tuning_number_of_rolling_forecast_days = 3  # usually greater than one

        hyper_parameter_space = CS.ConfigurationSpace()
        learning_rate = CSH.UniformFloatHyperparameter('learning_rate', lower=0.00001, upper=0.1, log=True)
        number_of_hidden_dimensions = CSH.UniformIntegerHyperparameter('number_of_hidden_dimensions', lower=2, upper=10)
        dropout_rate = CSH.UniformFloatHyperparameter('dropout_rate', lower=0, upper=0.7)
        hyper_parameters = [learning_rate, number_of_hidden_dimensions, dropout_rate]
        hyper_parameter_space.add_hyperparameters(hyper_parameters)

        hyper_parameter_space_seed = 0  # randomness in hyperparemter space sampling

        numpy_seed = 0  # controls randomness in BOHB optimizer

        CustomizedWorker = LSTMWorker  # the worker for runnning BOHB
        nameserver_run_id = 'example1'  # can use the same run id
        nameserver_address = '127.0.0.1'  # can use the same address
        nameserver_port = 65300  # dont be 80, 8080, 443 or below 1024; each instance (that runs on the same local computer) should have a unique nameserver_port
        min_budget = 10
        max_budget = 50  # also train the general model with max budget with early stopping
        n_iterations = 2

        neural_network_training_seed = 0
        loss_function = nn.MSELoss()

        trainable_parameter_name_list = ['linear.weight', 'linear.bias']

        do_normalization = False  # should be true in practice; but set to false to better inspect the process

        batch_size_for_tuning = 60
        batch_size_for_general_model_building = 60
        batch_size_for_fine_tuning = 60

        generate_train_prediction_during_tuning = True
        enable_data_caching_during_tuning = False

        verbose = True

        pipeLineAutoForecastVA = PipeLineAutoForecastVA(medical_center_subset=medical_center_subset, time_period_subset=time_period_subset, dataset_path=dataset_path, number_of_days_for_data_averaging=number_of_days_for_data_averaging, max_number_of_cluster=max_number_of_cluster, number_of_days_to_predict_ahead=number_of_days_to_predict_ahead, number_of_test_days_in_a_day_level_DataFrame=number_of_test_days_in_a_day_level_DataFrame, p_value_threshold=p_value_threshold, data_windowing_option=data_windowing_option, input_length=input_length, lstm_is_many_to_many=lstm_is_many_to_many, number_of_rolling_forecast_days=number_of_rolling_forecast_days, hyper_parameter_tuning_number_of_test_days_in_DataFrame=hyper_parameter_tuning_number_of_test_days_in_DataFrame, hyper_parameter_tuning_number_of_rolling_forecast_days=hyper_parameter_tuning_number_of_rolling_forecast_days, hyper_parameter_space=hyper_parameter_space, hyper_parameter_space_seed=hyper_parameter_space_seed, numpy_seed=numpy_seed, CustomizedWorker=CustomizedWorker, nameserver_run_id=nameserver_run_id, nameserver_address=nameserver_address, nameserver_port=nameserver_port, min_budget=min_budget, max_budget=max_budget, n_iterations=n_iterations, neural_network_training_seed=neural_network_training_seed, loss_function=loss_function, trainable_parameter_name_list=trainable_parameter_name_list, do_normalization=do_normalization, batch_size_for_tuning=batch_size_for_tuning, batch_size_for_general_model_building=batch_size_for_general_model_building, batch_size_for_fine_tuning=batch_size_for_fine_tuning, generate_train_prediction_during_tuning=generate_train_prediction_during_tuning, enable_data_caching_during_tuning=enable_data_caching_during_tuning, verbose=verbose)

        file_name = 'data/TestAnalyzerPipeLineAutoForecastVA_test_adaption_to_analyzer.dat'
        VariableSaverAndLoader(list_of_variables_to_save=[pipeLineAutoForecastVA], save=True, file_name=file_name)
        # pipeLineAutoForecastVA = VariableSaverAndLoader(load=True, file_name=file_name).list_of_loaded_variables[0]

        '''create analyzer'''
        analyzerPipeLineAutoForecastVA = AnalyzerPipeLineAutoForecastVA(pipeLineAutoForecastVA)

        '''compare'''
        self.assertListEqual(list(list(analyzerPipeLineAutoForecastVA.day_level_medical_center_level_test_prediction_target_dictionary.values())[0].values()), [11.5, 8.5, 7.5, 13.5, 6.0, 13.0, 6.0, 8.5, 21.0, 12.0])
        self.assertListEqual(list(list(analyzerPipeLineAutoForecastVA.day_level_medical_center_level_test_prediction_target_dictionary.values())[1].values()), [11.5, 6.0, 5.0, 4.5, 9.5, 23.5, 7.5, 4.0, 9.5, 9.0])
        self.assertListEqual([str(i) for i in list(list(analyzerPipeLineAutoForecastVA.day_level_medical_center_level_test_prediction_dictionary.values())[0].values())], [str(i) for i in [6.6389685, 9.0582285, 5.1609926, 9.07026, 9.006416, 8.795251, 6.385896, 7.0483756, 13.151907, 10.99342]])
        self.assertListEqual([str(i) for i in list(list(analyzerPipeLineAutoForecastVA.day_level_medical_center_level_test_prediction_dictionary.values())[1].values())], [str(i) for i in [11.529707, 10.706603, 11.030908, 9.960701, 9.676973, 17.011127, 6.5920467, 5.253193, 6.3969116, 6.6763873]])




