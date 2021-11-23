from autoForecastVA.src.analyzer.analyzer_pipeline_autoForecastVA import AnalyzerPipeLineAutoForecastVA
from autoForecastVA.src.components.data_filters.filter_controller import FilterController
from autoForecastVA.src.components.data_filters.filter_customized import FilterProportionOfMissingDays, FilterStandardDeviationDividedByMean

from autoForecastVA.src.components.data_preprocessing_global.data_clustering_and_normalization import DataCluseteringAndNormalization

from autoForecastVA.src.components.data_preprocessing_global.data_delimiting_by_day_and_medical_center import DataDelimitingByDayAndMedicalCenter

from autoForecastVA.src.components.data_preprocessing_global.data_imputation_and_averaging import DataImputationAndAveraging
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import ConfigSpace as CS
import ConfigSpace.hyperparameters as CSH
from torch import nn
import torch






from autoForecastVA.src.components.hyper_parameter_tuning.hyper_parameter_tuning_worker import LSTMWorker




import unittest
## data filter: find the subset of hospitals to perform autoforecast
from autoForecastVA.src.components.data_filters.filter_controller import FilterController
from autoForecastVA.src.components.data_filters.filter_customized import FilterProportionOfMissingDays
from autoForecastVA.src.components.data_preprocessing_global.data_imputation_and_averaging import DataImputationAndAveraging
from autoForecastVA.src.data_caching.variable_load_and_save import VariableSaverAndLoader
from autoForecastVA.src.pipelines.pipeline_autoForecastVA import PipeLineAutoForecastVA


class TestPipeLineAutoForecastVA(unittest.TestCase):
    def test_toy_pipeline(self):
        # maximum_allowable_proportion_of_missing_data = 0.05
        # maximum_allowable_divided_value = 1
        #
        # dataset_path = '../../data/coviddata07292020.csv'
        #
        # dataImputationAndAveraging = DataImputationAndAveraging(dataset_path=dataset_path, lazy=True, number_of_days_for_data_averaging=0)
        # dataImputationAndAveraging.read_dataset()
        # raw_dataset = dataImputationAndAveraging.get_processed_dataset()
        # filter_class_to_kwargs_dictionary = {FilterProportionOfMissingDays: {'dataset': raw_dataset, 'maximum_allowable_proportion_of_missing_data': maximum_allowable_proportion_of_missing_data, 'verbose': verbose}, FilterStandardDeviationDividedByMean: {'dataset': raw_dataset, 'maximum_allowable_divided_value': maximum_allowable_divided_value, 'verbose': verbose}}
        # filterController = FilterController(filter_class_to_kwargs_dictionary=filter_class_to_kwargs_dictionary)

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
        nameserver_port = 65308  # dont be 80, 8080, 443 or below 1024; each instance (that runs on the same local computer) should have a unique nameserver_port
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

        self.assertEqual(list(pipeLineAutoForecastVA.day_level_medical_center_level_modelFineTuning_dictionary.items())[0][1]['501'].evaluator.list_of_rolling_forecast_daily_list_of_normalization_reverted_test_prediction_targets[0][0], 8.5)
        self.assertEqual(list(pipeLineAutoForecastVA.day_level_medical_center_level_modelFineTuning_dictionary.items())[1][1]['501'].evaluator.list_of_rolling_forecast_daily_list_of_normalization_reverted_test_prediction_targets[0][0], 9.0)

        self.assertEqual(str(list(pipeLineAutoForecastVA.day_level_medical_center_level_modelFineTuning_dictionary.items())[0][1]['501'].evaluator.list_of_rolling_forecast_daily_list_of_normalization_reverted_test_predictions[0][0]), str(7.0483756))
        self.assertEqual(str(list(pipeLineAutoForecastVA.day_level_medical_center_level_modelFineTuning_dictionary.items())[1][1]['501'].evaluator.list_of_rolling_forecast_daily_list_of_normalization_reverted_test_predictions[0][0]), str(6.6763873))

        # AnalyzerPipeLineAutoForecastVA.save_analyzer_result_in_a_memory_efficient_way(pipeLineAutoForecastVA, file_name='data/toy_pipeline_memory_efficient_analyzer.dat')

    def test_toy_pipeline_more_options(self):
        '''We enable tuning interval, not do normalization and not do feature selection'''

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

        number_of_rolling_forecast_days = 5

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
        nameserver_port = 65308  # dont be 80, 8080, 443 or below 1024; each instance (that runs on the same local computer) should have a unique nameserver_port
        min_budget = 10
        max_budget = 50  # also train the general model with max budget with early stopping
        n_iterations = 2
        tuning_interval = 2

        neural_network_training_seed = 0
        loss_function = nn.MSELoss()

        trainable_parameter_name_list = ['linear.weight', 'linear.bias']

        do_clustering = False
        do_normalization = False  # should be true in practice; but set to false to better inspect the process
        do_feature_selection = False

        batch_size_for_tuning = 60
        batch_size_for_general_model_building = 60
        batch_size_for_fine_tuning = 60

        generate_train_prediction_during_tuning = True
        enable_data_caching_during_tuning = False

        verbose = True

        pipeLineAutoForecastVA = PipeLineAutoForecastVA(medical_center_subset=medical_center_subset, time_period_subset=time_period_subset, dataset_path=dataset_path, number_of_days_for_data_averaging=number_of_days_for_data_averaging, max_number_of_cluster=max_number_of_cluster, number_of_days_to_predict_ahead=number_of_days_to_predict_ahead, number_of_test_days_in_a_day_level_DataFrame=number_of_test_days_in_a_day_level_DataFrame, p_value_threshold=p_value_threshold, data_windowing_option=data_windowing_option, input_length=input_length, lstm_is_many_to_many=lstm_is_many_to_many, number_of_rolling_forecast_days=number_of_rolling_forecast_days, hyper_parameter_tuning_number_of_test_days_in_DataFrame=hyper_parameter_tuning_number_of_test_days_in_DataFrame, hyper_parameter_tuning_number_of_rolling_forecast_days=hyper_parameter_tuning_number_of_rolling_forecast_days, hyper_parameter_space=hyper_parameter_space, hyper_parameter_space_seed=hyper_parameter_space_seed, numpy_seed=numpy_seed, CustomizedWorker=CustomizedWorker, nameserver_run_id=nameserver_run_id, nameserver_address=nameserver_address, nameserver_port=nameserver_port, min_budget=min_budget, max_budget=max_budget, n_iterations=n_iterations, tuning_interval=tuning_interval, neural_network_training_seed=neural_network_training_seed, loss_function=loss_function, trainable_parameter_name_list=trainable_parameter_name_list, do_clustering=do_clustering, do_normalization=do_normalization, do_feature_selection=do_feature_selection, batch_size_for_tuning=batch_size_for_tuning, batch_size_for_general_model_building=batch_size_for_general_model_building, batch_size_for_fine_tuning=batch_size_for_fine_tuning, generate_train_prediction_during_tuning=generate_train_prediction_during_tuning, enable_data_caching_during_tuning=enable_data_caching_during_tuning, verbose=verbose)

        analyzerPipeLineAutoForecastVA = AnalyzerPipeLineAutoForecastVA(pipeLineAutoForecastVA)

        self.assertListEqual(analyzerPipeLineAutoForecastVA.day_to_clinic_to_test_prediction_dataframe.test_prediction.tolist()[12:15], [9.370858192443848, 8.79678726196289, 5.002796173095703])
        self.assertListEqual(analyzerPipeLineAutoForecastVA.day_to_clinic_to_test_prediction_target_dataframe.test_prediction_target.tolist()[:5], [13.5, 5.5, 8.0, 6.0, 2.5])
        self.assertListEqual(analyzerPipeLineAutoForecastVA.clinic_to_test_prediction_loss_dataframe.loss.tolist(), [74.25349761466842, 12.724102327337686, 13.368278386832026, 11.745059620072244, 6.125066349574854, 18.282861796305042, 23.294115918760507, 7.9167110980570214, 12.488128834346753, 4.763615706781001])
        self.assertEqual(analyzerPipeLineAutoForecastVA.test_prediction_loss, 18.496143765273555)


    @unittest.skip
    def test_minimally_signficant_pipeline(self):

        medical_center_subset = ['402', '405', '436', '437', '438', '442', '459', '460', '463', '501', '503', '506', '508', '509', '512', '515', '516', '518', '519', '521', '523', '528', '528A6', '528A7', '528A8']

        time_period_subset = None  # default to be None

        dataset_path = '../../data/coviddata07292020.csv'
        number_of_days_for_data_averaging = 3

        max_number_of_cluster = 3

        number_of_days_to_predict_ahead = 1
        number_of_test_days_in_a_day_level_DataFrame = 1

        p_value_threshold = 0.05  # for feature selection

        data_windowing_option = 2
        input_length = 10
        lstm_is_many_to_many = False

        number_of_rolling_forecast_days = 10

        hyper_parameter_tuning_number_of_test_days_in_DataFrame = 7  # usually equal to 1
        hyper_parameter_tuning_number_of_rolling_forecast_days = 1  # usually greater than one

        hyper_parameter_space = CS.ConfigurationSpace()
        learning_rate = CSH.UniformFloatHyperparameter('learning_rate', lower=0.00001, upper=0.1, log=True)
        number_of_hidden_dimensions = CSH.UniformIntegerHyperparameter('number_of_hidden_dimensions', lower=5, upper=50)
        dropout_rate = CSH.UniformFloatHyperparameter('dropout_rate', lower=0, upper=0.7)
        hyper_parameters = [learning_rate, number_of_hidden_dimensions, dropout_rate]
        hyper_parameter_space.add_hyperparameters(hyper_parameters)

        hyper_parameter_space_seed = 0  # randomness in hyperparemter space sampling

        numpy_seed = 0  # controls randomness in BOHB optimizer

        CustomizedWorker = LSTMWorker  # the worker for runnning BOHB
        nameserver_run_id = 'example1'  # can use the same run id
        nameserver_address = '127.0.0.1'  # can use the same address
        nameserver_port = 65300  # dont be 80, 8080, 443 or below 1024; each instance (that runs on the same local computer) should have a unique nameserver_port
        min_budget = 15
        max_budget = 160  # also train the general model with max budget with early stopping
        n_iterations = 4
        neural_network_training_seed = 0
        loss_function = nn.MSELoss()

        # trainable_parameter_name_list = ['linear.weight', 'linear.bias']
        trainable_parameter_name_list = ['lstm.weight_ih_l0', 'lstm.weight_hh_l0', 'lstm.bias_ih_l0', 'lstm.bias_hh_l0', 'linear.weight', 'linear.bias']

        do_normalization = True  # should be true in practice; but set to false to better inspect the process

        batch_size_for_tuning = 60
        batch_size_for_general_model_building = 60
        batch_size_for_fine_tuning = 60

        generate_train_prediction_during_tuning = True
        enable_data_caching_during_tuning = False

        verbose = True

        pipeLineAutoForecastVA = PipeLineAutoForecastVA(medical_center_subset=medical_center_subset, time_period_subset=time_period_subset, dataset_path=dataset_path, number_of_days_for_data_averaging=number_of_days_for_data_averaging, max_number_of_cluster=max_number_of_cluster, number_of_days_to_predict_ahead=number_of_days_to_predict_ahead, number_of_test_days_in_a_day_level_DataFrame=number_of_test_days_in_a_day_level_DataFrame, p_value_threshold=p_value_threshold, data_windowing_option=data_windowing_option, input_length=input_length, lstm_is_many_to_many=lstm_is_many_to_many, number_of_rolling_forecast_days=number_of_rolling_forecast_days, hyper_parameter_tuning_number_of_test_days_in_DataFrame=hyper_parameter_tuning_number_of_test_days_in_DataFrame, hyper_parameter_tuning_number_of_rolling_forecast_days=hyper_parameter_tuning_number_of_rolling_forecast_days, hyper_parameter_space=hyper_parameter_space, hyper_parameter_space_seed=hyper_parameter_space_seed, numpy_seed=numpy_seed, CustomizedWorker=CustomizedWorker, nameserver_run_id=nameserver_run_id, nameserver_address=nameserver_address, nameserver_port=nameserver_port, min_budget=min_budget, max_budget=max_budget, n_iterations=n_iterations, neural_network_training_seed=neural_network_training_seed, loss_function=loss_function, trainable_parameter_name_list=trainable_parameter_name_list, do_normalization=do_normalization, batch_size_for_tuning=batch_size_for_tuning, batch_size_for_general_model_building=batch_size_for_general_model_building, batch_size_for_fine_tuning=batch_size_for_fine_tuning, generate_train_prediction_during_tuning=generate_train_prediction_during_tuning, enable_data_caching_during_tuning=enable_data_caching_during_tuning, verbose=verbose)

        AnalyzerPipeLineAutoForecastVA.save_analyzer_result_in_a_memory_efficient_way(pipeLineAutoForecastVA, file_name='data/memory_efficient_analyzer.dat')



    @unittest.skip
    def test_minimally_signficant_pipeline_with_tuning_frequency(self):

        medical_center_subset = ['402', '405', '436', '437', '438', '442', '459', '460', '463', '501', '503', '506', '508', '509', '512', '515', '516', '518', '519', '521', '523', '528', '528A6', '528A7', '528A8']

        time_period_subset = None  # default to be None

        dataset_path = '../../data/coviddata07292020.csv'
        number_of_days_for_data_averaging = 3

        max_number_of_cluster = 3

        number_of_days_to_predict_ahead = 1
        number_of_test_days_in_a_day_level_DataFrame = 1

        p_value_threshold = 0.05  # for feature selection

        data_windowing_option = 2
        input_length = 10
        lstm_is_many_to_many = False

        number_of_rolling_forecast_days = 10

        hyper_parameter_tuning_number_of_test_days_in_DataFrame = 7  # usually equal to 1
        hyper_parameter_tuning_number_of_rolling_forecast_days = 1  # usually greater than one

        hyper_parameter_space = CS.ConfigurationSpace()
        learning_rate = CSH.UniformFloatHyperparameter('learning_rate', lower=0.00001, upper=0.1, log=True)
        number_of_hidden_dimensions = CSH.UniformIntegerHyperparameter('number_of_hidden_dimensions', lower=5, upper=50)
        dropout_rate = CSH.UniformFloatHyperparameter('dropout_rate', lower=0, upper=0.7)
        hyper_parameters = [learning_rate, number_of_hidden_dimensions, dropout_rate]
        hyper_parameter_space.add_hyperparameters(hyper_parameters)

        hyper_parameter_space_seed = 0  # randomness in hyperparemter space sampling

        numpy_seed = 0  # controls randomness in BOHB optimizer

        CustomizedWorker = LSTMWorker  # the worker for runnning BOHB
        nameserver_run_id = 'example1'  # can use the same run id
        nameserver_address = '127.0.0.1'  # can use the same address
        nameserver_port = 65304  # dont be 80, 8080, 443 or below 1024; each instance (that runs on the same local computer) should have a unique nameserver_port
        min_budget = 15
        max_budget = 160  # also train the general model with max budget with early stopping
        n_iterations = 4
        tuning_frequency = 3
        neural_network_training_seed = 0
        loss_function = nn.MSELoss()

        # trainable_parameter_name_list = ['linear.weight', 'linear.bias']
        trainable_parameter_name_list = ['lstm.weight_ih_l0', 'lstm.weight_hh_l0', 'lstm.bias_ih_l0', 'lstm.bias_hh_l0', 'linear.weight', 'linear.bias']

        do_normalization = True  # should be true in practice; but set to false to better inspect the process

        batch_size_for_tuning = 60
        batch_size_for_general_model_building = 60
        batch_size_for_fine_tuning = 60

        generate_train_prediction_during_tuning = True
        enable_data_caching_during_tuning = False

        verbose = True

        pipeLineAutoForecastVA = PipeLineAutoForecastVA(medical_center_subset=medical_center_subset, time_period_subset=time_period_subset, dataset_path=dataset_path, number_of_days_for_data_averaging=number_of_days_for_data_averaging, max_number_of_cluster=max_number_of_cluster, number_of_days_to_predict_ahead=number_of_days_to_predict_ahead, number_of_test_days_in_a_day_level_DataFrame=number_of_test_days_in_a_day_level_DataFrame, p_value_threshold=p_value_threshold, data_windowing_option=data_windowing_option, input_length=input_length, lstm_is_many_to_many=lstm_is_many_to_many, number_of_rolling_forecast_days=number_of_rolling_forecast_days, hyper_parameter_tuning_number_of_test_days_in_DataFrame=hyper_parameter_tuning_number_of_test_days_in_DataFrame, hyper_parameter_tuning_number_of_rolling_forecast_days=hyper_parameter_tuning_number_of_rolling_forecast_days, hyper_parameter_space=hyper_parameter_space, hyper_parameter_space_seed=hyper_parameter_space_seed, numpy_seed=numpy_seed, CustomizedWorker=CustomizedWorker, nameserver_run_id=nameserver_run_id, nameserver_address=nameserver_address, nameserver_port=nameserver_port, min_budget=min_budget, max_budget=max_budget, n_iterations=n_iterations, neural_network_training_seed=neural_network_training_seed, loss_function=loss_function, trainable_parameter_name_list=trainable_parameter_name_list, do_normalization=do_normalization, tuning_interval=tuning_frequency, batch_size_for_tuning=batch_size_for_tuning, batch_size_for_general_model_building=batch_size_for_general_model_building, batch_size_for_fine_tuning=batch_size_for_fine_tuning, generate_train_prediction_during_tuning=generate_train_prediction_during_tuning, enable_data_caching_during_tuning=enable_data_caching_during_tuning, verbose=verbose)

        AnalyzerPipeLineAutoForecastVA.save_analyzer_result_in_a_memory_efficient_way(pipeLineAutoForecastVA, file_name='data/tuning_frequency_memory_efficient_analyzer.dat')


    @unittest.skip
    def test_full_pipeline(self):

        medical_center_subset = ['402', '405', '436', '437', '438', '442', '459', '460', '463', '501', '503', '506', '508', '509', '512', '515', '516', '518', '519', '521', '523', '528', '528A6', '528A7', '528A8', '529', '531', '534', '537', '539', '541', '542', '544', '548', '549', '550', '552', '554', '556', '557', '558', '562', '568', '570', '573', '575', '578', '580', '581', '583', '585', '589', '589A5', '589A7', '590', '593', '595', '596', '600', '603', '605', '607', '608', '610', '612A4', '614', '618', '619', '621', '623', '626', '631', '635', '636', '636A6', '636A8', '637', '640', '642', '644', '646', '648', '649', '650', '652', '654', '655', '656', '657', '657A5', '658', '659', '660', '662', '663', '664', '668', '671', '673', '674', '675', '676', '678', '688', '689', '691', '693', '695', '756', '757']

        time_period_subset = None  # default to be None

        dataset_path = '../../data/coviddata07292020.csv'
        number_of_days_for_data_averaging = 3

        max_number_of_cluster = 10

        number_of_days_to_predict_ahead = 1
        number_of_test_days_in_a_day_level_DataFrame = 1

        p_value_threshold = 0.05  # for feature selection

        data_windowing_option = 2
        input_length = 10
        lstm_is_many_to_many = False

        number_of_rolling_forecast_days = 30  # TODO might consider changing this

        hyper_parameter_tuning_number_of_test_days_in_DataFrame = 10  # usually equal to 1
        hyper_parameter_tuning_number_of_rolling_forecast_days = 1  # usually greater than one

        hyper_parameter_space = CS.ConfigurationSpace()
        learning_rate = CSH.UniformFloatHyperparameter('learning_rate', lower=0.00001, upper=0.1, log=True)
        number_of_hidden_dimensions = CSH.UniformIntegerHyperparameter('number_of_hidden_dimensions', lower=5, upper=50)
        dropout_rate = CSH.UniformFloatHyperparameter('dropout_rate', lower=0, upper=0.7)
        hyper_parameters = [learning_rate, number_of_hidden_dimensions, dropout_rate]
        hyper_parameter_space.add_hyperparameters(hyper_parameters)

        hyper_parameter_space_seed = 0  # randomness in hyperparemter space sampling

        numpy_seed = 0  # controls randomness in BOHB optimizer

        CustomizedWorker = LSTMWorker  # the worker for runnning BOHB
        nameserver_run_id = 'example1'  # can use the same run id
        nameserver_address = '127.0.0.1'  # can use the same address
        nameserver_port = 65300  # dont be 80, 8080, 443 or below 1024; each instance (that runs on the same local computer) should have a unique nameserver_port
        min_budget = 15
        max_budget = 160  # also train the general model with max budget with early stopping
        n_iterations = 4
        tuning_frequency = 5  # TODO implement this
        neural_network_training_seed = 0
        loss_function = nn.MSELoss()

        # trainable_parameter_name_list = ['linear.weight', 'linear.bias']
        trainable_parameter_name_list = ['lstm.weight_ih_l0', 'lstm.weight_hh_l0', 'lstm.bias_ih_l0', 'lstm.bias_hh_l0', 'linear.weight', 'linear.bias']

        do_normalization = True  # should be true in practice; but set to false to better inspect the process

        batch_size_for_tuning = 60
        batch_size_for_general_model_building = 60
        batch_size_for_fine_tuning = 60

        generate_train_prediction_during_tuning = False
        enable_data_caching_during_tuning = True

        verbose = True

        pipeLineAutoForecastVA = PipeLineAutoForecastVA(medical_center_subset=medical_center_subset, time_period_subset=time_period_subset, dataset_path=dataset_path, number_of_days_for_data_averaging=number_of_days_for_data_averaging, max_number_of_cluster=max_number_of_cluster, number_of_days_to_predict_ahead=number_of_days_to_predict_ahead, number_of_test_days_in_a_day_level_DataFrame=number_of_test_days_in_a_day_level_DataFrame, p_value_threshold=p_value_threshold, data_windowing_option=data_windowing_option, input_length=input_length, lstm_is_many_to_many=lstm_is_many_to_many, number_of_rolling_forecast_days=number_of_rolling_forecast_days, hyper_parameter_tuning_number_of_test_days_in_DataFrame=hyper_parameter_tuning_number_of_test_days_in_DataFrame, hyper_parameter_tuning_number_of_rolling_forecast_days=hyper_parameter_tuning_number_of_rolling_forecast_days, hyper_parameter_space=hyper_parameter_space, hyper_parameter_space_seed=hyper_parameter_space_seed, numpy_seed=numpy_seed, CustomizedWorker=CustomizedWorker, nameserver_run_id=nameserver_run_id, nameserver_address=nameserver_address, nameserver_port=nameserver_port, min_budget=min_budget, max_budget=max_budget, n_iterations=n_iterations, neural_network_training_seed=neural_network_training_seed, loss_function=loss_function, trainable_parameter_name_list=trainable_parameter_name_list, do_normalization=do_normalization, batch_size_for_tuning=batch_size_for_tuning, batch_size_for_general_model_building=batch_size_for_general_model_building, batch_size_for_fine_tuning=batch_size_for_fine_tuning, generate_train_prediction_during_tuning=generate_train_prediction_during_tuning, enable_data_caching_during_tuning=enable_data_caching_during_tuning, verbose=verbose)

        AnalyzerPipeLineAutoForecastVA.save_analyzer_result_in_a_memory_efficient_way(pipeLineAutoForecastVA, file_name='data/memory_efficient_analyzer.dat')

    @unittest.skip
    def test_full_pipeline_with_tuning_frequency(self):

        medical_center_subset = ['402', '405', '436', '437', '438', '442', '459', '460', '463', '501', '503', '506', '508', '509', '512', '515', '516', '518', '519', '521', '523', '528', '528A6', '528A7', '528A8', '529', '531', '534', '537', '539', '541', '542', '544', '548', '549', '550', '552', '554', '556', '557', '558', '562', '568', '570', '573', '575', '578', '580', '581', '583', '585', '589', '589A5', '589A7', '590', '593', '595', '596', '600', '603', '605', '607', '608', '610', '612A4', '614', '618', '619', '621', '623', '626', '631', '635', '636', '636A6', '636A8', '637', '640', '642', '644', '646', '648', '649', '650', '652', '654', '655', '656', '657', '657A5', '658', '659', '660', '662', '663', '664', '668', '671', '673', '674', '675', '676', '678', '688', '689', '691', '693', '695', '756', '757']

        time_period_subset = None  # default to be None

        dataset_path = '../../data/coviddata07292020.csv'
        number_of_days_for_data_averaging = 3

        max_number_of_cluster = 10

        number_of_days_to_predict_ahead = 1
        number_of_test_days_in_a_day_level_DataFrame = 1

        p_value_threshold = 0.05  # for feature selection

        data_windowing_option = 2
        input_length = 10
        lstm_is_many_to_many = False

        number_of_rolling_forecast_days = 30  # TODO might consider changing this

        hyper_parameter_tuning_number_of_test_days_in_DataFrame = 10  # usually equal to 1
        hyper_parameter_tuning_number_of_rolling_forecast_days = 1  # usually greater than one

        hyper_parameter_space = CS.ConfigurationSpace()
        learning_rate = CSH.UniformFloatHyperparameter('learning_rate', lower=0.00001, upper=0.1, log=True)
        number_of_hidden_dimensions = CSH.UniformIntegerHyperparameter('number_of_hidden_dimensions', lower=5, upper=50)
        dropout_rate = CSH.UniformFloatHyperparameter('dropout_rate', lower=0, upper=0.7)
        hyper_parameters = [learning_rate, number_of_hidden_dimensions, dropout_rate]
        hyper_parameter_space.add_hyperparameters(hyper_parameters)

        hyper_parameter_space_seed = 0  # randomness in hyperparemter space sampling

        numpy_seed = 0  # controls randomness in BOHB optimizer

        CustomizedWorker = LSTMWorker  # the worker for runnning BOHB
        nameserver_run_id = 'example1'  # can use the same run id
        nameserver_address = '127.0.0.1'  # can use the same address
        nameserver_port = 65300  # dont be 80, 8080, 443 or below 1024; each instance (that runs on the same local computer) should have a unique nameserver_port
        min_budget = 15
        max_budget = 160  # also train the general model with max budget with early stopping
        n_iterations = 4
        tuning_frequency = 5
        neural_network_training_seed = 0
        loss_function = nn.MSELoss()

        # trainable_parameter_name_list = ['linear.weight', 'linear.bias']
        trainable_parameter_name_list = ['lstm.weight_ih_l0', 'lstm.weight_hh_l0', 'lstm.bias_ih_l0', 'lstm.bias_hh_l0', 'linear.weight', 'linear.bias']

        do_normalization = True  # should be true in practice; but set to false to better inspect the process

        batch_size_for_tuning = 60
        batch_size_for_general_model_building = 60
        batch_size_for_fine_tuning = 60

        generate_train_prediction_during_tuning = False
        enable_data_caching_during_tuning = True

        verbose = True

        pipeLineAutoForecastVA = PipeLineAutoForecastVA(medical_center_subset=medical_center_subset, time_period_subset=time_period_subset, dataset_path=dataset_path, number_of_days_for_data_averaging=number_of_days_for_data_averaging, max_number_of_cluster=max_number_of_cluster, number_of_days_to_predict_ahead=number_of_days_to_predict_ahead, number_of_test_days_in_a_day_level_DataFrame=number_of_test_days_in_a_day_level_DataFrame, p_value_threshold=p_value_threshold, data_windowing_option=data_windowing_option, input_length=input_length, lstm_is_many_to_many=lstm_is_many_to_many, number_of_rolling_forecast_days=number_of_rolling_forecast_days, hyper_parameter_tuning_number_of_test_days_in_DataFrame=hyper_parameter_tuning_number_of_test_days_in_DataFrame, hyper_parameter_tuning_number_of_rolling_forecast_days=hyper_parameter_tuning_number_of_rolling_forecast_days, hyper_parameter_space=hyper_parameter_space, hyper_parameter_space_seed=hyper_parameter_space_seed, numpy_seed=numpy_seed, CustomizedWorker=CustomizedWorker, nameserver_run_id=nameserver_run_id, nameserver_address=nameserver_address, nameserver_port=nameserver_port, min_budget=min_budget, max_budget=max_budget, n_iterations=n_iterations, neural_network_training_seed=neural_network_training_seed, loss_function=loss_function, trainable_parameter_name_list=trainable_parameter_name_list, do_normalization=do_normalization, tuning_interval=tuning_frequency, batch_size_for_tuning=batch_size_for_tuning, batch_size_for_general_model_building=batch_size_for_general_model_building, batch_size_for_fine_tuning=batch_size_for_fine_tuning, generate_train_prediction_during_tuning=generate_train_prediction_during_tuning, enable_data_caching_during_tuning=enable_data_caching_during_tuning, verbose=verbose)

        # AnalyzerPipeLineAutoForecastVA.save_analyzer_result_in_a_memory_efficient_way(pipeLineAutoForecastVA, file_name='data/memory_efficient_analyzer.dat')



