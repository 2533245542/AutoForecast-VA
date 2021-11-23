import hpbandster  # BOHB
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
import ConfigSpace as CS
import ConfigSpace.hyperparameters as CSH
import torch
from autoForecastVA.src.components.hyper_parameter_tuning.hyper_parameter_value_combination_evaluator import HyperParameterValueCombinationEvaluator
from torch import nn


from autoForecastVA.src.components.hyper_parameter_tuning.hyper_parameter_tuning_loop import HyperParameterTuningLoop
from autoForecastVA.src.components.hyper_parameter_tuning.hyper_parameter_tuning_worker import MyWorker
from autoForecastVA.src.components.hyper_parameter_tuning.hyper_parameter_tuning_worker import LSTMWorker


from autoForecastVA.src.components.data_preprocessing_global.data_clustering_and_normalization import DataCluseteringAndNormalization
from autoForecastVA.src.components.data_preprocessing_global.data_delimiting_by_day_and_medical_center import DataDelimitingByDayAndMedicalCenter
import unittest

from autoForecastVA.src.data_caching.variable_load_and_save import VariableSaverAndLoader


class TestHyperParameterTuningLoop(unittest.TestCase):


    '''
    for each hyper-parameter
        evaluate

    '''
    def test_toy_example(self):
        number_of_days_in_dataset = 10
        number_of_days_for_testing = 3
        number_of_test_days_in_DataFrame = 1
        number_of_days_to_predict_ahead = 2
        number_of_rolling_forecast_days = 3
        torch.manual_seed(0)

        loss_function = nn.MSELoss()
        data_windowing_option = 2
        input_length = 3
        lstm_is_many_to_many = False
        do_normalization = False
        batch_size = 1
        enable_data_caching = False

        hyper_parameter_space_seed = 0

        nameserver_run_id = 'example1'
        nameserver_address = '127.0.0.1'
        nameserver_port = 65305  # dont be 80, 8080, 443 or below 1024

        min_budget = 9
        max_budget = 28
        n_iterations = 3

        # convinient procedure of creating a DataFrame
        date = pd.date_range(start='2020-01-01', end='2020-01-{}'.format(number_of_days_in_dataset), name='date')
        case = pd.Series(list(range(1, 1 + number_of_days_in_dataset)))
        call = case * 3 + 10
        dayofweek = date.dayofweek
        weekofyear = date.isocalendar().week
        precursor_dataset = pd.DataFrame({'case': case.values.astype(np.float64), 'call': call.values.astype(np.float64), 'dayofweek': dayofweek.astype(np.float64), 'weekofyear': weekofyear.astype(np.float64)}, index=date)
        dataset_111 = precursor_dataset.copy()
        dataset_111.insert(0, 'clinic', '111')
        dataset_222 = precursor_dataset.copy()
        dataset_222.insert(0, 'clinic', '222')
        dataset_333 = precursor_dataset.copy()
        dataset_333.insert(0, 'clinic', '333')
        dataset = pd.concat([dataset_111, dataset_222, dataset_333], axis=0)

        hyper_parameter_space = CS.ConfigurationSpace()
        learning_rate = CSH.UniformFloatHyperparameter('learning_rate', lower=0.00001, upper=0.1, log=True)
        number_of_hidden_dimensions = CSH.UniformIntegerHyperparameter('number_of_hidden_dimensions', lower=30, upper=300)
        dropout_rate = CSH.UniformFloatHyperparameter('dropout_rate', lower=0, upper=0.7)
        hyper_parameters = [learning_rate, number_of_hidden_dimensions, dropout_rate]
        hyper_parameter_space.add_hyperparameters(hyper_parameters)
        hyper_parameter_space.seed(hyper_parameter_space_seed)

        worker_specific_parameter_to_value_dictionary = {'DataFrame': dataset,
            'loss_function': loss_function,
            'number_of_days_to_predict_ahead': number_of_days_to_predict_ahead,
            'number_of_test_days_in_a_day_level_DataFrame': number_of_test_days_in_DataFrame,
            'number_of_rolling_forecast_days': number_of_rolling_forecast_days,
            'data_windowing_option': data_windowing_option,
            'input_length': input_length,
            'lstm_is_many_to_many': lstm_is_many_to_many,
            'do_normalization': do_normalization,
            'batch_size': batch_size,
            'enable_data_caching': enable_data_caching
        }

        tuning_obect = HyperParameterTuningLoop(CustomizedWorker=LSTMWorker,
                                   worker_specific_parameter_to_value_dictionary=worker_specific_parameter_to_value_dictionary,
                                   hyper_parameter_space=hyper_parameter_space, nameserver_run_id=nameserver_run_id,
                                   nameserver_address=nameserver_address, nameserver_port=nameserver_port,
                                   min_budget=min_budget, max_budget=max_budget, n_iterations=n_iterations)

        self.assertEqual(tuning_obect.number_of_hyper_parameter_value_combination_evaluated, 10)
        self.assertEqual(tuning_obect.number_of_unique_hyper_parameter_value_combination_evaluated, 8)

        self.assertEqual(tuning_obect.number_of_total_budget_spent, 168.0)

        self.assertDictEqual(tuning_obect.optimal_partial_hyper_parameter_value_combination_found_among_max_budget_evaluations, {'dropout_rate': 0.5594109949517064, 'learning_rate': 0.0007013219779945792, 'number_of_hidden_dimensions': 241})

        self.assertDictEqual(tuning_obect.optimal_partial_hyper_parameter_value_combination_found_among_all_budget_evaluations, {'dropout_rate': 0.5594109949517064, 'learning_rate': 0.0007013219779945792, 'number_of_hidden_dimensions': 241})
        self.assertDictEqual(tuning_obect.info_of_optimal_partial_hyper_parameter_value_combination_found_among_all_budget_evaluations, {'id': (2, 0, 2), 'budget': 28.0, 'loss': 4.861063480377197, 'dropout_rate': 0.5594109949517064, 'learning_rate': 0.0007013219779945792, 'number_of_hidden_dimensions': 241})

    def test_development(self):
        import numpy as np
        np.random.seed(0)
        config_space = CS.ConfigurationSpace()
        config_space.add_hyperparameter(CS.UniformFloatHyperparameter('x', lower=0, upper=1))
        config_space.seed(0)
        config_space.sample_configuration()

        nameserver_run_id = 'example1'
        nameserver_address = '127.0.0.1'
        nameserver_port = 65510  # dont be 80, 8080, 443 or below 1024

        min_budget = 9
        max_budget = 243
        n_iterations = 4

        sleep_interval = 0.001

        tuning_obect = HyperParameterTuningLoop(CustomizedWorker=MyWorker,
                                   worker_specific_parameter_to_value_dictionary={'sleep_interval': sleep_interval},
                                   hyper_parameter_space=config_space, nameserver_run_id=nameserver_run_id,
                                   nameserver_address=nameserver_address, nameserver_port=nameserver_port,
                                   min_budget=min_budget, max_budget=max_budget, n_iterations=n_iterations)

        assert tuning_obect.number_of_hyper_parameter_value_combination_evaluated == 65
        assert tuning_obect.number_of_unique_hyper_parameter_value_combination_evaluated == 46

        assert tuning_obect.number_of_total_budget_spent == 3645.0
        assert tuning_obect.number_of_equivalent_max_budget_computation == 15.0

        assert tuning_obect.optimal_partial_hyper_parameter_value_combination_found_among_max_budget_evaluations == {
            'x': 3.4151701294818765e-06}
        assert tuning_obect.info_of_optimal_partial_hyper_parameter_value_combination_found_among_max_budget_evaluations == {
            'id': (2, 0, 2), 'budget': 243.0, 'loss': 243.00000341517014, 'x': 3.4151701294818765e-06}

        assert tuning_obect.optimal_partial_hyper_parameter_value_combination_found_among_all_budget_evaluations == {
            'x': 1.8673196127170794e-05}
        assert tuning_obect.info_of_optimal_partial_hyper_parameter_value_combination_found_among_all_budget_evaluations == {
            'id': (0, 0, 25), 'budget': 9.0, 'loss': 9.000018673196127, 'x': 1.8673196127170794e-05}

    def test_real_quick(self):
        '''As per instructed by v0.0.2, we make a quick tuning workflow. '''

        '''general inputs'''
        hyper_parameter_space_seed = 0
        numpy_seed = 0
        torch_seed = 0

        loss_function = nn.MSELoss()
        number_of_days_to_predict_ahead = 1
        number_of_test_days_in_a_day_level_DataFrame = 20
        number_of_rolling_forecast_days = 1
        data_windowing_option = 2
        input_length = 12
        lstm_is_many_to_many = False
        do_normalization = True
        batch_size = 20
        enable_data_caching = True
        generate_train_prediction = False

        total_data_size = 100 + 20 + input_length - 1 + number_of_days_to_predict_ahead
        DataFrame = pd.read_csv('jena_climate_2009_2016_autoforecastva.csv')
        date = pd.date_range(start='2020-01-01', end=pd.to_datetime('2020-01-01') + pd.Timedelta(len(DataFrame) - 1, unit='d'), name='date')
        DataFrame.index = date
        DataFrame = DataFrame.iloc[:total_data_size, 1:]
        DataFrame = DataFrame.iloc[:, [1,0,2,3,4,5,6]]
        DataFrame = DataFrame.rename(columns={'T (degC)': 'case'})
        DataFrame.insert(0, 'clinic', '111')

        '''inputs for searching hyperparameter value combination'''
        train_DataFrame = DataFrame.iloc[:-number_of_test_days_in_a_day_level_DataFrame, ]
        hyper_parameter_tuning_number_of_test_days_in_DataFrame = 20
        hyper_parameter_tuning_number_of_rolling_forecast_days = 1  # this should be greater than one in reality but we set it to one to speed up

        worker_specific_parameter_to_value_dictionary = {'DataFrame': train_DataFrame,
            'loss_function': loss_function,
            'number_of_days_to_predict_ahead': number_of_days_to_predict_ahead,
            'number_of_test_days_in_a_day_level_DataFrame': hyper_parameter_tuning_number_of_test_days_in_DataFrame,
            'number_of_rolling_forecast_days': hyper_parameter_tuning_number_of_rolling_forecast_days,
            'data_windowing_option': data_windowing_option,
            'input_length': input_length,
            'lstm_is_many_to_many': lstm_is_many_to_many,
            'do_normalization': do_normalization,
            'batch_size': batch_size,
            'enable_data_caching': enable_data_caching,
            'generate_train_prediction': generate_train_prediction
        }

        hyper_parameter_space = CS.ConfigurationSpace()
        learning_rate = CSH.UniformFloatHyperparameter('learning_rate', lower=0.00001, upper=0.1, log=True)
        number_of_hidden_dimensions = CSH.UniformIntegerHyperparameter('number_of_hidden_dimensions', lower=2, upper=10)
        dropout_rate = CSH.UniformFloatHyperparameter('dropout_rate', lower=0, upper=0.7)
        hyper_parameters = [learning_rate, number_of_hidden_dimensions, dropout_rate]
        hyper_parameter_space.add_hyperparameters(hyper_parameters)

        nameserver_run_id = 'example1'
        nameserver_address = '127.0.0.1'
        nameserver_port = 65301  # dont be 80, 8080, 443 or below 1024

        min_budget = 10
        max_budget = 50
        n_iterations = 3  # array([ 16.66666667,  50.        ])

        hyper_parameter_space.seed(hyper_parameter_space_seed)
        np.random.seed(numpy_seed)
        torch.manual_seed(0)

        '''do tuning (already did before, just skip it)'''
        tuning_object = HyperParameterTuningLoop(CustomizedWorker=LSTMWorker, worker_specific_parameter_to_value_dictionary=worker_specific_parameter_to_value_dictionary, hyper_parameter_space=hyper_parameter_space, nameserver_run_id=nameserver_run_id, nameserver_address=nameserver_address, nameserver_port=nameserver_port, min_budget=min_budget, max_budget=max_budget, n_iterations=n_iterations)
        self.assertListEqual( tuning_object.run_result_dataframe_sorted_by_loss_and_budget.loss.tolist(), [0.14093104004859924, 0.21238374710083008, 0.2765733301639557, 0.5349845290184021, 0.7288252115249634, 0.887844443321228, 0.9779404401779175, 3.7225799560546875, 4.737166404724121, 8.684465408325195])

        searched_hyper_parameter_value_combination = tuning_object.optimal_partial_hyper_parameter_value_combination_found_among_max_budget_evaluations
        searched_hyper_parameter_value_combination['number_of_training_epochs'] = max_budget

        '''start testing'''
        torch.manual_seed(torch_seed)
        searched_hyperParameterValueCombinationEvaluator = HyperParameterValueCombinationEvaluator(DataFrame=train_DataFrame, loss_function=loss_function, hyper_parameter_value_combination=searched_hyper_parameter_value_combination, number_of_days_to_predict_ahead=number_of_days_to_predict_ahead, number_of_test_days_in_a_day_level_DataFrame=hyper_parameter_tuning_number_of_test_days_in_DataFrame, number_of_rolling_forecast_days=hyper_parameter_tuning_number_of_rolling_forecast_days, data_windowing_option=data_windowing_option, input_length=input_length, lstm_is_many_to_many=lstm_is_many_to_many, do_normalization=do_normalization, batch_size=batch_size, enable_data_caching=enable_data_caching)
        self.assertEqual(searched_hyperParameterValueCombinationEvaluator.list_of_rolling_forecast_daily_normalization_reverted_prediction_loss[0].item(), 0.11798779666423798)

    def test_effectivenenss_of_tuning(self):
        '''In v0.0.2, we test the correctness of tuning. We ensure that the tuning component is able to find better hyperparaemter value combinations over time; and the found hyperparemter value combiantion is indeed a good one. The test results are in develoment_v0.0.2.md
        '''


        '''general inputs'''
        hyper_parameter_space_seed = 0
        numpy_seed = 0
        torch_seed = 0

        loss_function = nn.MSELoss()
        number_of_days_to_predict_ahead = 1
        number_of_test_days_in_a_day_level_DataFrame = 1981
        number_of_rolling_forecast_days = 1
        data_windowing_option = 2
        input_length = 20
        lstm_is_many_to_many = False
        do_normalization = True
        batch_size = 60
        enable_data_caching = True

        total_data_size = 7000 + 2000 + input_length - 1 + number_of_days_to_predict_ahead
        DataFrame = pd.read_csv('jena_climate_2009_2016_autoforecastva.csv')
        date = pd.date_range(start='2020-01-01', end=pd.to_datetime('2020-01-01') + pd.Timedelta(len(DataFrame) - 1, unit='d'), name='date')
        DataFrame.index = date
        DataFrame = DataFrame.iloc[:total_data_size, 1:]
        DataFrame = DataFrame.iloc[:, [1,0,2,3,4,5,6]]
        DataFrame = DataFrame.rename(columns={'T (degC)': 'case'})
        DataFrame.insert(0, 'clinic', '111')

        '''inputs for searching hyperparameter value combination'''
        train_DataFrame = DataFrame.iloc[:-number_of_test_days_in_a_day_level_DataFrame, ]
        hyper_parameter_tuning_number_of_test_days_in_DataFrame = 1500
        hyper_parameter_tuning_number_of_rolling_forecast_days = 1  # this should be greater than one in reality but we set it to one to speed up

        worker_specific_parameter_to_value_dictionary = {'DataFrame': train_DataFrame,
            'loss_function': loss_function,
            'number_of_days_to_predict_ahead': number_of_days_to_predict_ahead,
            'number_of_test_days_in_a_day_level_DataFrame': hyper_parameter_tuning_number_of_test_days_in_DataFrame,
            'number_of_rolling_forecast_days': hyper_parameter_tuning_number_of_rolling_forecast_days,
            'data_windowing_option': data_windowing_option,
            'input_length': input_length,
            'lstm_is_many_to_many': lstm_is_many_to_many,
            'do_normalization': do_normalization,
            'batch_size': batch_size,
            'enable_data_caching': enable_data_caching
        }

        hyper_parameter_space = CS.ConfigurationSpace()
        learning_rate = CSH.UniformFloatHyperparameter('learning_rate', lower=0.00001, upper=0.1, log=True)
        number_of_hidden_dimensions = CSH.UniformIntegerHyperparameter('number_of_hidden_dimensions', lower=30, upper=200)
        dropout_rate = CSH.UniformFloatHyperparameter('dropout_rate', lower=0, upper=0.7)
        hyper_parameters = [learning_rate, number_of_hidden_dimensions, dropout_rate]
        hyper_parameter_space.add_hyperparameters(hyper_parameters)

        nameserver_run_id = 'example1'
        nameserver_address = '127.0.0.1'
        nameserver_port = 65300  # dont be 80, 8080, 443 or below 1024

        min_budget = 10
        max_budget = 150
        n_iterations = 3  # array([ 16.66666667,  50.        , 150.        ])

        hyper_parameter_space.seed(hyper_parameter_space_seed)
        np.random.seed(numpy_seed)
        torch.manual_seed(0)

        '''do tuning (already did before, just skip it)'''
        # tuning_object = HyperParameterTuningLoop(CustomizedWorker=LSTMWorker, worker_specific_parameter_to_value_dictionary=worker_specific_parameter_to_value_dictionary, hyper_parameter_space=hyper_parameter_space, nameserver_run_id=nameserver_run_id, nameserver_address=nameserver_address, nameserver_port=nameserver_port, min_budget=min_budget, max_budget=max_budget, n_iterations=n_iterations)
        # VariableSaverAndLoader(list_of_variables_to_save=tuning_object, save=True, file_name='data/tuning_object_completed')
        # tuning_object = VariableSaverAndLoader(load=True, file_name='data/tuning_object_completed').list_of_loaded_variables

        '''candidate hyperparemter value combinations '''
        best_combination = {'dropout_rate': 0.559410995, 'learning_rate': 0.000701322, 'number_of_hidden_dimensions': 163, 'number_of_training_epochs': 150}
        moderate_combination = {'dropout_rate': 0.674048681, 'learning_rate': 0.000273317, 'number_of_hidden_dimensions': 159, 'number_of_training_epochs': 150}
        worst_combination = {'dropout_rate': 0.558870783, 'learning_rate': 1.43E-05, 'number_of_hidden_dimensions': 150, 'number_of_training_epochs': 150}
        existing_combination = {'dropout_rate': 0.0, 'learning_rate': 0.001, 'number_of_hidden_dimensions': 30, 'number_of_training_epochs': 30}

        '''start testing'''
        for searched_hyper_parameter_value_combination, loss in zip([best_combination, moderate_combination, worst_combination, existing_combination], [0.04174647107720375, 0.04702974110841751, 0.10888031125068665, 0.06216816231608391]):
            torch.manual_seed(torch_seed)
            searched_hyperParameterValueCombinationEvaluator = HyperParameterValueCombinationEvaluator(DataFrame=train_DataFrame, loss_function=loss_function, hyper_parameter_value_combination=searched_hyper_parameter_value_combination, number_of_days_to_predict_ahead=number_of_days_to_predict_ahead, number_of_test_days_in_a_day_level_DataFrame=hyper_parameter_tuning_number_of_test_days_in_DataFrame, number_of_rolling_forecast_days=hyper_parameter_tuning_number_of_rolling_forecast_days, data_windowing_option=data_windowing_option, input_length=input_length, lstm_is_many_to_many=lstm_is_many_to_many, do_normalization=do_normalization, batch_size=batch_size, enable_data_caching=enable_data_caching)
            self.assertEqual(searched_hyperParameterValueCombinationEvaluator.list_of_rolling_forecast_daily_normalization_reverted_prediction_loss[0].item(), loss)