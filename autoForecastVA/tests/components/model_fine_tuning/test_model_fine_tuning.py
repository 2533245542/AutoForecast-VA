import unittest
from torch import nn
import torch
from autoForecastVA.src.components.general_model_training.general_model_training import GeneralModelTraining
import pandas as pd
import numpy as np
import copy
# class TestModelFineTuning(unittest.TestCase):
from autoForecastVA.src.components.hyper_parameter_tuning.hyper_parameter_value_combination_evaluator import \
    HyperParameterValueCombinationEvaluator
from autoForecastVA.src.components.model_fine_tuning.model_fine_tuning import ModelFineTuning

class TestModelFineTuning(unittest.TestCase):
    def test_toy_example(self):
        number_of_days_in_dataset = 10
        date = pd.date_range(start='2020-01-01', end='2020-01-{}'.format(number_of_days_in_dataset), name='date')
        case = pd.Series(list(range(1, 1 + number_of_days_in_dataset)))
        call = case * 3 + 10
        dayofweek = date.dayofweek
        weekofyear = date.isocalendar().week
        precursor_dataset = pd.DataFrame(
            {'case': case.values.astype(np.float64), 'call': call.values.astype(np.float64),
             'dayofweek': dayofweek.astype(np.float64),
             'weekofyear': weekofyear.astype(np.float64)}, index=date)
        dataset_111 = precursor_dataset.copy()
        dataset_111.insert(0, 'clinic', '111')
        dataset_222 = precursor_dataset.copy()
        dataset_222.insert(0, 'clinic', '222')
        dataset_333 = precursor_dataset.copy()
        dataset_333.insert(0, 'clinic', '333')
        dataset = pd.concat([dataset_111, dataset_222, dataset_333], axis=0)

        hyper_parameter_value_combination = {'dropout_rate': 0.3841694527491273, 'learning_rate': 0.007257005721594277,
                                             'number_of_hidden_dimensions': 38, 'number_of_training_epochs': 95}
        train_DataFrame = dataset
        loss_function = nn.MSELoss()
        number_of_days_to_predict_ahead = 1  # one fake day
        data_windowing_option = 2
        input_length = 3
        lstm_is_many_to_many = False
        do_normalization = True

        ########################## finish preparation

        torch.manual_seed(0)
        generalModelTraining = GeneralModelTraining(hyper_parameter_value_combination=hyper_parameter_value_combination, train_DataFrame=train_DataFrame, loss_function=loss_function, data_windowing_option=data_windowing_option, input_length=input_length, lstm_is_many_to_many=lstm_is_many_to_many, number_of_days_to_predict_ahead=number_of_days_to_predict_ahead, do_normalization=do_normalization)

        # inputs
        hyper_parameter_value_combination = hyper_parameter_value_combination
        trainable_parameter_name_list = ['linear.weight', 'linear.bias']
        general_model = generalModelTraining.general_model
        dataframe = dataset_111  # should be unnormalized, and the last day in teh dataframe is the global test day
        number_of_days_to_predict_ahead = number_of_days_to_predict_ahead
        loss_function = loss_function
        data_windowing_option = data_windowing_option
        input_length = input_length
        lstm_is_many_to_many = lstm_is_many_to_many
        do_normalization = do_normalization
        number_of_test_days_in_a_day_level_DataFrame = 1  # the last day in teh dataframe is the global test day
        number_of_days_for_testing_in_a_hyper_parameter_tuning_run = 1  # only do one rollling forecast

        lazy = False
        verbose = False

        # outputs
        evaluator = None
        operator = None

        layer_name_to_layer_dictionary = None

        train_prediction_list = None
        train_target_list = None
        test_target_list = None
        test_prediction_list = None
        train_epoch_loss_list = None

        torch.manual_seed(0)
        modelFineTuning = ModelFineTuning(hyper_parameter_value_combination=hyper_parameter_value_combination, trainable_parameter_name_list=trainable_parameter_name_list, general_model=general_model, dataframe=dataframe, number_of_days_to_predict_ahead=number_of_days_to_predict_ahead, loss_function=loss_function, data_windowing_option=data_windowing_option, input_length=input_length, lstm_is_many_to_many=lstm_is_many_to_many, do_normalization=do_normalization, number_of_test_days_in_a_day_level_DataFrame=number_of_test_days_in_a_day_level_DataFrame, number_of_days_for_testing_in_a_hyper_parameter_tuning_run=number_of_days_for_testing_in_a_hyper_parameter_tuning_run, lazy=False, verbose=True)


        # tests
        self.assertEqual(dict(general_model.named_parameters())['lstm.bias_ih_l0'].requires_grad, True)
        self.assertEqual(dict(general_model.named_parameters())['lstm.weight_hh_l0'].requires_grad, True)
        self.assertEqual(dict(general_model.named_parameters())['linear.weight'].requires_grad, True)

        self.assertEqual(dict(modelFineTuning.partially_trainable_general_model.named_parameters())['lstm.bias_ih_l0'].requires_grad, False)
        self.assertEqual(dict(modelFineTuning.partially_trainable_general_model.named_parameters())['lstm.weight_hh_l0'].requires_grad, False)
        self.assertEqual(dict(modelFineTuning.partially_trainable_general_model.named_parameters())['linear.weight'].requires_grad, True)

        self.assertEqual(len(modelFineTuning.all_days_train_prediction_target_list), 6)
        self.assertEqual(len(modelFineTuning.all_days_train_prediction_list), 6)
        self.assertEqual(len(modelFineTuning.all_days_test_prediction_target_list), 1)  # must be one
        self.assertEqual(len(modelFineTuning.all_days_test_prediction_list), 1) # must be one
        self.assertEqual(len(modelFineTuning.train_epoch_loss_list), hyper_parameter_value_combination['number_of_training_epochs'])

        self.assertEqual(modelFineTuning.all_days_train_prediction_target_list[3].item(), 7.0)
        self.assertEqual(modelFineTuning.all_days_train_prediction_list[4].item(), 8.051521301269531)
        self.assertEqual(modelFineTuning.all_days_test_prediction_target_list[0], 10.0)
        self.assertEqual(modelFineTuning.all_days_test_prediction_list[0].item(), 9.530338287353516)
        self.assertEqual(modelFineTuning.train_epoch_loss_list[19], 0.026906061772933754)