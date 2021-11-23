import unittest
import pandas as pd
import numpy as np
import torch.nn as nn
import torch

from autoForecastVA.src.components.general_model_training.general_model_training import GeneralModelTraining

class TestGeneralModelTraining(unittest.TestCase):
    '''Train a model with the provided train numpy array and the hyper-paraemter value combination'''

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

        # inputs
        hyper_parameter_value_combination = {'dropout_rate': 0.3841694527491273, 'learning_rate': 0.007257005721594277,
                                             'number_of_hidden_dimensions': 38, 'number_of_training_epochs': 95}
        train_DataFrame = dataset
        loss_function = nn.MSELoss()
        number_of_days_to_predict_ahead = 1  # one fake day
        data_windowing_option = 2
        input_length = 3
        lstm_is_many_to_many = False
        do_normalization = False


        torch.manual_seed(0)
        generalModelTraining = GeneralModelTraining(hyper_parameter_value_combination=hyper_parameter_value_combination, train_DataFrame=train_DataFrame, loss_function=loss_function, data_windowing_option=data_windowing_option, input_length=input_length, lstm_is_many_to_many=lstm_is_many_to_many, number_of_days_to_predict_ahead=number_of_days_to_predict_ahead, do_normalization=do_normalization)

        self.assertTupleEqual(generalModelTraining.dataframe_with_one_extra_day.shape, (33, 5))
        self.assertEqual(generalModelTraining.hyperParameterValueCombinationEvaluator.rolling_forecast_overall_normalization_reverted_test_prediction_loss.item(), 0.32735589146614075)
        self.assertEqual(generalModelTraining.operator.list_of_train_predictions[3].item(), 6.7953081130981445)
        self.assertEqual(round(float(generalModelTraining.general_model.lstm.weight_ih_l0.data.numpy()[42, 2]), 9), 0.090487726)