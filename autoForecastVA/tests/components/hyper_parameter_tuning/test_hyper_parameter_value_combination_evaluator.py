import unittest
import pandas as pd
import numpy as np
import time
import os

from keras.models import Sequential
from keras.layers import Dense
from keras.layers import LSTM

from autoForecastVA.src.components.data_preprocessing_global.data_clustering_and_normalization import DataCluseteringAndNormalization
from autoForecastVA.src.components.data_preprocessing_global.data_delimiting_by_day_and_medical_center import DataDelimitingByDayAndMedicalCenter
from autoForecastVA.src.components.hyper_parameter_tuning.hyper_parameter_tuning_rolling_forecast_data_preparation import HyperParameterTuningRollingForecastDataPreparation
import torch
from torch import nn
import seaborn as sns
import matplotlib.pyplot as plt

from autoForecastVA.src.components.hyper_parameter_tuning.hyper_parameter_value_combination_evaluator import \
    HyperParameterValueCombinationEvaluator
from autoForecastVA.src.components.models.general_models.lstm_model_operator import LSTMModelOperator


class TestHyperParameterValueCombinationEvaluator(unittest.TestCase):
    '''
    given a dataset and hyper-paraemter value combination
    prepare rolling forecast dataset
    for each day in rolling forecast dataset
        train a model using the hyper-parameter value combination and the day's rolling forecast train dataset
        get prediction using day's rolling forecast test dataset
    calculates error on all days
    return the error
    '''

    def test_experiment_effect_of_features_on_model_performance(self):
        '''Experimenting how feature types affect model performance. By default no normalization is done.

        1467.04931640625: Adding a small feature that describes the pattern of the target.
        1522.12744140625: Adding a small feature that describes the pattern of the target and do normalization.

        1630.9935302734375: Adding a large feature that describes the pattern of the target.
        1522.12744140625:Adding a large feature that describes the pattern of the target and do normalization.

        4999.12939453125: Adding a very large feature that describes the pattern of the target results in a moderately good prediction curve..
        1522.12744140625: Adding a very large feature that describes the pattern of the target and do normalization.

        3786:Adding small random integers does not have much effect.
        2081.92578125:Adding small random integers and do normalization makes it good.
        7737: Adding large random integers (100-1000) has moderate effect. The predicted curve is still right but just not fitting well.
        2081.92578125: Adding large random integers (100-1000) and do normalization has perfect fit.
        49133: Adding very large random integers (1000 to 10000) has a huge effect. The prediction line is flat now..
        2081.92578125: Adding very large random integers (1000 to 10000) and do normalization has a huge effect. The prediction line is flat now..
        '''

        torch_seed = 0
        numpy_seed = 0

        hyper_parameter_value_combination = {'dropout_rate': 0.0, 'learning_rate': 0.0005,
                                             'number_of_hidden_dimensions': 100, 'number_of_training_epochs': 500}

        number_of_rolling_forecast_days = 1
        number_of_test_days_in_a_day_level_DataFrame = 12

        input_length = 12
        data_windowing_option = 2
        lstm_is_many_to_many = True
        # do_normalization = True
        do_normalization = False

        number_of_days_to_predict_ahead = 1

        loss_function = nn.MSELoss()
        torch.manual_seed(torch_seed)
        np.random.seed(numpy_seed)

        dataset = sns.load_dataset("flights")
        date = pd.date_range(start='2020-01-01',
                             end=pd.to_datetime('2020-01-01') + pd.Timedelta(len(dataset) - 1, unit='d'), name='date')
        dataset.index = date
        dataset = dataset.iloc[:, [2]]
        dataset.columns = ['case']
        dataset['pattern'] = list(range(12)) * 12
        dataset['pattern'] = dataset['pattern'] * 100
        # dataset['pattern'] = dataset['case']
        # dataset['pattern'] = np.random.random_integers(1, 10, dataset.shape[0]) * 1000
        dataset.insert(0, 'clinic', '111')

        hyperParameterValueCombinationEvaluator = HyperParameterValueCombinationEvaluator(DataFrame=dataset, loss_function=loss_function, hyper_parameter_value_combination=hyper_parameter_value_combination, number_of_days_to_predict_ahead=number_of_days_to_predict_ahead, number_of_test_days_in_a_day_level_DataFrame=number_of_test_days_in_a_day_level_DataFrame,
                                                                                          number_of_rolling_forecast_days=number_of_rolling_forecast_days, data_windowing_option=data_windowing_option, input_length=input_length, lstm_is_many_to_many=lstm_is_many_to_many, do_normalization=do_normalization)

        print(hyperParameterValueCombinationEvaluator.list_of_rolling_forecast_daily_normalization_reverted_prediction_loss[0].item())

        dataset.index = list(range(144))
        train_time = dataset.index[input_length:-input_length]
        test_time = dataset.index[-input_length:]

        plt.plot(train_time, hyperParameterValueCombinationEvaluator.list_of_rolling_forecast_daily_list_of_normalization_reverted_train_predictions[0], 'r--', label='Training Predictions', )
        plt.plot(test_time, hyperParameterValueCombinationEvaluator.list_of_rolling_forecast_daily_list_of_normalization_reverted_test_predictions[0], 'g--', label='Test Predictions')

        plt.plot(dataset.index, dataset['case'].to_numpy(), label='Actual')
        plt.xticks(np.arange(0, 145, 12))
        plt.legend()
        # plt.show()

    # def test_toy_example_flights_one_rolling_forecast_day_two_columns_automated(self):
    #     '''Given a dataset, build a model (once) using the training days and predict on the test days.
    #     Only build model once and make predictions for multiple inputs.
    #     Note that no normalization is done, but you could always enable it in the code.
    #     '''
    #
    #     torch_seed = 0
    #
    #     hyper_parameter_value_combination = {'dropout_rate': 0.0, 'learning_rate': 0.0005,
    #                                          'number_of_hidden_dimensions': 100, 'number_of_training_epochs': 500}
    #
    #     number_of_rolling_forecast_days = 1
    #     number_of_test_days_in_a_day_level_DataFrame = 12
    #
    #
    #     input_length = 12
    #     data_windowing_option = 2
    #     lstm_is_many_to_many = True
    #     do_normalization = False
    #
    #     number_of_days_to_predict_ahead = 1
    #
    #     loss_function = nn.MSELoss()
    #     torch.manual_seed(torch_seed)
    #
    #     dataset = sns.load_dataset("flights")
    #     date = pd.date_range(start='2020-01-01',
    #                          end=pd.to_datetime('2020-01-01') + pd.Timedelta(len(dataset) - 1, unit='d'), name='date')
    #     dataset.index = date
    #     dataset = dataset.iloc[:, [2]]
    #     dataset.columns = ['case']
    #     # dataset['pattern'] = list(range(12)) * 12
    #     # dataset['pattern'] = dataset['pattern'] * 10
    #     # dataset['pattern'] = dataset['case']
    #     # dataset['pattern'] = 300
    #     dataset.insert(0, 'clinic', '111')
    #
    #     hyperParameterValueCombinationEvaluator = HyperParameterValueCombinationEvaluator(DataFrame=dataset, loss_function=loss_function,hyper_parameter_value_combination=hyper_parameter_value_combination, number_of_days_to_predict_ahead=number_of_days_to_predict_ahead, number_of_test_days_in_a_day_level_DataFrame=number_of_test_days_in_a_day_level_DataFrame,number_of_rolling_forecast_days=number_of_rolling_forecast_days, data_windowing_option=data_windowing_option, input_length=input_length, lstm_is_many_to_many=lstm_is_many_to_many, do_normalization=do_normalization)
    #
    #
    #     # self.assertEqual(torch_seed, 0)
    #     # self.assertListEqual([38785.103801727295, 38138.37902069092, 37139.54291534424, 37748.61145401001, 37543.51226043701], hyperParameterValueCombinationEvaluator.list_of_rolling_forecast_daily_list_of_epoch_loss[0][-5:])
    #     # self.assertListEqual([509.0613098144531, 474.76947021484375, 404.8186950683594, 368.56072998046875, 366.8146667480469], [e.item() for e in hyperParameterValueCombinationEvaluator.list_of_rolling_forecast_daily_list_of_normalized_train_predictions[0][-5:]])
    #     # self.assertListEqual([e.item() for e in hyperParameterValueCombinationEvaluator.list_of_rolling_forecast_daily_list_of_normalized_test_predictions[0][-5:]], [517.1510009765625, 506.9619140625, 419.9571228027344, 411.85198974609375, 371.87200927734375])
    #     # self.assertEqual(hyperParameterValueCombinationEvaluator.list_of_rolling_forecast_daily_normalization_reverted_prediction_loss[0].item(), 3202.5517578125)
    #
    #     dataset.index = list(range(144))
    #     train_time = dataset.index[input_length:-input_length]
    #     test_time = dataset.index[-input_length:]
    #
    #     plt.plot(train_time, torch.stack(hyperParameterValueCombinationEvaluator.list_of_rolling_forecast_daily_list_of_normalized_train_predictions[0]), 'r--', label='Training Predictions', )
    #     plt.plot(test_time, torch.stack(hyperParameterValueCombinationEvaluator.list_of_rolling_forecast_daily_list_of_normalized_test_predictions[0]), 'g--', label='Test Predictions')
    #     plt.plot(dataset.index, dataset['case'].to_numpy(), label='Actual')
    #     plt.xticks(np.arange(0, 145, 12))
    #     plt.legend()
    #     plt.show()

    def test_toy_example_flights_one_rolling_forecast_day_automated(self):
        '''Given a dataset, build a model (once) using the training days and predict on the test days.
        Only build model once and make predictions for multiple inputs.
        Note that no normalization is done, but you could always enable it in the code.
        '''

        torch_seed = 0

        hyper_parameter_value_combination = {'dropout_rate': 0.0, 'learning_rate': 0.0005,
                                             'number_of_hidden_dimensions': 100, 'number_of_training_epochs': 500}

        number_of_rolling_forecast_days = 1
        number_of_test_days_in_a_day_level_DataFrame = 12


        input_length = 12
        data_windowing_option = 2
        lstm_is_many_to_many = True
        do_normalization = False

        number_of_days_to_predict_ahead = 1

        loss_function = nn.MSELoss()
        torch.manual_seed(torch_seed)

        dataset = sns.load_dataset("flights")
        date = pd.date_range(start='2020-01-01',
                             end=pd.to_datetime('2020-01-01') + pd.Timedelta(len(dataset) - 1, unit='d'), name='date')
        dataset.index = date
        dataset = dataset.iloc[:, [2]]
        dataset.columns = ['case']
        dataset.insert(0, 'clinic', '111')

        hyperParameterValueCombinationEvaluator = HyperParameterValueCombinationEvaluator(DataFrame=dataset, loss_function=loss_function, hyper_parameter_value_combination=hyper_parameter_value_combination, number_of_days_to_predict_ahead=number_of_days_to_predict_ahead, number_of_test_days_in_a_day_level_DataFrame=number_of_test_days_in_a_day_level_DataFrame,
                                                                                          number_of_rolling_forecast_days=number_of_rolling_forecast_days, data_windowing_option=data_windowing_option, input_length=input_length, lstm_is_many_to_many=lstm_is_many_to_many, do_normalization=do_normalization)


        self.assertEqual(torch_seed, 0)

        self.assertListEqual([38785.103801727295, 38138.37902069092, 37139.54291534424, 37748.61145401001, 37543.51226043701], hyperParameterValueCombinationEvaluator.list_of_rolling_forecast_daily_list_of_epoch_loss[0][-5:])

        self.assertListEqual([509.0613098144531, 474.76947021484375, 404.8186950683594, 368.56072998046875, 366.8146667480469], [e.item() for e in hyperParameterValueCombinationEvaluator.list_of_rolling_forecast_daily_list_of_normalized_train_predictions[0][-5:]])

        self.assertListEqual([e.item() for e in hyperParameterValueCombinationEvaluator.list_of_rolling_forecast_daily_list_of_normalized_test_predictions[0][-5:]], [517.1510009765625, 506.9619140625, 419.9571228027344, 411.85198974609375, 371.87200927734375])

        self.assertEqual(hyperParameterValueCombinationEvaluator.list_of_rolling_forecast_daily_normalization_reverted_prediction_loss[0].item(), 3202.5517578125)

        dataset.index = list(range(144))
        train_time = dataset.index[input_length:-input_length]
        test_time = dataset.index[-input_length:]

        plt.plot(train_time, torch.stack(hyperParameterValueCombinationEvaluator.list_of_rolling_forecast_daily_list_of_normalized_train_predictions[0]), 'r--', label='Training Predictions', )
        plt.plot(test_time, torch.stack(hyperParameterValueCombinationEvaluator.list_of_rolling_forecast_daily_list_of_normalized_test_predictions[0]), 'g--', label='Test Predictions')
        plt.plot(dataset.index, dataset['case'].to_numpy(), label='Actual')
        plt.xticks(np.arange(0, 145, 12))
        plt.legend()
        # plt.show()

    def test_toy_example_flights_twelve_rolling_forecast_day_without_normalization_automated(self):
        '''At each rolling forecast day, build a model and predict one day.
        Note that no normalization is done, but you could always enable it in the code.
        Note that in plotting, the training predictions are from the first rolling forecasat day.
        '''

        torch_seed = 0

        hyper_parameter_value_combination = {'dropout_rate': 0.0, 'learning_rate': 0.0005, 'number_of_hidden_dimensions': 100, 'number_of_training_epochs': 500}
        number_of_rolling_forecast_days = 12
        number_of_test_days_in_a_day_level_DataFrame = 1

        input_length = 12
        data_windowing_option = 2
        lstm_is_many_to_many = True
        do_normalization = False

        number_of_days_to_predict_ahead = 1

        loss_function = nn.MSELoss()
        torch.manual_seed(torch_seed)

        DataFrame = sns.load_dataset("flights")
        date = pd.date_range(start='2020-01-01',
                             end=pd.to_datetime('2020-01-01') + pd.Timedelta(len(DataFrame) - 1, unit='d'), name='date')
        DataFrame.index = date
        DataFrame = DataFrame.iloc[:, [2]]
        DataFrame.columns = ['case']
        DataFrame.insert(0, 'clinic', '111')
        '''calling constructor'''
        hyperParameterValueCombinationEvaluator = HyperParameterValueCombinationEvaluator(DataFrame=DataFrame, loss_function=loss_function, hyper_parameter_value_combination=hyper_parameter_value_combination, number_of_days_to_predict_ahead=number_of_days_to_predict_ahead, number_of_test_days_in_a_day_level_DataFrame=number_of_test_days_in_a_day_level_DataFrame,
                                                                                          number_of_rolling_forecast_days=number_of_rolling_forecast_days, data_windowing_option=data_windowing_option, input_length=input_length, lstm_is_many_to_many=lstm_is_many_to_many, do_normalization=do_normalization)

        self.assertEqual(torch_seed, 0)

        self.assertEqual(144 - input_length - 1,  len(hyperParameterValueCombinationEvaluator.list_of_rolling_forecast_daily_list_of_normalization_reverted_train_predictions[-1]))  # the train predictions of the last rolling forecast day
        print([e.item() for e in hyperParameterValueCombinationEvaluator.list_of_rolling_forecast_daily_list_of_normalized_train_predictions[-1][-5:]])
        print([e.item() for e in hyperParameterValueCombinationEvaluator.rolling_forecast_all_days_normalization_reverted_test_predictions[-5:]])
        print(hyperParameterValueCombinationEvaluator.rolling_forecast_overall_normalization_reverted_test_prediction_loss.item())
        self.assertListEqual([e.item() for e in hyperParameterValueCombinationEvaluator.list_of_rolling_forecast_daily_list_of_normalized_train_predictions[-1][-5:]], [537.76123046875, 546.1508178710938, 539.6898803710938, 426.3988037109375, 440.61346435546875])
        self.assertListEqual([e.item() for e in hyperParameterValueCombinationEvaluator.rolling_forecast_all_days_normalization_reverted_test_predictions[-5:]], [554.9495239257812, 535.3717651367188, 429.00439453125, 418.74005126953125, 415.1000671386719])

        DataFrame.index = list(range(144))
        train_time = DataFrame.index[input_length:-input_length]
        test_time = DataFrame.index[-input_length:]

        print(hyperParameterValueCombinationEvaluator.rolling_forecast_overall_normalization_reverted_test_prediction_loss.item())
        self.assertEqual(2245.9619140625, hyperParameterValueCombinationEvaluator.rolling_forecast_overall_normalization_reverted_test_prediction_loss.item())

        print(len(train_time))

        figure = hyperParameterValueCombinationEvaluator.plot_train_and_test_targets_and_predictions(list_of_target_index=DataFrame.index, list_of_train_prediction_index=train_time, list_of_test_prediction_index=test_time, list_of_targets=DataFrame['case'].to_numpy(), list_of_train_predictions=hyperParameterValueCombinationEvaluator.list_of_rolling_forecast_daily_list_of_normalization_reverted_train_predictions[0], list_of_test_predictions=hyperParameterValueCombinationEvaluator.rolling_forecast_all_days_normalization_reverted_test_predictions)
        # figure.show()

    def test_toy_example_flights_twelve_rolling_forecast_day_with_normalization_automated(self):
        '''At each rolling forecast day, build a model and predict one day.
        Note that no normalization is done, but you could always enable it in the code.
        Note that in plotting, the training predictions are from the first rolling forecasat day.
        '''

        torch_seed = 0

        hyper_parameter_value_combination = {'dropout_rate': 0.0, 'learning_rate': 0.0005, 'number_of_hidden_dimensions': 100, 'number_of_training_epochs': 500}
        number_of_rolling_forecast_days = 12
        number_of_test_days_in_a_day_level_DataFrame = 1

        input_length = 12
        data_windowing_option = 2
        lstm_is_many_to_many = True
        do_normalization = True

        number_of_days_to_predict_ahead = 1

        loss_function = nn.MSELoss()
        torch.manual_seed(torch_seed)

        DataFrame = sns.load_dataset("flights")
        date = pd.date_range(start='2020-01-01',
                             end=pd.to_datetime('2020-01-01') + pd.Timedelta(len(DataFrame) - 1, unit='d'), name='date')
        DataFrame.index = date
        DataFrame = DataFrame.iloc[:, [2]]
        DataFrame.columns = ['case']
        DataFrame.insert(0, 'clinic', '111')
        '''calling constructor'''
        hyperParameterValueCombinationEvaluator = HyperParameterValueCombinationEvaluator(DataFrame=DataFrame, loss_function=loss_function, hyper_parameter_value_combination=hyper_parameter_value_combination, number_of_days_to_predict_ahead=number_of_days_to_predict_ahead, number_of_test_days_in_a_day_level_DataFrame=number_of_test_days_in_a_day_level_DataFrame,
                                                                                          number_of_rolling_forecast_days=number_of_rolling_forecast_days, data_windowing_option=data_windowing_option, input_length=input_length, lstm_is_many_to_many=lstm_is_many_to_many, do_normalization=do_normalization)
        ''' '''

        self.assertEqual(torch_seed, 0)
        self.assertEqual(144 - input_length - 1,  len(hyperParameterValueCombinationEvaluator.list_of_rolling_forecast_daily_list_of_normalization_reverted_train_predictions[-1]))  # the train predictions of the last rolling forecast day
        self.assertListEqual([e.item() for e in hyperParameterValueCombinationEvaluator.list_of_rolling_forecast_daily_list_of_normalization_reverted_train_predictions[-1][-5:]], [624.50390625, 608.9237060546875, 508.03948974609375, 461.39385986328125, 389.5232849121094])
        self.assertListEqual([e.item() for e in hyperParameterValueCombinationEvaluator.rolling_forecast_all_days_normalization_reverted_test_predictions[-5:]], [598.877197265625, 494.6021423339844, 460.34710693359375, 407.8658447265625, 403.5831604003906])

        DataFrame.index = list(range(144))
        train_time = DataFrame.index[input_length:-input_length]
        test_time = DataFrame.index[-input_length:]

        print(hyperParameterValueCombinationEvaluator.rolling_forecast_overall_normalization_reverted_test_prediction_loss.item())
        self.assertEqual(1737.9947509765625, hyperParameterValueCombinationEvaluator.rolling_forecast_overall_normalization_reverted_test_prediction_loss.item())

        print(len(train_time))
        print(len(hyperParameterValueCombinationEvaluator.list_of_rolling_forecast_daily_list_of_normalization_reverted_train_predictions[0]))

        figure = hyperParameterValueCombinationEvaluator.plot_train_and_test_targets_and_predictions(list_of_target_index=DataFrame.index, list_of_train_prediction_index=train_time, list_of_test_prediction_index=test_time, list_of_targets=DataFrame['case'].to_numpy(), list_of_train_predictions=hyperParameterValueCombinationEvaluator.list_of_rolling_forecast_daily_list_of_normalization_reverted_train_predictions[0], list_of_test_predictions=hyperParameterValueCombinationEvaluator.rolling_forecast_all_days_normalization_reverted_test_predictions)

        # figure.show()

    # def test_experiment_correctness_of_multivariate_modeling_online_implementation(self):
    #     import torch
    #
    #     import numpy as np
    #     import pandas as pd
    #     import seaborn as sns
    #     from pylab import rcParams
    #     import matplotlib.pyplot as plt
    #     from sklearn.preprocessing import MinMaxScaler
    #     from sklearn.preprocessing import StandardScaler
    #     from pandas.plotting import register_matplotlib_converters
    #     from torch import nn, optim
    #
    #
    #     sns.set(style='whitegrid', palette='muted', font_scale=1.2)
    #     HAPPY_COLORS_PALETTE = ["#01BEFE", "#FFDD00", "#FF7D00", "#FF006D", "#93D30C", "#8F00FF"]
    #     sns.set_palette(sns.color_palette(HAPPY_COLORS_PALETTE))
    #     register_matplotlib_converters()
    #
    #     RANDOM_SEED = 0
    #     np.random.seed(RANDOM_SEED)
    #     torch.manual_seed(RANDOM_SEED)
    #
    #     df = pd.read_csv('time_series_19-covid-Confirmed.csv')
    #     df = df.iloc[:, 4:]
    #
    #     daily_cases = df.sum(axis=0)
    #     daily_cases.index = pd.to_datetime(daily_cases.index)
    #     daily_cases.head()
    #     daily_cases['random'] = np.random.randint(low=100, high=1000, size=len(daily_cases))
    #     daily_cases['order'] = list(range(len(daily_cases)))
    #
    #
    #
    #     daily_cases = daily_cases.diff().fillna(daily_cases[0]).astype(np.int64)
    #     # there is only 41 days though
    #
    #     test_data_size = 14
    #
    #     train_data = daily_cases[:-test_data_size]
    #     test_data = daily_cases[-test_data_size:]
    #
    #     """We have to scale the data (values will be between 0 and 1) if we want to increase the training speed and performance of the model. We'll use the `MinMaxScaler` from scikit-learn:"""
    #
    #
    #     scaler = StandardScaler()
    #
    #     scaler = scaler.fit(np.expand_dims(train_data, axis=1))
    #     scaled_daily_cases = scaler.transform(np.expand_dims(daily_cases, axis=1))
    #     # train_data = scaler.transform(np.expand_dims(train_data, axis=1))
    #     # test_data = scaler.transform(np.expand_dims(test_data, axis=1))
    #
    #     """Currently, we have a big sequence of daily cases. We'll convert it into smaller ones:"""
    #
    #     def create_sequences(data, seq_length):
    #         xs = []
    #         ys = []
    #
    #         for i in range(len(data) - seq_length - 1):
    #             x = data[i:(i + seq_length), :]
    #             y = data[i + seq_length:i + seq_length+1, :1]
    #             xs.append(x)
    #             ys.append(y)
    #
    #         return np.array(xs), np.array(ys)
    #
    #     seq_length = 5
    #
    #     X, y = create_sequences(scaled_daily_cases, seq_length)
    #
    #     X_train = X[:-test_data_size]
    #     X_test = X[-test_data_size:]
    #     y_train = y[:-test_data_size]
    #     y_test = y[-test_data_size:]
    #
    #     X_train = torch.from_numpy(X_train).float()
    #     y_train = torch.from_numpy(y_train).float()
    #
    #     X_test = torch.from_numpy(X_test).float()
    #     y_test = torch.from_numpy(y_test).float()
    #
    #
    #
    #     """## Building a model We'll encapsulate the complexity of our model into a class that extends from `torch.nn.Module` """
    #
    #     class CoronaVirusPredictor(nn.Module):
    #
    #         def __init__(self, n_features, n_hidden, seq_len, n_layers=1, dropout_rate=0.1):
    #             super().__init__()
    #
    #             self.n_hidden = n_hidden
    #             self.seq_len = seq_len
    #             self.n_layers = n_layers
    #             self.dropout_rate = dropout_rate
    #
    #             self.lstm = nn.LSTM(input_size=n_features, hidden_size=n_hidden, num_layers=n_layers, batch_first=True)
    #             self.dropout = nn.Dropout(dropout_rate)
    #             self.linear = nn.Linear(in_features=n_hidden, out_features=1)
    #
    #         def reset_hidden_state(self, number_of_sequences_in_a_batch):
    #             self.hidden = (torch.zeros(self.n_layers, number_of_sequences_in_a_batch, self.n_hidden), torch.zeros(self.n_layers, number_of_sequences_in_a_batch, self.n_hidden))
    #
    #         def forward(self, sequences):
    #             out, self.hidden = self.lstm(sequences.view(len(sequences), self.seq_len, -1), self.hidden)
    #             out = self.dropout(out)
    #             y_pred = self.linear(out)
    #             return y_pred
    #
    #     """Our `CoronaVirusPredictor` contains 3 methods:
    #     - constructor - initialize all helper data and create the layers
    #     - `reset_hidden_state` - we'll use a stateless LSTM, so we need to reset the state after each example
    #     - `forward` - get the sequences, pass all of them through the LSTM layer, at once. We take the output of the last time step and pass it through our linear layer to get the prediction.
    #
    #     ## Training
    #
    #     Let's build a helper function for the training of our model (we'll reuse it later):
    #     """
    #
    #     loss_fn = torch.nn.MSELoss()
    #
    #     def train_model(model, X_train, y_train, X_test=None, Y_test=None):
    #
    #         optimiser = torch.optim.Adam(model.parameters(), lr=0.001)
    #         num_epochs = 60
    #
    #         train_hist = np.zeros(num_epochs)
    #         test_hist = np.zeros(num_epochs)
    #
    #         for t in range(num_epochs):
    #             epoch_loss = 0
    #             for index_in_a_batch in range(X_train.shape[0]):
    #                 one_X_train = X_train[index_in_a_batch]
    #                 one_y_train = y_train[index_in_a_batch]
    #
    #                 one_X_train = torch.unsqueeze(one_X_train, 0)
    #                 one_y_train = torch.unsqueeze(one_y_train, 0)
    #                 model.reset_hidden_state(len(one_X_train))
    #                 y_pred = model(one_X_train)
    #                 loss = loss_fn(y_pred[:, -1:, :].float(), one_y_train)
    #
    #
    #                 epoch_loss += loss.item()
    #                 optimiser.zero_grad()
    #                 loss.backward()
    #                 optimiser.step()
    #
    #             train_hist[t] = epoch_loss
    #             print(epoch_loss)
    #         return model.eval(), train_hist, test_hist
    #
    #     model = CoronaVirusPredictor(n_features=1, n_hidden=50, seq_len=seq_length, n_layers=1, dropout_rate=0.1)
    #     model, train_hist, test_hist = train_model(model, X_train, y_train, None, None)
    #
    #     """Let's have a look at the train and test loss:"""
    #
    #     plt.plot(train_hist, label="Training loss")
    #
    #     plt.legend()
    #     plt.show()
    #
    #     """Our model's performance doesn't improve after 15 epochs or so. Recall that we have very little data. Maybe we shouldn't trust our model that much?
    #
    #     ## Predicting daily cases
    #
    #     Our model can (due to the way we've trained it) predict only a single day in the future. We'll employ a simple strategy to overcome this limitation. Use predicted values as input for predicting the next days:
    #     """
    #
    #     model.eval()
    #     test_predictions = []
    #     test_targets = []
    #     with torch.no_grad():
    #         for index_in_a_batch in range(X_test.shape[0]):
    #             one_X_test = X_test[index_in_a_batch]
    #             one_y_test = y_test[index_in_a_batch]
    #
    #             one_X_test = torch.unsqueeze(one_X_test, 0)
    #             one_y_test = torch.unsqueeze(one_y_test, 0)
    #
    #
    #             model.reset_hidden_state(len(one_X_test))
    #             y_pred = model(one_X_test)
    #             test_predictions.append(y_pred[0, -1, 0].item())
    #             test_targets.append(one_y_test.item())
    #
    #     # invert transform
    #     dir(scaler)
    #     reverted_test_predictions = [prediction * np.sqrt(scaler.var_) + scaler.mean_ for prediction in test_predictions]
    #     reverted_test_targets = [target * np.sqrt(scaler.var_) + scaler.mean_ for target in test_targets]
    #
    #
    #     loss = loss_fn(torch.Tensor(test_targets), torch.Tensor(test_predictions))
    #     reverted_loss = loss_fn(torch.Tensor(reverted_test_targets), torch.Tensor(reverted_test_predictions))
    #     print(loss)
    #     print(reverted_loss)
    #
    #
    #     plt.plot(reverted_test_predictions, label='predict')
    #     plt.plot(reverted_test_targets, label='target')
    #     plt.legend()
    #     plt.show()


    def test_batched_experiment_correctness_of_multivariate_modeling_autoforecastva_implementation3(self):
        '''Takes 2mins to finish'''

        torch_seed = 0
        hyper_parameter_value_combination = {'dropout_rate': 0.0, 'learning_rate': 0.001, 'number_of_hidden_dimensions': 30, 'number_of_training_epochs': 30}
        number_of_rolling_forecast_days = 1
        number_of_test_days_in_a_day_level_DataFrame = 1981

        input_length = 20
        data_windowing_option = 2
        lstm_is_many_to_many = False
        do_normalization = True
        batch_size = 40

        number_of_days_to_predict_ahead = 1

        total_data_size = 7000 + 2000 + input_length - 1 + number_of_days_to_predict_ahead

        loss_function = nn.MSELoss()
        torch.manual_seed(torch_seed)

        DataFrame = pd.read_csv('jena_climate_2009_2016_autoforecastva.csv')
        date = pd.date_range(start='2020-01-01',
                             end=pd.to_datetime('2020-01-01') + pd.Timedelta(len(DataFrame) - 1, unit='d'), name='date')
        DataFrame.index = date
        DataFrame = DataFrame.iloc[:total_data_size, 1:]
        DataFrame = DataFrame.iloc[:, [1,0,2,3,4,5,6]]
        DataFrame = DataFrame.rename(columns={'T (degC)': 'case'})
        DataFrame.insert(0, 'clinic', '111')
        '''calling constructor'''
        hyperParameterValueCombinationEvaluator = HyperParameterValueCombinationEvaluator(DataFrame=DataFrame, loss_function=loss_function, hyper_parameter_value_combination=hyper_parameter_value_combination, number_of_days_to_predict_ahead=number_of_days_to_predict_ahead, number_of_test_days_in_a_day_level_DataFrame=number_of_test_days_in_a_day_level_DataFrame, number_of_rolling_forecast_days=number_of_rolling_forecast_days, data_windowing_option=data_windowing_option, input_length=input_length, lstm_is_many_to_many=lstm_is_many_to_many, do_normalization=do_normalization, batch_size=batch_size)

        # figure = hyperParameterValueCombinationEvaluator.plot_train_and_test_targets_and_predictions(list_of_target_index=None, list_of_train_prediction_index=None, list_of_test_prediction_index=None, list_of_targets=DataFrame['case'].to_numpy(), list_of_train_predictions=hyperParameterValueCombinationEvaluator.list_of_rolling_forecast_daily_list_of_normalization_reverted_train_predictions[0], list_of_test_predictions=hyperParameterValueCombinationEvaluator.list_of_rolling_forecast_daily_list_of_normalization_reverted_test_predictions[0])
        # figure.show()
        # plt.plot(hyperParameterValueCombinationEvaluator.list_of_rolling_forecast_daily_list_of_epoch_loss[0]) # plot loss over time
        # plt.show()

        self.assertListEqual(hyperParameterValueCombinationEvaluator.list_of_rolling_forecast_daily_list_of_epoch_loss, [[84.60901719116373, 23.99606774642598, 10.866942111344542, 5.964687572748517, 4.314549271177384, 2.5761504335678183, 1.767782842849556, 0.9945048204899649, 0.827709584897093, 0.7069574522611219, 0.6490920525793626, 0.6078369500173721, 0.5658560130650585, 0.5434062630447443, 0.5221300906123361, 0.5018170396178903, 0.49345701287529664, 0.48263581174978754, 0.47332242421180126, 0.47101448684043135, 0.46480341509959544, 0.4544146201405965, 0.44594680052250624, 0.43638517357976525, 0.4246170538935985, 0.4143010299776506, 0.4054903973810724, 0.39572144429257605, 0.3852760926638439, 0.3756160955745145]])
        self.assertListEqual([i.item() for i in hyperParameterValueCombinationEvaluator.list_of_rolling_forecast_daily_list_of_normalization_reverted_train_predictions[ 0][:5]], [-8.920312881469727, -8.96170425415039, -8.87333869934082, -8.952061653137207, -9.025345802307129])
        self.assertListEqual([i.item() for i in hyperParameterValueCombinationEvaluator.list_of_rolling_forecast_daily_list_of_normalized_test_predictions[0][:5]], [-0.5286132097244263, -0.617034912109375, -0.6339290142059326, -0.676062822341919, -0.729236364364624])  # This has a problem
        self.assertListEqual([i.item() for i in hyperParameterValueCombinationEvaluator.list_of_rolling_forecast_daily_list_of_normalization_reverted_test_predictions[0][:5]], [-5.183105945587158, -5.624383926391602, -5.708695888519287, -5.91896915435791, -6.184337615966797])  # Only this has a problem
        self.assertListEqual([i.item() for i in hyperParameterValueCombinationEvaluator.rolling_forecast_all_days_normalization_reverted_train_prediction_targets[:5]], [-8.940000534057617, -8.860000610351562, -8.99000072479248, -9.050000190734863, -9.230000495910645])
        self.assertListEqual([i.item() for i in hyperParameterValueCombinationEvaluator.rolling_forecast_all_days_normalization_reverted_test_prediction_targets[:5]], [-5.490000247955322, -5.640000343322754, -5.800000190734863, -6.090000152587891, -6.190000534057617])

        self.assertEqual(hyperParameterValueCombinationEvaluator.list_of_rolling_forecast_daily_normalization_reverted_prediction_loss[0].item(), 0.04111247509717941)

    def test_slightly_differenct_results_between_batch_size_of_one_and_many(self):
        '''Debug: Maybe submit an issue to pytorch github
        It is found that, during model inferencing, feeding the same data but with different batch size(batch_size=1, batch_size=3, batch_size=5, and batch_size=all) yields (slightly) different result (very small, the difference starts to occur as far as the 7th digit or so). The difference is so small that it could only be found using tensor.item() instead of tensor.numpy().
        tensor.item() gives many digits while tensor.numpy() gives very few.

        I checked every possibilities. I ensured that the model weights are the same, the input data are the same and the initialized hidden states are the same.

        In the end, the problem is pinpointed to the hidden states returned by the model. They are slightly differnet.

        This is the reason of the difference.

        I suspect it is because the forward method in pytorch is designed differently for batch_size=1 and batch_size>1.

        Last thought, the initalized hidden states, although all appear to be 0, might be different in the deeper level.

        '''
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

        DataFrame = pd.read_csv('jena_climate_2009_2016_autoforecastva.csv')
        date = pd.date_range(start='2020-01-01',
                             end=pd.to_datetime('2020-01-01') + pd.Timedelta(len(DataFrame) - 1, unit='d'), name='date')
        DataFrame.index = date
        DataFrame = DataFrame.iloc[:total_data_size, 1:]
        DataFrame = DataFrame.iloc[:, [1,2]]
        DataFrame = DataFrame.rename(columns={'T (degC)': 'case'})
        DataFrame.insert(0, 'clinic', '111')

        '''calling constructor'''
        torch.manual_seed(torch_seed)
        hyperParameterValueCombinationEvaluator1 = HyperParameterValueCombinationEvaluator(DataFrame=DataFrame, loss_function=loss_function, hyper_parameter_value_combination=hyper_parameter_value_combination, number_of_days_to_predict_ahead=number_of_days_to_predict_ahead, number_of_test_days_in_a_day_level_DataFrame=number_of_test_days_in_a_day_level_DataFrame, number_of_rolling_forecast_days=number_of_rolling_forecast_days, data_windowing_option=data_windowing_option, input_length=input_length, lstm_is_many_to_many=lstm_is_many_to_many, do_normalization=do_normalization, speed_up_inference=speed_up_inference)

        torch.manual_seed(torch_seed)
        hyperParameterValueCombinationEvaluator2 = HyperParameterValueCombinationEvaluator(DataFrame=DataFrame, loss_function=loss_function, hyper_parameter_value_combination=hyper_parameter_value_combination, number_of_days_to_predict_ahead=number_of_days_to_predict_ahead, number_of_test_days_in_a_day_level_DataFrame=number_of_test_days_in_a_day_level_DataFrame, number_of_rolling_forecast_days=number_of_rolling_forecast_days, data_windowing_option=data_windowing_option, input_length=input_length, lstm_is_many_to_many=lstm_is_many_to_many, do_normalization=do_normalization, speed_up_inference=(not speed_up_inference))

        print(hyperParameterValueCombinationEvaluator1.list_of_rolling_forecast_daily_normalization_reverted_prediction_loss[0].item())
        print(hyperParameterValueCombinationEvaluator2.list_of_rolling_forecast_daily_normalization_reverted_prediction_loss[0].item())

        self.assertEqual(hyperParameterValueCombinationEvaluator1.list_of_rolling_forecast_daily_normalization_reverted_prediction_loss[0].item(), hyperParameterValueCombinationEvaluator2.list_of_rolling_forecast_daily_normalization_reverted_prediction_loss[0].item())

        with self.assertRaises(AssertionError):
            self.assertListEqual([i.item() for i in hyperParameterValueCombinationEvaluator1.list_of_rolling_forecast_daily_list_of_normalized_test_predictions[0]], [i.item() for i in hyperParameterValueCombinationEvaluator2.list_of_rolling_forecast_daily_list_of_normalized_test_predictions[0]])
            self.assertListEqual([i.item() for i in hyperParameterValueCombinationEvaluator1.list_of_rolling_forecast_daily_list_of_normalized_train_predictions[0]], [i.item() for i in hyperParameterValueCombinationEvaluator2.list_of_rolling_forecast_daily_list_of_normalized_train_predictions[0]])

    def test_fast_experiment_correctness_of_multivariate_modeling_autoforecastva_implementation3(self):
        '''Speed up the autoforecastva implement by reducing the train data size, test data size, hidden size, and train epoch'''

        torch_seed = 0
        hyper_parameter_value_combination = {'dropout_rate': 0.0, 'learning_rate': 0.001, 'number_of_hidden_dimensions': 5, 'number_of_training_epochs': 5}
        # hyper_parameter_value_combination = {'dropout_rate': 0.0, 'learning_rate': 0.001, 'number_of_hidden_dimensions': 2, 'number_of_training_epochs': 1}  # TODO change back to the top one
        number_of_rolling_forecast_days = 1
        number_of_test_days_in_a_day_level_DataFrame = 81

        input_length = 20
        data_windowing_option = 2
        lstm_is_many_to_many = False
        do_normalization = True

        number_of_days_to_predict_ahead = 1

        total_data_size = 500 + 100 + input_length - 1 + number_of_days_to_predict_ahead

        loss_function = nn.MSELoss()
        torch.manual_seed(torch_seed)

        DataFrame = pd.read_csv('jena_climate_2009_2016_autoforecastva.csv')
        date = pd.date_range(start='2020-01-01',
                             end=pd.to_datetime('2020-01-01') + pd.Timedelta(len(DataFrame) - 1, unit='d'), name='date')
        DataFrame.index = date
        DataFrame = DataFrame.iloc[:total_data_size, 1:]
        DataFrame = DataFrame.iloc[:, [1,0,2,3,4,5,6]]
        DataFrame = DataFrame.rename(columns={'T (degC)': 'case'})
        DataFrame.insert(0, 'clinic', '111')
        '''calling constructor'''
        hyperParameterValueCombinationEvaluator = HyperParameterValueCombinationEvaluator(DataFrame=DataFrame, loss_function=loss_function, hyper_parameter_value_combination=hyper_parameter_value_combination, number_of_days_to_predict_ahead=number_of_days_to_predict_ahead, number_of_test_days_in_a_day_level_DataFrame=number_of_test_days_in_a_day_level_DataFrame, number_of_rolling_forecast_days=number_of_rolling_forecast_days, data_windowing_option=data_windowing_option, input_length=input_length, lstm_is_many_to_many=lstm_is_many_to_many, do_normalization=do_normalization)

        # figure = hyperParameterValueCombinationEvaluator.plot_train_and_test_targets_and_predictions(list_of_target_index=None, list_of_train_prediction_index=None, list_of_test_prediction_index=None, list_of_targets=DataFrame['case'].to_numpy(), list_of_train_predictions=hyperParameterValueCombinationEvaluator.list_of_rolling_forecast_daily_list_of_normalization_reverted_train_predictions[0], list_of_test_predictions=hyperParameterValueCombinationEvaluator.list_of_rolling_forecast_daily_list_of_normalization_reverted_test_predictions[0])
        # figure.show()
        # plt.plot(hyperParameterValueCombinationEvaluator.list_of_rolling_forecast_daily_list_of_epoch_loss[0]) # plot loss over time
        # plt.show()

        self.assertListEqual(hyperParameterValueCombinationEvaluator.list_of_rolling_forecast_daily_list_of_epoch_loss, [[232.02363655449128, 62.11933676785384, 33.45067061540374, 21.98866432566342, 16.22591648282605]])
        self.assertListEqual([i.item() for i in hyperParameterValueCombinationEvaluator.list_of_rolling_forecast_daily_list_of_normalization_reverted_train_predictions[ 0][:5]], [-8.679691314697266, -8.703832626342773, -8.685844421386719, -8.692802429199219, -8.704523086547852])
        self.assertListEqual([i.item() for i in hyperParameterValueCombinationEvaluator.list_of_rolling_forecast_daily_list_of_normalized_test_predictions[0][:5]], [1.3549470901489258, 1.350167989730835, 1.3418126106262207, 1.339160442352295, 1.347153902053833])  # This has a problem
        self.assertListEqual([i.item() for i in hyperParameterValueCombinationEvaluator.list_of_rolling_forecast_daily_list_of_normalization_reverted_test_predictions[0][:5]], [-1.0584499835968018, -1.070512056350708, -1.0916006565093994, -1.0982944965362549, -1.0781195163726807])  # Only this has a problem
        self.assertListEqual([i.item() for i in hyperParameterValueCombinationEvaluator.rolling_forecast_all_days_normalization_reverted_train_prediction_targets[:5]], [-8.940000534057617, -8.860000610351562, -8.989999771118164, -9.050000190734863, -9.230000495910645])
        self.assertListEqual([i.item() for i in hyperParameterValueCombinationEvaluator.rolling_forecast_all_days_normalization_reverted_test_prediction_targets[:5]], [-1.510000228881836, -1.510000228881836, -1.510000228881836, -1.5600001811981201, -1.570000171661377])

        self.assertEqual(hyperParameterValueCombinationEvaluator.list_of_rolling_forecast_daily_normalization_reverted_prediction_loss[0].item(), 0.2912644147872925)


    def test_fast_batched_experiment_correctness_of_multivariate_modeling_autoforecastva_implementation3(self):
        '''Speed up the autoforecastva implement by reducing the train data size, test data size, hidden size, and train epoch
        This is for testing batch implementation. At this point, we only allow batch_size=1. We will try having batch_size > 1
        In this test, we try to implement batch_size=1 in a more general way which will allow us to eventually have batch_size>1
        '''

        torch_seed = 0
        hyper_parameter_value_combination = {'dropout_rate': 0.0, 'learning_rate': 0.001, 'number_of_hidden_dimensions': 5, 'number_of_training_epochs': 5}
        # hyper_parameter_value_combination = {'dropout_rate': 0.0, 'learning_rate': 0.001, 'number_of_hidden_dimensions': 2, 'number_of_training_epochs': 1}  # TODO change back to the top one
        number_of_rolling_forecast_days = 1
        number_of_test_days_in_a_day_level_DataFrame = 81

        input_length = 20
        data_windowing_option = 2
        lstm_is_many_to_many = False
        do_normalization = True
        batch_size = 15

        number_of_days_to_predict_ahead = 1

        total_data_size = 500 + 100 + input_length - 1 + number_of_days_to_predict_ahead

        loss_function = nn.MSELoss()
        torch.manual_seed(torch_seed)

        DataFrame = pd.read_csv('jena_climate_2009_2016_autoforecastva.csv')
        date = pd.date_range(start='2020-01-01',
                             end=pd.to_datetime('2020-01-01') + pd.Timedelta(len(DataFrame) - 1, unit='d'), name='date')
        DataFrame.index = date
        DataFrame = DataFrame.iloc[:total_data_size, 1:]
        DataFrame = DataFrame.iloc[:, [1,0,2,3,4,5,6]]
        DataFrame = DataFrame.rename(columns={'T (degC)': 'case'})
        DataFrame.insert(0, 'clinic', '111')
        '''calling constructor'''
        hyperParameterValueCombinationEvaluator = HyperParameterValueCombinationEvaluator(DataFrame=DataFrame, loss_function=loss_function, hyper_parameter_value_combination=hyper_parameter_value_combination, number_of_days_to_predict_ahead=number_of_days_to_predict_ahead, number_of_test_days_in_a_day_level_DataFrame=number_of_test_days_in_a_day_level_DataFrame, number_of_rolling_forecast_days=number_of_rolling_forecast_days, data_windowing_option=data_windowing_option, input_length=input_length, lstm_is_many_to_many=lstm_is_many_to_many, do_normalization=do_normalization, batch_size=batch_size)

        # figure = hyperParameterValueCombinationEvaluator.plot_train_and_test_targets_and_predictions(list_of_target_index=None, list_of_train_prediction_index=None, list_of_test_prediction_index=None, list_of_targets=DataFrame['case'].to_numpy(), list_of_train_predictions=hyperParameterValueCombinationEvaluator.list_of_rolling_forecast_daily_list_of_normalization_reverted_train_predictions[0], list_of_test_predictions=hyperParameterValueCombinationEvaluator.list_of_rolling_forecast_daily_list_of_normalization_reverted_test_predictions[0])
        # figure.show()
        # plt.plot(hyperParameterValueCombinationEvaluator.list_of_rolling_forecast_daily_list_of_epoch_loss[0]) # plot loss over time
        # plt.show()

        self.assertListEqual(hyperParameterValueCombinationEvaluator.list_of_rolling_forecast_daily_list_of_epoch_loss, [[33.0745446048677, 27.662159752566367, 23.702094867709093, 20.172226555645466, 17.066790678538382]])
        self.assertListEqual([i.item() for i in hyperParameterValueCombinationEvaluator.list_of_rolling_forecast_daily_list_of_normalization_reverted_train_predictions[ 0][:5]], [-5.696706771850586, -5.698071002960205, -5.696487903594971, -5.708098888397217, -5.707444190979004])
        self.assertListEqual([i.item() for i in hyperParameterValueCombinationEvaluator.list_of_rolling_forecast_daily_list_of_normalization_reverted_test_predictions[0][:5]], [-3.5422496795654297, -3.5481514930725098, -3.5636627674102783, -3.577505350112915, -3.559157609939575])
        self.assertListEqual([i.item() for i in hyperParameterValueCombinationEvaluator.rolling_forecast_all_days_normalization_reverted_train_prediction_targets[:5]], [-8.940000534057617, -8.860000610351562, -8.989999771118164, -9.050000190734863, -9.230000495910645])
        self.assertListEqual([i.item() for i in hyperParameterValueCombinationEvaluator.rolling_forecast_all_days_normalization_reverted_test_prediction_targets[:5]], [-1.510000228881836, -1.510000228881836, -1.510000228881836, -1.5600001811981201, -1.570000171661377])

        self.assertEqual(hyperParameterValueCombinationEvaluator.list_of_rolling_forecast_daily_normalization_reverted_prediction_loss[0].item(), 3.7309389114379883)

    def test_skipping_generate_train_predictions(self):
        '''As per instructed by v0.0.2, we enable LSTMOperator to not generate train predictions. This would have an effect on evaluator so we have to do a test.
        '''

        torch_seed = 0
        hyper_parameter_value_combination = {'dropout_rate': 0.0, 'learning_rate': 0.001, 'number_of_hidden_dimensions': 5, 'number_of_training_epochs': 5}
        number_of_rolling_forecast_days = 1
        number_of_test_days_in_a_day_level_DataFrame = 81

        input_length = 20
        data_windowing_option = 2
        lstm_is_many_to_many = False
        do_normalization = True
        batch_size = 15
        generate_train_prediction = False

        number_of_days_to_predict_ahead = 1

        total_data_size = 500 + 100 + input_length - 1 + number_of_days_to_predict_ahead

        loss_function = nn.MSELoss()
        torch.manual_seed(torch_seed)

        DataFrame = pd.read_csv('jena_climate_2009_2016_autoforecastva.csv')
        date = pd.date_range(start='2020-01-01',
                             end=pd.to_datetime('2020-01-01') + pd.Timedelta(len(DataFrame) - 1, unit='d'), name='date')
        DataFrame.index = date
        DataFrame = DataFrame.iloc[:total_data_size, 1:]
        DataFrame = DataFrame.iloc[:, [1,0,2,3,4,5,6]]
        DataFrame = DataFrame.rename(columns={'T (degC)': 'case'})
        DataFrame.insert(0, 'clinic', '111')
        '''calling constructor'''
        hyperParameterValueCombinationEvaluator = HyperParameterValueCombinationEvaluator(DataFrame=DataFrame, loss_function=loss_function, hyper_parameter_value_combination=hyper_parameter_value_combination, number_of_days_to_predict_ahead=number_of_days_to_predict_ahead, number_of_test_days_in_a_day_level_DataFrame=number_of_test_days_in_a_day_level_DataFrame, number_of_rolling_forecast_days=number_of_rolling_forecast_days, data_windowing_option=data_windowing_option, input_length=input_length, lstm_is_many_to_many=lstm_is_many_to_many, generate_train_prediction=generate_train_prediction, do_normalization=do_normalization, batch_size=batch_size)

        self.assertListEqual(hyperParameterValueCombinationEvaluator.list_of_rolling_forecast_daily_list_of_epoch_loss, [[33.0745446048677, 27.662159752566367, 23.702094867709093, 20.172226555645466, 17.066790678538382]])

        self.assertListEqual([i.item() for i in hyperParameterValueCombinationEvaluator.list_of_rolling_forecast_daily_list_of_normalization_reverted_test_predictions[0][:5]], [-3.5422496795654297, -3.5481514930725098, -3.5636627674102783, -3.577505350112915, -3.559157609939575])
        self.assertListEqual([i.item() for i in hyperParameterValueCombinationEvaluator.rolling_forecast_all_days_normalization_reverted_train_prediction_targets[:5]], [-8.940000534057617, -8.860000610351562, -8.989999771118164, -9.050000190734863, -9.230000495910645])
        self.assertListEqual([i.item() for i in hyperParameterValueCombinationEvaluator.rolling_forecast_all_days_normalization_reverted_test_prediction_targets[:5]], [-1.510000228881836, -1.510000228881836, -1.510000228881836, -1.5600001811981201, -1.570000171661377])

        self.assertEqual(hyperParameterValueCombinationEvaluator.list_of_rolling_forecast_daily_normalization_reverted_prediction_loss[0].item(), 3.7309389114379883)

    def test_experiment_correctness_of_multivariate_modeling_online_implementation3(self):

        """ Adapted from https://keras.io/examples/timeseries/timeseries_weather_forecasting/#training.
        Note that this acutally does not forecast anything because the data has missingness and we simply skipped them (like what the original code did, which was pretty bad written.)
        The model fits the curve pretty well.
        The test loss is 8.5291.
        It takes 21 mintes to run, compared to the batched size = 40 version which is only 2 mintes
        """

        # set for reproducibility
        import os
        os.environ['PYTHONHASHSEED'] = '0'
        os.environ['CUDA_VISIBLE_DEVICES'] = ''

        import pandas as pd
        import matplotlib.pyplot as plt
        import tensorflow as tf
        from tensorflow import keras

        from zipfile import ZipFile
        import random
        import os
        torch.manual_seed(0)
        np.random.seed(0)
        tf.random.set_seed(0)
        random.seed(0)

        uri = "https://storage.googleapis.com/tensorflow/tf-keras-datasets/jena_climate_2009_2016.csv.zip"
        # zip_path = keras.utils.get_file(origin=uri, fname="jena_climate_2009_2016.csv.zip")
        # zip_file = ZipFile(zip_path)
        # zip_file.extractall()
        csv_path = "jena_climate_2009_2016.csv"

        df = pd.read_csv(csv_path)
        df = df.iloc[:10000, :]

        titles = [ "Pressure", "Temperature", "Temperature in Kelvin", "Temperature (dew point)", "Relative Humidity", "Saturation vapor pressure", "Vapor pressure", "Vapor pressure deficit", "Specific humidity", "Water vapor concentration", "Airtight", "Wind speed", "Maximum wind speed", "Wind direction in degrees", ]

        feature_keys = [ "p (mbar)", "T (degC)", "Tpot (K)", "Tdew (degC)", "rh (%)", "VPmax (mbar)", "VPact (mbar)", "VPdef (mbar)", "sh (g/kg)", "H2OC (mmol/mol)", "rho (g/m**3)", "wv (m/s)", "max. wv (m/s)", "wd (deg)", ]

        date_time_key = "Date Time"
        """ """

        test = False
        step = 1
        sequence_length = 20
        predict_ahead = 3
        number_of_input_variables = 7
        train_size = 7000
        val_size = 2000  # predict on the last 1981 days


        learning_rate = 0.001
        epochs = 30

        hidden_dim = 30
        loss_function = nn.MSELoss()

        def normalize(data, train_split):
            data_mean = data[:train_split].mean(axis=0)
            data_std = data[:train_split].std(axis=0)
            return (data - data_mean) / data_std, data_mean, data_std


        print("The selected parameters are:", ", ".join([titles[i] for i in [0, 1, 5, 7, 8, 10, 11]]))
        selected_features = [feature_keys[i] for i in [0, 1, 5, 7, 8, 10, 11]]
        features = df[selected_features]
        features.index = df[date_time_key]
        features.to_csv('jena_climate_2009_2016_autoforecastva.csv')

        features, mean, std = normalize(features.values, train_size)
        features = pd.DataFrame(features)
        if test:
            features['order'] = list(range(len(features)))
            features.iloc[:, 1] = list(range(len(features)))

        train_data = features.iloc[:train_size, :]
        val_data = features.iloc[train_size:train_size+val_size, :]

        x_train = train_data.values
        y_train = features.iloc[sequence_length-1+predict_ahead:x_train.shape[0] + sequence_length-1 + predict_ahead, :][[1]]  # second features as target


        dataset_train = keras.preprocessing.timeseries_dataset_from_array(
            x_train,
            y_train,
            sequence_length=sequence_length,  # length of an input window
            sampling_rate=step,  # grab data in every sampling_rate period
            batch_size=train_size - (sequence_length-1),  # number of sequences in a batch
        )

        x_val = val_data.values
        y_val = features.iloc[train_size+sequence_length-1+predict_ahead: train_size+x_val.shape[0] + sequence_length - 1 + predict_ahead, :][[1]]  # second features as target

        dataset_val = keras.preprocessing.timeseries_dataset_from_array(
            x_val,
            y_val,
            sequence_length=sequence_length,
            sampling_rate=step,
            batch_size=df.shape[0] - train_size - (sequence_length-1),
        )

        for batch in dataset_train.take(1):
            inputs, targets = batch

        dataset_train.cardinality().numpy()  # number of batches=1
        dataset_val.cardinality().numpy()  # number of batches=1



        """
        ## Training
        """

        inputs = keras.layers.Input(shape=(sequence_length, number_of_input_variables))   # (times step, vars)
        lstm_out = keras.layers.LSTM(hidden_dim)(inputs)
        outputs = keras.layers.Dense(1)(lstm_out)

        model = keras.Model(inputs=inputs, outputs=outputs)
        model.compile(optimizer=keras.optimizers.Adam(learning_rate=learning_rate), loss="mse")
        model.summary()

        history = model.fit(
            dataset_train,
            epochs=epochs,
            validation_data=dataset_val
        )

        def visualize_loss(history, title):
            loss = history.history["loss"]
            val_loss = history.history["val_loss"]
            epochs = range(len(loss))
            plt.figure()
            plt.plot(epochs, loss, "b", label="Training loss")
            plt.plot(epochs, val_loss, "r", label="Validation loss")
            plt.title(title)
            plt.xlabel("Epochs")
            plt.ylabel("Loss")
            plt.legend()
            # plt.show()

        visualize_loss(history, "Training and Validation Loss")

        """
        ## Prediction
        The trained model above is now able to make predictions for 5 sets of values from
        validation set.
        """


        hyperParameterValueCombinationEvaluator = HyperParameterValueCombinationEvaluator(DataFrame=None, hyper_parameter_value_combination=None, loss_function=None, lazy=True)

        x, y = list(dataset_val.as_numpy_iterator())[0]
        test_predict = model.predict(x) * std[1] + mean[1]
        test_actual = y * std[1] + mean[1]

        x, y = list(dataset_train.as_numpy_iterator())[0]
        train_predict = model.predict(x) * std[1] + mean[1]
        train_actual = y * std[1] + mean[1]

        figure = hyperParameterValueCombinationEvaluator.plot_train_and_test_targets_and_predictions(list_of_targets=train_actual.tolist() + test_actual.tolist(), list_of_train_predictions=train_predict.tolist(), list_of_test_predictions=test_predict.tolist(), list_of_target_index=None, list_of_train_prediction_index=None, list_of_test_prediction_index=None)
        # figure.show()

        print(loss_function(torch.Tensor(test_predict), torch.Tensor(test_actual)))
        self.assertEqual(loss_function(torch.Tensor(test_predict), torch.Tensor(test_actual)).item(), 4.602758407592773)

    def test_experiment_correctness_of_multivariate_modeling_autoforecastva_implementation3(self):

        torch_seed = 0
        hyper_parameter_value_combination = {'dropout_rate': 0.0, 'learning_rate': 0.001, 'number_of_hidden_dimensions': 30, 'number_of_training_epochs': 30}
        # hyper_parameter_value_combination = {'dropout_rate': 0.0, 'learning_rate': 0.001, 'number_of_hidden_dimensions': 2, 'number_of_training_epochs': 1}  # TODO change back to the top one
        number_of_rolling_forecast_days = 1
        number_of_test_days_in_a_day_level_DataFrame = 1981

        input_length = 20
        data_windowing_option = 2
        lstm_is_many_to_many = False
        do_normalization = True
        batch_size = 1

        number_of_days_to_predict_ahead = 1

        total_data_size = 7000 + 2000 + input_length - 1 + number_of_days_to_predict_ahead

        loss_function = nn.MSELoss()
        torch.manual_seed(torch_seed)

        DataFrame = pd.read_csv('jena_climate_2009_2016_autoforecastva.csv')
        date = pd.date_range(start='2020-01-01',
                             end=pd.to_datetime('2020-01-01') + pd.Timedelta(len(DataFrame) - 1, unit='d'), name='date')
        DataFrame.index = date
        DataFrame = DataFrame.iloc[:total_data_size, 1:]
        DataFrame = DataFrame.iloc[:, [1,0,2,3,4,5,6]]
        DataFrame = DataFrame.rename(columns={'T (degC)': 'case'})
        DataFrame.insert(0, 'clinic', '111')
        '''calling constructor'''
        hyperParameterValueCombinationEvaluator = HyperParameterValueCombinationEvaluator(DataFrame=DataFrame, loss_function=loss_function, hyper_parameter_value_combination=hyper_parameter_value_combination, number_of_days_to_predict_ahead=number_of_days_to_predict_ahead, number_of_test_days_in_a_day_level_DataFrame=number_of_test_days_in_a_day_level_DataFrame, number_of_rolling_forecast_days=number_of_rolling_forecast_days, data_windowing_option=data_windowing_option, input_length=input_length, lstm_is_many_to_many=lstm_is_many_to_many, do_normalization=do_normalization, batch_size=batch_size)

        figure = hyperParameterValueCombinationEvaluator.plot_train_and_test_targets_and_predictions(list_of_target_index=None, list_of_train_prediction_index=None, list_of_test_prediction_index=None, list_of_targets=DataFrame['case'].to_numpy(), list_of_train_predictions=hyperParameterValueCombinationEvaluator.list_of_rolling_forecast_daily_list_of_normalization_reverted_train_predictions[0], list_of_test_predictions=hyperParameterValueCombinationEvaluator.list_of_rolling_forecast_daily_list_of_normalization_reverted_test_predictions[0])
        # figure.show()
        plt.plot(hyperParameterValueCombinationEvaluator.list_of_rolling_forecast_daily_list_of_epoch_loss[0]) # plot loss over time
        # plt.show()

        self.assertListEqual(hyperParameterValueCombinationEvaluator.list_of_rolling_forecast_daily_list_of_epoch_loss, [[247.47331099450594, 120.76995666539355, 78.2982306602436, 56.82300253916654, 55.048328588189364, 51.97613933966196, 51.298761443488864, 40.44007972702431, 32.79933354909914, 52.74979753242613, 34.179471421172245, 27.38174540107075, 23.547681324260154, 22.25670889663379, 24.051409493500252, 22.18793676393476, 37.85184871292419, 23.060092531118652, 25.041692295672085, 21.98617622684198, 20.20854089978821, 18.154769015190663, 16.206019138317608, 14.890027242043463, 14.058288102040905, 13.276056204544037, 12.751988617040073, 12.508561623899807, 11.885524409815627, 11.310298480648822]])
        self.assertListEqual([i.item() for i in hyperParameterValueCombinationEvaluator.list_of_rolling_forecast_daily_list_of_normalization_reverted_train_predictions[ 0][:5]], [-8.38880443572998, -8.087879180908203, -8.204429626464844, -8.312226295471191, -8.32050609588623])
        self.assertListEqual([i.item() for i in hyperParameterValueCombinationEvaluator.list_of_rolling_forecast_daily_list_of_normalization_reverted_test_predictions[0][:5]], [-5.018037796020508, -5.584997177124023, -5.517940998077393, -5.702569007873535, -5.922770023345947])
        self.assertListEqual([i.item() for i in hyperParameterValueCombinationEvaluator.rolling_forecast_all_days_normalization_reverted_train_prediction_targets[:5]], [-8.940000534057617, -8.860000610351562, -8.99000072479248, -9.050000190734863, -9.230000495910645])
        self.assertListEqual([i.item() for i in hyperParameterValueCombinationEvaluator.rolling_forecast_all_days_normalization_reverted_test_prediction_targets[:5]], [-5.490000247955322, -5.640000343322754, -5.800000190734863, -6.090000152587891, -6.190000534057617])


        self.assertEqual(hyperParameterValueCombinationEvaluator.list_of_rolling_forecast_daily_normalization_reverted_prediction_loss[0].item(), 1.0478646755218506)


    def test_caching(self):
        ''' We need to ensure that data caching works correctly.

            We run two same instances of hyperparamete value combination evaluator, one with caching and one without caching.

            We should set epoch to a small number to maximize the comparison effect.

            The first run should be slow than the second one by a certain amount of time.

            After runnning, there should be a clean up procedure to clear the cache.
        '''


        torch_seed = 0
        hyper_parameter_value_combination = {'dropout_rate': 0.0, 'learning_rate': 0.001, 'number_of_hidden_dimensions': 5, 'number_of_training_epochs': 1}
        number_of_rolling_forecast_days = 1
        number_of_test_days_in_a_day_level_DataFrame = 81

        input_length = 20
        data_windowing_option = 2
        lstm_is_many_to_many = False
        do_normalization = True
        batch_size = 2

        number_of_days_to_predict_ahead = 1

        total_data_size = 3000 + 100 + input_length - 1 + number_of_days_to_predict_ahead

        loss_function = nn.MSELoss()
        torch.manual_seed(torch_seed)

        DataFrame = pd.read_csv('jena_climate_2009_2016_autoforecastva.csv')
        date = pd.date_range(start='2020-01-01', end=pd.to_datetime('2020-01-01') + pd.Timedelta(len(DataFrame) - 1, unit='d'), name='date')
        DataFrame.index = date
        DataFrame = DataFrame.iloc[:total_data_size, 1:]
        DataFrame = DataFrame.iloc[:, [1,0,2,3,4,5,6]]
        DataFrame = DataFrame.rename(columns={'T (degC)': 'case'})
        DataFrame.insert(0, 'clinic', '111')

        '''The first run is a fresh run and it caches data'''

        start_time1 = time.time()
        torch.manual_seed(torch_seed)
        hyperParameterValueCombinationEvaluator1 = HyperParameterValueCombinationEvaluator(DataFrame=DataFrame, loss_function=loss_function, hyper_parameter_value_combination=hyper_parameter_value_combination, number_of_days_to_predict_ahead=number_of_days_to_predict_ahead, number_of_test_days_in_a_day_level_DataFrame=number_of_test_days_in_a_day_level_DataFrame, number_of_rolling_forecast_days=number_of_rolling_forecast_days, data_windowing_option=data_windowing_option, input_length=input_length, lstm_is_many_to_many=lstm_is_many_to_many, do_normalization=do_normalization, batch_size=batch_size, enable_data_caching=True)
        end_time1 = time.time()
        duration_time1 = end_time1 - start_time1

        '''The second run reads the data cached by the first run and speed up'''
        torch.manual_seed(torch_seed)
        start_time2 = time.time()
        hyperParameterValueCombinationEvaluator2 = HyperParameterValueCombinationEvaluator(DataFrame=DataFrame, loss_function=loss_function, hyper_parameter_value_combination=hyper_parameter_value_combination, number_of_days_to_predict_ahead=number_of_days_to_predict_ahead, number_of_test_days_in_a_day_level_DataFrame=number_of_test_days_in_a_day_level_DataFrame, number_of_rolling_forecast_days=number_of_rolling_forecast_days, data_windowing_option=data_windowing_option, input_length=input_length, lstm_is_many_to_many=lstm_is_many_to_many, do_normalization=do_normalization, batch_size=batch_size, enable_data_caching=True)
        end_time2 = time.time()
        duration_time2 = end_time2 - start_time2

        print(duration_time1)
        print(duration_time2)

        '''clear the cache to prepare for the next run'''
        os.remove(hyperParameterValueCombinationEvaluator1.cached_file_name_for_rolling_forecast_preparation)
        for file in hyperParameterValueCombinationEvaluator1.list_of_cached_file_name_for_train_input_tensor_list:
            os.remove(file)

        for file in hyperParameterValueCombinationEvaluator1.list_of_cached_file_name_for_test_input_tensor_list:
            os.remove(file)

        #
        print(hyperParameterValueCombinationEvaluator1.cached_file_name_for_rolling_forecast_preparation)
        print(hyperParameterValueCombinationEvaluator1.list_of_cached_file_name_for_train_input_tensor_list)
        print(hyperParameterValueCombinationEvaluator1.list_of_cached_file_name_for_test_input_tensor_list)

        #
        # tests
        self.assertEqual(hyperParameterValueCombinationEvaluator1.cached_file_name_for_rolling_forecast_preparation, 'data/40dabb98643469e9c582f802cc2aee8a')
        self.assertListEqual(hyperParameterValueCombinationEvaluator1.list_of_cached_file_name_for_train_input_tensor_list, ['data/440b75bb21afe41f29fe4c02544f2686'])
        self.assertListEqual(hyperParameterValueCombinationEvaluator1.list_of_cached_file_name_for_test_input_tensor_list, ['data/cfd5572dc057f39403328c7f218d44a4'])

        self.assertEqual(hyperParameterValueCombinationEvaluator2.cached_file_name_for_rolling_forecast_preparation, None)
        self.assertListEqual(hyperParameterValueCombinationEvaluator2.list_of_cached_file_name_for_train_input_tensor_list, [])
        self.assertListEqual(hyperParameterValueCombinationEvaluator2.list_of_cached_file_name_for_test_input_tensor_list, [])


        self.assertEqual(hyperParameterValueCombinationEvaluator1.list_of_rolling_forecast_daily_normalization_reverted_prediction_loss[0].item(), hyperParameterValueCombinationEvaluator2.list_of_rolling_forecast_daily_normalization_reverted_prediction_loss[0].item())
        self.assertListEqual([i.item() for i in hyperParameterValueCombinationEvaluator1.list_of_rolling_forecast_daily_list_of_normalized_test_predictions[0]], [i.item() for i in hyperParameterValueCombinationEvaluator2.list_of_rolling_forecast_daily_list_of_normalized_test_predictions[0]])
        self.assertListEqual([i.item() for i in hyperParameterValueCombinationEvaluator1.list_of_rolling_forecast_daily_list_of_normalized_train_predictions[0]], [i.item() for i in hyperParameterValueCombinationEvaluator2.list_of_rolling_forecast_daily_list_of_normalized_train_predictions[0]])

        self.assertGreaterEqual(duration_time1, duration_time2 * 1.5)