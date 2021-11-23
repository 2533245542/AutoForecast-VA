import unittest
import torch
from torch import nn, optim
from autoForecastVA.src.components.data_preprocessing_global.data_windowing import DataWindowing


import numpy as np
import pandas as pd
from sklearn.preprocessing import OneHotEncoder, MinMaxScaler

import matplotlib as mpl
import matplotlib.pyplot as plt
import seaborn as sns

from autoForecastVA.src.components.data_preprocessing_global.data_delimiting_by_day_and_medical_center import DataDelimitingByDayAndMedicalCenter
from autoForecastVA.src.components.models.general_models.lstm_model import LSTM

class TestLstmManyToOne(unittest.TestCase):
    def test_toy_example(self):
        '''We use the passenger dataset and ensure that output is the same as the expected one output. We expect, given the specific random seed and hyper-parameter value combination, the result is expected

        We create the input-output DataFrame, window the data, init the model, do training, do testing and confirm the result.

        The optimizer is fixed to be Adam.
        A number of evaluation results {seed:MSE, 0: 2458, 1:2579, 2:2875, 3:3973, 4: 5453}
        '''

        torch_seed = 0

        hyper_parameter_value_combination = {'dropout_rate': 0.0, 'learning_rate': 0.0005, 'number_of_hidden_dimensions': 100, 'number_of_training_epochs': 500}

        number_of_input_variables = 1
        number_of_output_variables = 1
        number_of_lstm_layers_stacked = 1
        number_of_sequences_in_a_batch = 1

        number_of_testing_day_in_a_day_level_DataFrame = 12
        input_length = 12
        data_windowing_option = 2
        lstm_is_many_to_many = False

        number_of_days_for_testing = 1
        number_of_days_to_predict_ahead = 1

        loss_function = nn.MSELoss()
        torch.manual_seed(torch_seed)

        dataset = sns.load_dataset("flights")
        date = pd.date_range(start='2020-01-01', end=pd.to_datetime('2020-01-01') + pd.Timedelta(len(dataset) - 1, unit='d'), name='date')
        dataset.index = date
        dataset = dataset.iloc[:, [2]]
        dataset.columns = ['case']
        dataset['clinic'] = '111'

        '''delimiting, and windowizing'''
        dataDelimitingByDayAndMedicalCenter = DataDelimitingByDayAndMedicalCenter(dataset, number_of_days_for_testing, number_of_days_to_predict_ahead)  # one day, one medical center
        day_level_cluster_level_medical_center_level_across_medical_center_normalized_DataFrame_dictionary = {}
        first_day = None
        for day in dataDelimitingByDayAndMedicalCenter.day_level_medical_center_level_DataFrame_dictionary.keys():
            first_day = day
            cluster_level_medical_center_level_across_medical_center_normalized_DataFrame_dictionary = {}
            cluster_level_medical_center_level_across_medical_center_normalized_DataFrame_dictionary[0] = dataDelimitingByDayAndMedicalCenter.day_level_medical_center_level_DataFrame_dictionary[day]
            day_level_cluster_level_medical_center_level_across_medical_center_normalized_DataFrame_dictionary[day] = cluster_level_medical_center_level_across_medical_center_normalized_DataFrame_dictionary

        dataWindowing = DataWindowing(day_level_cluster_level_medical_center_level_across_medical_center_normalized_DataFrame_dictionary, option=data_windowing_option, number_of_days_to_predict_ahead=number_of_days_to_predict_ahead, input_length=input_length, number_of_testing_day_in_a_day_level_DataFrame=number_of_testing_day_in_a_day_level_DataFrame, lstm_is_many_to_many=lstm_is_many_to_many)
        train_input_output_DataFrame_list = dataWindowing.day_level_cluster_level_medical_center_level_train_input_output_DataFrame_list_dictionary[first_day][0]['111']
        test_input_output_DataFrame_list = dataWindowing.day_level_cluster_level_medical_center_level_test_input_output_DataFrame_list_dictionary[first_day][0]['111']
        len(train_input_output_DataFrame_list)  # 120
        len(test_input_output_DataFrame_list)  # 12

        # train_input_output_sequence_list[0]
        # (tensor([112., 118., 132., 129., 121., 135., 148., 148., 136., 119., 104., 118.]),
        #  tensor([118., 132., 129., 121., 135., 148., 148., 136., 119., 104., 118., 115.]))
        # test_input_output_sequence_list[0]
        # (tensor([360., 342., 406., 396., 420., 472., 548., 559., 463., 407., 362., 405.]),
        #  tensor([342., 406., 396., 420., 472., 548., 559., 463., 407., 362., 405., 417.]))

        train_input_output_DataFrame_list[0]  # 112., 118., 132., 129., 121., 135., 148., 148., 136., 119., 104., 118. -> 118., 132., 129., 121., 135., 148., 148., 136., 119., 104., 118., 115.
        test_input_output_DataFrame_list[0]  # 360., 342., 406., 396., 420., 472., 548., 559., 463., 407., 362., 405., -> 417

        ''' Turn data to list of (input sequence, output sequence) format. 
        Input sequence should be of size (batch, seq_length, number of variables )
        Output sequence should be of size (batch, seq_length, number of variables )
        '''

        def get_train_input_output_tensor_list(train_input_output_DataFrame_list, lstm_is_many_to_many):
            '''
            When lstm_is_many_to_many, input length is always the same as output length
            When lstm_is_many_to_one, output length is always one
            '''
            train_input_output_tensor_list = []  # of shape ((batch, seq_length, number of variables ), (batch, seq_length, number of variables ))
            for train_input_output_DataFrame in train_input_output_DataFrame_list:
                if lstm_is_many_to_many:
                    input_length = int(train_input_output_DataFrame.shape[0] / 2)
                else:
                    input_length = train_input_output_DataFrame.shape[0] - 1

                train_input_values = train_input_output_DataFrame[:input_length].drop(columns='clinic').values
                train_input_values_with_batch_dimension = np.expand_dims(train_input_values, 0)
                train_input_tensor = torch.FloatTensor(train_input_values_with_batch_dimension)

                train_output_values = train_input_output_DataFrame[input_length:].drop(columns='clinic').values
                train_output_values_with_batch_dimension = np.expand_dims(train_output_values, 0)
                train_output_tensor = torch.FloatTensor(train_output_values_with_batch_dimension)

                train_input_output_tensor_list.append((train_input_tensor, train_output_tensor))
            return train_input_output_tensor_list

        train_input_output_tensor_list = get_train_input_output_tensor_list(train_input_output_DataFrame_list=train_input_output_DataFrame_list, lstm_is_many_to_many=lstm_is_many_to_many)

        def get_test_input_output_tensor_list(test_input_output_DataFrame_list):
            '''Output length is always 1'''
            test_input_output_tensor_list = []  # of shape ((batch, seq_length, number of variables ), (batch, seq_length, number of variables ))
            for test_input_output_DataFrame in test_input_output_DataFrame_list:
                test_input_values = test_input_output_DataFrame[:-1].drop(columns='clinic').values
                test_input_values_with_batch_dimension = np.expand_dims(test_input_values, 0)
                test_input_tensor = torch.FloatTensor(test_input_values_with_batch_dimension)

                test_output_values = test_input_output_DataFrame[-1:].drop(columns='clinic').values
                test_output_values_with_batch_dimension = np.expand_dims(test_output_values, 0)
                test_output_tensor = torch.FloatTensor(test_output_values_with_batch_dimension)
                test_input_output_tensor_list.append((test_input_tensor, test_output_tensor))
            return test_input_output_tensor_list

        test_input_output_tensor_list = get_test_input_output_tensor_list(test_input_output_DataFrame_list=test_input_output_DataFrame_list)

        model = LSTM(number_of_input_variables=train_input_output_tensor_list[0][0].shape[2],
                     number_of_hidden_dimensions=hyper_parameter_value_combination['number_of_hidden_dimensions'],
                     number_of_output_variables=train_input_output_tensor_list[0][1].shape[2], dropout_rate=hyper_parameter_value_combination['dropout_rate'])

        optimizer = torch.optim.Adam(model.parameters(), lr=hyper_parameter_value_combination['learning_rate'])

        def train_model(model, number_of_training_epochs, train_input_output_tensor_list):
            model.train()
            list_of_epoch_loss = []  # records the accumulated loss in each epoch, refreshed after each epoch ends
            for i in range(number_of_training_epochs):
                epoch_loss = 0
                for train_input_tensor, train_output_tensor in train_input_output_tensor_list:
                    optimizer.zero_grad()
                    h_0, c_0 = model.init_hidden(number_of_sequences_in_a_batch=train_input_tensor.shape[0])  # batch size
                    predicted_train_output_tensor, _ = model(input_tensor=train_input_tensor, tuple_of_h_0_c_0=(h_0, c_0))  # expect input of shape (batch_size, seq_len, input_size)
                    loss_of_a_batch = loss_function(predicted_train_output_tensor[:, -1:, :], train_output_tensor)  # note that batch size is always 1
                    loss_of_a_batch.backward()
                    optimizer.step()
                    epoch_loss += loss_of_a_batch.item()

                list_of_epoch_loss.append(epoch_loss)
                print(f'epoch: {i:3} loss: {epoch_loss:10.8f}')
            print('train end')
            return list_of_epoch_loss

        def get_predictions(model, input_output_tensor_list):
            '''Predict from each input_tensor's last time step by the number of time steps to predict ahead'''
            model.eval()
            list_of_predictions = []
            for input_tensor, output_tensor in input_output_tensor_list:
                with torch.no_grad():
                    h_0, c_0 = model.init_hidden(number_of_sequences_in_a_batch=input_tensor.shape[0])
                    predicted_output_tensor, _ = model(input_tensor, (h_0, c_0))  # expect input of shape (batch_size, seq_len, input_size)
                    prediction_of_the_last_one_in_a_batch_and_the_time_step = predicted_output_tensor[0, -1, 0]
                    list_of_predictions.append(prediction_of_the_last_one_in_a_batch_and_the_time_step)
            print('get train prediction end')
            return list_of_predictions

        list_of_epoch_loss = train_model(model, hyper_parameter_value_combination['number_of_training_epochs'], train_input_output_tensor_list)

        list_of_train_predictions = get_predictions(model, train_input_output_tensor_list)
        list_of_test_predictions = get_predictions(model, test_input_output_tensor_list)

        list_of_test_targets = [test_input_output[1][0, 0, 0] for test_input_output in test_input_output_tensor_list]
        prediction_loss = loss_function(torch.stack(list_of_test_predictions), torch.stack(list_of_test_targets))
        print('prediction loss', prediction_loss)

        self.assertEqual(torch_seed, 0)
        self.assertListEqual(list_of_epoch_loss[-5:], [18936.800285232253, 20647.07475671172, 28244.027122580446, 20746.809501715936, 20984.263245684095])
        self.assertListEqual([e.item() for e in list_of_train_predictions[-5:]], [518.5480346679688, 474.615478515625, 415.9562683105469, 368.33380126953125, 396.0769958496094])
        self.assertListEqual([e.item() for e in list_of_test_predictions[-5:]], [523.2666625976562, 520.3639526367188, 468.6636657714844, 468.3677062988281, 396.3254699707031])
        self.assertEqual(prediction_loss.item(), 2458.738037109375)

        dataset.index = list(range(144))
        train_time = dataset.index[input_length:-input_length]
        test_time = dataset.index[-input_length:]


        plt.plot(train_time, list_of_train_predictions, 'r--', label='Training Predictions', )
        plt.plot(test_time, list_of_test_predictions, 'g--', label='Test Predictions')
        plt.plot(test_time, list_of_test_targets, 'b--', label='Test Targets')
        plt.plot(dataset.index, dataset['case'].to_numpy(), label='Actual')
        plt.xticks(np.arange(0, 145, 12))
        plt.legend()
        # plt.show()