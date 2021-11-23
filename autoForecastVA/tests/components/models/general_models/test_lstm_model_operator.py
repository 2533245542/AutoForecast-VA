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
from autoForecastVA.src.components.models.general_models.lstm_model_operator import LSTMModelOperator

class TestLstmModelOperator(unittest.TestCase):
    def test_batching_data_function(self):
        # construct input data
        input_first_batch = torch.tensor([1,2,3,4,5,6]).reshape(1,2,3)
        input_second_batch = torch.tensor([7,8,9,10,11,12]).reshape(1,2,3)
        input_third_batch = torch.tensor([13,14,15,16,17,18]).reshape(1,2,3)

        output_first_batch = torch.tensor([10]).reshape(1,1,1)
        output_second_batch = torch.tensor([20]).reshape(1,1,1)
        output_third_batch = torch.tensor([30]).reshape(1,1,1)

        '''
        for each, batch size = 1, time steps =2, number of vars = 3
        [1,2,3,
        4,5,6],
        
        [7,8,9,
        10,11,12],
        
        [13,14,15,
        16,17,18],
        
        for this, batch size = 3, time steps = 2, numbe of vars = 3
        '''
        # run the function
        input_output_tensor_list = [(input_first_batch, output_first_batch), (input_second_batch, output_second_batch), (input_third_batch, output_third_batch)]
        batch_size = 2

        lSTMModelOperator = LSTMModelOperator(train_input_output_tensor_list=input_output_tensor_list, lazy=True)
        batched_input_output_list = lSTMModelOperator.create_batched_input_output_tensor_list(input_output_tensor_list=input_output_tensor_list, batch_size=batch_size)

        # construct output data
        batched_first_input_tensor = torch.cat([input_first_batch, input_second_batch], dim=0)
        batched_second_input_tensor = input_third_batch

        batched_first_output_tensor = torch.cat([output_first_batch, output_second_batch], dim=0)
        batched_second_output_tensor = output_third_batch

        # tests
        torch.equal(batched_input_output_list[0][0], batched_first_input_tensor)  # (2,2,3)
        torch.equal(batched_input_output_list[0][1], batched_first_output_tensor)  # (2,1,1)

        torch.equal(batched_input_output_list[1][0], batched_second_input_tensor)  # (1,2,3)
        torch.equal(batched_input_output_list[1][1], batched_second_output_tensor)  # (1,1,1)

    def test_early_stopping_function(self):
        list_of_epoch_loss = [8, 8, 8, 8, 6, 4, 4, 3, 3, 3]
        # no improve times: 2, should stop, when threshold = 0
        # no improve times: 4, should stop, when threshold = 1

        # early_stopping_delta: how much a loss should be smaller than the smallest loss such that we consider the loss the new smalleset loss (an improvement)

        self.assertTupleEqual(LSTMModelOperator.should_early_stop(list_of_epoch_loss=list_of_epoch_loss, early_stopping_patience=2, early_stopping_delta=0), (True, 2))
        self.assertTupleEqual(LSTMModelOperator.should_early_stop(list_of_epoch_loss=list_of_epoch_loss, early_stopping_patience=2, early_stopping_delta=1), (True, 4))

    def test_toy_example_many_to_one(self):
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

        lstmModelOperator = LSTMModelOperator(hyper_parameter_value_combination=hyper_parameter_value_combination, train_input_output_tensor_list=train_input_output_tensor_list, test_input_output_tensor_list=test_input_output_tensor_list, many_to_many=lstm_is_many_to_many, loss_function=loss_function, generate_train_prediction=True)

        list_of_epoch_loss = lstmModelOperator.list_of_epoch_loss
        list_of_train_predictions = lstmModelOperator.list_of_train_predictions
        list_of_test_predictions = lstmModelOperator.list_of_test_predictions
        prediction_loss = lstmModelOperator.test_prediction_loss
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
        plt.plot(dataset.index, dataset['case'].to_numpy(), label='Actual')
        plt.xticks(np.arange(0, 145, 12))
        plt.legend()
        # plt.show()

    def test_toy_example_many_to_many(self):
        '''Similar to above but many to many '''

        torch_seed = 0

        hyper_parameter_value_combination = {'dropout_rate': 0.0, 'learning_rate': 0.0005,
                                             'number_of_hidden_dimensions': 100, 'number_of_training_epochs': 500}

        number_of_input_variables = 1
        number_of_output_variables = 1
        number_of_lstm_layers_stacked = 1
        number_of_sequences_in_a_batch = 1

        number_of_testing_day_in_a_day_level_DataFrame = 12
        input_length = 12
        data_windowing_option = 2
        lstm_is_many_to_many = True

        number_of_days_for_testing = 1
        number_of_days_to_predict_ahead = 1

        loss_function = nn.MSELoss()
        torch.manual_seed(torch_seed)

        dataset = sns.load_dataset("flights")
        date = pd.date_range(start='2020-01-01',
                             end=pd.to_datetime('2020-01-01') + pd.Timedelta(len(dataset) - 1, unit='d'), name='date')
        dataset.index = date
        dataset = dataset.iloc[:, [2]]
        dataset.columns = ['case']
        dataset['clinic'] = '111'

        '''delimiting, and windowizing'''
        dataDelimitingByDayAndMedicalCenter = DataDelimitingByDayAndMedicalCenter(dataset, number_of_days_for_testing,
                                                                                  number_of_days_to_predict_ahead)  # one day, one medical center
        day_level_cluster_level_medical_center_level_across_medical_center_normalized_DataFrame_dictionary = {}
        first_day = None
        for day in dataDelimitingByDayAndMedicalCenter.day_level_medical_center_level_DataFrame_dictionary.keys():
            first_day = day
            cluster_level_medical_center_level_across_medical_center_normalized_DataFrame_dictionary = {}
            cluster_level_medical_center_level_across_medical_center_normalized_DataFrame_dictionary[0] = \
            dataDelimitingByDayAndMedicalCenter.day_level_medical_center_level_DataFrame_dictionary[day]
            day_level_cluster_level_medical_center_level_across_medical_center_normalized_DataFrame_dictionary[
                day] = cluster_level_medical_center_level_across_medical_center_normalized_DataFrame_dictionary

        dataWindowing = DataWindowing(
            day_level_cluster_level_medical_center_level_across_medical_center_normalized_DataFrame_dictionary,
            option=data_windowing_option, number_of_days_to_predict_ahead=number_of_days_to_predict_ahead,
            input_length=input_length,
            number_of_testing_day_in_a_day_level_DataFrame=number_of_testing_day_in_a_day_level_DataFrame,
            lstm_is_many_to_many=lstm_is_many_to_many)
        train_input_output_DataFrame_list = \
        dataWindowing.day_level_cluster_level_medical_center_level_train_input_output_DataFrame_list_dictionary[
            first_day][0]['111']
        test_input_output_DataFrame_list = \
        dataWindowing.day_level_cluster_level_medical_center_level_test_input_output_DataFrame_list_dictionary[
            first_day][0]['111']
        len(train_input_output_DataFrame_list)  # 120
        len(test_input_output_DataFrame_list)  # 12

        train_input_output_DataFrame_list[
            0]  # 112., 118., 132., 129., 121., 135., 148., 148., 136., 119., 104., 118. -> 118., 132., 129., 121., 135., 148., 148., 136., 119., 104., 118., 115.
        test_input_output_DataFrame_list[
            0]  # 360., 342., 406., 396., 420., 472., 548., 559., 463., 407., 362., 405., -> 417

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

        train_input_output_tensor_list = get_train_input_output_tensor_list(
            train_input_output_DataFrame_list=train_input_output_DataFrame_list,
            lstm_is_many_to_many=lstm_is_many_to_many)

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

        test_input_output_tensor_list = get_test_input_output_tensor_list(
            test_input_output_DataFrame_list=test_input_output_DataFrame_list)

        '''test start'''
        lstmModelOperator = LSTMModelOperator(hyper_parameter_value_combination=hyper_parameter_value_combination,
                                              train_input_output_tensor_list=train_input_output_tensor_list,
                                              test_input_output_tensor_list=test_input_output_tensor_list,
                                              many_to_many=lstm_is_many_to_many, loss_function=loss_function,
                                              generate_train_prediction=True)

        list_of_epoch_loss = lstmModelOperator.list_of_epoch_loss
        list_of_train_predictions = lstmModelOperator.list_of_train_predictions
        list_of_test_predictions = lstmModelOperator.list_of_test_predictions
        prediction_loss = lstmModelOperator.test_prediction_loss
        print('prediction loss', prediction_loss)

        self.assertEqual(torch_seed, 0)
        self.assertListEqual(list_of_epoch_loss[-5:], [38785.103801727295, 38138.37902069092, 37139.54291534424, 37748.61145401001, 37543.51226043701])
        self.assertListEqual([e.item() for e in list_of_train_predictions[-5:]], [509.0613098144531, 474.76947021484375, 404.8186950683594, 368.56072998046875, 366.8146667480469])
        self.assertListEqual([e.item() for e in list_of_test_predictions[-5:]], [517.1510009765625, 506.9619140625, 419.9571228027344, 411.85198974609375, 371.87200927734375])
        self.assertEqual(prediction_loss.item(), 3202.5517578125)


        dataset.index = list(range(144))
        train_time = dataset.index[input_length:-input_length]
        test_time = dataset.index[-input_length:]

        plt.plot(train_time, list_of_train_predictions, 'r--', label='Training Predictions', )
        plt.plot(test_time, list_of_test_predictions, 'g--', label='Test Predictions')
        plt.plot(dataset.index, dataset['case'].to_numpy(), label='Actual')
        plt.xticks(np.arange(0, 145, 12))
        plt.legend()
        # plt.show()

    def test_toy_example_many_to_one_test_only_one_day(self):
        '''Similar to the test above but only testing one test day (the first day in the test_input_output_dataframe, simulating rolling forecast). The predicted result should be the same.
        The prediction for the test input tensor should only result in one value and we assert this value to a constant '''

        torch_seed = 0

        hyper_parameter_value_combination = {'dropout_rate': 0.0, 'learning_rate': 0.0005,
                                             'number_of_hidden_dimensions': 100, 'number_of_training_epochs': 500}

        number_of_testing_day_in_a_day_level_DataFrame = 12
        input_length = 12
        data_windowing_option = 2
        lstm_is_many_to_many = False

        number_of_days_for_testing = 1
        number_of_days_to_predict_ahead = 1

        loss_function = nn.MSELoss()
        torch.manual_seed(torch_seed)

        dataset = sns.load_dataset("flights")
        date = pd.date_range(start='2020-01-01',
                             end=pd.to_datetime('2020-01-01') + pd.Timedelta(len(dataset) - 1, unit='d'), name='date')
        dataset.index = date
        dataset = dataset.iloc[:, [2]]
        dataset.columns = ['case']
        dataset['clinic'] = '111'

        '''delimiting, and windowizing'''
        dataDelimitingByDayAndMedicalCenter = DataDelimitingByDayAndMedicalCenter(dataset, number_of_days_for_testing,
                                                                                  number_of_days_to_predict_ahead)  # one day, one medical center
        day_level_cluster_level_medical_center_level_across_medical_center_normalized_DataFrame_dictionary = {}
        first_day = None
        for day in dataDelimitingByDayAndMedicalCenter.day_level_medical_center_level_DataFrame_dictionary.keys():
            first_day = day
            cluster_level_medical_center_level_across_medical_center_normalized_DataFrame_dictionary = {}
            cluster_level_medical_center_level_across_medical_center_normalized_DataFrame_dictionary[0] = \
            dataDelimitingByDayAndMedicalCenter.day_level_medical_center_level_DataFrame_dictionary[day]
            day_level_cluster_level_medical_center_level_across_medical_center_normalized_DataFrame_dictionary[
                day] = cluster_level_medical_center_level_across_medical_center_normalized_DataFrame_dictionary

        dataWindowing = DataWindowing(
            day_level_cluster_level_medical_center_level_across_medical_center_normalized_DataFrame_dictionary,
            option=data_windowing_option, number_of_days_to_predict_ahead=number_of_days_to_predict_ahead,
            input_length=input_length,
            number_of_testing_day_in_a_day_level_DataFrame=number_of_testing_day_in_a_day_level_DataFrame,
            lstm_is_many_to_many=lstm_is_many_to_many)
        train_input_output_DataFrame_list = \
        dataWindowing.day_level_cluster_level_medical_center_level_train_input_output_DataFrame_list_dictionary[
            first_day][0]['111']
        test_input_output_DataFrame_list = \
        dataWindowing.day_level_cluster_level_medical_center_level_test_input_output_DataFrame_list_dictionary[
            first_day][0]['111']
        len(train_input_output_DataFrame_list)  # 120
        len(test_input_output_DataFrame_list)  # 12

        train_input_output_DataFrame_list[
            0]  # 112., 118., 132., 129., 121., 135., 148., 148., 136., 119., 104., 118. -> 118., 132., 129., 121., 135., 148., 148., 136., 119., 104., 118., 115.
        test_input_output_DataFrame_list[
            0]  # 360., 342., 406., 396., 420., 472., 548., 559., 463., 407., 362., 405., -> 417

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

        train_input_output_tensor_list = get_train_input_output_tensor_list(
            train_input_output_DataFrame_list=train_input_output_DataFrame_list,
            lstm_is_many_to_many=lstm_is_many_to_many)

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

        test_input_output_tensor_list = get_test_input_output_tensor_list(
            test_input_output_DataFrame_list=test_input_output_DataFrame_list)

        lstmModelOperator = LSTMModelOperator(hyper_parameter_value_combination=hyper_parameter_value_combination,
                                              train_input_output_tensor_list=train_input_output_tensor_list,
                                              test_input_output_tensor_list=test_input_output_tensor_list,
                                              many_to_many=lstm_is_many_to_many, loss_function=loss_function,
                                              generate_train_prediction=True)

        list_of_epoch_loss = lstmModelOperator.list_of_epoch_loss
        list_of_train_predictions = lstmModelOperator.list_of_train_predictions
        list_of_test_predictions = lstmModelOperator.list_of_test_predictions
        prediction_loss = lstmModelOperator.test_prediction_loss
        print('prediction loss', prediction_loss)

        self.assertEqual(torch_seed, 0)
        self.assertListEqual(list_of_epoch_loss[-5:], [18936.800285232253, 20647.07475671172, 28244.027122580446, 20746.809501715936, 20984.263245684095])
        self.assertListEqual([e.item() for e in list_of_train_predictions[-5:]], [518.5480346679688, 474.615478515625, 415.9562683105469, 368.33380126953125, 396.0769958496094])
        self.assertListEqual([e.item() for e in list_of_test_predictions[-5:]], [523.2666625976562, 520.3639526367188, 468.6636657714844, 468.3677062988281, 396.3254699707031])
        self.assertEqual(prediction_loss.item(), 2458.738037109375)

        '''assert lstmModelOperator works even when making only one test prediction'''
        self.assertEqual(lstmModelOperator.get_the_last_time_step_of_the_predicted_output(test_input_output_tensor_list[:1])[0].item(), 415.6249084472656)


        dataset.index = list(range(144))
        train_time = dataset.index[input_length:-input_length]
        test_time = dataset.index[-input_length:]

        plt.plot(train_time, list_of_train_predictions, 'r--', label='Training Predictions', )
        plt.plot(test_time, list_of_test_predictions, 'g--', label='Test Predictions')
        plt.plot(dataset.index, dataset['case'].to_numpy(), label='Actual')
        plt.xticks(np.arange(0, 145, 12))
        plt.legend()
        # plt.show()