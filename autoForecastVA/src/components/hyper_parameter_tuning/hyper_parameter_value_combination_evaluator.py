'''
Given datasets, it makes a instance. When the instance is called with instance.train(hyper-parameter value combination), the instance returns a value.
'''
from os import path
from autoForecastVA.src.data_caching.compute_characterization_code import ComputeCharacterizationCodeAndMd5Code
from autoForecastVA.src.components.hyper_parameter_tuning.hyper_parameter_tuning_rolling_forecast_data_preparation import HyperParameterTuningRollingForecastDataPreparation
from autoForecastVA.src.components.models.general_models.lstm_model_operator import LSTMModelOperator
from autoForecastVA.src.data_caching.variable_load_and_save import VariableSaverAndLoader
import time

'''
given a dataset and hyper-parameter value combination
for each day
    train a model using the hyper-parameter value combination and dataset
    get prediction
calculates error on all days
return the error
'''
import torch
import numpy as np
import matplotlib.pyplot as plt
class HyperParameterValueCombinationEvaluator():
    def __init__(self, DataFrame, hyper_parameter_value_combination, loss_function, number_of_days_to_predict_ahead=1, number_of_test_days_in_a_day_level_DataFrame=1, number_of_rolling_forecast_days=1, data_windowing_option=2, input_length=2, lstm_is_many_to_many=False, generate_train_prediction=True, do_normalization=True, batch_size=1, number_of_output_variables=1, customized_model_layer_name_to_layer_dictionary=None, speed_up_inference=False, enable_data_caching=False, lazy=False):
        ''' DOES NOT SUPPORT WHEN BOTH number_of_test_days_in_a_day_level_DataFrame and number_of_rolling_forecast_days are > 1
        Please assume all outputs are in tensor format.

        The input DataFrame should be unnormalized, might contain multiple medical centers; for each medical center in the DataFrame, the days should be continous (no skipping days between two consective rows).

        The evaluator splits the input DataFrame to rolling forecast windows (by days); it then builds a model and evaluate the model for each rolling forecast window.

        We record the results generated for each rolling forecast day. See the self.variable fields for details.

        The evaluator gives a prediction loss for the whole input DataFrame.

        :param number_of_test_days_in_a_day_level_DataFrame: the number of predictions to create after building one model
        :param number_of_rolling_forecast_days: the number of times to build a new model
        :param do_normalization: when it is false, we nullify the normalization and normalization revertion procedures. So the normalized data and normalization reverted data are the same. This applies to all fields in this class.
        '''

        # inputs
        self.DataFrame = DataFrame
        self.hyper_parameter_value_combination = hyper_parameter_value_combination
        self.loss_function = loss_function
        self.number_of_days_to_predict_ahead = number_of_days_to_predict_ahead
        self.number_of_test_days_in_a_day_level_DataFrame = number_of_test_days_in_a_day_level_DataFrame
        self.number_of_rolling_forecast_days = number_of_rolling_forecast_days
        self.data_windowing_option = data_windowing_option
        self.input_length = input_length
        self.lstm_is_many_to_many = lstm_is_many_to_many
        self.generate_train_prediction = generate_train_prediction
        self.do_normalization = do_normalization
        self.batch_size = batch_size
        self.number_of_output_variables = number_of_output_variables
        self.customized_model_layer_name_to_layer_dictionary = customized_model_layer_name_to_layer_dictionary
        self.speed_up_inference = speed_up_inference
        self.enable_data_caching = enable_data_caching and (number_of_test_days_in_a_day_level_DataFrame > 1 or number_of_rolling_forecast_days > 1)  # only cache during tuning
        self.lazy = lazy

        # outputs
        self.dataframe_contains_more_than_one_clinic = None   # this function behaves differently when dataframe has 1. one clinic 2. more than one clinics

        self.list_of_rolling_forecast_daily_lstmModelOperators = []  # there is a lstmModelOperator for each rolling forecast day

        self.list_of_rolling_forecast_daily_list_of_epoch_loss = []  # there is a list of epoch loss for each rolling forecast day

        self.list_of_rolling_forecast_daily_list_of_normalized_train_prediction_targets = []  # difference between output and prediction targets: when many to many, output could contain multiple days (used for back propagation), but prediction target is only the final day
        self.list_of_rolling_forecast_daily_list_of_normalized_test_prediction_targets = []

        self.list_of_rolling_forecast_daily_list_of_normalization_reverted_train_prediction_targets = []   # each day has prediction target (a list of targets when more than one clinic, a list of targets when one rolling forecast day and many test days in a dataframe, a list of (one) target when one clinic and multiple rolling forecast days)
        self.list_of_rolling_forecast_daily_list_of_normalization_reverted_test_prediction_targets = []

        self.list_of_rolling_forecast_daily_list_of_normalized_train_predictions = []   # there is a list of train predictions for each rolling forecast day. We put these lists in a list and call the grand list as list of rolling forecast which means it emcompasses the whole rolling forecast.
        self.list_of_rolling_forecast_daily_list_of_normalized_test_predictions = []  # there is  a list of test predictions for each day; each day's list would have length > 1 if when number_of_test_days_in_a_day_level_DataFrame > 1

        self.list_of_rolling_forecast_daily_list_of_normalization_reverted_train_predictions = []  # the normalization reverted version of list_of_rolling_forecast_daily_list_of_normalized_train_predictions
        self.list_of_rolling_forecast_daily_list_of_normalization_reverted_test_predictions = []  # the normalization reverted version of list_of_rolling_forecast_daily_list_of_normalized_test_predictions

        self.list_of_rolling_forecast_daily_normalization_reverted_prediction_loss = []  # there is a prediction loss for each rolling forecast day (calculated using list_of_rolling_forecast_daily_list_of_normalization_reverted_test_predictions)

        self.rolling_forecast_all_days_normalization_reverted_train_prediction_targets = []  # simply flatten the list of lists
        self.rolling_forecast_all_days_normalization_reverted_test_prediction_targets = []

        self.rolling_forecast_all_days_normalization_reverted_train_predictions = []  #
        self.rolling_forecast_all_days_normalization_reverted_test_predictions = []  # simply flatten the nested list  list_of_rolling_forecast_daily_list_of_normalization_reverted_test_predictions (for one level) sequencially to create a flat list

        self.rolling_forecast_overall_normalization_reverted_test_prediction_loss = 0  # calculated from rolling_forecast_all_days_normalization_reverted_train_predictions and rolling_forecast_all_days_normalization_reverted_test_predictions; loss of this rolling forecast in general (calculated by take one prediction in each rolling forecast day. That said, the calculated value is not true if you want to use one rolling forecast day to predict many outputs. In that situation, use the loss stored in list_of_rolling_forecast_daily_normalization_reverted_prediction_loss)

        self.hyperParameterTuningRollingForecastDataPreparation = None

        self.figure = None

        self.cached_file_name_for_rolling_forecast_preparation = None
        self.list_of_cached_file_name_for_train_input_tensor_list = []
        self.list_of_cached_file_name_for_test_input_tensor_list = []

        if number_of_test_days_in_a_day_level_DataFrame > 1 and number_of_rolling_forecast_days > 1:
            raise NotImplemented('Not implemented')

        if not lazy:
            self.prepare_rolling_forecast_data()
            self.train_model_and_predict_for_each_rolling_forecast_day()

    def prepare_rolling_forecast_data(self):
        if self.enable_data_caching:  # only cache during tuning
            md5_code = ComputeCharacterizationCodeAndMd5Code.prepare_rolling_forecast_data_get_MD5_code(not_normalized_combined_train_DataFrame=self.DataFrame, number_of_days_to_predict_ahead=self.number_of_days_to_predict_ahead, number_of_test_days_in_a_day_level_DataFrame=self.number_of_test_days_in_a_day_level_DataFrame, number_of_rolling_forecast_days=self.number_of_rolling_forecast_days, windowing_option=self.data_windowing_option, input_length=self.input_length, lstm_is_many_to_many=self.lstm_is_many_to_many, do_normalization=self.do_normalization)
            cached_file_name = ComputeCharacterizationCodeAndMd5Code.get_default_cached_file_name(characterization_code=md5_code)
            if path.exists(cached_file_name):
                variableSaverAndLoader = VariableSaverAndLoader(load=True, file_name=cached_file_name)
                assert variableSaverAndLoader.load_is_successful == True
                print('Data load time:', variableSaverAndLoader.duration_time_load)
                self.hyperParameterTuningRollingForecastDataPreparation = variableSaverAndLoader.list_of_loaded_variables[0]
            else:
                self.hyperParameterTuningRollingForecastDataPreparation = HyperParameterTuningRollingForecastDataPreparation(not_normalized_combined_train_DataFrame=self.DataFrame, number_of_days_to_predict_ahead=self.number_of_days_to_predict_ahead, number_of_test_days_in_a_day_level_DataFrame=self.number_of_test_days_in_a_day_level_DataFrame, number_of_rolling_forecast_days=self.number_of_rolling_forecast_days, windowing_option=self.data_windowing_option, input_length=self.input_length, lstm_is_many_to_many=self.lstm_is_many_to_many, do_normalization=self.do_normalization)
                variableSaverAndLoader = VariableSaverAndLoader(list_of_variables_to_save=[self.hyperParameterTuningRollingForecastDataPreparation], save=True, file_name=cached_file_name)
                assert variableSaverAndLoader.save_is_successful == True
                print('Data save time:', variableSaverAndLoader.duration_time_save)
                self.cached_file_name_for_rolling_forecast_preparation = cached_file_name

        else:
            self.hyperParameterTuningRollingForecastDataPreparation = HyperParameterTuningRollingForecastDataPreparation(not_normalized_combined_train_DataFrame=self.DataFrame, number_of_days_to_predict_ahead=self.number_of_days_to_predict_ahead, number_of_test_days_in_a_day_level_DataFrame=self.number_of_test_days_in_a_day_level_DataFrame, number_of_rolling_forecast_days=self.number_of_rolling_forecast_days, windowing_option=self.data_windowing_option, input_length=self.input_length, lstm_is_many_to_many=self.lstm_is_many_to_many, do_normalization=self.do_normalization)

    def train_model_and_predict_for_each_rolling_forecast_day(self):
        '''
        This function has two usages: 1. one rolling forecast day with multiple test days in a dataframe 2. multiple rolling forecast days with one test day in a dataframe

        When many_to_many=True, the model back propagates using all provided output targets; when making a prediction, the model predicts only the last day's output target. Thus, the target in list_of_rolling_forecast_daily_list_of_normalization_reverted_train_targets is not the training target but rather the prediction target.

        '''
        self.dataframe_contains_more_than_one_clinic = len(self.DataFrame['clinic'].unique()) > 1   # this function behaves differently when dataframe has 1. one clinic 2. more than one

        '''For each day, train a and get the prediction.'''
        self.list_of_rolling_forecast_daily_list_of_epoch_loss = []

        self.list_of_rolling_forecast_daily_list_of_normalization_reverted_train_prediction_targets = []
        self.list_of_rolling_forecast_daily_list_of_normalization_reverted_test_prediction_targets = []

        self.list_of_rolling_forecast_daily_list_of_normalized_train_predictions = []  # Each rolling forecast day produces a list of train predictions. We put these lists in a list and call the grand list as list of rolling forecast which means it emcompasses the whole rolling forecast.
        self.list_of_rolling_forecast_daily_list_of_normalized_test_predictions = []

        self.list_of_rolling_forecast_daily_list_of_normalization_reverted_train_prediction_targets = []
        self.list_of_rolling_forecast_daily_list_of_normalization_reverted_test_prediction_targets = []

        self.list_of_rolling_forecast_daily_list_of_normalization_reverted_train_predictions = []
        self.list_of_rolling_forecast_daily_list_of_normalization_reverted_test_predictions = []

        self.list_of_rolling_forecast_daily_normalization_reverted_prediction_loss = []

        for day in self.hyperParameterTuningRollingForecastDataPreparation.day_level_train_input_output_DataFrame_list_dictionary.keys():
            '''statistics needed to revert normalization'''
            mean, std = self.hyperParameterTuningRollingForecastDataPreparation.day_level_normalization_statistics_dictionary[day]
            std_tensor = torch.Tensor(std).reshape(1, len(std))  # size is similar to condensing a dataframe to one row; batch size could only be 1, as coded LSTMModelOperator.get_predictions,
            mean_tensor = torch.Tensor(mean).reshape(1, len(mean))
            mean_tensor_of_first_variable = mean_tensor[0, 0]  # the prediction target; convert to one dimensional tensor to match how LSTMModelOperator.get_predictions generates predictions
            std_tensor_of_first_variable = std_tensor[0, 0]
            '''
            Little matrix multiplication note:
            torch.Tensor([1,2,3,4]).reshape(1,2,2) * torch.Tensor([1,2]).reshape(1,2)
            torch.Tensor([1,2,3,4]).reshape(1,2,2) + torch.Tensor([1,2]).reshape(1,2)
            When a 3D tensor times a 2D tensor, shape (1,any_1,any_2) * (1, any_3), and if any_2 == any_3, each column in the 2D tensor times the corresponding column in the 3D tensor. Same thing applied to addition. This is most useful when reverting the normalization.

            For example:

            tensor([[[1., 2.],            times       tensor([[1., 2.]])
                     [3., 4.]]])
            shape(1,2,2)                              shape(1,2)

            results in: 

            tensor([[[1., 4.],
                     [3., 8.]]])
            shape(1,2,2)
            '''

            '''get data prepared'''
            train_input_output_DataFrame_list = self.hyperParameterTuningRollingForecastDataPreparation.day_level_train_input_output_DataFrame_list_dictionary[day]
            test_input_output_DataFrame_list = self.hyperParameterTuningRollingForecastDataPreparation.day_level_test_input_output_DataFrame_list_dictionary[day]

            if self.enable_data_caching:
                md5_code = ComputeCharacterizationCodeAndMd5Code.train_input_output_tensor_list_get_MD5_code(day=day, not_normalized_combined_train_DataFrame=self.DataFrame, number_of_days_to_predict_ahead=self.number_of_days_to_predict_ahead, number_of_test_days_in_a_day_level_DataFrame=self.number_of_test_days_in_a_day_level_DataFrame, number_of_rolling_forecast_days=self.number_of_rolling_forecast_days, windowing_option=self.data_windowing_option, input_length=self.input_length, lstm_is_many_to_many=self.lstm_is_many_to_many, do_normalization=self.do_normalization, number_of_output_variables=self.number_of_output_variables)
                cached_file_name = ComputeCharacterizationCodeAndMd5Code.get_default_cached_file_name(characterization_code=md5_code)
                if path.exists(cached_file_name):
                    variableSaverAndLoader = VariableSaverAndLoader(load=True, file_name=cached_file_name)
                    assert variableSaverAndLoader.load_is_successful == True
                    print('Data load time: ', variableSaverAndLoader.duration_time_load)
                    train_input_output_tensor_list = variableSaverAndLoader.list_of_loaded_variables[0]
                else:
                    train_input_output_tensor_list = self.get_train_input_output_tensor_list( train_input_output_DataFrame_list=train_input_output_DataFrame_list, lstm_is_many_to_many=self.lstm_is_many_to_many, number_of_output_variables=self.number_of_output_variables)
                    variableSaverAndLoader = VariableSaverAndLoader(list_of_variables_to_save=[train_input_output_tensor_list], save=True, file_name=cached_file_name)
                    assert variableSaverAndLoader.save_is_successful == True
                    print('Data save time', variableSaverAndLoader.duration_time_save)
                    self.list_of_cached_file_name_for_train_input_tensor_list.append(cached_file_name)
            else:
                train_input_output_tensor_list = self.get_train_input_output_tensor_list(train_input_output_DataFrame_list=train_input_output_DataFrame_list, lstm_is_many_to_many=self.lstm_is_many_to_many, number_of_output_variables=self.number_of_output_variables)

            if self.enable_data_caching:
                md5_code = ComputeCharacterizationCodeAndMd5Code.test_input_output_tensor_list_get_MD5_code(day=day, not_normalized_combined_train_DataFrame=self.DataFrame, number_of_days_to_predict_ahead=self.number_of_days_to_predict_ahead, number_of_test_days_in_a_day_level_DataFrame=self.number_of_test_days_in_a_day_level_DataFrame, number_of_rolling_forecast_days=self.number_of_rolling_forecast_days, windowing_option=self.data_windowing_option, input_length=self.input_length, lstm_is_many_to_many=self.lstm_is_many_to_many, do_normalization=self.do_normalization, number_of_output_variables=self.number_of_output_variables)
                cached_file_name = ComputeCharacterizationCodeAndMd5Code.get_default_cached_file_name(characterization_code=md5_code)
                if path.exists(cached_file_name):
                    variableSaverAndLoader = VariableSaverAndLoader(load=True, file_name=cached_file_name)
                    assert variableSaverAndLoader.load_is_successful == True
                    print('Data load time', variableSaverAndLoader.duration_time_load)
                    test_input_output_tensor_list = variableSaverAndLoader.list_of_loaded_variables[0]
                else:
                    test_input_output_tensor_list = self.get_test_input_output_tensor_list(test_input_output_DataFrame_list=test_input_output_DataFrame_list, number_of_output_variables=self.number_of_output_variables)
                    variableSaverAndLoader = VariableSaverAndLoader(list_of_variables_to_save=[test_input_output_tensor_list], save=True, file_name=cached_file_name)
                    assert variableSaverAndLoader.save_is_successful == True
                    print('Data save time', variableSaverAndLoader.duration_time_save)
                    self.list_of_cached_file_name_for_test_input_tensor_list.append(cached_file_name)
            else:
                test_input_output_tensor_list = self.get_test_input_output_tensor_list(test_input_output_DataFrame_list=test_input_output_DataFrame_list, number_of_output_variables=self.number_of_output_variables)

            #  record normalized targets
            if self.lstm_is_many_to_many:
                list_of_normalized_train_prediction_targets = [train_input_output_tensor[1][0, -1, 0] for train_input_output_tensor in train_input_output_tensor_list]
            else:
                list_of_normalized_train_prediction_targets = [train_input_output_tensor[1][0, 0, 0] for train_input_output_tensor in train_input_output_tensor_list]

            list_of_normalized_test_targets = [test_input_output_tensor[1][0, 0, 0] for test_input_output_tensor in test_input_output_tensor_list]

            self.list_of_rolling_forecast_daily_list_of_normalized_train_prediction_targets.append(list_of_normalized_train_prediction_targets)
            self.list_of_rolling_forecast_daily_list_of_normalized_test_prediction_targets.append(list_of_normalized_test_targets)

            # record normalization reverted targets
            list_of_normalization_reverted_train_targets = [target.numpy() * std_tensor_of_first_variable.numpy() + mean_tensor_of_first_variable.numpy() for target in list_of_normalized_train_prediction_targets]

            list_of_normalization_reverted_test_prediction_targets = [target.numpy() * std_tensor_of_first_variable.numpy() + mean_tensor_of_first_variable.numpy() for target in list_of_normalized_test_targets]

            self.list_of_rolling_forecast_daily_list_of_normalization_reverted_train_prediction_targets.append(list_of_normalization_reverted_train_targets)
            self.list_of_rolling_forecast_daily_list_of_normalization_reverted_test_prediction_targets.append(list_of_normalization_reverted_test_prediction_targets)

            '''start and finish training'''
            lstmModelOperator = LSTMModelOperator(hyper_parameter_value_combination=self.hyper_parameter_value_combination, train_input_output_tensor_list=train_input_output_tensor_list, test_input_output_tensor_list=test_input_output_tensor_list, many_to_many=self.lstm_is_many_to_many, loss_function=self.loss_function, batch_size=self.batch_size, generate_train_prediction=self.generate_train_prediction, customized_model_layer_name_to_layer_dictionary=self.customized_model_layer_name_to_layer_dictionary, speed_up_inference=self.speed_up_inference)

            self.list_of_rolling_forecast_daily_lstmModelOperators.append(lstmModelOperator)

            list_of_epoch_loss = lstmModelOperator.list_of_epoch_loss
            list_of_normalized_train_predictions = lstmModelOperator.list_of_train_predictions  # there will only be one prediction for each input tensor (the last time step's), regardless the value of many_to_many
            list_of_normalized_test_predictions = lstmModelOperator.list_of_test_predictions  # when multiple rolling forecast day, one test day in a dataframe and dataframe has only one clinic, this is a list of length one

            # record normalized predictions
            self.list_of_rolling_forecast_daily_list_of_epoch_loss.append(list_of_epoch_loss)
            self.list_of_rolling_forecast_daily_list_of_normalized_train_predictions.append(list_of_normalized_train_predictions)
            self.list_of_rolling_forecast_daily_list_of_normalized_test_predictions.append(list_of_normalized_test_predictions)

            # record normalization reverted predictions
            list_of_normalization_reverted_train_predictions = [prediction.numpy() * std_tensor_of_first_variable.numpy() + mean_tensor_of_first_variable.numpy() for prediction in list_of_normalized_train_predictions]

            list_of_normalization_reverted_test_predictions = [prediction.numpy() * std_tensor_of_first_variable.numpy() + mean_tensor_of_first_variable.numpy() for prediction in list_of_normalized_test_predictions]

            self.list_of_rolling_forecast_daily_list_of_normalization_reverted_train_predictions.append(list_of_normalization_reverted_train_predictions)
            self.list_of_rolling_forecast_daily_list_of_normalization_reverted_test_predictions.append(list_of_normalization_reverted_test_predictions)

            # calculate loss for this rolling window
            # normalization_reverted_prediction_loss = self.loss_function(torch.stack(list_of_normalization_reverted_test_predictions), torch.stack(list_of_normalization_reverted_test_prediction_targets))
            normalization_reverted_prediction_loss = self.loss_function(torch.tensor(list_of_normalization_reverted_test_predictions), torch.tensor(list_of_normalization_reverted_test_prediction_targets))

            self.list_of_rolling_forecast_daily_normalization_reverted_prediction_loss.append(normalization_reverted_prediction_loss)

        '''only useful when multiple rolling forecast days, and each day has one test day'''
        self.rolling_forecast_all_days_normalization_reverted_train_prediction_targets = [train_prediction_target for list_of_normalization_reverted_train_prediction_targets in self.list_of_rolling_forecast_daily_list_of_normalization_reverted_train_prediction_targets for train_prediction_target in list_of_normalization_reverted_train_prediction_targets]

        self.rolling_forecast_all_days_normalization_reverted_test_prediction_targets = [test_prediction_target for list_of_normalization_reverted_test_prediction_targets in self.list_of_rolling_forecast_daily_list_of_normalization_reverted_test_prediction_targets for test_prediction_target in list_of_normalization_reverted_test_prediction_targets]

        self.rolling_forecast_all_days_normalization_reverted_train_predictions = [train_prediction for list_of_normalization_reverted_train_prediction in self.list_of_rolling_forecast_daily_list_of_normalization_reverted_train_predictions for train_prediction in list_of_normalization_reverted_train_prediction]

        self.rolling_forecast_all_days_normalization_reverted_test_predictions = [test_prediction for list_of_normalization_reverted_test_prediction in self.list_of_rolling_forecast_daily_list_of_normalization_reverted_test_predictions for test_prediction in list_of_normalization_reverted_test_prediction]

        # self.rolling_forecast_overall_normalization_reverted_test_prediction_loss = self.loss_function(torch.stack(self.rolling_forecast_all_days_normalization_reverted_test_prediction_targets), torch.stack(self.rolling_forecast_all_days_normalization_reverted_test_predictions))
        self.rolling_forecast_overall_normalization_reverted_test_prediction_loss = self.loss_function(torch.Tensor(self.rolling_forecast_all_days_normalization_reverted_test_prediction_targets), torch.Tensor(self.rolling_forecast_all_days_normalization_reverted_test_predictions))  #TODO replace


    def get_train_input_output_tensor_list(self, train_input_output_DataFrame_list, lstm_is_many_to_many, number_of_output_variables=1):
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

            train_output_values = train_input_output_DataFrame[input_length:].drop(columns='clinic').values[:, :number_of_output_variables]  # output's variable number is pre-defined
            train_output_values_with_batch_dimension = np.expand_dims(train_output_values, 0)
            train_output_tensor = torch.FloatTensor(train_output_values_with_batch_dimension)

            train_input_output_tensor_list.append((train_input_tensor, train_output_tensor))
        return train_input_output_tensor_list

    def get_test_input_output_tensor_list(self, test_input_output_DataFrame_list, number_of_output_variables=1):
        '''Convert dataframe to pytorch tensor. '''
        test_input_output_tensor_list = []  # of shape ((batch, seq_length, number of variables ), (batch, seq_length, number of variables ))
        for test_input_output_DataFrame in test_input_output_DataFrame_list:
            test_input_values = test_input_output_DataFrame[:-1].drop(columns='clinic').values
            test_input_values_with_batch_dimension = np.expand_dims(test_input_values, 0)
            test_input_tensor = torch.FloatTensor(test_input_values_with_batch_dimension)

            test_output_values = test_input_output_DataFrame[-1:].drop(columns='clinic').values[:, :number_of_output_variables]
            test_output_values_with_batch_dimension = np.expand_dims(test_output_values, 0)
            test_output_tensor = torch.FloatTensor(test_output_values_with_batch_dimension)
            test_input_output_tensor_list.append((test_input_tensor, test_output_tensor))
        return test_input_output_tensor_list

    def plot_train_and_test_targets_and_predictions(self, list_of_target_index, list_of_train_prediction_index, list_of_test_prediction_index, list_of_targets, list_of_train_predictions, list_of_test_predictions, train_prediction_label='Training Predictions From the Last Rolling Forecast Day', test_prediction_label='Test Predictions', target_label='Actual'):
        '''Plot prediction vs. actual'''
        if list_of_target_index is None:
            list_of_target_index = list(range(len(list_of_targets)))
        if list_of_train_prediction_index is None:
            list_of_train_prediction_index = list(range(len(list_of_train_predictions)))
        if list_of_test_prediction_index is None:
            list_of_test_prediction_index = list(range(len(list_of_train_predictions), len(list_of_train_predictions) + len(list_of_test_predictions)))

        figure, axis = plt.subplots(1)
        axis.plot(list_of_target_index, list_of_targets, label=target_label)
        axis.plot(list_of_train_prediction_index, list_of_train_predictions, 'r--',
                  label=train_prediction_label)  # -(input_length - 1) to exclude the predictions on rolling forecast days
        axis.plot(list_of_test_prediction_index, list_of_test_predictions, 'g--', label=test_prediction_label)
        axis.legend()
        return figure

