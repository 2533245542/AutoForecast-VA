import pandas as pd

from autoForecastVA.src.components.hyper_parameter_tuning.hyper_parameter_value_combination_evaluator import HyperParameterValueCombinationEvaluator


class GeneralModelTraining():
    '''
    Given datasets, it makes a instance. When the instance is called with instance.train(hyper-parameter value combination), the instance returns a value.
    inputs:
    hyper_parameter_value_combination: contains budget,
    train_DataFrame
    number_of_days_to_predict_ahead
    loss_function
    data_windowing_option
    input_length
    lstm_is_many_to_many
    do_normalization=True

    outputs:
    dataframe_with_one_extra_day
    hyperParameterValueCombinationEvaluator
    lstm_model_operator
    lstm_model

    steps:
    add one day to the end of the train_DataFrame to fake a rolling forecast
    do a evaluation with just one rolling forecast day and one (fake) test day
    get the model


    '''
    def __init__(self, hyper_parameter_value_combination, train_DataFrame, loss_function, data_windowing_option, input_length, lstm_is_many_to_many, number_of_days_to_predict_ahead, do_normalization=True, batch_size=1, lazy=False):

        # inputs
        self.hyper_parameter_value_combination = hyper_parameter_value_combination
        self.train_DataFrame = train_DataFrame
        self.loss_function = loss_function
        self.data_windowing_option = data_windowing_option
        self.input_length = input_length
        self.lstm_is_many_to_many = lstm_is_many_to_many
        self.number_of_days_to_predict_ahead = number_of_days_to_predict_ahead
        self.do_normalization = do_normalization
        self.batch_size = batch_size
        self.lazy = lazy

        # outputs
        self.dataframe_with_one_extra_day = None
        self.hyperParameterValueCombinationEvaluator = None
        self.operator = None
        self.general_model = None

        # constants: these constants are set to make HyperParameterValueCombinationEvaluator do one rolling forecast on the fake day, and test on the fake day, in this way, all train_DataFrame will be used for model building
        self.number_of_days_for_testing_in_a_hyper_parameter_tuning_run = 1
        self.number_of_test_days_in_a_day_level_DataFrame = 1  # one fake day

        if not lazy:
            self.add_extra_day()
            self.train_general_model()
            self.derive_model_related_outputs()

    def add_extra_day(self):
        self.dataframe_with_one_extra_day = self.append_one_fake_day_to_dataframe_for_each_clinic(dataframe=self.train_DataFrame)

    def train_general_model(self):
        self.hyperParameterValueCombinationEvaluator = HyperParameterValueCombinationEvaluator(DataFrame=self.dataframe_with_one_extra_day, loss_function=self.loss_function, hyper_parameter_value_combination=self.hyper_parameter_value_combination, number_of_days_to_predict_ahead=self.number_of_days_to_predict_ahead, number_of_test_days_in_a_day_level_DataFrame=self.number_of_test_days_in_a_day_level_DataFrame, number_of_rolling_forecast_days=self.number_of_days_for_testing_in_a_hyper_parameter_tuning_run, data_windowing_option=self.data_windowing_option, input_length=self.input_length, lstm_is_many_to_many=self.lstm_is_many_to_many, do_normalization=self.do_normalization, batch_size=self.batch_size)

    def derive_model_related_outputs(self):
        self.operator = self.hyperParameterValueCombinationEvaluator.list_of_rolling_forecast_daily_lstmModelOperators[0]
        self.general_model = self.operator.model

    @staticmethod
    def append_one_fake_day_to_dataframe_for_each_clinic(dataframe):
        '''
        inputs: dataframe
        outputs: dataframe with a fake day appended to the end, for each clinic
        '''

        dataframe_of_this_clinic_with_one_extra_day_list = []
        for clinic in dataframe.clinic.unique():
            dataframe_of_one_clinic = dataframe[
                dataframe.clinic == clinic].sort_index()  # sorted by time, from earlies to latest

            timestamp_of_the_last_day = dataframe_of_one_clinic.index[-1]
            timestamp_of_the_last_day_plus_one = timestamp_of_the_last_day + pd.Timedelta(1, unit='d')

            index_name = dataframe_of_one_clinic.index.name

            # fake a dataframe that has only one day, then append the row to the original dataframe
            dataframe_of_the_last_day = dataframe_of_one_clinic.iloc[-1:, ]
            dataframe_of_the_last_day_plus_one = dataframe_of_the_last_day.copy()
            dataframe_of_the_last_day_plus_one = dataframe_of_the_last_day_plus_one.set_index(
                pd.Index(data=[timestamp_of_the_last_day_plus_one], name=index_name))

            dataframe_of_one_clinic_with_one_extra_day = pd.concat(
                [dataframe_of_one_clinic, dataframe_of_the_last_day_plus_one], axis=0)
            dataframe_of_this_clinic_with_one_extra_day_list.append(dataframe_of_one_clinic_with_one_extra_day)

        dataframe_with_one_extra_day = pd.concat(dataframe_of_this_clinic_with_one_extra_day_list, axis=0)
        return (dataframe_with_one_extra_day)


