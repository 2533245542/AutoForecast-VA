from autoForecastVA.src.components.data_preprocessing_global.data_clustering_and_normalization import DataCluseteringAndNormalization
from autoForecastVA.src.components.data_preprocessing_global.data_delimiting_by_day_and_medical_center import DataDelimitingByDayAndMedicalCenter
from autoForecastVA.src.components.data_preprocessing_global.data_windowing import DataWindowing
from functools import reduce


class HyperParameterTuningRollingForecastDataPreparation():
    def __init__(self, not_normalized_combined_train_DataFrame, number_of_rolling_forecast_days=3, number_of_days_to_predict_ahead=1, number_of_test_days_in_a_day_level_DataFrame=1, windowing_option=1, input_length=2, lstm_is_many_to_many=False, do_normalization=True, lazy=False):
        '''
        The input DataFrame should be unnormalized, and might contain multiple medical centers; for each medical center in the DataFrame, the days should be continous (no skipping days between two consective rows)

        The preparator first delimits the DataFrame to day level and medical center level.
        It then skips clustering (pretending all centers are in one cluster), and do normalization across medical centers.
        The resulted normalized day level cluster level medical center level DataFrame dictionary is then sent to DataWindowing to get the input output DataFrames.
        DataWindowing will give the day_level cluster_level medical center_level input_output_DataFrame_list dictionary.

        In future calls, the day level [...] input_output_DataFrame_list dictionary can be used by evaluator to find a deep learning model's performance (along with a fixed hyper-paraemter value combination) on the input DataFrame (not_normalized_combined_train_DataFrame).

        :param not_normalized_combined_train_DataFrame: a train DataFrame as in dataCluseteringAndNormalization.day_level_cluster_level_not_normalized_combined_train_DataFrame_dictionary[day][cluster]
        :param number_of_test_days_in_a_day_level_DataFrame: in rolling forecast, this is usually set to 1 because we usually want to make the test day to be only one (using one input to test the model before building the model again); however, if you want to use more days to test the models (feed more than one inputs) before rolling one day forward (before building the model again and make the prediction for another day), you can set this to more than one. In that case, the model will be fed by more test inputs and thus make more predictions. Sometimes this can be used as a replacement of rolling forecast and it runs faster because it does not build the model over and over.
        For example, in rolling forecast, you can set number_of_rolling_forecast_days=3 and number_of_test_days_in_a_day_level_DataFrame=1 to build the model three times; you can also set number_of_rolling_forecast_days=1 and number_of_test_days_in_a_day_level_DataFrame=3 to build the model once.
        '''
        self.not_normalized_combined_train_DataFrame = not_normalized_combined_train_DataFrame
        self.number_of_rolling_forecast_days = number_of_rolling_forecast_days
        self.number_of_days_to_predict_ahead = number_of_days_to_predict_ahead
        self.number_of_test_days_in_a_day_level_DataFrame = number_of_test_days_in_a_day_level_DataFrame
        self.windowing_option = windowing_option
        self.input_length = input_length
        self.do_normalization = do_normalization
        self.lstm_is_many_to_many = lstm_is_many_to_many

        self.day_level_normalization_statistics_dictionary = {}
        self.day_level_cluster_level_medical_center_level_input_output_DataFrame_list_dictionary = {}
        self.day_level_cluster_level_medical_center_level_train_input_output_DataFrame_list_dictionary = {}
        self.day_level_cluster_level_medical_center_level_test_input_output_DataFrame_list_dictionary = {}

        self.day_level_medical_center_level_input_output_DataFrame_list_dictionary = {}
        self.day_level_medical_center_level_train_input_output_DataFrame_list_dictionary = {}
        self.day_level_medical_center_level_test_input_output_DataFrame_list_dictionary = {}

        self.day_level_input_output_DataFrame_list_dictionary = {}
        self.day_level_train_input_output_DataFrame_list_dictionary = {}
        self.day_level_test_input_output_DataFrame_list_dictionary = {}

        if not lazy:
            self.prepare_data()

    def prepare_data(self):
        '''Here shows the consequence of bad bad coding. We have to convert dataframe to DataDelimitingByDayAndMedicalCenter object (even though the medical center level does not matter) such that we can use the DataCluseteringAndNormalization.
        Also, we have to manually skip the clustering part of DataCluseteringAndNormalization such that we can only use the normalization part.
        '''

        dataDelimitingByDayAndMedicalCenter_in_a_hyper_parameter_tuning_run = DataDelimitingByDayAndMedicalCenter(dataset=self.not_normalized_combined_train_DataFrame, number_of_days_for_testing=self.number_of_rolling_forecast_days, number_of_days_to_predict_ahead=self.number_of_days_to_predict_ahead)
        dataCluseteringAndNormalization_in_a_hyper_parameter_tuning_run = DataCluseteringAndNormalization(dataDelimitingByDayAndMedicalCenter_in_a_hyper_parameter_tuning_run, number_of_test_day_in_day_level_DataFrame=self.number_of_test_days_in_a_day_level_DataFrame, max_number_of_cluster=2, do_normalization=self.do_normalization, lazy=True)  # whatever number of cluster; we don't do clustering anyways

        day_level_medical_center_to_cluster_dictionary = {}
        day_level_cluster_level_medical_center_list_dictionary = {}

        for day in dataDelimitingByDayAndMedicalCenter_in_a_hyper_parameter_tuning_run.day_level_medical_center_level_DataFrame_dictionary.keys():
            medical_center_to_cluster_dictionary = {}
            cluster_level_medical_center_list_dictionary = {0: list(dataDelimitingByDayAndMedicalCenter_in_a_hyper_parameter_tuning_run.day_level_medical_center_level_DataFrame_dictionary[day].keys())}
            for medical_center in dataDelimitingByDayAndMedicalCenter_in_a_hyper_parameter_tuning_run.day_level_medical_center_level_DataFrame_dictionary[day].keys():
                medical_center_to_cluster_dictionary[medical_center] = 0
            day_level_medical_center_to_cluster_dictionary[day] = medical_center_to_cluster_dictionary
            day_level_cluster_level_medical_center_list_dictionary[day] = cluster_level_medical_center_list_dictionary

        dataCluseteringAndNormalization_in_a_hyper_parameter_tuning_run.day_level_medical_center_to_cluster_dictionary = day_level_medical_center_to_cluster_dictionary
        dataCluseteringAndNormalization_in_a_hyper_parameter_tuning_run.day_level_cluster_level_medical_center_list_dictionary = day_level_cluster_level_medical_center_list_dictionary
        dataCluseteringAndNormalization_in_a_hyper_parameter_tuning_run.across_medical_center_normalization()
        dataCluseteringAndNormalization_in_a_hyper_parameter_tuning_run.split_and_combine_data_back_to_day_level_DataFrame()

        dataWindowing_in_a_hyper_parameter_tuning_run = DataWindowing(dataCluseteringAndNormalization_in_a_hyper_parameter_tuning_run.day_level_cluster_level_medical_center_level_across_medical_center_normalized_DataFrame_dictionary, option=self.windowing_option, input_length=self.input_length, number_of_days_to_predict_ahead=self.number_of_days_to_predict_ahead, number_of_testing_day_in_a_day_level_DataFrame=self.number_of_test_days_in_a_day_level_DataFrame, lstm_is_many_to_many=self.lstm_is_many_to_many)

        self.day_level_cluster_level_medical_center_level_input_output_DataFrame_list_dictionary = dataWindowing_in_a_hyper_parameter_tuning_run.day_level_cluster_level_medical_center_level_input_output_DataFrame_list_dictionary
        '''*The train dataset*'''
        self.day_level_cluster_level_medical_center_level_train_input_output_DataFrame_list_dictionary = dataWindowing_in_a_hyper_parameter_tuning_run.day_level_cluster_level_medical_center_level_train_input_output_DataFrame_list_dictionary
        '''*The test dataset*'''
        self.day_level_cluster_level_medical_center_level_test_input_output_DataFrame_list_dictionary = dataWindowing_in_a_hyper_parameter_tuning_run.day_level_cluster_level_medical_center_level_test_input_output_DataFrame_list_dictionary

        '''Make the day level normalization statistics dictionary'''
        for day in dataCluseteringAndNormalization_in_a_hyper_parameter_tuning_run.day_level_cluster_level_combined_normalization_statistics_dictionary.keys():
            self.day_level_normalization_statistics_dictionary[day] = dataCluseteringAndNormalization_in_a_hyper_parameter_tuning_run.day_level_cluster_level_combined_normalization_statistics_dictionary[day][0]

        '''Remove the cluster level'''
        for day in self.day_level_cluster_level_medical_center_level_input_output_DataFrame_list_dictionary.keys():
            self.day_level_medical_center_level_input_output_DataFrame_list_dictionary[day] = self.day_level_cluster_level_medical_center_level_input_output_DataFrame_list_dictionary[day][0]
            self.day_level_medical_center_level_train_input_output_DataFrame_list_dictionary[day] = self.day_level_cluster_level_medical_center_level_train_input_output_DataFrame_list_dictionary[day][0]
            self.day_level_medical_center_level_test_input_output_DataFrame_list_dictionary[day] = self.day_level_cluster_level_medical_center_level_test_input_output_DataFrame_list_dictionary[day][0]

        '''Remove the medical center level'''
        for day in self.day_level_medical_center_level_input_output_DataFrame_list_dictionary.keys():
            # reduce a list of lists to one list
            self.day_level_input_output_DataFrame_list_dictionary[day] = reduce(lambda list1, list2: list1+list2, self.day_level_medical_center_level_input_output_DataFrame_list_dictionary[day].values())
            self.day_level_train_input_output_DataFrame_list_dictionary[day] = reduce(lambda list1, list2: list1+list2, self.day_level_medical_center_level_train_input_output_DataFrame_list_dictionary[day].values())
            self.day_level_test_input_output_DataFrame_list_dictionary[day] = reduce(lambda list1, list2: list1+list2, self.day_level_medical_center_level_test_input_output_DataFrame_list_dictionary[day].values())

