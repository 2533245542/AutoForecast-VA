import unittest

import pandas as pd
import numpy as np

from autoForecastVA.src.components.data_preprocessing_global.data_imputation_and_averaging import DataImputationAndAveraging
from autoForecastVA.src.components.data_preprocessing_global.data_windowing import DataWindowing
from autoForecastVA.src.components.data_preprocessing_global.data_clustering_and_normalization import DataCluseteringAndNormalization
from autoForecastVA.src.components.data_preprocessing_global.data_delimiting_by_day_and_medical_center import DataDelimitingByDayAndMedicalCenter
from autoForecastVA.src.components.hyper_parameter_tuning.hyper_parameter_tuning_rolling_forecast_data_preparation import HyperParameterTuningRollingForecastDataPreparation

class TestHyperParameterTuningRollingForecastDataPreparation(unittest.TestCase):

    def test_toy_example(self):
        '''
        Test1
        We first create a dataset and go through (no data imputation because it is a toy and complete example) data delimiting and normalization. From the resulted dataset, we get the unnormalized data for a day and cluster. We then go thorugh another data delimiting. We lazily initiate a data clustering and normalization instance. We set the medical center to cluster dictionary and cluster to medical center dictionary. We then skip the clustering. We run normalization and split_and_combine_data_back_to_day_level_DataFrame. The resulted data is then fed into data windowing and we are done preparing data for a run of hyper-parameter tuning. These steps are done in two ways, manually and automatically. We test if the two ways generate the same results.
        Test2
        We also streamline the results by removing the cluster and medical center level sequentially. Originally, the results are day_level_cluster_level_medical_center_level dataframe list dictionary. After removing the cluster level, it becomes day level medical center level dataframe list dictionary; the content of the reduced dictionary should have the same dataframe list length as the non reduced one. We also make sure train dictionary plus test dictionary equals to the total dictionary by checking the change of length.
        Test3
        We proceed further by removing the medical center level. After removal, we obtain the day level dataframe list dictionary. The content should contain a dataframe list with length equaling to the sum of medical center level dataframe lists. We also make sure train dictionary plus test dictionary equals to the total dictionary by checking the change of length.

        Below shows the API flow.

        dataClusteringAndNormalization.day_level_cluster_level_not_normalized_combined_train_DataFrame_dictionary[day][cluster] -> DataDelimitingByDayAndMedicalCenter -> DataClusteringAndNormalization(lazy) -> set day_level_medical_center_to_cluster_dictionary and day_level_cluster_level_medical_center_list_dictionary -> run across_medical_center_normalization and split_and_combine_data_back_to_day_level_DataFrame -> dataCluseteringAndNormalization.day_level_cluster_level_medical_center_level_across_medical_center_normalized_DataFrame_dictionary -> DataWindowing ->  dataWindowing.day_level_cluster_level_medical_center_level_input_output_DataFrame_list_dictionary
        '''

        '''Test1'''

        number_of_days_in_dataset = 10
        number_of_days_for_testing = 3
        number_of_test_days_in_DataFrame = 1
        number_of_days_to_predict_ahead = 2
        number_of_days_for_testing_in_a_hyper_parameter_tuning_run = 3

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

        # start of API flow
        ## AutoForecastVA pipeline API flow
        dataDelimitingByDayAndMedicalCenter = DataDelimitingByDayAndMedicalCenter(dataset, number_of_days_for_testing, number_of_days_to_predict_ahead)
        dataCluseteringAndNormalization = DataCluseteringAndNormalization(dataDelimitingByDayAndMedicalCenter, number_of_test_day_in_day_level_DataFrame=number_of_test_days_in_DataFrame, max_number_of_cluster=2, lazy=False)

        ## assuming we are in a loop and is before making the general model
        day = sorted(list(dataCluseteringAndNormalization.day_level_cluster_level_not_normalized_combined_train_DataFrame_dictionary.keys()))[1]  # the second day 2020-01-09
        cluster = sorted(list(dataCluseteringAndNormalization.day_level_cluster_level_not_normalized_combined_train_DataFrame_dictionary[day].keys()))[0]  # cluster 0
        not_normalized_combined_train_DataFrame = dataCluseteringAndNormalization.day_level_cluster_level_not_normalized_combined_train_DataFrame_dictionary[day][cluster]  # from 2020-01-01 to 2020-01-07; contains clinic 111, 222, 333; features are case, call, dayofweek, weekofyear

        '''''in below we make the result manually'''''
        ## start of a hyper-parameter tuning run (the below job should be done automatically)
        dataDelimitingByDayAndMedicalCenter_in_a_hyper_parameter_tuning_run = DataDelimitingByDayAndMedicalCenter(not_normalized_combined_train_DataFrame, number_of_days_for_testing_in_a_hyper_parameter_tuning_run, number_of_days_to_predict_ahead)
        # dataDelimitingByDayAndMedicalCenter_in_a_hyper_parameter_tuning_run.day_level_medical_center_level_DataFrame_dictionary.keys()  # from 2020-01-05 to 2020-01-07

        dataCluseteringAndNormalization_in_a_hyper_parameter_tuning_run = DataCluseteringAndNormalization(dataDelimitingByDayAndMedicalCenter_in_a_hyper_parameter_tuning_run, number_of_test_day_in_day_level_DataFrame=number_of_test_days_in_DataFrame, max_number_of_cluster=2, lazy=True)  # whatever number of cluster; we don't do clustering anyways

        ## Create crafted cluster mapping dictionaries. We will have only one cluster (named 0)
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

        # dataCluseteringAndNormalization_in_a_hyper_parameter_tuning_run.day_level_cluster_level_medical_center_level_across_medical_center_normalized_DataFrame_dictionary.keys()   # from 2020-01-05 to 2020-01-07

        dataWindowing_in_a_hyper_parameter_tuning_run = DataWindowing(dataCluseteringAndNormalization_in_a_hyper_parameter_tuning_run.day_level_cluster_level_medical_center_level_across_medical_center_normalized_DataFrame_dictionary, option=1, number_of_days_to_predict_ahead=number_of_days_to_predict_ahead, number_of_testing_day_in_a_day_level_DataFrame=number_of_test_days_in_DataFrame)

        day_level_cluster_level_medical_center_level_input_output_DataFrame_list_dictionary = dataWindowing_in_a_hyper_parameter_tuning_run.day_level_cluster_level_medical_center_level_input_output_DataFrame_list_dictionary
        '''' *The train dataset* '''''
        day_level_cluster_level_medical_center_level_train_input_output_DataFrame_list_dictionary = dataWindowing_in_a_hyper_parameter_tuning_run.day_level_cluster_level_medical_center_level_train_input_output_DataFrame_list_dictionary
        '''''*The test dataset*'''''
        day_level_cluster_level_medical_center_level_test_input_output_DataFrame_list_dictionary = dataWindowing_in_a_hyper_parameter_tuning_run.day_level_cluster_level_medical_center_level_test_input_output_DataFrame_list_dictionary
        # then we can use the train input-output dataframes to train model and use the test input-output dataframes to test models
        test_day1 = sorted(list(day_level_cluster_level_medical_center_level_train_input_output_DataFrame_list_dictionary.keys()))[1]
        test_day2 = sorted(list(day_level_cluster_level_medical_center_level_train_input_output_DataFrame_list_dictionary.keys()))[2]
        test_medical_center1 = '111'
        test_medical_center2 = '222'

        expected_train_day1_medical_center1_DataFrame1 = day_level_cluster_level_medical_center_level_train_input_output_DataFrame_list_dictionary[test_day1][0][test_medical_center1][0]
        expected_train_day1_medical_center1_DataFrame2 = day_level_cluster_level_medical_center_level_train_input_output_DataFrame_list_dictionary[test_day1][0][test_medical_center1][1]
        expected_train_day1_medical_center2_DataFrame1 = day_level_cluster_level_medical_center_level_train_input_output_DataFrame_list_dictionary[test_day1][0][test_medical_center2][0]
        expected_train_day1_medical_center2_DataFrame2 = day_level_cluster_level_medical_center_level_train_input_output_DataFrame_list_dictionary[test_day1][0][test_medical_center2][1]

        expected_train_day2_medical_center1_DataFrame1 = day_level_cluster_level_medical_center_level_train_input_output_DataFrame_list_dictionary[test_day2][0][test_medical_center1][0]
        expected_train_day2_medical_center1_DataFrame2 = day_level_cluster_level_medical_center_level_train_input_output_DataFrame_list_dictionary[test_day2][0][test_medical_center1][1]
        expected_train_day2_medical_center2_DataFrame1 = day_level_cluster_level_medical_center_level_train_input_output_DataFrame_list_dictionary[test_day2][0][test_medical_center2][0]
        expected_train_day2_medical_center2_DataFrame2 = day_level_cluster_level_medical_center_level_train_input_output_DataFrame_list_dictionary[test_day2][0][test_medical_center2][1]

        expected_test_day1_medical_center1_DataFrame1 = day_level_cluster_level_medical_center_level_test_input_output_DataFrame_list_dictionary[test_day1][0][test_medical_center1][0]
        expected_test_day1_medical_center2_DataFrame2 = day_level_cluster_level_medical_center_level_test_input_output_DataFrame_list_dictionary[test_day1][0][test_medical_center2][0]

        expected_test_day2_medical_center1_DataFrame1 = day_level_cluster_level_medical_center_level_test_input_output_DataFrame_list_dictionary[test_day2][0][test_medical_center1][0]
        expected_test_day2_medical_center2_DataFrame2 = day_level_cluster_level_medical_center_level_test_input_output_DataFrame_list_dictionary[test_day2][0][test_medical_center2][0]

        '''''now we make the result automatically'''''
        hyperParameterTuningRollingForecastDataPreparation = HyperParameterTuningRollingForecastDataPreparation(not_normalized_combined_train_DataFrame, number_of_days_for_testing_in_a_hyper_parameter_tuning_run, number_of_days_to_predict_ahead, number_of_test_days_in_DataFrame)

        resulted_train_day1_medical_center1_DataFrame1 = hyperParameterTuningRollingForecastDataPreparation.day_level_cluster_level_medical_center_level_train_input_output_DataFrame_list_dictionary[test_day1][0][test_medical_center1][0]
        resulted_train_day1_medical_center1_DataFrame2 = hyperParameterTuningRollingForecastDataPreparation.day_level_cluster_level_medical_center_level_train_input_output_DataFrame_list_dictionary[test_day1][0][test_medical_center1][1]
        resulted_train_day1_medical_center2_DataFrame1 = hyperParameterTuningRollingForecastDataPreparation.day_level_cluster_level_medical_center_level_train_input_output_DataFrame_list_dictionary[test_day1][0][test_medical_center2][0]
        resulted_train_day1_medical_center2_DataFrame2 = hyperParameterTuningRollingForecastDataPreparation.day_level_cluster_level_medical_center_level_train_input_output_DataFrame_list_dictionary[test_day1][0][test_medical_center2][1]

        resulted_train_day2_medical_center1_DataFrame1 = hyperParameterTuningRollingForecastDataPreparation.day_level_cluster_level_medical_center_level_train_input_output_DataFrame_list_dictionary[test_day2][0][test_medical_center1][0]
        resulted_train_day2_medical_center1_DataFrame2 = hyperParameterTuningRollingForecastDataPreparation.day_level_cluster_level_medical_center_level_train_input_output_DataFrame_list_dictionary[test_day2][0][test_medical_center1][1]
        resulted_train_day2_medical_center2_DataFrame1 = hyperParameterTuningRollingForecastDataPreparation.day_level_cluster_level_medical_center_level_train_input_output_DataFrame_list_dictionary[test_day2][0][test_medical_center2][0]
        resulted_train_day2_medical_center2_DataFrame2 = hyperParameterTuningRollingForecastDataPreparation.day_level_cluster_level_medical_center_level_train_input_output_DataFrame_list_dictionary[test_day2][0][test_medical_center2][1]

        resulted_test_day1_medical_center1_DataFrame1 = hyperParameterTuningRollingForecastDataPreparation.day_level_cluster_level_medical_center_level_test_input_output_DataFrame_list_dictionary[test_day1][0][test_medical_center1][0]
        resulted_test_day1_medical_center2_DataFrame2 = hyperParameterTuningRollingForecastDataPreparation.day_level_cluster_level_medical_center_level_test_input_output_DataFrame_list_dictionary[test_day1][0][test_medical_center2][0]

        resulted_test_day2_medical_center1_DataFrame1 = hyperParameterTuningRollingForecastDataPreparation.day_level_cluster_level_medical_center_level_test_input_output_DataFrame_list_dictionary[test_day2][0][test_medical_center1][0]
        resulted_test_day2_medical_center2_DataFrame2 = hyperParameterTuningRollingForecastDataPreparation.day_level_cluster_level_medical_center_level_test_input_output_DataFrame_list_dictionary[test_day2][0][test_medical_center2][0]


        pd.testing.assert_frame_equal(expected_train_day1_medical_center1_DataFrame1, resulted_train_day1_medical_center1_DataFrame1)
        pd.testing.assert_frame_equal(expected_train_day1_medical_center1_DataFrame2, resulted_train_day1_medical_center1_DataFrame2)
        pd.testing.assert_frame_equal(expected_train_day1_medical_center2_DataFrame1, resulted_train_day1_medical_center2_DataFrame1)
        pd.testing.assert_frame_equal(expected_train_day1_medical_center2_DataFrame2, resulted_train_day1_medical_center2_DataFrame2)

        pd.testing.assert_frame_equal(expected_train_day2_medical_center1_DataFrame1, resulted_train_day2_medical_center1_DataFrame1)
        pd.testing.assert_frame_equal(expected_train_day2_medical_center1_DataFrame2, resulted_train_day2_medical_center1_DataFrame2)
        pd.testing.assert_frame_equal(expected_train_day2_medical_center2_DataFrame1, resulted_train_day2_medical_center2_DataFrame1)
        pd.testing.assert_frame_equal(expected_train_day2_medical_center2_DataFrame2, resulted_train_day2_medical_center2_DataFrame2)

        pd.testing.assert_frame_equal(expected_test_day1_medical_center1_DataFrame1, resulted_test_day1_medical_center1_DataFrame1)
        pd.testing.assert_frame_equal(expected_test_day1_medical_center2_DataFrame2, resulted_test_day1_medical_center2_DataFrame2)

        pd.testing.assert_frame_equal(expected_test_day2_medical_center1_DataFrame1, resulted_test_day2_medical_center1_DataFrame1)
        pd.testing.assert_frame_equal(expected_test_day2_medical_center2_DataFrame2, resulted_test_day2_medical_center2_DataFrame2)

        '''Test2'''
        for day in hyperParameterTuningRollingForecastDataPreparation.day_level_medical_center_level_input_output_DataFrame_list_dictionary.keys():
            for medical_center in hyperParameterTuningRollingForecastDataPreparation.day_level_medical_center_level_input_output_DataFrame_list_dictionary[day].keys():
                self.assertEqual(len(hyperParameterTuningRollingForecastDataPreparation.day_level_cluster_level_medical_center_level_input_output_DataFrame_list_dictionary[day][0][medical_center]), len(hyperParameterTuningRollingForecastDataPreparation.day_level_medical_center_level_input_output_DataFrame_list_dictionary[day][medical_center]))
                self.assertEqual(len(hyperParameterTuningRollingForecastDataPreparation.day_level_cluster_level_medical_center_level_train_input_output_DataFrame_list_dictionary[day][0][medical_center]), len(hyperParameterTuningRollingForecastDataPreparation.day_level_medical_center_level_train_input_output_DataFrame_list_dictionary[day][medical_center]))
                self.assertEqual(len(hyperParameterTuningRollingForecastDataPreparation.day_level_cluster_level_medical_center_level_test_input_output_DataFrame_list_dictionary[day][0][medical_center]), len(hyperParameterTuningRollingForecastDataPreparation.day_level_medical_center_level_test_input_output_DataFrame_list_dictionary[day][medical_center]))
                self.assertEqual(len(hyperParameterTuningRollingForecastDataPreparation.day_level_medical_center_level_input_output_DataFrame_list_dictionary[day][medical_center]), len(hyperParameterTuningRollingForecastDataPreparation.day_level_medical_center_level_train_input_output_DataFrame_list_dictionary[day][medical_center]) + len(hyperParameterTuningRollingForecastDataPreparation.day_level_medical_center_level_test_input_output_DataFrame_list_dictionary[day][medical_center]))

        '''Test3'''
        for day in hyperParameterTuningRollingForecastDataPreparation.day_level_medical_center_level_input_output_DataFrame_list_dictionary.keys():
            DataFrame_list_length_before_removing_medical_center_level = 0
            train_DataFrame_list_length_before_removing_medical_center_level = 0
            test_DataFrame_list_length_before_removing_medical_center_level = 0
            for medical_center in hyperParameterTuningRollingForecastDataPreparation.day_level_medical_center_level_input_output_DataFrame_list_dictionary[day].keys():
                DataFrame_list_length_before_removing_medical_center_level += len(hyperParameterTuningRollingForecastDataPreparation.day_level_medical_center_level_input_output_DataFrame_list_dictionary[day][medical_center])
                train_DataFrame_list_length_before_removing_medical_center_level += len(hyperParameterTuningRollingForecastDataPreparation.day_level_medical_center_level_train_input_output_DataFrame_list_dictionary[day][medical_center])
                test_DataFrame_list_length_before_removing_medical_center_level += len(hyperParameterTuningRollingForecastDataPreparation.day_level_medical_center_level_test_input_output_DataFrame_list_dictionary[day][medical_center])
            self.assertEqual(DataFrame_list_length_before_removing_medical_center_level, len(hyperParameterTuningRollingForecastDataPreparation.day_level_input_output_DataFrame_list_dictionary[day]))
            self.assertEqual(train_DataFrame_list_length_before_removing_medical_center_level, len(hyperParameterTuningRollingForecastDataPreparation.day_level_train_input_output_DataFrame_list_dictionary[day]))
            self.assertEqual(test_DataFrame_list_length_before_removing_medical_center_level, len(hyperParameterTuningRollingForecastDataPreparation.day_level_test_input_output_DataFrame_list_dictionary[day]))
            self.assertEqual(len(hyperParameterTuningRollingForecastDataPreparation.day_level_input_output_DataFrame_list_dictionary[day]), len(hyperParameterTuningRollingForecastDataPreparation.day_level_train_input_output_DataFrame_list_dictionary[day]) + len(hyperParameterTuningRollingForecastDataPreparation.day_level_test_input_output_DataFrame_list_dictionary[day]))

    def test_real_example(self):
        '''
        In day-level, medical center-level dataset generated from a rolling forecast prepartion, for each medical center, for each day, the length the of training input-output dataframe list should be larger than the length the of the previous day's training input-output dataframe list by 1, while the length of the testing input-output dataframe list should always be 1. Also, for each medical center, if we sum the length of train input-output dataframe list for all days, it should equal to n * (a1+an) / 2 (according to the formula of arithmetic progression) where
        a1 = length of the first training input-output dataframe list
        an = length of the first training input-output dataframe list + n - 1
        n = number_of_days_for_testing_in_a_hyper_parameter_tuning_run
        ; if we sum the length of the test input-output dataframe list for all days, it should equal to number_of_days_for_testing_in_a_hyper_parameter_tuning_run.

        For each day, we also calculate the number of input-output dataframes created. It should equal to the number of medical centers * the length of the input-output dataframe list for a medical center of that day)
        '''

        '''''set up a pipeline'''''
        number_of_days_for_testing = 30
        number_of_test_days_in_DataFrame = 1
        number_of_days_to_predict_ahead = 2
        number_of_days_for_testing_in_a_hyper_parameter_tuning_run = 5
        dataset = DataImputationAndAveraging(dataset_path='../../../data/coviddata07292020.csv', number_of_days_for_data_averaging=3, filter_by_medical_center=True, filter_by_time_period=False).get_processed_dataset()
        dataDelimitingByDayAndMedicalCenter = DataDelimitingByDayAndMedicalCenter(dataset, number_of_days_for_testing, number_of_days_to_predict_ahead)
        dataCluseteringAndNormalization = DataCluseteringAndNormalization(dataDelimitingByDayAndMedicalCenter, number_of_test_day_in_day_level_DataFrame=number_of_test_days_in_DataFrame, max_number_of_cluster=10)
        for day in dataCluseteringAndNormalization.day_level_cluster_level_not_normalized_combined_train_DataFrame_dictionary.keys():
            for cluster in dataCluseteringAndNormalization.day_level_cluster_level_not_normalized_combined_train_DataFrame_dictionary[day].keys():
                not_normalized_combined_train_DataFrame = dataCluseteringAndNormalization.day_level_cluster_level_not_normalized_combined_train_DataFrame_dictionary[day][cluster]
                hyperParameterTuningRollingForecastDataPreparation = HyperParameterTuningRollingForecastDataPreparation(not_normalized_combined_train_DataFrame, number_of_days_for_testing_in_a_hyper_parameter_tuning_run, number_of_days_to_predict_ahead, number_of_test_days_in_DataFrame)
                '''''The relevent part of the pipeline we need to test'''''
                # Iterate over medical centers then day. This is viable because the medical centers are the same in different days.
                list_of_hyper_parameter_tuning_rolling_forecast_day = list(hyperParameterTuningRollingForecastDataPreparation.day_level_medical_center_level_input_output_DataFrame_list_dictionary.keys())
                list_of_hyper_parameter_tuning_rolling_forecast_medical_center = list(hyperParameterTuningRollingForecastDataPreparation.day_level_medical_center_level_input_output_DataFrame_list_dictionary[list_of_hyper_parameter_tuning_rolling_forecast_day[0]].keys())
                for hyper_parameter_tuning_rolling_forecast_medical_center in list_of_hyper_parameter_tuning_rolling_forecast_medical_center:
                    first_train_DataFrame_list_length = 0
                    previous_train_DataFrame_list_length = -1
                    total_train_DataFrame_list_length = 0
                    total_test_DataFrame_list_length = 0
                    for hyper_parameter_tuning_rolling_forecast_day in list_of_hyper_parameter_tuning_rolling_forecast_day:
                        DataFrame_list_length = len(hyperParameterTuningRollingForecastDataPreparation.day_level_medical_center_level_input_output_DataFrame_list_dictionary[hyper_parameter_tuning_rolling_forecast_day][hyper_parameter_tuning_rolling_forecast_medical_center])
                        train_DataFrame_list_length = len(hyperParameterTuningRollingForecastDataPreparation.day_level_medical_center_level_train_input_output_DataFrame_list_dictionary[hyper_parameter_tuning_rolling_forecast_day][hyper_parameter_tuning_rolling_forecast_medical_center])
                        test_DataFrame_list_length = len(hyperParameterTuningRollingForecastDataPreparation.day_level_medical_center_level_test_input_output_DataFrame_list_dictionary[hyper_parameter_tuning_rolling_forecast_day][hyper_parameter_tuning_rolling_forecast_medical_center])
                        self.assertEqual(DataFrame_list_length, train_DataFrame_list_length + test_DataFrame_list_length)
                        if previous_train_DataFrame_list_length != -1:  # start testing from the second day when we have record for the prvious day's train dataframe list length
                            self.assertEqual(1, train_DataFrame_list_length - previous_train_DataFrame_list_length)
                        if previous_train_DataFrame_list_length == -1:  # record first day's train length
                            first_train_DataFrame_list_length = train_DataFrame_list_length
                        self.assertEqual(1, test_DataFrame_list_length)
                        previous_train_DataFrame_list_length = train_DataFrame_list_length  # record the last day's train dataframe list length
                        total_train_DataFrame_list_length += train_DataFrame_list_length
                        total_test_DataFrame_list_length += test_DataFrame_list_length

                    a_1 = first_train_DataFrame_list_length
                    a_n = first_train_DataFrame_list_length + number_of_days_for_testing_in_a_hyper_parameter_tuning_run - 1
                    n = number_of_days_for_testing_in_a_hyper_parameter_tuning_run

                    self.assertEqual(n * (a_1 + a_n) / 2, total_train_DataFrame_list_length)
                    self.assertEqual(number_of_days_for_testing_in_a_hyper_parameter_tuning_run, total_test_DataFrame_list_length)

                for hyper_parameter_tuning_rolling_forecast_day in list_of_hyper_parameter_tuning_rolling_forecast_day:
                    first_DataFrame_list_length = len(hyperParameterTuningRollingForecastDataPreparation.day_level_medical_center_level_input_output_DataFrame_list_dictionary[hyper_parameter_tuning_rolling_forecast_day][list_of_hyper_parameter_tuning_rolling_forecast_medical_center[0]])  # this day's any medical center's train_input_output dataframe list
                    self.assertEqual(len(list_of_hyper_parameter_tuning_rolling_forecast_medical_center) * first_DataFrame_list_length, len(hyperParameterTuningRollingForecastDataPreparation.day_level_input_output_DataFrame_list_dictionary[hyper_parameter_tuning_rolling_forecast_day]))