from autoForecastVA.src.components.data_preprocessing_global.data_clustering_and_normalization import \
    DataCluseteringAndNormalization
from autoForecastVA.src.components.data_preprocessing_global.data_delimiting_by_day_and_medical_center import \
    DataDelimitingByDayAndMedicalCenter
from autoForecastVA.src.components.data_preprocessing_global.data_imputation_and_averaging import DataImputationAndAveraging
from autoForecastVA.src.components.data_preprocessing_global.data_windowing import DataWindowing
import unittest
import pandas as pd
from pandas.api.types import is_numeric_dtype
from pandas.api.types import is_string_dtype
import numpy as np


class TestWindowing(unittest.TestCase):

    def test_toy_example(self):
        '''We have a time series data for a medical center at a day. Given the number of days to predict ahead and the two options: 1. A fixed sequence length 2. Using as much data as possible, we create the input-out DataFrame '''
        '''
        We will create as input-output dataframes
        option 1
        train input-output dataframes
        '2020-01-03', '2020-01-05'
        '2020-01-03', '2020-01-04','2020-01-06'
        
        test input-output dataframes
        '2020-01-03', '2020-01-04', '2020-01-05', '2020-01-07'
        '2020-01-03', '2020-01-04', '2020-01-05','2020-01-06','2020-01-08'
        
        option 2
        train input-output dataframes
        '2020-01-03', '2020-01-04', '2020-01-06'
        
        test input-output dataframes
        '2020-01-04', '2020-01-05', '2020-01-07'
        '2020-01-05','2020-01-06','2020-01-08'
        
        In option 1, the number of test input-output dataframes for tests are determined by checking if each test day has enough preceeding days. In order to create the input for one test day, we should have in the day-level DataFrame the day equaling to the test day minus the number of days to predict ahead. We test this condition for every day, and increments the count of generated input-output dataframes after such a test succeeds. Similarly, the number of train input-output dataframes are determined by checking if each train day has enough preceeding days.
        
        number_of_train_input_output_DataFrame = 0
        number_of_test_input_output_DataFrame = 0
        for  day  in day-level DataFrame.index[:-number of test days]
            day -= number of days to predict ahead
            if day in day-level DataFrame.index
                number_of_train_input_output_DataFrame += 1
            
        for  day  in day-level DataFrame.index[-number of test days:]
            day -= number of days to predict ahead
            if day in day-level DataFrame.index
                number_of_test_input_output_DataFrame += 1

        
        In option 2, the number of test input-output dataframes for tests are determined by checking if each test day has enough preceeding days. In order to create the input for one test day, we should have in the day-level DataFrame the day equaling to the test day minus the number of days to predict ahead and minus the sequence length + 1. We test this condition for every tes day, and increments the count of generated input-output dataframes after such a test succeeds. Similarly, the number of train input-output dataframes are determined by checking if each train day has enough preceeding days.
        number_of_train_input_output_DataFrame = 0
        number_of_test_input_output_DataFrame = 0
        for  day  in day-level DataFrame.index[:-number of test days]
            day -= number of days to predict ahead - input_length + 1
            if day in day-level DataFrame.index
                number_of_train_input_output_DataFrame += 1
            
        for  day  in day-level DataFrame.index[-number of test days:]
            day -= number of days to predict ahead - input_length + 1
            if day in day-level DataFrame.index
                number_of_test_input_output_DataFrame += 1

        Lastly, when we choose to do many-to-one windowing, some DatFrame has even rows and some has odd rows; but when doing many-to-many windowing, all train_input_output_DataFrames should have even rows(input has the same length as output).
        '''
        number_of_days_to_predict_ahead = 2
        number_of_testing_day_in_a_day_level_DataFrame = 2
        input_length = 2

        DataFrame = pd.DataFrame({'date':['2020-01-03', '2020-01-04', '2020-01-05','2020-01-06','2020-01-07','2020-01-08'], 'clinic': ['111','111','111','111','111', '111'], 'case':[1,2,3,4,5,6], 'call':[1,1,1,1,1,1]})

        DataFrame['date'] = pd.to_datetime(DataFrame.date, infer_datetime_format=True)
        DataFrame = DataFrame.set_index(['date'])


        for option in [1,2]:
            for lstm_is_many_to_many in [True, False]:
                dataWindowing = DataWindowing({}, option=option, number_of_days_to_predict_ahead=number_of_days_to_predict_ahead, number_of_testing_day_in_a_day_level_DataFrame=number_of_testing_day_in_a_day_level_DataFrame, input_length=input_length, lazy=True, lstm_is_many_to_many=lstm_is_many_to_many)

                train_input_output_DataFrame_list, test_input_output_DataFrame_list = dataWindowing.create_input_output_DataFrames(DataFrame)

                if option == 1:
                    '''option 1'''
                    if lstm_is_many_to_many:
                        # self.assertEqual(0, train_input_output_DataFrame_list[0].shape[0] % 2)
                        # self.assertEqual(0, train_input_output_DataFrame_list[1].shape[0] % 2)
                        self.assertEqual(2, train_input_output_DataFrame_list[0].shape[0])
                        self.assertEqual(4, train_input_output_DataFrame_list[1].shape[0])
                    else:
                        self.assertEqual(2, train_input_output_DataFrame_list[0].shape[0])
                        self.assertEqual(3, train_input_output_DataFrame_list[1].shape[0])
                elif option == 2:
                    '''option 2'''
                    if lstm_is_many_to_many:
                        self.assertEqual(4, train_input_output_DataFrame_list[0].shape[0])
                    else:
                        self.assertEqual(3, train_input_output_DataFrame_list[0].shape[0])
                else:
                    raise ValueError

                if lstm_is_many_to_many:
                    '''did not prepare below codes' lstm_is_many_to_many==True version, so just skip'''
                    continue

                # tests
                number_of_train_input_output_DataFrame = 0
                number_of_test_input_output_DataFrame = 0

                if option == 1:
                    for day in DataFrame.index[:-number_of_testing_day_in_a_day_level_DataFrame]:
                        day -= pd.Timedelta(number_of_days_to_predict_ahead, unit='d')
                        if day in DataFrame.index:
                            number_of_train_input_output_DataFrame += 1
                    for day in DataFrame.index[-number_of_testing_day_in_a_day_level_DataFrame:]:
                        day -= pd.Timedelta(number_of_days_to_predict_ahead, unit='d')
                        if day in DataFrame.index:
                            number_of_test_input_output_DataFrame += 1
                elif option == 2:
                    for day in DataFrame.index[:-number_of_testing_day_in_a_day_level_DataFrame]:
                        day -= pd.Timedelta(number_of_days_to_predict_ahead + input_length - 1, unit='d')
                        if day in DataFrame.index:
                            number_of_train_input_output_DataFrame += 1

                    for day in DataFrame.index[-number_of_testing_day_in_a_day_level_DataFrame:]:
                        day -= pd.Timedelta(number_of_days_to_predict_ahead + input_length - 1, unit='d')
                        if day in DataFrame.index:
                            number_of_test_input_output_DataFrame += 1
                else:
                    raise ValueError

                # option 1 has 2 train, 2 test; option 2 has 1 train, 2 test
                self.assertEqual(len(train_input_output_DataFrame_list), number_of_train_input_output_DataFrame)
                self.assertEqual(len(test_input_output_DataFrame_list), number_of_test_input_output_DataFrame)

                if option == 1:

                    self.assertListEqual(train_input_output_DataFrame_list[0].index.strftime('%Y-%m-%d').tolist(), ['2020-01-03', '2020-01-05'])
                    self.assertListEqual(train_input_output_DataFrame_list[1].index.strftime('%Y-%m-%d').tolist(), ['2020-01-03', '2020-01-04', '2020-01-06'])

                    self.assertListEqual(test_input_output_DataFrame_list[0].index.strftime('%Y-%m-%d').tolist(), ['2020-01-03', '2020-01-04', '2020-01-05', '2020-01-07'])
                    self.assertListEqual(test_input_output_DataFrame_list[1].index.strftime('%Y-%m-%d').tolist(), ['2020-01-03', '2020-01-04', '2020-01-05','2020-01-06','2020-01-08'])
                elif option == 2:
                    self.assertListEqual(train_input_output_DataFrame_list[0].index.strftime('%Y-%m-%d').tolist(), ['2020-01-03', '2020-01-04', '2020-01-06'])
                    self.assertListEqual(test_input_output_DataFrame_list[0].index.strftime('%Y-%m-%d').tolist(), ['2020-01-04', '2020-01-05', '2020-01-07'])
                    self.assertListEqual(test_input_output_DataFrame_list[1].index.strftime('%Y-%m-%d').tolist(), ['2020-01-05','2020-01-06','2020-01-08'])
                else:
                    raise ValueError



    def test_real_example(self):
        '''We have tested the function `create_input_output_DataFrames` in the toy example. In this test, we focus on testing the non-lazy initiation of DataWindowing. We calculate the number of train plus test input-output dataframes for each list in the day_level_cluster_level_medical_center_level_input_output_DataFrame_list_dictionary and confirm that the calculated ones and the actual ones in day_level_cluster_level_medical_center_level_input_output_DataFrame_list_dictionary are equal.

        We also make sure that the individual input-output DataFrame is at a regularly range of length. When in option 1, the longest input-output DataFrame should have length equals to the (last day - the first day + 1 - number of days to predict ahead); the shortest DataFrame should be of length 2; when in option 2, each input-output DataFrame should be of length input_length + 1'''

        number_of_days_to_predict_ahead = 2
        number_of_testing_day_in_a_day_level_DataFrame = 1
        input_length = 2
        number_of_days_to_for_testing = 30

        dataset = DataImputationAndAveraging(dataset_path='../../../data/coviddata07292020.csv', number_of_days_for_data_averaging=3, filter_by_medical_center=True, filter_by_time_period=True).get_processed_dataset()
        dataDelimitingByDayAndMedicalCenter = DataDelimitingByDayAndMedicalCenter(dataset, number_of_days_to_for_testing, number_of_days_to_predict_ahead)
        dataCluseteringAndNormalization = DataCluseteringAndNormalization(dataDelimitingByDayAndMedicalCenter, number_of_test_day_in_day_level_DataFrame=number_of_testing_day_in_a_day_level_DataFrame, max_number_of_cluster=10)
        for option in [1,2]:
            dataWindowing = DataWindowing(dataCluseteringAndNormalization.day_level_cluster_level_medical_center_level_across_medical_center_normalized_DataFrame_dictionary, option=option, number_of_days_to_predict_ahead=number_of_days_to_predict_ahead, number_of_testing_day_in_a_day_level_DataFrame=number_of_testing_day_in_a_day_level_DataFrame, input_length=input_length)
            day_level_cluster_level_medical_center_level_input_output_DataFrame_list_dictionary = dataWindowing.day_level_cluster_level_medical_center_level_input_output_DataFrame_list_dictionary
            day_level_cluster_level_medical_center_level_train_input_output_DataFrame_list_dictionary = dataWindowing.day_level_cluster_level_medical_center_level_train_input_output_DataFrame_list_dictionary
            day_level_cluster_level_medical_center_level_test_input_output_DataFrame_list_dictionary = dataWindowing.day_level_cluster_level_medical_center_level_test_input_output_DataFrame_list_dictionary

            for day in day_level_cluster_level_medical_center_level_input_output_DataFrame_list_dictionary.keys():
                for cluster in day_level_cluster_level_medical_center_level_input_output_DataFrame_list_dictionary[day].keys():
                    for medical_center in day_level_cluster_level_medical_center_level_input_output_DataFrame_list_dictionary[day][cluster].keys():
                        # using the original DataFrame, calculates the number of input-output DataFrames
                        DataFrame = dataDelimitingByDayAndMedicalCenter.get_day_level_medical_center_level_DataFrame_dictionary()[day][medical_center]
                        number_of_train_input_output_DataFrame = 0
                        number_of_test_input_output_DataFrame = 0

                        if option == 1:
                            for temp_day in DataFrame.index[:-number_of_testing_day_in_a_day_level_DataFrame]:  # use temp_day to not overwrite the day in the larger loop
                                temp_day -= pd.Timedelta(number_of_days_to_predict_ahead, unit='d')
                                if temp_day in DataFrame.index:
                                    number_of_train_input_output_DataFrame += 1
                            for temp_day in DataFrame.index[-number_of_testing_day_in_a_day_level_DataFrame:]:
                                temp_day -= pd.Timedelta(number_of_days_to_predict_ahead, unit='d')
                                if temp_day in DataFrame.index:
                                    number_of_test_input_output_DataFrame += 1
                        elif option == 2:
                            for temp_day in DataFrame.index[:-number_of_testing_day_in_a_day_level_DataFrame]:
                                temp_day -= pd.Timedelta(number_of_days_to_predict_ahead + input_length - 1, unit='d')
                                if temp_day in DataFrame.index:
                                    number_of_train_input_output_DataFrame += 1

                            for temp_day in DataFrame.index[-number_of_testing_day_in_a_day_level_DataFrame:]:
                                temp_day -= pd.Timedelta(number_of_days_to_predict_ahead + input_length - 1, unit='d')
                                if temp_day in DataFrame.index:
                                    number_of_test_input_output_DataFrame += 1
                        else:
                            raise ValueError

                        # the calculated value should equal to the generated value
                        self.assertEqual(len(day_level_cluster_level_medical_center_level_input_output_DataFrame_list_dictionary[day][cluster][medical_center]), number_of_train_input_output_DataFrame + number_of_test_input_output_DataFrame)
                        # the length of the concatenated DataFrame should equal to the sum length of its two parts
                        self.assertEqual(len(day_level_cluster_level_medical_center_level_input_output_DataFrame_list_dictionary[day][cluster][medical_center]), len(day_level_cluster_level_medical_center_level_train_input_output_DataFrame_list_dictionary[day][cluster][medical_center]) + len(day_level_cluster_level_medical_center_level_test_input_output_DataFrame_list_dictionary[day][cluster][medical_center]))

                        # calculate the length of individual input-output DataFrames
                        if option == 1:
                            longest_DataFrame = 0
                            shortest_DataFrame = 0
                            longest_DataFrame_length = np.NINF
                            shortest_DataFrame_length = np.PINF
                            for DataFrame in day_level_cluster_level_medical_center_level_input_output_DataFrame_list_dictionary[day][cluster][medical_center]:
                                if DataFrame.shape[0] > longest_DataFrame_length:
                                    longest_DataFrame_length = DataFrame.shape[0]
                                    longest_DataFrame = DataFrame
                                if DataFrame.shape[0] < shortest_DataFrame_length:
                                    shortest_DataFrame_length = DataFrame.shape[0]
                                    shortest_DataFrame = DataFrame
                            '''The longest input-output DataFrame should have length equals to the (last day - the first day + 1 - number of days to predict ahead); the shortest DataFrame should be of length 2'''
                            day_level_medical_center_level_DataFrame_dictionary = dataDelimitingByDayAndMedicalCenter.get_day_level_medical_center_level_DataFrame_dictionary()
                            first_day = day_level_medical_center_level_DataFrame_dictionary[day][medical_center].index[0]
                            last_day = day_level_medical_center_level_DataFrame_dictionary[day][medical_center].index[-1]
                            self.assertEqual(longest_DataFrame_length, (last_day - first_day).days + 1 - (number_of_days_to_predict_ahead - 1))
                            self.assertEqual(shortest_DataFrame_length, 2)
                        else:
                            for DataFrame in day_level_cluster_level_medical_center_level_input_output_DataFrame_list_dictionary[day][cluster][medical_center]:
                                self.assertEqual(DataFrame.shape[0], input_length+1)
                                '''Each input-output DataFrame should be of length input_length + 1'''


    def test_real_example_with_lstm_is_many_to_many(self):
        '''When we are windowing data for many-to-many, all train input-output dataframe should have even number of rows; when for many-to-one, sometimes it has even and sometimes it has odd'''

        number_of_days_to_predict_ahead = 2
        number_of_testing_day_in_a_day_level_DataFrame = 1
        input_length = 2
        number_of_days_to_for_testing = 30

        dataset = DataImputationAndAveraging(dataset_path='../../../data/coviddata07292020.csv', number_of_days_for_data_averaging=3, filter_by_medical_center=True, filter_by_time_period=True).get_processed_dataset()
        dataDelimitingByDayAndMedicalCenter = DataDelimitingByDayAndMedicalCenter(dataset, number_of_days_to_for_testing, number_of_days_to_predict_ahead)
        dataCluseteringAndNormalization = DataCluseteringAndNormalization(dataDelimitingByDayAndMedicalCenter, number_of_test_day_in_day_level_DataFrame=number_of_testing_day_in_a_day_level_DataFrame, max_number_of_cluster=10)
        for option in [1,2]:
            for lstm_is_many_to_many in [True, False]:
                dataWindowing = DataWindowing(dataCluseteringAndNormalization.day_level_cluster_level_medical_center_level_across_medical_center_normalized_DataFrame_dictionary, option=option, number_of_days_to_predict_ahead=number_of_days_to_predict_ahead, number_of_testing_day_in_a_day_level_DataFrame=number_of_testing_day_in_a_day_level_DataFrame, input_length=input_length, lstm_is_many_to_many=lstm_is_many_to_many)
                day_level_cluster_level_medical_center_level_input_output_DataFrame_list_dictionary = dataWindowing.day_level_cluster_level_medical_center_level_input_output_DataFrame_list_dictionary
                day_level_cluster_level_medical_center_level_train_input_output_DataFrame_list_dictionary = dataWindowing.day_level_cluster_level_medical_center_level_train_input_output_DataFrame_list_dictionary
                day_level_cluster_level_medical_center_level_test_input_output_DataFrame_list_dictionary = dataWindowing.day_level_cluster_level_medical_center_level_test_input_output_DataFrame_list_dictionary

                for day in day_level_cluster_level_medical_center_level_input_output_DataFrame_list_dictionary.keys():
                    for cluster in day_level_cluster_level_medical_center_level_input_output_DataFrame_list_dictionary[day].keys():
                        for medical_center in day_level_cluster_level_medical_center_level_input_output_DataFrame_list_dictionary[day][cluster].keys():
                            number_of_times_train_input_output_DataFrame_does_not_have_even_rows = 0
                            number_of_times_train_input_output_DataFrame_has_even_rows = 0
                            '''only look at train input-output dataframe'''
                            train_input_output_DataFrame_list = day_level_cluster_level_medical_center_level_train_input_output_DataFrame_list_dictionary[day][cluster][medical_center]
                            for train_input_output_DataFrame in train_input_output_DataFrame_list:
                                if lstm_is_many_to_many == True:
                                    self.assertEqual(0, train_input_output_DataFrame.shape[0] % 2)
                                elif lstm_is_many_to_many == False:
                                    if train_input_output_DataFrame.shape[0] % 2 == 0:
                                        number_of_times_train_input_output_DataFrame_has_even_rows += 1
                                    else:
                                        number_of_times_train_input_output_DataFrame_does_not_have_even_rows += 1
                                else:
                                    raise ValueError

                            # print(train_input_output_DataFrame_list)
                            # print(number_of_times_train_input_output_DataFrame_does_not_have_even_rows, number_of_times_train_input_output_DataFrame_has_even_rows)
                            # print(day)

                            if option == 1 and lstm_is_many_to_many == False:
                                self.assertNotEqual(0, number_of_times_train_input_output_DataFrame_has_even_rows)
                                self.assertNotEqual(0, number_of_times_train_input_output_DataFrame_does_not_have_even_rows)
                            if option == 2 and lstm_is_many_to_many == False:
                                self.assertEqual(0, number_of_times_train_input_output_DataFrame_has_even_rows)
                                self.assertNotEqual(0, number_of_times_train_input_output_DataFrame_does_not_have_even_rows)
