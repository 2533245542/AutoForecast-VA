from autoForecastVA.src.components.data_preprocessing_global.data_clustering_and_normalization import DataCluseteringAndNormalization
from autoForecastVA.src.components.data_preprocessing_global.data_imputation_and_averaging import DataImputationAndAveraging
from autoForecastVA.src.components.data_preprocessing_global.data_delimiting_by_day_and_medical_center import DataDelimitingByDayAndMedicalCenter
import unittest
import pandas as pd
import numpy as np

from autoForecastVA.src.utils.tuning_interval_moderator import OperationIntervalModerator


class TestDataCluseteringAndNormalization(unittest.TestCase):
    '''We first creat a toy example using a number of DataFrames for 3 days and 3 medical centers. Each medical center contains 3 features. Two medical centers have the same trend and one is distinct. We use the last day as the prediction day and predict one day ahead. We test both the toy example and the real case.'''
    number_of_days_for_testing = 30
    number_of_days_to_predict_ahead = 3
    medical_center_level_DataFrame_dictionary = {}
    medical_center_level_day_level_DataFrame_dictionary = {}
    day_level_medical_center_level_DataFrame_dictionary = {}
    def set_up_toy_example(self):
        dataset = pd.DataFrame({'date':['2020-01-03', '2020-01-04', '2020-01-05', '2020-01-03', '2020-01-04', '2020-01-05', '2020-01-03', '2020-01-04', '2020-01-05'], 'clinic': ['111','111','111','222','222','222','333','333','333'], 'case':[1,2,3,1,2,3,1,1,1], 'call':[1,2,3,1,2,3,1,2,3]})
        alt_clustering_dataset = pd.DataFrame({'date':['2020-01-03', '2020-01-04', '2020-01-05', '2020-01-03', '2020-01-04', '2020-01-05', '2020-01-03', '2020-01-04', '2020-01-05', '2020-01-03', '2020-01-04', '2020-01-05'], 'clinic': ['111','111','111','222','222','222','333','333','333', '444','444','444'], 'case':[1,2,3,1,2,3,1,1,1,1,0.5,1], 'call':[1,2,3,1,2,3,1,2,3,1,2,3]})

        dataset['date'] = pd.to_datetime(dataset.date, infer_datetime_format=True)
        dataset = dataset.set_index(['date'])
        alt_clustering_dataset['date'] = pd.to_datetime(alt_clustering_dataset.date, infer_datetime_format=True)
        alt_clustering_dataset = alt_clustering_dataset.set_index(['date'])

        dataDelimitingByDayAndMedicalCenter = DataDelimitingByDayAndMedicalCenter(dataset, 1, 1)
        self.medical_center_level_DataFrame_dictionary = dataDelimitingByDayAndMedicalCenter.get_medical_center_level_DataFrame_dictionary()
        self.medical_center_level_day_level_DataFrame_dictionary = dataDelimitingByDayAndMedicalCenter.get_medical_center_level_day_level_DataFrame_dictionary()
        self.day_level_medical_center_level_DataFrame_dictionary = dataDelimitingByDayAndMedicalCenter.get_day_level_medical_center_level_DataFrame_dictionary()
        self.dataCluseteringAndNormalization = DataCluseteringAndNormalization(dataDelimitingByDayAndMedicalCenter, number_of_test_day_in_day_level_DataFrame=1, max_number_of_cluster=2, lazy=True)

        test_clustering_dataDelimitingByDayAndMedicalCenter = DataDelimitingByDayAndMedicalCenter(alt_clustering_dataset, 1, 1)
        self.alt_clustering_dataCluseteringAndNormalization = DataCluseteringAndNormalization(test_clustering_dataDelimitingByDayAndMedicalCenter, number_of_test_day_in_day_level_DataFrame=1, max_number_of_cluster=3, lazy=True)


    def set_up_real_example(self, use_filter):
        dataset = DataImputationAndAveraging(dataset_path='../../../data/coviddata07292020.csv', number_of_days_for_data_averaging=3, filter_by_medical_center=use_filter).get_processed_dataset()
        dataDelimitingByDayAndMedicalCenter = DataDelimitingByDayAndMedicalCenter(dataset, self.number_of_days_for_testing, self.number_of_days_to_predict_ahead)
        self.dataCluseteringAndNormalization = DataCluseteringAndNormalization(dataDelimitingByDayAndMedicalCenter, number_of_test_day_in_day_level_DataFrame=1, max_number_of_cluster=10, lazy=True)


    def test_toy_example(self):
        '''We first used the previous compoent to create the medical center level and day level DataFrames. We then apply the clustering technique on the prediction day. In this case, we only have one prediction so it will only perform clustering once. After the clustering technique, we should have the mappings between clusters and medical center. It should indicate that medical center 111 and 222 are in the same cluster while 333 is in a cluster of its own.

        Then, we perform normalization within medical center on training data, which, in this case, is the data of all but the last day. A mean and a standard deviation should be calculated for each feature based on the training data. For `case` it should be 1.5 and 0.7 for 111 and 222; and it should be 1 and 1 for 333. For `call`, it should be 1.5 and 0.707107 for three medical centers. We then normalize the each feature within each medical center. For `case`, 111 and 222's values become [?] and 333's becomes [?] (because when standard deviation is 0, it means the feature is homogenous and we should not do normalization; however, to make this feature the in the same scale as the others, we replace the values them with 0, which is the mean of standard deviation). For `call`, 111,222 and 333's value shoud be [-0.707107, 0.707107, 2.121320]. We record the standard deviations in a dictionary.

        Next, we should do cross medical center normalization. We split each medical center's day-level DataFrame into a training and a testing DataFrame. We then combine the training together and combine the testing DataFrames together. We should keep two lists, one for training and one for testing DataFrame, showing which observation belongs to which medical center and we will add it to the DataFrame later. We then calculate the statistics for the combined training DataFrame, and normalize both combined DataFrames. Now we should have the normalized day-cluster level DataFrame to train the general model, as well as the normalized day-level DataFrame for fine-tuning.

        The next part will just be windowing the DataFrames and we provide two options.
        '''
        self.set_up_toy_example()

        self.dataCluseteringAndNormalization.clustering()
        self.alt_clustering_dataCluseteringAndNormalization.clustering()

        for day in self.alt_clustering_dataCluseteringAndNormalization.day_level_medical_center_level_DataFrame_dictionary.keys():
            self.assertEqual(len(self.alt_clustering_dataCluseteringAndNormalization.day_level_cluster_level_medical_center_list_dictionary[day].keys()), 2)
            self.assertListEqual(self.alt_clustering_dataCluseteringAndNormalization.day_level_cluster_level_medical_center_list_dictionary[day][0], ['333', '444'])
            self.assertListEqual(self.alt_clustering_dataCluseteringAndNormalization.day_level_cluster_level_medical_center_list_dictionary[day][1], ['111', '222'])

        for day in self.day_level_medical_center_level_DataFrame_dictionary.keys():
            # should contain only two clusters as it is the optimal split
            self.assertEqual(len(self.dataCluseteringAndNormalization.day_level_cluster_level_medical_center_list_dictionary[day].keys()), 2)

            self.assertEqual(self.dataCluseteringAndNormalization.day_level_medical_center_to_cluster_dictionary[day]['111'], 0) # only one day of data is used so we just use the medical_center_to_cluster_dictionary for simplicity
            self.assertEqual(self.dataCluseteringAndNormalization.day_level_medical_center_to_cluster_dictionary[day]['222'], 0)
            self.assertEqual(self.dataCluseteringAndNormalization.day_level_medical_center_to_cluster_dictionary[day]['333'], 1)
            self.assertListEqual(self.dataCluseteringAndNormalization.day_level_cluster_level_medical_center_list_dictionary[day][0], ['111', '222'])
            self.assertListEqual(self.dataCluseteringAndNormalization.day_level_cluster_level_medical_center_list_dictionary[day][1], ['333'])

        self.dataCluseteringAndNormalization.across_medical_center_normalization()

        ''' 
        We normalize in a cluster across medical centers based on the training data. Cluster 0 contains 111 and 222. The training data are {'case': [1,2,1,2], 'call': [1,2,1,2]}. The mean is [1.5,1.5] and the std is [0.577,0.577]. Cluster 1 contains 333. The training data are {'case': [1,1], 'call': [1,2]}. The mean is [1,1.5] and the std is [1,0.707106]. 
        We also make sure that the day_level_cluster_level_not_normalized_combined_DataFrame_dictionary combines the DataFrames in day_level_cluster_level_not_normalized_combined_train_DataFrame_dictionary and day_level_cluster_level_not_normalized_combined_test_DataFrame_dictionary. 
        Similarly, we make sure that day_level_cluster_level_normalized_combined_train_DataFrame_dictionary combines the DataFrames in day_level_cluster_level_normalized_combined_test_DataFrame_dictionary and day_level_cluster_level_normalized_combined_DataFrame_dictionary.
        '''

        for day in self.day_level_medical_center_level_DataFrame_dictionary.keys():
            for cluster in self.dataCluseteringAndNormalization.day_level_cluster_level_medical_center_list_dictionary[day].keys():
                if cluster == 0:
                    np.testing.assert_almost_equal(self.dataCluseteringAndNormalization.day_level_cluster_level_combined_normalization_statistics_dictionary[day][cluster][0], [1.5,1.5], decimal=1)
                    np.testing.assert_almost_equal(self.dataCluseteringAndNormalization.day_level_cluster_level_combined_normalization_statistics_dictionary[day][cluster][1], [0.577,0.577], decimal=3)
                    self.assertEqual(self.dataCluseteringAndNormalization.day_level_cluster_level_normalized_combined_train_DataFrame_dictionary[day][cluster].shape[0], 4)
                    self.assertEqual(self.dataCluseteringAndNormalization.day_level_cluster_level_normalized_combined_test_DataFrame_dictionary[day][cluster].shape[0], 2)
                    # for the not normalized and normalized combined DataFrame, the shape should be both (6, 3)
                    self.assertTupleEqual(self.dataCluseteringAndNormalization.day_level_cluster_level_not_normalized_combined_DataFrame_dictionary[day][cluster].shape, (6, 3))
                    self.assertEqual(self.dataCluseteringAndNormalization.day_level_cluster_level_normalized_combined_DataFrame_dictionary[day][cluster].shape, (6, 3))

                else:
                    np.testing.assert_almost_equal(self.dataCluseteringAndNormalization.day_level_cluster_level_combined_normalization_statistics_dictionary[day][cluster][0], [1,1.5], decimal=1)
                    np.testing.assert_almost_equal(self.dataCluseteringAndNormalization.day_level_cluster_level_combined_normalization_statistics_dictionary[day][cluster][1], [1,0.707106], decimal=3)
                    self.assertEqual(self.dataCluseteringAndNormalization.day_level_cluster_level_normalized_combined_train_DataFrame_dictionary[day][cluster].shape[0], 2)
                    self.assertEqual(self.dataCluseteringAndNormalization.day_level_cluster_level_normalized_combined_test_DataFrame_dictionary[day][cluster].shape[0], 1)
                    # for the not normalized and normalized combined DataFrame, the shape should be both (3, 3)
                    self.assertTupleEqual(self.dataCluseteringAndNormalization.day_level_cluster_level_not_normalized_combined_DataFrame_dictionary[day][cluster].shape, (3,3))
                    self.assertEqual(self.dataCluseteringAndNormalization.day_level_cluster_level_normalized_combined_DataFrame_dictionary[day][cluster].shape, (3,3))

        self.dataCluseteringAndNormalization.split_and_combine_data_back_to_day_level_DataFrame()

        for day in self.day_level_medical_center_level_DataFrame_dictionary.keys():
            for cluster in self.dataCluseteringAndNormalization.day_level_cluster_level_medical_center_list_dictionary[day].keys():
                for medical_center in self.dataCluseteringAndNormalization.day_level_cluster_level_medical_center_list_dictionary[day][cluster]:
                    dataFrame = self.dataCluseteringAndNormalization.day_level_medical_center_level_across_medical_center_normalized_DataFrame_dictionary[day][medical_center]
                    self.assertTupleEqual(dataFrame.shape, (3,3))

    def test_real_example(self):
        for use_filter in [True, False]:
            self.set_up_real_example(use_filter)
            self.dataCluseteringAndNormalization.clustering()
            # the number of clusters generated should be <= 10 for all days; the sum of the number of medical center in all cluster of all days should equal to day*medical centers; one medical center should map exactly to one cluster at any day
            total_number_of_medical_center = 0
            for day in self.dataCluseteringAndNormalization.day_level_cluster_level_medical_center_list_dictionary.keys():
                self.assertLessEqual(len(self.dataCluseteringAndNormalization.day_level_cluster_level_medical_center_list_dictionary[day].keys()), 10)
                for medical_center in self.dataCluseteringAndNormalization.day_level_medical_center_to_cluster_dictionary[day].keys():
                    total_number_of_medical_center += 1

            self.assertEqual(total_number_of_medical_center, len(
                self.dataCluseteringAndNormalization.day_level_cluster_level_medical_center_list_dictionary.keys()) * len(
                self.dataCluseteringAndNormalization.medical_center_level_DataFrame_dictionary.keys()))


            for day in self.dataCluseteringAndNormalization.day_level_medical_center_level_DataFrame_dictionary.keys():
                for medical_center in self.dataCluseteringAndNormalization.day_level_medical_center_level_DataFrame_dictionary[day].keys():
                    number_of_cluster_this_medical_center_belongs = 0
                    for cluster in self.dataCluseteringAndNormalization.day_level_cluster_level_medical_center_list_dictionary[day].keys():
                        if medical_center in self.dataCluseteringAndNormalization.day_level_cluster_level_medical_center_list_dictionary[day][cluster]:
                            number_of_cluster_this_medical_center_belongs += 1
                    self.assertEqual(number_of_cluster_this_medical_center_belongs, 1)


            self.dataCluseteringAndNormalization.across_medical_center_normalization()
            # for a day, counting the number of rows in the combined testing dataframe for all clusters should equal to the the number of test day times the number of medical centers;
            # we also make sure that the day_level_cluster_level_not_normalized_combined_DataFrame_dictionary combines the DataFrames in day_level_cluster_level_not_normalized_combined_train_DataFrame_dictionary and day_level_cluster_level_not_normalized_combined_test_DataFrame_dictionary.
            # similarly, we make sure that day_level_cluster_level_normalized_combined_train_DataFrame_dictionary combines the DataFrames in day_level_cluster_level_normalized_combined_test_DataFrame_dictionary and day_level_cluster_level_normalized_combined_DataFrame_dictionary.
            # after normaliztion, the values should also mostly different and smaller; revert the normalization and the data should be the same

            for day in self.dataCluseteringAndNormalization.day_level_cluster_level_not_normalized_combined_train_DataFrame_dictionary.keys():
                for cluster in self.dataCluseteringAndNormalization.day_level_cluster_level_not_normalized_combined_train_DataFrame_dictionary[day].keys():
                    not_normalized_combined_train_DataFrame = self.dataCluseteringAndNormalization.day_level_cluster_level_not_normalized_combined_train_DataFrame_dictionary[day][cluster]
                    not_normalized_combined_test_DataFrame = self.dataCluseteringAndNormalization.day_level_cluster_level_not_normalized_combined_test_DataFrame_dictionary[day][cluster]
                    not_normalized_combined_DataFrame = self.dataCluseteringAndNormalization.day_level_cluster_level_not_normalized_combined_DataFrame_dictionary[day][cluster]  # new
                    normalized_combined_train_DataFrame = self.dataCluseteringAndNormalization.day_level_cluster_level_normalized_combined_train_DataFrame_dictionary[day][cluster]
                    normalized_combined_test_DataFrame = self.dataCluseteringAndNormalization.day_level_cluster_level_normalized_combined_test_DataFrame_dictionary[day][cluster]
                    normalized_combined_DataFrame = self.dataCluseteringAndNormalization.day_level_cluster_level_normalized_combined_DataFrame_dictionary[day][cluster]  # new
                    self.assertTupleEqual(not_normalized_combined_train_DataFrame.shape, normalized_combined_train_DataFrame.shape)
                    self.assertTupleEqual(not_normalized_combined_test_DataFrame.shape, normalized_combined_test_DataFrame.shape)
                    self.assertTupleEqual(not_normalized_combined_DataFrame.shape, (not_normalized_combined_train_DataFrame.shape[0] + not_normalized_combined_test_DataFrame.shape[0], not_normalized_combined_train_DataFrame.shape[1]))
                    self.assertTupleEqual(normalized_combined_DataFrame.shape, (normalized_combined_train_DataFrame.shape[0] + normalized_combined_test_DataFrame.shape[0], normalized_combined_train_DataFrame.shape[1]))

                    number_of_train_different_values = 0
                    number_of_train_smaller_values = 0
                    number_of_train_total_values = not_normalized_combined_train_DataFrame.shape[0] * (not_normalized_combined_train_DataFrame.shape[1] - 1)

                    number_of_test_different_values = 0
                    number_of_test_smaller_values = 0
                    number_of_test_total_values = not_normalized_combined_test_DataFrame.shape[0] * (not_normalized_combined_test_DataFrame.shape[1] - 1)

                    for i in range(not_normalized_combined_train_DataFrame.shape[0]):
                        for j in range(1, not_normalized_combined_train_DataFrame.shape[1]):
                            if not_normalized_combined_train_DataFrame.iloc[i, j] != normalized_combined_train_DataFrame.iloc[i, j]:
                                number_of_train_different_values += 1
                            if not_normalized_combined_train_DataFrame.iloc[i, j] > normalized_combined_train_DataFrame.iloc[i, j]:
                                number_of_train_smaller_values += 1
                    for i in range(not_normalized_combined_test_DataFrame.shape[0]):
                        for j in range(1, not_normalized_combined_test_DataFrame.shape[1]):
                            if not_normalized_combined_test_DataFrame.iloc[i, j] != normalized_combined_test_DataFrame.iloc[i, j]:
                                number_of_test_different_values += 1
                            if not_normalized_combined_test_DataFrame.iloc[i, j] > normalized_combined_test_DataFrame.iloc[i, j]:
                                number_of_test_smaller_values += 1
                    self.assertGreater(number_of_train_different_values / number_of_train_total_values, 0.9)
                    self.assertGreater(number_of_train_smaller_values / number_of_train_total_values, 0.8)  # different value than double normalization because the we changed < comparison to > when finding number_of_test_smaller_values and number_of_train_smaller_values

                    self.assertGreater(number_of_test_different_values / number_of_test_total_values, 0.9)
                    self.assertGreater(number_of_test_smaller_values / number_of_test_total_values, 0.8)

                    mean, std = self.dataCluseteringAndNormalization.day_level_cluster_level_combined_normalization_statistics_dictionary[day][cluster]

                    revert_normalized_combined_train_DataFrame = normalized_combined_train_DataFrame.iloc[:, 1:] * std + mean
                    revert_normalized_combined_test_DataFrame = normalized_combined_test_DataFrame.iloc[:, 1:] * std + mean
                    revert_normalized_combined_train_DataFrame.insert(0, 'clinic', normalized_combined_train_DataFrame.clinic)
                    revert_normalized_combined_test_DataFrame.insert(0, 'clinic', normalized_combined_test_DataFrame.clinic)
                    pd.testing.assert_frame_equal(revert_normalized_combined_train_DataFrame, not_normalized_combined_train_DataFrame, check_dtype=False)
                    pd.testing.assert_frame_equal(revert_normalized_combined_test_DataFrame, not_normalized_combined_test_DataFrame, check_dtype=False)


            self.dataCluseteringAndNormalization.split_and_combine_data_back_to_day_level_DataFrame()

            '''Very likely that the below code is redundant as we only did one normalization instead of two, but whatever, we leave them there.'''
            # revert the normalization and they should be the same as the original data
            for day in self.dataCluseteringAndNormalization.day_level_medical_center_level_across_medical_center_normalized_DataFrame_dictionary.keys():
                for medical_center in self.dataCluseteringAndNormalization.day_level_medical_center_level_across_medical_center_normalized_DataFrame_dictionary[day].keys():
                    # revert the DataFrame normalized across medical centers, compare with the DataFrame normalized within medical centers; revert again, compare with the freshly delimited DataFrame
                    day_level_medical_center_level_across_cluster_normalized_DataFrame = self.dataCluseteringAndNormalization.day_level_medical_center_level_across_medical_center_normalized_DataFrame_dictionary[day][medical_center]

                    cluster = self.dataCluseteringAndNormalization.day_level_medical_center_to_cluster_dictionary[day][medical_center]
                    mean, std = self.dataCluseteringAndNormalization.day_level_cluster_level_combined_normalization_statistics_dictionary[day][cluster]

                    revert_across_medical_center_DataFrame = day_level_medical_center_level_across_cluster_normalized_DataFrame.iloc[:, 1:] * std + mean
                    revert_across_medical_center_DataFrame.insert(0, 'clinic', day_level_medical_center_level_across_cluster_normalized_DataFrame.clinic)
                    pd.testing.assert_frame_equal(revert_across_medical_center_DataFrame,self.dataCluseteringAndNormalization.day_level_medical_center_level_DataFrame_dictionary[day][medical_center], check_dtype=False)

            # the dictionaries created for future convientiice (day_level_cluster_level_medical_center_level_DataFrame_dictionary, day_level_cluster_level_medical_center_level_across_medical_center_normalized_DataFrame_dictionary) should contain the same data as the dictionaries where they synthesized on
            for day in self.dataCluseteringAndNormalization.day_level_cluster_level_medical_center_level_DataFrame_dictionary.keys():
                for cluster in self.dataCluseteringAndNormalization.day_level_cluster_level_medical_center_level_DataFrame_dictionary[day].keys():
                    for medical_center in self.dataCluseteringAndNormalization.day_level_cluster_level_medical_center_level_DataFrame_dictionary[day][cluster].keys():
                        # test equivalence of the original dataset
                        pd.testing.assert_frame_equal(self.dataCluseteringAndNormalization.day_level_cluster_level_medical_center_level_DataFrame_dictionary[day][cluster][medical_center], self.dataCluseteringAndNormalization.day_level_medical_center_level_DataFrame_dictionary[day][medical_center])
                        # test equuivalence of the normalized dataset
                        pd.testing.assert_frame_equal(self.dataCluseteringAndNormalization.day_level_cluster_level_medical_center_level_across_medical_center_normalized_DataFrame_dictionary[day][cluster][medical_center], self.dataCluseteringAndNormalization.day_level_medical_center_level_across_medical_center_normalized_DataFrame_dictionary[day][medical_center])
    def test_operation_interval(self):
        def assert_frame_not_equal(*args, **kwargs):
            try:
                pd.testing.assert_frame_equal(*args, **kwargs)
            except AssertionError:
                # frames are not equal
                pass
            else:
                # frames are equal
                raise AssertionError
        def assert_series_not_equal(*args, **kwargs):
            try:
                pd.testing.assert_series_equal(*args, **kwargs)
            except AssertionError:
                # series are not equal
                pass
            else:
                # series are equal
                raise AssertionError

        dataset = DataImputationAndAveraging(dataset_path='../../../data/coviddata07292020.csv', number_of_days_for_data_averaging=3).get_processed_dataset()
        dataDelimitingByDayAndMedicalCenter = DataDelimitingByDayAndMedicalCenter(dataset=dataset, number_of_days_for_testing=30, number_of_days_to_predict_ahead=1)

        operation_interval = 5
        dataCluseteringAndNormalization = DataCluseteringAndNormalization(dataDelimitingByDayAndMedicalCenter, number_of_test_day_in_day_level_DataFrame=1, max_number_of_cluster=10, operation_interval=operation_interval)

        operationIntervalModerator = OperationIntervalModerator(days=dataCluseteringAndNormalization.day_level_cluster_level_not_normalized_combined_train_DataFrame_dictionary.keys(), operation_interval=operation_interval)

        for day in dataCluseteringAndNormalization.day_level_cluster_level_not_normalized_combined_train_DataFrame_dictionary.keys():
            opeartion_day = operationIntervalModerator.get_operation_day_of_a_day(day)
            print(day, opeartion_day)
            self.assertDictEqual(dataCluseteringAndNormalization.day_level_medical_center_to_cluster_dictionary[day], dataCluseteringAndNormalization.day_level_medical_center_to_cluster_dictionary[opeartion_day])
            self.assertDictEqual(dataCluseteringAndNormalization.day_level_cluster_level_medical_center_list_dictionary[day], dataCluseteringAndNormalization.day_level_cluster_level_medical_center_list_dictionary[opeartion_day])

            for cluster in dataCluseteringAndNormalization.day_level_cluster_level_not_normalized_combined_train_DataFrame_dictionary[day].keys():
                if not operationIntervalModerator.day_is_operation_day(day):
                    assert_frame_not_equal(dataCluseteringAndNormalization.day_level_cluster_level_not_normalized_combined_train_DataFrame_dictionary[day][cluster], dataCluseteringAndNormalization.day_level_cluster_level_not_normalized_combined_train_DataFrame_dictionary[opeartion_day][cluster])
                    assert_frame_not_equal(dataCluseteringAndNormalization.day_level_cluster_level_not_normalized_combined_test_DataFrame_dictionary[day][cluster], dataCluseteringAndNormalization.day_level_cluster_level_not_normalized_combined_test_DataFrame_dictionary[opeartion_day][cluster])
                    assert_frame_not_equal(dataCluseteringAndNormalization.day_level_cluster_level_not_normalized_combined_DataFrame_dictionary[day][cluster], dataCluseteringAndNormalization.day_level_cluster_level_not_normalized_combined_DataFrame_dictionary[opeartion_day][cluster])
                    assert_series_not_equal(dataCluseteringAndNormalization.day_level_cluster_level_combined_normalization_statistics_dictionary[day][cluster], dataCluseteringAndNormalization.day_level_cluster_level_combined_normalization_statistics_dictionary[opeartion_day][cluster])
                    assert_frame_not_equal(dataCluseteringAndNormalization.day_level_cluster_level_normalized_combined_train_DataFrame_dictionary[day][cluster], dataCluseteringAndNormalization.day_level_cluster_level_normalized_combined_train_DataFrame_dictionary[opeartion_day][cluster])
                    assert_frame_not_equal(dataCluseteringAndNormalization.day_level_cluster_level_normalized_combined_test_DataFrame_dictionary[day][cluster], dataCluseteringAndNormalization.day_level_cluster_level_normalized_combined_test_DataFrame_dictionary[opeartion_day][cluster])
                    assert_frame_not_equal(dataCluseteringAndNormalization.day_level_cluster_level_normalized_combined_DataFrame_dictionary[day][cluster], dataCluseteringAndNormalization.day_level_cluster_level_normalized_combined_DataFrame_dictionary[opeartion_day][cluster])
                    assert_frame_not_equal(dataCluseteringAndNormalization.day_level_cluster_level_medical_center_level_DataFrame_dictionary[day][cluster], dataCluseteringAndNormalization.day_level_cluster_level_medical_center_level_DataFrame_dictionary[opeartion_day][cluster])
                    assert_frame_not_equal(dataCluseteringAndNormalization.day_level_cluster_level_medical_center_level_across_medical_center_normalized_DataFrame_dictionary[day][cluster], dataCluseteringAndNormalization.day_level_cluster_level_medical_center_level_across_medical_center_normalized_DataFrame_dictionary[opeartion_day][cluster])
                else:
                    if not operationIntervalModerator.day_is_operation_day(day):
                        pd.testing.assert_frame_equal( dataCluseteringAndNormalization.day_level_cluster_level_not_normalized_combined_train_DataFrame_dictionary[ day][cluster], dataCluseteringAndNormalization.day_level_cluster_level_not_normalized_combined_train_DataFrame_dictionary[ opeartion_day][cluster])
                        pd.testing.assert_frame_equal( dataCluseteringAndNormalization.day_level_cluster_level_not_normalized_combined_test_DataFrame_dictionary[ day][cluster], dataCluseteringAndNormalization.day_level_cluster_level_not_normalized_combined_test_DataFrame_dictionary[ opeartion_day][cluster])
                        pd.testing.assert_frame_equal( dataCluseteringAndNormalization.day_level_cluster_level_not_normalized_combined_DataFrame_dictionary[ day][cluster], dataCluseteringAndNormalization.day_level_cluster_level_not_normalized_combined_DataFrame_dictionary[ opeartion_day][cluster])
                        pd.testing.assert_series_equal( dataCluseteringAndNormalization.day_level_cluster_level_combined_normalization_statistics_dictionary[ day][cluster], dataCluseteringAndNormalization.day_level_cluster_level_combined_normalization_statistics_dictionary[ opeartion_day][cluster])
                        pd.testing.assert_frame_equal( dataCluseteringAndNormalization.day_level_cluster_level_normalized_combined_train_DataFrame_dictionary[ day][cluster], dataCluseteringAndNormalization.day_level_cluster_level_normalized_combined_train_DataFrame_dictionary[ opeartion_day][cluster])
                        pd.testing.assert_frame_equal( dataCluseteringAndNormalization.day_level_cluster_level_normalized_combined_test_DataFrame_dictionary[ day][cluster], dataCluseteringAndNormalization.day_level_cluster_level_normalized_combined_test_DataFrame_dictionary[ opeartion_day][cluster])
                        pd.testing.assert_frame_equal( dataCluseteringAndNormalization.day_level_cluster_level_normalized_combined_DataFrame_dictionary[ day][cluster], dataCluseteringAndNormalization.day_level_cluster_level_normalized_combined_DataFrame_dictionary[ opeartion_day][cluster])
                        pd.testing.assert_frame_equal( dataCluseteringAndNormalization.day_level_cluster_level_medical_center_level_DataFrame_dictionary[ day][cluster], dataCluseteringAndNormalization.day_level_cluster_level_medical_center_level_DataFrame_dictionary[ opeartion_day][cluster])
                        pd.testing.assert_frame_equal( dataCluseteringAndNormalization.day_level_cluster_level_medical_center_level_across_medical_center_normalized_DataFrame_dictionary[ day][cluster], dataCluseteringAndNormalization.day_level_cluster_level_medical_center_level_across_medical_center_normalized_DataFrame_dictionary[ opeartion_day][cluster])

    def test_no_clustering(self):
        dataset = DataImputationAndAveraging(dataset_path='../../../data/coviddata07292020.csv', number_of_days_for_data_averaging=3).get_processed_dataset()
        dataDelimitingByDayAndMedicalCenter = DataDelimitingByDayAndMedicalCenter(dataset=dataset, number_of_days_for_testing=30, number_of_days_to_predict_ahead=1)
        operation_interval = 5
        dataCluseteringAndNormalization = DataCluseteringAndNormalization(dataDelimitingByDayAndMedicalCenter, number_of_test_day_in_day_level_DataFrame=1, max_number_of_cluster=10, operation_interval=operation_interval, do_clustering=False)

        for day in dataCluseteringAndNormalization.day_level_medical_center_to_cluster_dictionary.keys():
            # for each day, each clinic is a cluster
            list_of_cluster = []
            for clinic in dataCluseteringAndNormalization.day_level_medical_center_to_cluster_dictionary[day].keys():
                list_of_cluster.append(dataCluseteringAndNormalization.day_level_medical_center_to_cluster_dictionary[day][clinic])
            self.assertListEqual(sorted(list_of_cluster), list(range(len(dataCluseteringAndNormalization.day_level_medical_center_to_cluster_dictionary[day].keys()))))

        for day in dataCluseteringAndNormalization.day_level_cluster_level_medical_center_list_dictionary.keys():
            medical_center_list = []
            for cluster in dataCluseteringAndNormalization.day_level_cluster_level_medical_center_list_dictionary[day].keys():
                self.assertEqual(len(dataCluseteringAndNormalization.day_level_cluster_level_medical_center_list_dictionary[day][cluster]), 1)
                medical_center_list += dataCluseteringAndNormalization.day_level_cluster_level_medical_center_list_dictionary[day][cluster]

            self.assertListEqual(sorted(medical_center_list), sorted(dataset.clinic.unique().tolist()))









