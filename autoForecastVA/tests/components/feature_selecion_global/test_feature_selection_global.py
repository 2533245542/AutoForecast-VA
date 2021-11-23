import unittest
import pandas as pd

from autoForecastVA.src.components.data_preprocessing_global.data_clustering_and_normalization import \
    DataCluseteringAndNormalization
from autoForecastVA.src.components.data_preprocessing_global.data_delimiting_by_day_and_medical_center import DataDelimitingByDayAndMedicalCenter
from autoForecastVA.src.components.data_preprocessing_global.data_imputation_and_averaging import \
    DataImputationAndAveraging
from autoForecastVA.src.components.data_preprocessing_global.data_preparation_for_feature_selection import DataPreparationForFeatureSelection
from autoForecastVA.src.components.feature_selection_global.feature_selection_global import FeatureSelectionGlobal
import numpy as np

from autoForecastVA.src.utils.tuning_interval_moderator import OperationIntervalModerator


class TestFeatureSelectionGlobal1(unittest.TestCase):
    def test_day_level_medical_center_level_feature_selection(self):
        '''Creat an original DataFrame and shifted DataFrame, feed them into the function, and check if the index of the significant features are found correctly.'''
        import statsmodels.api as sm
        spector_data = sm.datasets.spector.load(as_pandas=True)
        original_DataFrame = pd.concat([spector_data.raw_data.iloc[:, [-1]], spector_data.raw_data.iloc[:, :-1]], axis=1)
        original_DataFrame.insert(0, 'clinic', '111')
        original_DataFrame.index = pd.date_range(start='2020-01-01', end='2020-02-01', name='date')
        shifted_DataFrame = original_DataFrame.copy()
        featureSelectionGlobal = FeatureSelectionGlobal(dataPreparationForFeatureSelection=None, p_value_threshold=0.05, lazy=True)
        selected_set_of_feature_index_list, total_p_value_list, selected_p_value_list, not_selected_p_value_list = featureSelectionGlobal.day_level_medical_center_level_feature_selection(original_DataFrame, shifted_DataFrame)
        # should be [0]
        self.assertCountEqual(selected_set_of_feature_index_list, [0.0])

    def test_fit_ordinary_linear_regression_and_return_the_feature_with_the_largest_p_value(self):
        '''Create an X and Y, feed to the function and check if the name and p-value of the least significant feature is found correctly.'''

        import numpy as np
        import statsmodels.api as sm
        spector_data = sm.datasets.spector.load(as_pandas=True)
        X = pd.concat([spector_data.raw_data.iloc[:, [-1]], spector_data.raw_data.iloc[:, :-1]], axis=1)
        Y = spector_data.raw_data.iloc[:, [-1]]
        featureSelectionGlobal = FeatureSelectionGlobal(None, None, lazy=True)
        # requires 20 observations to build a regression model for 4 variables (5 variables if counting the constant)
        maximum_p_value_feature_name, maximum_p_value, p_value_list= featureSelectionGlobal.fit_ordinary_linear_regression_and_return_the_feature_with_the_largest_p_value(X, Y)
        # GRADE   GPA  TUCE  PSI -> GRADE

        # p-value should be 0.732, name should be GPA
        self.assertEqual(maximum_p_value_feature_name, 'GPA')
        np.testing.assert_approx_equal(0.732, maximum_p_value, 3)

    def test_medical_center_level_real_example(self):
        '''
        For each day and medical center, the feature index should contain 0.
        '''
        number_of_days_for_testing = 3
        number_of_test_days_in_DataFrame = 1
        number_of_days_to_predict_ahead = 2
        dataset = DataImputationAndAveraging(dataset_path='../../../data/coviddata07292020.csv', number_of_days_for_data_averaging=3, filter_by_medical_center=False, filter_by_time_period=False).get_processed_dataset()
        dataDelimitingByDayAndMedicalCenter = DataDelimitingByDayAndMedicalCenter(dataset, number_of_days_for_testing, number_of_days_to_predict_ahead)
        day_level_medical_center_level_DataFrame_dictionary = dataDelimitingByDayAndMedicalCenter.get_day_level_medical_center_level_DataFrame_dictionary()
        dataPreparationForFeatureSelection = DataPreparationForFeatureSelection(day_level_medical_center_level_DataFrame_dictionary, number_of_test_days_in_DataFrame=number_of_test_days_in_DataFrame, number_of_days_to_predict_ahead=number_of_days_to_predict_ahead)

        """subtest 1: when the threshold is loose, it should include 0 as the selected feature.
        The smallest index in the selected list of feature index should not be negative and the largest should not be greater than the number of features (excluding the hospital code column). The p-value of all features should not equal to -1 (the default value); the p-value of selected features should be <= threshold; the p-value of the non-selected features should be > threshold """

        p_value_threshold = 0.05
        featureSelectionGlobal1 = FeatureSelectionGlobal(dataPreparationForFeatureSelection=dataPreparationForFeatureSelection, p_value_threshold=p_value_threshold, enforce_using_the_prediction_target_as_a_selected_feature=False)
        for day in featureSelectionGlobal1.day_level_agency_level_feature_index_list_dictionary.keys():
            for medical_center in featureSelectionGlobal1.day_level_agency_level_feature_index_list_dictionary[day].keys():
                feature_index_list = featureSelectionGlobal1.day_level_agency_level_feature_index_list_dictionary[day][medical_center]
                self.assertTrue(0 in feature_index_list)
                self.assertGreaterEqual(min(feature_index_list), 0)
                self.assertLessEqual(max(feature_index_list), day_level_medical_center_level_DataFrame_dictionary[day][medical_center].shape[1] - 1)
                self.assertTrue(all([p_value != -1 for p_value in featureSelectionGlobal1.day_level_agency_level_total_p_value_list_dictionary[day][medical_center]]) )
                if any(np.isnan(featureSelectionGlobal1.day_level_agency_level_selected_p_value_list_dictionary[day][medical_center])):  # should have selected all features
                    self.assertLessEqual(len(feature_index_list), 17)
                else:
                    self.assertTrue(all([p_value <= p_value_threshold for p_value in featureSelectionGlobal1.day_level_agency_level_selected_p_value_list_dictionary[day][medical_center]]))
                self.assertTrue(all([p_value > p_value_threshold for p_value in featureSelectionGlobal1.day_level_agency_level_not_selected_p_value_list_dictionary[day][medical_center]]))



        """ subtest 2: when the threshold is very strict, in most of the time, the prediction target itself should be selected. The p-value of all features should not equal to -1 (the default value); the p-value of selected features should be <= threshold; the p-value of the non-selected features should be > threshold """
        p_value_threshold = 0.00000005
        featureSelectionGlobal2 = FeatureSelectionGlobal(dataPreparationForFeatureSelection=dataPreparationForFeatureSelection, p_value_threshold=p_value_threshold, enforce_using_the_prediction_target_as_a_selected_feature=False)
        number_of_feature_index_list = 0
        number_of_times_feature_index_list_does_not_contain_prediction_target = 0
        for day in featureSelectionGlobal2.day_level_agency_level_feature_index_list_dictionary.keys():
            for medical_center in featureSelectionGlobal2.day_level_agency_level_feature_index_list_dictionary[day].keys():
                feature_index_list = featureSelectionGlobal2.day_level_agency_level_feature_index_list_dictionary[day][medical_center]
                if 0 not in feature_index_list:
                    number_of_times_feature_index_list_does_not_contain_prediction_target += 1

                number_of_feature_index_list += 1
                self.assertTrue(all([p_value != -1 for p_value in featureSelectionGlobal2.day_level_agency_level_total_p_value_list_dictionary[day][medical_center]]))
                self.assertTrue(all([p_value <= p_value_threshold for p_value in featureSelectionGlobal2.day_level_agency_level_selected_p_value_list_dictionary[day][medical_center]]))
                self.assertTrue(all([p_value > p_value_threshold for p_value in featureSelectionGlobal2.day_level_agency_level_not_selected_p_value_list_dictionary[day][medical_center]]))

        self.assertLess(number_of_times_feature_index_list_does_not_contain_prediction_target/number_of_feature_index_list, 0.11)  # most of the time the index list should contain the prediction target

        """ subtest 3: when the threshold is very strict and the number of days to predict ahead is very large, sometimes no features would be selected. The p-value of all features should not equal to -1 (the default value); the p-value of selected features should be <= threshold; the p-value of the non-selected features should be > threshold """
        number_of_days_to_predict_ahead = 30
        dataset = DataImputationAndAveraging(dataset_path='../../../data/coviddata07292020.csv', number_of_days_for_data_averaging=3, filter_by_medical_center=False, filter_by_time_period=False).get_processed_dataset()
        dataDelimitingByDayAndMedicalCenter = DataDelimitingByDayAndMedicalCenter(dataset, number_of_days_for_testing, number_of_days_to_predict_ahead)
        day_level_medical_center_level_DataFrame_dictionary = dataDelimitingByDayAndMedicalCenter.get_day_level_medical_center_level_DataFrame_dictionary()
        dataPreparationForFeatureSelection = DataPreparationForFeatureSelection(day_level_medical_center_level_DataFrame_dictionary, number_of_test_days_in_DataFrame=number_of_test_days_in_DataFrame, number_of_days_to_predict_ahead=number_of_days_to_predict_ahead)
        featureSelectionGlobal3 = FeatureSelectionGlobal(dataPreparationForFeatureSelection=dataPreparationForFeatureSelection, p_value_threshold=p_value_threshold, enforce_using_the_prediction_target_as_a_selected_feature=False)
        number_of_times_no_feature_is_selected = 0
        number_of_feature_selection_performed = 0
        for day in featureSelectionGlobal3.day_level_agency_level_feature_index_list_dictionary.keys():
            for medical_center in featureSelectionGlobal3.day_level_agency_level_feature_index_list_dictionary[day].keys():
                feature_index_list = featureSelectionGlobal3.day_level_agency_level_feature_index_list_dictionary[day][medical_center]
                if len(feature_index_list) == 0:
                    number_of_times_no_feature_is_selected += 1
                number_of_feature_selection_performed += 1

                self.assertTrue(all([p_value != -1 for p_value in featureSelectionGlobal3.day_level_agency_level_total_p_value_list_dictionary[day][medical_center]]) )
                self.assertTrue(all([p_value <= p_value_threshold for p_value in featureSelectionGlobal3.day_level_agency_level_selected_p_value_list_dictionary[day][medical_center]]))
                self.assertTrue(all([p_value > p_value_threshold for p_value in featureSelectionGlobal3.day_level_agency_level_not_selected_p_value_list_dictionary[day][medical_center]]))

        self.assertGreater(number_of_times_no_feature_is_selected/number_of_feature_selection_performed, 0.23)


        """ subtest 4: when the threshold is very strict, the number of days to predict ahead is very large and the enforcement is True, sometimes only the 0th feature would be selected. The p-value of all features should not equal to -1 (the default value); the p-value of selected features should be <= threshold; the p-value of the non-selected features should be > threshold """
        number_of_times_no_feature_is_selected = 0
        number_of_feature_selection_performed = 0
        featureSelectionGlobal4 = FeatureSelectionGlobal(dataPreparationForFeatureSelection=dataPreparationForFeatureSelection, p_value_threshold=p_value_threshold, enforce_using_the_prediction_target_as_a_selected_feature=True)
        for day in featureSelectionGlobal4.day_level_agency_level_feature_index_list_dictionary.keys():
            for medical_center in featureSelectionGlobal4.day_level_agency_level_feature_index_list_dictionary[day].keys():
                feature_index_list = featureSelectionGlobal4.day_level_agency_level_feature_index_list_dictionary[day][medical_center]
                if len(feature_index_list) == 1 and 0 in feature_index_list:
                    number_of_times_no_feature_is_selected += 1
                number_of_feature_selection_performed += 1

                self.assertTrue(0 in feature_index_list)
                self.assertTrue(all([p_value != -1 for p_value in featureSelectionGlobal4.day_level_agency_level_total_p_value_list_dictionary[day][medical_center]]) )
                self.assertTrue(all([p_value <= p_value_threshold for p_value in featureSelectionGlobal4.day_level_agency_level_selected_p_value_list_dictionary[day][medical_center]]))
                self.assertTrue(all([p_value > p_value_threshold for p_value in featureSelectionGlobal4.day_level_agency_level_not_selected_p_value_list_dictionary[day][medical_center]]))

        self.assertGreater(number_of_times_no_feature_is_selected / number_of_feature_selection_performed, 0.23)

    def test_cluster_level_real_example(self):
        ''' For each day and cluster, the feature index should contain 0. Note that we only include a portion of hospitals in our dataset in the below tests.'''
        number_of_days_for_testing = 30
        number_of_test_days_in_DataFrame = 1
        number_of_days_to_predict_ahead = 2
        dataset = DataImputationAndAveraging(dataset_path='../../../data/coviddata07292020.csv', number_of_days_for_data_averaging=3, filter_by_medical_center=True, filter_by_time_period=False).get_processed_dataset()
        dataDelimitingByDayAndMedicalCenter = DataDelimitingByDayAndMedicalCenter(dataset, number_of_days_for_testing, number_of_days_to_predict_ahead)
        dataCluseteringAndNormalization = DataCluseteringAndNormalization(dataDelimitingByDayAndMedicalCenter, number_of_test_day_in_day_level_DataFrame=1, max_number_of_cluster=10)
        dataPreparationForFeatureSelection = DataPreparationForFeatureSelection(dataCluseteringAndNormalization.day_level_cluster_level_not_normalized_combined_DataFrame_dictionary, number_of_test_days_in_DataFrame=number_of_test_days_in_DataFrame, number_of_days_to_predict_ahead=number_of_days_to_predict_ahead)

        """subtest 1: when the threshold is loose and using clusters, we should always include 0 as the selected feature. We should have less than 64% features selected on average.
        The smallest index in the selected list of feature index should not be negative and the largest should not be greater than the number of features (excluding the hospital code column). The p-value of all features should not equal to -1 (the default value); the p-value of selected features should be <= threshold; the p-value of the non-selected features should be > threshold """

        p_value_threshold = 0.05
        featureSelectionGlobal1 = FeatureSelectionGlobal(dataPreparationForFeatureSelection=dataPreparationForFeatureSelection, p_value_threshold=p_value_threshold, enforce_using_the_prediction_target_as_a_selected_feature=False)
        number_of_feature_index_list = 0
        number_of_times_feature_index_list_does_not_contain_prediction_target = 0
        number_of_total_features = 0
        number_of_selected_features = 0
        for day in featureSelectionGlobal1.day_level_agency_level_feature_index_list_dictionary.keys():
            for cluster in featureSelectionGlobal1.day_level_agency_level_feature_index_list_dictionary[day].keys():
                feature_index_list = featureSelectionGlobal1.day_level_agency_level_feature_index_list_dictionary[day][cluster]
                self.assertGreaterEqual(min(feature_index_list), 0)
                self.assertLessEqual(max(feature_index_list),  dataCluseteringAndNormalization.day_level_cluster_level_not_normalized_combined_DataFrame_dictionary[day][cluster].shape[1] - 1)

                self.assertTrue(all([p_value != -1 for p_value in featureSelectionGlobal1.day_level_agency_level_total_p_value_list_dictionary[day][cluster]]) )
                if 0 not in feature_index_list:
                    number_of_times_feature_index_list_does_not_contain_prediction_target += 1
                number_of_feature_index_list += 1
                number_of_total_features += len(featureSelectionGlobal1.day_level_agency_level_total_p_value_list_dictionary[day][cluster])
                number_of_selected_features += len(featureSelectionGlobal1.day_level_agency_level_selected_p_value_list_dictionary[day][cluster])
                if any(np.isnan(featureSelectionGlobal1.day_level_agency_level_selected_p_value_list_dictionary[day][cluster])):  # should have selected all features
                    self.assertLessEqual(len(feature_index_list), 17)
                else:
                    self.assertTrue(all([p_value <= p_value_threshold for p_value in featureSelectionGlobal1.day_level_agency_level_selected_p_value_list_dictionary[day][cluster]]))
                self.assertTrue(all([p_value > p_value_threshold for p_value in featureSelectionGlobal1.day_level_agency_level_not_selected_p_value_list_dictionary[day][cluster]]))
        self.assertEqual(number_of_times_feature_index_list_does_not_contain_prediction_target/number_of_feature_index_list, 0.0)  # most of the time the index list should contain the prediction target
        self.assertLessEqual(number_of_selected_features/number_of_total_features, 0.65)


        """ subtest 2: when the threshold is very strict and using cluster, the prediction target itself should always be selected. The percentage of features selected should be less than 0.32. The p-value of all features should not equal to -1 (the default value); the p-value of selected features should be <= threshold; the p-value of the non-selected features should be > threshold """
        p_value_threshold = 0.00000005
        featureSelectionGlobal2 = FeatureSelectionGlobal(dataPreparationForFeatureSelection=dataPreparationForFeatureSelection, p_value_threshold=p_value_threshold, enforce_using_the_prediction_target_as_a_selected_feature=False)
        number_of_feature_index_list = 0
        number_of_times_feature_index_list_does_not_contain_prediction_target = 0
        number_of_total_features = 0
        number_of_selected_features = 0
        for day in featureSelectionGlobal2.day_level_agency_level_feature_index_list_dictionary.keys():
            for cluster in featureSelectionGlobal2.day_level_agency_level_feature_index_list_dictionary[day].keys():
                feature_index_list = featureSelectionGlobal2.day_level_agency_level_feature_index_list_dictionary[day][cluster]
                if 0 not in feature_index_list:
                    number_of_times_feature_index_list_does_not_contain_prediction_target += 1
                number_of_feature_index_list += 1
                number_of_total_features += len(featureSelectionGlobal2.day_level_agency_level_total_p_value_list_dictionary[day][cluster])
                number_of_selected_features += len(featureSelectionGlobal2.day_level_agency_level_selected_p_value_list_dictionary[day][cluster])
                self.assertTrue(all([p_value != -1 for p_value in featureSelectionGlobal2.day_level_agency_level_total_p_value_list_dictionary[day][cluster]]))
                self.assertTrue(all([p_value <= p_value_threshold for p_value in featureSelectionGlobal2.day_level_agency_level_selected_p_value_list_dictionary[day][cluster]]))
                self.assertTrue(all([p_value > p_value_threshold for p_value in featureSelectionGlobal2.day_level_agency_level_not_selected_p_value_list_dictionary[day][cluster]]))

        self.assertEqual(number_of_times_feature_index_list_does_not_contain_prediction_target/number_of_feature_index_list, 0.00)  # most of the time the index list should contain the prediction target
        self.assertLessEqual(number_of_selected_features/number_of_total_features, 0.32)

        """ subtest 3: when the threshold is very strict, using cluster and the number of days to predict ahead is large, the prediction target should always be selected. The percentage of feature selected should be smaller than 0.35. The p-value of all features should not equal to -1 (the default value); the p-value of selected features should be <= threshold; the p-value of the non-selected features should be > threshold """
        number_of_days_to_predict_ahead = 30
        dataset = DataImputationAndAveraging(dataset_path='../../../data/coviddata07292020.csv', number_of_days_for_data_averaging=3, filter_by_medical_center=True, filter_by_time_period=False).get_processed_dataset()
        dataDelimitingByDayAndMedicalCenter = DataDelimitingByDayAndMedicalCenter(dataset, number_of_days_for_testing, number_of_days_to_predict_ahead)
        dataCluseteringAndNormalization = DataCluseteringAndNormalization(dataDelimitingByDayAndMedicalCenter, number_of_test_day_in_day_level_DataFrame=1, max_number_of_cluster=10)
        dataPreparationForFeatureSelection = DataPreparationForFeatureSelection( dataCluseteringAndNormalization.day_level_cluster_level_not_normalized_combined_DataFrame_dictionary, number_of_test_days_in_DataFrame=number_of_test_days_in_DataFrame, number_of_days_to_predict_ahead=number_of_days_to_predict_ahead)
        featureSelectionGlobal3 = FeatureSelectionGlobal(dataPreparationForFeatureSelection=dataPreparationForFeatureSelection, p_value_threshold=p_value_threshold, enforce_using_the_prediction_target_as_a_selected_feature=False)
        number_of_times_no_feature_is_selected = 0
        number_of_feature_selection_performed = 0
        number_of_total_features = 0
        number_of_selected_features = 0
        for day in featureSelectionGlobal3.day_level_agency_level_feature_index_list_dictionary.keys():
            for cluster in featureSelectionGlobal3.day_level_agency_level_feature_index_list_dictionary[day].keys():
                feature_index_list = featureSelectionGlobal3.day_level_agency_level_feature_index_list_dictionary[day][cluster]
                if len(feature_index_list) == 0:
                    number_of_times_no_feature_is_selected += 1
                number_of_feature_selection_performed += 1
                number_of_total_features += len(featureSelectionGlobal3.day_level_agency_level_total_p_value_list_dictionary[day][cluster])
                number_of_selected_features += len(featureSelectionGlobal3.day_level_agency_level_selected_p_value_list_dictionary[day][cluster])

                self.assertTrue(all([p_value != -1 for p_value in featureSelectionGlobal3.day_level_agency_level_total_p_value_list_dictionary[day][cluster]]) )
                self.assertTrue(all([p_value <= p_value_threshold for p_value in featureSelectionGlobal3.day_level_agency_level_selected_p_value_list_dictionary[day][cluster]]))
                self.assertTrue(all([p_value > p_value_threshold for p_value in featureSelectionGlobal3.day_level_agency_level_not_selected_p_value_list_dictionary[day][cluster]]))

        self.assertEqual(number_of_times_no_feature_is_selected/number_of_feature_selection_performed, 0.0)
        self.assertLessEqual(number_of_selected_features/number_of_total_features, 0.35)


        """ subtest 4: when the threshold is very strict, the number of days to predict ahead is very large and the enforcement is True, the prediction target would always be selected. The percertage of features selected should be smaller than 0.35. The p-value of all features should not equal to -1 (the default value); the p-value of selected features should be <= threshold; the p-value of the non-selected features should be > threshold """
        number_of_times_no_feature_is_selected = 0
        number_of_feature_selection_performed = 0
        number_of_total_features = 0
        number_of_selected_features = 0
        featureSelectionGlobal4 = FeatureSelectionGlobal(dataPreparationForFeatureSelection=dataPreparationForFeatureSelection, p_value_threshold=p_value_threshold, enforce_using_the_prediction_target_as_a_selected_feature=True)
        for day in featureSelectionGlobal4.day_level_agency_level_feature_index_list_dictionary.keys():
            for cluster in featureSelectionGlobal4.day_level_agency_level_feature_index_list_dictionary[day].keys():
                feature_index_list = featureSelectionGlobal4.day_level_agency_level_feature_index_list_dictionary[day][cluster]
                if len(feature_index_list) == 1 and 0 in feature_index_list:
                    number_of_times_no_feature_is_selected += 1
                number_of_feature_selection_performed += 1
                number_of_total_features += len(featureSelectionGlobal3.day_level_agency_level_total_p_value_list_dictionary[day][cluster])
                number_of_selected_features += len(featureSelectionGlobal3.day_level_agency_level_selected_p_value_list_dictionary[day][cluster])
                self.assertTrue(0 in feature_index_list)
                self.assertTrue(all([p_value != -1 for p_value in featureSelectionGlobal4.day_level_agency_level_total_p_value_list_dictionary[day][cluster]]) )
                self.assertTrue(all([p_value <= p_value_threshold for p_value in featureSelectionGlobal4.day_level_agency_level_selected_p_value_list_dictionary[day][cluster]]))
                self.assertTrue(all([p_value > p_value_threshold for p_value in featureSelectionGlobal4.day_level_agency_level_not_selected_p_value_list_dictionary[day][cluster]]))

        self.assertEqual(number_of_times_no_feature_is_selected / number_of_feature_selection_performed, 0.0)
        self.assertLessEqual(number_of_selected_features/number_of_total_features, 0.35)

    def test_cluster_level_real_example_tuning_interval(self):
        ''' For each day and cluster, the feature index should contain 0. Note that we only include a portion of hospitals in our dataset in the below tests.'''
        number_of_days_for_testing = 30
        number_of_test_days_in_DataFrame = 1
        number_of_days_to_predict_ahead = 2
        dataset = DataImputationAndAveraging(dataset_path='../../../data/coviddata07292020.csv', number_of_days_for_data_averaging=3, filter_by_medical_center=True, filter_by_time_period=False).get_processed_dataset()
        dataDelimitingByDayAndMedicalCenter = DataDelimitingByDayAndMedicalCenter(dataset, number_of_days_for_testing, number_of_days_to_predict_ahead)
        dataCluseteringAndNormalization = DataCluseteringAndNormalization(dataDelimitingByDayAndMedicalCenter, number_of_test_day_in_day_level_DataFrame=1, max_number_of_cluster=10)
        dataPreparationForFeatureSelection = DataPreparationForFeatureSelection(dataCluseteringAndNormalization.day_level_cluster_level_not_normalized_combined_DataFrame_dictionary, number_of_test_days_in_DataFrame=number_of_test_days_in_DataFrame, number_of_days_to_predict_ahead=number_of_days_to_predict_ahead, )
        operation_interval = 5
        featureSelectionGlobal = FeatureSelectionGlobal(dataPreparationForFeatureSelection=dataPreparationForFeatureSelection, p_value_threshold=0.05, operation_interval=operation_interval)
        operationIntervalModerator = OperationIntervalModerator(days=featureSelectionGlobal.day_level_agency_level_feature_index_list_dictionary.keys(), operation_interval=operation_interval)

        for day in featureSelectionGlobal.day_level_agency_level_feature_index_list_dictionary.keys():
            operation_day = operationIntervalModerator.get_operation_day_of_a_day(day)
            for cluster in featureSelectionGlobal.day_level_agency_level_feature_index_list_dictionary[day].keys():
                print(day, operation_day)
                self.assertListEqual(featureSelectionGlobal.day_level_agency_level_feature_index_list_dictionary[day][cluster], featureSelectionGlobal.day_level_agency_level_feature_index_list_dictionary[operation_day][cluster])
                self.assertListEqual(featureSelectionGlobal.day_level_agency_level_selected_p_value_list_dictionary[day][cluster], featureSelectionGlobal.day_level_agency_level_selected_p_value_list_dictionary[operation_day][cluster])
                self.assertListEqual(featureSelectionGlobal.day_level_agency_level_not_selected_p_value_list_dictionary[day][cluster], featureSelectionGlobal.day_level_agency_level_not_selected_p_value_list_dictionary[operation_day][cluster])
                self.assertListEqual(featureSelectionGlobal.day_level_agency_level_total_p_value_list_dictionary[day][cluster], featureSelectionGlobal.day_level_agency_level_total_p_value_list_dictionary[operation_day][cluster])

    def test_cluster_level_real_example_no_feature_selection(self):
        ''' For each day and cluster, the feature index should contain 0. Note that we only include a portion of hospitals in our dataset in the below tests.'''
        number_of_days_for_testing = 30
        number_of_test_days_in_DataFrame = 1
        number_of_days_to_predict_ahead = 2
        dataset = DataImputationAndAveraging(dataset_path='../../../data/coviddata07292020.csv', number_of_days_for_data_averaging=3, filter_by_medical_center=True, filter_by_time_period=False).get_processed_dataset()
        dataDelimitingByDayAndMedicalCenter = DataDelimitingByDayAndMedicalCenter(dataset, number_of_days_for_testing, number_of_days_to_predict_ahead)
        dataCluseteringAndNormalization = DataCluseteringAndNormalization(dataDelimitingByDayAndMedicalCenter, number_of_test_day_in_day_level_DataFrame=1, max_number_of_cluster=10)
        dataPreparationForFeatureSelection = DataPreparationForFeatureSelection( dataCluseteringAndNormalization.day_level_cluster_level_not_normalized_combined_DataFrame_dictionary, number_of_test_days_in_DataFrame=number_of_test_days_in_DataFrame, number_of_days_to_predict_ahead=number_of_days_to_predict_ahead, )
        operation_interval = 5
        featureSelectionGlobal = FeatureSelectionGlobal(dataPreparationForFeatureSelection=dataPreparationForFeatureSelection, p_value_threshold=0.05, operation_interval=operation_interval, do_feature_selection=False)

        for day in featureSelectionGlobal.day_level_agency_level_feature_index_list_dictionary.keys():
            for cluster in featureSelectionGlobal.day_level_agency_level_feature_index_list_dictionary[day].keys():
                print(featureSelectionGlobal.day_level_agency_level_feature_index_list_dictionary[day][cluster])
                print(featureSelectionGlobal.day_level_agency_level_not_selected_p_value_list_dictionary[day][cluster])
                self.assertListEqual(featureSelectionGlobal.day_level_agency_level_feature_index_list_dictionary[day][cluster], list(range(len(dataset.columns) - 1)))