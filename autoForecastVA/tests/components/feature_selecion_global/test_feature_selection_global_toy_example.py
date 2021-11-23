import unittest
import pandas as pd
from autoForecastVA.src.components.data_preprocessing_global.data_delimiting_by_day_and_medical_center import DataDelimitingByDayAndMedicalCenter
from autoForecastVA.src.components.data_preprocessing_global.data_imputation_and_averaging import \
    DataImputationAndAveraging
from autoForecastVA.src.components.data_preprocessing_global.data_preparation_for_feature_selection import DataPreparationForFeatureSelection
from autoForecastVA.src.components.feature_selection_global.feature_selection_global import FeatureSelectionGlobal
import numpy as np
"""
We purposely make a standalone toy example test because sometimes, when toy example is run with other tests together, python will mark the toy example as failed even when it succeed when runing individually.
"""

class TestFeatureSelectionGlobalToyExample(unittest.TestCase):
    def test_toy_example(self):
        '''
        We will create three days of dataframe. For each day, the significant features should only be case, call and dayofweek; weekofyear and random should be insiginificant.
        case:0, call:1, dayofweek:2, weekofyear:3, random:4
        The p-value of all features should not equal to -1 (the default value); the p-value of selected features should be <= threshold; the p-value of the non-selected features should be > threshold
        '''
        number_of_days_for_testing = 3
        number_of_test_days_in_DataFrame = 1
        number_of_days_to_predict_ahead = 2
        p_value_threshold = 0.05

        # convinient procedure of creating a DataFrame
        date = pd.date_range(start='2020-01-01', end='2020-01-30', name='date')
        case = pd.Series(list(range(1, 31)))
        call = case * 3 + 10 + np.random.normal(0, 1, 30)
        dayofweek = date.dayofweek
        weekofyear = date.isocalendar().week
        random = np.random.normal(10, 3, 30)
        precursor_dataset = pd.DataFrame({'case': case.values.astype(np.float64), 'call': call.values.astype(np.float64), 'dayofweek': dayofweek.astype(np.float64), 'weekofyear': weekofyear.astype(np.float64), 'random': random.astype(np.float64)}, index=date)
        dataset_111 = precursor_dataset.copy()
        dataset_111.insert(0, 'clinic', '111')
        dataset_222 = precursor_dataset.copy()
        dataset_222.insert(0, 'clinic', '222')
        dataset_333 = precursor_dataset.copy()
        dataset_333.insert(0, 'clinic', '333')
        dataset = pd.concat([dataset_111, dataset_222, dataset_333], axis=0)

        dataDelimitingByDayAndMedicalCenter = DataDelimitingByDayAndMedicalCenter(dataset, number_of_days_for_testing, number_of_days_to_predict_ahead)
        day_level_medical_center_level_DataFrame_dictionary = dataDelimitingByDayAndMedicalCenter.get_day_level_medical_center_level_DataFrame_dictionary()

        dataPreparationForFeatureSelection = DataPreparationForFeatureSelection(day_level_medical_center_level_DataFrame_dictionary, number_of_test_days_in_DataFrame=number_of_test_days_in_DataFrame, number_of_days_to_predict_ahead=number_of_days_to_predict_ahead)
        featureSelectionGlobal = FeatureSelectionGlobal(dataPreparationForFeatureSelection=dataPreparationForFeatureSelection, p_value_threshold=p_value_threshold)
        day_level_medical_center_level_feature_index_list_dictionary = featureSelectionGlobal.day_level_agency_level_feature_index_list_dictionary
        for day in day_level_medical_center_level_feature_index_list_dictionary.keys():
            for medical_center in day_level_medical_center_level_feature_index_list_dictionary[day].keys():
                feature_index_list = day_level_medical_center_level_feature_index_list_dictionary[day][medical_center]
                self.assertTrue(4 not in feature_index_list)  # the random feature
                self.assertTrue(0 in feature_index_list)  # the prediction target
                self.assertTrue(all([p_value != -1 for p_value in featureSelectionGlobal.day_level_agency_level_total_p_value_list_dictionary[day][medical_center]]) )
                self.assertTrue(all([p_value <= p_value_threshold for p_value in featureSelectionGlobal.day_level_agency_level_selected_p_value_list_dictionary[day][medical_center]]))
                self.assertTrue(all([p_value > p_value_threshold for p_value in featureSelectionGlobal.day_level_agency_level_not_selected_p_value_list_dictionary[day][medical_center]]))


