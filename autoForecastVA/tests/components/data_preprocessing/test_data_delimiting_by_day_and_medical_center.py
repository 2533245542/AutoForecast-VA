from autoForecastVA.src.components.data_preprocessing_global.data_imputation_and_averaging import DataImputationAndAveraging
from autoForecastVA.src.components.data_preprocessing_global.data_delimiting_by_day_and_medical_center import DataDelimitingByDayAndMedicalCenter
import unittest
import pandas as pd
from pandas.api.types import is_numeric_dtype
from pandas.api.types import is_string_dtype

class TestDataDelimitingByDayAndMedicalCenter(unittest.TestCase):

    def test_data_delimiting_by_day_and_medical_center(self):
        '''
        For each day and each medical center, the data accessed by the dictionary should contain the amount of observations equaling to the prediction day minus the number of days to predict ahead and then minus the first day and then plus one.
        The last day in the dataset should be the test day.
        The last day and the second to the last day have a day difference of the number of days to predict ahead.
        '''

        number_of_days_for_testing = 30
        number_of_days_to_predict_ahead = 3

        dataset = DataImputationAndAveraging(dataset_path='../../../data/coviddata07292020.csv', number_of_days_for_data_averaging=3).get_processed_dataset()
        dataDelimitingByDayAndMedicalCenter = DataDelimitingByDayAndMedicalCenter(dataset, number_of_days_for_testing, number_of_days_to_predict_ahead)
        medical_center_level_DataFrame_dictionary = dataDelimitingByDayAndMedicalCenter.get_medical_center_level_DataFrame_dictionary()
        medical_center_level_day_level_DataFrame_dictionary = dataDelimitingByDayAndMedicalCenter.get_medical_center_level_day_level_DataFrame_dictionary()
        first_day = dataset.index.unique().sort_values()[0]
        for medical_center in dataset.clinic.unique():
            self.assertTupleEqual(medical_center_level_DataFrame_dictionary[medical_center].shape, (207, 18))
            for day in dataset.index.unique()[-number_of_days_for_testing:]:
                # medical_center_level_day_level_DataFrame_dictionary[medical_center][day+pd.Timedelta(3, unit='d')].shape[0]
                # e.g. (Tuesday - Monday).days = 1
                self.assertEqual(medical_center_level_day_level_DataFrame_dictionary[medical_center][day].shape[0], (day - first_day).days + 1 - number_of_days_to_predict_ahead + 1)  # tip: when calculating the number of days between two date, calculate it as (date1 - date2 + 1)
                self.assertEqual((medical_center_level_day_level_DataFrame_dictionary[medical_center][day].index[-1] - first_day).days + 1, (day - first_day).days + 1)
                self.assertEqual((medical_center_level_day_level_DataFrame_dictionary[medical_center][day].index[-1] - medical_center_level_day_level_DataFrame_dictionary[medical_center][day].index[-2]).days, number_of_days_to_predict_ahead)

# site 358 has 177
                # 207 - 30 = 177 is the first prediction day, while for the training data, it should be 1 to 177-3=174. Total number of days should be 175 days