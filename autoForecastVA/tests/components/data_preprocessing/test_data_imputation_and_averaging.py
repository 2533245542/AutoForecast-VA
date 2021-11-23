from autoForecastVA.src.components.data_preprocessing_global.data_imputation_and_averaging import DataImputationAndAveraging
import unittest
import pandas as pd
import numpy as np
from pandas.api.types import is_numeric_dtype
from pandas.api.types import is_dtype_equal
from pandas.api.types import is_string_dtype


class TestDataImputationAndAveraging(unittest.TestCase):

    def test_data_imputation_and_averaging(self):
        number_of_days_for_data_averaging = 3
        dataset_path = '../../../data/coviddata07292020.csv'

        dataImputationAndAveraging = DataImputationAndAveraging(dataset_path=dataset_path, number_of_days_for_data_averaging=number_of_days_for_data_averaging, lazy=True)
        ''' This should read the dataset and the dataset should 16 columns and 29260 rows. It also contains 140 medical centers. The COVID-19 related calls feature contains 701 NA values'''
        dataImputationAndAveraging.read_dataset()
        self.assertTupleEqual(dataImputationAndAveraging.get_processed_dataset().shape, (29260, 16))
        self.assertEqual(len(dataImputationAndAveraging.get_processed_dataset().clinic.unique()), 140)
        self.assertEqual(dataImputationAndAveraging.get_processed_dataset().case.isna().sum(), 701)

        ''' After imputation, there should be no missing numbers'''
        dataImputationAndAveraging.impute_dataset()
        self.assertEqual(dataImputationAndAveraging.get_processed_dataset().case.isna().sum(), 0)
        self.assertEqual(dataImputationAndAveraging.get_processed_dataset().shape[1], 16)

        ''' After averaging, each medical center should only contain data for 207 days.
        The first column should be of string type, and the rest should be float64.
        And it has two more date-related features (day of week, week of year) generated from the date index'''
        dataImputationAndAveraging.average_dataset()
        for clinic in dataImputationAndAveraging.get_processed_dataset().clinic.unique().tolist():
            self.assertEqual(dataImputationAndAveraging.get_processed_dataset()[dataImputationAndAveraging.get_processed_dataset().clinic == clinic].shape[0], 207)

        for i in range(len(dataImputationAndAveraging.get_processed_dataset().columns)):
            if i == 0:
                self.assertTrue(is_string_dtype(dataImputationAndAveraging.get_processed_dataset().iloc[:, i]))
            else:
                self.assertTrue(is_dtype_equal(np.float64, dataImputationAndAveraging.get_processed_dataset().iloc[:, i]))

        self.assertEqual(dataImputationAndAveraging.get_processed_dataset().shape[1], 18)

        '''The processed_dataset retreived from the the non-lazy initiation should have the same result as that for lazy loading
        We compare the difference by the number of rows and columns, number of medical centers, existence of missing values, and data.
        '''
        non_lazy_dataImputationAndAveraging = DataImputationAndAveraging(dataset_path=dataset_path, number_of_days_for_data_averaging=number_of_days_for_data_averaging, lazy=False)
        self.assertTupleEqual(dataImputationAndAveraging.get_processed_dataset().shape, non_lazy_dataImputationAndAveraging.get_processed_dataset().shape)
        self.assertEqual(len(dataImputationAndAveraging.get_processed_dataset().clinic.unique()), len(non_lazy_dataImputationAndAveraging.get_processed_dataset().clinic.unique()))
        self.assertEqual(dataImputationAndAveraging.get_processed_dataset().case.isna().sum(), non_lazy_dataImputationAndAveraging.get_processed_dataset().case.isna().sum())
        self.assertEqual(dataImputationAndAveraging.get_processed_dataset().iloc[3441, 4], non_lazy_dataImputationAndAveraging.get_processed_dataset().iloc[3441, 4])
        self.assertEqual(dataImputationAndAveraging.get_processed_dataset().iloc[4093, 0], non_lazy_dataImputationAndAveraging.get_processed_dataset().iloc[4093, 0])
        self.assertEqual(dataImputationAndAveraging.get_processed_dataset().iloc[12330, 12], non_lazy_dataImputationAndAveraging.get_processed_dataset().iloc[12330, 12])

        '''With medical center filtering, dataset should only contain 11 medical centers. And the exact same medical centers'''
        medical_center_filtered_non_lazy_dataImputationAndAveraging = DataImputationAndAveraging(dataset_path=dataset_path, number_of_days_for_data_averaging=number_of_days_for_data_averaging, filter_by_medical_center=True)
        self.assertEqual(len(medical_center_filtered_non_lazy_dataImputationAndAveraging.get_processed_dataset().clinic.unique()), 11)
        self.assertTupleEqual(medical_center_filtered_non_lazy_dataImputationAndAveraging.get_processed_dataset().shape, (2277,18))
        self.assertCountEqual(medical_center_filtered_non_lazy_dataImputationAndAveraging.get_processed_dataset().clinic.unique().tolist(), ['608', '621', '537', '549', '554', '663', '519', '521', '538', '509', '619'])

        '''With time period filtering, dataset should only contain (62 - number of days of to average + 1) days for each medical center.'''
        time_period_filtered_non_lazy_dataImputationAndAveraging = DataImputationAndAveraging(dataset_path=dataset_path, number_of_days_for_data_averaging=number_of_days_for_data_averaging, filter_by_time_period=True)
        for medical_center in time_period_filtered_non_lazy_dataImputationAndAveraging.get_processed_dataset().clinic.unique():
            self.assertTupleEqual(time_period_filtered_non_lazy_dataImputationAndAveraging.get_processed_dataset()[time_period_filtered_non_lazy_dataImputationAndAveraging.get_processed_dataset().clinic == medical_center].shape, (62-number_of_days_for_data_averaging+1, 18))

        '''With both medical center and time filtering, dataset should only contain 11 medical centers, the exact same medical center, and for each medical center, the shape is fixed'''
        medical_center_and_time_period_filtered_non_lazy_dataImputationAndAveraging = DataImputationAndAveraging(dataset_path=dataset_path, number_of_days_for_data_averaging=number_of_days_for_data_averaging, filter_by_medical_center=True, filter_by_time_period=True)
        self.assertEqual(len(medical_center_and_time_period_filtered_non_lazy_dataImputationAndAveraging.get_processed_dataset().clinic.unique()), 11)
        self.assertCountEqual(medical_center_and_time_period_filtered_non_lazy_dataImputationAndAveraging.get_processed_dataset().clinic.unique().tolist(), ['608', '621', '537', '549', '554', '663', '519', '521', '538', '509', '619'])
        for medical_center in ['608', '621', '537', '549', '554', '663', '519', '521', '538', '509', '619']:
            self.assertTupleEqual(medical_center_and_time_period_filtered_non_lazy_dataImputationAndAveraging.get_processed_dataset()[medical_center_and_time_period_filtered_non_lazy_dataImputationAndAveraging.get_processed_dataset().clinic == medical_center].shape, (62-number_of_days_for_data_averaging+1, 18))