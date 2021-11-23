import pandas as pd
import numpy as np
'''
data_imputation_and_averaging.py
This file implements the "Data imputation and data averaging" part of the "Data preprocessing component" as described in the system design document.
This part reads the dataset with a given the path, imputes the missing values, and performs data averaging with on a given number of days.
We also set all numeric data to have the data type of float64

It returns the processed dataset in `DataFrame` format.
'''
class DataImputationAndAveraging():
    def __init__(self, dataset_path, number_of_days_for_data_averaging, lazy=False, filter_by_medical_center=False, medical_center_subset=['608', '621', '537', '549', '554', '663', '519', '521', '538', '509', '619'], filter_by_time_period=False, time_period=['2020-3-1', '2020-5-1']):
        '''
        :param dataset_path: The path (relative to the path of this file) to find the dataset.
        :param number_of_days_for_data_averaging: The number of days to perform forward averaging; the number_of_days_for_data_averaging days at the beginning of each medical center will be discarded.
        :param lazy: This class is design to process data as soon as an instance of this class is created, but for the purpose of testing, we can wait untill it is explicitly called to process data when lazy=True.
        :param filter_by_medical_center: This filters the dataset by the subset of medical centers during dataset reading.

        '''
        self.dataset_path = dataset_path
        self.number_of_days_for_data_averaging = number_of_days_for_data_averaging
        self.filter_by_medical_center = filter_by_medical_center
        self.medical_center_subset = medical_center_subset
        self.filter_by_time_period = filter_by_time_period
        self.time_period = time_period

        self.original_dataset = None
        self.processed_dataset = None

        if lazy == False:
            self.read_dataset()
            self.impute_dataset()
            self.average_dataset()

    def read_dataset(self):
        '''Reads the dataset, set up the date index for the pandas DataFrame, but does not add the two features generated from the date index.
        Overwrite this function if you have a different dataset.
        The requirement is that the index should be continous days, the second column should be named clinic, and the third should be named case.
        '''
        df = pd.read_csv(self.dataset_path)
        df = df.iloc[:, :-1]  # no hospital location name
        df.columns = ['date', 'clinic', 'case', 'call', 'par_sta_sdi', 'stayinghome', 'tripsperson', 'outofcountytrips', 'milesperson', 'nonworktripsperson', 'par_sta_cases', 'par_sta_pop', 'par_sta_case_rt', 'par_sta_test_rt', 'par_sta_tests', 'par_sta_pos_rt', 'hospitalcovid']
        def convert(s):
            return s + '-2020'

        if self.filter_by_medical_center:
            df = df[df.clinic.isin(self.medical_center_subset)]

        df['date'] = pd.to_datetime(df.date.apply(convert).apply(str), infer_datetime_format=True)
        df = df.set_index(['date'])

        if self.filter_by_time_period:
            df = df.loc[self.time_period[0]:self.time_period[1]]

        self.processed_dataset = df

    def impute_dataset(self):
        '''Impute with forward fill and backward fill'''
        def resampleFillNa(df):
            return df.resample('D').asfreq().fillna(method='ffill').fillna(method='bfill')

        dfSite = self.processed_dataset.groupby('clinic', group_keys=False, as_index=False).apply(resampleFillNa)

        self.processed_dataset = dfSite

    def average_dataset(self):
        '''For each medical center, performs the moving average. Convert the data types of dayofweek and weekofyear from int to float'''

        def per_medical_center_moving_averge(df):
            clinic = df.clinic.tolist()[0]
            df = df.rolling(window=self.number_of_days_for_data_averaging).mean().dropna()
            df['clinic'] = clinic
            return df

        self.processed_dataset = self.processed_dataset.groupby('clinic', as_index=False, group_keys=False).apply(per_medical_center_moving_averge)
        self.processed_dataset['dayofweek'] = self.processed_dataset.index.dayofweek.astype(np.float64)
        self.processed_dataset['weekofyear'] = self.processed_dataset.index.isocalendar().week.astype(np.float64)

    def get_original_dataset(self):
        return self.original_dataset
    def get_processed_dataset(self):
        return self.processed_dataset