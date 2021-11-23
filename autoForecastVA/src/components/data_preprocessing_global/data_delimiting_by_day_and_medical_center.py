import pandas as pd
'''
Given the dataset, number of days used for testing and the number of days to predict ahead, we create the amount of DataFrames equal to the number of days for testing times the number of medical centers. The created DataFrames are stored in two nested dictionaries and can be query by the code of the medical center and the prediction day.
'''
class DataDelimitingByDayAndMedicalCenter():
    def __init__(self, dataset, number_of_days_for_testing, number_of_days_to_predict_ahead, lazy=False):
        self.dataset = dataset
        self.number_of_days_for_testing = number_of_days_for_testing
        self.number_of_days_to_predict_ahead = number_of_days_to_predict_ahead
        self.medical_center_level_DataFrame_dictionary = {}  # a complete DataFrame containing data of all days
        self.medical_center_level_day_level_DataFrame_dictionary = {}
        self.day_level_medical_center_level_DataFrame_dictionary = {}

        if not lazy:
            self.split_by_medical_center()
            self.split_by_prediction_day()

    def split_by_medical_center(self):
        '''
        Create a dictionary of DataFrames where each DataFrame consists of all data for a medical center.
        '''
        for medical_center in self.dataset.clinic.unique():
            self.medical_center_level_DataFrame_dictionary[medical_center] = self.dataset[self.dataset.clinic == medical_center]

    def split_by_prediction_day(self):
        '''
        Creat a dictionary for each medical that maps a date to a DataFrame. The dictionary of each medial center is stored in another dictionary.
        '''
        for medical_center in self.dataset.clinic.unique():
            day_to_day_level_DataFrame_dictionary = {}  # the day-level DataFrame for all days and for one medical center
            for day in self.dataset.index.unique()[-self.number_of_days_for_testing:]:
                # data of the prediction day
                data_of_prediction_day = self.dataset[self.dataset.clinic == medical_center][day - pd.Timedelta(1, unit='d'):day].iloc[-1:,:]  # code hack to get the data for the prediction day
                # data of the rest
                data_of_previous_days = self.dataset[self.dataset.clinic == medical_center][:day - pd.Timedelta(self.number_of_days_to_predict_ahead, unit='d')]
                # form the day-level DataFrame
                day_level_DataFrame = pd.concat([data_of_previous_days, data_of_prediction_day])
                day_to_day_level_DataFrame_dictionary[day] = day_level_DataFrame
            self.medical_center_level_day_level_DataFrame_dictionary[medical_center] = day_to_day_level_DataFrame_dictionary
        # populate day_level_DataFrame_dictionary which is the reshaping of medical_center_day_level_DataFrame_dictionary
        for day in self.dataset.index.unique()[-self.number_of_days_for_testing:]:
            medical_center_to_day_level_DataFrame_dictionary = {}
            for medical_center in self.dataset.clinic.unique():
                medical_center_to_day_level_DataFrame_dictionary[medical_center] = self.medical_center_level_day_level_DataFrame_dictionary[medical_center][day]  # the day-level DataFrame for all days and for one medical center
            self.day_level_medical_center_level_DataFrame_dictionary[day] = medical_center_to_day_level_DataFrame_dictionary

    def get_medical_center_level_DataFrame_dictionary(self):
        return self.medical_center_level_DataFrame_dictionary

    def get_medical_center_level_day_level_DataFrame_dictionary(self):
        return self.medical_center_level_day_level_DataFrame_dictionary
    def get_day_level_medical_center_level_DataFrame_dictionary(self):
        return self.day_level_medical_center_level_DataFrame_dictionary
