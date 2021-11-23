import pandas as pd

class DataWindowing():
    def __init__(self, day_level_cluster_level_medical_center_level_across_medical_center_normalized_DataFrame_dictionary, option=1, number_of_days_to_predict_ahead=2, number_of_testing_day_in_a_day_level_DataFrame=1, input_length=2, lstm_is_many_to_many=False, lazy=False):
        '''
        Convert DataFrames to supervised datasets  applicable for training deep learning algorithms.
        For each day, cluster and medical center, create a list of DataFrames.

        By default, many_to_many is False which means we are windowing data for a many-to-one LSTM -- one input sequence corresponds to one output;
        A DataFrame is converted to a list of shorter DataFrames (where the last row is the prediction target).

        When many_to_many is True, we are windowing data for a many-to-many LSTM -- one input sequence corresponds to one output sequence.
        A DataFrame is converted to a list of shorter DataFrames (where the former half of the rows are input and the later half are prediction targets). Note that test DataFrames is the same as the many_to_one case and the last row is the prediction target.
        '''

        self.day_level_cluster_level_medical_center_level_across_medical_center_normalized_DataFrame_dictionary = day_level_cluster_level_medical_center_level_across_medical_center_normalized_DataFrame_dictionary
        self.option = option  # 1 is using all previous data, 2 is creating fixed length sequences
        self.number_of_days_to_predict_ahead = number_of_days_to_predict_ahead
        self.number_of_testing_day_in_a_day_level_DataFrame = number_of_testing_day_in_a_day_level_DataFrame  # the number of input-output DataFrames for testing
        self.input_length = input_length
        self.many_to_many = lstm_is_many_to_many
        self.day_level_cluster_level_medical_center_level_input_output_DataFrame_list_dictionary = {}
        self.day_level_cluster_level_medical_center_level_train_input_output_DataFrame_list_dictionary = {}
        self.day_level_cluster_level_medical_center_level_test_input_output_DataFrame_list_dictionary = {}

        if not lazy:
            for day in day_level_cluster_level_medical_center_level_across_medical_center_normalized_DataFrame_dictionary.keys():
                cluster_level_medical_center_level_input_output_DataFrame_list_dictionary = {}
                cluster_level_medical_center_level_train_input_output_DataFrame_list_dictionary = {}
                cluster_level_medical_center_level_test_input_output_DataFrame_list_dictionary = {}
                for cluster in day_level_cluster_level_medical_center_level_across_medical_center_normalized_DataFrame_dictionary[day].keys():
                    medical_center_level_input_output_DataFrame_list_dictionary = {}
                    medical_center_level_train_input_output_DataFrame_list_dictionary = {}
                    medical_center_level_test_input_output_DataFrame_list_dictionary = {}
                    for medical_center in day_level_cluster_level_medical_center_level_across_medical_center_normalized_DataFrame_dictionary[day][cluster].keys():
                        DataFrame = day_level_cluster_level_medical_center_level_across_medical_center_normalized_DataFrame_dictionary[day][cluster][medical_center]
                        train_input_output_DataFrame_list, test_input_output_DataFrame_list  = self.create_input_output_DataFrames(DataFrame)
                        input_output_DataFrames_list = train_input_output_DataFrame_list + test_input_output_DataFrame_list
                        medical_center_level_input_output_DataFrame_list_dictionary[medical_center] = input_output_DataFrames_list
                        medical_center_level_train_input_output_DataFrame_list_dictionary[medical_center] = train_input_output_DataFrame_list
                        medical_center_level_test_input_output_DataFrame_list_dictionary[medical_center] = test_input_output_DataFrame_list
                    cluster_level_medical_center_level_input_output_DataFrame_list_dictionary[cluster] = medical_center_level_input_output_DataFrame_list_dictionary
                    cluster_level_medical_center_level_train_input_output_DataFrame_list_dictionary[cluster] = medical_center_level_train_input_output_DataFrame_list_dictionary
                    cluster_level_medical_center_level_test_input_output_DataFrame_list_dictionary[cluster] = medical_center_level_test_input_output_DataFrame_list_dictionary
                self.day_level_cluster_level_medical_center_level_input_output_DataFrame_list_dictionary[day] = cluster_level_medical_center_level_input_output_DataFrame_list_dictionary
                self.day_level_cluster_level_medical_center_level_train_input_output_DataFrame_list_dictionary[day] = cluster_level_medical_center_level_train_input_output_DataFrame_list_dictionary
                self.day_level_cluster_level_medical_center_level_test_input_output_DataFrame_list_dictionary[day] = cluster_level_medical_center_level_test_input_output_DataFrame_list_dictionary

    def create_input_output_DataFrames(self, DataFrame):
        '''From a DataFrame, create a list of input-output DataFrames (where the last row is the prediction target) for training, and a list of input-output DataFrames for testing. The size of the list of input-output DataFrames for testing equals to self.number_of_testing_day_in_a_day_level_DataFrame.'''

        first_day = DataFrame.index[0]
        train_input_output_DataFrame_list = []
        test_input_output_DataFrame_list = []

        # first handle the testing day which we do not need to offset number_of_days_to_predict_ahead because it has been done in data delimiating
        for day in DataFrame.index[-self.number_of_testing_day_in_a_day_level_DataFrame:]:
            output_DataFrame = DataFrame.loc[[day]]
            # option 1: all previous data
            if self.option == 1:
                if day - pd.Timedelta(self.number_of_days_to_predict_ahead, unit='d') in DataFrame.index:
                    input_DataFrame = DataFrame[first_day: day - pd.Timedelta(self.number_of_days_to_predict_ahead, unit='d')]
                else:
                    continue
            elif self.option == 2:
                # option 2: fixed length data. Both head must be within index
                if day - pd.Timedelta(self.number_of_days_to_predict_ahead + self.input_length - 1, unit='d') in DataFrame.index and day - pd.Timedelta(self.number_of_days_to_predict_ahead, unit='d') in DataFrame.index:
                    input_DataFrame = DataFrame[day - pd.Timedelta(self.number_of_days_to_predict_ahead + self.input_length - 1, unit='d'): day - pd.Timedelta(self.number_of_days_to_predict_ahead, unit='d')]
                else:
                    continue
            else:
                raise ValueError
            input_output_DataFrame = pd.concat([input_DataFrame, output_DataFrame])
            test_input_output_DataFrame_list.append(input_output_DataFrame)

        # then we take care of the training days
        for day in DataFrame.index[:-self.number_of_testing_day_in_a_day_level_DataFrame]:
            # option 1: all previous data
            if self.option == 1:
                '''first generate input(depends on option only), then generate output (depends on both option and LSTM type)'''
                if day - pd.Timedelta(self.number_of_days_to_predict_ahead, unit='d') in DataFrame.index:
                    input_DataFrame = DataFrame[first_day: day - pd.Timedelta(self.number_of_days_to_predict_ahead, unit='d')]
                    if self.many_to_many:
                        output_length = (day - pd.Timedelta(self.number_of_days_to_predict_ahead, unit='d') - first_day).days + 1  # output length varies based on the input length and should equal to input length
                        output_DataFrame = DataFrame[day - pd.Timedelta(output_length - 1, unit='d'):day]
                    else:
                        output_DataFrame = DataFrame.loc[[day]]
                else:
                    continue
            elif self.option == 2:
                # option 2: fixed length data
                '''first generate input(depends on option only), then generate output (depends on both option and LSTM type)'''
                if day - pd.Timedelta(self.number_of_days_to_predict_ahead + self.input_length - 1, unit='d') in DataFrame.index and day - pd.Timedelta(self.number_of_days_to_predict_ahead, unit='d') in DataFrame.index:
                    input_DataFrame = DataFrame[day - pd.Timedelta(self.number_of_days_to_predict_ahead + self.input_length - 1, unit='d'): day - pd.Timedelta(self.number_of_days_to_predict_ahead, unit='d')]
                    if self.many_to_many:
                        output_DataFrame = DataFrame[day - pd.Timedelta(self.input_length - 1, unit='d'): day]
                    else:
                        output_DataFrame = DataFrame.loc[[day]]
                else:
                    continue
            else:
                raise ValueError
            input_output_DataFrame = pd.concat([input_DataFrame, output_DataFrame])
            train_input_output_DataFrame_list.append(input_output_DataFrame)

        return train_input_output_DataFrame_list, test_input_output_DataFrame_list


