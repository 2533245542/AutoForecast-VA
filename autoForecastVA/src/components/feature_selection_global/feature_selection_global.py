import copy

import statsmodels.api as sm
import pandas as pd
import numpy as np

from autoForecastVA.src.utils.tuning_interval_moderator import OperationIntervalModerator


class FeatureSelectionGlobal():
    '''
    We do not consider clinic as a feature and thus the first feature index (0th) starts from the number of COVID-19 related calls.
    '''

    def __init__(self, dataPreparationForFeatureSelection=None, p_value_threshold=0.05, enforce_using_the_prediction_target_as_a_selected_feature=True, do_feature_selection=True, operation_interval=1, lazy=False):
        # inputs
        self.dataPreparationForFeatureSelection = dataPreparationForFeatureSelection
        self.p_value_threshold = p_value_threshold  # only features with p-value lower than or equal to this value will be kept
        self.enforce_using_the_prediction_target_as_a_selected_feature = enforce_using_the_prediction_target_as_a_selected_feature
        self.do_feature_selection = do_feature_selection  # if False select all features
        self.operation_interval = operation_interval
        self.lazy = lazy

        # outputs
        self.day_level_agency_level_feature_index_list_dictionary = {}
        self.day_level_agency_level_selected_p_value_list_dictionary = {}  # testing: should all not greater than threshold
        self.day_level_agency_level_not_selected_p_value_list_dictionary = {}  # testing: should all not greater than threshold
        self.day_level_agency_level_total_p_value_list_dictionary = {}  # testing: should all not equal to -1

        if not self.lazy:
            self.day_level_agency_level_original_train_DataFrame_dictionary = dataPreparationForFeatureSelection.day_level_agency_level_original_train_DataFrame_dictionary
            self.day_level_agency_level_shifted_train_DataFrame_dictionary = dataPreparationForFeatureSelection.day_level_agency_level_shifted_train_DataFrame_dictionary
            if not self.do_feature_selection:
                self.p_value_threshold = 999999999  # HACK Why not 1? Because I set a feature's p_value to 999999999 when it is homogenous
            self.perform_feature_selection()
            # enforce using the 0th numeric feature
            self.enforce_prediction_target_as_selected_feature()


    def perform_feature_selection(self):
        '''
        for each day and agency
            perform feature selection
        '''
        for day in self.day_level_agency_level_original_train_DataFrame_dictionary.keys():
            agency_level_feature_index_list_dictionary = {}
            agency_level_total_p_value_list_dictionary = {}
            agency_level_selected_p_value_list_dictionary = {}
            agency_level_not_selected_p_value_list_dictionary = {}
            for agency in self.day_level_agency_level_original_train_DataFrame_dictionary[day].keys():
                original_DataFrame = self.day_level_agency_level_original_train_DataFrame_dictionary[day][agency]
                shifted_DataFrame = self.day_level_agency_level_shifted_train_DataFrame_dictionary[day][agency]
                selected_index_list, total_p_value_list, selected_p_value_list, not_selected_p_value_list = self.day_level_medical_center_level_feature_selection(original_DataFrame, shifted_DataFrame)
                agency_level_feature_index_list_dictionary[agency] = selected_index_list
                agency_level_total_p_value_list_dictionary[agency] = total_p_value_list
                agency_level_selected_p_value_list_dictionary[agency] = selected_p_value_list
                agency_level_not_selected_p_value_list_dictionary[agency] = not_selected_p_value_list
            self.day_level_agency_level_feature_index_list_dictionary[day] = agency_level_feature_index_list_dictionary
            self.day_level_agency_level_total_p_value_list_dictionary[day] = agency_level_total_p_value_list_dictionary
            self.day_level_agency_level_selected_p_value_list_dictionary[day] = agency_level_selected_p_value_list_dictionary
            self.day_level_agency_level_not_selected_p_value_list_dictionary[day] = agency_level_not_selected_p_value_list_dictionary

            operationIntervalModerator = OperationIntervalModerator(days=self.day_level_agency_level_original_train_DataFrame_dictionary.keys(), operation_interval=self.operation_interval)
            current_operation_day = operationIntervalModerator.get_operation_day_of_a_day(day=day)
            self.day_level_agency_level_feature_index_list_dictionary[day] = copy.deepcopy(self.day_level_agency_level_feature_index_list_dictionary[current_operation_day])
            self.day_level_agency_level_total_p_value_list_dictionary[day] = copy.deepcopy(self.day_level_agency_level_total_p_value_list_dictionary[current_operation_day])
            self.day_level_agency_level_selected_p_value_list_dictionary[day] = copy.deepcopy(self.day_level_agency_level_selected_p_value_list_dictionary[current_operation_day])
            self.day_level_agency_level_not_selected_p_value_list_dictionary[day] = copy.deepcopy(self.day_level_agency_level_not_selected_p_value_list_dictionary[current_operation_day])


    def day_level_medical_center_level_feature_selection(self, original_DataFrame, shifted_DataFrame):
        '''
        Given a day and a medical center's DataFrame (original and shifted), select the top few rows from the original DataFrame to make the original DataFrame have the same number of observations as the shifted DataFrame; we then extract the 1st to the last columns in the edited original DataFrame as X. We ensure the ordering of data in both X and Y are the same. We then extract the 1st column in the shifted DataFrame as Y (exlcuding the 0th column hospital code).
        Then, we do feature selection by performing many linear regressions. We perform linear regression where the target is Y and the predictor is X.
        To perform feature selection, we first maintain a list of selected feature index. In a loop, we select from X the list of selected features. We then fit a linear regression model using the selected X and Y. From the linear regression, we find the feature with the largest p-value(feature name) and its p-value. We check if the p-value is greater than a predefined threshold; if it is, remove it from the set of selected features and continue the loop; if not, our feature selection algorithm converges and we exist the loop.
        After each linear regression, we record the most up-to-date p-values of each feature; if the feature has been eliminated in the last loop iteration, its p-value would not be updated; if not, it would be updated (note that the soon-to-be-removed [in this loop iteration] feature's p-value WILL be updated). We also create a list that contains the p-values of the selected features, as well as the not selected ones.

        :return: A list of feature index. The feature index is counted assuming that the clinic code feature does not exist.
        '''

        original_first_day = original_DataFrame.sort_index().index[0]
        original_edited_last_day = original_first_day + pd.Timedelta(len(shifted_DataFrame.index.unique()) - 1, 'd')
        aligned_original_DataFrame = original_DataFrame.sort_index().loc[:original_edited_last_day].sort_values(by=['clinic', 'date'])  # align original dataframe to shifted dataframe
        X = aligned_original_DataFrame.iloc[:, 1:]  # exclude the hospital code column
        Y = shifted_DataFrame.sort_values(by=['clinic', 'date']).iloc[:, [1]]  # exclude the hospital code column


        selected_set_of_feature_index_list = list(range(X.shape[1]))
        total_p_value_list = [-1] * X.shape[1]  # ensure this not any -1

        while True:
            feature_selected_X = X.iloc[:, selected_set_of_feature_index_list]
            maximum_p_value_feature_name, maximum_p_value, p_value_list = self.fit_ordinary_linear_regression_and_return_the_feature_with_the_largest_p_value(feature_selected_X, Y)
            maximum_p_value_feature_index = X.columns.tolist().index(maximum_p_value_feature_name)

            for index, p_value in zip(selected_set_of_feature_index_list, p_value_list):
                total_p_value_list[index] = p_value  # ensure this is all not -1

            if maximum_p_value > self.p_value_threshold:
                selected_set_of_feature_index_list.remove(maximum_p_value_feature_index)
                if len(selected_set_of_feature_index_list) == 0:
                    break
            else:
                break
        selected_features_p_value_list = [total_p_value_list[index] for index in selected_set_of_feature_index_list]  # ensure this is all <= threshold
        not_selected_features_p_value_list = [total_p_value_list[index] for index in list(range(X.shape[1])) if index not in selected_set_of_feature_index_list]  # ensure this is all > threshold
        return selected_set_of_feature_index_list, total_p_value_list, selected_features_p_value_list, not_selected_features_p_value_list

    def fit_ordinary_linear_regression_and_return_the_feature_with_the_largest_p_value(self, X, Y):
        '''
        We fit a linear regression using the input X and Y, and retrieve the p-value of each X feature. We find the maximum p-value, along with its index. Using the index, we find the name of the feature with the maximum p-value.

        :return feature name, feature p-value
        '''

        feature_name_list = X.columns.tolist()

        # find if there is any homogenous column; if a column is homogenous, return its name, with a p-value of 999999999, and with a p-value list where other p-values are -1
        for index, feature_name in enumerate(feature_name_list):
            if X[feature_name].nunique() == 1:
                p_value_list = [-1] * len(feature_name_list)
                p_value_list[feature_name_list.index(feature_name)] = 999999999
                return feature_name, 999999999, p_value_list

        # fit model
        X2 = sm.add_constant(X.to_numpy(), has_constant='add')  # has_constant='add' ensures that a 1 column is still added even when X contains a column with a zero variance
        Y2 = Y.to_numpy()
        ols = sm.OLS(Y2, X2)
        res = ols.fit()
        p_value_list = res.pvalues[1:]  # exclude the p-value of the added constant)

        # Do not do feature selection when the number of observations is smaller than the number of features.
        # This is achieved by returning any feature name with a 0 p-value
        if any(np.isnan(p_value_list)):  # if any p-value is nan, it means we failed to construct the linear model. Stop doing feature selection. We claim termination.
            return feature_name_list[0], 0, p_value_list

        maximum_p_value = np.max(p_value_list)
        maximum_p_value_feature_index = np.where(p_value_list == maximum_p_value)[0].item()  # np.where always returns a tuple containing one item so we need to select it
        maximum_p_value_feature_name = feature_name_list[maximum_p_value_feature_index]  # use the index to find the name

        return maximum_p_value_feature_name, maximum_p_value, p_value_list

    def enforce_prediction_target_as_selected_feature(self):
        if self.enforce_using_the_prediction_target_as_a_selected_feature:
            for day in self.day_level_agency_level_feature_index_list_dictionary.keys():
                for agency in self.day_level_agency_level_feature_index_list_dictionary[day].keys():
                    feature_index_list = self.day_level_agency_level_feature_index_list_dictionary[day][agency]
                    if 0 not in feature_index_list:
                        feature_index_list.insert(0, 0)  # it is done in-place. No need to put it back to the dictionary