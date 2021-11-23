import pandas as pd
import copy
import matplotlib as mpl
import matplotlib.pyplot as plt
import seaborn as sns
from functools import cached_property
from sklearn import metrics

from autoForecastVA.src.data_caching.variable_load_and_save import VariableSaverAndLoader
from autoForecastVA.src.utils.dictionary_wrangling import DictionaryWrangling


class AnalyzerPipeLineAutoForecastVA():
    def __init__(self, pipeLineAutoForecastVA, lazy=False):
        '''
        A PipeLineAutoForecastVA instance contains very rich information for the sake of simplicity.

        This analyzer acts as a helper for analyzing a run result for a PipeLineAutoForecastVA.

        We will be able to get much insight.

        The available properties include

        day_level_medical_center_level_used_hyper_parameter_value_combination_dictionary  # one hyperparemter value combiantion for each day and medical center
        day_to_clinic_to_hyper_parameter_value_combination_dataframe

        day_level_cluster_level_general_model_train_epoch_loss_list_dictionary  # one list of general model training loss for each day and cluser
        day_to_clinic_to_epoch_to_general_model_loss_dataframe

        day_level_medical_center_level_fine_tuned_model_train_epoch_loss_list_dictionary  # one list of fine tuned model training loss for each day and medical center
        day_to_clinic_to_epoch_to_fine_tuned_model_loss_dataframe

        day_level_medical_center_level_train_prediction_target_list_dictionary   # one list of train prediction target for each day and medical center
        day_to_clinic_to_index_to_train_prediction_target_dataframe

        day_level_medical_center_level_train_prediction_list_dictionary  # one list of train prediction for each day and medical center
        day_to_clinic_to_index_to_train_prediction_dataframe

        day_level_medical_center_level_daily_test_prediction_target_dictionary   # one test prediction target (one value) for each day and medical center
        day_to_clinic_to_test_prediction_target_dataframe

        day_level_medical_center_level_daily_test_prediction_dictionary  # one test prediction (one value)
        day_to_clinic_to_test_prediction_dataframe


        '''
        if not lazy:
            # inputs
            self.pipeLineAutoForecastVA = pipeLineAutoForecastVA

            self.dataCluseteringAndNormalization = pipeLineAutoForecastVA.dataCluseteringAndNormalization
            self.featureSelectionGlobal = pipeLineAutoForecastVA.featureSelectionGlobal

            self.day_level_cluster_level_hyperParameterTuningLoop_dictionary = self.pipeLineAutoForecastVA.day_level_cluster_level_hyperParameterTuningLoop_dictionary
            self.day_level_cluster_level_generalModelTraining_dictionary = self.pipeLineAutoForecastVA.day_level_cluster_level_generalModelTraining_dictionary
            self.day_level_medical_center_level_modelFineTuning_dictionary = self.pipeLineAutoForecastVA.day_level_medical_center_level_modelFineTuning_dictionary

            # outputs
            self.day_level_cluster_level_medical_center_list_dictionary = self.get_day_level_cluster_level_medical_center_list_dictionary()
            self.day_level_medical_center_to_cluster_dictionary = self.get_day_level_medical_center_to_cluster_dictionary()

            self.day_level_agency_level_feature_index_list_dictionary = self.featureSelectionGlobal.day_level_agency_level_feature_index_list_dictionary
            self.day_level_agency_level_total_p_value_list_dictionary = self.featureSelectionGlobal.day_level_agency_level_total_p_value_list_dictionary

            self.day_level_medical_center_level_test_prediction_target_dictionary = self.get_day_level_medical_center_level_test_prediction_target_dictionary()
            self.day_to_clinic_to_test_prediction_target_dataframe = self.get_day_to_clinic_to_test_prediction_target_dataframe()

            self.day_level_cluster_level_tuning_record_dataframe_dictionary = self.get_day_level_cluster_level_tuning_record_dataframe_dictionary()
            self.day_level_cluster_level_used_hyper_parameter_value_combination_dictionary = self.get_day_level_cluster_level_used_hyper_parameter_value_combination_dictionary()
            self.day_to_cluster_to_hyper_parameter_to_value_combination_dataframe = self.get_day_to_cluster_to_hyper_parameter_to_value_combination_dataframe()

            self.day_level_cluster_level_general_model_train_epoch_loss_list_dictionary = self.get_day_level_cluster_level_general_model_train_epoch_loss_list_dictionary()
            self.day_to_cluster_to_epoch_to_general_model_loss_dataframe = self.get_day_to_cluster_to_epoch_to_general_model_loss_dataframe()

            self.day_level_medical_center_level_fine_tuned_model_train_epoch_loss_list_dictionary = self.get_day_level_medical_center_level_fine_tuned_model_train_epoch_loss_list_dictionary()
            self.day_to_clinic_to_epoch_to_fine_tuned_model_loss_dataframe = self.get_day_to_clinic_to_epoch_to_fine_tuned_model_loss_dataframe()
            self.day_level_medical_center_level_fine_tuned_model_train_last_epoch_loss_dictionary = self.get_day_level_medical_center_level_fine_tuned_model_train_last_epoch_loss_dictionary(self.day_level_medical_center_level_fine_tuned_model_train_epoch_loss_list_dictionary)
            self.day_to_clinic_to_fine_tuned_model_last_epoch_loss_dataframe = self.get_day_to_clinic_to_fine_tuned_model_last_epoch_loss_dataframe(self.day_level_medical_center_level_fine_tuned_model_train_last_epoch_loss_dictionary)

            self.day_level_medical_center_level_train_prediction_target_list_dictionary = self.get_day_level_medical_center_level_train_prediction_target_list_dictionary()
            self.day_to_clinic_to_index_to_train_prediction_target_dataframe = self.get_day_to_clinic_to_index_to_train_prediction_target_dataframe()

            self.day_level_medical_center_level_train_prediction_list_dictionary = self.get_day_level_medical_center_level_train_prediction_list_dictionary()
            self.day_to_clinic_to_index_to_train_prediction_dataframe = self.get_day_to_clinic_to_index_to_train_prediction_dataframe()

            self.day_level_medical_center_level_test_prediction_dictionary = self.get_day_level_medical_center_level_test_prediction_dictionary()
            self.day_to_clinic_to_test_prediction_dataframe = self.get_day_to_clinic_to_test_prediction_dataframe()

            self.day_to_clinic_to_test_prediction_loss_dataframe = self.get_day_to_clinic_to_test_prediction_loss_dataframe( day_to_clinic_to_test_prediction_target_dataframe=self.day_to_clinic_to_test_prediction_target_dataframe, day_to_clinic_to_test_prediction_dataframe=self.day_to_clinic_to_test_prediction_dataframe)
            self.clinic_to_test_prediction_loss_dataframe = self.get_clinic_to_test_prediction_loss_dataframe( day_to_clinic_to_test_prediction_target_dataframe=self.day_to_clinic_to_test_prediction_target_dataframe, day_to_clinic_to_test_prediction_dataframe=self.day_to_clinic_to_test_prediction_dataframe)
            self.test_prediction_loss = self.get_test_prediction_loss( day_to_clinic_to_test_prediction_target_dataframe=self.day_to_clinic_to_test_prediction_target_dataframe, day_to_clinic_to_test_prediction_dataframe=self.day_to_clinic_to_test_prediction_dataframe)

    def get_day_level_cluster_level_medical_center_list_dictionary(self):
        return self.dataCluseteringAndNormalization.day_level_cluster_level_medical_center_list_dictionary

    def get_day_level_medical_center_to_cluster_dictionary(self):
        return self.dataCluseteringAndNormalization.day_level_medical_center_to_cluster_dictionary

    def get_day_level_medical_center_level_test_prediction_target_dictionary(self):
        return self.looper_double(self.day_level_medical_center_level_modelFineTuning_dictionary, lambda x: x.evaluator.rolling_forecast_all_days_normalization_reverted_test_prediction_targets[0])

    def get_day_to_clinic_to_test_prediction_target_dataframe(self):
        day_to_clinic_to_test_prediction_target_dataframe = self.convert_nested_dictionary_to_others(self.day_level_medical_center_level_test_prediction_target_dictionary)
        day_to_clinic_to_test_prediction_target_dataframe.columns = ['day', 'clinic', 'test_prediction_target']
        day_to_clinic_to_test_prediction_target_dataframe = day_to_clinic_to_test_prediction_target_dataframe.set_index('day')
        return day_to_clinic_to_test_prediction_target_dataframe

    def get_day_level_cluster_level_tuning_record_dataframe_dictionary(self):
        return self.looper_double(first_level_second_level_item_dictionary=self.day_level_cluster_level_hyperParameterTuningLoop_dictionary, value_getter_function=lambda x: x.run_result_dataframe_sorted_by_loss_and_budget)

    def get_day_level_cluster_level_used_hyper_parameter_value_combination_dictionary(self):
        return self.looper_double(first_level_second_level_item_dictionary=self.day_level_cluster_level_hyperParameterTuningLoop_dictionary, value_getter_function=lambda x: x.optimal_partial_hyper_parameter_value_combination_found_among_max_budget_evaluations)

    def get_day_to_cluster_to_hyper_parameter_to_value_combination_dataframe(self):
        day_to_cluster_to_hyper_parameter_to_value_dataframe = self.convert_triple_nested_dictionary_to_others(first_level_second_level_third_level_item_dictionary=self.day_level_cluster_level_used_hyper_parameter_value_combination_dictionary)
        day_to_cluster_to_hyper_parameter_to_value_dataframe.columns = ['day', 'cluster', 'hyper_parameter', 'value']
        day_to_cluster_to_hyper_parameter_to_value_dataframe = day_to_cluster_to_hyper_parameter_to_value_dataframe.set_index('day')
        return day_to_cluster_to_hyper_parameter_to_value_dataframe

    def get_day_level_cluster_level_general_model_train_epoch_loss_list_dictionary(self):
        return self.looper_double(first_level_second_level_item_dictionary=self.day_level_cluster_level_generalModelTraining_dictionary, value_getter_function=lambda x: x.hyperParameterValueCombinationEvaluator.list_of_rolling_forecast_daily_list_of_epoch_loss[0])

    def get_day_to_cluster_to_epoch_to_general_model_loss_dataframe(self):
        day_to_clinic_to_epoch_to_general_model_loss_dataframe = self.convert_nested_key_then_list_dictionary_to_others(self.day_level_cluster_level_general_model_train_epoch_loss_list_dictionary)
        day_to_clinic_to_epoch_to_general_model_loss_dataframe.columns = ['day', 'cluster', 'epoch', 'loss']
        day_to_clinic_to_epoch_to_general_model_loss_dataframe = day_to_clinic_to_epoch_to_general_model_loss_dataframe.set_index('day')
        return day_to_clinic_to_epoch_to_general_model_loss_dataframe

    def get_day_level_medical_center_level_fine_tuned_model_train_epoch_loss_list_dictionary(self):
        return self.looper_double(first_level_second_level_item_dictionary=self.day_level_medical_center_level_modelFineTuning_dictionary, value_getter_function=lambda x: x.evaluator.list_of_rolling_forecast_daily_list_of_epoch_loss[0])

    def get_day_to_clinic_to_epoch_to_fine_tuned_model_loss_dataframe(self):
        day_to_clinic_to_epoch_to_fine_tuned_model_loss_dataframe = self.convert_nested_key_then_list_dictionary_to_others(self.day_level_medical_center_level_fine_tuned_model_train_epoch_loss_list_dictionary)
        day_to_clinic_to_epoch_to_fine_tuned_model_loss_dataframe.columns = ['day', 'clinic', 'epoch', 'loss']
        day_to_clinic_to_epoch_to_fine_tuned_model_loss_dataframe = day_to_clinic_to_epoch_to_fine_tuned_model_loss_dataframe.set_index('day')
        return day_to_clinic_to_epoch_to_fine_tuned_model_loss_dataframe

    @staticmethod
    def get_day_level_medical_center_level_fine_tuned_model_train_last_epoch_loss_dictionary(day_level_medical_center_level_fine_tuned_model_train_epoch_loss_list_dictionary):
        day_level_clinic_level_train_loss_dictionary = DictionaryWrangling.create_empty_nested_dictionary(
            keys=day_level_medical_center_level_fine_tuned_model_train_epoch_loss_list_dictionary.keys())

        for day in sorted(day_level_medical_center_level_fine_tuned_model_train_epoch_loss_list_dictionary.keys()):
            for clinic in day_level_medical_center_level_fine_tuned_model_train_epoch_loss_list_dictionary[day].keys():
                train_epoch_loss_list = day_level_medical_center_level_fine_tuned_model_train_epoch_loss_list_dictionary[day][clinic]
                last_epoch_loss = train_epoch_loss_list[-1] if len(train_epoch_loss_list) > 0 else -1
                day_level_clinic_level_train_loss_dictionary[day][clinic] = last_epoch_loss
                # day_level_clinic_level_train_loss_dictionary[day][clinic] = day_level_medical_center_level_fine_tuned_model_train_epoch_loss_list_dictionary[day][clinic][-1]

        return day_level_clinic_level_train_loss_dictionary

    @staticmethod
    def get_day_to_clinic_to_fine_tuned_model_last_epoch_loss_dataframe(day_level_medical_center_level_fine_tuned_model_train_last_epoch_loss_dictionary):
        day_to_clinic_to_fine_tuned_model_last_epoch_loss_dataframe = AnalyzerPipeLineAutoForecastVA.convert_nested_dictionary_to_others(day_level_medical_center_level_fine_tuned_model_train_last_epoch_loss_dictionary)
        day_to_clinic_to_fine_tuned_model_last_epoch_loss_dataframe.columns = ['day', 'clinic', 'loss']
        day_to_clinic_to_fine_tuned_model_last_epoch_loss_dataframe = day_to_clinic_to_fine_tuned_model_last_epoch_loss_dataframe.set_index('day')
        return day_to_clinic_to_fine_tuned_model_last_epoch_loss_dataframe

    def get_day_level_medical_center_level_train_prediction_target_list_dictionary(self):
        return self.looper_double(first_level_second_level_item_dictionary=self.day_level_medical_center_level_modelFineTuning_dictionary, value_getter_function=lambda x: x.evaluator.rolling_forecast_all_days_normalization_reverted_train_prediction_targets)

    def get_day_to_clinic_to_index_to_train_prediction_target_dataframe(self):
        day_to_clinic_to_index_to_train_prediction_target_dataframe = self.convert_nested_key_then_list_dictionary_to_others(self.day_level_medical_center_level_train_prediction_target_list_dictionary)
        day_to_clinic_to_index_to_train_prediction_target_dataframe.columns = ['day', 'clinic', 'index', 'train_prediction_target']
        day_to_clinic_to_index_to_train_prediction_target_dataframe = day_to_clinic_to_index_to_train_prediction_target_dataframe.set_index('day')
        return day_to_clinic_to_index_to_train_prediction_target_dataframe

    def get_day_level_medical_center_level_train_prediction_list_dictionary(self):
        return self.looper_double(first_level_second_level_item_dictionary=self.day_level_medical_center_level_modelFineTuning_dictionary, value_getter_function=lambda x: x.evaluator.rolling_forecast_all_days_normalization_reverted_train_predictions)

    def get_day_to_clinic_to_index_to_train_prediction_dataframe(self):
        day_to_clinic_to_index_to_train_prediction_dataframe = self.convert_nested_key_then_list_dictionary_to_others(self.day_level_medical_center_level_train_prediction_list_dictionary)
        day_to_clinic_to_index_to_train_prediction_dataframe.columns = ['day', 'clinic', 'index', 'train_prediction']
        day_to_clinic_to_index_to_train_prediction_dataframe = day_to_clinic_to_index_to_train_prediction_dataframe.set_index('day')
        return day_to_clinic_to_index_to_train_prediction_dataframe

    def get_day_level_medical_center_level_test_prediction_dictionary(self):
        return self.looper_double(self.day_level_medical_center_level_modelFineTuning_dictionary, lambda x: x.evaluator.rolling_forecast_all_days_normalization_reverted_test_predictions[0])

    def get_day_to_clinic_to_test_prediction_dataframe(self):
        day_to_clinic_to_test_prediction_dataframe = self.convert_nested_dictionary_to_others(self.day_level_medical_center_level_test_prediction_dictionary)
        day_to_clinic_to_test_prediction_dataframe.columns = ['day', 'clinic', 'test_prediction']
        day_to_clinic_to_test_prediction_dataframe = day_to_clinic_to_test_prediction_dataframe.set_index('day')
        return day_to_clinic_to_test_prediction_dataframe

    @staticmethod
    def get_day_to_clinic_to_test_prediction_loss_dataframe(day_to_clinic_to_test_prediction_target_dataframe, day_to_clinic_to_test_prediction_dataframe, loss_function=metrics.mean_squared_error):
        day_to_clinic_to_test_prediction_loss_dataframe = pd.merge(left=day_to_clinic_to_test_prediction_target_dataframe, right=day_to_clinic_to_test_prediction_dataframe, on=['day', 'clinic'], validate='one_to_one')

        list_of_daily_clinic_loss = []
        for i in range(day_to_clinic_to_test_prediction_loss_dataframe.shape[0]):
            list_of_daily_clinic_loss.append(loss_function( day_to_clinic_to_test_prediction_loss_dataframe.iloc[[i],].test_prediction_target.tolist(), day_to_clinic_to_test_prediction_loss_dataframe.iloc[[i],].test_prediction.tolist()))

        day_to_clinic_to_test_prediction_loss_dataframe['loss'] = list_of_daily_clinic_loss
        return day_to_clinic_to_test_prediction_loss_dataframe

    @staticmethod
    def get_clinic_to_test_prediction_loss_dataframe(day_to_clinic_to_test_prediction_target_dataframe, day_to_clinic_to_test_prediction_dataframe, loss_function=metrics.mean_squared_error):
        '''usage
        clinic_to_test_prediction_loss_dataframe = AnalyzerPipeLineAutoForecastVA.get_clinic_to_test_prediction_loss_dataframe( day_to_clinic_to_test_prediction_target_dataframe=day_to_clinic_to_test_prediction_target_dataframe, day_to_clinic_to_test_prediction_dataframe=day_to_clinic_to_test_prediction_dataframe, loss_function=metrics.mean_squared_error)
        '''
        day_to_clinic_to_test_prediction_loss_dataframe = pd.merge( left=day_to_clinic_to_test_prediction_target_dataframe, right=day_to_clinic_to_test_prediction_dataframe, on=['day', 'clinic'], validate='one_to_one')

        list_of_clinic = []
        list_of_loss = []
        for clinic in day_to_clinic_to_test_prediction_loss_dataframe.clinic.unique():
            day_to_test_prediction_loss_dataframe = day_to_clinic_to_test_prediction_loss_dataframe[ day_to_clinic_to_test_prediction_loss_dataframe.clinic == clinic]
            single_clinic_test_prediction_loss = loss_function( day_to_test_prediction_loss_dataframe.test_prediction_target.tolist(), day_to_test_prediction_loss_dataframe.test_prediction.tolist())
            list_of_clinic.append(clinic)
            list_of_loss.append(single_clinic_test_prediction_loss)

        clinic_to_test_prediction_loss_dataframe = pd.DataFrame({'clinic': list_of_clinic, 'loss': list_of_loss})
        return clinic_to_test_prediction_loss_dataframe

    @staticmethod
    def get_test_prediction_loss(day_to_clinic_to_test_prediction_target_dataframe, day_to_clinic_to_test_prediction_dataframe, loss_function=metrics.mean_squared_error):
        '''usage
        test_prediction_loss = AnalyzerPipeLineAutoForecastVA.get_test_prediction_loss( day_to_clinic_to_test_prediction_target_dataframe=day_to_clinic_to_test_prediction_target_dataframe, day_to_clinic_to_test_prediction_dataframe=day_to_clinic_to_test_prediction_dataframe)
        '''

        day_to_clinic_to_test_prediction_loss_dataframe = pd.merge(left=day_to_clinic_to_test_prediction_target_dataframe, right=day_to_clinic_to_test_prediction_dataframe, on=['day', 'clinic'], validate='one_to_one')
        test_prediction_loss = loss_function(day_to_clinic_to_test_prediction_loss_dataframe.test_prediction_target, day_to_clinic_to_test_prediction_loss_dataframe.test_prediction)
        return test_prediction_loss


    # helper functions
    '''loopers'''
    def looper_double(self, first_level_second_level_item_dictionary, value_getter_function):
        first_level_second_level_value_dictionary = DictionaryWrangling.create_empty_nested_dictionary(first_level_second_level_item_dictionary.keys())
        for first_level_key in first_level_second_level_item_dictionary.keys():
            for second_level_key in first_level_second_level_item_dictionary[first_level_key].keys():
                item = first_level_second_level_item_dictionary[first_level_key][second_level_key]
                value = value_getter_function(item)
                first_level_second_level_value_dictionary[first_level_key][second_level_key] = value

        return first_level_second_level_value_dictionary
    @staticmethod
    def transform_value_of_two_level_dictionary(first_level_second_level_item_dictionary, value_transform_function):
        """Transform the value of a two level dictionary"""
        first_level_second_level_value_dictionary = DictionaryWrangling.create_empty_nested_dictionary(first_level_second_level_item_dictionary.keys())
        for first_level_key in first_level_second_level_item_dictionary.keys():
            for second_level_key in first_level_second_level_item_dictionary[first_level_key].keys():
                value = first_level_second_level_item_dictionary[first_level_key][second_level_key]
                transformed_value = value_transform_function(value)
                first_level_second_level_value_dictionary[first_level_key][second_level_key] = transformed_value

        return first_level_second_level_value_dictionary


    '''convert dictionary to dataframe'''
    @staticmethod
    def convert_nested_dictionary_to_others(first_level_second_level_item_dictionary, first_level_key_converter=lambda x: x, second_level_key_converter=lambda x: x, item_converter=lambda x: x, return_type='dataframe'):
        '''
        return_type: 'dictionary', 'dataframe'
        '''
        list_of_first_level_key = []
        list_of_second_level_key = []
        list_of_items = []
        for first_level_key in first_level_second_level_item_dictionary.keys():
            for second_level_key in first_level_second_level_item_dictionary[first_level_key].keys():
                list_of_first_level_key.append(first_level_key_converter(first_level_key))
                list_of_second_level_key.append(second_level_key_converter(second_level_key))
                list_of_items.append(item_converter(first_level_second_level_item_dictionary[first_level_key][second_level_key]))
        result_dictionary = {'first_level_key': list_of_first_level_key, 'second_level_key': list_of_second_level_key, 'item': list_of_items}
        if return_type == 'dictionary':
            return result_dictionary
        elif return_type == 'dataframe':
            return pd.DataFrame(result_dictionary)
        else:
            raise NotImplementedError

    @staticmethod
    def convert_triple_nested_dictionary_to_others(first_level_second_level_third_level_item_dictionary, first_level_key_converter=lambda x: x, second_level_key_converter=lambda x: x, item_converter=lambda x: x, third_level_key_converter=lambda x:x, return_type='dataframe'):   # for hyperpaemter value combination
        list_of_first_level_key = []
        list_of_second_level_key = []
        list_of_third_level_key = []
        list_of_items = []
        for first_level_key in first_level_second_level_third_level_item_dictionary.keys():
            for second_level_key in first_level_second_level_third_level_item_dictionary[first_level_key].keys():
                for third_level_key in first_level_second_level_third_level_item_dictionary[first_level_key][second_level_key].keys():
                    list_of_first_level_key.append(first_level_key_converter(first_level_key))
                    list_of_second_level_key.append(second_level_key_converter(second_level_key))
                    list_of_third_level_key.append(third_level_key_converter(third_level_key))
                    list_of_items.append(item_converter(first_level_second_level_third_level_item_dictionary[first_level_key][second_level_key][third_level_key]))
        result_dictionary = {'first_level_key': list_of_first_level_key, 'second_level_key': list_of_second_level_key, 'third_level_key': list_of_third_level_key, 'item': list_of_items}
        if return_type == 'dictionary':
            return result_dictionary
        elif return_type == 'dataframe':
            return pd.DataFrame(result_dictionary)
        else:
            raise NotImplementedError

    @staticmethod
    def convert_nested_key_then_list_dictionary_to_others(first_level_second_level_list_dictionary, first_level_key_converter=lambda x: x, second_level_key_converter=lambda x: x, sub_item_converter=lambda x: x, return_type='dataframe'):   # for list-like item
        list_of_first_level_key = []
        list_of_second_level_key = []
        list_of_index = []
        list_of_sub_items = []  # a sub_item is like an elment of a list
        for first_level_key in first_level_second_level_list_dictionary.keys():
            for second_level_key in first_level_second_level_list_dictionary[first_level_key].keys():
                for index, sub_item in enumerate(first_level_second_level_list_dictionary[first_level_key][second_level_key]):
                    list_of_first_level_key.append(first_level_key_converter(first_level_key))
                    list_of_second_level_key.append(second_level_key_converter(second_level_key))
                    list_of_index.append(index)
                    list_of_sub_items.append(sub_item_converter(sub_item))
        result_dictionary = {'first_level_key': list_of_first_level_key, 'second_level_key': list_of_second_level_key, 'index': list_of_index, 'sub_item': list_of_sub_items}
        if return_type == 'dictionary':
            return result_dictionary
        elif return_type == 'dataframe':
            return pd.DataFrame(result_dictionary)
        else:
            raise NotImplementedError

    @staticmethod
    def reformat_dataframe(input_dataframe, new_column_names=None, name_of_index_column='first_level_key'):
        """Reformat the column names and index of a dataframe."""
        reformatted_dataframe = input_dataframe.copy()
        reformatted_dataframe.columns = new_column_names
        reformatted_dataframe = reformatted_dataframe.set_index(keys=name_of_index_column)
        return reformatted_dataframe

    '''make figures'''
    @staticmethod
    def make_figure_for_clusterization_trend(day_level_medical_center_to_cluster_dictionary, rotation_degree=90, height=20.27, aspect=11.7 / 20.27):
        '''
        def make_figure_for_clusterization_trend
        input:
        day_level_medical_center_to_cluster_dictionary
        rotation_degree

        output:
        a figure
        steps:

        usage:
        g = make_figure_for_clusterization_trend(day_level_medical_center_to_cluster_dictionary)

        '''

        day_to_clinic_to_cluster_dataframe = AnalyzerPipeLineAutoForecastVA.convert_nested_dictionary_to_others( first_level_second_level_item_dictionary=day_level_medical_center_to_cluster_dictionary)
        day_to_clinic_to_cluster_dataframe.columns = ['day', 'clinic', 'cluster']
        day_to_clinic_to_cluster_dataframe = day_to_clinic_to_cluster_dataframe.set_index('day')

        ### debug start
        # figure, ax = plt.subplots(figsize=(30, 30))
        # number_of_unique_colors = len(day_to_clinic_to_cluster_dataframe.cluster.unique())
        # palette = ['deepskyblue', 'goldenrod', 'grey', 'orchid'][:number_of_unique_colors]
        #
        # list_of_marker = ['o', 'v', 's', 'd']
        #
        # for cluster in day_to_clinic_to_cluster_dataframe.cluster.unique():
        #     cluster_df = day_to_clinic_to_cluster_dataframe.loc[day_to_clinic_to_cluster_dataframe.cluster == cluster, ]
        #     ax.scatter(x=cluster_df.index, y=cluster_df.clinic, label=cluster)
        #
        # ax.figure.show()
        #
        #
        # figure_for_clusterzation_trend = sns.scatterplot(data=day_to_clinic_to_cluster_dataframe, x='day', y='clinic', hue='cluster', style='cluster', palette=palette, ax=ax)
        # figure_for_clusterzation_trend.figure.show()
        ### debug end

        figure_for_clusterzation_trend = sns.relplot(data=day_to_clinic_to_cluster_dataframe, x='day', y='clinic', hue='cluster', height=height, aspect=aspect)
        figure_for_clusterzation_trend.set_xticklabels(rotation=rotation_degree)
        return figure_for_clusterzation_trend

    @staticmethod
    def make_figure_for_cluster_size_trend(day_level_cluster_level_medical_center_list_dictionary):
        '''
        def make_figure_for_cluster_size_trend
        inputs:
        day_level_cluster_level_medical_center_list_dictionary
        outputs:
        a figure
        steps:
        usage:
        figure_for_cluster_size_trend = make_figure_for_cluster_size_trend(day_level_cluster_level_medical_center_list_dictionary=day_level_cluster_level_medical_center_list_dictionary)
        figure_for_cluster_size_trend.savefig('figure_for_cluster_size_trend.png')
        '''
        palette = ['deepskyblue', 'goldenrod', 'grey', 'orchid']

        day_to_cluster_to_size_dataframe = AnalyzerPipeLineAutoForecastVA.convert_nested_dictionary_to_others(first_level_second_level_item_dictionary=day_level_cluster_level_medical_center_list_dictionary, item_converter=lambda x: len(x), return_type='dataframe')
        day_to_cluster_to_size_dataframe.columns = ['day', 'cluster', 'size']
        day_to_cluster_to_size_dataframe = day_to_cluster_to_size_dataframe.set_index('day')
        plot_df = day_to_cluster_to_size_dataframe.pivot_table(index='day', columns='cluster', values='size').fillna(0)
        plot_df.index = plot_df.index.strftime('%Y-%m-%d')

        # mpl.style.use('tableau-colorblind10')
        f, ax = plt.subplots()
        # plot_df.plot(kind='bar', stacked=True, ax=ax, color=['tab:blue', 'tab:orange', 'tab:black', 'tab:purple'])
        plot_df.plot(kind='bar', stacked=True, ax=ax, color=palette)
        # plot_df.plot(kind='bar', stacked=True, ax=ax)
        plt.legend(title='cluster', shadow=True)

        return ax.figure


    @staticmethod
    def make_figure_for_selected_features_trend(day_level_agency_level_feature_index_list_dictionary, rotation_degree=30):
        day_to_cluster_to_index_of_feature_index_to_feature_index_dataframe = AnalyzerPipeLineAutoForecastVA.convert_nested_key_then_list_dictionary_to_others( day_level_agency_level_feature_index_list_dictionary)
        day_to_cluster_to_index_of_feature_index_to_feature_index_dataframe.columns = ['day', 'cluster', 'index_of_feature_index', 'feature_index']
        day_to_cluster_to_feature_index_dataframe = day_to_cluster_to_index_of_feature_index_to_feature_index_dataframe[ ['day', 'cluster', 'feature_index']]
        day_to_cluster_to_feature_index_dataframe = day_to_cluster_to_feature_index_dataframe.set_index('day')

        # ### debug start
        # figure, ax = plt.subplots(figsize=(15, 20))
        # number_of_unique_colors = len(day_to_clinic_to_cluster_dataframe.cluster.unique())
        # palette = ['deepskyblue', 'goldenrod', 'grey', 'orchid'][:number_of_unique_colors]
        #
        # list_of_marker = ['o', 'v', 's', 'd']
        # ### debug end
        figure_for_selected_features_trends = sns.relplot(data=day_to_cluster_to_feature_index_dataframe, x='day', y='feature_index', row='cluster')
        figure_for_selected_features_trends.set_xticklabels(rotation=rotation_degree)
        return figure_for_selected_features_trends

    @staticmethod
    def make_figure_for_number_of_selected_features_trend(day_level_agency_level_feature_index_list_dictionary, rotation_degree=30):
        day_to_cluster_to_number_of_selected_features = AnalyzerPipeLineAutoForecastVA.convert_nested_dictionary_to_others( day_level_agency_level_feature_index_list_dictionary, item_converter=lambda x: len(x))
        day_to_cluster_to_number_of_selected_features.columns = ['day', 'cluster', 'number_of_selected_features']
        day_to_cluster_to_feature_index_dataframe = day_to_cluster_to_number_of_selected_features.set_index('day')

        figure_for_number_of_selected_features_trend = sns.relplot(data=day_to_cluster_to_feature_index_dataframe, x='day', y='number_of_selected_features', row='cluster')
        figure_for_number_of_selected_features_trend.set_xticklabels(rotation=rotation_degree)
        return figure_for_number_of_selected_features_trend

    @staticmethod
    def make_figure_for_percentage_of_features_selected(day_level_agency_level_feature_index_list_dictionary, rotation_degree=90):
        day_level_agency_level_feature_index_list_dictionary = copy.deepcopy(day_level_agency_level_feature_index_list_dictionary)
        day_to_cluster_to_index_of_feature_index_to_feature_index_dataframe = AnalyzerPipeLineAutoForecastVA.convert_nested_key_then_list_dictionary_to_others( day_level_agency_level_feature_index_list_dictionary)
        day_to_cluster_to_index_of_feature_index_to_feature_index_dataframe.columns = ['day', 'cluster', 'index_of_feature_index', 'feature_index']
        day_to_cluster_to_feature_index_dataframe = day_to_cluster_to_index_of_feature_index_to_feature_index_dataframe[ ['day', 'cluster', 'feature_index']]
        day_to_cluster_to_feature_index_dataframe = day_to_cluster_to_feature_index_dataframe.set_index('day')

        day_level_feature_level_feature_selected_rate_dictionary = DictionaryWrangling.create_empty_nested_dictionary(day_level_agency_level_feature_index_list_dictionary.keys())

        for day in day_level_feature_level_feature_selected_rate_dictionary.keys():
            for feature_index in sorted(day_to_cluster_to_feature_index_dataframe.feature_index.unique()):
                number_of_times_selected = 0
                for cluster in day_level_agency_level_feature_index_list_dictionary[day].keys():
                    if feature_index in day_level_agency_level_feature_index_list_dictionary[day][cluster]:
                        number_of_times_selected += 1
                day_level_feature_level_feature_selected_rate_dictionary[day][feature_index] = number_of_times_selected / len(day_level_agency_level_feature_index_list_dictionary[day].keys())

        day_to_feature_index_to_selected_rate_dataframe = AnalyzerPipeLineAutoForecastVA.convert_nested_dictionary_to_others(first_level_second_level_item_dictionary=day_level_feature_level_feature_selected_rate_dictionary)
        day_to_feature_index_to_selected_rate_dataframe.columns = ['day', 'feature_index', 'selected_rate']
        day_to_feature_index_to_selected_rate_dataframe.day = day_to_feature_index_to_selected_rate_dataframe.day.dt.strftime('%Y-%m-%d')

        pivoted_day_to_feature_index_to_selected_rate_dataframe = day_to_feature_index_to_selected_rate_dataframe.pivot(index='feature_index', columns='day', values='selected_rate')
        ### debug start
        f, ax = plt.subplots()
        sns.heatmap(pivoted_day_to_feature_index_to_selected_rate_dataframe, ax=ax)
        ax.set_xticklabels(labels=ax.get_xticklabels(), rotation=rotation_degree)
        ax.set_xlabel('Date')
        ax.set_ylabel('Feature index')
        ax.set_yticklabels(list(range(1, len(day_to_feature_index_to_selected_rate_dataframe.feature_index.unique()) + 1)))
        f.tight_layout()
        # f.show()
        # f.savefig('asdf.png')
        return f

        ### debug end
        # f, ax = plt.subplots()
        # sns.heatmap(pivoted_day_to_feature_index_to_selected_rate_dataframe, ax=ax)
        # ax.set_xticklabels(labels=ax.get_xticklabels(), rotation=rotation_degree)
        # return ax.figure

    @staticmethod
    def make_figure_for_hyper_parameter_value_combination_trend(day_to_cluster_to_hyper_parameter_to_value_combination_dataframe, rotation_degree=30, kind='scatter'):
        ''' kind = 'scatter' or 'line' '''
        day_to_cluster_to_hyper_parameter_to_value_combination_dataframe = day_to_cluster_to_hyper_parameter_to_value_combination_dataframe.reset_index()
        figure_for_hyper_parameter_value_combination_trend = sns.relplot(data=day_to_cluster_to_hyper_parameter_to_value_combination_dataframe, x='day', y='value', kind=kind, row='hyper_parameter', facet_kws={'sharey': False, 'sharex': True})
        figure_for_hyper_parameter_value_combination_trend.set_xticklabels(rotation=rotation_degree)
        return figure_for_hyper_parameter_value_combination_trend

    @staticmethod
    def make_figure_for_all_clinic_test_target_and_prediction(day_to_clinic_to_test_prediction_dataframe, day_to_clinic_to_test_prediction_target_dataframe, rotation_degree=30):
        '''
        inputs:
        day_to_clinic_to_test_prediction_dataframe
        day_to_clinic_to_test_prediction_target_dataframe

        outputs:
        a figure

        steps:
        for both dataframe,
        add a column,
        reset index
        rename the column names to  ['day', 'clinic', 'value', 'target_or_prediction']
        concate them to create plot_df

        g = sns.FacetGrid(plot_df, col='clinic', hue='target_or_prediction')
        g.map(plt.scatter, "day", "value")
        g.add_legend()
        plt.show()

        usage:
        plot_predictions_all_clinic(result_dictionary['day_to_clinic_to_test_prediction_target_dataframe'], result_dictionary['day_to_clinic_to_test_prediction_dataframe'])
        '''

        day_to_clinic_to_test_prediction_target_dataframe = day_to_clinic_to_test_prediction_target_dataframe.reset_index()  # HACK must do reset index first to create a copy of the dataframe
        day_to_clinic_to_test_prediction_target_dataframe['target_or_prediction'] = 'target'
        day_to_clinic_to_test_prediction_target_dataframe.columns = ['day', 'clinic', 'value', 'target_or_prediction']

        day_to_clinic_to_test_prediction_dataframe = day_to_clinic_to_test_prediction_dataframe.reset_index()  # HACK
        day_to_clinic_to_test_prediction_dataframe['target_or_prediction'] = 'prediction'
        day_to_clinic_to_test_prediction_dataframe.columns = ['day', 'clinic', 'value', 'target_or_prediction']

        plot_df = pd.concat([day_to_clinic_to_test_prediction_target_dataframe, day_to_clinic_to_test_prediction_dataframe], axis=0)

        g = sns.relplot(data=plot_df, x='day', y='value', hue='target_or_prediction', col='clinic', col_wrap=10, kind='line')
        g.set_xticklabels(rotation=rotation_degree)

        return g

    @staticmethod
    def make_figure_for_test_prediction_loss(day_to_clinic_to_test_prediction_loss_dataframe, col_wrap=10, rotation_degree=30):
        '''
        usage:
        figure_for_test_prediction_loss = AnalyzerPipeLineAutoForecastVA.make_figure_for_test_prediction_loss(day_to_clinic_to_test_prediction_loss_dataframe=day_to_clinic_to_test_prediction_loss_dataframe, col_wrap=10, rotation_degree=30)
        figure_for_test_prediction_loss.savefig('figure_for_test_prediction_loss.png')
        '''
        plot_dataframe = day_to_clinic_to_test_prediction_loss_dataframe.melt(id_vars='clinic', value_vars=['test_prediction_target', 'test_prediction', 'loss'], ignore_index=False)
        figure_for_test_prediction_loss = sns.relplot(data=plot_dataframe, x='day', y='value', hue='variable', col='clinic', col_wrap=col_wrap, kind='line')
        figure_for_test_prediction_loss.set_xticklabels(rotation=rotation_degree)
        return figure_for_test_prediction_loss

    '''data saver'''
    @staticmethod
    def save_analyzer_result_in_a_memory_efficient_way(pipeLineAutoForecastVA, file_name='data/memory_efficient_analyzer.dat'):
        analyzerPipeLineAutoForecastVA = AnalyzerPipeLineAutoForecastVA(pipeLineAutoForecastVA)
        analyzerPipeLineAutoForecastVA.pipeLineAutoForecastVA = None
        analyzerPipeLineAutoForecastVA.dataCluseteringAndNormalization = None
        analyzerPipeLineAutoForecastVA.featureSelectionGlobal = None
        analyzerPipeLineAutoForecastVA.day_level_cluster_level_hyperParameterTuningLoop_dictionary = None
        analyzerPipeLineAutoForecastVA.day_level_cluster_level_generalModelTraining_dictionary = None
        analyzerPipeLineAutoForecastVA.day_level_medical_center_level_modelFineTuning_dictionary = None

        VariableSaverAndLoader(save=True, list_of_variables_to_save=[analyzerPipeLineAutoForecastVA], file_name=file_name)