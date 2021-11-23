import copy

from torch import nn
import numpy as np
import torch

from autoForecastVA.src.components.data_preprocessing_global.data_clustering_and_normalization import DataCluseteringAndNormalization
from autoForecastVA.src.components.data_preprocessing_global.data_delimiting_by_day_and_medical_center import DataDelimitingByDayAndMedicalCenter
from autoForecastVA.src.components.data_preprocessing_global.data_imputation_and_averaging import DataImputationAndAveraging
from autoForecastVA.src.components.data_preprocessing_global.data_preparation_for_feature_selection import DataPreparationForFeatureSelection
from autoForecastVA.src.components.feature_selection_global.feature_selection_global import FeatureSelectionGlobal
from autoForecastVA.src.components.general_model_training.general_model_training import GeneralModelTraining
from autoForecastVA.src.components.hyper_parameter_tuning.hyper_parameter_tuning_loop import HyperParameterTuningLoop
from autoForecastVA.src.components.model_fine_tuning.model_fine_tuning import ModelFineTuning
from autoForecastVA.src.utils.dictionary_wrangling import DictionaryWrangling
from autoForecastVA.src.utils.tuning_interval_moderator import OperationIntervalModerator


class PipeLineAutoForecastVA():
    def __init__(self, medical_center_subset=None, time_period_subset=None, dataset_path=None, number_of_days_for_data_averaging=2, max_number_of_cluster=3, number_of_days_to_predict_ahead=1, number_of_test_days_in_a_day_level_DataFrame=1, p_value_threshold=0.05, data_windowing_option=2, input_length=3, lstm_is_many_to_many=False, number_of_rolling_forecast_days=2, hyper_parameter_tuning_number_of_test_days_in_DataFrame=1, hyper_parameter_tuning_number_of_rolling_forecast_days=3, hyper_parameter_space=None, hyper_parameter_space_seed=0, numpy_seed=0, CustomizedWorker=None, nameserver_run_id='example1', nameserver_address='127.0.0.1', nameserver_port=65300, min_budget=10, max_budget=50, n_iterations=2, neural_network_training_seed=0, loss_function=None, do_fine_tuning=False, trainable_parameter_name_list=None, do_clustering=True, do_general_model_training=True, do_normalization=True, do_feature_selection=True, tuning_interval=1, batch_size_for_tuning=60, batch_size_for_general_model_building=60, batch_size_for_fine_tuning=60, generate_train_prediction_during_tuning=True, enable_data_caching_during_tuning=False, verbose=True, lazy=False):
        '''
        Several requirements:
        The first column of the dataset must be clinic, and named clinic
        It is assumed the rolling forecast is doing in a continous day by day fashion, no skipping days
        '''
        # inputs
        self.medical_center_subset = medical_center_subset  # default to be None
        self.time_period_subset = time_period_subset  # default to be None

        self.dataset_path = dataset_path
        self.number_of_days_for_data_averaging = number_of_days_for_data_averaging

        self.max_number_of_cluster = max_number_of_cluster

        self.number_of_days_to_predict_ahead = number_of_days_to_predict_ahead
        self.number_of_test_days_in_a_day_level_DataFrame = number_of_test_days_in_a_day_level_DataFrame

        self.p_value_threshold = p_value_threshold  # for feature selection

        self.data_windowing_option = data_windowing_option
        self.input_length = input_length
        self.lstm_is_many_to_many = lstm_is_many_to_many

        self.number_of_rolling_forecast_days = number_of_rolling_forecast_days

        self.hyper_parameter_tuning_number_of_test_days_in_DataFrame = hyper_parameter_tuning_number_of_test_days_in_DataFrame  # usually equal to 1
        self.hyper_parameter_tuning_number_of_rolling_forecast_days = hyper_parameter_tuning_number_of_rolling_forecast_days  # usually greater than one

        self.hyper_parameter_space = hyper_parameter_space

        self.hyper_parameter_space_seed = hyper_parameter_space_seed  # randomness in hyperparemter space sampling

        self.numpy_seed = numpy_seed  # controls randomness in BOHB optimizer

        self.CustomizedWorker = CustomizedWorker  # the worker for runnning BOHB
        self.nameserver_run_id = nameserver_run_id  # can use the same run id
        self.nameserver_address = nameserver_address  # can use the same address
        self.nameserver_port = nameserver_port  # dont be 80, 8080, 443 or below 1024; each instance (that runs on the same local computer) should have a unique nameserver_port
        self.min_budget = min_budget
        self.max_budget = max_budget  # also train the general model with max budget with early stopping
        self.n_iterations = n_iterations

        self.neural_network_training_seed = neural_network_training_seed
        self.loss_function = loss_function

        self.do_fine_tuning = do_fine_tuning
        self.trainable_parameter_name_list = trainable_parameter_name_list

        self.do_clustering = do_clustering
        self.do_general_model_training = do_general_model_training
        self.do_normalization = do_normalization  # should be true in practice; but set to false to better inspect the process

        self.do_feature_selection = do_feature_selection

        self.tuning_interval = tuning_interval

        self.batch_size_for_tuning = batch_size_for_tuning
        self.batch_size_for_general_model_building = batch_size_for_general_model_building
        self.batch_size_for_fine_tuning = batch_size_for_fine_tuning

        self.generate_train_prediction_during_tuning = generate_train_prediction_during_tuning
        self.enable_data_caching_during_tuning = enable_data_caching_during_tuning

        self.verbose = verbose

        # outputs
        self.filter_by_medical_center = False
        self.filter_by_time_period = False

        self.dataImputationAndAveraging = None
        self.processed_dataset = None

        self.dataDelimitingByDayAndMedicalCenter = None

        self.dataCluseteringAndNormalization = None

        self.dataPreparationForFeatureSelection = None
        self.featureSelectionGlobal = None

        self.day_level_cluster_level_hyperParameterTuningLoop_dictionary = None
        self.day_level_cluster_level_generalModelTraining_dictionary = None
        self.day_level_medical_center_level_modelFineTuning_dictionary = None

        if not lazy:
            self.execute_data_imputation_and_averaging()
            self.execute_data_delimiting()
            self.execute_clustering()
            self.execute_feature_selection()
            self.execute_populating_dictionary_outputs()
            self.execute_autoforecast_loop()


    # core functions
    def execute_data_imputation_and_averaging(self):
        if self.medical_center_subset is not None:
            self.filter_by_medical_center = True

        if self.time_period_subset is not None:
            self.filter_by_time_period = True

        dataImputationAndAveraging = DataImputationAndAveraging(dataset_path=self.dataset_path, number_of_days_for_data_averaging=self.number_of_days_for_data_averaging, filter_by_medical_center=self.filter_by_medical_center, filter_by_time_period=self.filter_by_time_period, medical_center_subset=self.medical_center_subset, time_period=self.time_period_subset)
        self.processed_dataset = dataImputationAndAveraging.processed_dataset

    def execute_data_delimiting(self):
        self.dataDelimitingByDayAndMedicalCenter = DataDelimitingByDayAndMedicalCenter(dataset=self.processed_dataset, number_of_days_for_testing=self.number_of_rolling_forecast_days, number_of_days_to_predict_ahead=self.number_of_days_to_predict_ahead)

    def execute_clustering(self):
        self.dataCluseteringAndNormalization = DataCluseteringAndNormalization(dataDelimitingByDayAndMedicalCenter=self.dataDelimitingByDayAndMedicalCenter, number_of_test_day_in_day_level_DataFrame=self.number_of_test_days_in_a_day_level_DataFrame, max_number_of_cluster=self.max_number_of_cluster, do_clustering=self.do_clustering, do_normalization=False, operation_interval=self.tuning_interval)  # do not do normalization as we will do it again in evaluator anyways; only do clustering

    def execute_feature_selection(self):
        self.dataPreparationForFeatureSelection = DataPreparationForFeatureSelection( self.dataCluseteringAndNormalization.day_level_cluster_level_not_normalized_combined_DataFrame_dictionary, number_of_test_days_in_DataFrame=self.number_of_test_days_in_a_day_level_DataFrame, number_of_days_to_predict_ahead=self.number_of_days_to_predict_ahead)
        self.featureSelectionGlobal = FeatureSelectionGlobal(dataPreparationForFeatureSelection=self.dataPreparationForFeatureSelection, p_value_threshold=self.p_value_threshold, enforce_using_the_prediction_target_as_a_selected_feature=True, do_feature_selection=self.do_feature_selection, operation_interval=self.tuning_interval)

    def execute_populating_dictionary_outputs(self):
        day_keys = self.dataCluseteringAndNormalization.day_level_cluster_level_not_normalized_combined_DataFrame_dictionary.keys()

        self.day_level_cluster_level_hyperParameterTuningLoop_dictionary = DictionaryWrangling.create_empty_nested_dictionary(day_keys)
        self.day_level_cluster_level_generalModelTraining_dictionary = DictionaryWrangling.create_empty_nested_dictionary(day_keys)
        self.day_level_medical_center_level_modelFineTuning_dictionary = DictionaryWrangling.create_empty_nested_dictionary(day_keys)

    def execute_autoforecast_loop(self):
        for day in self.dataCluseteringAndNormalization.day_level_cluster_level_not_normalized_combined_DataFrame_dictionary.keys():
            for cluster in self.dataCluseteringAndNormalization.day_level_cluster_level_not_normalized_combined_DataFrame_dictionary[day].keys():
                dataframe_for_building_general_model = self.dataCluseteringAndNormalization.day_level_cluster_level_not_normalized_combined_train_DataFrame_dictionary[day][cluster]
                index_of_selected_features = self.featureSelectionGlobal.day_level_agency_level_feature_index_list_dictionary[day][cluster]  # feature selection assumes dataframe starts from the case column
                adapted_index_of_selected_features = [i+1 for i in index_of_selected_features]  # move the index to the right by 1, accouting for the clinic column
                clinic_column_added_adapted_index_of_selected_features = [0] + adapted_index_of_selected_features  # include clinic column as well so normalization can be done
                feature_selected_dataframe_for_building_general_model = dataframe_for_building_general_model.iloc[:, clinic_column_added_adapted_index_of_selected_features]

                worker_specific_parameter_to_value_dictionary = {'DataFrame': feature_selected_dataframe_for_building_general_model,
                    'loss_function': self.loss_function,
                    'number_of_days_to_predict_ahead': self.number_of_days_to_predict_ahead,
                    'number_of_test_days_in_a_day_level_DataFrame': self.hyper_parameter_tuning_number_of_test_days_in_DataFrame,
                    'number_of_rolling_forecast_days': self.hyper_parameter_tuning_number_of_rolling_forecast_days,
                    'data_windowing_option': self.data_windowing_option,
                    'input_length': self.input_length,
                    'lstm_is_many_to_many': self.lstm_is_many_to_many,
                    'do_normalization': self.do_normalization,
                    'batch_size': self.batch_size_for_tuning,
                    'enable_data_caching': self.enable_data_caching_during_tuning,
                    'generate_train_prediction': self.generate_train_prediction_during_tuning
                }

                self.hyper_parameter_space.seed(self.hyper_parameter_space_seed)
                np.random.seed(self.numpy_seed)
                torch.manual_seed(self.neural_network_training_seed)

                operationIntervalModerator = OperationIntervalModerator(days=self.dataCluseteringAndNormalization.day_level_cluster_level_not_normalized_combined_DataFrame_dictionary.keys(), operation_interval=self.tuning_interval)
                if operationIntervalModerator.day_is_operation_day(day=day):
                    hyperParameterTuningLoop = HyperParameterTuningLoop(CustomizedWorker=self.CustomizedWorker, worker_specific_parameter_to_value_dictionary=worker_specific_parameter_to_value_dictionary, hyper_parameter_space=self.hyper_parameter_space, nameserver_run_id=self.nameserver_run_id, nameserver_address=self.nameserver_address, nameserver_port=self.nameserver_port, min_budget=self.min_budget, max_budget=self.max_budget, n_iterations=self.n_iterations)
                else:
                    current_operation_day = operationIntervalModerator.get_operation_day_of_a_day(day=day)
                    hyperParameterTuningLoop = copy.deepcopy(self.day_level_cluster_level_hyperParameterTuningLoop_dictionary[current_operation_day][cluster])

                self.day_level_cluster_level_hyperParameterTuningLoop_dictionary[day][cluster] = hyperParameterTuningLoop
                searched_hyper_parameter_value_combination = hyperParameterTuningLoop.optimal_partial_hyper_parameter_value_combination_found_among_max_budget_evaluations
                searched_hyper_parameter_value_combination['number_of_training_epochs'] = self.max_budget

                torch.manual_seed(self.neural_network_training_seed)
                if self.do_general_model_training:
                    generalModelTraining = GeneralModelTraining(hyper_parameter_value_combination=searched_hyper_parameter_value_combination, train_DataFrame=feature_selected_dataframe_for_building_general_model, loss_function=self.loss_function, data_windowing_option=self.data_windowing_option, input_length=self.input_length, lstm_is_many_to_many=self.lstm_is_many_to_many, number_of_days_to_predict_ahead=self.number_of_days_to_predict_ahead, do_normalization=self.do_normalization, batch_size=self.batch_size_for_general_model_building)
                else:
                    no_general_model_training_searched_hyper_parameter_value_combination = copy.deepcopy(searched_hyper_parameter_value_combination)
                    no_general_model_training_searched_hyper_parameter_value_combination['number_of_training_epochs'] = 0
                    generalModelTraining = GeneralModelTraining(hyper_parameter_value_combination=no_general_model_training_searched_hyper_parameter_value_combination, train_DataFrame=feature_selected_dataframe_for_building_general_model, loss_function=self.loss_function, data_windowing_option=self.data_windowing_option, input_length=self.input_length, lstm_is_many_to_many=self.lstm_is_many_to_many, number_of_days_to_predict_ahead=self.number_of_days_to_predict_ahead, do_normalization=self.do_normalization, batch_size=self.batch_size_for_general_model_building)
                # generalModelTraining = GeneralModelTraining(hyper_parameter_value_combination=searched_hyper_parameter_value_combination, train_DataFrame=feature_selected_dataframe_for_building_general_model, loss_function=self.loss_function, data_windowing_option=self.data_windowing_option, input_length=self.input_length, lstm_is_many_to_many=self.lstm_is_many_to_many, number_of_days_to_predict_ahead=self.number_of_days_to_predict_ahead, do_normalization=self.do_normalization, batch_size=self.batch_size_for_general_model_building)
                self.day_level_cluster_level_generalModelTraining_dictionary[day][cluster] = generalModelTraining
                general_model = generalModelTraining.general_model

                for medical_center in self.dataCluseteringAndNormalization.day_level_cluster_level_medical_center_list_dictionary[day][cluster]:
                    dataframe_for_building_fine_tuned_model = self.dataDelimitingByDayAndMedicalCenter.day_level_medical_center_level_DataFrame_dictionary[day][medical_center]
                    feature_selected_dataframe_for_building_fine_tuned_model = dataframe_for_building_fine_tuned_model.iloc[:, clinic_column_added_adapted_index_of_selected_features]

                    torch.manual_seed(self.neural_network_training_seed)
                    # '''
                    if self.do_fine_tuning:
                        modelFineTuning = ModelFineTuning(hyper_parameter_value_combination=searched_hyper_parameter_value_combination, trainable_parameter_name_list=self.trainable_parameter_name_list, general_model=general_model, dataframe=feature_selected_dataframe_for_building_fine_tuned_model, number_of_days_to_predict_ahead=self.number_of_days_to_predict_ahead, loss_function=self.loss_function, data_windowing_option=self.data_windowing_option, input_length=self.input_length, lstm_is_many_to_many=self.lstm_is_many_to_many, do_normalization=self.do_normalization, batch_size=self.batch_size_for_fine_tuning, number_of_test_days_in_a_day_level_DataFrame=self.number_of_test_days_in_a_day_level_DataFrame, number_of_days_for_testing_in_a_hyper_parameter_tuning_run=1, verbose=self.verbose)  # number_of_days_for_testing_in_a_hyper_parameter_tuning_run should be 1 because we are in a rolling forecast day and only one test day is available
                    else:
                        no_fine_tuning_searched_hyper_parameter_value_combination = copy.deepcopy(searched_hyper_parameter_value_combination)
                        no_fine_tuning_searched_hyper_parameter_value_combination['number_of_training_epochs'] = 0
                        modelFineTuning = ModelFineTuning(hyper_parameter_value_combination=no_fine_tuning_searched_hyper_parameter_value_combination, trainable_parameter_name_list=self.trainable_parameter_name_list, general_model=general_model, dataframe=feature_selected_dataframe_for_building_fine_tuned_model, number_of_days_to_predict_ahead=self.number_of_days_to_predict_ahead, loss_function=self.loss_function, data_windowing_option=self.data_windowing_option, input_length=self.input_length, lstm_is_many_to_many=self.lstm_is_many_to_many, do_normalization=self.do_normalization, batch_size=self.batch_size_for_fine_tuning, number_of_test_days_in_a_day_level_DataFrame=self.number_of_test_days_in_a_day_level_DataFrame, number_of_days_for_testing_in_a_hyper_parameter_tuning_run=1, verbose=self.verbose)  # number_of_days_for_testing_in_a_hyper_parameter_tuning_run should be 1 because we are in a rolling forecast day and only one test day is available
                    # '''
                    # modelFineTuning = ModelFineTuning(hyper_parameter_value_combination=searched_hyper_parameter_value_combination, trainable_parameter_name_list=self.trainable_parameter_name_list, general_model=general_model, dataframe=feature_selected_dataframe_for_building_fine_tuned_model, number_of_days_to_predict_ahead=self.number_of_days_to_predict_ahead, loss_function=self.loss_function, data_windowing_option=self.data_windowing_option, input_length=self.input_length, lstm_is_many_to_many=self.lstm_is_many_to_many, do_normalization=self.do_normalization, batch_size=self.batch_size_for_fine_tuning, number_of_test_days_in_a_day_level_DataFrame=self.number_of_test_days_in_a_day_level_DataFrame, number_of_days_for_testing_in_a_hyper_parameter_tuning_run=1, verbose=self.verbose)  # number_of_days_for_testing_in_a_hyper_parameter_tuning_run should be 1 because we are in a rolling forecast day and only one test day is available
                    self.day_level_medical_center_level_modelFineTuning_dictionary[day][medical_center] = modelFineTuning