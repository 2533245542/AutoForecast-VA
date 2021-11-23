from autoForecastVA.src.components.data_preprocessing_global.data_clustering_and_normalization import \
    DataCluseteringAndNormalization
from autoForecastVA.src.components.data_preprocessing_global.data_delimiting_by_day_and_medical_center import \
    DataDelimitingByDayAndMedicalCenter
from autoForecastVA.src.components.data_preprocessing_global.data_imputation_and_averaging import \
    DataImputationAndAveraging
from autoForecastVA.src.components.data_preprocessing_global.data_preparation_for_feature_selection import \
    DataPreparationForFeatureSelection
from autoForecastVA.src.components.feature_selection_global.feature_selection_global import FeatureSelectionGlobal
from autoForecastVA.src.components.models.statistical_models.varma_model import VarmaModel
from autoForecastVA.src.utils.dictionary_wrangling import DictionaryWrangling
from autoForecastVA.src.utils.tuning_interval_moderator import OperationIntervalModerator


class PipelineVarma():
    '''Do a run of VARMA on all test data'''
    def __init__(self, medical_center_subset=None, time_period_subset=None, dataset_path='../../data/coviddata07292020.csv', number_of_days_for_data_averaging=3, number_of_days_to_predict_ahead=1, number_of_test_days_in_a_day_level_DataFrame=1, p_value_threshold=0.05, number_of_rolling_forecast_days=30, tuning_frequency=9999999, do_feature_selection=True, order_of_autoregression_lower=0, order_of_autoregression_upper=5, order_of_moving_average_lower=0, order_of_moving_average_upper=5, max_model_building_convergence_iteration=1000, verbose=True, lazy=False):
        if medical_center_subset is None:
            self.medical_center_subset = ['402', '405', '436', '437', '438', '442', '459', '460', '463', '501', '503', '506', '508', '509', '512', '515', '516', '518', '519', '521', '523', '528', '528A6', '528A7', '528A8', '529', '531', '534', '537', '539', '541', '542', '544', '548', '549', '550', '552', '554', '556', '557', '558', '562', '568', '570', '573', '575', '578', '580', '581', '583', '585', '589', '589A5', '589A7', '590', '593', '595', '596', '600', '603', '605', '607', '608', '610', '612A4', '614', '618', '619', '621', '623', '626', '631', '635', '636', '636A6', '636A8', '637', '640', '642', '644', '646', '648', '649', '650', '652', '654', '655', '656', '657', '657A5', '658', '659', '660', '662', '663', '664', '668', '671', '673', '674', '675', '676', '678', '688', '689', '691', '693', '695', '756', '757']
        else:
            self.medical_center_subset = medical_center_subset
        self.time_period_subset = time_period_subset
        self.dataset_path = dataset_path

        self.number_of_days_for_data_averaging = number_of_days_for_data_averaging
        self.number_of_days_to_predict_ahead = number_of_days_to_predict_ahead
        self.number_of_test_days_in_a_day_level_DataFrame = number_of_test_days_in_a_day_level_DataFrame

        self.p_value_threshold = p_value_threshold

        self.number_of_rolling_forecast_days = number_of_rolling_forecast_days

        self.tuning_frequency = tuning_frequency

        self.do_feature_selection = do_feature_selection  # default to True; if False, all features are selected

        self.order_of_autoregression_lower = order_of_autoregression_lower
        self.order_of_autoregression_upper = order_of_autoregression_upper
        self.order_of_moving_average_lower = order_of_moving_average_lower
        self.order_of_moving_average_upper = order_of_moving_average_upper

        self.max_model_building_convergence_iteration = max_model_building_convergence_iteration

        self.verbose = verbose

        self.lazy = lazy

        # outputs
        self.filter_by_medical_center = False
        self.filter_by_time_period = False

        self.dataImputationAndAveraging = None
        self.processed_dataset = None

        self.dataDelimitingByDayAndMedicalCenter = None

        self.dataCluseteringAndNormalization = None

        self.dataPreparationForFeatureSelection = None
        self.featureSelectionGlobal = None

        self.day_level_clinic_level_varmaModel_dictionary = None

        if not lazy:
            self.execute_data_imputation_and_averaging()
            self.execute_data_delimiting()
            self.execute_clustering()
            self.execute_feature_selection()
            self.execute_populating_dictionary_outputs()
            self.execute_varma_loop()

    # core functions
    def execute_data_imputation_and_averaging(self):
        if self.medical_center_subset is not None:
            self.filter_by_medical_center = True

        if self.time_period_subset is not None:
            self.filter_by_time_period = True

        dataImputationAndAveraging = DataImputationAndAveraging(dataset_path=self.dataset_path, number_of_days_for_data_averaging=self.number_of_days_for_data_averaging, filter_by_medical_center=self.filter_by_medical_center, filter_by_time_period=self.filter_by_time_period, medical_center_subset=self.medical_center_subset, time_period=self.time_period_subset)
        self.processed_dataset = dataImputationAndAveraging.processed_dataset
        dataImputationAndAveraging.processed_dataset.std()

    def execute_data_delimiting(self):
        self.dataDelimitingByDayAndMedicalCenter = DataDelimitingByDayAndMedicalCenter(dataset=self.processed_dataset, number_of_days_for_testing=self.number_of_rolling_forecast_days, number_of_days_to_predict_ahead=self.number_of_days_to_predict_ahead)

    def execute_clustering(self):
        self.dataCluseteringAndNormalization = DataCluseteringAndNormalization(dataDelimitingByDayAndMedicalCenter=self.dataDelimitingByDayAndMedicalCenter, number_of_test_day_in_day_level_DataFrame=self.number_of_test_days_in_a_day_level_DataFrame, operation_interval=self.tuning_frequency, max_number_of_cluster=3, do_clustering=False, do_normalization=False)  # do not do normalization as we will do it again in evaluator anyways; only do clustering; HACK: the implemetation of DataCluseteringAndNormalization makes it always do clustering, so set max_number_of_cluster=3 to reduce time.

    def execute_feature_selection(self):
        self.dataPreparationForFeatureSelection = DataPreparationForFeatureSelection( self.dataCluseteringAndNormalization.day_level_cluster_level_not_normalized_combined_DataFrame_dictionary, number_of_test_days_in_DataFrame=self.number_of_test_days_in_a_day_level_DataFrame, number_of_days_to_predict_ahead=self.number_of_days_to_predict_ahead)
        self.featureSelectionGlobal = FeatureSelectionGlobal(dataPreparationForFeatureSelection=self.dataPreparationForFeatureSelection, p_value_threshold=self.p_value_threshold, enforce_using_the_prediction_target_as_a_selected_feature=True, do_feature_selection=self.do_feature_selection, operation_interval=self.tuning_frequency)

    def execute_populating_dictionary_outputs(self):
        day_keys = self.dataCluseteringAndNormalization.day_level_cluster_level_not_normalized_combined_DataFrame_dictionary.keys()

        self.day_level_clinic_level_varmaModel_dictionary = DictionaryWrangling.create_empty_nested_dictionary(day_keys)

    def execute_varma_loop(self):
        for day in self.dataCluseteringAndNormalization.day_level_cluster_level_not_normalized_combined_DataFrame_dictionary.keys():
            print('Pipeline VARMA before runnining on day ' + str(day))   # from 2020-06-28 to 2020-07-27
            for cluster in self.dataCluseteringAndNormalization.day_level_cluster_level_not_normalized_combined_DataFrame_dictionary[day].keys():
                index_of_selected_features = self.featureSelectionGlobal.day_level_agency_level_feature_index_list_dictionary[day][cluster]  # feature selection assumes dataframe starts from the case column
                adapted_index_of_selected_features = [i+1 for i in index_of_selected_features]  # move the index to the right by 1, accouting for the clinic column
                clinic_column_added_adapted_index_of_selected_features = [0] + adapted_index_of_selected_features  # include clinic column as well so normalization can be done

                for medical_center in self.dataCluseteringAndNormalization.day_level_cluster_level_medical_center_list_dictionary[day][cluster]:
                    print('Pipeline VARMA before runnining on clinic ' + str(medical_center))
                    dataframe_for_building_fine_tuned_model = self.dataDelimitingByDayAndMedicalCenter.day_level_medical_center_level_DataFrame_dictionary[day][medical_center]
                    feature_selected_dataframe_for_building_fine_tuned_model = dataframe_for_building_fine_tuned_model.iloc[:, clinic_column_added_adapted_index_of_selected_features]
                    dataframe_for_building_varma_model = feature_selected_dataframe_for_building_fine_tuned_model.drop(columns='clinic')

                    operationIntervalModerator = OperationIntervalModerator(days=self.dataCluseteringAndNormalization.day_level_cluster_level_not_normalized_combined_DataFrame_dictionary.keys(), operation_interval=self.tuning_frequency)

                    if operationIntervalModerator.day_is_operation_day(day=day):
                        varmaModel = VarmaModel(dataframe=dataframe_for_building_varma_model, order_of_autoregression_lower=self.order_of_autoregression_lower, order_of_autoregression_upper=self.order_of_autoregression_upper, order_of_moving_average_lower=self.order_of_moving_average_lower, order_of_moving_average_upper=self.order_of_moving_average_upper, max_model_building_convergence_iteration=self.max_model_building_convergence_iteration)

                    else:
                        # create varmaModel with the VA and MA order found in the tuning day for this clinic
                        tuning_day = operationIntervalModerator.get_operation_day_of_a_day(day=day)
                        tuning_day_varmaModel = self.day_level_clinic_level_varmaModel_dictionary[tuning_day][medical_center]
                        selected_order_of_autoregression = tuning_day_varmaModel.selected_order_of_autoregression
                        selected_order_of_moving_average = tuning_day_varmaModel.selected_order_of_moving_average  # tuning_day_varmaModel.built_model_order_to_info_dictionary
                        varmaModel = VarmaModel(dataframe=dataframe_for_building_varma_model, order_of_autoregression_lower=selected_order_of_autoregression, order_of_autoregression_upper=selected_order_of_autoregression, order_of_moving_average_lower=selected_order_of_moving_average, order_of_moving_average_upper=selected_order_of_moving_average, max_model_building_convergence_iteration=self.max_model_building_convergence_iteration)

                    # varmaModel = VarmaModel(dataframe=dataframe_for_building_varma_model, order_of_autoregression_lower=self.order_of_autoregression_lower, order_of_autoregression_upper=self.order_of_autoregression_upper, order_of_moving_average_lower=self.order_of_moving_average_lower, order_of_moving_average_upper=self.order_of_moving_average_upper, max_model_building_convergence_iteration=self.max_model_building_convergence_iteration) # TODO delete this line

                    self.day_level_clinic_level_varmaModel_dictionary[day][medical_center] = varmaModel
