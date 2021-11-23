import unittest
import pickle
import inspect
import sys
from itertools import chain
from collections import deque
from sys import getsizeof, stderr

from autoForecastVA.src.analyzer.analyzer_pipeline_varma import AnalyzerPipeLineVarma
from autoForecastVA.src.data_caching.variable_load_and_save import VariableSaverAndLoader
from autoForecastVA.src.pipelines.pipeline_varma import PipelineVarma


class TestPipelineVarma(unittest.TestCase):
    def test_toy_example(self):
        """Test a VARMA pipeline."""
        medical_center_subset = ['402', '405', '436']
        # time_period_subset = ['2020-3-1', '2020-5-1']
        time_period_subset = None
        pipelineVarma = PipelineVarma(medical_center_subset=medical_center_subset, time_period_subset=time_period_subset, number_of_rolling_forecast_days=2, order_of_autoregression_upper=2, order_of_moving_average_upper=2, tuning_frequency=1, max_model_building_convergence_iteration=20, do_feature_selection=True, verbose=True)
        VariableSaverAndLoader(save=True, list_of_variables_to_save=[pipelineVarma], file_name='data/toy_pipeline_varma.dat')
        pipelineVarma = VariableSaverAndLoader(load=True, file_name='data/toy_pipeline_varma.dat').list_of_loaded_variables[0]
        days = list(pipelineVarma.day_level_clinic_level_arimaModel_dictionary.keys())
        clinics = list(pipelineVarma.day_level_clinic_level_arimaModel_dictionary[days[0]].keys())

        varmaModel = pipelineVarma.day_level_clinic_level_arimaModel_dictionary[days[0]][clinics[0]]
        self.assertAlmostEqual(varmaModel.built_model_order_to_info_dictionary[(0,2)]['aic'], 10785.33942, places=5)
        self.assertAlmostEqual(varmaModel.selected_built_model.aic, -1202.17012, places=5)
        self.assertAlmostEqual(varmaModel.test_predictions.iloc[0,0], 4.23953, places=5)

    def test_toy_example_no_feature_selection_and_tuning_frequency_is_9999999(self):
        """Test a VARMA pipeline."""
        medical_center_subset = ['402', '405', '436']
        time_period_subset = None
        pipelineVarma = PipelineVarma(medical_center_subset=medical_center_subset, time_period_subset=time_period_subset, number_of_rolling_forecast_days=2, order_of_autoregression_upper=2, order_of_moving_average_upper=2, max_model_building_convergence_iteration=20, do_feature_selection=False, verbose=True)
        VariableSaverAndLoader(save=True, list_of_variables_to_save=[pipelineVarma], file_name='data/toy_pipeline_varma_no_feature_selection_and_tuning_frequency_is_9999999.dat')
        pipelineVarma = VariableSaverAndLoader(load=True, file_name='data/toy_pipeline_varma_no_feature_selection_and_tuning_frequency_is_9999999.dat').list_of_loaded_variables[0]
        days = list(pipelineVarma.day_level_clinic_level_arimaModel_dictionary.keys())
        clinics = list(pipelineVarma.day_level_clinic_level_arimaModel_dictionary[days[0]].keys())

        varmaModel = pipelineVarma.day_level_clinic_level_arimaModel_dictionary[days[0]][clinics[0]]
        self.assertAlmostEqual(varmaModel.built_model_order_to_info_dictionary[(1,1)]['aic'], 1030830.02978, places=5)
        self.assertAlmostEqual(varmaModel.selected_built_model.aic, 1030318.029782535, places=5)
        self.assertAlmostEqual(varmaModel.test_predictions.iloc[0,0], 10.01626, places=5)

    def test_experimenting(self):
        """Just experimenting."""
        medical_center_subset = ['402', '405', '436', '437', '438', '442', '459', '460', '463', '501', '503', '506', '508', '509', '512', '515', '516', '518', '519', '521', '523', '528', '528A6', '528A7', '528A8', '529', '531', '534', '537', '539', '541', '542', '544', '548', '549', '550', '552', '554', '556', '557', '558', '562', '568', '570', '573', '575', '578', '580', '581', '583', '585', '589', '589A5', '589A7', '590', '593', '595', '596', '600', '603', '605', '607', '608', '610', '612A4', '614', '618', '619', '621', '623', '626', '631', '635', '636', '636A6', '636A8', '637', '640', '642', '644', '646', '648', '649', '650', '652', '654', '655', '656', '657', '657A5', '658', '659', '660', '662', '663', '664', '668', '671', '673', '674', '675', '676', '678', '688', '689', '691', '693', '695', '756', '757']
        time_period_subset = None
        # time_period_subset = ['2020-3-1', '2020-7-27']
        pipelineVarma = PipelineVarma(medical_center_subset=medical_center_subset, time_period_subset=time_period_subset, number_of_rolling_forecast_days=2, order_of_autoregression_upper=3, order_of_moving_average_upper=3, max_model_building_convergence_iteration=1000, verbose=True)
        VariableSaverAndLoader(save=True, list_of_variables_to_save=[pipelineVarma], file_name='data/test_experimenting.dat')
        pipelineVarma = VariableSaverAndLoader(load=True, file_name='data/test_experimenting.dat').list_of_loaded_variables[0]
        days = list(pipelineVarma.day_level_clinic_level_arimaModel_dictionary.keys())
        clinics = list(pipelineVarma.day_level_clinic_level_arimaModel_dictionary[days[0]].keys())
        varmaModel = pipelineVarma.day_level_clinic_level_arimaModel_dictionary[days[0]][clinics[0]]

        analyzerPipeLineVarma = AnalyzerPipeLineVarma(pipelineVarma)
