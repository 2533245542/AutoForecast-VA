import unittest
import seaborn as sns
import pandas as pd
import os
import ConfigSpace as CS
import ConfigSpace.hyperparameters as CSH
from torch import nn
import matplotlib.pylab as plt

from autoForecastVA.src.analyzer.analyzer_pipeline_autoForecastVA import AnalyzerPipeLineAutoForecastVA
from autoForecastVA.src.analyzer.analyzer_pipeline_varma import AnalyzerPipeLineVarma
from autoForecastVA.src.components.hyper_parameter_tuning.hyper_parameter_tuning_loop import HyperParameterTuningLoop
from autoForecastVA.src.components.hyper_parameter_tuning.hyper_parameter_tuning_worker import LSTMWorker
from autoForecastVA.src.data_caching.variable_load_and_save import VariableSaverAndLoader
from autoForecastVA.src.pipelines.pipeline_autoForecastVA import PipeLineAutoForecastVA


class TestAnalyzerPipeLineVarma(unittest.TestCase):
    def test_toy_to_analyzer(self):
        pipelineVarma = VariableSaverAndLoader(load=True, file_name='../pipelines/data/toy_pipeline_varma.dat').list_of_loaded_variables[0]

        # days = list(pipelineVarma.day_level_clinic_level_varmaModel_dictionary.keys())
        # clinics = list(pipelineVarma.day_level_clinic_level_varmaModel_dictionary[days[0]].keys())

        analyzerPipeLineVarma = AnalyzerPipeLineVarma(pipelineVarma)

        self.assertAlmostEqual(analyzerPipeLineVarma.day_to_clinic_to_build_info_dataframe.iloc[107, 2], 5237.83557, places=5)
        self.assertAlmostEqual(analyzerPipeLineVarma.day_to_clinic_to_index_to_train_prediction_target_dataframe.iloc[1232, 2], 3.333333, places=5)
        self.assertAlmostEqual(analyzerPipeLineVarma.day_to_clinic_to_test_prediction_target_dataframe.iloc[5, 1], 4.33333, places=5)
        self.assertAlmostEqual(analyzerPipeLineVarma.day_to_clinic_to_index_to_train_prediction_dataframe.iloc[1032, 2], 7.60145, places=5)
        self.assertAlmostEqual(analyzerPipeLineVarma.day_to_clinic_to_test_prediction_dataframe.iloc[5, 1], 4.30135, places=5)
        self.assertAlmostEqual(analyzerPipeLineVarma.day_to_clinic_to_test_prediction_loss_dataframe.iloc[5, 3], 0.001023, places=5)
        self.assertAlmostEqual(analyzerPipeLineVarma.clinic_to_test_prediction_loss_dataframe.iloc[2, 1], 9.319289, places=5)
        self.assertAlmostEqual(analyzerPipeLineVarma.test_prediction_loss, 3.458374, places=5)
