import unittest
import seaborn as sns
import pandas as pd
import os
import ConfigSpace as CS
import ConfigSpace.hyperparameters as CSH
from autoForecastVA.src.analyzer.analyzer_pipeline_arima import AnalyzerPipeLineArima
from torch import nn
import matplotlib.pylab as plt

from autoForecastVA.src.analyzer.analyzer_pipeline_autoForecastVA import AnalyzerPipeLineAutoForecastVA
from autoForecastVA.src.analyzer.analyzer_pipeline_varma import AnalyzerPipeLineVarma
from autoForecastVA.src.components.hyper_parameter_tuning.hyper_parameter_tuning_loop import HyperParameterTuningLoop
from autoForecastVA.src.components.hyper_parameter_tuning.hyper_parameter_tuning_worker import LSTMWorker
from autoForecastVA.src.data_caching.variable_load_and_save import VariableSaverAndLoader
from autoForecastVA.src.pipelines.pipeline_autoForecastVA import PipeLineAutoForecastVA


class TestAnalyzerPipeLineArima(unittest.TestCase):
    def test_toy_to_analyzer(self):
        # analyzerPipeLineArima = VariableSaverAndLoader(load=True, file_name='../benchmark/test_results_arima/299dfbb14252cd8084f8c9d4c6afc08e.dat').list_of_loaded_variables[0]  # avg = 3
        # analyzerPipeLineArima = VariableSaverAndLoader(load=True, file_name='../benchmark/test_results_arima/606d03270f2bc5f7bf62fc49aa5930ef.dat').list_of_loaded_variables[0]  # avg = 2
        # analyzerPipeLineArima = VariableSaverAndLoader(load=True, file_name='../benchmark/test_results_arima/7ffaaeac00c07fdc07343da41e0c7319.dat').list_of_loaded_variables[0]  # avg = 1
        analyzerPipeLineArima = VariableSaverAndLoader(load=True, file_name='../benchmark/test_results_arima_one_tune/9f411b79f40136748f07a17c762d00a7.dat').list_of_loaded_variables[0]  # tuning interval=30

        print(analyzerPipeLineArima.day_to_clinic_to_index_to_train_prediction_target_dataframe.iloc[1232, 2])
        print(analyzerPipeLineArima.day_to_clinic_to_test_prediction_target_dataframe.iloc[5, 1])
        print(analyzerPipeLineArima.day_to_clinic_to_index_to_train_prediction_dataframe.iloc[1032, 2])
        print(analyzerPipeLineArima.day_to_clinic_to_test_prediction_dataframe.iloc[5, 1])
        print(analyzerPipeLineArima.day_to_clinic_to_test_prediction_loss_dataframe.iloc[5, 3])
        print(analyzerPipeLineArima.clinic_to_test_prediction_loss_dataframe.iloc[2, 1])
        print(analyzerPipeLineArima.test_prediction_loss)

        # fig = AnalyzerPipeLineArima.make_figure_for_all_clinic_test_target_and_prediction(analyzerPipeLineArima.day_to_clinic_to_test_prediction_dataframe, analyzerPipeLineArima.day_to_clinic_to_test_prediction_target_dataframe)
        # fig.savefig('all_clinic_test_prediction_vs_target.png')
        # fig.savefig('all_clinic_test_prediction_vs_target_one_day_average.png')
