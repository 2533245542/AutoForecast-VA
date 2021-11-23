import unittest
import os
import sys
import logging

from autoForecastVA.src.analyzer.analyzer_aggregate import AnalyzerAggregate
from autoForecastVA.src.analyzer.analyzer_pipeline_autoForecastVA import AnalyzerPipeLineAutoForecastVA


class TestAnalyzerAggregate(unittest.TestCase):
    def test_toy_example(self):
        pipeline_to_seeded_clones_mapping = {
            'tuning_frequency_30':
                ['../../benchmark/results_no_averaging/64a00f6965f53ab104d91a84493eee72',
                 '../../benchmark/results_no_averaging/fe8e43483a1ab7c23c6baef38f78ea70',
                 '../../benchmark/results_no_averaging/6047b93f88cb8a50952d369598db3f6f',
                 '../../benchmark/results_no_averaging/bb6829805b7d0d8aaae040686778af3f',
                 '../../benchmark/results_no_averaging/2a745dc595491c9e19e941f8c9e6a034'],

            'tuning_frequency_10': [
                '../../benchmark/results_no_averaging/70fc3d92923ed60c57285c7deb219104',
                '../../benchmark/results_no_averaging/706e4ec90b965635e16eb1c2a6037a31',
                '../../benchmark/results_no_averaging/7a8d07a464e014626de6005236571cd8',
                '../../benchmark/results_no_averaging/a7c6b24154b769f2a20b63ea7f2c319e',
                '../../benchmark/results_no_averaging/997ff12785136880a07c8bb6e5e97140'],

            'tuning_frequency_5': [
                '../../benchmark/results_no_averaging/f1e562838e22e91f27bc8bfa0f3e9a61',
                '../../benchmark/results_no_averaging/164b15f698a4016bcc75a9ae6993f1d6',
                '../../benchmark/results_no_averaging/7ceea049caf1f9f95d44284e76e3bf2b',
                '../../benchmark/results_no_averaging/f6c1c5cec0366b7a1dfce510b6096d3e',
                '../../benchmark/results_no_averaging/662e6c32a7e8477c5ef597d3e7704bb9']
        }

        list_of_aggregatable_variable_names = ['day_to_clinic_to_test_prediction_dataframe', 'day_to_clinic_to_test_prediction_target_dataframe', 'test_prediction_loss']

        for pipeline_experiment_name in pipeline_to_seeded_clones_mapping.keys():
            analyzerAggregate = AnalyzerAggregate(list_of_result_files=pipeline_to_seeded_clones_mapping[pipeline_experiment_name], list_of_aggregatable_variable_names=list_of_aggregatable_variable_names)
            day_to_clinic_to_test_prediction_target_dataframe = analyzerAggregate.variable_name_to_aggregated_variable_mapping['day_to_clinic_to_test_prediction_target_dataframe']
            day_to_clinic_to_test_prediction_dataframe = analyzerAggregate.variable_name_to_aggregated_variable_mapping['day_to_clinic_to_test_prediction_dataframe']

            # clinic_to_test_prediction_loss_dataframe = AnalyzerPipeLineAutoForecastVA.get_clinic_to_test_prediction_loss_dataframe(day_to_clinic_to_test_prediction_target_dataframe=day_to_clinic_to_test_prediction_target_dataframe, day_to_clinic_to_test_prediction_dataframe=day_to_clinic_to_test_prediction_dataframe)

            test_prediction_loss = AnalyzerPipeLineAutoForecastVA.get_test_prediction_loss(day_to_clinic_to_test_prediction_target_dataframe=day_to_clinic_to_test_prediction_target_dataframe, day_to_clinic_to_test_prediction_dataframe=day_to_clinic_to_test_prediction_dataframe)
            test_prediction_loss_mean = analyzerAggregate.variable_name_to_aggregated_variable_mapping['test_prediction_loss_mean']
            test_prediction_loss_std = analyzerAggregate.variable_name_to_aggregated_variable_mapping['test_prediction_loss_std']
            print(pipeline_experiment_name + ' single loss: ' + str(test_prediction_loss))
            print(pipeline_experiment_name + ' mean loss: ' + str(test_prediction_loss_mean))
            print(pipeline_experiment_name + ' std loss: ' + str(test_prediction_loss_std))