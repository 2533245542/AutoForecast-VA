import unittest

from autoForecastVA.src.analyzer.analyzer_pipeline_autoForecastVA import AnalyzerPipeLineAutoForecastVA
from autoForecastVA.src.pipelines.pipeline_ablation_studies import PipeLineTuningInterval


class TestPipeLineAblationStudies(unittest.TestCase):
    def test_toy(self):
        medical_center_subset = ['402', '405', '436', '437', '438', '442', '459', '460', '463', '501']  # default to be None
        time_period_subset = ['2020-3-1', '2020-4-1']  # default to be None
        max_number_of_cluster = 3
        input_length = 3
        number_of_rolling_forecast_days = 5
        hyper_parameter_tuning_number_of_test_days_in_DataFrame = 3  # usually greater than one
        n_iterations = 2
        tuning_interval = 2

        # pipeLineTuningInterval = PipeLineTuningInterval(do_clustering=False, do_general_model_training=False, medical_center_subset=medical_center_subset, time_period_subset=time_period_subset, max_number_of_cluster=max_number_of_cluster, input_length=input_length, number_of_rolling_forecast_days=number_of_rolling_forecast_days, hyper_parameter_tuning_number_of_rolling_forecast_days=hyper_parameter_tuning_number_of_rolling_forecast_days, n_iterations=n_iterations, tuning_frequency=tuning_interval)  # test the removal of general model tuning
        # pipeLineTuningInterval = PipeLineTuningInterval(do_fine_tuning=False, medical_center_subset=medical_center_subset, time_period_subset=time_period_subset, max_number_of_cluster=max_number_of_cluster, input_length=input_length, number_of_rolling_forecast_days=number_of_rolling_forecast_days, hyper_parameter_tuning_number_of_test_days_in_DataFrame=hyper_parameter_tuning_number_of_test_days_in_DataFrame, n_iterations=n_iterations, tuning_frequency=tuning_interval)  # test the removal of fine tuning
        # AnalyzerPipeLineAutoForecastVA.save_analyzer_result_in_a_memory_efficient_way(pipeLineTuningInterval.pipeLineAutoForecastVA, file_name='data/toy_analyzer.dat')

        print(10)