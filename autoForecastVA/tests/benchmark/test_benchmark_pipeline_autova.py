import unittest

from autoForecastVA.src.benchmark.benchmark_parameter_util import AutoVAParameterUtil
from autoForecastVA.src.benchmark.benchmark_pipeline_autova import BenchMarkPipelineAutova
from autoForecastVA.src.benchmark.benchmark_varma_pipeline import BenchmarkVarmaPipeline

class TestBenchMarkPipelineAutovaLargeSpaceAndLimitedFineTuning(unittest.TestCase):
    def test_running_limited_fine_tuning(self):
        '''Restrict fine-tuning to the linear layer.'''

        list_of_argument_dictionary = [
            {
                # 'number_of_hidden_dimensions_lower': 10,
                # 'number_of_hidden_dimensions_upper': 150,
             'trainable_parameter_name_list': ['linear.weight', 'linear.bias']}
        ]

        # tuning frequency = 30
        list_of_output_dictionary = AutoVAParameterUtil(list_of_template_dictionary=list_of_argument_dictionary, list_of_tuning_interval=[30], list_of_seeds=None, generate_no_fine_tuning=False, generate_no_clustering=False, generate_no_clustering_and_no_feature_selection=False).port_inserted_seeded_total_list_of_dictionary

        # set up save
        folder_name_to_save_results = 'results_original_search_space_and_limited_fine_tuning'
        mapping_record_file_path = 'autova_mapping_md5_to_parameter_to_value.csv'
        log_output_file_path = 'autova_multiprocessing_progress_logger.log'
        logger_name = 'autova_larger_search_space_progress_logger'
        benchMarkPipelineAutova = BenchMarkPipelineAutova(max_runner_cap=2, list_of_input_dictionary=list_of_output_dictionary, do_multiprocessing_dry_run=True, mapping_record_file_name=mapping_record_file_path, log_output_file_name=log_output_file_path, logger_name=logger_name, folder_name_to_save_results=folder_name_to_save_results)

        print(benchMarkPipelineAutova.output)