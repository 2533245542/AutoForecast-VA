import pexecute
import copy
from pexecute.process import ProcessLoom

from autoForecastVA.src.analyzer.analyzer_pipeline_autoForecastVA import AnalyzerPipeLineAutoForecastVA
from autoForecastVA.src.data_caching.compute_characterization_code import ComputeCharacterizationCodeAndMd5Code
from autoForecastVA.src.data_caching.variable_load_and_save import VariableSaverAndLoader
from autoForecastVA.src.loggers.causual_loggers import get_progress_logger
from autoForecastVA.src.pipelines.pipeline_ablation_studies import PipeLineTuningInterval
import logging

from autoForecastVA.src.pipelines.pipeline_transcend import PipeLineTuningIntervalTranscend


class BenchMarkPipelineIntervalTranscend():
    def __init__(self, base_nameserver_port=60000, max_tuning_interval=9999999, number_of_repeat_for_each_pipeline=5, max_runner_cap=1, mapping_record_file_path='data/mapping_md5_to_parameter_to_value.csv', run_multiprocessing=True, do_multiprocessing_dry_run=False, lazy=False):
        '''Create a list of kwargs, each for running a pipeline, run the pipelines and store the results '''
        # inputs
        self.base_nameserver_port = base_nameserver_port
        self.max_tuning_interval = max_tuning_interval
        self.number_of_repeat_for_each_pipeline = number_of_repeat_for_each_pipeline
        self.max_runner_cap = max_runner_cap
        self.md5_to_parameter_to_value_mapping_file_path = mapping_record_file_path  # the path to save the mapping between md5 code to kwargs
        self.run_multiprocessing = run_multiprocessing
        self.do_multiprocessing_dry_run = do_multiprocessing_dry_run  # do nothing in the run function
        self.lazy = lazy

        # outputs
        self.list_of_parameter_to_modified_value_dictionary = None
        self.pool = None
        self.output = None
        if not self.lazy:
            self.list_of_parameter_to_modified_value_dictionary = self.generate_parameter_to_modified_value_dictionary_for_each_pipeline(base_nameserver_port=self.base_nameserver_port, max_tuning_interval=self.max_tuning_interval, number_of_repeat_for_each_pipeline=self.number_of_repeat_for_each_pipeline)

            if self.run_multiprocessing:
                self.start_multiprocessing()

    def do_a_run_of_pipeline(self, parameter_to_modified_value_dictionary, file_path_for_saving_result, do_multiprocessing_dry_run=False):
        progress_logger = get_progress_logger()
        progress_logger.info('start doing one run of pipeline: ' + file_path_for_saving_result)
        progress_logger.info('the param dict for ' + file_path_for_saving_result + ' is: ' + str(parameter_to_modified_value_dictionary))
        print('start running pipeline: ')
        print(parameter_to_modified_value_dictionary)
        if not do_multiprocessing_dry_run:
            pipeLineTuningIntervalTranscend = PipeLineTuningIntervalTranscend(**parameter_to_modified_value_dictionary)
            AnalyzerPipeLineAutoForecastVA.save_analyzer_result_in_a_memory_efficient_way(pipeLineTuningIntervalTranscend.pipeLineAutoForecastVA, file_name=file_path_for_saving_result)
        progress_logger.info('end doing one run of pipeline: ' + file_path_for_saving_result)
        print('completed: ' + file_path_for_saving_result)
        print(parameter_to_modified_value_dictionary)

    def start_multiprocessing(self):
        progress_logger = get_progress_logger()
        progress_logger.info('starting multiprocessing')
        multiprocess_manager = ProcessLoom(max_runner_cap=self.max_runner_cap)
        for parameter_to_modified_value_dictionary in self.list_of_parameter_to_modified_value_dictionary:
            md5_code = ComputeCharacterizationCodeAndMd5Code.get_class_MD5_code(target_class=PipeLineTuningIntervalTranscend, parameter_to_modified_value_dictionary=parameter_to_modified_value_dictionary, store=True, md5_to_parameter_to_value_mapping_file_path=self.md5_to_parameter_to_value_mapping_file_path)
            file_path_for_saving_result = 'results/'+md5_code
            multiprocess_manager.add_function(self.do_a_run_of_pipeline, kwargs={'parameter_to_modified_value_dictionary': parameter_to_modified_value_dictionary, 'file_path_for_saving_result': file_path_for_saving_result, 'do_multiprocessing_dry_run': self.do_multiprocessing_dry_run}, key=md5_code)
        self.output = multiprocess_manager.execute()

    @staticmethod
    def generate_parameter_to_modified_value_dictionary_for_each_pipeline(base_nameserver_port=65100, max_tuning_interval=999999999, number_of_repeat_for_each_pipeline=5):
        '''Generate a list of kwargs, one for each pipeline'''

        # tuning interval
        list_of_tuning_interval_dict = []
        for interval in [5]:
            parameter_to_modified_value_dictionary = {}
            parameter_to_modified_value_dictionary['tuning_frequency'] = interval

            list_of_tuning_interval_dict.append(parameter_to_modified_value_dictionary)


        total_list_of_dict = list_of_tuning_interval_dict

        # repeat each pipeline by assigning different seeds
        repeated_total_list_of_dict = []
        for parameter_to_modified_value_dictionary in total_list_of_dict:
            repeated_total_list_of_dict += BenchMarkPipelineIntervalTranscend.repeat_a_pipeline_by_seed(
                parameter_to_modified_value_dictionary=parameter_to_modified_value_dictionary, number_of_repeat_for_one_pipeline=number_of_repeat_for_each_pipeline)

        # add unique nameserver port for each dictionary
        for parameter_to_modified_value_dictionary in repeated_total_list_of_dict:
            base_nameserver_port += 1  # each kwarg dict will have a nameserver port so pipelines don't interfere with each other
            parameter_to_modified_value_dictionary['nameserver_port'] = base_nameserver_port
            parameter_to_modified_value_dictionary['nameserver_run_id'] = 'run_id' + str(base_nameserver_port)

        return repeated_total_list_of_dict

    @staticmethod
    def repeat_a_pipeline_by_seed(parameter_to_modified_value_dictionary, number_of_repeat_for_one_pipeline=5):
        list_of_repeated_dictionary = []
        for seed in range(number_of_repeat_for_one_pipeline):
            replicated_argument_dictionary = copy.deepcopy(parameter_to_modified_value_dictionary)
            replicated_argument_dictionary['hyper_parameter_space_seed'] = seed
            replicated_argument_dictionary['numpy_seed'] = seed
            replicated_argument_dictionary['neural_network_training_seed'] = seed
            list_of_repeated_dictionary.append(replicated_argument_dictionary)
        return list_of_repeated_dictionary
