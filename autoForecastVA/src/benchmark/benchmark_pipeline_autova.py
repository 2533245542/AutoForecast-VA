import copy

from autoForecastVA.src.utils.timer_util import TimerDefault
from pexecute.process import ProcessLoom

from autoForecastVA.src.analyzer.analyzer_pipeline_autoForecastVA import AnalyzerPipeLineAutoForecastVA
from autoForecastVA.src.data_caching.compute_characterization_code import ComputeCharacterizationCodeAndMd5Code
from autoForecastVA.src.loggers.causual_loggers import get_progress_logger
from autoForecastVA.src.pipelines.pipeline_ablation_studies import PipeLineTuningInterval
import os

class BenchMarkPipelineAutova():
    def __init__(self, list_of_input_dictionary=None, max_runner_cap=1, mapping_record_file_name='autova_mapping_md5_to_parameter_to_value.csv', log_output_file_name='autova_multiprocessing_progress_logger.log', logger_name='autova_progress_logger', folder_name_to_save_results='results_autova', run_multiprocessing=True, do_multiprocessing_dry_run=False, lazy=False):
        '''Create a list of kwargs, each for running a pipeline, run the pipelines and store the results '''
        # inputs
        self.list_of_input_dictionary = list_of_input_dictionary
        self.max_runner_cap = max_runner_cap
        self.md5_to_parameter_to_value_mapping_file_name = mapping_record_file_name  # the path to save the mapping between md5 code to kwargs
        self.log_output_file_name = log_output_file_name
        self.logger_name = logger_name
        self.folder_name_to_save_results = folder_name_to_save_results
        self.run_multiprocessing = run_multiprocessing
        self.do_multiprocessing_dry_run = do_multiprocessing_dry_run  # do nothing in the run function
        self.lazy = lazy

        if self.list_of_input_dictionary == None:
            self.list_of_input_dictionary = [{}]

        # outputs
        self.pool = None
        self.output = None
        if not self.lazy:
            if self.run_multiprocessing:
                self.start_multiprocessing()

    def do_a_run_of_pipeline(self, parameter_to_modified_value_dictionary, file_path_for_saving_result, do_multiprocessing_dry_run=False):
        progress_logger = get_progress_logger(logger_name=self.logger_name, log_output_file_path=os.path.join(self.folder_name_to_save_results, self.log_output_file_name))
        progress_logger.info('start doing one run of pipeline: ' + file_path_for_saving_result)
        progress_logger.info('the param dict for ' + file_path_for_saving_result + ' is: ' + str(parameter_to_modified_value_dictionary))
        print('start running pipeline: ')
        print(parameter_to_modified_value_dictionary)
        timerDefault = TimerDefault(time_unit='hour')
        timerDefault.start()
        if not do_multiprocessing_dry_run:
            pipeLineTuningInterval = PipeLineTuningInterval(**parameter_to_modified_value_dictionary)
            AnalyzerPipeLineAutoForecastVA.save_analyzer_result_in_a_memory_efficient_way(pipeLineTuningInterval.pipeLineAutoForecastVA, file_name=file_path_for_saving_result)
        timerDefault.end()
        progress_logger.info('end doing one run of pipeline: ' + file_path_for_saving_result + ' ' + 'duration(hours): ' + str(timerDefault.get_duration()))
        print('completed: ' + file_path_for_saving_result)
        print(parameter_to_modified_value_dictionary)

    def start_multiprocessing(self):
        # progress_logger = get_progress_logger(logger_name=self.logger_name, log_output_file_path=os.path.join(self.folder_name_to_save_results, self.log_output_file_name))
        # progress_logger.info('starting multiprocessing')  # cannot do logging untill creates the folder
        multiprocess_manager = ProcessLoom(max_runner_cap=self.max_runner_cap)
        for parameter_to_modified_value_dictionary in self.list_of_input_dictionary:
            md5_code = ComputeCharacterizationCodeAndMd5Code.get_class_MD5_code(target_class=PipeLineTuningInterval, parameter_to_modified_value_dictionary=parameter_to_modified_value_dictionary, store=True, md5_to_parameter_to_value_mapping_file_path=os.path.join(self.folder_name_to_save_results, self.md5_to_parameter_to_value_mapping_file_name))
            file_path_for_saving_result = self.folder_name_to_save_results + '/' + md5_code + '.dat'
            multiprocess_manager.add_function(self.do_a_run_of_pipeline, kwargs={'parameter_to_modified_value_dictionary': parameter_to_modified_value_dictionary, 'file_path_for_saving_result': file_path_for_saving_result, 'do_multiprocessing_dry_run': self.do_multiprocessing_dry_run}, key=md5_code)
        self.output = multiprocess_manager.execute()