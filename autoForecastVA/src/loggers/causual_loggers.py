import logging
import os
def get_progress_logger(log_output_file_path='multiprocessing_progress_logger.log', mode='a', log_format='%(asctime)s - %(levelname)s - %(message)s',
                        logger_name='progress_logger'):
    progress_logger_handler = logging.FileHandler(log_output_file_path, mode=mode)
    progress_logger_formatter = logging.Formatter(log_format)
    progress_logger_handler.setFormatter(progress_logger_formatter)

    progress_logger = logging.getLogger(logger_name)
    progress_logger.setLevel(logging.DEBUG)
    progress_logger.addHandler(progress_logger_handler)
    return progress_logger