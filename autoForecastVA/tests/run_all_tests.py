# run this file with the python version that is supposed to run individual AutoForecast-VA tests
import subprocess
import time
from datetime import datetime


# function: run a test
# input: a directory to cd into, and the file to test (should only input directory and file)
# output: the test result; whether the test is succeeded or not
def run_a_test(path, test_file_name):
    start_time = time.time()

    print("Testing:", path, test_file_name)
    run_test_result = subprocess.run(args=['python', '-m', 'unittest', test_file_name], cwd=path, capture_output=True, text=True)
    # run_test_result = subprocess.run(args=['python', '-m', 'pytest', test_file_name], cwd=path, capture_output=True, text=True)

    end_time = time.time()

    run_time_of_the_test_in_seconds = (end_time - start_time)

    succeeded = False
    if run_test_result.returncode == 0:
        succeeded = True
    result_std_out = run_test_result.stdout
    result_std_err = run_test_result.stderr

    now = datetime.now()  # show time
    current_time = now.strftime("%H:%M:%S")

    if succeeded:
        print('SUCCEEDED TEST: ' + path + ' ' + test_file_name + ' time spent(minutes):' + str(
            round(run_time_of_the_test_in_seconds / 60, 2)) + ' current time:' + current_time +'\n')
    else:
        print('FAILED TEST:' + path + ' ' + test_file_name + ' time spent(minutes):' + str(
            round(run_time_of_the_test_in_seconds / 60, 2)) + ' current time:' + current_time + '\n')
        print(result_std_err)

    return succeeded, result_std_out, result_std_err, run_time_of_the_test_in_seconds

# function: run loop
# input: a dictionry
# output: print how many tests fails, who they are, and write the error info of each test into a file; will create RUNALLTESTSLOG
def run_test_loop(path_to_test_file_name_list_dictionary):
    path_list = []
    test_file_name_list = []
    succeeded_list = []
    result_std_out_list = []
    result_std_err_list = []
    test_run_time_in_seconds_list = []
    for path in path_to_test_file_name_list_dictionary.keys():
        for test_file_name in path_to_test_file_name_list_dictionary[path]:
            # print(path, test_file_name)
            succeeded, result_std_out, result_std_err, run_time_of_the_test_in_seconds = run_a_test(path=path, test_file_name=test_file_name)
            succeeded_list.append(succeeded)
            result_std_out_list.append(result_std_out)
            result_std_err_list.append(result_std_err)
            test_run_time_in_seconds_list.append(run_time_of_the_test_in_seconds)
            path_list.append(path)
            test_file_name_list.append(test_file_name)

    print('Number of succeeded tests:', sum(succeeded_list))
    print('Number of failed tests:', len(succeeded_list) - sum(succeeded_list))
    print('See RUN_ALL_TESTS_LOG for detail')

    with open('RUN_ALL_TESTS_LOG', 'w') as f:
        for path, test_file_name, succeeded, result_std_out, result_std_err, run_time_of_the_test_in_seconds in zip(path_list, test_file_name_list, succeeded_list, result_std_out_list, result_std_err_list, test_run_time_in_seconds_list):
            if succeeded:
                f.write('SUCCEEDED TEST: ' + path + ' ' + test_file_name + ' time spent(minutes):' + str(round(run_time_of_the_test_in_seconds/60, 2)) + '\n')
            else:
                f.write('FAILED TEST:' + path + ' ' + test_file_name + ' time spent(minutes):' + str(round(run_time_of_the_test_in_seconds/60, 2)) + '\n')
                f.write(result_std_err)

    print('Written logs to RUN_ALL_TESTS_LOG')

if __name__ == '__main__':
    # path_to_test_file_name_list_dictionary = {
    #     'components/data_preprocessing': ['test_data_delimiting_by_day_and_medical_center.py', 'test_error_run.py'],
    #     'components/feature_selecion_global': ['test_feature_selection_global_toy_example.py']
    # }
    #
    # run_test_loop(path_to_test_file_name_list_dictionary)

    path_to_test_file_name_list_dictionary = {
        'components/data_filters': ['test_filter_controller.py', 'test_filter_customized.py'],
        'components/data_preprocessing': ['test_data_clustering_and_normalization.py', 'test_data_delimiting_by_day_and_medical_center.py', 'test_data_imputation_and_averaging.py', 'test_data_preparation_for_feature_selection.py', 'test_data_windowing.py'],
        'components/feature_selecion_global': ['test_feature_selection_global.py', 'test_feature_selection_global_toy_example.py'],
        'components/general_model_training': ['test_general_model_training.py'],
        'components/hyper_parameter_tuning': ['test_hyper_parameter_tuning_loop.py', 'test_hyper_parameter_tuning_rolling_forecast_data_preparation.py', 'test_hyper_parameter_value_combination_evaluator.py'],
        'components/model_fine_tuning': ['test_model_fine_tuning.py'],
        'components/models/general_models': ['test_lstm_many_to_many.py', 'test_lstm_many_to_one.py', 'test_lstm_model_operator.py'],
        'components/models/fine_tuned_models': [],
        'components/models/statistical_models': ['test_varma_model.py'],
        'analyzer': ['test_analyzer_pipeline_varma.py'],
        'components/other_tests': [],
        'data_caching/': ['test_compute_characterization_code.py'],
        'pipelines/': ['test_pipeline.py']
    }
    run_test_loop(path_to_test_file_name_list_dictionary)
    # keywords to search for succeeded and failed tests in log: 'SUCCEEDED TEST:', 'FAILED TEST:'