from autoForecastVA.src.components.hyper_parameter_tuning.hyper_parameter_tuning_worker import LSTMWorker
from autoForecastVA.src.pipelines.pipeline_autoForecastVA import PipeLineAutoForecastVA
import ConfigSpace as CS
import ConfigSpace.hyperparameters as CSH
from torch import nn


class PipeLineTuningIntervalTranscend():

    # def __init__(self, medical_center_subset=None, time_period_subset=None, dataset_path='../../data/coviddata07292020.csv', number_of_days_for_data_averaging=3, max_number_of_cluster=10, number_of_days_to_predict_ahead=1, number_of_test_days_in_a_day_level_DataFrame=1, p_value_threshold=0.05, data_windowing_option=2, input_length=10, lstm_is_many_to_many=False, number_of_rolling_forecast_days=30, hyper_parameter_tuning_number_of_test_days_in_DataFrame=10, hyper_parameter_tuning_number_of_rolling_forecast_days=1, learning_rate_lower=0.00001, learning_rate_upper=0.1, number_of_hidden_dimensions_lower=5, number_of_hidden_dimensions_upper=50, dropout_rate_lower=0, dropout_rate_upper=0.7, hyper_parameter_space_seed=0, numpy_seed=0, CustomizedWorker=LSTMWorker, nameserver_run_id='example1', nameserver_address='127.0.0.1', nameserver_port=65300, min_budget=15, max_budget=160, n_iterations=4, tuning_frequency=1, neural_network_training_seed=0, loss_function=nn.MSELoss(), trainable_parameter_name_list=None, do_clustering=True, do_normalization=True, do_feature_selection=True, batch_size_for_tuning=60, batch_size_for_general_model_building=60, batch_size_for_fine_tuning=60, generate_train_prediction_during_tuning=False, enable_data_caching_during_tuning=True, verbose=True, lazy=False):  # default
    # def __init__(self, medical_center_subset=None, time_period_subset=None, dataset_path='../../data/coviddata07292020.csv', number_of_days_for_data_averaging=3, max_number_of_cluster=10, number_of_days_to_predict_ahead=1, number_of_test_days_in_a_day_level_DataFrame=1, p_value_threshold=0.05, data_windowing_option=2, input_length=10, lstm_is_many_to_many=False, number_of_rolling_forecast_days=30, hyper_parameter_tuning_number_of_test_days_in_DataFrame=10, hyper_parameter_tuning_number_of_rolling_forecast_days=1, learning_rate_lower=0.00001, learning_rate_upper=0.1, number_of_hidden_dimensions_lower=10, number_of_hidden_dimensions_upper=150, dropout_rate_lower=0, dropout_rate_upper=0.7, hyper_parameter_space_seed=0, numpy_seed=0, CustomizedWorker=LSTMWorker, nameserver_run_id='example1', nameserver_address='127.0.0.1', nameserver_port=65300, min_budget=15, max_budget=160, n_iterations=4, tuning_frequency=1, neural_network_training_seed=0, loss_function=nn.MSELoss(), trainable_parameter_name_list=None, do_clustering=True, do_normalization=True, do_feature_selection=True, batch_size_for_tuning=60, batch_size_for_general_model_building=60, batch_size_for_fine_tuning=60, generate_train_prediction_during_tuning=False, enable_data_caching_during_tuning=True, verbose=True, lazy=False):  # experiment 1
    def __init__(self, medical_center_subset=None, time_period_subset=None, dataset_path='../../data/coviddata07292020.csv', number_of_days_for_data_averaging=1, max_number_of_cluster=10, number_of_days_to_predict_ahead=1, number_of_test_days_in_a_day_level_DataFrame=1, p_value_threshold=0.05, data_windowing_option=2, input_length=10, lstm_is_many_to_many=False, number_of_rolling_forecast_days=30, hyper_parameter_tuning_number_of_test_days_in_DataFrame=10, hyper_parameter_tuning_number_of_rolling_forecast_days=1, learning_rate_lower=0.00001, learning_rate_upper=0.1, number_of_hidden_dimensions_lower=5, number_of_hidden_dimensions_upper=50, dropout_rate_lower=0, dropout_rate_upper=0.7, hyper_parameter_space_seed=0, numpy_seed=0, CustomizedWorker=LSTMWorker, nameserver_run_id='example1', nameserver_address='127.0.0.1', nameserver_port=65300, min_budget=15, max_budget=160, n_iterations=4, tuning_frequency=1, neural_network_training_seed=0, loss_function=nn.MSELoss(), trainable_parameter_name_list=None, do_clustering=True, do_normalization=True, do_feature_selection=True, batch_size_for_tuning=60, batch_size_for_general_model_building=60, batch_size_for_fine_tuning=60, generate_train_prediction_during_tuning=False, enable_data_caching_during_tuning=True, verbose=True, lazy=False):  # experiment 1


        if medical_center_subset is None:
            self.medical_center_subset = ['402', '405', '436', '437', '438', '442', '459', '460', '463', '501', '503', '506', '508', '509', '512', '515', '516', '518', '519', '521', '523', '528', '528A6', '528A7', '528A8', '529', '531', '534', '537', '539', '541', '542', '544', '548', '549', '550', '552', '554', '556', '557', '558', '562', '568', '570', '573', '575', '578', '580', '581', '583', '585', '589', '589A5', '589A7', '590', '593', '595', '596', '600', '603', '605', '607', '608', '610', '612A4', '614', '618', '619', '621', '623', '626', '631', '635', '636', '636A6', '636A8', '637', '640', '642', '644', '646', '648', '649', '650', '652', '654', '655', '656', '657', '657A5', '658', '659', '660', '662', '663', '664', '668', '671', '673', '674', '675', '676', '678', '688', '689', '691', '693', '695', '756', '757']
        else:
            self.medical_center_subset = medical_center_subset
        self.time_period_subset = time_period_subset
        self.dataset_path = dataset_path

        self.number_of_days_for_data_averaging = number_of_days_for_data_averaging
        self.max_number_of_cluster = max_number_of_cluster
        self.number_of_days_to_predict_ahead = number_of_days_to_predict_ahead
        self.number_of_test_days_in_a_day_level_DataFrame = number_of_test_days_in_a_day_level_DataFrame

        self.p_value_threshold = p_value_threshold

        self.data_windowing_option = data_windowing_option
        self.input_length = input_length
        self.lstm_is_many_to_many = lstm_is_many_to_many
        self.number_of_rolling_forecast_days = number_of_rolling_forecast_days

        self.hyper_parameter_tuning_number_of_test_days_in_DataFrame = hyper_parameter_tuning_number_of_test_days_in_DataFrame  # default to be 1 to save time
        self.hyper_parameter_tuning_number_of_rolling_forecast_days = hyper_parameter_tuning_number_of_rolling_forecast_days  # default to 10

        self.learning_rate_lower = learning_rate_lower
        self.learning_rate_upper = learning_rate_upper
        self.number_of_hidden_dimensions_lower = number_of_hidden_dimensions_lower
        self.number_of_hidden_dimensions_upper = number_of_hidden_dimensions_upper
        self.dropout_rate_lower = dropout_rate_lower
        self.dropout_rate_upper = dropout_rate_upper

        self.hyper_parameter_space = CS.ConfigurationSpace()
        learning_rate = CSH.UniformFloatHyperparameter('learning_rate', lower=self.learning_rate_lower, upper=self.learning_rate_upper, log=True)
        number_of_hidden_dimensions = CSH.UniformIntegerHyperparameter('number_of_hidden_dimensions', lower=self.number_of_hidden_dimensions_lower, upper=self.number_of_hidden_dimensions_upper)
        dropout_rate = CSH.UniformFloatHyperparameter('dropout_rate', lower=self.dropout_rate_lower, upper=self.dropout_rate_upper)
        hyper_parameters = [learning_rate, number_of_hidden_dimensions, dropout_rate]
        self.hyper_parameter_space.add_hyperparameters(hyper_parameters)

        self.hyper_parameter_space_seed = hyper_parameter_space_seed  # controls randomness for hyperparemter space (random) sampling. BOHB does random sampling at the begininig of the tuning
        self.numpy_seed = numpy_seed  # controls randomness for BOHB optimizer

        self.CustomizedWorker = CustomizedWorker  # the worker for runnning BOHB
        self.nameserver_run_id = nameserver_run_id  # can be repeatable
        self.nameserver_address = nameserver_address  # can be repeatable
        self.nameserver_port = nameserver_port  # each instance (that runs on the same local computer) should have a unique nameserver_port; avoid using 80, 8080, 443 or below 1024;

        self.min_budget = min_budget
        self.max_budget = max_budget
        self.n_iterations = n_iterations

        self.tuning_frequency = tuning_frequency

        self.neural_network_training_seed = neural_network_training_seed  # control randomness for nueral network training

        self.loss_function = loss_function

        if trainable_parameter_name_list is None:
            self.trainable_parameter_name_list = ['lstm.weight_ih_l0', 'lstm.weight_hh_l0', 'lstm.bias_ih_l0', 'lstm.bias_hh_l0', 'linear.weight', 'linear.bias']
            # trainable_parameter_name_list = ['linear.weight', 'linear.bias']
        else:
            self.trainable_parameter_name_list = trainable_parameter_name_list

        self.do_clustering = do_clustering  # default to True; if False, each clinic is a cluster
        self.do_normalization = do_normalization  # default to True
        self.do_feature_selection = do_feature_selection  # default to True; if False, all features are selected

        self.batch_size_for_tuning = batch_size_for_tuning
        self.batch_size_for_general_model_building = batch_size_for_general_model_building
        self.batch_size_for_fine_tuning = batch_size_for_fine_tuning

        self.generate_train_prediction_during_tuning = generate_train_prediction_during_tuning  # default to False to save time
        self.enable_data_caching_during_tuning = enable_data_caching_during_tuning  # default to True to save time;
        self.verbose = verbose

        # self.file_name_save_analyzer_result_in_a_memory_efficient_way = analyzer_result_file_name

        self.lazy = lazy

        if not lazy:
            self.run_pipeline()

    def run_pipeline(self):
        '''
        This is a full pipeline with tuning interval. By setting tuning interval from 1 to infinite, one could test how autoforecastVA's performance changes with tuning interval.
        `pipe_line_tuning_interval`

        Full-Full form:
        cluster, feature selection and hyperparameter tuning for every day

        Full-Intermediate form:
        cluster, feature selection and hyperparameter tuning with an interval

        Full-Base form:
        cluster, feature selection and hyperparameter tuning for only the first day

        '''

        self.pipeLineAutoForecastVA = PipeLineAutoForecastVA(medical_center_subset=self.medical_center_subset, time_period_subset=self.time_period_subset, dataset_path=self.dataset_path, number_of_days_for_data_averaging=self.number_of_days_for_data_averaging, max_number_of_cluster=self.max_number_of_cluster, number_of_days_to_predict_ahead=self.number_of_days_to_predict_ahead, number_of_test_days_in_a_day_level_DataFrame=self.number_of_test_days_in_a_day_level_DataFrame, p_value_threshold=self.p_value_threshold, data_windowing_option=self.data_windowing_option, input_length=self.input_length, lstm_is_many_to_many=self.lstm_is_many_to_many, number_of_rolling_forecast_days=self.number_of_rolling_forecast_days, hyper_parameter_tuning_number_of_test_days_in_DataFrame=self.hyper_parameter_tuning_number_of_test_days_in_DataFrame, hyper_parameter_tuning_number_of_rolling_forecast_days=self.hyper_parameter_tuning_number_of_rolling_forecast_days, hyper_parameter_space=self.hyper_parameter_space, hyper_parameter_space_seed=self.hyper_parameter_space_seed, numpy_seed=self.numpy_seed, CustomizedWorker=self.CustomizedWorker, nameserver_run_id=self.nameserver_run_id, nameserver_address=self.nameserver_address, nameserver_port=self.nameserver_port, min_budget=self.min_budget, max_budget=self.max_budget, n_iterations=self.n_iterations, neural_network_training_seed=self.neural_network_training_seed, loss_function=self.loss_function, trainable_parameter_name_list=self.trainable_parameter_name_list, do_clustering=self.do_clustering, do_normalization=self.do_normalization, do_feature_selection=self.do_feature_selection, tuning_interval=self.tuning_frequency, batch_size_for_tuning=self.batch_size_for_tuning, batch_size_for_general_model_building=self.batch_size_for_general_model_building, batch_size_for_fine_tuning=self.batch_size_for_fine_tuning, generate_train_prediction_during_tuning=self.generate_train_prediction_during_tuning, enable_data_caching_during_tuning=self.enable_data_caching_during_tuning, verbose=self.verbose)
