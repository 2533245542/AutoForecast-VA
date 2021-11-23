import copy

from autoForecastVA.src.components.hyper_parameter_tuning.hyper_parameter_value_combination_evaluator import HyperParameterValueCombinationEvaluator


class ModelFineTuning():
    '''
    Convert the general model to the customized model (with costumozed parameter settings), train and predict, and get the fine-tuned model.

    Given a trained model, a hyper-parameter value combination, a data set, and an index list specifying which layer to fix, we fine tune the model and return the fine tuned model.
    '''
    def __init__(self, hyper_parameter_value_combination, trainable_parameter_name_list, parameter_to_weights_dictionary=None, general_model=None, dataframe=None, number_of_days_to_predict_ahead=1, loss_function=None, data_windowing_option=2, input_length=3, lstm_is_many_to_many=False, do_normalization=True, batch_size=1, number_of_test_days_in_a_day_level_DataFrame=1, number_of_days_for_testing_in_a_hyper_parameter_tuning_run=1, lazy=False, verbose=False):
        #inputs
        self.hyper_parameter_value_combination = hyper_parameter_value_combination
        self.trainable_parameter_name_list = trainable_parameter_name_list  # describe the customized model; how each parameter should be trainable or not
        self.parameter_to_weights_dictionary = parameter_to_weights_dictionary  # describe the customized model; how each parameter should be initialized; if this is not provided, use the exiting weights in general model
        self.general_model = general_model
        self.dataframe = dataframe  # should be unnormalized; should only contain one clinic; the last day in teh dataframe is the global test day
        self.number_of_days_to_predict_ahead = number_of_days_to_predict_ahead
        self.loss_function = loss_function
        self.data_windowing_option = data_windowing_option
        self.input_length = input_length
        self.lstm_is_many_to_many = lstm_is_many_to_many
        self.do_normalization = do_normalization
        self.batch_size = batch_size
        self.number_of_test_days_in_a_day_level_DataFrame = number_of_test_days_in_a_day_level_DataFrame  # the last day in teh dataframe is the global test day
        self.number_of_rolling_forecast_days = number_of_days_for_testing_in_a_hyper_parameter_tuning_run  # only do one rollling forecast

        self.lazy = lazy
        self.verbose = verbose

        # outputs
        self.evaluator = None
        self.operator = None
        self.fine_tuned_model = None

        self.partially_trainable_general_model = None
        self.customized_model_layer_name_to_layer_dictionary = None   # the model with our costumozed parameter settings; represented in a dictionary of layers

        self.all_days_train_prediction_list = None
        self.all_days_train_prediction_target_list = None
        self.all_days_test_prediction_target_list = None
        self.all_days_test_prediction_list = None
        self.train_epoch_loss_list = None

        if self.parameter_to_weights_dictionary is not None:
            raise NotImplemented('parameter_to_weights_dictionary not implemented')

        if not lazy:
            self.customize_model()
            self.build_model_from_the_customized_layers_and_train_and_predict()

    def customize_model(self):
        # set trainable parameters
        self.partially_trainable_general_model = self.make_partially_trainable_model(model=self.general_model, trainable_parameter_name_list=self.trainable_parameter_name_list, verbose=self.verbose)

        # set parameters' initial weights (here we use the old weights -- weights in the general model)
        self.customized_model_layer_name_to_layer_dictionary = dict(self.partially_trainable_general_model.named_children())   # assume model only has the simple layer-after -layer structure

    def build_model_from_the_customized_layers_and_train_and_predict(self):
        hyperParameterValueCombinationEvaluator = HyperParameterValueCombinationEvaluator(DataFrame=self.dataframe, loss_function=self.loss_function, hyper_parameter_value_combination=self.hyper_parameter_value_combination, number_of_days_to_predict_ahead=self.number_of_days_to_predict_ahead, number_of_test_days_in_a_day_level_DataFrame=self.number_of_test_days_in_a_day_level_DataFrame, number_of_rolling_forecast_days=self.number_of_rolling_forecast_days, data_windowing_option=self.data_windowing_option, input_length=self.input_length, lstm_is_many_to_many=self.lstm_is_many_to_many, do_normalization=self.do_normalization, batch_size=self.batch_size, customized_model_layer_name_to_layer_dictionary=self.customized_model_layer_name_to_layer_dictionary)

        self.evaluator = hyperParameterValueCombinationEvaluator
        self.operator = hyperParameterValueCombinationEvaluator.list_of_rolling_forecast_daily_lstmModelOperators[0]
        self.fine_tuned_model = self.operator.model

        self.all_days_train_prediction_target_list = hyperParameterValueCombinationEvaluator.rolling_forecast_all_days_normalization_reverted_train_prediction_targets
        self.all_days_train_prediction_list = hyperParameterValueCombinationEvaluator.rolling_forecast_all_days_normalization_reverted_train_predictions
        self.all_days_test_prediction_target_list = hyperParameterValueCombinationEvaluator.rolling_forecast_all_days_normalization_reverted_test_prediction_targets
        self.all_days_test_prediction_list = hyperParameterValueCombinationEvaluator.rolling_forecast_all_days_normalization_reverted_test_predictions
        self.train_epoch_loss_list = hyperParameterValueCombinationEvaluator.list_of_rolling_forecast_daily_list_of_epoch_loss[0]

    @staticmethod
    def make_partially_trainable_model(model, trainable_parameter_name_list, verbose=False):
        '''All parameters not in trainable_parameter_name_list are not trainable'''
        complete_parameter_name_list = [name for name, param in model.named_parameters()]
        if verbose:
            print('All parameters: {}'.format(complete_parameter_name_list))

        trainable_adjusted_generabl_model = copy.deepcopy(model)

        for name, param in list(trainable_adjusted_generabl_model.named_parameters()):
            if name in trainable_parameter_name_list:
                if verbose:
                    print('Set trainable parameter: {}'.format(name))
                param.requires_grad = True
            else:
                if verbose:
                    print('Set not trainable parameter: {}'.format(name))
                param.requires_grad = False
        return trainable_adjusted_generabl_model





