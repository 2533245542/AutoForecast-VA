from hpbandster.core.worker import Worker
import time

from autoForecastVA.src.components.hyper_parameter_tuning.hyper_parameter_value_combination_evaluator import HyperParameterValueCombinationEvaluator


class MyWorker(Worker):

    def __init__(self, *args, customized_worker_specific_parameter_to_value_dictionary=None, **kwargs):
        super().__init__(*args, **kwargs)

        self.sleep_interval = customized_worker_specific_parameter_to_value_dictionary['sleep_interval']

    def compute(self, config, budget, **kwargs):
        """
        Simple example for a compute function
        The loss is just a the config + some noise (that decreases with the budget)

        For dramatization, the function can sleep for a given interval to emphasizes
        the speed ups achievable with parallel workers.

        Args:
            config: dictionary containing the sampled configurations by the optimizer
            budget: (float) amount of time/epochs/etc. the model can use to train

        Returns:
            dictionary with mandatory fields:
                'loss' (scalar)
                'info' (dict)
        """

        res = config['x'] + budget
        time.sleep(self.sleep_interval)

        return({
                    'loss': float(res),  # this is the a mandatory field to run hyperband
                    'info': res  # can be used for any user-defined information - also mandatory
                })


class LSTMWorker(Worker):

    def __init__(self, *args, customized_worker_specific_parameter_to_value_dictionary=None, **kwargs):
        super().__init__(*args, **kwargs)

        self.customized_worker_specific_parameter_to_value_dictionary = customized_worker_specific_parameter_to_value_dictionary

    def compute(self, config, budget, **kwargs):
        hyper_parameter_value_combination = {
            'learning_rate': config['learning_rate'],
            'number_of_hidden_dimensions': config['number_of_hidden_dimensions'],
            'dropout_rate': config['dropout_rate'],
            'number_of_training_epochs': int(budget)  # budget is float, need to convert to int to make LSTM work
        }

        hyperParameterValueCombinationEvaluator = HyperParameterValueCombinationEvaluator(hyper_parameter_value_combination=hyper_parameter_value_combination, **self.customized_worker_specific_parameter_to_value_dictionary)

        overall_loss = hyperParameterValueCombinationEvaluator.rolling_forecast_overall_normalization_reverted_test_prediction_loss

        return({
                'loss': float(overall_loss.item()),  # this is the a mandatory field to run hyperband
                'info': overall_loss.item()  # can be used for any user-defined information - also mandatory
        })