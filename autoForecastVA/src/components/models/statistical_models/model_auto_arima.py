import pmdarima as pm
import pandas as pd
import statsmodels.api as sm
import numpy as np

class ModelAutoArima():
    def __init__(self, dataframe, number_of_test_days=1, number_of_days_to_predict_ahead=1, autoarima_argument_dictioanry=None, generate_train_predictions=True, lazy=False):
        """Find an optimal ARIMA model by using pmdarima, and use the best model to make predictions."""
        # inputs
        self.dataframe = dataframe
        self.number_of_test_days = number_of_test_days
        self.number_of_days_to_predict_ahead = number_of_days_to_predict_ahead
        self.autoarima_argument_dictioanry = autoarima_argument_dictioanry
        self.generate_train_predictions = generate_train_predictions

        # outputs
        self.train_dataframe = None
        self.test_dataframe = None
        self.selected_built_model = None
        self.selected_order_of_autoregression = None
        self.selected_order_of_difference = None
        self.selected_order_of_moving_average = None
        self.train_predictions = []
        self.test_predictions = []

        if not lazy:
            self.train_dataframe = self.dataframe.iloc[:len(self.dataframe) - self.number_of_test_days, :]
            self.test_dataframe = self.dataframe.iloc[len(self.dataframe) - self.number_of_test_days:, :]
            self.selected_built_model = self.auto_select_arima_model(list_of_train_values=self.train_dataframe.case.tolist(), autoarima_argument_dictioanry=self.autoarima_argument_dictioanry)
            self.selected_order_of_autoregression = self.selected_built_model.order[0]
            self.selected_order_of_difference = self.selected_built_model.order[1]
            self.selected_order_of_moving_average = self.selected_built_model.order[2]
            if self.generate_train_predictions:
                self.train_predictions = self.make_train_prediction(built_model=self.selected_built_model)
            self.test_predictions = self.make_test_prediction(built_model=self.selected_built_model, number_of_days_to_forecast=number_of_test_days, number_of_days_to_predict_ahead=self.number_of_days_to_predict_ahead)

    @staticmethod
    def auto_select_arima_model(list_of_train_values, autoarima_argument_dictioanry):
        selected_model = pm.auto_arima(list_of_train_values, **autoarima_argument_dictioanry)
        return selected_model


    @staticmethod
    def make_train_prediction(built_model):
        """Make in-sample prediction for all samples."""
        train_predictions = built_model.predict_in_sample().tolist()
        train_predictions_dataframe = pd.DataFrame({'case': train_predictions})
        return train_predictions_dataframe

    @staticmethod
    def make_test_prediction(built_model, number_of_days_to_forecast=1, number_of_days_to_predict_ahead=1):
        """Return a number of forecasts for a model on all variables starting from the forecast day."""
        number_of_skipped_days = number_of_days_to_predict_ahead - 1  # predict() defaults to skip 1 day already
        total_forecast_days = number_of_skipped_days + number_of_days_to_forecast
        test_predictions = built_model.predict(n_periods=total_forecast_days).tolist()[number_of_skipped_days:]
        test_prediction_dataframe = pd.DataFrame({'case': test_predictions})
        return test_prediction_dataframe


