import pmdarima as pm
import pandas as pd
import statsmodels.api as sm
import numpy as np
from statsmodels.tsa.arima.model import ARIMA
class ModelManualArima():
    def __init__(self, dataframe, number_of_test_days=1, number_of_days_to_predict_ahead=1, order_of_autoregression=1, order_of_difference=1, order_of_moving_average=1, generate_train_predictions=True, lazy=False):
        """Fit an ARIMA model with the given [p,d,q], and make predictions."""
        # inputs
        self.dataframe = dataframe
        self.number_of_test_days = number_of_test_days
        self.number_of_days_to_predict_ahead = number_of_days_to_predict_ahead
        self.order_of_autoregression = order_of_autoregression
        self.order_of_difference = order_of_difference
        self.order_of_moving_average = order_of_moving_average
        self.generate_train_predictions = generate_train_predictions

        # outputs
        self.train_dataframe = None
        self.test_dataframe = None
        self.built_model = None
        self.train_predictions = []
        self.test_predictions = []

        if not lazy:
            self.train_dataframe = self.dataframe.iloc[:len(self.dataframe) - self.number_of_test_days, :]
            self.test_dataframe = self.dataframe.iloc[len(self.dataframe) - self.number_of_test_days:, :]
            self.built_model = self.build_arima_model(list_of_train_values=self.train_dataframe.case.tolist(), order_of_autoregression=self.order_of_autoregression, order_of_difference=self.order_of_difference, order_of_moving_average=self.order_of_moving_average)
            if self.generate_train_predictions:
                self.train_predictions = self.make_train_prediction(built_model=self.built_model)
            self.test_predictions = self.make_test_prediction(built_model=self.built_model, number_of_days_to_forecast=number_of_test_days, number_of_days_to_predict_ahead=self.number_of_days_to_predict_ahead)

    @staticmethod
    def build_arima_model(list_of_train_values, order_of_autoregression, order_of_difference, order_of_moving_average):
        naive_model = ARIMA(list_of_train_values, order=(order_of_autoregression, order_of_difference, order_of_moving_average))
        print(list_of_train_values, (order_of_autoregression, order_of_difference, order_of_moving_average))
        built_model = naive_model.fit()
        return built_model

    @staticmethod
    def make_train_prediction(built_model):
        """Make in-sample prediction for all samples."""
        train_predictions = built_model.predict().tolist()
        train_predictions_dataframe = pd.DataFrame({'case': train_predictions})
        return train_predictions_dataframe

    @staticmethod
    def make_test_prediction(built_model, number_of_days_to_forecast=1, number_of_days_to_predict_ahead=1):
        """Return a number of forecasts for a model on all variables starting from the forecast day."""
        number_of_skipped_days = number_of_days_to_predict_ahead - 1  # predict() defaults to skip 1 day already
        total_forecast_days = number_of_skipped_days + number_of_days_to_forecast
        test_predictions = built_model.forecast(steps=total_forecast_days).tolist()[number_of_skipped_days:]
        test_prediction_dataframe = pd.DataFrame({'case': test_predictions})
        return test_prediction_dataframe
