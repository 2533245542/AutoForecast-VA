import statsmodels.api as sm
import numpy as np

class VarmaModel():
    def __init__(self, dataframe, number_of_test_days=1, number_of_days_to_predict_ahead=1, max_model_building_convergence_iteration=1000, order_of_autoregression_lower=0, order_of_autoregression_upper=5, order_of_moving_average_lower=0, order_of_moving_average_upper=5, generate_train_predictions=True, reduce_memory=True, verbose=True, lazy=False):
        """Find an optimal VARMA model by grid searching the order of AR and MA, and use the best model to make predictions."""
        # inputs
        self.dataframe = dataframe
        self.number_of_test_days = number_of_test_days
        self.number_of_days_to_predict_ahead = number_of_days_to_predict_ahead
        self.max_model_building_convergence_iteration = max_model_building_convergence_iteration
        self.order_of_autoregression_lower = order_of_autoregression_lower  # inclusive
        self.order_of_autoregression_upper = order_of_autoregression_upper  # inclusive
        self.order_of_moving_average_lower = order_of_moving_average_lower  # inclusive
        self.order_of_moving_average_upper = order_of_moving_average_upper  # inclusive
        self.generate_train_predictions = generate_train_predictions
        self.reduce_memory = reduce_memory

        # outputs
        self.train_dataframe = None
        self.test_dataframe = None
        self.processed_dataframe = None
        # self.processed_train_dataframe = None
        # self.processed_test_dataframe = None
        self.list_of_constant_features = None
        self.list_of_built_models = []  # use built_model.model.order to get the AR and MA order
        self.built_model_order_to_info_dictionary = []
        self.selected_built_model = None
        self.selected_order_of_autoregression = None
        self.selected_order_of_moving_average = None
        self.train_predictions = []
        self.test_predictions = []

        if not lazy:
            self.train_dataframe = self.dataframe.iloc[:len(self.dataframe) - self.number_of_test_days, :]
            self.test_dataframe = self.dataframe.iloc[len(self.dataframe) - self.number_of_test_days:, :]
            self.processed_dataframe, self.list_of_constant_features = self.process_dataframe(self.dataframe)
            # self.processed_train_dataframe = self.processed_dataframe.iloc[:len(self.processed_dataframe) - self.number_of_test_days, :]
            # self.processed_test_dataframe = self.processed_dataframe.iloc[len(self.processed_dataframe) - self.number_of_test_days:, :]
            self.list_of_built_models = self.build_list_of_model(dataframe=self.processed_dataframe, number_of_test_days=number_of_test_days, max_model_building_convergence_iteration=self.max_model_building_convergence_iteration, order_of_autoregression_lower=self.order_of_autoregression_lower, order_of_autoregression_upper=self.order_of_autoregression_upper, order_of_moving_average_lower=self.order_of_moving_average_lower, order_of_moving_average_upper=self.order_of_moving_average_upper)
            self.built_model_order_to_info_dictionary = self.gather_info_for_built_models(self.list_of_built_models)
            self.selected_built_model = self.select_model(list_of_built_models=self.list_of_built_models)
            self.selected_order_of_autoregression = self.selected_built_model.model.order[0]
            self.selected_order_of_moving_average = self.selected_built_model.model.order[1]
            if self.generate_train_predictions:
                self.train_predictions = self.make_train_prediction(built_model=self.selected_built_model)
            self.test_predictions = self.make_test_prediction(built_model=self.selected_built_model, number_of_days_to_forecast=number_of_test_days, number_of_days_to_predict_ahead=self.number_of_days_to_predict_ahead)
            if self.reduce_memory:
                self.list_of_built_models = None

    @staticmethod
    def process_dataframe(input_dataframe):
        """Remove the constant columns in a dataframe.

        Note: Theoretically this is not the correct implementation. The correct implementation should find constant columns based on input_dataframe.iloc[:dataframe.shape[0]-number_of_test_days, ] instead of input_dataframe. It is because input_dataframe.iloc[:dataframe.shape[0]-number_of_test_days, ] is the data for building model. But untill there is an numpy.linalg.LinAlgError('Schur decomposition solver error.') or Matrix not positive definite (probably looks like this), just let it be what it is now.
        """
        constant_removed_df, list_of_constant_features = VarmaModel.remove_dataframe_constant_column(input_dataframe)
        return constant_removed_df, list_of_constant_features

    @staticmethod
    def remove_dataframe_constant_column(input_dataframe):
        feature_to_non_constant_column_mapping = (input_dataframe != input_dataframe.iloc[0, :]).any()
        constant_removed_df = input_dataframe.loc[:, feature_to_non_constant_column_mapping]
        list_of_constant_feature_names = input_dataframe.columns[feature_to_non_constant_column_mapping == False].tolist()
        return constant_removed_df, list_of_constant_feature_names

    @staticmethod
    def build_model(dataframe, number_of_test_days=1, max_model_building_convergence_iteration=1000, order_of_autoregression=2, order_of_moving_average=3):
        """Try to build a VARMA model with the given dataframe and order."""
        build_success = True
        built_model = None

        try:
            train_dataframe = dataframe.iloc[:dataframe.shape[0]-number_of_test_days, ]
            naive_model = sm.tsa.VARMAX(train_dataframe, order=(order_of_autoregression, order_of_moving_average)) # does not rely on index of train_endog, can just do train_endog.reset_index(drop=True)
            built_model = naive_model.fit(maxiter=max_model_building_convergence_iteration)
        except ValueError:  # when both AR and MA orders are 0
            build_success = False
        except np.linalg.LinAlgError as e:
            # when there is no constant column but model fit still has problem
            error_not_due_to_positive_definite = '-th leading minor of the array is not positive definite' not in str(e)
            error_not_due_to_schur_decomposition_solver = 'Schur decomposition solver error' not in str(e)
            error_not_due_to_lu_decomposition = 'LU decomposition error' not in str(e)
            error_not_due_to_matrix_not_positive_definite = 'Matrix is not positive definite' not in str(e)

            unexpected_error = all([error_not_due_to_positive_definite, error_not_due_to_schur_decomposition_solver, error_not_due_to_lu_decomposition, error_not_due_to_matrix_not_positive_definite])

            if unexpected_error:
                print('VARMA build model unexpected error'+str(e))
                raise NotImplementedError
            build_success = False
        return built_model, build_success

    @staticmethod
    def build_list_of_model(dataframe, number_of_test_days, max_model_building_convergence_iteration, order_of_autoregression_lower, order_of_autoregression_upper, order_of_moving_average_lower, order_of_moving_average_upper):
        """Build a list of models with different orders on the same dataframe using grid search."""
        list_of_built_models = []
        for order_of_autoregression in range(order_of_autoregression_lower, order_of_autoregression_upper+1):
            for order_of_moving_average in range(order_of_moving_average_lower, order_of_moving_average_upper+1):
                print('Start building VARMA model of order {}, {}'.format(order_of_autoregression, order_of_moving_average))
                built_model, build_success = VarmaModel.build_model(dataframe=dataframe, number_of_test_days=number_of_test_days, max_model_building_convergence_iteration=max_model_building_convergence_iteration, order_of_autoregression=order_of_autoregression, order_of_moving_average=order_of_moving_average)

                if build_success:
                    list_of_built_models.append(built_model)

        return list_of_built_models

    @staticmethod
    def gather_info_for_built_models(list_of_built_models):
        """Create a mapping between built models and AIC"""
        built_model_order_to_info_dictionary = {}
        for built_model in list_of_built_models:
            built_model_order_to_info_dictionary[built_model.model.order] = {
                'aic': built_model.aic,
                'bic': built_model.bic
            }

        return built_model_order_to_info_dictionary

    @staticmethod
    def select_model(list_of_built_models, sort_function=lambda built_model: built_model.aic):
        """Select the best model among a set of built VARMA models based on (the lowest) AIC."""
        selected_model = sorted(list_of_built_models, key=sort_function)[0]
        return selected_model

    @staticmethod
    def make_test_prediction(built_model, number_of_days_to_forecast=1, number_of_days_to_predict_ahead=1):
        """Return a number of forecasts for a model on all variables starting from the forecast day."""
        number_of_skipped_days = number_of_days_to_predict_ahead - 1  # forecast() defaults to skip 1 day already
        total_forecast_days = number_of_skipped_days  +number_of_days_to_forecast
        test_predictions = built_model.forecast(steps=total_forecast_days).iloc[number_of_skipped_days:, :]
        return test_predictions

    @staticmethod
    def make_train_prediction(built_model):
        """Make in-sample prediction for all samples."""
        train_predictions = built_model.predict()
        return train_predictions