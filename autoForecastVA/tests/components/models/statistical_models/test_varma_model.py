import pandas as pd
import numpy as np
import statsmodels.api as sm
import matplotlib.pyplot as plt

import unittest

from autoForecastVA.src.components.models.statistical_models.varma_model import VarmaModel


class TestVarmaModel(unittest.TestCase):
    def test_toy_example(self):

        # inputs
        number_of_test_days = 20  # should be 1 in pipelien
        order_of_autoregression = 2
        order_of_moving_average = 3

        # outputs
        # VAR
        dta = sm.datasets.webuse('lutkepohl2', 'https://www.stata-press.com/data/r12/')
        dta.index = dta.qtr
        dta.index.freq = dta.index.inferred_freq
        endog = dta.loc['1960-04-01':'1978-10-01', ['dln_inv', 'dln_inc', 'dln_consump']]
        endog.shape  # 75, 3

        train_endog = endog.iloc[:-number_of_test_days, ]
        mod = sm.tsa.VARMAX(train_endog, order=(order_of_autoregression, order_of_moving_average)) # does not rely on index of train_endog, can just do train_endog.reset_index(drop=True)
        built_model = mod.fit(maxiter=1000)
        print(built_model.summary())
        print(dir(built_model))
        built_model.aic
        built_model.bic
        built_model.model.order
        built_model.forecast(10).iloc[0, 0]
        built_model.forecast(1)


        mod1 = sm.tsa.VARMAX(train_endog, order=(1, 3))
        res1 = mod1.fit(maxiter=1000)
        mod2 = sm.tsa.VARMAX(train_endog, order=(1, 5))
        res2 = mod2.fit(maxiter=1000)
        mod3 = sm.tsa.VARMAX(train_endog, order=(3, 5))
        res3 = mod3.fit(maxiter=1000)

        res1_automated, built_success = VarmaModel.build_model(dataframe=train_endog, number_of_test_days=0, order_of_autoregression=1, order_of_moving_average=3)
        self.assertEqual(res1_automated.aic, res1.aic)

        res1.aic  # -825 should be this
        res2.aic  # -799
        res3.aic  # -768
        list_of_built_models = [res1, res2, res3]

        self.assertEqual(VarmaModel.select_model(list_of_built_models).model.order, (1,3))
        self.assertAlmostEqual(VarmaModel.select_model(list_of_built_models).aic, -825.62905, places=5)
        self.assertAlmostEqual(VarmaModel.make_test_prediction(res1).iloc[0, 0], 0.02762, places=5)
        varmaModel = VarmaModel(dataframe=train_endog, order_of_autoregression_upper=1, order_of_moving_average_upper=1, number_of_days_to_predict_ahead=2)
        self.assertAlmostEqual(varmaModel.test_predictions.iloc[0, 0], 0.020551, places=5)








