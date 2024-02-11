import logging
from abc import ABC, abstractmethod

import optuna
import pandas as pd
import xgboost as xgb
from lightgbm import LGBMRegressor
from sklearn.ensemble import RandomForestRegressor
from sklearn.linear_model import LinearRegression


class Model(ABC):

    @abstractmethod
    def train(self, X_train, y_train):
        pass


"""
    @abstractmethod
    def optimize(self, trial, X_train, y_train, X_test, y_test):

        pass
"""

"""
class LightGBMModel(Model):


    def train(self, X_train, y_train, **kwargs):
        reg = LGBMRegressor(**kwargs)
        reg.fit(X_train, y_train)
        return reg

    def optimize(self, trial, X_train, y_train, X_test, y_test):
        n_estimators = trial.suggest_int("n_estimators", 1, 200)
        max_depth = trial.suggest_int("max_depth", 1, 20)
        learning_rate = trial.suggest_uniform("learning_rate", 0.01, 0.99)
        reg = self.train(X_train, y_train, n_estimators=n_estimators,
                         learning_rate=learning_rate, max_depth=max_depth)
        return reg.score(X_test, y_test)


class XGBoostModel(Model):
    """
# XGBoostModel that implements the Model interface.

"""
    def train(self, X_train, y_train, **kwargs):
        reg = xgb.XGBRegressor(**kwargs)
        reg.fit(X_train, y_train)
        return reg

    def optimize(self, trial, X_train, y_train, X_test, y_test):
        n_estimators = trial.suggest_int("n_estimators", 1, 200)
        max_depth = trial.suggest_int("max_depth", 1, 30)
        learning_rate = trial.suggest_loguniform("learning_rate", 1e-7, 10.0)
        reg = self.train(X_train, y_train, n_estimators=n_estimators,
                         learning_rate=learning_rate, max_depth=max_depth)
        return reg.score(X_test, y_test)
"""


class LinearRegressionModel(Model):

    def train(self, X_train, y_train, **kwargs):

        try:
            reg = LinearRegression(**kwargs)
            reg.fit(X_train, y_train)
            return reg
        except Exception as e:
            logging.error("")
            raise e


""" # For linear regression, there might not be hyperparameters that we want to tune, so we can simply return the score
    def optimize(self, trial, X_train, y_train, X_test, y_test):
        reg = self.train(X_train, y_train)
        return reg.score(X_test, y_test)
"""
