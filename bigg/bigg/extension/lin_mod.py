### For Linear Regression Prediction
import pandas as pd
import numpy as np
from sklearn import linear_model as lm
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

from sklearn.linear_model import SGDRegressor
from sklearn.datasets import load_boston
from sklearn.datasets import make_regression
from sklearn.preprocessing import scale
import matplotlib.pyplot as plt 


class EdgeWeightLinearModel:
    
    def __init__(self, args):
        super().__init__(args)
        self.lin_mod = make_pipeline(StandardScaler(),lm.SGDRegressor(max_iter=1000, tol=1e-3, warm_start = True))
    
    def train(features, weights):
        self.lin_mod.fit(features, weights)
    
    def rep_train(list_features, list_weights):
        for features, weights in zip(list_features, list_weights):
            self.drain(features, weights)
    
    def predict(features):
        weights = self.lin_mod.predict(features)
        return features
