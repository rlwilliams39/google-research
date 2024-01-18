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

####### AD HOC APPROACH

def collected_weights(train_graphs):
    coll_weights = []
    for g in train_graphs:
        weights = [w for (_, _, w) in g.edges(data = True)]
        coll_weights += weights
    return coll_weights

def weights_dict(train_graphs):
    w_dict = dict()
    for g in train_graphs:
        for (n1, n2, w) in g.edges(data=True):
            if (n1, n2) not in w_dict:
                w_dict[(n1, n2)] = [w['weight']]
            else:
                w_dict[(n1, n2)] += [w['weight']]
    return w_dict

def edge_estimator(train_graphs):
    n = len(train_graphs)
    edge_probs = dict()
    
    for g in train_graphs:
        for (n1, n2) in g.edges(data=False):
            if (n1, n2) not in edge_probs:
                edge_probs[(n1, n2)] = 1/n
            else:
                edge_probs[(n1, n2)] += 1/n
    
    return edge_probs

def weight_generator_AH(edges, train_graphs):
    w_dict = weights_dict(train_graphs)
    global_weights = collected_weights(train_graphs)
    edge_feats = []
    for (n1, n2) in edges:
        if (n1, n2) in w_dict:
            weights = w_dict[(n1, n2)]
            w = random.sample(weights, 1).pop()
            edge_feats.append(w)
        else:
           w = random.sample(global_weights, 1).pop()
           edge_feats.append(w)
    return edge_feats

def weight_generator_SN(edges, train_graphs):
    global_weights = collected_weights(train_graphs)
    np_all_weights = np.log(np.exp(np.array(all_weights))-1)
    
    mu_hat = np.mean(np_all_weights)
    s_hat = np.std(np_all_weights, ddof = 1)
    
    pred_edge_feats = np.random.normal(mu_hat, s_hat, len(edges))
    pred_edge_feats = np.log(np.exp(np.array(pred_edge_feats))+1)
    return pred_edge_feats

def weight_generator_AN(edges, train_graphs):
    ### IMPLEMENTATION IN PROGRESS
    return 0


def weight_generator(arg, edges, edge_feats, train_graphs):
    print("Current Weight Generator: ", arg)
    print("Note: Currently supports 'Simple Normal'; 'Auto Normal'; 'Ad Hoc'; 'Identity'")
    if arg == "Simple Normal":
        edge_feats = weight_generator_SN(edges, train_graphs)
    if arg == "Auto Normal":
        edge_feats = weight_generator_AN(edges, train_graphs)
    if arg == "Ad Hoc":
        edge_feats = weight_generator_AH(edges, train_graphs)
    return(edge_feats)




#class EdgeWeightLinearModel:
#    
#    def __init__(self, args):
#        #super().__init__(args)
#        self.lin_mod = make_pipeline(StandardScaler(),lm.SGDRegressor(max_iter=1000, tol=1e-3, warm_start = True))
#    
#    def train(self, features, weights):
#        weights = weights.cpu().detach().numpy()
#        print(features)
#        print(weights)
#        self.lin_mod.fit(features, weights)
#    
#    def rep_train(self, list_features, list_weights):
#        for features, weights in zip(list_features, list_weights):
#            self.drain(features, weights)
#    
#    def predict(self, features):
#        weights = self.lin_mod.predict(features)
#        return features
