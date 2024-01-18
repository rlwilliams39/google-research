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

####### Helper Functions
### 1: Collect Global Weights as List
### 2: Collect Dictionary of Weights by Edge

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

####### Weight Generator Functions
### 1: Ad Hoc Approach
### 2: Simple Normal Approach
### 3: Autoregressive Normal Approach

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
    
    ## First, get global statistics
    global_weights = collected_weights(train_graphs)
    np_all_weights = np.log(np.exp(np.array(all_weights))-1)
    
    mu_hat_glob = np.mean(np_all_weights)
    s_hat_glob = np.std(np_all_weights, ddof = 1)
    
    ## Next, get edge-specific statistics
    w_dict = weights_dict(train_graphs)
    params_dict = dict()
    
    for key in w_dict:
        if len(w_dict[key] <= 1):
            continue
        else:
            weights = w_dict[key]
            np_weights = np.log(np.exp(np.array(weight)))
            
            mu_hat = np.mean(np_weights)
            s_hat = np.std(np_weights, ddof = 1)
            
            params_dict[key] = [mu_hat, s_hat]
    
    ## Generate Edge Weights
    edges_feats = []
    for (n1, n2) in edges:
        if (n1, n2) in params_dict:
            mu_hat = params_dict[(n1, n2)][0]
            s_hat = params_dict[(n1, n2)][1]
        else:
            mu_hat = mu_hat_glob
            s_hat = s_hat_glob
        w = np.random.normal(mu_hat, s_hat, 1)
        w = np.log(np.exp(w)+1)
        edge_feats.append(w)
    
    return edge_feats



#### Weight Generator Selector
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
