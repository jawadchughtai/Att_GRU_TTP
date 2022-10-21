# -*- coding: utf-8 -*-
import pandas as pd
import numpy as np
from sklearn.metrics import mean_squared_error,mean_absolute_error
import numpy.linalg as la
import math
from sklearn.svm import SVR
from statsmodels.tsa.arima.model import ARIMA
from xgboost import XGBRegressor 


def load_qt_data(dataset):
    qt_tf = pd.read_csv('D:/Latest/T-GCN-master/data/merged_speed_road_data_travel_time_tt_matrix.csv')    # sz_tf = pd.read_csv(r'/home/abid/Desktop/Progress_report/rush_hours_matrix.csv')
    return qt_tf

def data_preprocessing(ds, ts, rate, len_hist, len_pred):
    ds1 = np.mat(ds)
    size_train = int(ts * rate)
    data_train = ds1[0:size_train]
    data_test = ds1[size_train:ts]
    
    Xtr, Ytr, Xtst, Ytst = [], [], [], []
    for j in range(len(data_train) - len_hist - len_pred):
        x = data_train[j: j + len_hist + len_pred]
        Xtr.append(x[0 : len_hist])
        Ytr.append(x[len_hist : len_hist + len_pred])
    for j in range(len(data_test) - len_hist -len_pred):
        y = data_test[j: j + len_hist + len_pred]
        Xtst.append(y[0 : len_hist])
        Ytst.append(y[len_hist : len_hist + len_pred])
    return Xtr, Ytr, Xtst, Ytst
    
def eval_measure(target,predicted):
    rmse_score = math.sqrt(mean_squared_error(target,predicted))
    mae_score = mean_absolute_error(target, predicted)
    mape_score = np.mean(np.abs((target - predicted) / target)) * 100
    r2_score = 1-((target-predicted)**2).sum()/((target-target.mean())**2).sum()
    F_norm_score = la.norm(target-predicted)/la.norm(target)
    return rmse_score, mae_score, mape_score, r2_score, 1-F_norm_score

### Loading Dataset
ds = load_qt_data('qt')
ts = ds.shape[0]
nodes_total = ds.shape[1]
data_split_tr = 0.8
len_hist = 4
len_pred = 1
Xtr,Ytr,Xtst,Ytst = data_preprocessing(ds, ts, data_split_tr, len_hist, len_pred)
approach = 'SVR' ####  HA or SVR or XGB or MLP or ARIMA

########### HA #############
if approach == 'HA':
    outcome = []  
    
    
    for i in range(len(Xtst)):
        a = np.array(Xtst[i])
        tempOutcome = []

        a1 = np.mean(a, axis=0)
        tempOutcome.append(a1)
        # a = a[1:]
        # a = np.append(a, [a1], axis=0)
        # a1 = np.mean(a, axis=0)
        # tempOutcome.append(a1)
        # a = a[1:]
        # a = np.append(a, [a1], axis=0)
        # a1 = np.mean(a, axis=0)
        # tempOutcome.append(a1)
        # a = a[1:]
        # a = np.append(a, [a1], axis=0)
        # a1 = np.mean(a, axis=0)
        # tempOutcome.append(a1)
        outcome.append(tempOutcome)
    outcome1 = np.array(outcome)
    outcome1 = np.reshape(outcome1, [-1,nodes_total])
    Ytst1 = np.array(Ytst)
    Ytst1 = np.reshape(Ytst1, [-1,nodes_total])
    print(Ytst1.shape, outcome1.shape)
    rmse_score, mae_score, mape_score, r2_score, accuracy_score = eval_measure(Ytst1, outcome1)  
    print('rmse_HA:%r'%rmse_score,
          'mae_HA:%r'%mae_score,
          'mape_HA:%r'%mape_score,
          'r2_HA:%r'%r2_score,
          'accur_HA:%r'%accuracy_score)


############ SVR #############
if approach == 'SVR':  
    rmse_overall, mae_overall, accur_overall, outcome = [],[],[],[]
    for i in range(nodes_total):
        ds1 = np.mat(ds)
        a = ds1[:,i]
        a_X, a_Y, t_X, t_Y = data_preprocessing(a, ts, data_split_tr, len_hist, len_pred)
        a_X = np.array(a_X)
        a_X = np.reshape(a_X,[-1, len_hist])
        a_Y = np.array(a_Y)
        a_Y = np.reshape(a_Y,[-1, len_pred])
        a_Y = np.mean(a_Y, axis=1)
        t_X = np.array(t_X)
        t_X = np.reshape(t_X,[-1, len_hist])
        t_Y = np.array(t_Y)
        t_Y = np.reshape(t_Y,[-1, len_pred])    
       
        svr_model=SVR(kernel='rbf')
        svr_model.fit(a_X, a_Y)
        pre = svr_model.predict(t_X)
        pre = np.array(np.transpose(np.mat(pre)))
        pre = pre.repeat(len_pred ,axis=1)
        outcome.append(pre)
    outcome1 = np.array(outcome)
    outcome1 = np.reshape(outcome1, [nodes_total,-1])
    outcome1 = np.transpose(outcome1)
    Ytst1 = np.array(Ytst)
    Ytst1 = np.reshape(Ytst1, [-1,nodes_total])
    overall = np.mat(accur_overall)
    overall[overall<0] = 0
    rmse_score, mae_score, mape_score, r2_score, accur_score = eval_measure(Ytst1, outcome1)
    print('rmse_SVR:%r'%rmse_score,
          'mae_SVR:%r'%mae_score,
          'mape_SVR:%r'%mape_score,
          'r2_SVR:%r'%r2_score,
          'accur_SVR:%r'%accur_score)

############ XGB #############
if approach == 'XGB':  
    rmse_overall, mae_overall, accur_overall, outcome = [],[],[],[]
    for i in range(nodes_total):
        ds1 = np.mat(ds)
        a = ds1[:,i]
        a_X, a_Y, t_X, t_Y = data_preprocessing(a, ts, data_split_tr, len_hist, len_pred)
        a_X = np.array(a_X)
        a_X = np.reshape(a_X,[-1, len_hist])
        a_Y = np.array(a_Y)
        a_Y = np.reshape(a_Y,[-1, len_pred])
        a_Y = np.mean(a_Y, axis=1)
        t_X = np.array(t_X)
        t_X = np.reshape(t_X,[-1, len_hist])
        t_Y = np.array(t_Y)
        t_Y = np.reshape(t_Y,[-1, len_pred])    
       
        xgb_model = XGBRegressor(objective = 'reg:squarederror',max_depth = 7)        
        xgb_model.fit(a_X, a_Y)
        pre = xgb_model.predict(t_X)
        pre = np.array(np.transpose(np.mat(pre)))
        pre = pre.repeat(len_pred ,axis=1)
        outcome.append(pre)
    outcome1 = np.array(outcome)
    outcome1 = np.reshape(outcome1, [nodes_total,-1])
    outcome1 = np.transpose(outcome1)
    Ytst1 = np.array(Ytst)


    Ytst1 = np.reshape(Ytst1, [-1,nodes_total])
    overall = np.mat(accur_overall)
    overall[overall<0] = 0
    rmse_score, mae_score, mape_score, r2_score, accur_score = eval_measure(Ytst1, outcome1)
    print('rmse_XGB:%r'%rmse_score,
          'mae_XGB:%r'%mae_score,
          'mape_XGB:%r'%mape_score,
          'r2_score_XGB:%r'%r2_score,
          'accur_XGB:%r'%accur_score,)


from sklearn.neural_network import MLPRegressor   
if approach == 'MLP':  
    rmse_overall, mae_overall, accur_overall, outcome = [],[],[],[]
    for i in range(nodes_total):
        ds1 = np.mat(ds)
        a = ds1[:,i]
        a_X, a_Y, t_X, t_Y = data_preprocessing(a, ts, data_split_tr, len_hist, len_pred)
        a_X = np.array(a_X)
        a_X = np.reshape(a_X,[-1, len_hist])
        a_Y = np.array(a_Y)
        a_Y = np.reshape(a_Y,[-1, len_pred])
        a_Y = np.mean(a_Y, axis=1)
        t_X = np.array(t_X)
        t_X = np.reshape(t_X,[-1, len_hist])
        t_Y = np.array(t_Y)
        t_Y = np.reshape(t_Y,[-1, len_pred])    
       
        mlp_model=MLPRegressor(hidden_layer_sizes=(64,64,64),activation="relu", max_iter=100).fit(a_X, a_Y)
        # mlp_model.fit(a_X, a_Y)
        pre = mlp_model.predict(t_X)
        pre = np.array(np.transpose(np.mat(pre)))
        pre = pre.repeat(len_pred ,axis=1)
        outcome.append(pre)
    outcome1 = np.array(outcome)
    outcome1 = np.reshape(outcome1, [nodes_total,-1])
    outcome1 = np.transpose(outcome1)
    Ytst1 = np.array(Ytst)


    Ytst1 = np.reshape(Ytst1, [-1,nodes_total])
    overall = np.mat(accur_overall)
    overall[overall<0] = 0
    rmse_score, mae_score, mape_score, r2_score, accur_score = eval_measure(Ytst1, outcome1)
    print('rmse_MLP}:%r'%rmse_score,
          'mae_MLP:%r'%mae_score,
          'mape_MLP:%r'%mape_score,
          'r2_score_MLP:%r'%r2_score,
          'accur_MLP:%r'%accur_score)
    
    
######## Auto Regressive Integrated Moving Average #########'

data30=ds.groupby(np.arange(len(ds))//2).mean()
data45=ds.groupby(np.arange(len(ds))//3).mean()
data60=ds.groupby(np.arange(len(ds))//4).mean()

import warnings
warnings.filterwarnings("ignore")
if approach == 'ARIMA':
    range1 = pd.date_range('4/1/2017',periods=4880, freq='15min')
    d = pd.DatetimeIndex(range1)

    ds.index = d
    rmse_score,mae_score,accur_score,r2_score,pred,ori,mape_score = [],[],[],[],[],[],[],[]
    for j in range(15073):
        ts = ds.iloc[:,j]
        log1=np.log(ts)    
        log1=np.array(log1,dtype=np.float)
        where_are_inf = np.isinf(log1)
        log1[where_are_inf] = 0
        log1 = pd.Series(log1)
        log1.index = a1
        approach = ARIMA(log1,order=[2,0,1])
        Modelfit = approach.fit()
        Ypred = Modelfit.predict(3904, dynamic=True)
        Ypred1 = np.exp(Ypred)
        ts = ts[Ypred1.index]
        rmse1,mae1, mape1, r2_score1, accur1 = eval_measure(ts,Ypred1)
        rmse_score.append(rmse1)
        mae_score.append(mae1)
        mape_score.append(mape1)
        r2_score.append(r2_score1)
        accur_score.append(accur1)
    accur = np.mat(accur_score)
    accur[accur < 0] = 0
    print('rmse_ARIMA:%r'%(np.mean(rmse_score)),
          'mae_ARIMA:%r'%(np.mean(mae_score)),
          'mape_ARIMA:%r'%(np.mean(mape_score)),
          'r2_score_ARIMA:%r'%(np.mean(r2_score)),
          'accur_ARIMA:%r'%(np.mean(accur)))
  