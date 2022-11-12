import pickle as pkl
import tensorflow as tf
import pandas as pd
import numpy as np
import math
import os
import numpy.linalg as la
from RNN import RNN 
# from plotting import calc_graphs, calc_errors
from plotting import calc_graphs
from sklearn.metrics import mean_squared_error,mean_absolute_error
import matplotlib.pyplot as plt
from sklearn.preprocessing import MinMaxScaler


def load_qt_data(dataset):
    #qt_tf = pd.read_csv(r'D:/Latest/T-GCN-master/data/merged_speed_road_data_travel_time_tt_matrix.csv')
    qt_tf = pd.read_csv(r'D:/Latest/T-GCN-master/data/df_merged.csv') ### concatenate the two files in Dataset folder with pd.concat as df_merged = pd.concat([df_a, df_b], axis=1)

    return qt_tf

def data_preprocessing(ds, ts, split_tr, len_hist, len_pred):
    split_size_tr = int(ts * split_tr)
    data_tr = ds[0:split_size_tr]
    data_tst = ds[split_size_tr:ts]
    
    Xtr, Ytr, Xtst, Ytst = [], [], [], []
    for i in range(len(data_tr) - len_hist - len_pred):
        temp = data_tr[i: i + len_hist + len_pred]
        Xtr.append(temp[0 : len_hist])
        Ytr.append(temp[len_hist : len_hist + len_pred])
    for i in range(len(data_tst) - len_hist -len_pred):
        temp1 = data_tst[i: i + len_hist + len_pred]
        Xtst.append(temp1[0 : len_hist])
        Ytst.append(temp1[len_hist : len_hist + len_pred])
      
    Xtr = np.array(Xtr)
    Ytr = np.array(Ytr)
    Xtst = np.array(Xtst)
    Ytst = np.array(Ytst)
    return Xtr, Ytr, Xtst, Ytst

def att_GRU(_D, wts, bias):
    ###
    def_cell = RNN(hidden_units_gru, nodes_total)
    cell_dfd = tf.nn.rnn_cell.MultiRNNCell([def_cell], state_is_tuple=True)
    _D = tf.unstack(_D, axis=1)
    outcomes, stt = tf.nn.static_rnn(cell_dfd, _D, dtype=tf.float32)

    res = tf.concat(outcomes, axis=0)
    res = tf.reshape(res, shape=[len_hist,-1,nodes_total,hidden_units_gru])
    res = tf.transpose(res, perm=[1,0,2,3])

    outcome_e,alp = comp_attention(res, attention_wts, attention_bias)

    outcome = tf.reshape(outcome_e,shape=[-1,len_hist])
    outcome = tf.matmul(outcome, wts['output']) + bias['output']
    outcome = tf.reshape(outcome,shape=[-1,nodes_total,len_pred])
    outcome = tf.transpose(outcome, perm=[0,2,1])
    outcome = tf.reshape(outcome, shape=[-1,nodes_total])

    return outcome

def GRU(_D, model_weights, model_bias):
    ###
    def_cell = RNN(hidden_units_gru, nodes_total=nodes_total)
    grucell = tf.nn.rnn_cell.MultiRNNCell([def_cell], state_is_tuple=True)
    _D = tf.unstack(_D, axis=1)
    outcomes, _ = tf.nn.static_rnn(grucell, _D, dtype=tf.float32)
    last = []
    for j in outcomes:
        k = tf.reshape(j,shape=[-1,nodes_total,hidden_units_gru])
        k = tf.reshape(k,shape=[-1,hidden_units_gru])
        last.append(k)
    outcome_last = last[-1]
    outcome = tf.matmul(outcome_last, model_weights['output']) + model_bias['output']
    outcome = tf.reshape(outcome,shape=[-1,nodes_total,len_pred])
    outcome = tf.transpose(outcome, perm=[0,2,1])
    outcome = tf.reshape(outcome, shape=[-1,nodes_total])
    return outcome
    
def comp_attention(x, weight_att,bias_att):
    x = tf.matmul(tf.reshape(x,[-1,hidden_units_gru]),weight_att['wts1']) + bias_att['bias1']
    f = tf.matmul(tf.reshape(x, [-1, nodes_total]), weight_att['wts2']) + bias_att['bias2']
    g = tf.matmul(tf.reshape(x, [-1, nodes_total]), weight_att['wts2']) + bias_att['bias2']
    h = tf.matmul(tf.reshape(x, [-1, nodes_total]), weight_att['wts2']) + bias_att['bias2']

    f1 = tf.reshape(f, [-1,len_hist])
    g1 = tf.reshape(g, [-1,len_hist])
    h1 = tf.reshape(h, [-1,len_hist])
    s = g1 * f1

    beta = tf.nn.softmax(s, dim=-1)
    context = tf.expand_dims(beta,2) * tf.reshape(x,[-1,len_hist,nodes_total])

    context = tf.transpose(context,perm=[0,2,1])
    return context, beta

def eval_measure(target,predicted):
    rmse_score = math.sqrt(mean_squared_error(target,predicted))
    mae_score = mean_absolute_error(target, predicted)
    mape_score = np.mean(np.abs((target - predicted) / target)) * 100
    r2_score = 1-((target-predicted)**2).sum()/((target-target.mean())**2).sum()
    normF_score = la.norm(target-predicted,'fro')/la.norm(target,'fro')
    return rmse_score, mae_score, mape_score, r2_score, 1-normF_score

########################### MOdel Parameter Definitions ###########################
approach = 'att_GRU' ### GRU or att_GRU
dst_name = 'qt' ### dataset name 
data_split_tr =  0.8 ### 80:20 ratio
len_hist = 4 ### previous one hour i.e., 4
out_dimension = len_pred = 1 ### predicted upto 1 hour i.e., 4
b_s = 32 ### 16,32,64
lr = 0.001 ### set to 0.001
number_of_epochs = 600 ### set to 600
hidden_units_gru = 32 ### 8,16,32,64,128
lambda_val = 0.0015  ### set to 0.0015

### Loading Dataset
ds = load_qt_data('qt')
# ds,adj_mat = load_qt_data('qt')

ts = ds.shape[0]
nodes_total = ds.shape[1]
ds1 =np.mat(ds,dtype=np.float32)

# # ### Generalization Test
# noise_guass = np.random.normal(0,0.2,size=ds.shape)
# sc = MinMaxScaler()
# sc.fit(noise_guass)
# noise_guass = sc.transform(noise_guass)
# ds1 = ds1 + noise_guass

#### Normalizing Data
max_val = np.max(ds1)
ds1  = ds1/max_val

#### Splitting Data into Training and Test
Xtr, Ytr, Xtst, Ytst = data_preprocessing(ds1, ts, data_split_tr, len_hist, len_pred)


#### Defining Placeholders 
data = tf.placeholder(tf.float32, shape=[None, len_hist, nodes_total])
targets = tf.placeholder(tf.float32, shape=[None, len_pred, nodes_total])

model_weights = {
    'output': tf.Variable(tf.random_normal([hidden_units_gru, len_pred], mean=1.0), name='gru_weights')}
model_bias = {
    'output': tf.Variable(tf.random_normal([len_pred]),name='gru_biases')}
att_weights = {
    'output': tf.Variable(tf.random_normal([len_hist, len_pred], mean=1.0), name='output_weight')}
att_bias = {
    'output': tf.Variable(tf.random_normal([len_pred]),name='output_bias')}
attention_wts={
    'wts1':tf.Variable(tf.random_normal([hidden_units_gru,1], stddev=0.1),name='attention_wts1'),
    'wts2':tf.Variable(tf.random_normal([nodes_total,1], stddev=0.1),name='attention_wts2')}
attention_bias = {
    'bias1': tf.Variable(tf.random_normal([1]),name='attention_bias1'),
    'bias2': tf.Variable(tf.random_normal([1]),name='attention_bias2')}



if approach == 'att_GRU':
    prediction = att_GRU(data, att_weights, att_bias)
if approach == 'GRU':
    prediction = GRU(data, model_weights, model_bias)    
    
predY = prediction
             
#### Adding Normalization term for Regularization
norm = lambda_val * sum(tf.nn.l2_loss(tf_var) for tf_var in tf.trainable_variables())
target = tf.reshape(targets, [-1,nodes_total])
calc_loss = tf.reduce_mean(tf.nn.l2_loss(predY-target) + norm)
calc_error = tf.sqrt(tf.reduce_mean(tf.square(predY-target)))
opt = tf.train.AdamOptimizer(lr).minimize(calc_loss)

#### Other Settings
varss = tf.global_variables()
svr = tf.train.Saver(tf.global_variables())  
features_GPU = tf.GPUOptions(per_process_gpu_memory_fraction=0.333)
det_session = tf.Session(config=tf.ConfigProto(gpu_options=features_GPU))
det_session.run(tf.global_variables_initializer())

Results = 'Results/%s'%(approach)
pth = '%s_%s_lr%r_bs%r_GRUunit%r_len_hist%r_len_pred%r_numepoch%r'%(approach,dst_name,lr,b_s,hidden_units_gru,len_hist,len_pred,number_of_epochs)
# path1 = '%s_%s_lr%r_bs%r_GRUunit%r_len_hist%r_len_pred%r_numepoch%r_guassian_0.2'%(approach,dst_name,lr,b_s,hidden_units_gru,len_hist,len_pred,number_of_epochs)
dir_path = os.path.join(Results,pth)
if not os.path.exists(dir_path):
    os.makedirs(dir_path)
          
# loss_trn,rmse_trn,mae_trn,acc_trn,r2_trn,pred_trn, mape_trn = [],[],[],[],[],[],[]
loss_tst,rmse_tst,mae_tst,acc_tst,r2_tst,pred_tst, mape_tst = [],[],[],[],[],[],[]
bt_los,bt_rmse = [], []

tot_b_s = int(Xtr.shape[0]/b_s)
for iterations in range(number_of_epochs):
    for j in range(tot_b_s):
        bt_min = Xtr[j * b_s : (j+1) * b_s]
        lbl_min = Ytr[j * b_s : (j+1) * b_s]
        _, los, rms, _ = det_session.run([opt, calc_loss, calc_error, predY],
                                                 feed_dict = {data:bt_min, targets:lbl_min})
        bt_los.append(los)
        bt_rmse.append(rms * max_val)

    los1, rms1, out_tst = det_session.run([calc_loss, calc_error, predY],
                                         feed_dict = {data:Xtst, targets:Ytst})
    # lbl_trn = np.reshape(trainY,[-1,num_nodes])
    lbl_tst = np.reshape(Ytst,[-1,nodes_total])
    # rmsetr, maetr, mapetr, r2tr, accurtr = eval_measure(lbl_trn, out_trn)
    rmsetst, maetst, mapetst, r2tst, accurtst  = eval_measure(lbl_tst, out_tst)
    lbl_tst1 = lbl_tst * max_val
    # out_trn1 = out_trn * max_val
    out_tst1 = out_tst * max_val
    loss_tst.append(los1)
    rmse_tst.append(rmsetst * max_val)
    # mae_trn.append(maetr * max_val)
    mae_tst.append(maetst * max_val)
    # acc_trn.append(accurtr)
    # r2_trn.append(r2tr)
    # pred_trn.append(out_trn1)
    # mape_trn.append(mapetr)
    
    acc_tst.append(accurtst)
    r2_tst.append(r2tst)
    pred_tst.append(out_tst1)
    mape_tst.append(mapetst)

    print('Iteration:{}'.format(iterations),
          'rmse_tr:{:.4}'.format(bt_rmse[-1]),
          'loss_tst:{:.4}'.format(los1),
          'rmse_tst:{:.4}'.format(rmsetst),
          'accur_tst:{:.4}'.format(accurtst),
          'mape_tst:{:.4}'.format(mapetst))
    if (iterations % 50 == 0):        
        svr.save(det_session, dir_path+'/approach_32/pred_GRU%r'%iterations, global_step = iterations)

############################ Results ############################
x = int(len(bt_rmse)/tot_b_s)
rmse_bt = [k for k in bt_rmse]
rmse_trn = [(sum(rmse_bt[k*tot_b_s:(k+1)*tot_b_s])/tot_b_s) for k in range(x)]
loss_bt = [k for k in bt_los]
loss_trn = [(sum(loss_bt[k*tot_b_s:(k+1)*tot_b_s])/tot_b_s) for k in range(x)]
result = pd.DataFrame(loss_bt)
result.to_csv(dir_path+'/batch_loss.csv',index = False,header = False)
result = pd.DataFrame(loss_trn)
result.to_csv(dir_path+'/train_loss.csv',index = False,header = False)
result = pd.DataFrame(rmse_bt)
result.to_csv(dir_path+'/batch_rmse.csv',index = False,header = False)
result = pd.DataFrame(rmse_trn)
result.to_csv(dir_path+'/train_rmse.csv',index = False,header = False)
result = pd.DataFrame(loss_tst)
result.to_csv(dir_path+'/loss_test_data.csv',index = False,header = False)
# result = pd.DataFrame(acc_trn)
# result.to_csv(path+'/accur_train_data.csv',index = False,header = False)
result = pd.DataFrame(acc_tst)
result.to_csv(dir_path+'/accur_test_data.csv',index = False,header = False)
result = pd.DataFrame(rmse_tst)
result.to_csv(dir_path+'/rmse_test_data.csv',index = False,header = False)
# result = pd.DataFrame(mae_trn)
# result.to_csv(path+'/mae_train_data.csv',index = False,header = False)
result = pd.DataFrame(mae_tst)
result.to_csv(dir_path+'/mae_test_data.csv',index = False,header = False)
# result = pd.DataFrame(r2_trn)
# result.to_csv(path+'/r2_train_data.csv',index = False,header = False)
# result = pd.DataFrame(r2_tst)
# result.to_csv(path+'/r2_test_data.csv',index = False,header = False)
# result = pd.DataFrame(mape_trn)
# result.to_csv(path+'/mape_train.data.csv',index = False,header = False)
# result = pd.DataFrame(mape_tst)
# result.to_csv(path+'/mape_test_data.csv',index = False,header = False)

inx = rmse_tst.index(np.min(rmse_tst))
res_tst = pred_tst[inx]
result = pd.DataFrame(res_tst)
result.to_csv(dir_path+'/result_test_data.csv',index = False,header = False)
calc_graphs(res_tst,lbl_tst1,dir_path)
# calc_errors(rmse_trn,loss_trn,rmse_tst,loss_tst,mae_tst,acc_tst,dir_path)
# calc_errors(rmse_trn,loss_trn,rmse_tst,mae_trn,
#            mae_tst,mape_trn,mape_tst,acc_trn,acc_tst,r2_trn,r2_tst,path)

# fig1 = plt.figure(figsize=(7,3))
# ax1 = fig1.add_subplot(1,1,1)
# plt.plot(np.sum(alpha1,0))
# plt.savefig(path+'/alpha.jpg',dpi=500)
# plt.show()


# plt.imshow(np.mat(np.sum(alpha1,0)))
# plt.savefig(path+'/alpha11.jpg',dpi=500)
# plt.show()

print('rmse_min:%r'%(np.min(rmse_tst)),
      'mae_min:%r'%(mae_tst[inx]),
      'accur_max:%r'%(acc_tst[inx]),
      'r2_score:%r'%(r2_tst[inx]),
      'mape_score:%r'%mape_tst[inx])
