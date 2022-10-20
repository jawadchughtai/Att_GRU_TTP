import matplotlib.pyplot as plt
import matplotlib.dates as md
import pandas as pd

def calc_graphs(res_tst,lbl_tst,pth):
    # times = pd.date_range('2017-05-19', periods=1167, freq='15min')
    # res_tst.index = times
    # lbl_tst = pd.DataFrame(lbl_tst)      
    # lbl_tst.index = times
    # a_pred1 = res_tst.iloc[:1167,0]
    # a_true = test_label1.iloc[:1167,0]
    a_pred = res_tst[:,0]
    a_true = lbl_tst[:,0]
    plt.plot(a_pred,label='predicted')
    plt.plot(a_true,label='actual')
    plt.legend(loc='best',fontsize=10)
    # plt.xticks(rotation=10)
    # plt.gca().xaxis.set_major_formatter(md.DateFormatter('%y-%m-%d'))  
    # plt.gca().xaxis.set_major_locator(md.HourLocator(interval=10)) 
    plt.savefig(pth+'/test_all15.eps') ### test_all30, test_all45, test_all60
    plt.show()
    
    # predicted = res_tst[0:96,0]
    # target = lbl_tst[0:96,0]
    # plt.plot(predicted,label="predicted")
    # plt.plot(target,label="actual")
    # plt.legend(loc='best',fontsize=10)
    # plt.xticks(rotation=10)
    # plt.gca().xaxis.set_major_formatter(md.DateFormatter('%y-%m-%d'))  
    # plt.gca().xaxis.set_major_locator(md.HourLocator(interval=10)) 
    # plt.savefig(pth+'/test_1day15.eps')
    # plt.show()
    
    predicted = res_tst[0:192,0]
    target = lbl_tst[0:192,0]
    plt.plot(predicted,label="predicted")
    plt.plot(target,label="actual")
    plt.legend(loc='best',fontsize=10)
    # plt.xticks(rotation=10)
    # plt.gca().xaxis.set_major_formatter(md.DateFormatter('%y-%m-%d'))  
    # plt.gca().xaxis.set_major_locator(md.HourLocator(interval=10)) 
    plt.savefig(pth+'/test_2days15.eps') ### test_2days30, test_2days45, test_2days60
    plt.show()
    
    # predicted = res_tst[0:672,0]
    # target = lbl_tst[0:672,0]
    # plt.plot(predicted,label="prediction")
    # plt.plot(target,label="true")
    # plt.legend(loc='best',fontsize=10)
    # plt.xticks(rotation=10)
    # plt.gca().xaxis.set_major_formatter(md.DateFormatter('%y-%m-%d'))  
    # plt.gca().xaxis.set_major_locator(md.HourLocator(interval=10)) 
    # plt.savefig(pth+'/test_7days15.eps')
    # plt.show()


# def calc_errors(rmse_trn,loss_trn,rmse_tst,loss_tst,mae_tst,accu_tst,pth):
#     plt.plot(rmse_trn,  label="train_rmse")
#     plt.plot(rmse_tst,  label="test_rmse")
#     plt.legend(loc='best',fontsize=10)
#     plt.savefig(pth+'/rmse.eps')
#     plt.show()
    
#     plt.plot(loss_trn,  label="train_loss")
#     plt.plot(loss_tst,  label="test_loss")
#     plt.legend(loc='best',fontsize=10)
#     plt.savefig(pth+'/loss.eps')
#     plt.show()

#     plt.plot(loss_trn, label='train_loss')
#     plt.legend(loc='best',fontsize=10)
#     plt.savefig(pth+'/train_loss.eps')
#     plt.show()

#     plt.plot(rmse_trn, label='train_rmse')
#     plt.legend(loc='best',fontsize=10)
#     plt.savefig(pth+'/train_rmse.eps')
#     plt.show()

#     plt.plot(accu_tst,  label="test_acc")
#     plt.legend(loc='best',fontsize=10)
#     plt.savefig(pth+'/test_acc.eps')
#     plt.show()
    
#     plt.plot(rmse_tst,  label="test_rmse")
#     plt.legend(loc='best',fontsize=10)
#     plt.savefig(pth+'/test_rmse.eps')
#     plt.show()
    
#     plt.plot(mae_tst, label="test_mae")
#     plt.legend(loc='best',fontsize=10)
#     plt.savefig(pth+'/test_mae.eps')
#     plt.show()
    
