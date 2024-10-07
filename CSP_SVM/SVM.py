import numpy as np
from sklearn.svm import LinearSVC
from sklearn.metrics import accuracy_score
import pandas as pd
import scipy.io
from sklearn.model_selection import KFold

def standardize_trial(EEG):
    # Calculate the mean and standard deviation along the last axis (data points axis)
    mean = np.mean(EEG, axis=2, keepdims=True)
    std = np.std(EEG, axis=2, keepdims=True)
    
    # Standardize the data
    standardized_data = (EEG - mean) / std
    
    return standardized_data

def avg_cov(EEG,if_std=True):
    if if_std==True:
        EEG=standardize_trial(EEG)
    EEG_center = EEG - np.mean(EEG, axis=2, keepdims=True)
    avg_cov_all=np.zeros((EEG.shape[1],EEG.shape[1]))
    trial_num=0
    for trial in EEG_center:
        EEG_cov=np.cov(trial)
        avg_cov_all+=EEG_cov/np.trace(EEG_cov)
        trial_num+=1
    return avg_cov_all/trial_num

def CSP(EEG,label,num_filter,if_std=True):
    EEG_c1=EEG[label==1]
    EEG_c2=EEG[label==-1]
    avg_cov_c1=avg_cov(EEG_c1,if_std)
    avg_cov_c2=avg_cov(EEG_c2,if_std)
    
    comb_cov=avg_cov_c1+avg_cov_c2
    evalue_c, evect_c = np.linalg.eig(comb_cov)
    diagw = np.diag(1/(evalue_c**0.5))
    diagw = diagw.real.round(4) #convert to real and round off
    P=np.dot(diagw, evect_c.T)
    S_c1=np.dot(np.dot(P,avg_cov_c1),P.T)
    S_c2=np.dot(np.dot(P,avg_cov_c2),P.T)
    evalue_c1, evect_c1 = np.linalg.eig(S_c1)
    evalue_c2, evect_c2 = np.linalg.eig(S_c2)
    
    eig_pairs_c1 = [(np.abs(evalue_c1[i]), evect_c1[:,i]) for i in range(len(evalue_c1))]
    eig_pairs_c1.sort(reverse=True)
    evect_sorted_c1=np.array([ele[1] for ele in eig_pairs_c1[:num_filter//2]])

    eig_pairs_c2 = [(np.abs(evalue_c2[i]), evect_c2[:,i]) for i in range(len(evalue_c2))]
    eig_pairs_c2.sort(reverse=True)
    evect_sorted_c2=np.array([ele[1] for ele in eig_pairs_c2[:num_filter//2]])
    
    evect_comb=np.concatenate((evect_sorted_c1,evect_sorted_c2))
    
    W=np.dot(P.T,evect_comb.T)
    
    return W    
    
def Proj_(EEG,SpatialFilter,iflog=True):
    WTX=np.dot(SpatialFilter.T,EEG)
    var_WTX=np.var(WTX,axis=1)
    # print(var_WTX.shape)
    if iflog==True:
        # print('yes')
        spatial_feature=np.log10(var_WTX/np.sum(var_WTX))
    else:
        spatial_feature=var_WTX
    # spatial_feature=var_WTX/np.sum(var_WTX)
    return spatial_feature

def Proj_EEG(EEG,SpatialFilter,if_std=True,if_log=True):
    Feature=[]
    if if_std==True:
        EEG=standardize_trial(EEG)
    for trial in EEG:
        Feature.append(Proj_(trial,SpatialFilter,if_log))
    Feature=np.array(Feature)
    return Feature

if __name__=='__main__':
        
    cost_range = [0.001,0.01,0.1, 1, 10, 100]
    # cost_range = [1]
    num_fold=4
    filters_num=4
    CV_result=[]
    for [lowband,highband] in [[8,13],[13,30],[8,30],[30,40]]:
        print('Fre band:'+str(lowband)+'-'+str(highband)+'-----------------------------')
        for partID in ['P1','P2','P3']:
            for stage in ['pre','post']:
                print(f'----{partID}-{stage}----')
                train_dict=scipy.io.loadmat(f'./filtered_data/filtered_EEG_{partID}_{stage}_training_{lowband}_{highband}.mat')
                test_dict=scipy.io.loadmat(f'./filtered_data/filtered_EEG_{partID}_{stage}_test_{lowband}_{highband}.mat')
                train_data=train_dict['EEGdata']
                train_label=train_dict['labellist'][:,0]
                test_data=test_dict['EEGdata']
                test_label=test_dict['labellist'][:,0]
                best_score = 0
                best_params = {'cost': None}
                for cost in cost_range:
                    print('cost:'+str(cost),end=' ')
                    score_in_one_CV=0
                    print('CV fold',end=' ')
                    kf = KFold(n_splits=num_fold, shuffle=True, random_state=42)
                    for fold_idx,[train_index, val_index] in enumerate(kf.split(train_data)):
                        print(fold_idx, end=' ')
                        # Split the data into training and validation sets
                        EEG_CV_train, EEG_CV_val = train_data[train_index], train_data[val_index]
                        label_CV_train, label_CV_val = train_label[train_index], train_label[val_index]
                        W=CSP(EEG_CV_train,label_CV_train,filters_num,True)
                        SF_CV_train=Proj_EEG(EEG_CV_train,W,True,True)
                        SF_CV_val=Proj_EEG(EEG_CV_val,W,True,True)
                        # train model
                        svm = LinearSVC(C=cost,max_iter=10000,fit_intercept = True,dual='auto')
                        svm.fit(SF_CV_train, label_CV_train)
                        # test model
                        val_score = svm.score(SF_CV_val, label_CV_val)          
                        score_in_one_CV+=val_score
                    score_in_one_CV/=num_fold
                    print(' score: '+str(score_in_one_CV))
                    if score_in_one_CV > best_score:
                        best_score = score_in_one_CV
                        best_params['cost'] = cost
                best_svm = LinearSVC(C=best_params['cost'],max_iter=10000,fit_intercept = True,dual='auto')
                W_best=CSP(train_data,train_label,filters_num,if_std=True)
                SF_train=Proj_EEG(train_data,W_best,if_std=True,if_log=False)
                SF_test=Proj_EEG(test_data,W_best,if_std=True,if_log=False)
                
                best_svm.fit(SF_train, train_label)
                label_pred_test = best_svm.predict(SF_test)
                
                acc_total=accuracy_score(label_pred_test,test_label)
                print(f'{partID}-{stage}-{lowband}-{highband}')
                print("Best parameters:", best_params)
                print("Best validation score:", best_score)
                print("Test set score with best parameters:", acc_total)
                CV_result.append([partID,stage,f'{lowband}-{highband} Hz',best_params['cost'],best_score,acc_total])
                
    CV_result_df=pd.DataFrame(CV_result,columns=['PID','stage','frequency band','best cost','best validation score','test score'])
    print(CV_result_df)
    CV_result_df.to_csv('./svm_result_.csv')
        
            