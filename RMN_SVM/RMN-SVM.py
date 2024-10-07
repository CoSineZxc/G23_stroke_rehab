import numpy as np
from sklearn.svm import SVC
from sklearn.model_selection import KFold
from sklearn.metrics import accuracy_score
import scipy.io
import pandas as pd
from pyriemann.estimation import Covariances
from pyriemann.tangentspace import TangentSpace

if __name__ == "__main__":
    cost_range = [0.001, 0.01, 0.1, 1, 10, 100]
    num_fold = 4
    CV_result = []
    
    for [lowband, highband] in [[8, 13], [13, 30], [8, 30], [30, 40]]:
        print(f'Frequency band: {lowband}-{highband} Hz -----------------------------')
        for partID in ['P1', 'P2', 'P3']:
            for stage in ['pre', 'post']:
                print(f'----{partID}-{stage}----')
                train_dict = scipy.io.loadmat(f'./filtered_data/filtered_EEG_{partID}_{stage}_training_{lowband}_{highband}.mat')
                test_dict = scipy.io.loadmat(f'./filtered_data/filtered_EEG_{partID}_{stage}_test_{lowband}_{highband}.mat')
                
                train_data = train_dict['EEGdata']  # shape: (n_trials, n_channels, n_samples)
                train_label = train_dict['labellist'][:, 0]
                test_data = test_dict['EEGdata']
                test_label = test_dict['labellist'][:, 0]
                
                # Compute covariance matrices for each trial
                cov_estimator = Covariances(estimator='oas')
                cov_train = cov_estimator.fit_transform(train_data)
                cov_test = cov_estimator.transform(test_data)
                
                # Map covariance matrices to Tangent Space
                ts = TangentSpace(metric='riemann')
                ts_train = ts.fit_transform(cov_train)
                ts_test = ts.transform(cov_test)
                
                best_score = 0
                best_params = {'cost': None}
                
                # Perform k-fold cross-validation
                for cost in cost_range:
                    print(f'cost: {cost}', end=' ')
                    score_in_one_CV = 0
                    print('CV fold', end=' ')
                    
                    kf = KFold(n_splits=num_fold, shuffle=True, random_state=42)
                    for fold_idx, (train_index, val_index) in enumerate(kf.split(ts_train)):
                        print(fold_idx, end=' ')
                        
                        # Split the data into training and validation sets
                        X_train, X_val = ts_train[train_index], ts_train[val_index]
                        y_train, y_val = train_label[train_index], train_label[val_index]
                        
                        # Train SVM on Tangent Space features
                        svm = SVC(C=cost, kernel='linear', max_iter=10000)
                        svm.fit(X_train, y_train)
                        
                        # Validate the model
                        val_score = svm.score(X_val, y_val)
                        score_in_one_CV += val_score
                    
                    score_in_one_CV /= num_fold
                    print(f' score: {score_in_one_CV}')
                    
                    # Update the best parameters if the current score is better
                    if score_in_one_CV > best_score:
                        best_score = score_in_one_CV
                        best_params['cost'] = cost
                
                # Train the best model on the full training data
                best_svm = SVC(C=best_params['cost'], kernel='linear', max_iter=10000)
                best_svm.fit(ts_train, train_label)
                
                # Test the model on the test set
                label_pred_test = best_svm.predict(ts_test)
                acc_total = accuracy_score(label_pred_test, test_label)
                
                print(f'{partID}-{stage}-{lowband}-{highband}')
                print('Best parameters:', best_params)
                print('Best validation score:', best_score)
                print('Test set score with best parameters:', acc_total)
                
                # Append results to the list
                CV_result.append([partID, stage, f'{lowband}-{highband} Hz', best_params['cost'], best_score, acc_total])
    
    # Convert results to a DataFrame and save to CSV
    CV_result_df = pd.DataFrame(CV_result, columns=['PID', 'stage', 'frequency band', 'best cost', 'best validation score', 'test score'])
    print(CV_result_df)
    CV_result_df.to_csv('./riemannian_svm_result.csv', index=False)