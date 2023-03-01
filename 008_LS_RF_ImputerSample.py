# -*- coding: utf-8 -*-
"""
Created on Tue Aug  9 11:21:27 2022
Objective: Run 150 loops, determine MSE

@author: lsanto6
"""
#%% === Import libraries ===
import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler
from sklearn.impute import SimpleImputer 
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.ensemble import RandomForestRegressor
from sklearn import metrics, linear_model
import tkinter as tk
import os.path, time
import winsound
#from pprint import pprint
#%% === Filelog ===
root=tk.Tk()
root.withdraw()
#%% === Loading Data ===
main_dir = 'G:/My Drive/Leticia_Santos/Soybeans'
Ex1=pd.read_csv(main_dir+'/Data/Processed/TG_CleanData_12_05_2018.csv')
#%% === Constants ===
n=150 #Number of loops in the model
c1=20 #Start sample size testing
c2=125  #Finish sample size testing
c3=5 #Range to increase each time
#%% === Wragling
cols = list(Ex1) 
cols.insert(0, cols.pop(cols.index('YLD')))
Ex1=Ex1.loc[:,cols]
df1_desc=Ex1.describe()
#%% === Get Dummies
Ex1feat = pd.get_dummies(Ex1)
Ex1num = Ex1feat.apply(np.float64)
#%% === Imputing Data on the NA data ===
Ex1num.isnull().sum() #Code on how to check missing values on each columns
Ex1num.isnull().sum().sum() #How many missing values on the table
fill_NaN = SimpleImputer(missing_values=np.nan, strategy='median')
Ex2 = pd.DataFrame(fill_NaN.fit_transform(Ex1num))
Ex2.isnull().sum().sum() #Checking if the code worked to apply median on missing values
#%% === Data normalization ===
sc = StandardScaler()
Ex2 = pd.DataFrame(sc.fit_transform(Ex2))
#%% Split The Input and output values ============================== 
X=Ex2.iloc[:,1:len(Ex2)].values
y=Ex2.iloc[:,0].values.flatten() #Skitlearn condition
#%% === Create empty lists to store results ===
results_1 = pd.DataFrame(columns=['rf_RMSE', "rf_MSE", 'lr_RMSE', 'lr_MSE', 'Time', "Size"])

#%% === Model and looping sample size ===
for i in range(n):
    for c in range(c1,c2,c3):
        Ex3=Ex2.sample(n=c, replace=False)
        X=Ex3.iloc[:,1:len(Ex3)].values
        y=Ex3.iloc[:,0].values.flatten()
        Size=(str(c))

    
        start=time.time()
        X_train,X_test,y_train,y_test=train_test_split(X,y,test_size=0.2,shuffle=True)
            
        #Random Forest
        regressor=RandomForestRegressor()
        model_hyp={'max_depth' : [24],
                   'n_estimators': [100],
                   "max_features": ['sqrt'],
                   "bootstrap": [True]}
        rf_opt=GridSearchCV(regressor,model_hyp)
        rf_opt.fit(X_train,y_train)
        y_pred_rf=rf_opt.predict(X_test)
        RMSE_rf=np.sqrt(metrics.mean_squared_error(y_test,y_pred_rf))
        MSE_rf=metrics.mean_squared_error(y_test,y_pred_rf)
        hyp = rf_opt.best_params_ 
    
            
        #Linear Regression
        regr=linear_model.LinearRegression()
        regr.fit(X_train,y_train)
        y_pred_lr=regr.predict(X_test)
        RMSE_lr=np.sqrt(metrics.mean_squared_error(y_test, y_pred_lr))
        MSE_lr=metrics.mean_squared_error(y_test, y_pred_lr)

    
        
        #Append other results
        end = time.time()
        tm = end-start
        results_1 = results_1.append({'rf_RMSE': RMSE_rf, "rf_MSE": MSE_rf,
                                      'lr_RMSE': RMSE_lr, "lr_MSE": MSE_lr,
                                      "hyperparams": hyp, 'Size':Size, 'Time': tm}, ignore_index=True)
        
winsound.Beep(440,500)

#%%===Combine Results in dataset


results_MSE = results_1[['rf_MSE', 'lr_MSE','Size']]
results_TimeHyp = results_1[['hyperparams', 'Time']]

results_longMSE = pd.melt(results_MSE, id_vars='Size', var_name="Method", value_name="MSE")
results_longMSE['Method'] = results_longMSE['Method'].replace(
    {'lr_MSE': "Linear Regression", 'rf_MSE': "Random Forest"})

results_longMSE.to_csv(main_dir+'/Results/Results_paper/RawResultsMSE_008ImpSam.csv')

results_1.to_csv(main_dir+'/Results/Results_paper/TIMERawResultsMSE_008ImpSam.csv')



#Run code in R - Error Analysis

