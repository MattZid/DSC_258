# %%
import numpy as np
import sklearn as sk
from sklearn import linear_model
import matplotlib.pyplot as plt

import gzip
import json
import dateutil.parser
import random
import datetime

# %%
def Q1(dataset):
    ratings = [d['rating'] for d in dataset]
    lengths = [len(d['review_text']) for d in dataset]
    print("first record of lengths: "+  str(lengths[0]))
    print("longest length review: " +  str(max(lengths)))
    
    lengths_norm = [word/max(lengths) for word in lengths]
    X_norm = np.array([[1,l] for l in lengths_norm]) # Note the inclusion of the constant term
    y = np.array(ratings).T
    model = sk.linear_model.LinearRegression(fit_intercept=False)
    model.fit(X_norm, y)
    
    theta = model.coef_
    
    y_pred = model.predict(X_norm)
    sse = sum(x**2 for x in (y-y_pred))
    mse = sse/len(y)
    return theta, mse
    

# %%
def Q2(dataset):

    day_of_week = []
    month_list = []

    ratings = [d['rating'] for d in dataset]
    lengths = [len(d['review_text']) for d in dataset]
    print("first record of lengths: "+  str(lengths[0]))
    print("longest length review: " +  str(max(lengths)))
    
    lengths_norm = [word/max(lengths) for word in lengths]
    X_norm = np.array([[1,l] for l in lengths_norm]) # Note the inclusion of the constant term
    y = np.array(ratings).T


    for d in dataset:
        dow = str(d['date_added']).split(' ')[0]    
        t = dateutil.parser.parse(d['date_added'])
    
        month_list.append(t.month)
        day_of_week.append(dow)
        
    month_arr = np.array(month_list).reshape(-1,1)
    dow_arr = np.array(day_of_week).reshape(-1,1)

    unique_days = np.unique(dow_arr)
    baseline = unique_days[0]
    unique_days_reduced = unique_days[1:,]
    print(baseline)
    print(unique_days_reduced)
    print("Number of Unique Days: " + str(len(unique_days)))
    print("Dimensions of Hot Encoded DOW feature: " + str(len(unique_days_reduced)))
    print(dow_arr.shape)
    dow_arr = np.reshape(dow_arr,(-1))

    dow_hot_encode = (dow_arr[:,None] == unique_days_reduced).astype(int)
    X_norm = np.hstack((X_norm,dow_hot_encode))


    unique_mo = np.unique(month_arr)
    baseline_m = unique_mo[0]
    unique_mo_reduced = unique_mo[1:,]
    print(baseline_m)
    print(unique_mo_reduced)
    print("Number of Unique Months: " + str(len(unique_mo)))
    print("Dimensions of Hot Encoded DOW feature: " + str(len(unique_mo_reduced)))
    print(month_arr.shape)
    month_arr = np.reshape(month_arr,(-1))

    mo_hot_encode = (month_arr[:,None] == unique_mo_reduced).astype(int)
    X_norm = np.hstack((X_norm,mo_hot_encode))

    model = sk.linear_model.LinearRegression(fit_intercept=False)
    model.fit(X_norm, y)
    theta = model.coef_

    y_pred = model.predict(X_norm)
    sse = sum(x**2 for x in (y-y_pred))
    mse = sse/len(y)
    return X_norm,y,mse

# %%
def Q3(dataset):
    day_of_week = []
    weekday_num = []
    month_list = []
    month_num = []

    ratings = [d['rating'] for d in dataset]
    lengths = [len(d['review_text']) for d in dataset]

    for d in dataset:
        dow = str(d['date_added']).split(' ')[0]    
        t = dateutil.parser.parse(d['date_added'])

        month_list.append(t.month)
        day_of_week.append(dow)
        weekday_num.append(t.weekday())
        month_num.append(t.month)

    print("first record of lengths: "+  str(lengths[0]))
    print("longest length review: " +  str(max(lengths)))
    weekday_num = np.array(weekday_num).reshape(-1,1)
    month_num = np.array(month_num).reshape(-1,1)

    X_norm = np.array([[1,l] for l in lengths]) # Note the inclusion of the constant term
    X_norm = np.hstack((X_norm,weekday_num))
    X_norm = np.hstack((X_norm,month_num))
    y = np.array(ratings).T

    model = sk.linear_model.LinearRegression(fit_intercept=False)
    model.fit(X_norm, y)
    theta = model.coef_

    y_pred = model.predict(X_norm)
    sse = sum(x**2 for x in (y-y_pred))
    mse = sse/len(y)
    
    return X_norm,y,mse