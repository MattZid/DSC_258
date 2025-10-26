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

# %%
def processing_hotcode(dataset):
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
    return X_norm,y

# %%

def processing_std(dataset): 
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
    return X_norm,y

# %%
def Q4(dataset):
    dataset4 = dataset[:]
    # random.seed(0)
    # random.shuffle(dataset4)
    cut = int(len(dataset4)/2)
    end = int(len(dataset4))
    dataset_train = dataset4[0:cut]
    dataset_test = dataset4[cut:end]

    print("Processing Training Data Hot Encode:")    
    print()
    X_norm,y = processing_hotcode(dataset_test)

    model = sk.linear_model.LinearRegression(fit_intercept=False)
    model.fit(X_norm, y)
    theta = model.coef_

    X_norm_test, y_test = processing_hotcode(dataset_test)
   
    y_pred = model.predict(X_norm_test)
    sse = sum(x**2 for x in (y_test-y_pred))
    mse1 = sse/len(y_test)
    print("test mse:")

    X_norm, y = processing_std(dataset_train)
    model = sk.linear_model.LinearRegression(fit_intercept=False)
    model.fit(X_norm, y)
    theta = model.coef_

    X_norm_test, y_test = processing_std(dataset_test)
    y_pred = model.predict(X_norm_test)
    sse = sum(x**2 for x in (y_test-y_pred))
    mse2 = sse/len(y_test)
    print("test mse2:")
    
    return mse1, mse2

# %%
def featureQ5(dataset):
    X = [[1, len(d['review/text'])] for d in dataset]
    y = [d['review/overall'] >= 4 for d in dataset]
    return X, y

# %% 
def featureQ7(dataset):
    X = [[1, len(d['review/text']),d['review/palate']] for d in dataset]
    y = [d['review/overall'] >= 4 for d in dataset]
    return X, y
# %%

def Q5(dataset, feature_fn):
    X, y = feature_fn(dataset)

    mod = sk.linear_model.LogisticRegression(class_weight='balanced')
    mod.fit(X,y)
    predictions = mod.predict(X)
    correct = predictions == y
    sum(correct) / len(correct)

    TP = sum([(p and l) for (p,l) in zip(predictions, y)])
    FP = sum([(p and not l) for (p,l) in zip(predictions, y)])
    TN = sum([(not p and not l) for (p,l) in zip(predictions, y)])
    FN = sum([(not p and l) for (p,l) in zip(predictions, y)])

    TPR = TP / (TP + FN)
    TNR = TN / (TN + FP)

    BER = 1 - 1/2 * (TPR + TNR)
    BER
    return TP, TN, FP, FN, BER

# %%
def Q5_Q6(dataset, feature_fn):
    X, y = feature_fn(dataset)

    mod = sk.linear_model.LogisticRegression(class_weight='balanced')
    mod.fit(X,y)
    predictions = mod.predict(X)
    score = mod.predict_proba(X)[:,1]

    return score, y   

# %%
def Q6(dataset):
    scores, y = Q5_Q6(dataset, featureQ5)
    ranked = sorted(zip(scores, y), key=lambda x: x[0], reverse=True)
    predictions,y = zip(*ranked)
    precisions = []

    k_values = [1, 100, 1000, 10000]
    for k in k_values:
        if k == 1:
            TPR_k = 1.0 if (predictions[0] and y[0]) else 0.0
        else:
            preds_k = predictions[0:k]
            y_k = y[0:k]
            TP_k = sum([(p and l) for (p,l) in zip(preds_k, y_k)])
            FP_k = sum([(p and not l) for (p,l) in zip(preds_k, y_k)])
            TPR_k = float(TP_k / k)
        precisions.append(TPR_k)
    return precisions