# %%
from collections import defaultdict
from sklearn import linear_model
import numpy
import math

# %%


# %%
def feat(d, catID, maxLength, includeCat = True, includeReview = True, includeLength = True):
    feat = []

    if includeCat:
        # 
        num_cats = len(catID)
        cat_features = [0.0] * max(num_cats - 1, 0)
        if num_cats > 1:
            category = d.get('beer/style')
            if category in catID:
                idx = catID[category]
                if idx > 0:
                    cat_features[idx - 1] = 1.0
        feat.extend(cat_features)

    if includeReview:
        review_keys = [
            'review/appearance',
            'review/aroma',
            'review/palate',
            'review/taste',
            'review/overall'
        ]
        feat.extend(float(d.get(key, 0.0)) for key in review_keys)

    if includeLength:
        text = d.get('review/text', '')
        length = len(text) if isinstance(text, str) else 0
        feat.append((length / maxLength) if maxLength else 0.0)

    return feat + [1]

# %%
def pipeline(reg, catID, dataTrain, dataValid, dataTest, includeCat=True, includeReview=True, includeLength=True):
    mod = linear_model.LogisticRegression(C=reg, class_weight='balanced')

    maxLength = max([len(d['review/text']) for d in dataTrain])
    
    Xtrain = [feat(d, catID, maxLength, includeCat, includeReview, includeLength) for d in dataTrain]
    Xvalid = [feat(d, catID, maxLength, includeCat, includeReview, includeLength) for d in dataValid]
    Xtest = [feat(d, catID, maxLength, includeCat, includeReview, includeLength) for d in dataTest]
    
    yTrain = numpy.array([d['beer/ABV'] > 7 for d in dataTrain], dtype=int)
    yValid = numpy.array([d['beer/ABV'] > 7 for d in dataValid], dtype=int)
    yTest = numpy.array([d['beer/ABV'] > 7 for d in dataTest], dtype=int)
    

    mod.fit(Xtrain, yTrain)

    def balanced_error_rate(y_true, y_pred):
        y_true = numpy.asarray(y_true, dtype=int)
        y_pred = numpy.asarray(y_pred, dtype=int)
        positives = (y_true == 1)
        negatives = (y_true == 0)
        pos_count = positives.sum()
        neg_count = negatives.sum()
        tpr = ((y_pred[positives] == 1).sum() / pos_count) if pos_count else 0.0
        tnr = ((y_pred[negatives] == 0).sum() / neg_count) if neg_count else 0.0
        return 1 - 0.5 * (tpr + tnr)

    yValidPred = mod.predict(Xvalid)
    vBER = balanced_error_rate(yValid, yValidPred)

    yTestPred = mod.predict(Xtest)
    tBER = balanced_error_rate(yTest, yTestPred)

    return mod, vBER, tBER

# %%


#%%
## Question 1

# %%
def Q1(catID, dataTrain, dataValid, dataTest):
    # No need to modify this if you've implemented the functions above
    mod, validBER, testBER = pipeline(10, catID, dataTrain, dataValid, dataTest, True, False, False)
    return mod, validBER, testBER

# %%


# %%
# ### Question 2

# %%
def Q2(catID, dataTrain, dataValid, dataTest):
    mod, validBER, testBER = pipeline(10, catID, dataTrain, dataValid, dataTest, True, True, True)
    return mod, validBER, testBER

# %%


# %%
# ## Question 3

# %%
def Q3(catID, dataTrain, dataValid, dataTest):
    bestModel = None
    bestValidBER = None
    bestTestBER = None

    for C in [0.001, 0.01, 0.1, 1, 10]:
        mod, validBER, testBER = pipeline(C, catID, dataTrain, dataValid, dataTest, True, True, True)
        if bestValidBER is None or validBER < bestValidBER:
            bestModel = mod
            bestValidBER = validBER
            bestTestBER = testBER

    return bestModel, bestValidBER, bestTestBER

# %%


# %%
### Question 4

# %%
def Q4(C, catID, dataTrain, dataValid, dataTest):
    mod, validBER, testBER_noCat = pipeline(C, catID, dataTrain, dataValid, dataTest, False, True, True)
    mod, validBER, testBER_noReview = pipeline(C, catID, dataTrain, dataValid, dataTest, True, False, True)
    mod, validBER, testBER_noLength = pipeline(C, catID, dataTrain, dataValid, dataTest, True, True, False)
    return testBER_noCat, testBER_noReview, testBER_noLength

# %%


# %%
### Question 5

# %%
def Jaccard(s1, s2):
    # Implement
    # Placeholder return to keep earlier questions testable
    return 0.0

# %%
def mostSimilar(i, N, usersPerItem):
    # Implement...

    # Should be a list of (similarity, itemID) pairs
    similarities = []
    return similarities[:N]

# %%


# %%
# ### Question 6

# %%
def MSE(y, ypred):
    # Implement...
    # Placeholder
    return 0.0

# %%
def getMeanRating(dataTrain):
    # Implement...
    return 0.0

def getUserAverages(itemsPerUser, ratingDict):
    # Implement (should return a dictionary mapping users to their averages)
    userAverages = {}
    return userAverages

def getItemAverages(usersPerItem, ratingDict):
    # Implement...
    itemAverages = {}
    return itemAverages

# %%


# %%
def predictRating(user,item,ratingMean,reviewsPerUser,usersPerItem,itemsPerUser,userAverages,itemAverages):
    # Solution for Q6, should return a rating
    return ratingMean

# %%


# %%
### Question 7

# %%
def predictRatingQ7(user,item,ratingMean,reviewsPerUser,usersPerItem,itemsPerUser,userAverages,itemAverages):
    # Your solution here
    return ratingMean

# %%
