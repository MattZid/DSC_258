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
    set1 = s1 if isinstance(s1, set) else set(s1)
    set2 = s2 if isinstance(s2, set) else set(s2)
    if not set1 and not set2:
        return 0.0
    union = set1.union(set2)
    if not union:
        return 0.0
    intersection = set1.intersection(set2)
    return len(intersection) / len(union)

# %%
def mostSimilar(i, N, usersPerItem):
    target_users = usersPerItem.get(i)
    if target_users is None:
        return []
    if not target_users:
        return []
    similarities = []

    for j, users in usersPerItem.items():
        if j == i:
            continue
        sim = Jaccard(target_users, users)
        similarities.append((sim, j))

    similarities.sort(key=lambda x: (-x[0], x[1]))
    return similarities[:N]

# %%


# %%
# ### Question 6

# %%
def MSE(y, ypred):
    if not y:
        return 0.0
    if len(y) != len(ypred):
        raise ValueError("Predictions and labels must have the same length")
    se = [(float(yi) - float(yp)) ** 2 for yi, yp in zip(y, ypred)]
    return sum(se) / len(se)

# %%
def getMeanRating(dataTrain):
    if not dataTrain:
        return 0.0
    total = sum(d['star_rating'] for d in dataTrain)
    return total / len(dataTrain)

def getUserAverages(itemsPerUser, ratingDict):
    # Compute the training-set average rating for each user
    userAverages = {}
    for user, items in itemsPerUser.items():
        if not items:
            continue
        ratings = [ratingDict[(user, item)] for item in items if (user, item) in ratingDict]
        if ratings:
            userAverages[user] = sum(ratings) / len(ratings)
    return userAverages

def getItemAverages(usersPerItem, ratingDict):
    itemAverages = {}
    for item, users in usersPerItem.items():
        if not users:
            continue
        ratings = [ratingDict[(user, item)] for user in users if (user, item) in ratingDict]
        if ratings:
            itemAverages[item] = sum(ratings) / len(ratings)
    return itemAverages

# %%


# %%
def predictRating(user,item,ratingMean,reviewsPerUser,usersPerItem,itemsPerUser,userAverages,itemAverages):
    # Solution for Q6, should return a rating
    base_rating = itemAverages.get(item, ratingMean)
    user_reviews = reviewsPerUser.get(user, [])

    target_users = usersPerItem.get(item, set())
    if not user_reviews or not target_users:
        return base_rating

    numerator = 0.0
    denominator = 0.0

    for review in user_reviews:
        neighbor_item = review['product_id']
        if neighbor_item == item:
            continue
        neighbor_users = usersPerItem.get(neighbor_item, set())
        if not neighbor_users:
            continue
        sim = Jaccard(target_users, neighbor_users)
        if sim <= 0:
            continue
        neighbor_rating = review['star_rating']
        neighbor_avg = itemAverages.get(neighbor_item, ratingMean)
        numerator += (neighbor_rating - neighbor_avg) * sim
        denominator += sim

    if denominator == 0:
        return base_rating

    prediction = base_rating + numerator / denominator
    return min(max(prediction, 1.0), 5.0)

# %%


# %%
### Question 7

# %%
def predictRatingQ7(user,item,ratingMean,reviewsPerUser,usersPerItem,itemsPerUser,userAverages,itemAverages):
    target_users = usersPerItem.get(item, set())
    user_reviews = reviewsPerUser.get(user, [])

    item_avg = itemAverages.get(item)
    user_avg = userAverages.get(user)

    baseline_components = [ratingMean]
    if item_avg is not None:
        baseline_components.append(item_avg)
    if user_avg is not None:
        baseline_components.append(user_avg)
    baseline = sum(baseline_components) / len(baseline_components)

    if not target_users or not user_reviews:
        return min(max(baseline, 1.0), 5.0)

    numerator = 0.0
    denom = 0.0

    for review in user_reviews:
        neighbor_item = review['product_id']
        if neighbor_item == item:
            continue
        neighbor_users = usersPerItem.get(neighbor_item, set())
        if not neighbor_users:
            continue
        overlap = len(target_users & neighbor_users)
        if overlap == 0:
            continue
        sim = Jaccard(target_users, neighbor_users)
        if sim <= 0:
            continue
        significance = overlap / (overlap + 2.0)
        weight = sim * significance
        neighbor_rating = review['star_rating']
        neighbor_avg = itemAverages.get(neighbor_item, ratingMean)
        numerator += (neighbor_rating - neighbor_avg) * weight
        denom += weight

    prediction = baseline
    if denom > 0:
        adjustment = numerator / denom
        shrink = denom / (denom + 1.0)
        prediction = baseline + shrink * adjustment

    if user_avg is not None:
        prediction = 0.8 * prediction + 0.2 * user_avg

    return min(max(prediction, 1.0), 5.0)

# %%
