from collections import defaultdict
import random
import string
import os
import math

BETTER_WORDS = None
BETTER_WORDID = None
BETTER_IDF = None
BETTER_NW = 800

## comment

def getGlobalAverage(trainRatings):
    # data = list(readCSV(trainRatings))
    ratings = []
    for entry in trainRatings:
        ratings.append(entry)
    mean_rating = sum(ratings)/ len(ratings)
    return mean_rating

##

def trivialValidMSE(ratingsValid, globalAverage):
    se = 0.0
    for entry in ratingsValid:
        se += (entry[2] - globalAverage) ** 2
    mse = se / len(ratingsValid)
    return mse



##
def alphaUpdate(ratingsTrain, alpha, betaU, betaI, lamb):
    total = 0.0
    for u,b,r in ratingsTrain:
        bu = betaU.get(u, 0.0)
        bi = betaI.get(b, 0.0)
        total += r - (bu + bi)
    return total / len(ratingsTrain)

##


def betaUUpdate(ratingsPerUser, alpha, betaU, betaI, lamb):
    newBetaU = {}
    for u, items in ratingsPerUser.items():
        denom = lamb + len(items)
        if denom == 0:
            newBetaU[u] = 0.0
            continue
        total = 0.0
        for b,r in items:
            total += r - (alpha + betaI.get(b, 0.0))
        newBetaU[u] = total / denom
    return newBetaU

##



def betaIUpdate(ratingsPerItem, alpha, betaU, betaI, lamb):
    newBetaI = {}
    for b, users in ratingsPerItem.items():
        denom = lamb + len(users)
        if denom == 0:
            newBetaI[b] = 0.0
            continue
        total = 0.0
        for u,r in users:
            total += r - (alpha + betaU.get(u, 0.0))
        newBetaI[b] = total / denom
    return newBetaI

##

def msePlusReg(ratingsTrain, alpha, betaU, betaI, lamb):
    se = 0.0
    for u,b,r in ratingsTrain:
        bu = betaU.get(u, 0.0)
        bi = betaI.get(b, 0.0)
        err = r - (alpha + bu + bi)
        se += err * err
    mse = se / len(ratingsTrain)

    regularizer = 0.0
    for val in betaU.values():
        regularizer += val * val
    for val in betaI.values():
        regularizer += val * val

    return mse, mse + lamb * regularizer

##

def validMSE(ratingsValid, alpha, betaU, betaI):
    se = 0.0
    for u,b,r in ratingsValid:
        bu = betaU.get(u, 0.0)
        bi = betaI.get(b, 0.0)
        pred = alpha + bu + bi
        se += (r - pred) ** 2
    return se / len(ratingsValid)

##

def goodModel(ratingsTrain, ratingsPerUser, ratingsPerItem, alpha, betaU, betaI):
    lamb = 1.0
    bestAlpha, bestBetaU, bestBetaI = alpha, betaU.copy(), betaI.copy()
    bestMSE = float('inf')
    for _ in range(15):
        alpha = alphaUpdate(ratingsTrain, alpha, betaU, betaI, lamb)
        betaU = betaUUpdate(ratingsPerUser, alpha, betaU, betaI, lamb)
        betaI = betaIUpdate(ratingsPerItem, alpha, betaU, betaI, lamb)
        mse, _ = msePlusReg(ratingsTrain, alpha, betaU, betaI, lamb)
        if mse + 1e-6 < bestMSE:
            bestMSE = mse
            bestAlpha, bestBetaU, bestBetaI = alpha, betaU.copy(), betaI.copy()
        else:
            break
    return bestAlpha, bestBetaU, bestBetaI
##

def generateValidation(allRatings, ratingsValid):
    # Using ratingsValid, generate two sets:
    # readValid: set of (u,b) pairs in the validation set
    # notRead: set of (u,b') pairs, containing one negative (not read) for each row (u) in readValid  
    # Both should have the same size as ratingsValid
    userHistory = defaultdict(set)
    allItems = set()
    for u,b,_ in allRatings:
        userHistory[u].add(b)
        allItems.add(b)
    itemsList = list(allItems)
    rng = random.Random(0)
    readValid = set()
    notRead = set()
    for u,b,_ in ratingsValid:
        readValid.add((u,b))
        seen = userHistory[u]
        candidate = None
        attempts = 0
        while candidate is None:
            c = rng.choice(itemsList)
            if c not in seen and (u,c) not in readValid and (u,c) not in notRead:
                candidate = c
                break
            attempts += 1
            if attempts > 1000:
                for alt in itemsList:
                    if alt not in seen and (u,alt) not in readValid and (u,alt) not in notRead:
                        candidate = alt
                        break
                if candidate is None:
                    candidate = c
                    break
        notRead.add((u, candidate))
    return readValid, notRead

##

def baseLineStrategy(mostPopular, totalRead):
    return1 = set()

    # Compute the set of items for which we should return "True"
    # This is the same strategy implemented in the baseline code for Assignment 1
    count = 0
    threshold = totalRead / 2.0
    for ic, i in mostPopular:
        return1.add(i)
        count += ic
        if count >= threshold:
            break

    return return1

##

def improvedStrategy(mostPopular, totalRead):
    return1 = set()

    # Same as above function, just find an item set that'll have higher accuracy
    target = totalRead * 0.05
    minReads = 20
    count = 0
    for ic, i in mostPopular:
        if ic < minReads and count > target:
            break
        return1.add(i)
        count += ic
        if count >= target and ic < minReads:
            break

    return return1

##


def evaluateStrategy(return1, readValid, notRead):

    # Compute the accuracy of a strategy which just returns "true" for a set of items (those in return1)
    # readValid: instances with positive label
    # notRead: instances with negative label
    correct = 0
    for _, b in readValid:
        if b in return1:
            correct += 1
    for _, b in notRead:
        if b not in return1:
            correct += 1
    total = len(readValid) + len(notRead)
    if total == 0:
        return 0.0
    return correct / total

##



def jaccardThresh(u,b,ratingsPerItem,ratingsPerUser):
    
    # Compute the similarity of the query item (b) compared to the most similar item in the user's history
    # Return true if the similarity is high or the item is popular
    usersB = set([user for user,_ in ratingsPerItem[b]])
    maxSim = 0.0
    for otherB,_ in ratingsPerUser.get(u, []):
        usersOther = set([user for user,_ in ratingsPerItem[otherB]])
        sim = Jaccard(usersB, usersOther)
        if sim > maxSim:
            maxSim = sim
    if maxSim > 0.013 or len(ratingsPerItem[b]) > 40: # Keep these thresholds as-is
        return 1
    return 0

##


def featureCat(datum, words, wordId, wordSet):
    feat = [0]*len(words)

    # Compute features counting instance of each word in "words"
    # after converting to lower case and removing punctuation
    punctuation = set(string.punctuation)
    review = ''.join([c for c in datum['review_text'].lower() if c not in punctuation])
    for w in review.split():
        if w in wordSet:
            feat[wordId[w]] += 1

    feat.append(1) # offset (put at the end)
    return feat

##


def betterFeatures(data):
    # Produce better features than those from the above question
    # Return matrix (each row is the feature vector for one entry in the dataset)
    global BETTER_WORDS, BETTER_WORDID, BETTER_IDF
    punctuation = set(string.punctuation)
    need_vocab = BETTER_WORDS is None
    if need_vocab:
        wordCount = defaultdict(int)
        docFreq = defaultdict(int)
    tokenized = []
    for d in data:
        review = ''.join([c for c in d['review_text'].lower() if c not in punctuation])
        tokens = review.split()
        tokenized.append(tokens)
        if need_vocab:
            for w in tokens:
                wordCount[w] += 1
            for w in set(tokens):
                docFreq[w] += 1

    if need_vocab:
        common = sorted(wordCount.items(), key=lambda kv: kv[1], reverse=True)[:BETTER_NW]
        BETTER_WORDS = [w for w,_ in common]
        BETTER_WORDID = {w:i for i,w in enumerate(BETTER_WORDS)}
        Ndocs = len(tokenized)
        BETTER_IDF = {}
        for w in BETTER_WORDS:
            df = docFreq.get(w, 1)
            BETTER_IDF[w] = math.log((1 + Ndocs)/(1 + df)) + 1.0

    words = BETTER_WORDS
    wordId = BETTER_WORDID
    idf = BETTER_IDF
    X = []
    for datum, tokens in zip(data, tokenized):
        feat = [0.0]*len(words)
        length_raw = len(tokens)
        norm = 1.0 / length_raw if length_raw else 0.0
        for w in tokens:
            idx = wordId.get(w)
            if idx is not None:
                feat[idx] += 1
        for i in range(len(feat)):
            if feat[i] > 0:
                feat[i] = (feat[i] * norm) * idf[words[i]]
        length_feat = math.log1p(length_raw)
        unique = math.log1p(len(set(tokens)))
        avglen = math.log1p((sum(len(w) for w in tokens) / length_raw) if length_raw else 0.0)
        rating = datum.get('rating', 0)
        votes = math.log1p(max(0, datum.get('n_votes', 0)))
        feat.extend([length_feat, unique, avglen, rating, votes, 1])
        X.append(feat)

    return X

##


def Jaccard(s1, s2):
    numer = len(s1.intersection(s2))
    denom = len(s1.union(s2))
    if denom > 0:
        return numer/denom
    return 0

##


def writePredictionsRead(ratingsPerItem, ratingsPerUser):
    predictions = open("predictions_Read.csv", 'w')
    for l in open("pairs_Read.csv"):
        if l.startswith("userID"):
            predictions.write(l)
            continue
        u,b = l.strip().split(',')
        pred = jaccardThresh(u,b,ratingsPerItem,ratingsPerUser)
        _ = predictions.write(u + ',' + b + ',' + str(pred) + '\n')

    predictions.close()

##

def writePredictionsCategory(pred_test):
    predictions = open("predictions_Category.csv", 'w')
    pos = 0

    pairs_path = "pairs_Category.csv"
    if not os.path.exists(pairs_path):
        pairs_path = os.path.join("..", "assignment_1", "pairs_Category.csv")

    for l in open(pairs_path):
        if l.startswith("userID"):
            predictions.write(l)
            continue
        u,b = l.strip().split(',')
        _ = predictions.write(u + ',' + b + ',' + str(pred_test[pos]) + '\n')
        pos += 1

    predictions.close()

##

def runOnTest(data_test, mod):
    Xtest = [featureCat(d) for d in data_test]
    pred_test = mod.predict(Xtest)
