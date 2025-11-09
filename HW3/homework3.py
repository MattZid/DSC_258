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

