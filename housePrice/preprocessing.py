import pandas as pd
import sklearn.tree as sktree
import sklearn.preprocessing as skpre
import sklearn.ensemble as skensemble
import sklearn.neural_network as skneural
import sklearn.svm as sksvm
import sklearn.linear_model as sklinear
import sklearn.metrics as skmetrics
import sklearn.model_selection as sksel

import numpy as np


def dataSetSpecificFeaturePreProcessing(inputFeatures):
    inputFeatures["LotFrontage"] = inputFeatures[["LotFrontage"]].fillna(0)
    inputFeatures["MasVnrArea"] = inputFeatures[["MasVnrArea"]].fillna(0)
    inputFeatures["GarageYrBlt"] = inputFeatures[["GarageYrBlt"]].apply(lambda x: str(x))

    return inputFeatures


def dataSetSpecificResultPreProcessing(inputFeatures, results):
    return results - inputFeatures["MiscVal"]


def predictionPostProcessing(inputFeatures, predictions):
    return predictions + inputFeatures["MiscVal"]


def preProcessFeatures(inputFeatures):
    knownNumericalFeatures = ["LotFrontage", "LotArea", "MasVnrArea", "BsmtFinSF1", "BsmtFinSF2", "BsmtUnfSF", "TotalBsmtSF",
                              "1stFlrSF", "2ndFlrSF", "LowQualFinSF", "GrLivArea", "GarageArea", "WoodDeckSF", "OpenPorchSF",
                              "EnclosedPorch", "3SsnPorch", "ScreenPorch", "PoolArea", "MiscVal"]

    inputFeatures = dataSetSpecificFeaturePreProcessing(inputFeatures)
    for colName, datatype in zip(inputFeatures.dtypes.index, inputFeatures.dtypes):
        if colName not in knownNumericalFeatures:
            # encode all the strings into enums
            cleanedInputFeatures = inputFeatures[colName].apply(lambda x: str(x))
            encoder = skpre.LabelEncoder()
            encoder.fit(cleanedInputFeatures)
            inputFeatures[colName] = encoder.transform(cleanedInputFeatures)
        else:
            # scale the features correctly
            inputFeatures[colName] = skpre.scale(inputFeatures[colName])

    return inputFeatures


def trainAndTest(classifierName, classifier, trainingFeatures, trainingResults, testFeatures, testResults):
    processedTrainingResults = dataSetSpecificResultPreProcessing(trainingFeatures, trainingResults)

    processedTrainingFeatures = trainingFeatures.drop(["MiscVal"], axis=1)
    processedTestFeatures = testFeatures.drop(["MiscVal"], axis=1)

    classifier.fit(processedTrainingFeatures, processedTrainingResults)
    predictions = classifier.predict(processedTestFeatures)
    postProcessedPredictions = predictionPostProcessing(testFeatures, predictions)
    numTestRows = testResults.shape[0]

    sqError = 0
    for prediction, expected in zip(postProcessedPredictions, testResults):
        # print(str(prediction) + " -> " + str(expected) + " error : " + str(abs(prediction - expected)))
        sqError += (prediction - expected) * (prediction - expected)
    rmse = np.sqrt(sqError/numTestRows)

    scoreResult = classifier.score(processedTestFeatures, testResults)
    mae = skmetrics.mean_absolute_error(testResults, postProcessedPredictions)
    print(classifierName + "\trmse =\t" + str(rmse) + "\tmae =\t" + str(mae) + "\tscoreResult = \t" + str(scoreResult))



# ________________________________________________________________________

housePrices = pd.read_csv("train.csv")

splitRowId = int(housePrices.shape[0] * 4/5)

trainingHousePrices, testHousePrices = sksel.train_test_split(housePrices, test_size=(1/5))

# trainingHousePrices = housePrices.iloc[: splitRowId]
trainingHousePricesFeatures = preProcessFeatures(trainingHousePrices.iloc[:, 1:-1])
trainingHousePricesResults = trainingHousePrices.iloc[:, -1]

# testHousePrices = housePrices.iloc[splitRowId:]
testHousePricesFeatures = preProcessFeatures(testHousePrices.iloc[:, 1:-1])
testHousePricesResults = testHousePrices.iloc[:, -1]

# testingHousePricesFeatures = pd.read_csv("test.csv")

rng = np.random.RandomState(1)

linearClassifier = sklinear.LinearRegression()
trainAndTest("linearClassifier", linearClassifier, trainingHousePricesFeatures, trainingHousePricesResults,
             testHousePricesFeatures, testHousePricesResults)

plainTreeClassifier = sktree.DecisionTreeRegressor(criterion='mse', min_samples_leaf=1,
                                                   min_samples_split=2, min_weight_fraction_leaf=0.0,
                                                   presort=False, splitter='best')
trainAndTest("plainTree\t\t", plainTreeClassifier, trainingHousePricesFeatures, trainingHousePricesResults,
             testHousePricesFeatures, testHousePricesResults)

forest300 = skensemble.RandomForestRegressor(n_estimators=300, random_state=rng)
trainAndTest("decisionForest300", forest300, trainingHousePricesFeatures, trainingHousePricesResults,
             testHousePricesFeatures, testHousePricesResults)

forest50 = skensemble.RandomForestRegressor(n_estimators=50, random_state=rng)
trainAndTest("decisionForest50", forest50, trainingHousePricesFeatures, trainingHousePricesResults,
             testHousePricesFeatures, testHousePricesResults)

boostedTreeClassifier300 = skensemble.AdaBoostRegressor(plainTreeClassifier, n_estimators=300, random_state=rng)
trainAndTest("adaBoostedTree300", boostedTreeClassifier300, trainingHousePricesFeatures, trainingHousePricesResults,
             testHousePricesFeatures, testHousePricesResults)

boostedTreeClassifier50 = skensemble.AdaBoostRegressor(plainTreeClassifier, n_estimators=50, random_state=rng)
trainAndTest("adaBoostedTree50", boostedTreeClassifier50, trainingHousePricesFeatures, trainingHousePricesResults,
             testHousePricesFeatures, testHousePricesResults)

neuralClassifier1500x200 = skneural.MLPRegressor(solver='lbfgs', hidden_layer_sizes=(150, 200), random_state=rng)
trainAndTest("neural1500x200\t", neuralClassifier1500x200, trainingHousePricesFeatures, trainingHousePricesResults,
             testHousePricesFeatures, testHousePricesResults)

svmClassifier = sksvm.SVR()
trainAndTest("svmClassifier\t", svmClassifier, trainingHousePricesFeatures, trainingHousePricesResults,
             testHousePricesFeatures, testHousePricesResults)

# baseBoostedTree = skensemble.AdaBoostRegressor(plainTreeClassifier, random_state=rng)
# boostedTreeGridSearch = sksel.GridSearchCV(baseBoostedTree, {'n_estimators': [25, 50, 100, 300, 500]})
# trainAndTest("boostedTreeGridSearch\t", boostedTreeGridSearch, trainingHousePricesFeatures, trainingHousePricesResults,
#             testHousePricesFeatures, testHousePricesResults)

# summarize the results of the grid search
# print(boostedTreeGridSearch.best_estimator_)
# print(boostedTreeGridSearch.best_params_)
