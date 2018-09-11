import pandas as pd
import numpy as np
import sklearn.preprocessing as skpre
import sklearn.model_selection as sksel
import sklearn.tree as sktree
import sklearn.linear_model as sklinear
import sklearn.ensemble as skensemble
import sklearn.neural_network as skneural
import sklearn.svm as sksvm
from keras.models import Sequential
from keras.layers import Dense, Dropout, Activation, Flatten
from keras.utils import np_utils

def summarizeData(data):
    for colName, dataType in zip(data.columns, data.dtypes):
        print(colName + "\t->\t" + str(dataType) + "\t->\t" + str(np.unique(data[colName]).size))


def preProcessData(data):
    cleanedData = data.drop(labels=["PassengerId", "Name", "Ticket", "Cabin"], axis=1)

    avgAge = cleanedData["Age"].mean()
    cleanedData["Age"] = cleanedData["Age"].fillna(avgAge)
    # cleanedData["Cabin"] = cleanedData["Cabin"].fillna("Unknown")
    cleanedData["Embarked"] = cleanedData["Embarked"].fillna("Unknown")

    data['FamilySize'] = data.SibSp + data.Parch + 1

    encodedData = toEnums(cleanedData)
    # summarizeData(encodedData)
    return encodedData.drop(['SibSp', 'Parch'], axis=1)


def toEnums(data):
    knownNumericalFeatures = ["Age", "Fare", "SibSp", "Parch"]

    for colName in data.columns:
        if colName not in knownNumericalFeatures:
            # encode all the strings into enums
            cleanedInputFeatures = data[colName].apply(lambda x: str(x))
            encoder = skpre.LabelEncoder()
            encoder.fit(cleanedInputFeatures)
            data[colName] = encoder.transform(cleanedInputFeatures)
        else:
            # scale the features correctly
            data[colName] = skpre.scale(data[colName])

    return data


def trainAndTest(classifierName, classifier, trainingFeatures, trainingResults, testFeatures, testResults):
    classifier.fit(trainingFeatures, trainingResults)
    predictions = classifier.predict(testFeatures)

    tp = 0
    fp = 0
    tn = 0
    fn = 0
    for prediction, expected in zip(predictions, testResults):
        if prediction == expected and prediction == 1:
            tp += 1
        if prediction == expected and prediction == 0:
            tn += 1
        if prediction != expected and prediction == 1:
            fp += 1
        if prediction != expected and prediction == 0:
            fn += 1

    recall = tp / testResults[testResults == 1].size
    precision = tp / (tp + fp)

    f1Score = 2 * precision * recall / (precision + recall)

    print(classifierName + "\tprecision =\t" + str(precision) + "\trecall =\t" + str(recall) + "\tf1Score = \t" + str(f1Score))
#    print(classifierName + "\ttp =\t" + str(tp) + "\tfp =\t" + str(fp) + "\ttn = \t" + str(tn) + "\tfn = \t" + str(fn))


# _____________________________________________________________

titanicData = pd.read_csv("train.csv")
processedTitanicData = preProcessData(titanicData)

trainData, testData = sksel.train_test_split(processedTitanicData, test_size=(4/5))

trainingResults = trainData["Survived"]
trainingFeatures = trainData.drop("Survived", axis=1)

testResults = testData["Survived"]
testFeatures = testData.drop("Survived", axis=1)


# ___________________________________________________________
rng = np.random.RandomState(1)

ridge = sklinear.RidgeClassifier()
trainAndTest("ridge\t\t", ridge, trainingFeatures, trainingResults, testFeatures, testResults)

dTree = sktree.DecisionTreeClassifier()
trainAndTest("plainTree\t", dTree, trainingFeatures, trainingResults, testFeatures, testResults)
print(dTree.feature_importances_)

forest1000 = skensemble.RandomForestClassifier(bootstrap=True, max_leaf_nodes=82, random_state=rng, n_estimators=1000)
trainAndTest("forest1000\t", forest1000, trainingFeatures, trainingResults, testFeatures, testResults)

print(forest1000.feature_importances_)

forest300 = skensemble.RandomForestClassifier(n_estimators=300, random_state=rng)
trainAndTest("forest300\t", forest300, trainingFeatures, trainingResults, testFeatures, testResults)

boostedTreeClassifier50 = skensemble.AdaBoostClassifier(dTree, n_estimators=50, random_state=rng)
trainAndTest("boostedTree50", boostedTreeClassifier50, trainingFeatures, trainingResults, testFeatures, testResults)

boostedTreeClassifier300 = skensemble.AdaBoostClassifier(dTree, n_estimators=300, random_state=rng)
trainAndTest("boostedTree300", boostedTreeClassifier300, trainingFeatures, trainingResults, testFeatures, testResults)

svmClassifier = sksvm.LinearSVC()
trainAndTest("svm\t\t\t", svmClassifier, trainingFeatures, trainingResults, testFeatures, testResults)

neural = skneural.MLPClassifier(hidden_layer_sizes=(150, 200), random_state=rng)
trainAndTest("neural\t\t", neural, trainingFeatures, trainingResults, testFeatures, testResults)


# ____________________________

# model = Sequential()
# model.add(Dense())
