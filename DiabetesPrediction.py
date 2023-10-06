import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.preprocessing import MinMaxScaler,StandardScaler
from sklearn.ensemble import RandomForestClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.svm import SVC
from sklearn.metrics import accuracy_score
import pickle

df = pd.read_csv('diabetes_data.csv')


df.head()


df.info()


df.shape


x = df.drop(columns = 'Outcome', axis = 1)
y = df['Outcome']


x.head()


sc = StandardScaler()


x = sc.fit_transform(x)

from sklearn.model_selection import train_test_split as tts
x_train, x_test, y_train, y_test = tts(x, y, test_size = 0.2, stratify = y, random_state = 5)

class AdaBoostClassifier:
    def __init__(self, n_estimators=10):
        self.n_estimators = n_estimators
        self.estimators = []
        self.estimator_weights = []

    def fit(self, X, y):
        # Initialize sample weights
        sample_weights = np.ones(len(X)) / len(X)

        for _ in range(self.n_estimators):
            estimator = DecisionTreeClassifier(max_depth=1)
            estimator.fit(X, y, sample_weight=sample_weights)
            
            y_pred = estimator.predict(X)
            
            error = np.sum(sample_weights * (y_pred != y))
        
            estimator_weight = 0.5 * np.log((1 - error) / error)
            
            sample_weights *= np.exp(-estimator_weight * y * y_pred)
            sample_weights /= np.sum(sample_weights)

            self.estimators.append(estimator)
            self.estimator_weights.append(estimator_weight)

    def predict(self, X):
        predictions = np.zeros(len(X))
        for estimator, estimator_weight in zip(self.estimators, self.estimator_weights):
            predictions += estimator_weight * estimator.predict(X)
        return np.sign(predictions)


classifier_adb = AdaBoostClassifier()
classifier_rfc = RandomForestClassifier()
classifier_svm = SVC(kernel = 'linear')


classifier_adb.fit(x_train, y_train)
classifier_rfc.fit(x_train, y_train)
classifier_svm.fit(x_train, y_train)



y_pred_adb = classifier_adb.predict(x_test)
y_pred_rfc = classifier_rfc.predict(x_test)
y_pred_svm = classifier_svm.predict(x_test)


print(accuracy_score(y_pred_adb, y_test))
print(accuracy_score(y_pred_rfc, y_test))
print(accuracy_score(y_pred_svm, y_test))

pickle.dump(classifier_svm, open("DiabetesPrediction.pkl",'wb'))




