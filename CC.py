import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import cross_val_score
from sklearn import tree
from sklearn.linear_model import Perceptron
from sklearn.neural_network import MLPClassifier
from sklearn import svm
from sklearn.naive_bayes import GaussianNB
from sklearn.linear_model import LogisticRegression
from sklearn.neighbors import KNeighborsClassifier
from sklearn.ensemble import BaggingClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.ensemble import AdaBoostClassifier
from sklearn.ensemble import GradientBoostingClassifier


def numOfNull(data):
    print("Number of null values in each column: ")
    for col in data.columns:
        print(col, ":", data[col].isnull().sum())
    return


def formatData(data):
    for col in data.columns:
        col_values = data[col]
        if col_values.dtype == int:
            data.ix[:, col] = col_values.astype(float)
        elif col_values.dtype != float:
            data.ix[:, col] = col_values.astype('category')
    return


#import ionosphere data
na_strings = [
    '',
    'na', 'n.a', 'n.a.',
    'nan', 'n.a.n', 'n.a.n.',
    'NA', 'N.A', 'N.A.',
    'NaN', 'N.a.N', 'N.a.N.',
    'NAN', 'N.A.N', 'N.A.N.',
    'nil', 'Nil', 'NIL',
    'null', 'Null', 'NULL']
ionosphere = pd.read_table("ionosphere.data", sep=',', header=None, na_values=na_strings)


# convert response variable as categorical variable and numerical variable to float type
formatData(ionosphere)
numOfNull(ionosphere)
# Check for correlation between two columns of the data set
correlation = ionosphere.corr()
# np.savetxt('correlation.csv', (correlation), delimiter=',')

# Standardize the data set
ionosphere.ix[:, 0:33] = StandardScaler().fit_transform(ionosphere.ix[:, 0:33].values)

#Column Rings is the target
ionosphere.data = ionosphere.ix[:, 0:33]
# print(ionosphere.data)
ionosphere.target = ionosphere.ix[:, 34]
# print(ionosphere.target)

#Decision Tree
clfDT = tree.DecisionTreeClassifier(max_depth=5)
scores = cross_val_score(clfDT, ionosphere.data, ionosphere.target, cv=10)
precision = cross_val_score(clfDT, ionosphere.data, ionosphere.target, cv=10, scoring='precision_micro')
DTprecision=np.mean(precision)
DTaccuracy=np.mean(scores)
print("The accuracy of Decision Tree is:",DTaccuracy)
print("The precision of Decision Tree is:",DTprecision)

#Perceptron
clfPT = Perceptron(n_iter=100, random_state=16)
scores = cross_val_score(clfPT, ionosphere.data, ionosphere.target, cv=10)
precision = cross_val_score(clfPT, ionosphere.data, ionosphere.target, cv=10, scoring='precision_micro')
PTprecision=np.mean(precision)
PTaccuracy=np.mean(scores)
print("The accuracy of Perceptron is:",PTaccuracy)
print("The precision of Perceptron is:",PTprecision)

#Neural Net
clfNN = MLPClassifier(solver='lbfgs', alpha=1e-6,hidden_layer_sizes=(5,5,5),activation='logistic')
scores = cross_val_score(clfNN, ionosphere.data, ionosphere.target, cv=10)
precision = cross_val_score(clfNN, ionosphere.data, ionosphere.target, cv=10, scoring='precision_micro')
NNprecision=np.mean(precision)
NNaccuracy=np.mean(scores)
print("The accuracy of Neural Net is:",NNaccuracy)
print("The precision of Neural Net is:",NNprecision)

#SVM
clfSVM= svm.SVC(kernel='poly',C=1.0)
scores = cross_val_score(clfSVM, ionosphere.data, ionosphere.target, cv=10)
precision = cross_val_score(clfSVM, ionosphere.data, ionosphere.target, cv=10, scoring='precision_micro')
SVMprecision=np.mean(precision)
SVMaccuracy=np.mean(scores)
print("The accuracy of SVM is:",SVMaccuracy)
print("The precision of SVM is:",SVMprecision)

#Naive Bayes
clfNB= GaussianNB()
scores = cross_val_score(clfNB, ionosphere.data, ionosphere.target, cv=10)
precision = cross_val_score(clfNB, ionosphere.data, ionosphere.target, cv=10, scoring='precision_micro')
NBprecision=np.mean(precision)
NBaccuracy=np.mean(scores)
print("The accuracy of Naive Bayes is:",NBaccuracy)
print("The precision of Naive Bayes is:",NBprecision)

#Logistic Regression
clfLR= LogisticRegression(penalty='l2',C=1.0,solver='liblinear',tol=1e-5)
scores = cross_val_score(clfLR, ionosphere.data, ionosphere.target, cv=10)
precision = cross_val_score(clfLR, ionosphere.data, ionosphere.target, cv=10, scoring='precision_micro')
LRprecision=np.mean(precision)
LRaccuracy=np.mean(scores)
print("The accuracy of Logistic Regression is:",LRaccuracy)
print("The precision of Logistic Regression is:",LRprecision)


#k-Nearest Neighbors
clfKNN= KNeighborsClassifier(n_neighbors=20)
scores = cross_val_score(clfKNN, ionosphere.data, ionosphere.target, cv=10)
precision = cross_val_score(clfKNN, ionosphere.data, ionosphere.target, cv=10, scoring='precision_micro')
KNNprecision=np.mean(precision)
KNNaccuracy=np.mean(scores)
print("The accuracy of k-Nearest Neighbors is:",KNNaccuracy)
print("The precision of k-Nearest Neighbors is:",KNNprecision)

#Bagging
clfBC= BaggingClassifier(base_estimator=clfSVM,n_estimators=20)
scores = cross_val_score(clfBC, ionosphere.data, ionosphere.target, cv=10)
precision = cross_val_score(clfBC, ionosphere.data, ionosphere.target, cv=10, scoring='precision_micro')
BCprecision=np.mean(precision)
BCaccuracy=np.mean(scores)
print("The accuracy of Bagging is:",BCaccuracy)
print("The precision of Bagging is:",BCprecision)

#Random Forests
clfRF= RandomForestClassifier(n_estimators=20,max_depth=5)
scores = cross_val_score(clfRF, ionosphere.data, ionosphere.target, cv=10)
precision = cross_val_score(clfRF, ionosphere.data, ionosphere.target, cv=10, scoring='precision_micro')
RFprecision=np.mean(precision)
RFaccuracy=np.mean(scores)
print("The accuracy of Random Forests is:",RFaccuracy)
print("The precision of Random Forests is:",RFprecision)

#Adaboost
clfAB= AdaBoostClassifier()
scores = cross_val_score(clfAB, ionosphere.data, ionosphere.target, cv=10)
precision = cross_val_score(clfAB, ionosphere.data, ionosphere.target, cv=10, scoring='precision_micro')
ABprecision=np.mean(precision)
ABaccuracy=np.mean(scores)
print("The accuracy of Adaboost is:",ABaccuracy)
print("The precision of Adaboost is:",ABprecision)

#Gradient Boosting
clfGB= GradientBoostingClassifier()
scores = cross_val_score(clfGB, ionosphere.data, ionosphere.target, cv=10)
precision = cross_val_score(clfGB, ionosphere.data, ionosphere.target, cv=10, scoring='precision_micro')
GBprecision=np.mean(precision)
GBaccuracy=np.mean(scores)
print("The accuracy of Gradient Boosting is:",GBaccuracy)
print("The precision of Gradient Boosting is:",GBprecision)


