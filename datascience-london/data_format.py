


import numpy as np
import math
import pandas as pd
from sklearn import svm
from sklearn.cross_validation import StratifiedKFold, cross_val_score
from sklearn.decomposition import PCA
from sklearn.linear_model.logistic import LogisticRegression
from sklearn.svm import SVC
from sklearn import cross_validation, grid_search, tree


train_set = open("train.csv")
test_set = open("test.csv")

train_set = np.genfromtxt(open("train.csv", 'r'), delimiter=',', dtype='float')
test_set = np.genfromtxt(open("test.csv", 'r'), delimiter=',', dtype='float')


target = np.genfromtxt(open("trainLabels.csv", 'r'), dtype='float', delimiter=',')

pca = PCA(n_components=12, whiten=True)

train_set_comp = pca.fit_transform(train_set)
test_set_comp = pca.transform(test_set)

train_train_set, train_validate, target_train, target_validate = cross_validation.train_test_split(train_set_comp, target, test_size = 0.9)

#clf = SVC(kernel='linear', C=1).fit(train_train_set, target_train)

#skf = StratifiedKFold()

####### code for svc #########################
C_range = 10 ** np.arange(3,7, 0.5)
gamma_range = 10 ** np.arange(-1, -3, -0.5)

params = [{'C':C_range, 'gamma':gamma_range}]
cvk = StratifiedKFold(target, n_folds=5)

svc = SVC(kernel = 'rbf', probability=True)
clf = grid_search.GridSearchCV(svc, param_grid=params, cv=cvk)

clf.fit(train_set_comp, target)
score = cross_val_score(clf.best_estimator_, train_set_comp, target, cv = 20)
print("best classifier:" , clf.best_estimator_)

test_prob_values = clf.best_estimator_.predict_proba(test_set_comp)
train_prob_values = clf.best_estimator_.predict_proba(train_set_comp)
#
#

############ code for decision tree  ###################
#
DT = tree.DecisionTreeClassifier()
depth_list = [3, 4, 5, 6, 7, 8]
min_samples_leaf_list = [3, 5, 7, 10 ]
min_samples_split_list = [2,3,4,5]
params_dt = [{"max_depth": depth_list, 'min_samples_leaf': min_samples_leaf_list, 'min_samples_split' : min_samples_split_list}]
##
clfDT = grid_search.GridSearchCV(DT, param_grid=params_dt, cv=cvk)
clfDT.fit(train_set_comp, target)
scoreDT = cross_val_score(clfDT.best_estimator_, train_set_comp, target, cv = 10)
print("best classifier:" , clfDT.best_estimator_)
print(np.median(scoreDT))


############ code for Logistic Regression ####################

clfLR = LogisticRegression(penalty='l2', dual=False, class_weight='auto')
clfLR.fit(train_set_comp, target)
scoreLR = cross_val_score(clfLR, train_set_comp, target, cv= 20)
print(np.median(scoreLR))


#result = clfDT.best_estimator_.predict(test_set_comp)
result= clf.best_estimator_.predict(test_set_comp)

f = open("result.csv", 'w')
f.write('Id,Solution\n')

count =1
while count <= 9000:
    f.write('%d,%d\n' % (count, result[count-1]))
    count += 1

f.close()


result2 = np.genfromtxt(open("result2.csv", 'r'), dtype=float, delimiter=',')

#result = clfLR.predict(test_set_comp)
result2 = result2[1:9001,]
print(result2)

op = [value for key, value in result2]

matchscore = sum([a==b for a,b in zip(result, op)])


print(float(matchscore/9000))