


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
C_range = 10.0 ** np.arange(6.5,7.5,.25)
gamma_range = 10.0 ** np.arange(-1.5,0.5,.1)

params = [{'C':C_range, 'gamma':gamma_range}]
cvk = StratifiedKFold(target, n_folds=5)

svc = SVC(kernel = 'rbf', probability=True)
clf = grid_search.GridSearchCV(svc, param_grid=params)

clf.fit(train_set_comp, target)
score = cross_val_score(clf.best_estimator_, train_set_comp, target, cv = 20)
print("best classifier:" , clf.best_estimator_)

test_prob_values = clf.best_estimator_.predict_proba(test_set_comp)
train_prob_values = clf.best_estimator_.predict_proba(train_set_comp)
#
#
max_values = map(lambda x: max(x[0],x[1]), test_prob_values)
indices = np.where(np.asarray(max_values) > 0.95)

C_range2 = 10.0 ** np.arange(6.5,7.5,.25)
gamma_range2 = 10.0 ** np.arange(-1.5,0.5,.1)

params2 = [{'C':C_range2, 'gamma':gamma_range2}]

svc2 = SVC(kernel = 'rbf', probability=True)
clf2 = grid_search.GridSearchCV(svc, param_grid=params2,cv = cvk)

test_set_comp_subset = test_set_comp[indices]
target_test_subset = clf.best_estimator_.predict(test_set_comp_subset)

train_set_second = np.concatenate((train_set_comp, test_set_comp_subset),axis= 0)
target_second = np.concatenate((target, target_test_subset), axis = 0)
#target_second = target + target_test_subset

clf2.fit(train_set_second, target_second)

score2 = cross_val_score(clf2.best_estimator_, train_set_second, target_second, cv = 20)
print("best classifier:" , clf2.best_estimator_)
print(np.median(score2))

#final_set_comp = train_set_second
#final_target = target_second
#
#for i in range(1, 3):
#    svc = SVC(kernel = 'linear', C=10, probability=True)
#    clf2 = grid_search.GridSearchCV(svc, param_grid=params)
#    clf2.fit(train_set_comp, target)
#    final_prob_values = clf2.best_estimator_.predict(final_set_comp)
##values = filter(lambda x: max(x[0],x[1]) > 0.9, test_prob_values)
#
#    max_values = map(lambda x: max(x[0],x[1]), final_prob_values)
#    indices = np.where(np.asarray(max_values) > 0.9)
#    final_set_comp = final_set_comp[indices]
#    final_target = final_target[indices]



#result = clfDT.best_estimator_.predict(test_set_comp)
result= clf.best_estimator_.predict(test_set_comp)
result2 = clf2.best_estimator_.predict(test_set_comp)

f = open("result3.csv", 'w')
f.write('Id,Solution\n')

count =1
while count <= 9000:
    f.write('%d,%d\n' % (count, result[count-1]))
    count += 1

f.close()


result3 = np.genfromtxt(open("result2.csv", 'r'), dtype=float, delimiter=',')

#result = clfLR.predict(test_set_comp)
result3 = result3[1:9001,]
print(result3)

op = [value for key, value in result3]

matchscore = sum([a==b for a,b in zip(result, op)])


print(float(matchscore/9000))