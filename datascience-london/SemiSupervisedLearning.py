import numpy as np
import math
from sklearn.cross_validation import StratifiedKFold, cross_val_score
from sklearn.decomposition import PCA
from sklearn.svm import SVC
from sklearn import cross_validation, grid_search


class SemiSupervisedLearning:
    def __init__(self, traindata, testdata, target, C_range = 10 ** np.arange(3,7, 0.5), gamma_range = 10 ** np.arange(0, -2, -0.25)):

        if 3*len(traindata) > len(testdata):
            raise Exception("This technique is only useful if test data is much bigger than training data.")
        self._traindata = traindata
        self._testdata = testdata
        self.target = target
        self.C_range = C_range
        self.gamma_range = gamma_range
        self.dimreduce()

    def dimreduce(self, dims = 10):
        pca = PCA(n_components=dims, whiten=True)
        self._traindata_comp = pca.fit_transform(self._traindata)
        self._testdata_comp = pca.transform(self._testdata)

    def train(self, threshold = 0.97):
        params = [{'C':self.C_range, 'gamma':self.gamma_range}]
        cvk = StratifiedKFold(self.target, n_folds=5)
        svc = SVC(kernel = 'rbf', probability=True)
        clf = grid_search.GridSearchCV(svc, param_grid=params)
        clf.fit(self._traindata_comp, self.target)
        score = cross_val_score(clf.best_estimator_, self._traindata_comp, self.target, cv = 20)
        print("best classifier:" , clf.best_estimator_)

        test_prob_values = clf.best_estimator_.predict_proba(self._testdata_comp)

        max_values = map(lambda x: max(x[0],x[1]), test_prob_values)
        indices = np.where(np.asarray(max_values) > threshold)

        C_range2 = 10.0 ** np.arange(6.75,7.5,.25)
        gamma_range2 = 10.0 ** np.arange(-1.5,0,.05)

        params2 = [{'C':C_range2, 'gamma':gamma_range2}]

        self.clf2 = grid_search.GridSearchCV(svc, param_grid=params2,cv = cvk)

        test_set_comp_subset = self._testdata_comp[indices]
        target_test_subset = clf.best_estimator_.predict(test_set_comp_subset)

        train_set_second = np.concatenate((self._traindata_comp, test_set_comp_subset),axis= 0)
        target_second = np.concatenate((self.target, target_test_subset), axis = 0)

        self.clf2.fit(train_set_second, target_second)
        score2 = cross_val_score(self.clf2.best_estimator_, train_set_second, target_second, cv = 20)
        print("best classifier:", self.clf2.best_estimator_)
        return self.clf2




