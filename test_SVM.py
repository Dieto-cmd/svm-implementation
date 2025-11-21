import sklearn as sk
import unittest
import numpy as np
import mfup


class TestSVM(unittest.TestCase):
    def test_digits(self):
        features = sk.datasets.load_digits(as_frame = True)['data']
        targets = sk.datasets.load_digits(as_frame = True)['target'] #labels

        print('\n')
        print("\tDigits Recognition results:")
        print('\n')

        Classifier = sk.svm.LinearSVC(random_state = 19)
        scores = sk.model_selection.cross_val_score(Classifier, features, targets, scoring = 'accuracy', cv=5) 
        scores = np.round(scores,4) * 100
        print("Scikit-learn SVM scores:", scores)
        print("Mean Scikit-learn SVM accuracy:", round(scores.mean(), 2), '%')

        Classifier = mfup.svm.SVM(C = 0.06, n_epochs = 22, random_state = 19)
        scores = sk.model_selection.cross_val_score(Classifier, features, targets, scoring = 'accuracy', cv=5) 
        scores = np.round(scores,4) * 100
        print('\n')
        print("MFUP SVM scores:", scores)
        print("Mean MFUP SVM accuracy:", round(scores.mean(), 2), '%')


    def test_breast_cancer(self):
        features = sk.datasets.load_breast_cancer(as_frame = True)['data']
        targets = sk.datasets.load_breast_cancer(as_frame = True)['target']

        print('\n')
        print("\tBreast Cancer Recognition results:")
        print('\n')

        Classifier = sk.svm.LinearSVC(random_state = 19)
        scores = sk.model_selection.cross_val_score(Classifier, features, targets, scoring = 'accuracy', cv=5) 
        scores = np.round(scores,4) * 100
        print("Scikit-learn SVM scores:", scores)
        print("Mean Scikit-learn SVM accuracy:", round(scores.mean(), 2), '%')

        Classifier = mfup.svm.SVM(C = 0.06, n_epochs = 22, random_state = 19)
        scores = sk.model_selection.cross_val_score(Classifier, features, targets, scoring = 'accuracy', cv=5) 
        scores = np.round(scores,4) * 100
        print('\n')
        print("MFUP SVM scores:", scores)
        print("Mean MFUP SVM accuracy:", round(scores.mean(), 2), '%')


if __name__ == "__main__":
    unittest.main()