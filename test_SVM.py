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

    def test_mnist(self):
    # 1. Wczytanie danych MNIST
        (X_train, y_train), (X_test, y_test) = mnist.load_data()
        # X_train: (60000, 28, 28)
        # y_train: (60000,)

        # 2. Zmiana kształtu na 2D (flatten)
        #    z (n_samples, 28, 28) -> (n_samples, 28*28)
        n_train = X_train.shape[0]
        n_test = X_test.shape[0]

        X_train_flat = X_train.reshape(n_train, -1)
        X_test_flat = X_test.reshape(n_test, -1)

        # 3. Normalizacja danych do [0, 1]
        X_train_flat = X_train_flat.astype(np.float32) / 255.0
        X_test_flat = X_test_flat.astype(np.float32) / 255.0

        # 4. Inicjalizacja Twojego SVM
        #    możesz oczywiście zmieniać C, lr, n_epochs, batch_size
        Classifier = mfup.svm.SVM(C = 0.02, n_epochs = 24, batch_size = 16, lr = 0.01, random_state = 19)
        Classifier.fit(X_train_flat, y_train)

        y_pred = Classifier.predict(X_test_flat)

        print('\n')
        print("\tMNIST results:")
        print('\n')
        
        # 7. Ocena jakości
        accuracy = (y_pred == y_test).mean()
        print(f"mfup svm accuracy: {accuracy:.4f}")

        (X_train, y_train), (X_test, y_test) = mnist.load_data()

        n_train = X_train.shape[0]
        n_test = X_test.shape[0]

        X_train_flat = X_train.reshape(n_train, -1).astype(np.float32) / 255.0
        X_test_flat = X_test.reshape(n_test, -1).astype(np.float32) / 255.0

        Classifier = sk.svm.LinearSVC()
        Classifier.fit(X_train_flat, y_train)

        y_pred = Classifier.predict(X_test_flat)
        accuracy = (y_pred == y_test).mean()
        print(f"LinearSVC accuracy: {accuracy:.4f}")


if __name__ == "__main__":
    unittest.main()
