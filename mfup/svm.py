import numpy as np
from sklearn.base import BaseEstimator, ClassifierMixin

class SVM(BaseEstimator, ClassifierMixin):
    #it inherits from BaseEstimator and ClassifierMixin only for the sake of beeing able to run cross_val_score() on it 
    """
    Prosta implementacja liniowego SVM z hinge loss,
    obsługująca wiele klas strategią One-vs-Rest (OvR).

    Parametry:
    ----------
    C : float
        Siła kary za błędy (soft margin). Im większe C, tym mniejsza regularizacja.
    lr : float
        Learning rate dla SGD.
    n_epochs : int
        Liczba epok (przejść po całym zbiorze).
    random_state : int | None
        Ziarno generatora losowego (dla powtarzalności).
    """

    def __init__(self, C=1.0, lr=0.01, n_epochs=5, random_state=None):
        self.C = C
        self.lr = lr
        self.n_epochs = n_epochs
        self.random_state = random_state

    def fit(self, X, y):
        X = np.asarray(X, dtype=float)
        y = np.asarray(y)

        n_samples, n_features = X.shape

        # Zapamiętujemy unikalne klasy (mogą być 0,1,2,... lub inne)
        self.classes_ = np.unique(y)
        n_classes = len(self.classes_)

        # Inicjalizujemy macierze wag dla każdej klasy:
        # w_: shape (n_classes, n_features)
        # b_: shape (n_classes,)
        rng = np.random.default_rng(self.random_state)
        self.w_ = np.zeros((n_classes, n_features), dtype=float)
        self.b_ = np.zeros(n_classes, dtype=float)

        # Trenujemy osobny model OvR dla każdej klasy k
        for class_idx, class_label in enumerate(self.classes_):
            # Tworzymy etykiety binarne: +1 dla aktualnej klasy, -1 dla reszty
            y_binary = np.where(y == class_label, 1.0, -1.0)

            w = self.w_[class_idx]
            b = self.b_[class_idx]

            batch_size = 64

            for epoch in range(self.n_epochs):
                indices = np.arange(n_samples)
                rng.shuffle(indices)

                for start in range(0, n_samples, batch_size):
                    end = start + batch_size
                    batch_idx = indices[start:end]

                    Xb = X[batch_idx]            # shape: (B, n_features)
                    yb = y_binary[batch_idx]     # shape: (B,)

                    margins = yb * (Xb @ w + b)  # shape: (B,)

                    # maska, które próbki są "złe" (margin < 1)
                    mask = margins < 1

                    if np.any(mask):
                        # tylko błędne wnoszą wkład
                        X_bad = Xb[mask]
                        y_bad = yb[mask]         # shape: (B_bad,)

                        # gradient po hinge loss dla batcha:
                        grad_w_hinge = -self.C * (y_bad[:, None] * X_bad).sum(axis=0)
                        grad_b_hinge = -self.C * y_bad.sum()
                    else:
                        grad_w_hinge = 0.0
                        grad_b_hinge = 0.0

                    # dodajemy regularizację
                    grad_w = w + grad_w_hinge
                    grad_b = grad_b_hinge

                    w = w - self.lr * grad_w
                    b = b - self.lr * grad_b

            # zapisujemy wagi dla tej klasy
            self.w_[class_idx] = w
            self.b_[class_idx] = b

        return self

    def _decision_function_ovr(self, X):
        """
        Zwraca macierz score'ów (marginów) dla każdej klasy:
        shape: (n_samples, n_classes)
        """
        X = np.asarray(X, dtype=float)
        # scores[i, k] = w_k · x_i + b_k
        scores = X @ self.w_.T + self.b_
        return scores

    def decision_function(self, X):
        """
        Dla binary classification: zwraca 1D array score'ów.
        Dla multi-class (OvR): zwraca 2D array (n_samples, n_classes).
        """
        scores = self._decision_function_ovr(X)
        if scores.shape[1] == 1:
            return scores.ravel()
        return scores

    def predict(self, X):
        """
        Wybiera klasę o największym score (marginie).
        """
        scores = self._decision_function_ovr(X)
        # argmax po klasach
        indices = np.argmax(scores, axis=1)
        return self.classes_[indices]