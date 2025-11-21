import mfup
import sklearn as sk
import numpy as np

Classifier = mfup.svm.SVM(random_state = 19)
features = sk.datasets.load_breast_cancer(as_frame = True)['data']
targets = sk.datasets.load_breast_cancer(as_frame = True)['target']
C_values =np.arange(0.01,0.1,0.01)
epochs_values = np.arange(1,25,1)
params = {'C': C_values, 'n_epochs': epochs_values }
grid_search = sk.model_selection.GridSearchCV(estimator =Classifier, param_grid =params, verbose=3, scoring ='accuracy', cv = 5)
grid_search.fit(features, targets)
print("Best score:", grid_search.best_score_)
print("Best hyperparameters:", grid_search.best_params_)