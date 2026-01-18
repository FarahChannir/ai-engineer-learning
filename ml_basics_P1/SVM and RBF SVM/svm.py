import numpy as np
from sklearn.datasets import load_breast_cancer
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC
from sklearn.metrics import accuracy_score, precision_score, recall_score
def SVM_RBF():
    data = load_breast_cancer()
    X = data.data
    y = data.target

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=80
    )

    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)
      
    linear_svm = SVC(kernel="linear", C=1)
    linear_svm.fit(X_train, y_train)

    y_pred_linear = linear_svm.predict(X_test)
     
    print("Linear SVM Results")
    print("Accuracy:", accuracy_score(y_test, y_pred_linear))
    print("Precision:", precision_score(y_test, y_pred_linear))
    print("Recall:", recall_score(y_test, y_pred_linear))

    param_grid = {
        'C': [0.1, 1, 10, 100],
        'gamma': [0.001, 0.01, 0.1, 1]
    }

    svc = SVC(kernel='rbf')
    grid_search = GridSearchCV(svc, param_grid, cv=5)
    grid_search.fit(X_train_scaled, y_train)
    print("Best parameters:", grid_search.best_params_)


    best_svc = grid_search.best_estimator_
    y_pred = best_svc.predict(X_test_scaled)

    accuracy = accuracy_score(y_test, y_pred)
    precision = precision_score(y_test, y_pred)
    recall = recall_score(y_test, y_pred)

    print(f"Accuracy: {accuracy:.4f}, Precision: {precision:.4f}, Recall: {recall:.4f}")

SVM_RBF()