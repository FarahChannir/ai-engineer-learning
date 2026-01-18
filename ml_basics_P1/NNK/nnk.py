from sklearn.datasets import load_breast_cancer
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import accuracy_score, precision_score, recall_score



def KN():
    data = load_breast_cancer()
    print(data.keys())
    X = data.data
    y = data.target
    print(X.shape, y.shape)
    X_train, X_test, y_train, y_test = train_test_split(
       X, y, test_size=0.2, random_state=100
     )
    print(X_train.shape, X_test.shape, y_train.shape, y_test.shape)

    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)
    print(X_train_scaled.shape, X_test_scaled.shape)

    k_values = [3, 5, 7]

    for k in range(1, 21):
        knn = KNeighborsClassifier(n_neighbors=k)
        knn.fit(X_train_scaled, y_train)
        y_pred = knn.predict(X_test_scaled)
        accuracy = accuracy_score(y_test, y_pred)
        precision = precision_score(y_test, y_pred)
        recall = recall_score(y_test, y_pred)
        print(f"K={k}: Accuracy={accuracy:.4f}, Precision={precision:.4f}, Recall={recall:.4f}")

KN()
