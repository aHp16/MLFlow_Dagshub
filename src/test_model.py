import pytest
from sklearn.datasets import load_wine
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score

@pytest.fixture
def wine_data():
    """Load wine dataset and split into train/test."""
    wine = load_wine()
    X_train, X_test, y_train, y_test = train_test_split(
        wine.data, wine.target, test_size=0.10, random_state=42
    )
    return X_train, X_test, y_train, y_test

def test_random_forest_accuracy(wine_data):
    """Ensure RandomForest gets at least 85% accuracy."""
    X_train, X_test, y_train, y_test = wine_data
    clf = RandomForestClassifier(max_depth=5, n_estimators=10, random_state=42)
    clf.fit(X_train, y_train)
    y_pred = clf.predict(X_test)

    acc = accuracy_score(y_test, y_pred)
    print(f"Test accuracy: {acc:.4f}")
    
    assert acc >= 0.85, f"Model accuracy too low: {acc}"
