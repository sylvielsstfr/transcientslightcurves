"""
Classifieur pour les supernovae (Random Forest).
"""
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report, confusion_matrix
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler


def extract_features(results_dict):
    """
    Extrait les features de plusieurs résultats de fit_single_event.
    """
    X = []
    for snid, result in results_dict.items():
        if result['success']:
            X.append(result['features'])
    return pd.DataFrame(X).fillna(0)  # Remplacer les NaN par 0

def train_classifier(X, y, test_size=0.2, random_state=42):
    """
    Entraîne un classifieur Random Forest sur les features X et labels y.
    """
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=test_size, random_state=random_state
    )

    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)

    clf = RandomForestClassifier(
        n_estimators=100,
        max_depth=10,
        random_state=random_state,
        class_weight='balanced'
    )
    clf.fit(X_train_scaled, y_train)

    y_pred = clf.predict(X_test_scaled)
    print(classification_report(y_test, y_pred, target_names=['Ia', 'non-Ia']))
    print(confusion_matrix(y_test, y_pred))

    return clf, scaler

def predict_type(clf, scaler, result):
    """
    Prédit le type d'une supernova à partir de ses features.
    """
    if not result['success']:
        return None

    features = pd.DataFrame([result['features']]).fillna(0)
    features_scaled = scaler.transform(features)
    return clf.predict(features_scaled)[0]
