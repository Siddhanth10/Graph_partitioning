from sklearn.ensemble import RandomForestClassifier
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

def build_model() -> Pipeline:
    return Pipeline([
        ("scale", StandardScaler()),
        ("classifier", RandomForestClassifier(n_estimators=250, random_state=42, class_weight="balanced", n_jobs=-1)),
    ])
