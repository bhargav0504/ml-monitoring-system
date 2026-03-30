import os
import pandas as pd
from sklearn.datasets import load_iris
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
import joblib

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
DATA_DIR = os.path.join(os.path.dirname(BASE_DIR), 'data')
MODEL_DIR = os.path.join(os.path.dirname(BASE_DIR), 'model')

os.makedirs(DATA_DIR, exist_ok=True)
os.makedirs(MODEL_DIR, exist_ok=True)


def main():
    iris = load_iris(as_frame=True)
    df = pd.concat([iris.data, iris.target.rename('target')], axis=1)

    train, _ = train_test_split(df, test_size=0.2, random_state=42, stratify=df['target'])
    train.to_csv(os.path.join(DATA_DIR, 'reference_data.csv'), index=False)

    X = train[iris.feature_names]
    y = train['target']

    model = RandomForestClassifier(random_state=42)
    model.fit(X, y)

    joblib.dump(model, os.path.join(MODEL_DIR, 'baseline_model.pkl'))

    preds = model.predict(X)
    print('Train accuracy:', accuracy_score(y, preds))


if __name__ == '__main__':
    main()