import pandas as pd
from sklearn.metrics import accuracy_score, f1_score, roc_auc_score


def compute_metrics(y_true, y_pred, y_proba=None):
    metrics = {
        'accuracy': accuracy_score(y_true, y_pred),
        'f1': f1_score(y_true, y_pred, average='weighted')
    }
    if y_proba is not None:
        try:
            metrics['auc'] = roc_auc_score(y_true, y_proba, multi_class='ovr')
        except Exception:
            metrics['auc'] = None
    return metrics


def evaluate_time_series(history_df):
    results = []
    for ts, slice_df in history_df.groupby('date'):
        row = compute_metrics(slice_df['target'], slice_df['prediction'], slice_df.get('probability'))
        row['date'] = ts
        results.append(row)
    return pd.DataFrame(results)
