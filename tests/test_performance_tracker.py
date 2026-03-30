from monitoring.performance_tracker import compute_metrics


def test_compute_metrics_classification():
    y_true = [0, 1, 1, 0]
    y_pred = [0, 1, 0, 0]
    metrics = compute_metrics(y_true, y_pred)

    assert metrics['accuracy'] == 0.75
    assert 0 <= metrics['f1'] <= 1


def test_compute_metrics_proba():
    y_true = [0, 1, 1, 0]
    y_pred = [0, 1, 0, 0]
    y_proba = [[0.8, 0.2], [0.2, 0.8], [0.6, 0.4], [0.9, 0.1]]
    metrics = compute_metrics(y_true, y_pred, y_proba)
    assert 'auc' in metrics
