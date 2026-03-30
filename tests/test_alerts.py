from monitoring.alerts import threshold_alerts, drift_alert


def test_threshold_alerts():
    metrics = {'accuracy': 0.75, 'f1': 0.68, 'auc': 0.72}
    alerts = threshold_alerts(metrics)
    assert len(alerts) == 3


def test_drift_alert():
    drift_report = {
        'feature1': {'psi': 0.15, 'ks_p': 0.02},
        'feature2': {'p_value': 0.01},
    }
    notifs = drift_alert(drift_report)
    assert any('high PSI' in n for n in notifs)
    assert any('KS p-value' in n for n in notifs)
    assert any('chi2 p-value' in n for n in notifs)
