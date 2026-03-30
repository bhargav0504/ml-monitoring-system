def threshold_alerts(metrics, thresholds=None):
    if thresholds is None:
        thresholds = {'accuracy': 0.80, 'f1': 0.70, 'auc': 0.75, 'psi': 0.10}

    alerts = []
    for metric, value in metrics.items():
        if metric in thresholds and value is not None and value < thresholds[metric]:
            alerts.append(f'{metric} below threshold: {value:.4f} < {thresholds[metric]:.4f}')

    return alerts


def drift_alert(drift_report, psi_threshold=0.1, ks_p_threshold=0.05, chi2_p_threshold=0.05):
    notifications = []
    for feature, m in drift_report.items():
        if 'psi' in m and m['psi'] > psi_threshold:
            notifications.append(f'{feature} has high PSI {m["psi"]:.4f}')
        if 'ks_p' in m and m['ks_p'] < ks_p_threshold:
            notifications.append(f'{feature} KS p-value {m["ks_p"]:.4f} below {ks_p_threshold}')
        if 'p_value' in m and m['p_value'] < chi2_p_threshold:
            notifications.append(f'{feature} chi2 p-value {m["p_value"]:.4f} below {chi2_p_threshold}')

    return notifications
