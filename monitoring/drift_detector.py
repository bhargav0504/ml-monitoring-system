import os
import numpy as np
import pandas as pd
from scipy.stats import ks_2samp, chi2_contingency


def _safe_quantiles(values, bins=10):
    values = np.sort(np.array(values, dtype=float))
    if len(values) < bins:
        bins = max(2, len(values) - 1)
    return np.quantile(values, np.linspace(0, 1, bins + 1))


def psi(expected, actual, bins=10):
    expected = np.array(expected, dtype=float)
    actual = np.array(actual, dtype=float)
    edges = _safe_quantiles(expected, bins=bins)
    expected_perc, _ = np.histogram(expected, bins=edges)
    actual_perc, _ = np.histogram(actual, bins=edges)
    expected_perc = expected_perc / np.sum(expected_perc)
    actual_perc = actual_perc / np.sum(actual_perc)
    expected_perc = np.where(expected_perc == 0, 1e-8, expected_perc)
    actual_perc = np.where(actual_perc == 0, 1e-8, actual_perc)
    return np.sum((expected_perc - actual_perc) * np.log(expected_perc / actual_perc))


def run_drift_check(reference_path, production_path, exclude_columns=None, psi_threshold=0.1, ks_p_threshold=0.05):
    ref = pd.read_csv(reference_path)
    prod = pd.read_csv(production_path)

    if exclude_columns is None:
        exclude_columns = []

    report = {}
    numeric_columns = [col for col in ref.select_dtypes(include=[np.number]).columns if col in prod.columns and col not in exclude_columns]
    categorical_columns = [col for col in ref.select_dtypes(include=['object', 'category']).columns if col in prod.columns and col not in exclude_columns]

    for col in numeric_columns:
        ks_stat, ks_p = ks_2samp(ref[col], prod[col])
        col_psi = psi(ref[col], prod[col])
        report[col] = {
            'ks_stat': float(ks_stat),
            'ks_p': float(ks_p),
            'psi': float(col_psi),
            'drift_detected': bool((ks_p < ks_p_threshold) or (col_psi > psi_threshold)),
        }

    for col in categorical_columns:
        cross = pd.crosstab(ref[col], prod[col])
        if cross.shape[0] > 1 and cross.shape[1] > 1:
            chi2, p, _, _ = chi2_contingency(cross)
            report[col] = {
                'chi2': float(chi2),
                'p_value': float(p),
                'drift_detected': bool(p < ks_p_threshold),
            }

    return report


def save_drift_report(report, output_html):
    rows = []
    for feature, metrics in report.items():
        row = {'feature': feature}
        row.update(metrics)
        rows.append(row)
    pd.DataFrame(rows).to_html(output_html, index=False)


if __name__ == '__main__':
    base_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    ref_file = os.path.join(base_dir, 'data', 'reference_data.csv')
    prod_file = os.path.join(base_dir, 'data', 'production_data.csv')

    output = run_drift_check(ref_file, prod_file, exclude_columns=['target'])
    print('Drift report:')
    for feature, metrics in output.items():
        print(feature, metrics)

    os.makedirs(os.path.join(base_dir, 'reports'), exist_ok=True)
    out_html = os.path.join(base_dir, 'reports', 'drift_report.html')
    save_drift_report(output, out_html)
    print('Saved report to', out_html)