import os
import pandas as pd
from monitoring.drift_detector import run_drift_check, save_drift_report


def test_run_drift_check():
    base = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
    ref = os.path.join(base, 'data', 'reference_data.csv')
    prod = os.path.join(base, 'data', 'production_data.csv')
    report = run_drift_check(ref, prod)
    assert isinstance(report, dict)
    assert 'sepal length (cm)' in report
    assert 'psi' in report['sepal length (cm)']


def test_save_drift_report(tmp_path):
    base = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
    ref = os.path.join(base, 'data', 'reference_data.csv')
    prod = os.path.join(base, 'data', 'production_data.csv')
    report = run_drift_check(ref, prod)
    out = tmp_path / 'drift.html'
    save_drift_report(report, str(out))
    assert out.exists()
