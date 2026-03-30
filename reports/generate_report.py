import os
import sys
import pandas as pd

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, BASE_DIR)

from monitoring.drift_detector import run_drift_check
from monitoring.performance_tracker import compute_metrics


def generate_evidently_report(ref_path, prod_path, output_path):
    """Generate a data-drift report using Evidently, with a plain-HTML fallback."""
    ref = pd.read_csv(ref_path)
    prod = pd.read_csv(prod_path)
    os.makedirs(os.path.dirname(output_path), exist_ok=True)

    try:
        from evidently import Report
        from evidently.presets import DataDriftPreset, DataSummaryPreset

        report = Report([DataDriftPreset(), DataSummaryPreset()])
        snapshot = report.run(current_data=prod, reference_data=ref)
        snapshot.save_html(output_path)
        print(f"Evidently report saved: {output_path}")
    except ImportError:
        print("evidently not installed — generating fallback HTML report.")
        _generate_fallback_html(ref_path, prod_path, output_path)

    return output_path


def _generate_fallback_html(ref_path, prod_path, output_path):
    report = run_drift_check(ref_path, prod_path, exclude_columns=["target"])
    rows = [{"feature": feat, **metrics} for feat, metrics in report.items()]
    drift_df = pd.DataFrame(rows)

    html = f"""<!DOCTYPE html>
<html lang="en">
<head>
  <meta charset="UTF-8">
  <title>ML Monitoring Report</title>
  <style>
    body {{ font-family: Arial, sans-serif; margin: 2rem; color: #222; }}
    h1, h2 {{ color: #2c3e50; }}
    table {{ border-collapse: collapse; width: 100%; margin-top: 1rem; }}
    th {{ background: #2c3e50; color: #fff; padding: 8px 12px; text-align: left; }}
    td {{ padding: 8px 12px; border-bottom: 1px solid #ddd; }}
    tr.drift-yes td {{ background: #fff0f0; }}
    tr:hover td {{ background: #f5f5f5; }}
  </style>
</head>
<body>
  <h1>ML Monitoring Report</h1>
  <h2>Data Drift</h2>
  {_styled_table(drift_df)}
</body>
</html>"""

    with open(output_path, "w", encoding="utf-8") as fh:
        fh.write(html)
    print(f"Fallback HTML report saved: {output_path}")


def _styled_table(df: pd.DataFrame) -> str:
    headers = "".join(f"<th>{c}</th>" for c in df.columns)
    rows_html = ""
    for _, row in df.iterrows():
        drift = str(row.get("drift_detected", False)).lower()
        cls = "drift-yes" if drift == "true" else ""
        cells = "".join(f"<td>{v}</td>" for v in row.values)
        rows_html += f'<tr class="{cls}">{cells}</tr>\n'
    return f"<table><thead><tr>{headers}</tr></thead><tbody>{rows_html}</tbody></table>"


if __name__ == "__main__":
    ref = os.path.join(BASE_DIR, "data", "reference_data.csv")
    prod = os.path.join(BASE_DIR, "data", "production_data.csv")
    out = os.path.join(BASE_DIR, "reports", "monitoring_report.html")
    generate_evidently_report(ref, prod, out)
