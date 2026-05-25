import html
import json
from pathlib import Path

import numpy as np
import pandas as pd


REPORTS_DIR = Path("reports")
DASHBOARD_PATH = REPORTS_DIR / "model3_dashboard.html"

METRICS_PATH = REPORTS_DIR / "metrics.json"
BACKTEST_METRICS_PATH = REPORTS_DIR / "backtest_metrics.json"
BACKTEST_PATH = REPORTS_DIR / "backtest.csv"
BACKTEST_SWEEP_PATH = REPORTS_DIR / "backtest_sweep.csv"
FEATURE_IMPORTANCE_PATH = REPORTS_DIR / "feature_importance.csv"
PREDICTIONS_PATH = REPORTS_DIR / "predictions.csv"

SIGNAL_NAMES = ["SELL", "HOLD", "BUY"]
SIGNAL_COLORS = {
    "SELL": "#d1495b",
    "HOLD": "#6c757d",
    "BUY": "#2a9d8f",
}


def load_json(path):
    if not path.exists():
        return {}
    with path.open("r", encoding="utf-8") as file:
        return json.load(file)


def read_csv(path, **kwargs):
    if not path.exists():
        return pd.DataFrame()
    return pd.read_csv(path, **kwargs)


def format_pct(value):
    if value is None or pd.isna(value):
        return "N/A"
    return f"{float(value):.2f}%"


def format_number(value, digits=2):
    if value is None or pd.isna(value):
        return "N/A"
    return f"{float(value):,.{digits}f}"


def metric_card(title, value, caption=""):
    caption_html = f"<span>{html.escape(caption)}</span>" if caption else ""
    return (
        '<section class="metric-card">'
        f"<p>{html.escape(title)}</p>"
        f"<strong>{html.escape(str(value))}</strong>"
        f"{caption_html}"
        "</section>"
    )


def make_confusion_matrix(metrics):
    matrix = np.array(metrics.get("Confusion_Matrix", []), dtype=float)
    if matrix.shape != (3, 3):
        return '<p class="empty">Confusion matrix is not available.</p>'

    max_value = matrix.max() or 1
    rows = []
    for row_index, row_name in enumerate(SIGNAL_NAMES):
        cells = [f"<th>{row_name}</th>"]
        row_total = matrix[row_index].sum() or 1
        for col_index, value in enumerate(matrix[row_index]):
            intensity = value / max_value
            recall_share = value / row_total * 100
            bg = f"rgba(42, 157, 143, {0.12 + intensity * 0.68:.3f})"
            cells.append(
                '<td style="background: {bg}">'
                '<b>{value:,.0f}</b><small>{share:.1f}% row</small>'
                "</td>".format(bg=bg, value=value, share=recall_share)
            )
        rows.append("<tr>" + "".join(cells) + "</tr>")

    header = "".join(f"<th>{name}</th>" for name in ["Actual / Pred", *SIGNAL_NAMES])
    return (
        '<table class="confusion-matrix">'
        f"<thead><tr>{header}</tr></thead>"
        f"<tbody>{''.join(rows)}</tbody>"
        "</table>"
    )


def make_bar_chart(df, label_col, value_col, color="#2a9d8f", limit=12):
    if df.empty or label_col not in df or value_col not in df:
        return '<p class="empty">No data available.</p>'

    chart_df = df[[label_col, value_col]].dropna().head(limit).copy()
    if chart_df.empty:
        return '<p class="empty">No data available.</p>'

    max_value = chart_df[value_col].max() or 1
    rows = []
    for _, row in chart_df.iterrows():
        label = html.escape(str(row[label_col]))
        value = float(row[value_col])
        width = max(2, value / max_value * 100)
        rows.append(
            '<div class="bar-row">'
            f'<span title="{label}">{label}</span>'
            '<div class="bar-track">'
            f'<div class="bar-fill" style="width:{width:.2f}%;background:{color}"></div>'
            "</div>"
            f"<b>{value:.4f}</b>"
            "</div>"
        )
    return '<div class="bar-chart">' + "".join(rows) + "</div>"


def make_line_chart(df):
    required = {"cumulative_return", "cumulative_return_net", "benchmark_cumulative_return"}
    if df.empty or not required.issubset(df.columns):
        return '<p class="empty">Backtest curve is not available.</p>'

    values = df[list(required)].apply(pd.to_numeric, errors="coerce").dropna()
    if values.empty:
        return '<p class="empty">Backtest curve is not available.</p>'

    width, height, pad = 780, 300, 34
    min_y = float(values.min().min())
    max_y = float(values.max().max())
    if min_y == max_y:
        max_y = min_y + 1

    def points_for(series):
        y_values = series.to_numpy(dtype=float)
        x_values = np.linspace(pad, width - pad, len(y_values))
        y_scaled = height - pad - ((y_values - min_y) / (max_y - min_y)) * (height - 2 * pad)
        return " ".join(f"{x:.2f},{y:.2f}" for x, y in zip(x_values, y_scaled))

    lines = [
        ("cumulative_return", "#2a9d8f", "Gross"),
        ("cumulative_return_net", "#1d3557", "Net"),
        ("benchmark_cumulative_return", "#d1495b", "Benchmark"),
    ]
    polylines = []
    legend = []
    for column, color, label in lines:
        polylines.append(
            f'<polyline points="{points_for(values[column])}" '
            f'fill="none" stroke="{color}" stroke-width="3" stroke-linejoin="round" />'
        )
        legend.append(f'<span><i style="background:{color}"></i>{label}</span>')

    y0 = height - pad
    return (
        '<div class="line-chart">'
        f'<svg viewBox="0 0 {width} {height}" role="img" aria-label="Backtest curve">'
        f'<line x1="{pad}" y1="{y0}" x2="{width - pad}" y2="{y0}" stroke="#cfd8dc" />'
        f'<line x1="{pad}" y1="{pad}" x2="{pad}" y2="{y0}" stroke="#cfd8dc" />'
        f'<text x="{pad}" y="22">{max_y:.1%}</text>'
        f'<text x="{pad}" y="{height - 8}">{min_y:.1%}</text>'
        f'{"".join(polylines)}'
        "</svg>"
        f'<div class="legend">{"".join(legend)}</div>'
        "</div>"
    )


def make_signal_distribution():
    usecols = ["target_signal", "predicted_signal"]
    predictions = read_csv(PREDICTIONS_PATH, usecols=usecols)
    if predictions.empty:
        return '<p class="empty">Prediction distribution is not available.</p>'

    rows = []
    for column, title in [("target_signal", "Actual"), ("predicted_signal", "Predicted")]:
        counts = predictions[column].value_counts().reindex(SIGNAL_NAMES, fill_value=0)
        total = counts.sum() or 1
        items = []
        for signal, count in counts.items():
            pct = count / total * 100
            color = SIGNAL_COLORS.get(signal, "#6c757d")
            items.append(
                '<div class="dist-item">'
                f'<span><i style="background:{color}"></i>{signal}</span>'
                f"<b>{count:,}</b><small>{pct:.1f}%</small>"
                "</div>"
            )
        rows.append(f'<div class="dist-card"><h3>{title}</h3>{"".join(items)}</div>')
    return '<div class="distribution-grid">' + "".join(rows) + "</div>"


def make_backtest_sweep_table():
    sweep = read_csv(BACKTEST_SWEEP_PATH)
    if sweep.empty:
        return '<p class="empty">Backtest sweep is not available.</p>'

    columns = [
        "Top_K",
        "Min_Volume",
        "Min_Close",
        "Min_Buy_Probability",
        "Cumulative_Return_Net",
        "Sharpe_Ratio_Net",
        "Trade_Days",
    ]
    table_df = sweep[[col for col in columns if col in sweep.columns]].head(8)
    if table_df.empty:
        return '<p class="empty">Backtest sweep is not available.</p>'

    head = "".join(f"<th>{html.escape(col)}</th>" for col in table_df.columns)
    body_rows = []
    for _, row in table_df.iterrows():
        cells = []
        for col in table_df.columns:
            value = row[col]
            if "Return" in col:
                value = f"{float(value):.2%}"
            elif isinstance(value, float):
                value = f"{value:.3f}"
            cells.append(f"<td>{html.escape(str(value))}</td>")
        body_rows.append("<tr>" + "".join(cells) + "</tr>")
    return f'<table class="data-table"><thead><tr>{head}</tr></thead><tbody>{"".join(body_rows)}</tbody></table>'


def build_dashboard_html():
    metrics = load_json(METRICS_PATH)
    backtest_metrics = load_json(BACKTEST_METRICS_PATH)
    feature_importance = read_csv(FEATURE_IMPORTANCE_PATH)
    backtest = read_csv(BACKTEST_PATH)

    cards = [
        metric_card(
            "Accuracy",
            format_pct(metrics.get("Accuracy")),
            f"Baseline: {format_pct(metrics.get('Baseline_Accuracy'))}",
        ),
        metric_card("Macro F1", format_number(metrics.get("Macro_F1"), 4)),
        metric_card(
            "Net Return",
            f"{float(backtest_metrics.get('Cumulative_Return_Net', 0)):.2%}",
            "After transaction cost and slippage",
        ),
        metric_card(
            "Net Sharpe",
            format_number(backtest_metrics.get("Sharpe_Ratio_Net"), 3),
            f"Trade days: {backtest_metrics.get('Trade_Days', 'N/A')}",
        ),
    ]

    return f"""<!doctype html>
<html lang="en">
<head>
  <meta charset="utf-8">
  <meta name="viewport" content="width=device-width, initial-scale=1">
  <title>Model 3 Trading Signal Dashboard</title>
  <style>
    :root {{
      --ink: #18202a;
      --muted: #5f6c7b;
      --line: #d9e1e8;
      --bg: #f5f7f9;
      --panel: #ffffff;
      --accent: #2a9d8f;
    }}
    * {{ box-sizing: border-box; }}
    body {{
      margin: 0;
      background: var(--bg);
      color: var(--ink);
      font-family: Arial, Helvetica, sans-serif;
      line-height: 1.45;
    }}
    header {{
      padding: 28px clamp(18px, 4vw, 44px) 18px;
      border-bottom: 1px solid var(--line);
      background: var(--panel);
    }}
    header h1 {{ margin: 0; font-size: 28px; letter-spacing: 0; }}
    header p {{ margin: 6px 0 0; color: var(--muted); }}
    main {{ padding: 22px clamp(18px, 4vw, 44px) 40px; }}
    .metric-grid {{
      display: grid;
      grid-template-columns: repeat(4, minmax(160px, 1fr));
      gap: 12px;
    }}
    .metric-card, .panel {{
      background: var(--panel);
      border: 1px solid var(--line);
      border-radius: 8px;
    }}
    .metric-card {{ padding: 16px; min-height: 116px; }}
    .metric-card p {{ margin: 0 0 10px; color: var(--muted); font-size: 13px; }}
    .metric-card strong {{ display: block; font-size: 28px; }}
    .metric-card span {{ display: block; margin-top: 8px; color: var(--muted); font-size: 12px; }}
    .grid-2 {{
      display: grid;
      grid-template-columns: minmax(0, 1.25fr) minmax(280px, 0.75fr);
      gap: 14px;
      margin-top: 14px;
    }}
    .panel {{ padding: 18px; overflow: hidden; }}
    .panel h2 {{ margin: 0 0 14px; font-size: 18px; }}
    .confusion-matrix, .data-table {{ width: 100%; border-collapse: collapse; }}
    th, td {{ border-bottom: 1px solid var(--line); padding: 10px; text-align: left; }}
    th {{ color: var(--muted); font-size: 12px; font-weight: 700; }}
    .confusion-matrix td {{ text-align: center; border: 1px solid #fff; }}
    .confusion-matrix b, .confusion-matrix small {{ display: block; }}
    .confusion-matrix small {{ color: var(--muted); margin-top: 3px; }}
    .bar-row {{
      display: grid;
      grid-template-columns: minmax(92px, 150px) 1fr 72px;
      gap: 10px;
      align-items: center;
      margin: 9px 0;
      font-size: 13px;
    }}
    .bar-row span {{ overflow: hidden; text-overflow: ellipsis; white-space: nowrap; }}
    .bar-track {{ height: 12px; background: #e8eef2; border-radius: 999px; overflow: hidden; }}
    .bar-fill {{ height: 100%; border-radius: 999px; }}
    .line-chart svg {{ width: 100%; height: auto; display: block; }}
    .line-chart text {{ fill: var(--muted); font-size: 12px; }}
    .legend {{ display: flex; flex-wrap: wrap; gap: 14px; color: var(--muted); font-size: 13px; }}
    .legend i, .dist-item i {{
      display: inline-block;
      width: 10px;
      height: 10px;
      margin-right: 6px;
      border-radius: 50%;
    }}
    .distribution-grid {{ display: grid; grid-template-columns: repeat(2, 1fr); gap: 12px; }}
    .dist-card h3 {{ margin: 0 0 8px; font-size: 15px; }}
    .dist-item {{
      display: grid;
      grid-template-columns: 1fr auto auto;
      gap: 8px;
      padding: 8px 0;
      border-bottom: 1px solid var(--line);
      align-items: center;
    }}
    .dist-item small {{ color: var(--muted); }}
    .empty {{ color: var(--muted); margin: 0; }}
    @media (max-width: 900px) {{
      .metric-grid, .grid-2 {{ grid-template-columns: 1fr; }}
      .distribution-grid {{ grid-template-columns: 1fr; }}
      .bar-row {{ grid-template-columns: 96px 1fr 56px; }}
    }}
  </style>
</head>
<body>
  <header>
    <h1>Model 3 Trading Signal Dashboard</h1>
    <p>XGBoost classifier for SELL / HOLD / BUY signals, generated from the latest files in reports/.</p>
  </header>
  <main>
    <section class="metric-grid">{"".join(cards)}</section>
    <section class="grid-2">
      <div class="panel">
        <h2>Backtest Curve</h2>
        {make_line_chart(backtest)}
      </div>
      <div class="panel">
        <h2>Signal Distribution</h2>
        {make_signal_distribution()}
      </div>
    </section>
    <section class="grid-2">
      <div class="panel">
        <h2>Confusion Matrix</h2>
        {make_confusion_matrix(metrics)}
      </div>
      <div class="panel">
        <h2>Top Feature Importance</h2>
        {make_bar_chart(feature_importance, "feature", "importance")}
      </div>
    </section>
    <section class="panel" style="margin-top:14px">
      <h2>Best Backtest Configurations</h2>
      {make_backtest_sweep_table()}
    </section>
  </main>
</body>
</html>
"""


def main():
    REPORTS_DIR.mkdir(parents=True, exist_ok=True)
    DASHBOARD_PATH.write_text(build_dashboard_html(), encoding="utf-8")
    print(f"Dashboard saved to {DASHBOARD_PATH}")


if __name__ == "__main__":
    main()
