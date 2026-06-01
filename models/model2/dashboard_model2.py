import json
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go

from src.config import REPORT_DIR


def create_dashboard():
    metrics_path = REPORT_DIR / "metrics.json"
    importance_path = REPORT_DIR / "feature_importance.csv"
    backtest_path = REPORT_DIR / "backtest.csv"

    with open(metrics_path, "r", encoding="utf-8") as f:
        metrics = json.load(f)

    importance_df = pd.read_csv(importance_path)
    backtest_df = pd.read_csv(backtest_path)

    fig_importance = px.bar(
        importance_df.head(15),
        x="importance",
        y="feature",
        orientation="h",
        title="Top 15 Feature Importance"
    )

    fig_backtest = px.line(
        backtest_df,
        x="trading_date",
        y="cumulative_return",
        title="Cumulative Return"
    )

    html = f"""
    <html>
    <head>
        <title>Model 2 Dashboard</title>
    </head>

    <body>
        <h1>Future Return Prediction Dashboard</h1>

        <h2>Model Metrics</h2>

        <ul>
            <li>MAE: {metrics["mae"]:.6f}</li>
            <li>RMSE: {metrics["rmse"]:.6f}</li>
            <li>R2: {metrics["r2_score"]:.6f}</li>
            <li>MAPE: {metrics["mape"]:.6f}</li>
        </ul>

        {fig_importance.to_html(full_html=False)}

        {fig_backtest.to_html(full_html=False)}

    </body>
    </html>
    """

    output_path = REPORT_DIR / "model2_dashboard.html"

    with open(output_path, "w", encoding="utf-8") as f:
        f.write(html)

    print(f"Dashboard saved: {output_path}")


if __name__ == "__main__":
    create_dashboard()
