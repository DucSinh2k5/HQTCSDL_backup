import numpy as np

from .config import REPORT_DIR
from .utils import save_json


def run_backtest(prediction_df, top_n=10):
    df = prediction_df.copy()

    backtest_rows = []

    for date, group in df.groupby("trading_date"):
        selected = group.sort_values(
            "predicted_future_return_5d",
            ascending=False
        ).head(top_n)

        avg_actual_return = selected["actual_future_return_5d"].mean()
        avg_predicted_return = selected["predicted_future_return_5d"].mean()
        win_rate = (selected["actual_future_return_5d"] > 0).mean()

        backtest_rows.append({
            "trading_date": date,
            "top_n": top_n,
            "avg_predicted_return": avg_predicted_return,
            "avg_actual_return": avg_actual_return,
            "win_rate": win_rate
        })

    import pandas as pd
    backtest_df = pd.DataFrame(backtest_rows)
    backtest_df["cumulative_return"] = (
        1 + backtest_df["avg_actual_return"]
    ).cumprod() - 1

    backtest_df.to_csv(
        REPORT_DIR / "backtest.csv",
        index=False,
        encoding="utf-8-sig"
    )

    metrics = calculate_backtest_metrics(backtest_df)
    save_json(metrics, REPORT_DIR / "backtest_metrics.json")

    return backtest_df, metrics


def calculate_backtest_metrics(backtest_df):
    returns = backtest_df["avg_actual_return"]

    avg_return = returns.mean()
    win_rate = (returns > 0).mean()

    cumulative = (1 + returns).cumprod()
    peak = cumulative.cummax()
    drawdown = cumulative / peak - 1
    max_drawdown = drawdown.min()

    sharpe_ratio = returns.mean() / returns.std() if returns.std() != 0 else 0

    return {
        "avg_return": float(avg_return),
        "win_rate": float(win_rate),
        "max_drawdown": float(max_drawdown),
        "sharpe_ratio": float(sharpe_ratio)
    }


def run_backtest_sweep(prediction_df, top_n_values=(5, 10, 20)):
    rows = []

    for top_n in top_n_values:
        backtest_df, metrics = run_backtest(prediction_df, top_n=top_n)

        rows.append({
            "top_n": top_n,
            "avg_return": metrics["avg_return"],
            "win_rate": metrics["win_rate"],
            "max_drawdown": metrics["max_drawdown"],
            "sharpe_ratio": metrics["sharpe_ratio"]
        })

    import pandas as pd
    sweep_df = pd.DataFrame(rows)

    sweep_df.to_csv(
        REPORT_DIR / "backtest_sweep.csv",
        index=False,
        encoding="utf-8-sig"
    )

    return sweep_df