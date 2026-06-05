# Model1 Marts Insights Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Build dashboard-ready Model 1 marts and daily insights from existing Model 1 reports, with optional ClickHouse upload.

**Architecture:** Add a focused mart builder module under `models/model1/src` that transforms existing reports into normalized mart DataFrames. Add a CLI script that writes mart CSV outputs locally by default and uploads to ClickHouse only when requested.

**Tech Stack:** Python, pandas, clickhouse-connect, unittest.

---

### Task 1: Mart Builder

**Files:**
- Create: `models/model1/src/marts.py`
- Test: `models/model1/tests/test_marts.py`

- [ ] Write tests for building price forecast, top expected return, backtest daily, metrics, and daily insight marts from sample report data.
- [ ] Implement pure transformation functions with stable columns and `created_at` / `model_run_id`.
- [ ] Verify tests pass.

### Task 2: CLI

**Files:**
- Create: `models/model1/generate_marts.py`
- Test: `models/model1/tests/test_generate_marts.py`

- [ ] Write tests for creating configured local mart output paths.
- [ ] Implement CLI with local CSV output by default and `--upload-clickhouse` opt-in.
- [ ] Verify tests pass.

### Task 3: Verification

**Files:**
- Modify if needed: `.gitignore`

- [ ] Run Model 1 unit tests.
- [ ] Run `python generate_marts.py` from `models/model1`.
- [ ] Confirm mart CSV files are written under `models/model1/reports/marts`.
- [ ] Confirm ClickHouse upload path is explicit opt-in.
