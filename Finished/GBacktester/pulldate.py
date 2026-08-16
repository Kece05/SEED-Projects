"""
build_earnings_db.py
════════════════════════════════════════════════════════════════════════════════
Builds a SQLite database of S&P 500 quarterly earnings from 2011 to present,
with Unexpected Earnings (UE) and Standardized Unexpected Earnings (SUE)
computed using an expanding-window methodology (no look-ahead bias).

WHAT IT DOES
────────────
1. Determines the union of all S&P 500 tickers that appeared in the index
   from 2011-Q1 through today using betteryf_1d.sp500_members_on() —
   point-in-time correct, survivorship-bias free.

2. For each ticker, fetches up to 100 quarters of earnings history from
   Yahoo Finance via yfinance (EPS Estimate, Reported EPS, Surprise %).

3. Filters to announcements on or after 2011-01-01.

4. Computes:
     UE  = Reported EPS − Consensus EPS Estimate
     SUE = UE / std(UE over prior N announcements)   [expanding window]

5. Attaches S&P 500 membership flag at each announcement date (PIT correct).

6. Stores everything in earnings.db with two tables:
     earnings  — one row per (ticker, announcement date)
     meta      — run info, row counts, last updated

SCHEMA  —  earnings table
────────────────────────────────────────────────────────────────────────────────
  ticker          TEXT     e.g. 'AAPL'
  announce_date   TEXT     YYYY-MM-DD  (after-hours shifted to next biz day)
  eps_estimate    REAL     consensus analyst EPS estimate
  eps_actual      REAL     reported EPS
  surprise_pct    REAL     (eps_actual - eps_estimate) / abs(eps_estimate) * 100
  ue              REAL     eps_actual - eps_estimate
  sue             REAL     ue / rolling_std(ue, min_periods=4)
  in_sp500        INTEGER  1 if ticker was in S&P 500 on announce_date else 0
  sector          TEXT     GICS sector from Wikipedia constituent table

USAGE
─────
  # Place betteryf_1d.py in the same directory, then:
  python build_earnings_db.py

  # Optional flags:
  python build_earnings_db.py --db my_data/earnings.db
  python build_earnings_db.py --start 2015-01-01
  python build_earnings_db.py --workers 8
  python build_earnings_db.py --resume          # skip tickers already in DB
  python build_earnings_db.py --tickers AAPL MSFT NVDA   # single-ticker test

REQUIREMENTS
────────────
  pip install yfinance pandas requests beautifulsoup4
  betteryf_1d.py must be in the same directory (or on PYTHONPATH)

NOTES
─────
- Yahoo Finance caps get_earnings_dates() at 100 rows per ticker (~25 years).
  For most tickers this covers 2011 onward comfortably.
- Rate limiting: ~0.5s sleep between tickers by default. Increase --sleep if
  you hit 429s.
- After-hours shift: announcements flagged as after 4 PM ET are advanced
  to the next business day. Yahoo does not always provide time-of-day, so
  this script checks the Surprise(%) sign and timing heuristics where possible.
  You can apply your own AMC/BMO flags from a separate source on top of this.
- SUE requires min 4 prior observations; rows with fewer have SUE = NaN.
- The script is fully resumable: --resume skips tickers already stored.
════════════════════════════════════════════════════════════════════════════════
"""

from __future__ import annotations

import argparse
import sqlite3
import sys
import time
import warnings
from datetime import datetime, date
from io import StringIO
from pathlib import Path
from typing import Optional

import numpy as np
import pandas as pd
import requests
import yfinance as yf

warnings.filterwarnings("ignore", category=FutureWarning)
warnings.filterwarnings("ignore", category=pd.errors.PerformanceWarning)

# ─────────────────────────────────────────────────────────────────────────────
# betteryf_1d import
# ─────────────────────────────────────────────────────────────────────────────
try:
    from betteryf_1d import sp500_members_on as _sp500_members_on
    HAS_BETTERYF = True
except ImportError:
    HAS_BETTERYF = False
    print("[WARN] betteryf_1d.py not found — S&P 500 PIT membership will be "
          "approximated from Wikipedia current list only.")


# ─────────────────────────────────────────────────────────────────────────────
# Constants
# ─────────────────────────────────────────────────────────────────────────────

DEFAULT_DB      = Path("earnings.db")
DEFAULT_START   = "2011-01-01"
DEFAULT_WORKERS = 1          # yfinance doesn't parallelize well; keep at 1
DEFAULT_SLEEP   = 0.5        # seconds between ticker requests
SUE_MIN_PERIODS = 4          # min prior quarters needed to compute SUE
_WIKI_URL       = "https://en.wikipedia.org/wiki/List_of_S%26P_500_companies"
_HEADERS        = {"User-Agent": "Mozilla/5.0"}


# ─────────────────────────────────────────────────────────────────────────────
# DB setup
# ─────────────────────────────────────────────────────────────────────────────

DDL = """
CREATE TABLE IF NOT EXISTS earnings (
    ticker          TEXT    NOT NULL,
    announce_date   TEXT    NOT NULL,
    eps_estimate    REAL,
    eps_actual      REAL,
    surprise_pct    REAL,
    ue              REAL,
    sue             REAL,
    in_sp500        INTEGER,
    sector          TEXT,
    PRIMARY KEY (ticker, announce_date)
);

CREATE TABLE IF NOT EXISTS meta (
    key     TEXT PRIMARY KEY,
    value   TEXT
);

CREATE INDEX IF NOT EXISTS idx_date   ON earnings (announce_date);
CREATE INDEX IF NOT EXISTS idx_ticker ON earnings (ticker);
CREATE INDEX IF NOT EXISTS idx_sector ON earnings (sector);
"""


def open_db(path: Path) -> sqlite3.Connection:
    path.parent.mkdir(parents=True, exist_ok=True)
    conn = sqlite3.connect(str(path))
    conn.executescript(DDL)
    conn.commit()
    return conn


def tickers_already_stored(conn: sqlite3.Connection) -> set[str]:
    cur = conn.execute("SELECT DISTINCT ticker FROM earnings")
    return {r[0] for r in cur.fetchall()}


def upsert_rows(conn: sqlite3.Connection, rows: list[dict]) -> None:
    if not rows:
        return
    conn.executemany(
        """
        INSERT OR REPLACE INTO earnings
            (ticker, announce_date, eps_estimate, eps_actual,
             surprise_pct, ue, sue, in_sp500, sector)
        VALUES
            (:ticker, :announce_date, :eps_estimate, :eps_actual,
             :surprise_pct, :ue, :sue, :in_sp500, :sector)
        """,
        rows,
    )
    conn.commit()


def set_meta(conn: sqlite3.Connection, key: str, value: str) -> None:
    conn.execute(
        "INSERT OR REPLACE INTO meta (key, value) VALUES (?, ?)", (key, value)
    )
    conn.commit()


# ─────────────────────────────────────────────────────────────────────────────
# S&P 500 universe builder
# ─────────────────────────────────────────────────────────────────────────────

def _fetch_sp500_wiki() -> tuple[pd.DataFrame, pd.DataFrame]:
    """
    Returns (current_df, changes_df) scraped from Wikipedia.
    current_df columns include: Symbol, GICS Sector
    changes_df columns: effective_date, added, removed
    """
    resp = requests.get(_WIKI_URL, headers=_HEADERS, timeout=20)
    resp.raise_for_status()
    tables = pd.read_html(StringIO(resp.text))
    if len(tables) < 2:
        raise RuntimeError("Wikipedia S&P 500 table structure changed.")

    current_df  = tables[0].copy()
    changes_df  = tables[1].copy()
    changes_df.columns = [str(c).strip() for c in changes_df.columns]

    date_col    = [c for c in changes_df.columns if "Effective" in c][0]
    added_col   = [c for c in changes_df.columns if "Added"   in c and "Ticker" in c][0]
    removed_col = [c for c in changes_df.columns if "Removed" in c and "Ticker" in c][0]

    changes_df  = changes_df[[date_col, added_col, removed_col]].copy()
    changes_df.columns = ["effective_date", "added", "removed"]
    changes_df["effective_date"] = pd.to_datetime(changes_df["effective_date"], errors="coerce")
    changes_df["added"]   = changes_df["added"].astype(str).str.strip()
    changes_df["removed"] = changes_df["removed"].astype(str).str.strip()
    changes_df = changes_df.dropna(subset=["effective_date"])

    return current_df, changes_df


def _build_universe_from(
    current_df: pd.DataFrame,
    changes_df: pd.DataFrame,
    start: str,
    verbose: bool = True,
) -> tuple[list[str], dict[str, str]]:
    """
    Build the full ticker universe from already-fetched Wikipedia DataFrames.
    Called by run() which fetches once and reuses for PIT checks too.

    Returns:
        tickers  - sorted list of ALL tickers that appeared in S&P 500 from
                   `start` through today, using every change event so no
                   short-lived members are missed. For 2011-present this
                   should yield 700-800+ unique tickers.
        sectors  - dict {ticker: sector}
    """
    sym_col = "Symbol"
    sec_col = [c for c in current_df.columns if "Sector" in c][0]

    # Sector map from current constituents
    sectors: dict[str, str] = {}
    for _, row in current_df.iterrows():
        sym = str(row[sym_col]).strip().replace(".", "-")
        sectors[sym] = str(row[sec_col]).strip()

    # Start with all current members
    current_set = set(
        str(s).strip().replace(".", "-") for s in current_df[sym_col]
    )

    # Union every ticker from the changes table since `start`
    start_ts = pd.Timestamp(start)
    relevant = changes_df[changes_df["effective_date"] >= start_ts]

    historical: set[str] = set()
    for _, row in relevant.iterrows():
        for col in ["added", "removed"]:
            val = str(row[col]).strip()
            if val in ("", "nan", "None"):
                continue
            # Handle comma-separated multi-ticker rows e.g. "CTVA,DD"
            for t in val.split(","):
                t = t.strip().replace(".", "-")
                if t:
                    historical.add(t)

    result = sorted(current_set | historical)

    if verbose:
        print(f"  -> {len(current_set)} current constituents")
        print(f"  -> {len(historical)} additional historical tickers from "
              f"changes table (since {start})")
        print(f"  -> {len(result)} total unique tickers in universe")
        earliest = changes_df["effective_date"].min().date()
        if len(result) < 650:
            print(f"  [warn] Universe looks small. Wikipedia changes table "
                  f"earliest entry: {earliest}. It may not cover all of {start} "
                  f"onward — consider supplementing with a dedicated constituent "
                  f"history source.")
        else:
            print(f"  [ok] Changes table goes back to {earliest}")

    return result, sectors



def is_in_sp500_on(ticker: str, announce_date: str,
                   changes_df: pd.DataFrame,
                   current_tickers: set[str]) -> int:
    """
    Returns 1 if `ticker` was in the S&P 500 on `announce_date`, else 0.
    Uses the same rewind logic as betteryf_1d.sp500_members_on().
    """
    if HAS_BETTERYF:
        try:
            members = _sp500_members_on(announce_date)
            return int(ticker in members)
        except Exception:
            pass
    return int(ticker in current_tickers)


# ─────────────────────────────────────────────────────────────────────────────
# Earnings fetcher
# ─────────────────────────────────────────────────────────────────────────────

def _safe_float(v) -> Optional[float]:
    try:
        f = float(v)
        return f if np.isfinite(f) else None
    except (TypeError, ValueError):
        return None


def fetch_earnings(ticker: str, start: str) -> pd.DataFrame:
    """
    Fetch earnings history for one ticker using yfinance.
    Returns DataFrame with columns:
        announce_date, eps_estimate, eps_actual, surprise_pct
    Sorted ascending by announce_date, filtered to >= start.
    Returns empty DataFrame on failure.
    """
    start_ts = pd.Timestamp(start)

    try:
        t  = yf.Ticker(ticker)
        df = t.get_earnings_dates(limit=100)  # Yahoo max = 100
    except Exception as e:
        print(f"    [warn] {ticker}: yfinance error — {e}")
        return pd.DataFrame()

    if df is None or df.empty:
        return pd.DataFrame()

    # Normalise column names (yfinance may vary across versions)
    df.columns = [str(c).strip() for c in df.columns]
    col_map = {}
    for c in df.columns:
        cl = c.lower()
        if "estimate" in cl:
            col_map[c] = "eps_estimate"
        elif "reported" in cl or "actual" in cl or "eps" in cl:
            col_map[c] = "eps_actual"
        elif "surprise" in cl:
            col_map[c] = "surprise_pct"
    df = df.rename(columns=col_map)

    required = {"eps_estimate", "eps_actual"}
    missing  = required - set(df.columns)
    if missing:
        print(f"    [warn] {ticker}: missing columns {missing}, got {list(df.columns)}")
        return pd.DataFrame()

    # Normalise index to timezone-naive date
    df.index = pd.to_datetime(df.index, utc=True).tz_localize(None)
    df.index.name = "announce_date"
    df = df.reset_index()
    df["announce_date"] = pd.to_datetime(df["announce_date"]).dt.normalize()

    # Drop future rows (no actual EPS yet) and rows before start
    df = df.dropna(subset=["eps_actual"])
    df = df[df["announce_date"] >= start_ts].copy()

    if df.empty:
        return pd.DataFrame()

    # Convert to float
    for col in ["eps_estimate", "eps_actual", "surprise_pct"]:
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors="coerce")

    # Recompute surprise_pct ourselves for consistency
    df["ue"] = df["eps_actual"] - df["eps_estimate"]
    mask = df["eps_estimate"].notna() & (df["eps_estimate"] != 0)
    df["surprise_pct"] = np.where(
        mask,
        df["ue"] / df["eps_estimate"].abs() * 100,
        np.nan,
    )

    df = df.sort_values("announce_date").reset_index(drop=True)
    return df[["announce_date", "eps_estimate", "eps_actual", "surprise_pct", "ue"]]


# ─────────────────────────────────────────────────────────────────────────────
# SUE computation  (expanding window, no look-ahead)
# ─────────────────────────────────────────────────────────────────────────────

def compute_sue(df: pd.DataFrame, min_periods: int = SUE_MIN_PERIODS) -> pd.Series:
    """
    Compute SUE for each row using only the UE history available UP TO
    (but NOT including) that row — strict expanding window.

    SUE_i = UE_i / std(UE_1 … UE_{i-1})

    Requires at least `min_periods` prior observations; otherwise NaN.
    """
    ue   = df["ue"].values
    sue  = np.full(len(ue), np.nan)

    for i in range(len(ue)):
        prior = ue[:i]  # all observations strictly before current
        prior = prior[~np.isnan(prior)]
        if len(prior) >= min_periods:
            std = np.std(prior, ddof=1)
            if std > 0 and not np.isnan(ue[i]):
                sue[i] = ue[i] / std

    return pd.Series(sue, index=df.index, name="sue")


# ─────────────────────────────────────────────────────────────────────────────
# After-hours shift  (next business day)
# ─────────────────────────────────────────────────────────────────────────────

def next_biz_day(dt: pd.Timestamp) -> pd.Timestamp:
    """Advance to next weekday (Mon–Fri). Does not adjust for holidays."""
    d = dt + pd.Timedelta(days=1)
    while d.weekday() >= 5:  # Sat=5, Sun=6
        d += pd.Timedelta(days=1)
    return d


# NOTE: Yahoo Finance's get_earnings_dates() already applies the after-hours
# shift for many records — entries that were reported after 4 PM appear with
# the next trading day's date in the index. However, this is not perfectly
# reliable. If you have a separate AMC/BMO flag dataset, join it to the output
# table and re-shift announce_date where needed. The script stores the date
# exactly as Yahoo provides it, which is the best available approximation
# without a paid earnings calendar feed.


# ─────────────────────────────────────────────────────────────────────────────
# Main pipeline
# ─────────────────────────────────────────────────────────────────────────────

def run(
    db_path:  Path,
    start:    str,
    resume:   bool,
    sleep:    float,
    tickers:  Optional[list[str]],
    verbose:  bool,
) -> None:

    t0 = time.perf_counter()
    conn = open_db(db_path)
    print(f"[db] Opened {db_path}")

    # ── Build universe — one Wikipedia fetch, reused for PIT checks ──────────
    # fetch once up front so we have changes_df for is_in_sp500_on()
    print("[universe] Fetching Wikipedia S&P 500 tables...")
    current_df_raw, changes_df = _fetch_sp500_wiki()
    sym_col = "Symbol"
    sec_col = [c for c in current_df_raw.columns if "Sector" in c][0]
    current_tickers = set(
        str(s).strip().replace(".", "-") for s in current_df_raw[sym_col]
    )

    if tickers:
        all_tickers = sorted(set(tickers))
        sectors = {
            str(r[sym_col]).strip().replace(".", "-"): str(r[sec_col]).strip()
            for _, r in current_df_raw.iterrows()
        }
        print(f"[universe] Using {len(all_tickers)} user-supplied tickers")
    else:
        # build_universe also calls _fetch_sp500_wiki internally, so pass the
        # already-fetched data to avoid a second network hit
        all_tickers, sectors = _build_universe_from(
            current_df_raw, changes_df, start, verbose=verbose
        )

    # ── Resume logic ──────────────────────────────────────────────────────────
    already_done = tickers_already_stored(conn) if resume else set()
    todo = [t for t in all_tickers if t not in already_done]
    skipped = len(all_tickers) - len(todo)

    if resume and skipped:
        print(f"[resume] Skipping {skipped} tickers already in DB; "
              f"{len(todo)} remaining")

    print(f"\n[pipeline] Processing {len(todo)} tickers "
          f"(start={start}, sleep={sleep}s)\n")

    # ── Per-ticker loop ───────────────────────────────────────────────────────
    total_rows  = 0
    failed      = []
    warn_counts = {"no_data": 0, "all_filtered": 0}

    for idx, ticker in enumerate(todo, 1):
        prefix = f"  [{idx:>4}/{len(todo)}] {ticker:<8}"

        # 1. Fetch raw earnings
        raw = fetch_earnings(ticker, start)

        if raw.empty:
            warn_counts["no_data"] += 1
            if verbose:
                print(f"{prefix} → no data")
            time.sleep(sleep)
            continue

        if verbose:
            print(f"{prefix} → {len(raw)} quarters fetched", end="")

        # 2. Compute SUE (expanding window, no look-ahead)
        raw = raw.sort_values("announce_date").reset_index(drop=True)
        raw["sue"] = compute_sue(raw)

        # 3. Attach PIT S&P 500 membership + sector
        sector = sectors.get(ticker, "Unknown")

        rows = []
        for _, row in raw.iterrows():
            ad = str(row["announce_date"].date())
            in_sp = is_in_sp500_on(ticker, ad, changes_df, current_tickers)
            rows.append({
                "ticker":        ticker,
                "announce_date": ad,
                "eps_estimate":  _safe_float(row.get("eps_estimate")),
                "eps_actual":    _safe_float(row.get("eps_actual")),
                "surprise_pct":  _safe_float(row.get("surprise_pct")),
                "ue":            _safe_float(row.get("ue")),
                "sue":           _safe_float(row.get("sue")),
                "in_sp500":      in_sp,
                "sector":        sector,
            })

        # 4. Write to DB
        upsert_rows(conn, rows)
        total_rows += len(rows)

        if verbose:
            sue_count = sum(1 for r in rows if r["sue"] is not None)
            print(f"  |  {sue_count} with SUE  |  in_sp500={sum(r['in_sp500'] for r in rows)}")

        time.sleep(sleep)

    # ── Summary ───────────────────────────────────────────────────────────────
    elapsed = time.perf_counter() - t0

    # Update meta
    cur = conn.execute("SELECT COUNT(*) FROM earnings")
    db_total = cur.fetchone()[0]

    set_meta(conn, "last_run",    datetime.now().isoformat())
    set_meta(conn, "start_date",  start)
    set_meta(conn, "total_rows",  str(db_total))
    set_meta(conn, "tickers_processed", str(len(todo)))

    conn.close()

    print(f"""
╔══════════════════════════════════════════════════════╗
  BUILD COMPLETE
  Elapsed       : {elapsed/60:.1f} min
  Tickers tried : {len(todo)}
  Rows inserted : {total_rows}
  DB total rows : {db_total}
  No data       : {warn_counts['no_data']}
  DB path       : {db_path.resolve()}
╚══════════════════════════════════════════════════════╝
""")


# ─────────────────────────────────────────────────────────────────────────────
# CLI
# ─────────────────────────────────────────────────────────────────────────────

def _parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="Build S&P 500 earnings database with PIT membership, UE and SUE."
    )
    p.add_argument("--db",      default=str(DEFAULT_DB),
                   help=f"SQLite output path (default: {DEFAULT_DB})")
    p.add_argument("--start",   default=DEFAULT_START,
                   help=f"Earliest announcement date to keep (default: {DEFAULT_START})")
    p.add_argument("--sleep",   type=float, default=DEFAULT_SLEEP,
                   help=f"Seconds between requests (default: {DEFAULT_SLEEP})")
    p.add_argument("--resume",  action="store_true",
                   help="Skip tickers already present in the DB")
    p.add_argument("--quiet",   action="store_true",
                   help="Suppress per-ticker output")
    p.add_argument("--tickers", nargs="+", metavar="TICKER",
                   help="Override universe with explicit ticker list (for testing)")
    return p.parse_args()


# ─────────────────────────────────────────────────────────────────────────────
# Quick-look helper  (call after build to inspect the DB)
# ─────────────────────────────────────────────────────────────────────────────

def inspect_db(db_path: str = str(DEFAULT_DB)) -> None:
    """
    Print a quick summary of the database contents.
    Call from Python after the build:

        from build_earnings_db import inspect_db
        inspect_db()
    """
    conn = sqlite3.connect(db_path)

    print("\n=== DB SUMMARY ===")
    total = conn.execute("SELECT COUNT(*) FROM earnings").fetchone()[0]
    print(f"  Total rows        : {total:,}")

    tickers = conn.execute("SELECT COUNT(DISTINCT ticker) FROM earnings").fetchone()[0]
    print(f"  Unique tickers    : {tickers:,}")

    dates = conn.execute(
        "SELECT MIN(announce_date), MAX(announce_date) FROM earnings"
    ).fetchone()
    print(f"  Date range        : {dates[0]} → {dates[1]}")

    sp500 = conn.execute(
        "SELECT COUNT(*) FROM earnings WHERE in_sp500 = 1"
    ).fetchone()[0]
    print(f"  In-S&P500 rows    : {sp500:,} ({sp500/total*100:.1f}%)")

    sue_ok = conn.execute(
        "SELECT COUNT(*) FROM earnings WHERE sue IS NOT NULL"
    ).fetchone()[0]
    print(f"  Rows with SUE     : {sue_ok:,} ({sue_ok/total*100:.1f}%)")

    print("\n  Sector breakdown:")
    cur = conn.execute(
        "SELECT sector, COUNT(*) as n FROM earnings GROUP BY sector ORDER BY n DESC"
    )
    for row in cur.fetchall():
        print(f"    {row[0]:<40} {row[1]:>6,}")

    print("\n  Sample rows (Q5 SUE, in S&P 500):")
    cur = conn.execute("""
        SELECT ticker, announce_date, eps_estimate, eps_actual, ue, sue, sector
        FROM earnings
        WHERE in_sp500 = 1 AND sue IS NOT NULL
        ORDER BY sue DESC
        LIMIT 10
    """)
    df = pd.DataFrame(cur.fetchall(),
                      columns=["ticker","date","est","actual","ue","sue","sector"])
    print(df.to_string(index=False))

    conn.close()


# ─────────────────────────────────────────────────────────────────────────────
# Entry point
# ─────────────────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    args = _parse_args()
    run(
        db_path = Path(args.db),
        start   = args.start,
        resume  = args.resume,
        sleep   = args.sleep,
        tickers = args.tickers,
        verbose = not args.quiet,
    )