"""
update.py
════════════════════════════════════════════════════════════════════════════════
Gap-fill script for earnings.db

WHAT IT DOES
────────────
For every ticker already in earnings.db:

  1. Determine the ticker's S&P 500 membership window (first date in → last
     date out) by sampling quarterly from 2011-01-01 → today via
     betteryf_1d.sp500_members_on().

  2. Set required_start = first_membership_date − 1 year  (SUE warmup buffer)

  3. Compare required_start against the earliest row already in the DB for
     that ticker. If earliest_in_db > required_start → gap exists.

  4. For gapped tickers, re-fetch earnings from Yahoo with a higher limit
     (400 rows) targeted at covering the full window.

  5. Merge new + existing rows (deduplicate on ticker+announce_date), sort
     ascending, then recompute SUE from scratch over the full history using
     the same expanding-window logic as the original build script.

  6. Option C SUE nulling: any quarter where fewer than SUE_MIN_PERIODS prior
     quarters exist gets sue = NULL. This means the backtest (WHERE sue IS NOT
     NULL) will never generate a signal from thin-history quarters.

  7. Upsert everything back with INSERT OR REPLACE.

USAGE
─────
  python update.py                        # uses earnings.db in current dir
  python update.py --db path/to/earnings.db
  python update.py --dry-run              # audit only, no writes
  python update.py --ticker AAPL MSFT     # limit to specific tickers

════════════════════════════════════════════════════════════════════════════════
"""

from __future__ import annotations

import argparse
import sqlite3
import time
from datetime import datetime, timedelta
from pathlib import Path
from typing import Optional

import numpy as np
import pandas as pd
import yfinance as yf

from betteryf_1d import sp500_members_on

# ─────────────────────────────────────────────────────────────────────────────
# Constants — must match build_earnings_db.py / pulldate.py exactly
# ─────────────────────────────────────────────────────────────────────────────

DEFAULT_DB       = Path("earnings.db")
BACKTEST_START   = "2011-01-01"          # earliest quarter we care about
SUE_MIN_PERIODS  = 4                     # min prior quarters to compute SUE
YAHOO_PAGE_SIZE  = 100                   # Yahoo hard cap per call
YAHOO_MAX_PAGES  = 6                     # 6 x 100 rows = up to ~150 distinct quarters
MEMBERSHIP_STEP  = "QS"                  # quarterly sampling for membership check
RATE_LIMIT_SLEEP = 0.4                   # seconds between pages (be polite)


# ─────────────────────────────────────────────────────────────────────────────
# S&P 500 membership window per ticker
# ─────────────────────────────────────────────────────────────────────────────

def build_membership_windows(
    tickers: list[str],
    start: str = BACKTEST_START,
) -> dict[str, tuple[pd.Timestamp, pd.Timestamp]]:
    """
    For each ticker, find (first_in_date, last_in_date) within [start, today].

    Samples quarterly to avoid calling sp500_members_on() for every single day.
    Returns only tickers that appear in the S&P 500 at least once.
    """
    today     = pd.Timestamp.today().normalize()
    dates     = pd.date_range(start=start, end=today, freq=MEMBERSHIP_STEP)
    ticker_set = set(tickers)

    # {ticker: [date, date, ...]} — all dates the ticker was a member
    membership: dict[str, list[pd.Timestamp]] = {t: [] for t in ticker_set}

    total = len(dates)
    print(f"[membership] Sampling {total} quarterly dates from {start} → {today.date()} ...")

    for i, dt in enumerate(dates):
        if i % 20 == 0:
            print(f"  {i}/{total}  ({dt.date()})", end="\r", flush=True)
        members = set(sp500_members_on(dt))
        for t in ticker_set:
            if t in members:
                membership[t].append(dt)

    print()  # newline after \r progress

    windows: dict[str, tuple[pd.Timestamp, pd.Timestamp]] = {}
    for t, dates_in in membership.items():
        if dates_in:
            windows[t] = (min(dates_in), max(dates_in))

    print(f"[membership] {len(windows)} / {len(tickers)} tickers found in S&P 500")
    return windows


# ─────────────────────────────────────────────────────────────────────────────
# DB helpers
# ─────────────────────────────────────────────────────────────────────────────

def get_db_tickers(conn: sqlite3.Connection) -> list[str]:
    rows = conn.execute("SELECT DISTINCT ticker FROM earnings ORDER BY ticker").fetchall()
    return [r[0] for r in rows]


def get_earliest_date(conn: sqlite3.Connection, ticker: str) -> Optional[pd.Timestamp]:
    row = conn.execute(
        "SELECT MIN(announce_date) FROM earnings WHERE ticker = ?", (ticker,)
    ).fetchone()
    if row and row[0]:
        return pd.Timestamp(row[0])
    return None


def fetch_all_rows(conn: sqlite3.Connection, ticker: str) -> pd.DataFrame:
    """Load all existing earnings rows for a ticker into a DataFrame."""
    df = pd.read_sql_query(
        "SELECT * FROM earnings WHERE ticker = ? ORDER BY announce_date",
        conn,
        params=(ticker,),
    )
    if not df.empty:
        df["announce_date"] = pd.to_datetime(df["announce_date"])
    return df


def upsert_rows(conn: sqlite3.Connection, rows: list[dict]) -> None:
    """
    INSERT OR REPLACE rows into earnings table.
    Matches the schema from build_earnings_db.py.
    """
    if not rows:
        return

    conn.executemany(
        """
        INSERT OR REPLACE INTO earnings
            (ticker, announce_date, eps_actual, eps_estimate,
             ue, sue, sector, in_sp500)
        VALUES
            (:ticker, :announce_date, :eps_actual, :eps_estimate,
             :ue, :sue, :sector, :in_sp500)
        """,
        rows,
    )
    conn.commit()


# ─────────────────────────────────────────────────────────────────────────────
# Yahoo fetch
# ─────────────────────────────────────────────────────────────────────────────

def fetch_yahoo_earnings(ticker: str,
                          required_start: Optional[pd.Timestamp] = None) -> pd.DataFrame:
    """
    Fetch earnings dates from Yahoo via yfinance using offset-based pagination.

    Yahoo hard-caps limit=100 per call, but exposes an `offset` parameter so
    we can page through history:
        page 0 : offset=0   → most-recent 100 rows
        page 1 : offset=100 → next 100 rows (older)
        page 2 : offset=200 → etc.

    We stop early once all rows on a page pre-date required_start (no point
    fetching further back than needed). Deduplicates by fiscal quarter (period
    "Q") keeping the confirmed actual over estimates.

    Returns DataFrame sorted ascending with columns:
        announce_date, eps_actual, eps_estimate
    """
    all_frames = []
    t = yf.Ticker(ticker)

    for page in range(YAHOO_MAX_PAGES):
        offset = page * YAHOO_PAGE_SIZE
        try:
            df_page = t.get_earnings_dates(limit=YAHOO_PAGE_SIZE, offset=offset)
            time.sleep(RATE_LIMIT_SLEEP)
        except Exception as e:
            if page == 0:
                print(f"  [warn] Yahoo fetch failed for {ticker}: {e}")
            break

        if df_page is None or df_page.empty:
            break

        df_page = df_page.reset_index()
        df_page.columns = [str(c).strip() for c in df_page.columns]

        rename_map = {}
        for c in df_page.columns:
            cl = c.lower()
            if "earnings date" in cl or cl == "index":
                rename_map[c] = "announce_date"
            elif "eps actual" in cl or "reported eps" in cl:
                rename_map[c] = "eps_actual"
            elif "eps estimate" in cl or "estimated eps" in cl:
                rename_map[c] = "eps_estimate"
        df_page = df_page.rename(columns=rename_map)

        needed = {"announce_date", "eps_actual", "eps_estimate"}
        if not needed.issubset(df_page.columns):
            break

        df_page["announce_date"] = pd.to_datetime(
            df_page["announce_date"], utc=True, errors="coerce"
        )
        df_page["announce_date"] = df_page["announce_date"].dt.tz_localize(None).dt.normalize()
        df_page = df_page.dropna(subset=["announce_date"])
        df_page["eps_actual"]   = pd.to_numeric(df_page["eps_actual"],   errors="coerce")
        df_page["eps_estimate"] = pd.to_numeric(df_page["eps_estimate"], errors="coerce")
        df_page = df_page.dropna(subset=["eps_actual", "eps_estimate"], how="all")

        if df_page.empty:
            break

        all_frames.append(df_page[["announce_date", "eps_actual", "eps_estimate"]])

        # Early stop: oldest date on this page is already before what we need
        if required_start is not None:
            oldest_on_page = df_page["announce_date"].min()
            if oldest_on_page < required_start:
                break  # we have enough history, no need for more pages

    if not all_frames:
        return pd.DataFrame()

    df = pd.concat(all_frames, ignore_index=True)

    # Deduplicate by fiscal quarter — keep confirmed actual over estimate-only rows
    df["_qkey"]       = df["announce_date"].dt.to_period("Q")
    df["_has_actual"] = df["eps_actual"].notna().astype(int)
    df = df.sort_values(["_qkey", "_has_actual", "announce_date"],
                        ascending=[True, False, False])
    df = df.drop_duplicates(subset=["_qkey"], keep="first")
    df = df.drop(columns=["_qkey", "_has_actual"])

    return df[["announce_date", "eps_actual", "eps_estimate"]].sort_values(
        "announce_date"
    ).reset_index(drop=True)


# ─────────────────────────────────────────────────────────────────────────────
# SUE computation  (must match pulldate.py logic exactly)
# ─────────────────────────────────────────────────────────────────────────────

def compute_sue(df: pd.DataFrame) -> pd.DataFrame:
    """
    Compute SUE over the full sorted history for a single ticker.

    SUE = UE / std(UE over all prior quarters)
    UE  = eps_actual - eps_estimate

    Option C: if fewer than SUE_MIN_PERIODS prior rows exist when computing
    a given row's SUE, set sue = None (NULL in DB). The backtest filters
    WHERE sue IS NOT NULL so these quarters never generate signals.

    Expects df sorted ascending by announce_date with eps_actual, eps_estimate.
    Returns df with ue and sue columns added/replaced.
    """
    df = df.copy().sort_values("announce_date").reset_index(drop=True)
    df["ue"]  = df["eps_actual"] - df["eps_estimate"]
    df["sue"] = np.nan

    for i in range(len(df)):
        if i < SUE_MIN_PERIODS:
            # Not enough history — Option C: leave as NaN → NULL in DB
            continue
        prior_ues = df["ue"].iloc[:i]
        std = prior_ues.std(ddof=1)
        if std == 0 or pd.isna(std):
            continue
        df.at[i, "sue"] = df.at[i, "ue"] / std

    return df


# ─────────────────────────────────────────────────────────────────────────────
# in_sp500 flag (point-in-time)
# ─────────────────────────────────────────────────────────────────────────────

def annotate_in_sp500(
    df: pd.DataFrame,
    membership_windows: dict[str, tuple[pd.Timestamp, pd.Timestamp]],
    ticker: str,
) -> pd.DataFrame:
    """
    Set in_sp500 = 1 if the announce_date falls within the ticker's
    membership window (inclusive), else 0.

    Using the window bounds (first/last quarterly sample) is a reasonable
    approximation and avoids calling sp500_members_on() per row.
    """
    df = df.copy()
    if ticker in membership_windows:
        first_in, last_in = membership_windows[ticker]
        df["in_sp500"] = (
            (df["announce_date"] >= first_in) &
            (df["announce_date"] <= last_in)
        ).astype(int)
    else:
        df["in_sp500"] = 0
    return df


# ─────────────────────────────────────────────────────────────────────────────
# Sector lookup (reuse whatever is already in the DB for this ticker)
# ─────────────────────────────────────────────────────────────────────────────

def get_sector(conn: sqlite3.Connection, ticker: str) -> Optional[str]:
    row = conn.execute(
        "SELECT sector FROM earnings WHERE ticker = ? AND sector IS NOT NULL LIMIT 1",
        (ticker,),
    ).fetchone()
    return row[0] if row else None


# ─────────────────────────────────────────────────────────────────────────────
# Core per-ticker update logic
# ─────────────────────────────────────────────────────────────────────────────

def process_ticker(
    conn: sqlite3.Connection,
    ticker: str,
    membership_windows: dict[str, tuple[pd.Timestamp, pd.Timestamp]],
    dry_run: bool = False,
) -> dict:
    """
    Full pipeline for one ticker. Returns a status dict for the audit log.
    """
    result = {
        "ticker":          ticker,
        "status":          "skipped",
        "required_start":  None,
        "earliest_in_db":  None,
        "gap_quarters":    0,
        "rows_added":      0,
        "rows_recomputed": 0,
        "error":           None,
    }

    # ── 1. Determine required coverage start ─────────────────────────────────
    if ticker not in membership_windows:
        result["status"] = "not_in_sp500"
        return result

    first_in, _ = membership_windows[ticker]
    required_start = first_in - pd.DateOffset(years=1)
    result["required_start"] = required_start.date()

    # ── 2. Check what's already in DB ────────────────────────────────────────
    earliest_in_db = get_earliest_date(conn, ticker)
    result["earliest_in_db"] = earliest_in_db.date() if earliest_in_db else None

    gap_exists = (earliest_in_db is None) or (earliest_in_db > required_start)

    if not gap_exists:
        result["status"] = "ok_no_gap"
        # Still recompute SUE in case rows were previously added without recompute
        # (skip recompute here for performance; only do it if gap exists)
        return result

    # ── 3. Re-fetch from Yahoo ────────────────────────────────────────────────
    yahoo_df = fetch_yahoo_earnings(ticker, required_start=required_start)
    time.sleep(RATE_LIMIT_SLEEP)

    if yahoo_df.empty:
        result["status"] = "yahoo_empty"
        return result

    # ── 4. Merge with existing DB rows ────────────────────────────────────────
    existing_df = fetch_all_rows(conn, ticker)
    sector      = get_sector(conn, ticker)

    if not existing_df.empty:
        # Align columns for concat
        existing_slim = existing_df[["announce_date", "eps_actual", "eps_estimate"]].copy()
        combined = pd.concat([existing_slim, yahoo_df], ignore_index=True)
    else:
        combined = yahoo_df.copy()

    combined["announce_date"] = pd.to_datetime(combined["announce_date"])
    combined = combined.sort_values("announce_date").drop_duplicates(
        subset=["announce_date"], keep="last"
    ).reset_index(drop=True)

    rows_added = len(combined) - (len(existing_df) if not existing_df.empty else 0)
    result["rows_added"] = max(0, rows_added)

    # ── 5. Recompute SUE from scratch ─────────────────────────────────────────
    combined = compute_sue(combined)
    combined["ticker"]   = ticker
    combined["sector"]   = sector
    combined             = annotate_in_sp500(combined, membership_windows, ticker)
    result["rows_recomputed"] = len(combined)

    # ── 6. Count gap quarters (pre-DB rows recovered) ────────────────────────
    if earliest_in_db is not None:
        result["gap_quarters"] = int(
            (combined["announce_date"] < earliest_in_db).sum()
        )
    else:
        result["gap_quarters"] = len(combined)

    # ── 7. Upsert ─────────────────────────────────────────────────────────────
    rows_to_upsert = []
    for _, row in combined.iterrows():
        announce_str = row["announce_date"].strftime("%Y-%m-%d")
        sue_val      = None if pd.isna(row["sue"]) else float(row["sue"])
        ue_val       = None if pd.isna(row["ue"])  else float(row["ue"])

        rows_to_upsert.append({
            "ticker":        ticker,
            "announce_date": announce_str,
            "eps_actual":    None if pd.isna(row["eps_actual"])   else float(row["eps_actual"]),
            "eps_estimate":  None if pd.isna(row["eps_estimate"]) else float(row["eps_estimate"]),
            "ue":            ue_val,
            "sue":           sue_val,
            "sector":        sector,
            "in_sp500":      int(row["in_sp500"]),
        })

    if not dry_run:
        upsert_rows(conn, rows_to_upsert)

    result["status"] = "updated" if not dry_run else "dry_run"
    return result


# ─────────────────────────────────────────────────────────────────────────────
# Main
# ─────────────────────────────────────────────────────────────────────────────

def run(
    db_path:  Path,
    dry_run:  bool,
    tickers:  Optional[list[str]],
) -> None:

    if not db_path.exists():
        print(f"[error] DB not found: {db_path}")
        return

    conn = sqlite3.connect(str(db_path))

    # ── Ticker universe ───────────────────────────────────────────────────────
    all_tickers = get_db_tickers(conn)
    if tickers:
        all_tickers = [t for t in all_tickers if t in set(tickers)]
        print(f"[filter] Limiting to {len(all_tickers)} specified tickers")
    else:
        print(f"[db] Found {len(all_tickers)} tickers in earnings.db")

    # ── Build membership windows (batched — one call per quarterly date) ──────
    membership_windows = build_membership_windows(all_tickers)

    # ── Process each ticker ───────────────────────────────────────────────────
    audit = []
    n     = len(all_tickers)

    print(f"\n[update] Processing {n} tickers ...")
    if dry_run:
        print("  DRY RUN — no writes will occur\n")

    for i, ticker in enumerate(all_tickers):
        print(f"  [{i+1:>4}/{n}] {ticker:<8}", end=" ", flush=True)
        try:
            r = process_ticker(conn, ticker, membership_windows, dry_run=dry_run)
            audit.append(r)
            status = r["status"]
            if status == "updated":
                print(f"✓  gap={r['gap_quarters']}q  recomputed={r['rows_recomputed']}r")
            elif status == "dry_run":
                print(f"[dry]  gap={r['gap_quarters']}q  would_recompute={r['rows_recomputed']}r")
            elif status == "ok_no_gap":
                print("ok  (no gap)")
            elif status == "not_in_sp500":
                print("—  (not in S&P 500)")
            elif status == "yahoo_empty":
                print("⚠  Yahoo returned no data")
            else:
                print(status)
        except Exception as e:
            msg = str(e)
            print(f"✗  ERROR: {msg[:80]}")
            audit.append({"ticker": ticker, "status": "error", "error": msg})

    conn.close()

    # ── Audit summary ─────────────────────────────────────────────────────────
    audit_df = pd.DataFrame(audit)
    print("\n" + "═" * 60)
    print("AUDIT SUMMARY")
    print("═" * 60)
    if not audit_df.empty and "status" in audit_df.columns:
        print(audit_df["status"].value_counts().to_string())
        updated = audit_df[audit_df["status"].isin(["updated", "dry_run"])]
        if not updated.empty:
            total_gap = updated["gap_quarters"].sum()
            total_new = updated["rows_recomputed"].sum()
            print(f"\nGap quarters recovered  : {total_gap:,}")
            print(f"Rows recomputed/upserted: {total_new:,}")

    # Save audit CSV next to the DB
    audit_path = db_path.parent / "update_audit.csv"
    if not dry_run:
        audit_df.to_csv(audit_path, index=False)
        print(f"\nAudit log saved → {audit_path}")

    print("\nDone.")


# ─────────────────────────────────────────────────────────────────────────────
# CLI
# ─────────────────────────────────────────────────────────────────────────────

def _parse() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="Gap-fill and SUE recompute for earnings.db"
    )
    p.add_argument("--db",      default=str(DEFAULT_DB),
                   help=f"Path to earnings.db (default: {DEFAULT_DB})")
    p.add_argument("--dry-run", action="store_true",
                   help="Audit only — print what would change, no writes")
    p.add_argument("--ticker",  nargs="+", metavar="TICKER",
                   help="Limit update to specific tickers (e.g. --ticker AAPL MSFT)")
    return p.parse_args()


if __name__ == "__main__":
    args = _parse()
    run(
        db_path = Path(args.db),
        dry_run = args.dry_run,
        tickers = args.ticker,
    )