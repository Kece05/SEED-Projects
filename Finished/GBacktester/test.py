"""
pead_backtest.py
════════════════════════════════════════════════════════════════════════════════
Triple-Filtered PEAD Strategy backtest using backtest_framework.py

STRATEGY RULES (from paper)
────────────────────────────
  Universe  : S&P 500 constituents (point-in-time, via betteryf_1d)
  Signal    : Top 20% SUE (Q5) from earnings.db — expanding window, no
              look-ahead bias. After-hours announcements already shifted to
              next trading day in the DB.
  Sectors   : Industrials and Technology only
  Holding   : 21 trading days per position, independent clocks
  Max slots : 7 simultaneous positions
  Idle cash : Remainder allocated to SPY
  Costs     : 16 bps/side  →  commission=0.001, slippage=0.0005
  Lag       : execution_lag=1 (fill at next-day close)

HOW IT FITS THE FRAMEWORK
──────────────────────────
  The framework's _apply() sets the ENTIRE portfolio to whatever weights you
  return on each rebalance day. To run independent 21-day clocks we maintain
  a `slots` dict {ticker: exit_date} in closure state, update it on each
  daily call, then return the full current target weights including SPY for
  idle capital. The framework handles execution, slippage, and commission.

USAGE
─────
  python pead_backtest.py                          # full run 2013-2025
  python pead_backtest.py --start 2018-01-01       # custom date range
  python pead_backtest.py --db path/to/earnings.db # custom DB path
  python pead_backtest.py --max-slots 5            # vary position count
  python pead_backtest.py --no-spy-parking         # stress test: no idle SPY
  python pead_backtest.py --plot                   # show chart after run

FILES NEEDED (all in same directory)
─────────────────────────────────────
  pead_backtest.py        ← this file
  backtest_framework.py
  betteryf_1d.py
  earnings.db             ← built by build_earnings_db.py

════════════════════════════════════════════════════════════════════════════════
"""

from __future__ import annotations

import argparse
import sqlite3
from collections import defaultdict
from datetime import date
from pathlib import Path
from typing import Dict, List, Optional

import numpy as np
import pandas as pd

from backtest_framework import backtest, DataStore

# ─────────────────────────────────────────────────────────────────────────────
# Constants
# ─────────────────────────────────────────────────────────────────────────────

DEFAULT_DB       = Path("earnings.db")
DEFAULT_START    = "2013-01-01"
DEFAULT_END      = "2025-12-31"
HOLD_DAYS        = 21          # trading days per position
MAX_SLOTS        = 7           # maximum simultaneous PEAD positions
IDLE_TICKER      = "SPY"       # where uninvested capital parks
TARGET_SECTORS   = {"Industrials", "Technology"}
COMMISSION       = 0.0010      # 10 bps  ─┐ = 16 bps/side total
SLIPPAGE         = 0.0006      # 6  bps  ─┘   (paper uses 16 bps)
INITIAL_CAPITAL  = 100_000.0
BENCHMARK        = "SPY"


# ─────────────────────────────────────────────────────────────────────────────
# Load signals from earnings.db
# ─────────────────────────────────────────────────────────────────────────────

def load_signals(db_path: Path, start: str, end: str) -> pd.DataFrame:
    """
    Load Q5 SUE signals for Industrials + Technology from the earnings DB.

    Returns a DataFrame with columns:
        announce_date  (datetime, the entry-eligible date)
        ticker
        sue
        sector

    Only rows where:
      - in_sp500 = 1              (PIT membership enforced)
      - sector in TARGET_SECTORS  (Industrials or Technology)
      - sue is not null           (needs >= 4 prior quarters)
      - announce_date in [start, end]

    SUE quintile is computed WITHIN each announce_date group using only
    the companies that reported on that same date — exactly as the paper
    describes. This is already correct because the DB stores the expanding-
    window SUE per ticker, so we just rank across same-day reporters.
    """
    conn = sqlite3.connect(str(db_path))

    query = """
        SELECT
            announce_date,
            ticker,
            sue,
            sector
        FROM earnings
        WHERE
            in_sp500    = 1
            AND sue     IS NOT NULL
            AND sector  IN ('Industrials', 'Technology')
            AND announce_date >= ?
            AND announce_date <= ?
        ORDER BY announce_date, sue DESC
    """
    df = pd.read_sql_query(query, conn, params=(start, end))
    conn.close()

    df["announce_date"] = pd.to_datetime(df["announce_date"])

    # ── Assign Q5 flag within each announce_date ──────────────────────────────
    # Rank same-day reporters: top 20% SUE on each announcement date gets Q5.
    # Require a minimum group size of 5 — fewer reporters produces noisy signals
    # that would inflate signal count and pollute Q5 with low-conviction names.
    def mark_q5(group: pd.DataFrame) -> pd.Series:
        if len(group) < 5:
            # Too few reporters to rank meaningfully — no signals from this date
            return pd.Series(0, index=group.index)
        cutoff = max(1, int(np.ceil(len(group) * 0.20)))   # top 20%
        ranked = group["sue"].rank(ascending=False, method="first")
        return (ranked <= cutoff).astype(int)

    df["q5"] = df.groupby("announce_date", group_keys=False).apply(mark_q5)
    signals  = df[df["q5"] == 1].copy()

    print(f"[signals] Loaded {len(signals):,} Q5 signals "
          f"({signals['ticker'].nunique()} unique tickers, "
          f"{start} → {end})")
    print(f"  Sector breakdown:")
    print(signals["sector"].value_counts().to_string())
    print(f"  Signals per year:")
    print(signals.groupby(signals["announce_date"].dt.year).size().to_string())

    return signals.reset_index(drop=True)


# ─────────────────────────────────────────────────────────────────────────────
# Build the signals DataFrame for the framework
# ─────────────────────────────────────────────────────────────────────────────

def build_signal_df(signals: pd.DataFrame) -> pd.DataFrame:
    """
    Convert to the format backtest() expects via df= / date_col= / ticker_col=:
        date      pd.Timestamp   — announce_date (already AMC-shifted in DB)
        ticker    str
        sue       float
        sector    str

    The framework passes rows whose date == current bar as `signals` list
    to the strategy function.
    """
    out = signals[["announce_date", "ticker", "sue", "sector"]].copy()
    out = out.rename(columns={"announce_date": "date"})
    return out


# ─────────────────────────────────────────────────────────────────────────────
# Strategy function factory
# ─────────────────────────────────────────────────────────────────────────────

def make_pead_strategy(
    max_slots:   int  = MAX_SLOTS,
    hold_days:   int  = HOLD_DAYS,
    spy_parking: bool = True,
) -> callable:
    """
    Returns the PEAD strategy function with state captured in closure.

    State:
        slots  dict {ticker: trading_days_remaining}
            Tracks active positions and when their 21-day clock expires.
            Initialised to hold_days + 1 to account for execution_lag=1:
            the fill happens at next-day close, so the actual hold starts
            one bar after the signal — without the +1 we'd only hold 20
            post-fill days instead of the intended 21.

    The framework calls fn(date, data, signals) on every trading day.
    We maintain slots across calls and return the full target weight dict
    each day. The framework's _apply() handles the actual order execution
    with execution_lag=1 (next-day fill).

    Weight allocation:
        Each active PEAD slot gets 1/max_slots of capital (fixed, not
        1/n_active) so that adding a new position never forces trimming
        of existing ones. Idle fraction goes to SPY if spy_parking=True.
        All weights sum to <= 1.0.
    """

    slots: Dict[str, int] = {}     # {ticker: trading_days_remaining}
    bar_count = [0]                # mutable counter via list trick

    def pead_strategy(dt: pd.Timestamp,
                      data: dict,
                      signals: list) -> dict:
        """
        Called every trading day by the framework.

        signals: list of dicts with keys: date, ticker, sue, sector
                 These are rows from signal_df whose date == dt.
                 An empty list means no earnings announcements today.
        """
        bar_count[0] += 1

        # ── 1. Age existing positions — decrement countdown ───────────────────
        to_exit = []
        for ticker, days_left in list(slots.items()):
            slots[ticker] = days_left - 1
            if slots[ticker] <= 0:
                to_exit.append(ticker)

        for ticker in to_exit:
            del slots[ticker]

        # ── 2. Process today's new signals ────────────────────────────────────
        # Sort by SUE descending so highest-conviction signals fill slots first
        new_signals = sorted(signals, key=lambda s: s.get("sue", 0), reverse=True)

        for sig in new_signals:
            ticker = sig["ticker"]

            # Skip if already holding this ticker
            if ticker in slots:
                continue

            # Skip if no price data available
            if ticker not in data or data[ticker].empty:
                continue

            # Skip if slots full
            if len(slots) >= max_slots:
                break

            # Enter: hold_days + 1 accounts for execution_lag=1.
            # The fill happens at next-day close, so day 0 is consumed by lag.
            # Without +1 we'd exit after 20 post-fill days, not 21.
            slots[ticker] = hold_days + 1

        # ── 3. Build target weight dict ───────────────────────────────────────
        # Verify all slot tickers still have price data; drop any that don't
        active = [t for t in slots if t in data and not data[t].empty]

        n_active = len(active)
        weights: dict = {}

        if n_active == 0:
            # No active positions — park everything in SPY
            if spy_parking and IDLE_TICKER in data:
                weights[IDLE_TICKER] = 1.0
            return weights

        # Fixed 1/max_slots weight per slot — never changes mid-hold so the
        # framework never needs to trim an existing position to make room.
        slot_weight   = 1.0 / max_slots
        idle_fraction = 1.0 - (n_active * slot_weight)

        for ticker in active:
            weights[ticker] = slot_weight

        if spy_parking and idle_fraction > 0.001 and IDLE_TICKER in data:
            # Guard: don't double-count if SPY is itself an active PEAD position
            existing_spy = weights.get(IDLE_TICKER, 0.0)
            weights[IDLE_TICKER] = existing_spy + idle_fraction

        return weights

    pead_strategy.__name__ = "Triple_Filtered_PEAD"
    return pead_strategy


# ─────────────────────────────────────────────────────────────────────────────
# Universe builder
# ─────────────────────────────────────────────────────────────────────────────

def build_ticker_universe(signals: pd.DataFrame, extra: list[str] = None) -> list[str]:
    """
    All tickers that ever appear in the signal set, plus SPY for idle parking
    and the benchmark. The framework will fetch price data for all of these.
    """
    tickers = set(signals["ticker"].unique())
    tickers.add(IDLE_TICKER)
    tickers.add(BENCHMARK)
    if extra:
        tickers.update(extra)
    return sorted(tickers)


# ─────────────────────────────────────────────────────────────────────────────
# Main
# ─────────────────────────────────────────────────────────────────────────────

def run(
    db_path:     Path,
    start:       str,
    end:         str,
    max_slots:   int,
    spy_parking: bool,
    show_plot:   bool,
    save_plot:   Optional[str],
    verbose:     bool,
) -> object:

    # ── Load signals ──────────────────────────────────────────────────────────
    signals    = load_signals(db_path, start, end)
    signal_df  = build_signal_df(signals)

    if signals.empty:
        print("[error] No signals found. Run build_earnings_db.py first.")
        return None

    # ── Build universe ────────────────────────────────────────────────────────
    universe = build_ticker_universe(signals)
    print(f"\n[universe] {len(universe)} tickers to fetch price data for")

    # ── Build strategy function ───────────────────────────────────────────────
    strategy = make_pead_strategy(
        max_slots   = max_slots,
        hold_days   = HOLD_DAYS,
        spy_parking = spy_parking,
    )

    print(f"\n[config]")
    print(f"  Period      : {start} → {end}")
    print(f"  Max slots   : {max_slots}")
    print(f"  Hold days   : {HOLD_DAYS} trading days (+1 lag = {HOLD_DAYS+1} countdown)")
    print(f"  Idle capital: {'SPY' if spy_parking else 'cash (0%)'}")
    print(f"  Commission  : {COMMISSION:.4f} ({COMMISSION*10000:.0f} bps)")
    print(f"  Slippage    : {SLIPPAGE:.4f} ({SLIPPAGE*10000:.0f} bps)")
    print(f"  Total cost  : ~{(COMMISSION + SLIPPAGE)*10000:.0f} bps/side")
    print(f"  Benchmark   : {BENCHMARK}")

    # ── Run backtest ──────────────────────────────────────────────────────────
    result = backtest(
        strategy,
        df             = signal_df,
        date_col       = "date",
        ticker_col     = "ticker",
        tickers        = universe,
        start          = start,
        end            = end,
        benchmark      = BENCHMARK,
        initial_capital= INITIAL_CAPITAL,
        commission     = COMMISSION,
        slippage       = SLIPPAGE,
        execution_lag  = 1,          # next-day close fill — look-ahead safe
        rebalance_freq = "D",        # called daily to manage 21-day clocks
        lookback       = 63,         # 3 months warm-up (ample for this strategy)
        verbose        = verbose,
    )

    # ── Results ───────────────────────────────────────────────────────────────
    result.summary()

    if show_plot or save_plot:
        result.plot(
            save_path = save_plot,
            show      = show_plot,
        )

    # ── Extra diagnostics ─────────────────────────────────────────────────────
    if not result.trades.empty:
        trades = result.trades.copy()
        entries = trades[trades["action"] == "ENTER"]
        exits   = trades[trades["action"].isin(["EXIT", "TRIM"])]

        print("\n[trade diagnostics]")
        print(f"  Total fills          : {len(trades):,}")
        print(f"  Entries              : {len(entries):,}")
        print(f"  Exits/Trims          : {len(exits):,}")

        if "return" in exits.columns:
            rets = exits["return"].dropna()
            print(f"  Avg trade return     : {rets.mean():+.3%}")
            print(f"  Win rate             : {(rets > 0).mean():.1%}")
            print(f"  Avg winner           : {rets[rets > 0].mean():+.3%}")
            print(f"  Avg loser            : {rets[rets < 0].mean():+.3%}")

        if "hold_days" in exits.columns:
            hd = exits["hold_days"].dropna()
            print(f"  Avg hold (cal days)  : {hd.mean():.1f}")

        if "ticker" in entries.columns:
            top = entries["ticker"].value_counts().head(10)
            print(f"\n  Most-traded tickers:")
            for t, n in top.items():
                print(f"    {t:<8} {n:>4} entries")

    return result


# ─────────────────────────────────────────────────────────────────────────────
# CLI
# ─────────────────────────────────────────────────────────────────────────────

def _parse() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="Triple-Filtered PEAD backtest (Industrials + Tech, Q5 SUE, 21-day hold)"
    )
    p.add_argument("--db",         default=str(DEFAULT_DB),
                   help=f"Path to earnings.db (default: {DEFAULT_DB})")
    p.add_argument("--start",      default=DEFAULT_START,
                   help=f"Backtest start date (default: {DEFAULT_START})")
    p.add_argument("--end",        default=DEFAULT_END,
                   help=f"Backtest end date (default: {DEFAULT_END})")
    p.add_argument("--max-slots",  type=int, default=MAX_SLOTS,
                   help=f"Max simultaneous PEAD positions (default: {MAX_SLOTS})")
    p.add_argument("--no-spy-parking", action="store_true",
                   help="Hold cash instead of SPY when slots are idle")
    p.add_argument("--plot",       action="store_true",
                   help="Display performance chart after run")
    p.add_argument("--save-plot",  default=None, metavar="PATH",
                   help="Save chart to file (e.g. pead_results.png)")
    p.add_argument("--quiet",      action="store_true",
                   help="Suppress per-ticker download progress")
    return p.parse_args()


if __name__ == "__main__":
    args = _parse()
    run(
        db_path     = Path(args.db),
        start       = args.start,
        end         = args.end,
        max_slots   = args.max_slots,
        spy_parking = not args.no_spy_parking,
        show_plot   = args.plot,
        save_plot   = args.save_plot,
        verbose     = not args.quiet,
    )