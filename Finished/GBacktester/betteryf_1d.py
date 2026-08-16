from __future__ import annotations

import asyncio
import threading
from datetime import datetime
from io import StringIO
from typing import Optional

import aiohttp
import pandas as pd
import requests
from bs4 import BeautifulSoup
import re


# ============================================================
# GLOBAL S&P500 CACHE
# ============================================================

_sp500_loaded = False
_sp500_lock = threading.Lock()
_changes_df: Optional[pd.DataFrame] = None
_current_tickers: Optional[set[str]] = None

_HEADERS = {"User-Agent": "Mozilla/5.0"}
_WIKI_URL = "https://en.wikipedia.org/wiki/List_of_S%26P_500_companies"


# ============================================================
# GLOBAL ETF CACHE
# ============================================================

_etfs_loaded = False
_etfs_lock = threading.Lock()
_all_etfs: Optional[list[str]] = None

_ETF_URL = "https://en.wikipedia.org/wiki/List_of_American_exchange-traded_funds"
_ETF_HEADERS = {
    "User-Agent": "MyPythonScraper/1.0 (https://example.com/contact) Requests/2.x"
}


# ============================================================
# ETF EXTRACTION LOGIC
# ============================================================

def _extract_etf_codes(text: str) -> list[str]:
    """
    Extract all text inside parentheses, take last 4 chars,
    strip non-letters, keep only all-capital strings.
    """
    inside = re.findall(r'\((.*?)\)', text)
    results = []

    for s in inside:
        last4 = s[-4:]
        cleaned = re.sub(r'[^A-Za-z]', '', last4)
        if cleaned.isalpha() and cleaned.isupper():
            results.append(cleaned)

    return results


def get_etfs() -> list[str]:
    """
    Returns a list of U.S. ETFs by scraping Wikipedia:
    - Fetch the page
    - Extract all text
    - Extract (...) contents
    - Take last 4 chars
    - Remove non-letters
    - Keep only all-capital codes
    Results are cached after first call.
    """
    global _etfs_loaded, _all_etfs

    if _etfs_loaded and _all_etfs is not None:
        return _all_etfs

    with _etfs_lock:
        if _etfs_loaded and _all_etfs is not None:
            return _all_etfs

        resp = requests.get(_ETF_URL, headers=_ETF_HEADERS, timeout=20)
        resp.raise_for_status()

        soup = BeautifulSoup(resp.text, "html.parser")
        all_text = soup.get_text()

        tickers = _extract_etf_codes(all_text)
        _all_etfs = sorted(set(tickers))
        _etfs_loaded = True

        return _all_etfs


# ============================================================
# S&P 500 MEMBERSHIP LOADER
# ============================================================

def _load_sp500_data() -> None:
    global _sp500_loaded, _changes_df, _current_tickers

    if _sp500_loaded:
        return

    with _sp500_lock:
        if _sp500_loaded:
            return

        resp = requests.get(_WIKI_URL, headers=_HEADERS, timeout=15)
        resp.raise_for_status()

        tables = pd.read_html(StringIO(resp.text))
        if len(tables) < 2:
            raise RuntimeError("Wikipedia S&P500 table structure changed.")

        # Correct tables
        current_df = tables[0]
        changes_df = tables[1].copy()

        # Normalize columns
        changes_df.columns = [str(c).strip() for c in changes_df.columns]

        # Correct column selection
        date_col = [c for c in changes_df.columns if "Effective" in c][0]
        added_col = [c for c in changes_df.columns if ("Added" in c and "Ticker" in c)][0]
        removed_col = [c for c in changes_df.columns if ("Removed" in c and "Ticker" in c)][0]

        changes_df = changes_df[[date_col, added_col, removed_col]].copy()
        changes_df.columns = ["effective_date", "added", "removed"]

        # Required cleaning
        changes_df["effective_date"] = pd.to_datetime(
            changes_df["effective_date"], errors="coerce"
        )
        changes_df["added"] = changes_df["added"].astype(str).str.strip()
        changes_df["removed"] = changes_df["removed"].astype(str).str.strip()

        changes_df = changes_df.dropna(subset=["effective_date"]).copy()

        _current_tickers = set(current_df["Symbol"].astype(str).str.strip())
        _changes_df = changes_df

        _sp500_loaded = True



def sp500_members_on(date: str | datetime) -> list[str]:
    """
    Return S&P 500 tickers on a given date.
    Synchronous function.
    """
    _load_sp500_data()
    assert _current_tickers is not None
    assert _changes_df is not None

    target_date = pd.to_datetime(date)
    members = set(_current_tickers)

    future_changes = _changes_df[_changes_df["effective_date"] > target_date]
    future_changes = future_changes.sort_values("effective_date", ascending=False)

    for _, row in future_changes.iterrows():
        if isinstance(row["added"], str) and row["added"] not in ["", "nan", "None"]:
            if row["added"] in members:
                members.remove(row["added"])
        if isinstance(row["removed"], str) and row["removed"] not in ["", "nan", "None"]:
            members.add(row["removed"])


    return sorted([t.replace(".", "-") for t in members])


# ============================================================
# ASYNC: FETCH ONE TICKER
# ============================================================

async def _fetch_ticker(session, ticker, start_unix, end_unix, interval, verbose):
    url = f"https://query1.finance.yahoo.com/v8/finance/chart/{ticker}"
    params = {
        "period1": start_unix,
        "period2": end_unix,
        "interval": interval,
        "events": "history",
        "includeAdjustedClose": "true",
    }

    try:
        async with session.get(url, params=params) as r:
            if r.status != 200:
                if verbose:
                    print(f"[WARN] {ticker}: HTTP {r.status}")
                return None
            data = await r.json()
    except Exception as e:
        if verbose:
            print(f"[ERROR] {ticker}: {e}")
        return None

    chart = data.get("chart", {})
    results = chart.get("result", [])
    if not results:
        if verbose:
            print(f"[WARN] {ticker}: empty chart response")
        return None

    result = results[0]
    if "timestamp" not in result or not result["timestamp"]:
        if verbose:
            print(f"[WARN] {ticker}: no price history")
        return None

    timestamps = result["timestamp"]
    indicators = result.get("indicators", {}).get("quote", [{}])[0]

    df = pd.DataFrame({
        "Date": pd.to_datetime(timestamps, unit="s"),
        "Open": indicators.get("open"),
        "High": indicators.get("high"),
        "Low": indicators.get("low"),
        "Close": indicators.get("close"),
        "Volume": indicators.get("volume"),
    })

    adj = result.get("indicators", {}).get("adjclose", [{}])[0]
    df["Adj Close"] = adj.get("adjclose", df["Close"])

    df.set_index("Date", inplace=True)
    df.columns = pd.MultiIndex.from_product([df.columns, [ticker]])

    return df


# ============================================================
# ASYNC ORCHESTRATOR
# ============================================================

async def _yahoo_chart_data_async(
    tickers, start_date, end_date, interval, max_concurrent, verbose, use_rich
):
    if isinstance(tickers, str):
        tickers = [tickers]

    start_unix = int(pd.Timestamp(start_date).timestamp())
    end_unix = int(pd.Timestamp(end_date).timestamp())
    cutoff_ts = pd.Timestamp(start_date) + pd.Timedelta(days=7)

    connector = aiohttp.TCPConnector(limit=max_concurrent)
    headers = {"User-Agent": "Mozilla/5.0"}
    timeout = aiohttp.ClientTimeout(total=20)

    tasks = []
    async with aiohttp.ClientSession(connector=connector, headers=headers, timeout=timeout) as session:
        for t in tickers:
            tasks.append(
                asyncio.create_task(
                    _fetch_ticker(session, t, start_unix, end_unix, interval, verbose)
                )
            )

        if use_rich:
            from rich.progress import Progress, SpinnerColumn, TextColumn
            results = [None] * len(tasks)
            with Progress(
                SpinnerColumn(),
                TextColumn("[progress.description]{task.description}")
            ) as progress:
                task_id = progress.add_task("Downloading...", total=len(tasks))
                for idx, coro in enumerate(asyncio.as_completed(tasks)):
                    results[idx] = await coro
                    progress.update(task_id, advance=1)
        else:
            results = await asyncio.gather(*tasks)

    valid, delisted, mismatched = [], [], []
    for ticker, r in zip(tickers, results):
        if r is None:
            delisted.append(ticker)
            continue
        if r.index.min() > cutoff_ts:
            mismatched.append(ticker)
            continue
        valid.append(r)

    if not valid:
        return None, delisted, mismatched

    return (pd.concat(valid, axis=1).sort_index(), delisted, mismatched)


# ============================================================
# PUBLIC ASYNC DOWNLOAD FUNCTION
# ============================================================

async def download(
    tickers, start, end, interval="1d", max_concurrent=10, verbose=True, use_rich=False
):
    """
    Async download function.
    Usage:
      - Jupyter: await download([...])
      - Scripts: asyncio.run(download([...]))
    """
    return await _yahoo_chart_data_async(
        tickers, start, end, interval, max_concurrent, verbose, use_rich
    )


# ============================================================
# SYNC WRAPPER FOR SCRIPTS
# ============================================================

def download_sync(*args, **kwargs):
    """
    Synchronous wrapper for Python scripts (no await needed).
    Cannot be used inside Jupyter.
    """
    try:
        loop = asyncio.get_running_loop()
        raise RuntimeError(
            "download_sync() cannot be used in Jupyter notebooks; use 'await download(...)' instead."
        )
    except RuntimeError:
        return asyncio.run(download(*args, **kwargs))


__all__ = [
    "sp500_members_on",
    "download",
    "download_sync",
    "get_etfs",
]