"""
backtest_framework.py  v5
─────────────────────────────────────────────────────────────────────────────
One public function: backtest().  Everything else is an implementation detail.

QUICKSTART
──────────
  from backtest_framework import backtest

  # ── Option A: explicit ticker list (attach to fn, no double-declaration) ──
  def sma_cross(date, data, signals):
      close = data["SPY"]["Close"]
      if len(close) < 200:
          return {}
      return {"SPY": 1.0} if close.iloc[-50:].mean() > close.iloc[-200:].mean() else {}

  sma_cross.tickers = ["SPY"]          # attach once — no need to also pass tickers=

  result = backtest(sma_cross, start="2018-01-01", end="2024-01-01")
  result.summary()
  result.plot()

  # ── Option B: S&P 500 universe (point-in-time correct, via betteryf_1d) ──
  def momentum(date, data, signals):
      scores = {}
      for ticker, df in data.items():
          if len(df) < 252:
              continue
          ret = df["Close"].iloc[-1] / df["Close"].iloc[-252] - 1
          scores[ticker] = ret
      top10 = sorted(scores, key=scores.get, reverse=True)[:10]
      return {t: 1/10 for t in top10}

  result = backtest(momentum, universe="sp500",
                    start="2015-01-01", end="2024-01-01")
  result.summary()
  result.plot()

  # External signal DataFrame (PEAD, ML scores, alt-data events …)
  def event_fn(date, data, signals):
      return {s["ticker"]: 0.10 for s in signals if s["score"] > 0.5}

  result = backtest(event_fn, df=my_df, date_col="event_date",
                    ticker_col="ticker", start="2018-01-01", end="2024-01-01")

─────────────────────────────────────────────────────────────────────────────
STRATEGY FUNCTION SIGNATURE
────────────────────────────
  def fn(date, data, signals) -> dict[str, float]

  date     pd.Timestamp      current bar being evaluated
  data     dict[ticker -> pd.DataFrame]
               OHLCV sliced strictly up to `date` (no look-ahead)
               tickers come from: fn.tickers / universe= / tickers= arg
               when universe="sp500", data contains only the current-day
               S&P 500 constituents (point-in-time correct)
  signals  list[dict]        rows from your df whose date_col == date
               empty list if no df was passed
  returns  {ticker: weight}  long-only portfolio weights; sum <= 1.0
               {} or all-zeros = hold cash
               negative weights are silently ignored (long only)
─────────────────────────────────────────────────────────────────────────────
TICKER UNIVERSE — THREE WAYS (pick one)
────────────────────────────────────────
  1. fn.tickers = [...]          attach list to strategy fn (recommended)
  2. universe="sp500"            S&P 500 with point-in-time constituents
  3. tickers=[...]               explicit list passed to backtest()
  Priority order: tickers= > universe= > fn.tickers
─────────────────────────────────────────────────────────────────────────────
BIAS CONTROLS
─────────────
  Look-ahead prices    data[t] is .loc[:date] — future bars are invisible
  S&P 500 membership   universe="sp500" uses betteryf_1d.sp500_members_on()
                       — avoids survivorship bias
  Execution lag        execution_lag=1 (default) fills at next-day close
  Warm-up data         lookback days fetched before start; equity starts at start
  Cost realism         commission + slippage applied symmetrically per fill
"""

from __future__ import annotations

import sqlite3
import threading
import time
import warnings
from dataclasses import dataclass, field
from datetime import timedelta
from pathlib import Path
from typing import Callable, Dict, List, Optional, Tuple, Union

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
from matplotlib.ticker import FuncFormatter

try:
    import yfinance as yf
    HAS_YF = True
except ImportError:
    HAS_YF = False
    warnings.warn("yfinance not installed — price downloads disabled.")

try:
    import statsmodels.api as sm
    HAS_SM = True
except ImportError:
    HAS_SM = False

try:
    from betteryf_1d import sp500_members_on as _sp500_members_on
    HAS_BETTERYF = True
except ImportError:
    HAS_BETTERYF = False
    _sp500_members_on = None  # type: ignore[assignment]

warnings.filterwarnings("ignore", category=FutureWarning)


# ─────────────────────────────────────────────────────────────────────────────
# Constants
# ─────────────────────────────────────────────────────────────────────────────

DEFAULT_DB      = Path.cwd() / ".backtest_cache" / "market_data.db"
RF_TICKER       = "^IRX"
DEFAULT_BENCH   = "SPY"
TDAYS           = 252
_CAL_BUFFER     = 1.6   # multiply trading-day lookback → calendar days

# ── SEED brand color palette ─────────────────────────────────────────────────
SEED_COLORS = dict(
    bg      = "#0a0f1e",   # deep navy background
    panel   = "#111827",   # dark card/panel
    strat   = "#00c9a7",   # SEED teal — strategy line
    bench   = "#f59e0b",   # amber — benchmark line
    dd_s    = "#f43f5e",   # rose — strategy drawdown fill
    dd_b    = "#6b7280",   # grey — benchmark drawdown fill
    pos     = "#10b981",   # emerald — positive bars / fill
    neg     = "#f43f5e",   # rose — negative bars / fill
    text    = "#e2e8f0",   # near-white text
    grid    = "#1e293b",   # subtle gridline
    gold    = "#fbbf24",   # gold — period markers / accents
)


# ─────────────────────────────────────────────────────────────────────────────
# S&P 500 universe helper
# ─────────────────────────────────────────────────────────────────────────────

def _sp500_union(start: str, end: str, verbose: bool = True) -> List[str]:
    """
    Return the union of all S&P 500 tickers that appeared in the index
    between *start* and *end*. Samples quarterly so the union is complete
    without needing daily lookups.

    Requires betteryf_1d.py to be importable.
    """
    if not HAS_BETTERYF or _sp500_members_on is None:
        raise ImportError(
            "betteryf_1d.py is required for universe='sp500'. "
            "Place betteryf_1d.py in the same directory as backtest_framework.py."
        )

    sample_dates = pd.date_range(start, end, freq="QS").tolist()
    sample_dates += [pd.Timestamp(start), pd.Timestamp(end)]
    sample_dates = sorted(set(sample_dates))

    all_tickers: set = set()
    for d in sample_dates:
        try:
            members = _sp500_members_on(d)
            all_tickers.update(members)
        except Exception as e:
            if verbose:
                print(f"  [sp500] Warning on {d.date()}: {e}")

    result = sorted(all_tickers)
    if verbose:
        print(f"  [sp500] {len(result)} unique constituent tickers "
              f"across {len(sample_dates)} sample dates")
    return result


# ─────────────────────────────────────────────────────────────────────────────
# DataStore  -  SQLite price cache
# ─────────────────────────────────────────────────────────────────────────────

class DataStore:
    """
    SQLite-backed OHLCV cache. Only ever downloads uncached date ranges.
    Shared across all calls in a process via _get_store().

    Usage
    ─────
      store = DataStore()                        # default .backtest_cache/
      store = DataStore("./my_project/data.db")  # custom path
      df    = store.get("AAPL", "2018-01-01", "2024-01-01", lookback=252)
      store.info()         # show what is cached
      store.clear("AAPL")  # force re-download for one ticker
    """

    def __init__(self, path: Optional[str] = None):
        self.path = Path(path or DEFAULT_DB)
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self._conn = sqlite3.connect(str(self.path), check_same_thread=False)
        self._lock = threading.Lock()
        self._setup()

    def _setup(self):
        self._conn.executescript("""
            CREATE TABLE IF NOT EXISTS prices (
                ticker TEXT, interval TEXT, date TEXT,
                open REAL, high REAL, low REAL, close REAL, volume REAL,
                PRIMARY KEY (ticker, interval, date)
            );
            CREATE TABLE IF NOT EXISTS meta (
                ticker TEXT, interval TEXT,
                min_date TEXT, max_date TEXT, updated TEXT,
                PRIMARY KEY (ticker, interval)
            );
        """)
        self._conn.commit()

    def _meta(self, ticker: str, interval: str) -> Optional[dict]:
        cur = self._conn.execute(
            "SELECT min_date, max_date FROM meta WHERE ticker=? AND interval=?",
            (ticker, interval))
        row = cur.fetchone()
        return {"min": row[0], "max": row[1]} if row else None

    def _upsert(self, ticker: str, interval: str, df: pd.DataFrame):
        if df.empty:
            return
        rows = list(zip(
            [ticker]   * len(df),
            [interval] * len(df),
            df.index.strftime("%Y-%m-%d"),
            [_f(v) for v in df["Open"]],
            [_f(v) for v in df["High"]],
            [_f(v) for v in df["Low"]],
            [_f(v) for v in df["Close"]],
            [_f(v) for v in df["Volume"]],
        ))
        with self._lock:
            self._conn.executemany(
                "INSERT OR REPLACE INTO prices VALUES(?,?,?,?,?,?,?,?)", rows)
            self._conn.execute(
                "INSERT OR REPLACE INTO meta VALUES(?,?,?,?,datetime('now'))",
                (ticker, interval,
                 str(df.index.min().date()), str(df.index.max().date())))
            self._conn.commit()

    def _read(self, ticker: str, interval: str,
              start: str, end: str) -> pd.DataFrame:
        cur = self._conn.execute(
            "SELECT date,open,high,low,close,volume FROM prices "
            "WHERE ticker=? AND interval=? AND date>=? AND date<=? ORDER BY date",
            (ticker, interval, start, end))
        rows = cur.fetchall()
        if not rows:
            return pd.DataFrame()
        df = pd.DataFrame(rows, columns=["Date","Open","High","Low","Close","Volume"])
        df["Date"] = pd.to_datetime(df["Date"])
        return df.set_index("Date")

    def _download(self, ticker: str, interval: str,
                  start: str, end: str) -> pd.DataFrame:
        if not HAS_YF:
            raise RuntimeError(
                "yfinance is not installed. "
                "Install it with:  pip install yfinance")
        try:
            raw = yf.download(ticker, start=start, end=end,
                              interval=interval, auto_adjust=True, progress=False)
            if raw.empty:
                return pd.DataFrame()
            if isinstance(raw.columns, pd.MultiIndex):
                raw.columns = raw.columns.get_level_values(0)
            raw.index = pd.to_datetime(raw.index).tz_localize(None)
            return raw[["Open","High","Low","Close","Volume"]].dropna(how="all")
        except Exception as e:
            warnings.warn(f"[DataStore] Download failed for '{ticker}': {e}")
            return pd.DataFrame()

    def get(self, ticker: str, start: str, end: str,
            lookback: int = 0, interval: str = "1d") -> pd.DataFrame:
        """Return OHLCV DataFrame, fetching only uncached date ranges."""
        warmup = (
            pd.to_datetime(start) - timedelta(days=int(lookback * _CAL_BUFFER))
        ).strftime("%Y-%m-%d")
        meta = self._meta(ticker, interval)
        if meta is None:
            self._upsert(ticker, interval,
                         self._download(ticker, interval, warmup, end))
        else:
            if warmup < meta["min"]:
                self._upsert(ticker, interval,
                             self._download(ticker, interval, warmup, meta["min"]))
            if end > meta["max"]:
                self._upsert(ticker, interval,
                             self._download(ticker, interval, meta["max"], end))
        return self._read(ticker, interval, warmup, end)

    def prefetch(self, tickers: List[str], start: str, end: str,
                 lookback: int = 0, interval: str = "1d"):
        """
        Batch-download tickers that are missing or whose cached range
        does not fully cover the requested warmup→end window.
        """
        warmup = (
            pd.to_datetime(start) - timedelta(days=int(lookback * _CAL_BUFFER))
        ).strftime("%Y-%m-%d")

        to_fetch: List[str] = []
        already = 0
        for t in tickers:
            meta = self._meta(t, interval)
            if meta is None or warmup < meta["min"] or end > meta["max"]:
                to_fetch.append(t)
            else:
                already += 1

        if already:
            print(f"  [cache] {already}/{len(tickers)} already cached")
        if to_fetch:
            print(f"  [cache] Fetching {len(to_fetch)} ticker(s)…")
            for i, t in enumerate(to_fetch, 1):
                df = self._download(t, interval, warmup, end)
                if df.empty:
                    warnings.warn(
                        f"  [cache] No data returned for '{t}'. "
                        "Check the ticker symbol and your internet connection.")
                else:
                    self._upsert(t, interval, df)
                if i % 20 == 0:
                    print(f"    {i}/{len(to_fetch)}")

    def clear(self, ticker: str, interval: str = "1d"):
        """Force re-download of a single ticker on the next get() call."""
        with self._lock:
            self._conn.execute(
                "DELETE FROM prices WHERE ticker=? AND interval=?", (ticker, interval))
            self._conn.execute(
                "DELETE FROM meta WHERE ticker=? AND interval=?", (ticker, interval))
            self._conn.commit()
        print(f"  [cache] Cleared '{ticker}' ({interval})")

    def info(self) -> pd.DataFrame:
        """Show what is currently cached."""
        cur = self._conn.execute(
            "SELECT ticker,interval,min_date,max_date,updated FROM meta ORDER BY ticker")
        return pd.DataFrame(cur.fetchall(),
                            columns=["Ticker","Interval","From","To","Updated"])


# ─────────────────────────────────────────────────────────────────────────────
# Analytics
# ─────────────────────────────────────────────────────────────────────────────

def _ret(eq: pd.Series) -> pd.Series:
    return eq.pct_change(fill_method=None).dropna()

def cagr(eq: pd.Series) -> float:
    if len(eq) < 2:
        return np.nan
    yrs = (eq.index[-1] - eq.index[0]).days / 365.25
    return (eq.iloc[-1] / eq.iloc[0]) ** (1 / yrs) - 1 if yrs > 0 else np.nan

def sharpe(r: pd.Series, rf: float = 0.0) -> float:
    exc = r - rf / TDAYS
    return (exc.mean() / exc.std() * np.sqrt(TDAYS)) if exc.std() > 0 else np.nan

def sortino(r: pd.Series, rf: float = 0.0) -> float:
    exc = r - rf / TDAYS
    d   = exc[exc < 0].std()
    return (exc.mean() / d * np.sqrt(TDAYS)) if d > 0 else np.nan

def max_dd(eq: pd.Series) -> float:
    return ((eq - eq.cummax()) / eq.cummax()).min()

def calmar(eq: pd.Series) -> float:
    mdd = abs(max_dd(eq))
    return cagr(eq) / mdd if mdd > 0 else np.nan

def beta(rp: pd.Series, rb: pd.Series) -> float:
    al = pd.concat([rp, rb], axis=1).dropna()
    if len(al) < 10:
        return np.nan
    return al.iloc[:,0].cov(al.iloc[:,1]) / al.iloc[:,1].var()

def jensens_alpha(rp: pd.Series, rb: pd.Series, rf: float = 0.0) -> float:
    b = beta(rp, rb)
    if np.isnan(b):
        return np.nan
    rf_d = rf / TDAYS
    return (rp.mean() - (rf_d + b * (rb.mean() - rf_d))) * TDAYS

def info_ratio(rp: pd.Series, rb: pd.Series) -> float:
    active = (rp - rb).dropna()
    te     = active.std()
    return (active.mean() / te * np.sqrt(TDAYS)) if te > 0 else np.nan

def dd_series(eq: pd.Series) -> pd.Series:
    return (eq - eq.cummax()) / eq.cummax()


# ─────────────────────────────────────────────────────────────────────────────
# Stats  -  OLS + NW-HAC significance tests
# ─────────────────────────────────────────────────────────────────────────────

@dataclass
class Stats:
    alpha_ann:   float = np.nan
    alpha_t:     float = np.nan
    alpha_p:     float = np.nan
    alpha_nw_t:  float = np.nan
    alpha_nw_p:  float = np.nan
    beta_:       float = np.nan
    r2:          float = np.nan
    ir:          float = np.nan
    ir_t:        float = np.nan
    ir_p:        float = np.nan
    ci_lo:       float = np.nan
    ci_hi:       float = np.nan
    yoy:         pd.DataFrame = field(default_factory=pd.DataFrame)
    beat_years:  int = 0
    total_years: int = 0


def compute_stats(rp: pd.Series, rb: pd.Series, rf: float) -> Stats:
    s  = Stats()
    al = pd.concat([rp, rb], axis=1).dropna()
    al.columns = ["rp", "rb"]
    if len(al) < 30:
        return s

    # CAPM OLS + NW-HAC
    if HAS_SM:
        try:
            X   = sm.add_constant(al["rb"] - rf / TDAYS)
            y   = al["rp"] - rf / TDAYS
            res = sm.OLS(y, X).fit()
            s.alpha_ann = float(res.params.iloc[0]) * TDAYS
            s.alpha_t   = float(res.tvalues.iloc[0])
            s.alpha_p   = float(res.pvalues.iloc[0])
            s.beta_     = float(res.params.iloc[1])
            s.r2        = float(res.rsquared)
            nw = res.get_robustcov_results(cov_type="HAC",
                                           maxlags=int(TDAYS**0.5))
            s.alpha_nw_t = float(nw.tvalues.iloc[0])
            s.alpha_nw_p = float(nw.pvalues.iloc[0])
        except Exception:
            pass
    else:
        s.alpha_ann = jensens_alpha(al["rp"], al["rb"], rf)
        s.beta_     = beta(al["rp"], al["rb"])

    # IR t-test
    active = al["rp"] - al["rb"]
    s.ir   = info_ratio(al["rp"], al["rb"])
    if len(active) > 1:
        try:
            from scipy import stats as sp
            t, p   = sp.ttest_1samp(active.dropna(), 0)
            s.ir_t = float(t)
            s.ir_p = float(p)
        except Exception:
            pass

    # Moving-block bootstrap CAGR 95% CI
    try:
        n         = len(al)
        block     = max(1, int(TDAYS**0.5))
        n_blocks  = int(np.ceil(n / block))
        max_start = max(0, n - block)
        boot      = []
        for _ in range(2_000):
            starts = np.random.randint(0, max_start + 1, n_blocks)
            idx_b  = np.concatenate(
                [np.arange(s_, min(s_ + block, n)) for s_ in starts]
            )[:n]
            samp   = al["rp"].iloc[idx_b].reset_index(drop=True)
            eq_s   = (1 + samp).cumprod()
            boot.append(eq_s.iloc[-1] ** (TDAYS / n) - 1)
        s.ci_lo = float(np.percentile(boot, 2.5))
        s.ci_hi = float(np.percentile(boot, 97.5))
    except Exception:
        pass

    # Year-by-year
    try:
        yp = (1 + al["rp"]).resample("YE").prod() - 1
        yb = (1 + al["rb"]).resample("YE").prod() - 1
        yy = pd.concat([yp, yb], axis=1).dropna()
        yy.columns    = ["strategy", "benchmark"]
        yy.index      = yy.index.year
        s.yoy         = yy
        s.beat_years  = int((yy["strategy"] > yy["benchmark"]).sum())
        s.total_years = len(yy)
    except Exception:
        pass

    return s


# ─────────────────────────────────────────────────────────────────────────────
# Result
# ─────────────────────────────────────────────────────────────────────────────

class Result:
    """
    Returned by backtest().

    Attributes
    ──────────
    result.equity      pd.Series       daily portfolio value
    result.benchmark   pd.Series       daily benchmark value
    result.trades      pd.DataFrame    every executed trade
    result.metrics     dict            all computed metrics (lazily cached)

    Methods
    ───────
    result.summary()                    print full metrics table
    result.plot()                       5-panel performance chart
    result.plot(save_path="out.png")    save chart without displaying
    """

    def __init__(self, equity, benchmark, trades, rf, name,
                 bench_name, commission, slippage, lag):
        self.equity     = equity
        self.benchmark  = benchmark
        self.trades     = trades
        self.rf         = rf
        self.name       = name
        self.bench_name = bench_name
        self.commission = commission
        self.slippage   = slippage
        self.lag        = lag

        self._rp = _ret(equity)
        self._rb = _ret(benchmark).reindex(self._rp.index).fillna(0)

    def __repr__(self) -> str:
        m   = self.metrics
        c   = m["CAGR"]
        sh  = m["Sharpe Ratio"]
        mdd = m["Max Drawdown"]
        return (
            f"<Result '{self.name}'  "
            f"CAGR={'n/a' if np.isnan(c)   else f'{c:+.2%}'}  "
            f"Sharpe={'n/a' if np.isnan(sh) else f'{sh:.2f}'}  "
            f"MDD={'n/a' if np.isnan(mdd)  else f'{mdd:.1%}'}>"
        )

    @property
    def stats(self) -> Stats:
        if not hasattr(self, "_stats"):
            self._stats = compute_stats(self._rp, self._rb, self.rf)
        return self._stats

    @property
    def metrics(self) -> dict:
        if not hasattr(self, "_metrics"):
            s = self.stats
            self._metrics = {
                "CAGR":                   cagr(self.equity),
                "Benchmark CAGR":         cagr(self.benchmark),
                "Excess CAGR":            cagr(self.equity) - cagr(self.benchmark),
                "Total Return":           self.equity.iloc[-1] / self.equity.iloc[0] - 1,
                "Sharpe Ratio":           sharpe(self._rp, self.rf),
                "Sortino Ratio":          sortino(self._rp, self.rf),
                "Max Drawdown":           max_dd(self.equity),
                "Benchmark Max Drawdown": max_dd(self.benchmark),
                "Calmar Ratio":           calmar(self.equity),
                "Volatility (ann.)":      self._rp.std() * np.sqrt(TDAYS),
                "Win Rate":               float((self._rp > 0).mean()),
                "Num Trades":             len(self.trades),
                "Risk-Free Rate":         self.rf,
                "Jensen's Alpha":         s.alpha_ann,
                "Alpha t (OLS)":          s.alpha_t,
                "Alpha p (OLS)":          s.alpha_p,
                "Alpha t (NW HAC)":       s.alpha_nw_t,
                "Alpha p (NW HAC)":       s.alpha_nw_p,
                "Beta":                   s.beta_,
                "R2 (CAPM)":              s.r2,
                "Information Ratio":      s.ir,
                "IR t (NW HAC)":          s.ir_t,
                "IR p (NW HAC)":          s.ir_p,
                "CAGR CI lo (95%)":       s.ci_lo,
                "CAGR CI hi (95%)":       s.ci_hi,
            }
        return self._metrics

    def summary(self) -> "Result":
        m, s = self.metrics, self.stats
        SEP  = "=" * 54
        na   = "   n/a"

        def pct(v):  return f"{v:>+.2%}" if not np.isnan(v) else na
        def f3(v):   return f"{v:>+.3f}" if not np.isnan(v) else na
        def pct0(v): return f"{v:>.2%}"  if not np.isnan(v) else na

        print(f"\n{SEP}")
        print(f"  {self.name}  >>  {self.bench_name}")
        print(f"  comm={self.commission:.2%}  slip={self.slippage:.3%}  lag={self.lag}d")
        print(f"{SEP}")
        print("  -- PERFORMANCE ---")
        for k, v in [
            ("CAGR",               pct(m["CAGR"])),
            ("CAGR 95% CI",        f"[{pct(s.ci_lo)}, {pct(s.ci_hi)}]"),
            ("Benchmark CAGR",     pct(m["Benchmark CAGR"])),
            ("Excess CAGR",        pct(m["Excess CAGR"])),
            ("Total Return",       pct(m["Total Return"])),
            ("Sharpe Ratio",       f3(m["Sharpe Ratio"])),
            ("Sortino Ratio",      f3(m["Sortino Ratio"])),
            ("Calmar Ratio",       f3(m["Calmar Ratio"])),
            ("Max Drawdown",       pct0(m["Max Drawdown"])),
            ("Bench Max Drawdown", pct0(m["Benchmark Max Drawdown"])),
            ("Volatility (ann.)",  pct0(m["Volatility (ann.)"])),
            ("Win Rate",           pct0(m["Win Rate"])),
            ("Num Trades",         str(int(m["Num Trades"]))),
        ]:
            print(f"  {k:<30} {v}")

        print("  -- SIGNIFICANCE --")
        ols_t, ols_p = m["Alpha t (OLS)"], m["Alpha p (OLS)"]
        nw_t,  nw_p  = m["Alpha t (NW HAC)"], m["Alpha p (NW HAC)"]
        ir_t,  ir_p  = m["IR t (NW HAC)"],    m["IR p (NW HAC)"]
        for k, v in [
            ("Jensen's Alpha",       pct(m["Jensen's Alpha"])),
            ("Alpha t / p (OLS)",    f"{ols_t:>+.3f}  /  {ols_p:.4f}" if not np.isnan(ols_t) else na),
            ("Alpha t / p (NW HAC)", f"{nw_t:>+.3f}  /  {nw_p:.4f}"  if not np.isnan(nw_t)  else na),
            ("Beta",                 f3(m["Beta"])),
            ("R2 (CAPM)",            f"{m['R2 (CAPM)']:.3f}"          if not np.isnan(m["R2 (CAPM)"]) else na),
            ("Information Ratio",    f3(m["Information Ratio"])),
            ("IR t / p (NW HAC)",    f"{ir_t:>+.3f}  /  {ir_p:.4f}"  if not np.isnan(ir_t)  else na),
        ]:
            print(f"  {k:<30} {v}")

        if not s.yoy.empty:
            print(f"  -- YEAR-BY-YEAR  (beat {s.beat_years}/{s.total_years}) --")
            print(f"  {'Year':<6} {'Strategy':>10} {'Benchmark':>10} {'Edge':>8}")
            for yr, row in s.yoy.iterrows():
                sym = "^" if row["strategy"] > row["benchmark"] else "v"
                print(f"  {yr:<6} {row['strategy']:>+9.1%} {row['benchmark']:>+9.1%}"
                      f" {row['strategy']-row['benchmark']:>+7.1%}  {sym}")

        print(f"{SEP}\n")
        return self

    def plot(self, save_path: Optional[str] = None, show: bool = True,
             figsize=(16, 11)) -> "Result":
        C = SEED_COLORS

        m   = self.metrics
        s   = self.stats
        peq = self.equity    / self.equity.iloc[0]
        beq = self.benchmark / self.benchmark.iloc[0]

        fig = plt.figure(figsize=figsize, facecolor=C["bg"])
        gs  = gridspec.GridSpec(3, 2, figure=fig,
                                height_ratios=[2.5, 1.5, 1.8],
                                hspace=0.50, wspace=0.28)

        nw_p      = m["Alpha p (NW HAC)"]
        nw_sig    = " *" if (not np.isnan(nw_p) and nw_p < 0.05) else ""
        alpha_str = f"{s.alpha_ann:+.2%}" if not np.isnan(s.alpha_ann) else "n/a"
        nw_str    = f"{nw_p:.3f}"         if not np.isnan(nw_p)        else "n/a"

        fig.suptitle(
            f"{self.name}  >>  "
            f"CAGR {m['CAGR']:+.1%} vs {self.bench_name} {m['Benchmark CAGR']:+.1%}  |  "
            f"alpha {alpha_str} (NW p={nw_str}{nw_sig})  |  "
            f"Sharpe {m['Sharpe Ratio']:.2f}  |  lag={self.lag}d",
            color=C["text"], fontsize=10, y=0.995)

        # 1. Equity curves
        ax1 = fig.add_subplot(gs[0, :])
        ax1.plot(peq.index, peq, color=C["strat"], lw=2.0,
                 label=f"{self.name}  {m['CAGR']:+.1%}")
        ax1.plot(beq.index, beq, color=C["bench"], lw=1.4, alpha=0.85,
                 label=f"{self.bench_name}  {m['Benchmark CAGR']:+.1%}")
        ax1.fill_between(peq.index, peq, beq, where=peq >= beq,
                         color=C["strat"], alpha=0.12)
        ax1.fill_between(peq.index, peq, beq, where=peq < beq,
                         color=C["bench"], alpha=0.12)
        ax1.yaxis.set_major_formatter(FuncFormatter(lambda x, _: f"{x:.1f}x"))
        ax1.legend(loc="upper left", facecolor=C["panel"],
                   labelcolor=C["text"], fontsize=9)
        _sax(ax1, C)

        # 2. Drawdown
        ax2 = fig.add_subplot(gs[1, 0], sharex=ax1)
        ddp = dd_series(self.equity)
        ddb = dd_series(self.benchmark)
        ax2.fill_between(ddp.index, ddp, 0, color=C["dd_s"], alpha=0.55,
                         label=f"Strategy  {m['Max Drawdown']:.1%}")
        ax2.fill_between(ddb.index, ddb, 0, color=C["dd_b"], alpha=0.30,
                         label=f"{self.bench_name}  {m['Benchmark Max Drawdown']:.1%}")
        ax2.yaxis.set_major_formatter(FuncFormatter(lambda x, _: f"{x:.0%}"))
        ax2.set_title("Drawdown", color=C["text"], fontsize=9, pad=5)
        ax2.legend(fontsize=8, facecolor=C["panel"], labelcolor=C["text"])
        _sax(ax2, C)

        # 3. Rolling 12M excess return
        ax3 = fig.add_subplot(gs[1, 1], sharex=ax1)
        exc  = (self._rp - self._rb).dropna()
        roll = exc.rolling(TDAYS).mean() * TDAYS * 100
        ax3.plot(roll.index, roll, color=C["gold"], lw=1.4)
        ax3.axhline(0, color=C["text"], lw=0.6, alpha=0.4, ls="--")
        ax3.fill_between(roll.index, roll, 0,
                         where=roll >= 0, color=C["pos"], alpha=0.20)
        ax3.fill_between(roll.index, roll, 0,
                         where=roll < 0,  color=C["neg"], alpha=0.20)
        ax3.yaxis.set_major_formatter(FuncFormatter(lambda x, _: f"{x:+.0f}%"))
        ir_str = f"{s.ir:+.2f}" if not np.isnan(s.ir) else "n/a"
        ax3.set_title(f"Rolling 12M Excess Return  IR={ir_str}",
                      color=C["text"], fontsize=9, pad=5)
        _sax(ax3, C)

        # 4. Year-by-year bars
        ax4 = fig.add_subplot(gs[2, 0])
        yy  = s.yoy
        if not yy.empty:
            x = np.arange(len(yy)); w = 0.35
            ax4.bar(x - w/2, yy["strategy"]  * 100, w, alpha=0.85,
                    color=[C["pos"] if v >= 0 else C["neg"] for v in yy["strategy"]])
            ax4.bar(x + w/2, yy["benchmark"] * 100, w, alpha=0.55,
                    color=C["bench"], label=self.bench_name)
            ax4.axhline(0, color=C["text"], lw=0.5, alpha=0.3)
            ax4.set_xticks(x)
            ax4.set_xticklabels(yy.index, rotation=45, fontsize=7, color=C["text"])
            ax4.yaxis.set_major_formatter(FuncFormatter(lambda x, _: f"{x:+.0f}%"))
            ax4.set_title(f"Year-by-Year  (beat {s.beat_years}/{s.total_years})",
                          color=C["text"], fontsize=9, pad=5)
            ax4.legend(fontsize=8, facecolor=C["panel"], labelcolor=C["text"])
        _sax(ax4, C)

        # 5. Monthly heatmap
        ax5 = fig.add_subplot(gs[2, 1])
        _heatmap(ax5, self._rp, C)

        if save_path:
            fig.savefig(save_path, dpi=150, bbox_inches="tight",
                        facecolor=fig.get_facecolor())
            print(f"  [Result] Saved → {save_path}")
        if show:
            plt.show()
        return self


# ─────────────────────────────────────────────────────────────────────────────
# Core simulation loop  (internal)
# ─────────────────────────────────────────────────────────────────────────────

def _simulate(fn, bundle, bench_df, signals_by_date, start, end, rf,
              name, bench_name, initial_capital, commission, slippage,
              execution_lag, rebalance_freq, verbose,
              universe_fn=None):
    """
    universe_fn: optional callable(date) -> list[str]
        When provided (e.g. for universe="sp500"), the view dict passed to fn
        is filtered to only the tickers returned by universe_fn(date).
        This gives point-in-time correct S&P 500 membership at each rebalance.
    """
    if bench_df.empty:
        raise ValueError(
            f"No price data for benchmark '{bench_name}'. "
            f"Try:  store.clear('{bench_name}')  and rerun.")

    idx = bench_df.loc[start:end].index
    if len(idx) == 0:
        raise ValueError(
            f"No trading days found between {start} and {end} "
            f"for benchmark '{bench_name}'. "
            "Verify your date range and that the benchmark has data.")

    if rebalance_freq == "D":
        rebal = set(idx)
    else:
        rs    = pd.Series(idx, index=idx).resample(rebalance_freq).last()
        rebal = set(rs.dropna().values)

    cash       = float(initial_capital)
    positions: Dict[str, dict] = {}
    trades_log: List[dict]     = []
    eq:         List[Tuple]    = []
    pending:    Optional[dict] = None

    def _nav(prices: dict) -> float:
        return cash + sum(
            p["shares"] * prices.get(t, p["entry_price"])
            for t, p in positions.items()
        )

    def _apply(target: dict, prices: dict, dt):
        nonlocal cash

        target = {t: w for t, w in target.items() if w > 0}
        total  = sum(target.values())
        if total > 1.0:
            target = {t: w / total for t, w in target.items()}

        nav = _nav(prices)

        for t in list(positions):
            if target.get(t, 0) == 0:
                pos   = positions.pop(t)
                px    = prices.get(t, pos["entry_price"]) * (1 - slippage)
                gross = pos["shares"] * px
                fee   = gross * commission
                cash += gross - fee
                trades_log.append({
                    "date": dt, "ticker": t, "action": "EXIT",
                    "shares": -pos["shares"], "price": px,
                    "dollars": gross, "commission": fee,
                    "return": px / pos["entry_price"] - 1,
                    "hold_days": (dt - pos["entry_date"]).days,
                })

        for t, w in target.items():
            if t not in prices or prices[t] <= 0:
                continue
            target_val = nav * w
            cur_val    = positions[t]["shares"] * prices[t] if t in positions else 0.0
            delta      = target_val - cur_val
            if abs(delta) < 1.0:
                continue

            if delta > 0:
                px   = prices[t] * (1 + slippage)
                cost = delta * (1 + commission)
                if cost > cash:
                    delta = cash / (1 + commission)
                    if delta < 1.0:
                        continue
                shr = delta / px
                fee = delta * commission
                cash -= delta + fee
                if t in positions:
                    old   = positions[t]
                    total_shr = old["shares"] + shr
                    positions[t]["entry_price"] = (
                        (old["shares"] * old["entry_price"] + shr * px) / total_shr
                    )
                    positions[t]["shares"] = total_shr
                else:
                    positions[t] = {"shares": shr, "entry_price": px, "entry_date": dt}
                trades_log.append({
                    "date": dt, "ticker": t, "action": "ENTER",
                    "shares": shr, "price": px,
                    "dollars": delta, "commission": fee,
                })

            else:
                pos = positions.get(t)
                if pos is None:
                    continue
                px    = prices[t] * (1 - slippage)
                shr   = min(abs(delta) / px, pos["shares"])
                gross = shr * px
                fee   = gross * commission
                cash += gross - fee
                trades_log.append({
                    "date": dt, "ticker": t, "action": "TRIM",
                    "shares": -shr, "price": px,
                    "dollars": gross, "commission": fee,
                    "return": px / pos["entry_price"] - 1,
                    "hold_days": (dt - pos["entry_date"]).days,
                })
                remaining = pos["shares"] - shr
                if remaining < 1e-8:
                    del positions[t]
                else:
                    positions[t]["shares"] = remaining

    for dt in idx:
        prices: Dict[str, float] = {}
        for t, df in bundle.items():
            if dt in df.index:
                v = df.loc[dt, "Close"]
            else:
                past = df.loc[:dt, "Close"].dropna()
                v    = past.iloc[-1] if not past.empty else np.nan
            if pd.notna(v) and v > 0:
                prices[t] = float(v)

        if pending is not None:
            _apply(pending, prices, dt)
            pending = None

        if dt in rebal:
            sigs = signals_by_date.get(dt, [])

            # Build point-in-time view — filter to current universe if sp500 mode
            if universe_fn is not None:
                try:
                    current_members = set(universe_fn(dt))
                except Exception:
                    current_members = set(bundle.keys())
                view = {
                    t: df.loc[:dt]
                    for t, df in bundle.items()
                    if t in current_members
                }
            else:
                view = {t: df.loc[:dt] for t, df in bundle.items()}

            try:
                weights = fn(dt, view, sigs) or {}
            except Exception as e:
                if verbose:
                    print(f"  [sim] Strategy error on {dt.date()}: {e}")
                weights = {}

            weights = {t: w for t, w in weights.items() if w > 0}

            if execution_lag > 0:
                pending = weights
            else:
                _apply(weights, prices, dt)

        eq.append((dt, _nav(prices)))

    equity = pd.Series({dt: v for dt, v in eq}, dtype=float, name="Portfolio")
    equity.index = pd.to_datetime(equity.index)

    bench_close  = bench_df["Close"].reindex(equity.index, method="ffill")
    bench_equity = bench_close / bench_close.iloc[0] * initial_capital
    bench_equity.name = bench_name

    result = Result(
        equity=equity, benchmark=bench_equity,
        trades=pd.DataFrame(trades_log), rf=rf,
        name=name, bench_name=bench_name,
        commission=commission, slippage=slippage, lag=execution_lag,
    )

    if verbose:
        m = result.metrics
        print(f"  -> CAGR {m['CAGR']:+.2%}  "
              f"MDD {m['Max Drawdown']:.1%}  "
              f"Sharpe {m['Sharpe Ratio']:.2f}  "
              f"trades {int(m['Num Trades'])}")

    return result


# ─────────────────────────────────────────────────────────────────────────────
# Store registry  -  one connection per unique db path
# ─────────────────────────────────────────────────────────────────────────────

_STORE_REGISTRY: Dict[str, DataStore] = {}


def _get_store(db_path: Optional[str] = None) -> DataStore:
    key = str(Path(db_path).resolve()) if db_path else "__default__"
    if key not in _STORE_REGISTRY:
        _STORE_REGISTRY[key] = DataStore(db_path) if db_path else DataStore()
    return _STORE_REGISTRY[key]


# ─────────────────────────────────────────────────────────────────────────────
# _backtest_inner()  -  internal single-run engine
# ─────────────────────────────────────────────────────────────────────────────

def _backtest_inner(
    fn:              Callable,
    *,
    universe:        Optional[Union[str, List[str]]] = None,
    tickers:         Optional[List[str]] = None,
    df:              Optional[pd.DataFrame] = None,
    date_col:        str   = "date",
    ticker_col:      Optional[str] = "ticker",
    start:           str   = "2018-01-01",
    end:             str   = "2024-01-01",
    benchmark:       str   = DEFAULT_BENCH,
    initial_capital: float = 100_000.0,
    commission:      float = 0.001,
    slippage:        float = 0.0005,
    execution_lag:   int   = 1,
    rebalance_freq:  str   = "D",
    lookback:        int   = 252,
    db_path:         Optional[str] = None,
    store:           Optional[DataStore] = None,
    verbose:         bool  = True,
) -> Result:
    name = fn.__name__
    ds   = store or _get_store(db_path)

    # ── Resolve ticker universe ───────────────────────────────────────────────
    # Priority: tickers= > universe= > fn.tickers attribute
    sp500_mode  = False
    universe_fn = None

    if tickers is not None:
        # Explicit list wins
        all_tickers: List[str] = list(tickers)

    elif universe == "sp500":
        sp500_mode = True
        universe_fn = _sp500_members_on
        if verbose:
            print(f"\n[{name}] Building S&P 500 constituent union "
                  f"({start} → {end})…")
        all_tickers = _sp500_union(start, end, verbose=verbose)

    elif isinstance(universe, list):
        all_tickers = list(universe)

    elif hasattr(fn, "tickers"):
        # Convenience: attach fn.tickers = [...] to the strategy function
        all_tickers = list(fn.tickers)

    else:
        all_tickers = []

    # Add any tickers from a signal DataFrame
    if df is not None and ticker_col and ticker_col in df.columns:
        for t in df[ticker_col].dropna().astype(str).str.strip().unique():
            if t and t not in all_tickers:
                all_tickers.append(t)

    if benchmark not in all_tickers:
        all_tickers.append(benchmark)

    if not all_tickers:
        raise ValueError(
            "No tickers found. Options:\n"
            "  1. fn.tickers = ['AAPL', 'MSFT']   (attach list to strategy fn)\n"
            "  2. universe='sp500'                 (S&P 500 point-in-time)\n"
            "  3. tickers=['AAPL', 'MSFT']         (explicit list to backtest())\n"
            "  4. df= with a ticker_col column     (event-driven signal)")

    if execution_lag not in (0, 1):
        raise ValueError(
            f"execution_lag must be 0 or 1, got {execution_lag}. "
            "Use 1 (next-day fill) for realistic backtests.")

    if verbose:
        mode_str = "S&P 500 (point-in-time)" if sp500_mode else "explicit list"
        print(f"\n[{name}] {start} -> {end}  bench={benchmark}  "
              f"lag={execution_lag}d  comm={commission:.2%}  "
              f"universe={mode_str}")
        print(f"  tickers ({len(all_tickers)}): "
              f"{all_tickers[:8]}{'…' if len(all_tickers) > 8 else ''}")

    ds.prefetch(all_tickers, start, end, lookback=lookback)

    bundle: Dict[str, pd.DataFrame] = {}
    missing: List[str] = []
    for t in all_tickers:
        price_df = ds.get(t, start, end, lookback=lookback)
        if price_df.empty:
            missing.append(t)
        else:
            bundle[t] = price_df

    if missing:
        warnings.warn(
            f"No price data for: {missing}. "
            "These tickers will be excluded. "
            "Check symbols and internet connection, or call store.clear(ticker) "
            "to force a fresh download.")

    if benchmark not in bundle:
        raise ValueError(
            f"Benchmark '{benchmark}' has no price data — cannot run backtest. "
            f"Try:  store.clear('{benchmark}')  then rerun.")

    bench_df = bundle[benchmark]

    try:
        rf_df = ds.get(RF_TICKER, start, end, lookback=0)
        rf    = float(rf_df["Close"].mean() / 100) if not rf_df.empty else 0.04
    except Exception:
        rf = 0.04

    signals_by_date: Dict[pd.Timestamp, List[dict]] = {}
    if df is not None:
        _df = df.copy()
        _df[date_col] = pd.to_datetime(_df[date_col], errors="coerce").dt.normalize()
        bad = int(_df[date_col].isna().sum())
        if bad:
            warnings.warn(
                f"{bad} row(s) in df have unparseable dates in column '{date_col}' "
                "and will be dropped.")
        _df = _df.dropna(subset=[date_col])
        for _, row in _df.iterrows():
            d = row[date_col]
            signals_by_date.setdefault(d, []).append(row.to_dict())

    return _simulate(
        fn=fn, bundle=bundle, bench_df=bench_df,
        signals_by_date=signals_by_date,
        start=start, end=end, rf=rf,
        name=name, bench_name=benchmark,
        initial_capital=initial_capital,
        commission=commission, slippage=slippage,
        execution_lag=execution_lag,
        rebalance_freq=rebalance_freq,
        verbose=verbose,
        universe_fn=universe_fn,
    )


# ─────────────────────────────────────────────────────────────────────────────
# backtest()  -  the only public entry point
# ─────────────────────────────────────────────────────────────────────────────

def backtest(
    fn:              Callable,
    *,
    universe:        Optional[Union[str, List[str]]] = None,
    tickers:         Optional[List[str]] = None,
    df:              Optional[pd.DataFrame] = None,
    date_col:        str   = "date",
    ticker_col:      Optional[str] = "ticker",
    start:           str   = "2018-01-01",
    end:             str   = "2024-01-01",
    benchmark:       str   = DEFAULT_BENCH,
    initial_capital: float = 100_000.0,
    commission:      float = 0.001,
    slippage:        float = 0.0005,
    execution_lag:   int   = 1,
    rebalance_freq:  str   = "D",
    lookback:        int   = 252,
    db_path:         Optional[str] = None,
    store:           Optional[DataStore] = None,
    verbose:         bool  = True,
) -> Result:
    """
    Run a backtest. The only function you need to import.

    Parameters
    ──────────
    fn              Strategy: fn(date, data, signals) -> {ticker: weight}
                    Long-only — negative weights are silently ignored.

    universe        Ticker universe. Three options (pick one):
                    ┌─────────────────────────────────────────────────────────┐
                    │  "sp500"       S&P 500 with point-in-time constituents  │
                    │                (requires betteryf_1d.py alongside this  │
                    │                 file). Avoids survivorship bias.        │
                    │                                                          │
                    │  ["AAPL",...]  Explicit ticker list (alias for tickers=) │
                    │                                                          │
                    │  None          Falls back to tickers= or fn.tickers     │
                    └─────────────────────────────────────────────────────────┘
                    You can also skip this entirely and just set:
                      fn.tickers = ["AAPL", "MSFT"]
                    The framework picks it up automatically.

    tickers         Explicit ticker list (overrides universe= and fn.tickers).
    df              Optional signal DataFrame (PEAD table, ML scores, events…).
    date_col        Column in df containing signal/event dates.
    ticker_col      Column in df containing ticker symbols (None = no auto-fetch).
    start / end     Backtest date range (ISO strings, inclusive).
    benchmark       Ticker used as benchmark (default "SPY").
    initial_capital Starting NAV in USD (default 100,000).
    commission      Fraction of trade value charged per fill (e.g. 0.001 = 10 bps).
    slippage        Price friction per side (e.g. 0.0005 = 5 bps).
    execution_lag   1 = fill at next-day close (default, look-ahead safe).
                    0 = fill at same-day close (only use for MOC strategies).
    rebalance_freq  "D" daily | "W" weekly | "ME" monthly end.
    lookback        Trading days of OHLCV history needed before `start` for
                    indicator warm-up. Automatically padded to calendar days.
    db_path         Custom SQLite cache path (default: ./.backtest_cache/).
    store           Pass an existing DataStore to share the cache across calls.
    verbose         Print progress and a one-line summary.

    Returns
    ───────
    Result
        .equity      pd.Series  — daily portfolio value
        .benchmark   pd.Series  — daily benchmark value
        .trades      pd.DataFrame — every fill
        .metrics     dict       — all metrics (lazily computed)
        .summary()   print full stats table
        .plot()      5-panel performance chart

    Examples
    ────────
    # 1. Attach tickers to fn — no duplication
    def my_fn(date, data, signals): ...
    my_fn.tickers = ["AAPL", "MSFT", "NVDA"]

    result = backtest(my_fn, start="2020-01-01", end="2024-01-01")

    # 2. S&P 500 universe (point-in-time, survivorship-bias free)
    result = backtest(my_fn, universe="sp500",
                      start="2015-01-01", end="2024-01-01")

    # 3. Signal DataFrame (PEAD / events / alt-data)
    result = backtest(my_fn, df=signals_df, date_col="entry_date",
                      ticker_col="Ticker", start="2013-01-01", end="2025-01-01")

    # 4. Shared DataStore across multiple strategy runs
    store    = DataStore()
    result_a = backtest(fn_a, universe=["SPY","QQQ"], store=store, ...)
    result_b = backtest(fn_b, universe=["SPY","QQQ"], store=store, ...)
    """
    t0 = time.perf_counter()

    result = _backtest_inner(
        fn,
        universe=universe, tickers=tickers,
        df=df, date_col=date_col, ticker_col=ticker_col,
        start=start, end=end, benchmark=benchmark,
        initial_capital=initial_capital,
        commission=commission, slippage=slippage,
        execution_lag=execution_lag, rebalance_freq=rebalance_freq,
        lookback=lookback, db_path=db_path, store=store, verbose=verbose,
    )

    elapsed = time.perf_counter() - t0
    if verbose:
        print(f"  [backtest] done in {elapsed:.1f}s")

    return result


# ─────────────────────────────────────────────────────────────────────────────
# Private helpers
# ─────────────────────────────────────────────────────────────────────────────

def _f(v) -> float:
    """Safe float conversion — returns nan for non-finite values."""
    try:
        f = float(v)
        return f if np.isfinite(f) else np.nan
    except (TypeError, ValueError):
        return np.nan


def _sax(ax, C: dict):
    """Apply SEED dark-theme styling to a matplotlib Axes."""
    ax.set_facecolor(C["panel"])
    ax.tick_params(colors=C["text"], labelsize=7)
    for sp in ax.spines.values():
        sp.set_color(C["grid"])
    ax.grid(True, color=C["grid"], lw=0.5, alpha=0.7)


def _heatmap(ax, r: pd.Series, C: dict):
    """Render a monthly-returns heatmap on ax."""
    mon = r.resample("ME").apply(lambda x: (1 + x).prod() - 1)
    if mon.empty:
        ax.set_visible(False)
        return
    df  = mon.to_frame("r")
    df["yr"] = df.index.year
    df["mo"] = df.index.month
    yrs = sorted(df["yr"].unique())
    mat = np.full((len(yrs), 12), np.nan)
    for _, row in df.iterrows():
        mat[yrs.index(int(row["yr"])), int(row["mo"]) - 1] = row["r"]
    vm  = np.nanmax(np.abs(mat)) or 0.05
    im  = ax.imshow(mat, aspect="auto", cmap="RdYlGn",
                    vmin=-vm, vmax=vm, alpha=0.85)
    ax.set_xticks(range(12))
    ax.set_xticklabels(list("JFMAMJJASOND"), fontsize=7, color=C["text"])
    ax.set_yticks(range(len(yrs)))
    ax.set_yticklabels(yrs, fontsize=6, color=C["text"])
    ax.set_title("Monthly Returns", color=C["text"], fontsize=9, pad=5)
    ax.set_facecolor(C["panel"])
    for sp in ax.spines.values():
        sp.set_color(C["grid"])
    for yi in range(len(yrs)):
        for mi in range(12):
            v = mat[yi, mi]
            if not np.isnan(v):
                ax.text(mi, yi, f"{v:.1%}", ha="center", va="center",
                        fontsize=4.5,
                        color="white" if abs(v) > vm * 0.5 else C["text"])
    plt.colorbar(im, ax=ax, fraction=0.015, pad=0.02,
                 format=FuncFormatter(lambda x, _: f"{x:.0%}"))


# ─────────────────────────────────────────────────────────────────────────────
# Public API
# ─────────────────────────────────────────────────────────────────────────────

__all__ = [
    # The one function you need
    "backtest",
    # DataStore — only needed when sharing a cache across multiple runs
    "DataStore",
    # Result and Stats — for type hints and direct metric access
    "Result", "Stats",
    # Individual metric functions — useful for custom post-processing
    "cagr", "sharpe", "sortino", "max_dd", "calmar",
    "beta", "jensens_alpha", "info_ratio", "dd_series",
    # SEED color palette — use in your own matplotlib charts for consistency
    "SEED_COLORS",
]