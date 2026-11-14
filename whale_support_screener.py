"""
Binance USDT screener v3
1) Asosiy alert: sariq sham + support + hajm (15m, 4h, 1d)
2) Alohida alert: 15m da ketma-ket FRESH_N-sariq ustun
Telegram alert, har 15 minutda (sham yopilgandan keyin).
"""
import os
import time
import requests
import numpy as np
import pandas as pd
from concurrent.futures import ThreadPoolExecutor
from datetime import datetime, timezone

# =========================== SOZLAMALAR ===========================
TFS = ["15m", "4h", "1d"]

TOP_N = 400
MIN_QUOTE_VOL_24H = 500_000

# --- Whale Pump (sariq sham) ---
WHALE_MIN = 1.0
WHALE_LOOKBACK = 1

# --- Support ---
PIVOT_LEFT = 5
PIVOT_RIGHT = 5
CLUSTER_PCT = 0.6
MIN_TOUCHES = 2
SUP_TOL_PCT = {"15m": 0.5, "4h": 1.0, "1d": 1.5}
SUP_BREAK_PCT = 0.3

# --- Hajm va sdelka ---
SURGE_TF = "15m"
SURGE_BASE = 20
VOL_X = 1.5
TRD_X = 1.5

# --- Asosiy alert sharti ---
MIN_YELLOW_TF = 1
MIN_SUPPORT_TF = 1
REQUIRE_SURGE = True

# --- Alohida alert: 15m da ketma-ket sariq ustunlar ---
FRESH15 = True
WP15_MIN = 1.0     # ustun shu qiymatdan oshsa "sariq ustun" deymiz
WP15_MAX = 500     # bundan katta qiymat hisob xatosi
FRESH_N = 5        # nechanchi ketma-ket ustunda xabar beradi
FRESH_TOP = 8      # bir xabarda ko'pi bilan nechta coin

DEBUG = True

STABLES = {"U", "XAUT", "PAXG", "USDC", "FDUSD", "TUSD", "USDP", "DAI", "USDE", "BUSD", "EUR", "AEUR", "XUSD", "USD1", "PYUSD", "RLUSD", "UST", "USTC", "EURI", "BFUSD"}
STOCKS = {"TSLA", "AAPL", "NVDA", "COIN", "MSTR", "HOOD", "CRCL", "AMZN", "GOOGL", "META", "MSFT"}
EXCLUDE = STABLES | STOCKS

BASES = ["https://data-api.binance.vision", "https://api.binance.com", "https://api1.binance.com"]


# =========================== SECRETS ===========================
def _secret(name):
    v = os.getenv(name, "")
    if v:
        return v
    try:
        from google.colab import userdata
        return userdata.get(name) or ""
    except Exception:
        return ""


BOT_TOKEN = _secret("TG_BOT_TOKEN").strip().strip('"').strip("'")
if BOT_TOKEN.lower().startswith("bot"):
    BOT_TOKEN = BOT_TOKEN[3:]
CHAT_ID = _secret("TG_CHAT_ID").strip()


# =========================== BINANCE ===========================
def api_get(path, params=None):
    last = None
    for base in BASES:
        try:
            r = requests.get(base + path, params=params, timeout=15)
            if r.status_code == 200:
                return r.json()
            last = f"{base} -> HTTP {r.status_code}"
        except Exception as e:
            last = f"{base} -> {e}"
    raise RuntimeError(f"Binance so'rovi muvaffaqiyatsiz: {last}")


def get_symbols():
    tick = api_get("/api/v3/ticker/24hr")
    info = api_get("/api/v3/exchangeInfo")
    trading = {s["symbol"]: s["baseAsset"] for s in info["symbols"]
               if s["quoteAsset"] == "USDT" and s["status"] == "TRADING" and s.get("isSpotTradingAllowed")}
    rows = []
    for t in tick:
        sym = t["symbol"]
        if sym not in trading:
            continue
        base = trading[sym]
        if base in EXCLUDE or base.endswith(("UP", "DOWN", "BULL", "BEAR")):
            continue
        qv = float(t["quoteVolume"])
        if qv >= MIN_QUOTE_VOL_24H:
            rows.append((sym, base, qv))
    rows.sort(key=lambda x: -x[2])
    return rows[:TOP_N]


def get_klines(symbol, interval, limit=500):
    data = api_get("/api/v3/klines", {"symbol": symbol, "interval": interval, "limit": limit})
    df = pd.DataFrame(data, columns=["open_time", "open", "high", "low", "close", "volume", "close_time",
                                     "quote_vol", "trades", "tb_base", "tb_quote", "ignore"])
    for c in ["open", "high", "low", "close", "volume", "quote_vol", "trades"]:
        df[c] = df[c].astype(float)
    now_ms = int(time.time() * 1000)
    df = df[df["close_time"] < now_ms].reset_index(drop=True)
    return df


def day_info(symbol):
    """Qaytaradi: (hozirgi narx, bugungi kunlik sham ochilishidan o'zgarish %)."""
    try:
        k = api_get("/api/v3/klines", {"symbol": symbol, "interval": "1d", "limit": 1})[0]
        o, c = float(k[1]), float(k[4])
        return c, (c / o - 1) * 100
    except Exception:
        return None, None


# =========================== INDIKATOR: WHALE PUMP ===========================
def xsa(src, length, wei):
    n = len(src)
    out = np.full(n, np.nan)
    sumf = np.zeros(n)
    for i in range(n):
        prev = sumf[i - 1] if i > 0 and not np.isnan(sumf[i - 1]) else 0.0
        old = src[i - length] if i >= length and not np.isnan(src[i - length]) else 0.0
        sumf[i] = prev - old + src[i]
        ma = sumf[i] / length if (i >= length and not np.isnan(src[i - length])) else np.nan
        if i == 0 or np.isnan(out[i - 1]):
            out[i] = ma
        else:
            out[i] = (src[i] * wei + out[i - 1] * (length - wei)) / length
    return out


def ema(arr, span):
    return pd.Series(arr).ewm(span=span, adjust=False).mean().values


def whale_pump(df, fast_len=38, base_len=90):
    low = df["low"].values
    var1 = np.roll(low, 1)
    var1[0] = np.nan
    up = np.maximum(low - var1, 0)
    num = xsa(np.abs(low - var1), 3, 1)
    den = xsa(up, 3, 1)
    with np.errstate(divide="ignore", invalid="ignore"):
        var2 = num / den * 100
    var2[~np.isfinite(var2)] = np.nan
    var3 = ema(var2 * 10, 3)
    s_low = pd.Series(low)
    var4 = s_low.rolling(fast_len, min_periods=fast_len).min().values
    var5 = pd.Series(var3).rolling(fast_len, min_periods=fast_len).max().values
    var6 = (s_low.rolling(base_len, min_periods=base_len).min().values != 0).astype(float)
    cond = low <= var4
    inner = np.where(cond, (var3 + var5 * 2) / 2, 0.0)
    return ema(inner, 3) / 618 * var6


# =========================== SUPPORT ===========================
def find_support_zones(df):
    lows = df["low"].values
    n = len(lows)
    piv = []
    for i in range(PIVOT_LEFT, n - PIVOT_RIGHT):
        if lows[i] == lows[i - PIVOT_LEFT:i + PIVOT_RIGHT + 1].min():
            piv.append(lows[i])
    piv.sort()
    zones, grp = [], []
    for p in piv:
        if grp and (p - grp[0]) / grp[0] * 100 > CLUSTER_PCT:
            zones.append((float(np.mean(grp)), len(grp)))
            grp = []
        grp.append(p)
    if grp:
        zones.append((float(np.mean(grp)), len(grp)))
    return [z for z in zones if z[1] >= MIN_TOUCHES]


def check_support(df, tf):
    zones = find_support_zones(df)
    c = df["close"].iloc[-1]
    lo = df["low"].iloc[-1]
    tol = SUP_TOL_PCT[tf]
    best = None
    for level, touches in zones:
        if c >= level * (1 - SUP_BREAK_PCT / 100) and lo <= level * (1 + tol / 100) and c <= level * (1 + 2 * tol / 100):
            dist = (c - level) / level * 100
            if best is None or abs(dist) < abs(best[3]):
                best = (True, level, touches, dist)
    return best if best else (False, None, 0, None)


# =========================== RSI ===========================
def rsi(close, n=14):
    d = close.diff()
    up = d.clip(lower=0).ewm(alpha=1 / n, adjust=False).mean()
    dn = (-d.clip(upper=0)).ewm(alpha=1 / n, adjust=False).mean()
    rs = up / dn.replace(0, np.nan)
    return 100 - 100 / (1 + rs)


# =========================== SKANER ===========================
def scan_symbol(item):
    sym, base, _ = item
    res = {"symbol": sym, "base": base, "yellow": {}, "support": {}, "rsi": {}, "whale": {}}
    try:
        for tf in TFS:
            df = get_klines(sym, tf)
            if len(df) < 120:
                continue
            wp = whale_pump(df)
            recent = wp[-WHALE_LOOKBACK:]
            res["whale"][tf] = float(np.nanmax(recent)) if len(recent) else 0.0
            res["yellow"][tf] = bool(res["whale"][tf] > WHALE_MIN)
            if tf == "15m":
                streak = 0
                for v in wp[::-1]:
                    if np.isnan(v) or v < WP15_MIN or v > WP15_MAX:
                        break
                    streak += 1
                res["wp15"] = float(wp[-1])
                res["wp15_streak"] = streak
            ok, level, touches, dist = check_support(df, tf)
            res["support"][tf] = (ok, level, touches, dist)
            res["rsi"][tf] = float(rsi(df["close"]).iloc[-1])
            if tf == SURGE_TF:
                qv, tr = df["quote_vol"].values, df["trades"].values
                base_qv = qv[-SURGE_BASE - 1:-1].mean()
                base_tr = tr[-SURGE_BASE - 1:-1].mean()
                res["vol_x"] = float(qv[-1] / base_qv) if base_qv > 0 else 0.0
                res["trd_x"] = float(tr[-1] / base_tr) if base_tr > 0 else 0.0
                res["candle_time"] = int(df["open_time"].iloc[-1])
        res["surge"] = res.get("vol_x", 0) >= VOL_X and res.get("trd_x", 0) >= TRD_X
        return res
    except Exception as e:
        print(f"  xato {sym}: {e}")
        return None


def run_scan():
    symbols = get_symbols()
    print(f"{datetime.now().strftime('%H:%M:%S')} | {len(symbols)} ta coin skanerlanmoqda...")
    with ThreadPoolExecutor(max_workers=8) as ex:
        results = [r for r in ex.map(scan_symbol, symbols) if r]
    out = []
    for r in results:
        y = sum(r["yellow"].values())
        s = sum(1 for v in r["support"].values() if v[0])
        r["y_count"], r["s_count"] = y, s
        r["score"] = y * 2 + s + (2 if r.get("surge") else 0)
        out.append(r)
    return out


def passes(r):
    if r["y_count"] < MIN_YELLOW_TF or r["s_count"] < MIN_SUPPORT_TF:
        return False
    if REQUIRE_SURGE and not r.get("surge"):
        return False
    return True


def is_fresh(r):
    return r.get("wp15_streak", 0) == FRESH_N


def tv_link(sym):
    return f"https://www.tradingview.com/chart/?symbol=BINANCE:{sym}"


def fmt(r):
    ys = "".join(f"{tf} " for tf in TFS if r["yellow"].get(tf)) or "-"
    ss = "".join(f"{tf} " for tf in TFS if r["support"].get(tf, (False,))[0]) or "-"
    rsi4 = r["rsi"].get("4h")
    _, ch = day_info(r["symbol"])
    chs = f"{ch:+.1f}%" if ch is not None else "-"
    txt = (f"*{r['base']}/USDT*  (ball {r['score']})\n"
           f"📈 kun: {chs}\n"
           f"🟡 sariq: {ys}\n"
           f"🟢 support: {ss}\n"
           f"⚡ hajm x{r.get('vol_x', 0):.1f} | sdelka x{r.get('trd_x', 0):.1f}")
    if rsi4 is not None:
        txt += f"\nRSI 4h: {rsi4:.0f}"
    txt += f"\n[📈 TradingView'da ko'rish]({tv_link(r['symbol'])})"
    return txt


def fmt_fresh(r):
    price, ch = day_info(r["symbol"])
    ps = f"{price:.6g}" if price is not None else "-"
    chs = f"{ch:+.1f}%" if ch is not None else "-"
    return (f"*{r['base']}/USDT*\n"
            f"💰 narx: {ps}\n"
            f"📈 kun: {chs}\n"
            f"🟡 15m: {FRESH_N}-ustun ({r['wp15']:.2f})\n"
            f"[📈 TradingView'da ko'rish]({tv_link(r['symbol'])})")


def telegram(text):
    if not BOT_TOKEN or not CHAT_ID:
        print("[Telegram sozlanmagan] \n" + text)
        return
    for i in range(0, len(text), 3800):
        try:
            r = requests.post(f"https://api.telegram.org/bot{BOT_TOKEN}/sendMessage",
                              json={"chat_id": CHAT_ID, "text": text[i:i + 3800], "parse_mode": "Markdown",
                                    "disable_web_page_preview": True}, timeout=15)
            print("Telegram:", r.status_code, r.text[:200])
        except Exception as e:
            print("Telegram xato:", e)


def print_table(results):
    rows = []
    for r in sorted(results, key=lambda x: -x["score"])[:25]:
        rows.append({
            "coin": r["base"], "ball": r["score"],
            "sariq": "".join(tf + " " for tf in TFS if r["yellow"].get(tf)),
            "support": "".join(tf + " " for tf in TFS if r["support"].get(tf, (False,))[0]),
            "hajm_x": round(r.get("vol_x", 0), 1), "sdelka_x": round(r.get("trd_x", 0), 1),
            "whale": {tf: round(v, 2) for tf, v in r["whale"].items()} if DEBUG else "",
        })
    print(pd.DataFrame(rows).to_string(index=False))


alerted = set()


def cycle():
    results = run_scan()
    print_table(results)
    seq = [f"{r['base']} {r['wp15_streak']}" for r in results if r.get("wp15_streak", 0) >= 1]
    print("15m ustun seriyalari:", ", ".join(seq) or "yo'q")

    # 1) asosiy alert
    hits = [r for r in results if passes(r)]
    new = [r for r in hits if (r["symbol"], r.get("candle_time")) not in alerted]
    for r in new:
        alerted.add((r["symbol"], r.get("candle_time")))
    if new:
        new.sort(key=lambda x: -x["score"])
        head = f"🔔 {datetime.now(timezone.utc).strftime('%H:%M')} UTC | {len(new)} ta signal\n\n"
        telegram(head + "\n\n".join(fmt(r) for r in new))
    else:
        print("Yangi signal yo'q.")

    # 2) alohida alert: 15m da ketma-ket FRESH_N-ustun
    if FRESH15:
        fresh = [r for r in results
                 if is_fresh(r) and (r["symbol"], "wp15", r.get("candle_time")) not in alerted]
        fresh.sort(key=lambda x: -x["wp15"])
        fresh = fresh[:FRESH_TOP]
        for r in fresh:
            alerted.add((r["symbol"], "wp15", r.get("candle_time")))
        if fresh:
            head = f"🟡 15m seriya: {FRESH_N}-ustun | {len(fresh)} ta\n\n"
            telegram(head + "\n\n".join(fmt_fresh(r) for r in fresh))
        else:
            print("Yangi ustun yo'q.")


def sleep_to_next_15m():
    now = time.time()
    nxt = (int(now // 900) + 1) * 900 + 8
    time.sleep(max(nxt - now, 1))


if __name__ == "__main__":
    cycle()
    if os.getenv("RUN_ONCE"):
        raise SystemExit(0)
    while True:
        sleep_to_next_15m()
        try:
            cycle()
        except Exception as e:
            print("Sikl xatosi:", e)
