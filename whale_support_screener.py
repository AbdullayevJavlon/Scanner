            r = requests.post(f"https://api.telegram.org/bot{BOT_TOKEN}/sendMessage",
                              json={"chat_id": CHAT_ID, "text": text[i:i + 3800], "parse_mode": "Markdown"}, timeout=15)
            print("Telegram:", r.status_code, r.text[:200])"""
Binance USDT screener v1
1) [blackcat] Whale Pump (sariq sham) - 15m, 4h, 1d
2) Support zonasida turgan coinlar - 15m, 4h, 1d
3) Hajm + sdelkalar soni oshishi - 15m
Telegram alert, har 15 minutda (sham yopilgandan keyin).

Colab: pip install requests pandas numpy
Telegram: TG_BOT_TOKEN va TG_CHAT_ID ni Colab "Secrets" ga (kalit belgisi) qo'shing.
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

TOP_N = 150                    # 24s hajm bo'yicha eng katta N ta USDT juftlik
MIN_QUOTE_VOL_24H = 3_000_000  # 24s hajmi shundan kam coinlar olinmaydi (USDT)

# --- Whale Pump (sariq sham) ---
WHALE_MIN = 1.0       # shu qiymatdan katta bo'lsa "sariq sham" deymiz (TradingView bilan solishtirib sozlang)
WHALE_LOOKBACK = 1    # oxirgi nechta YOPILGAN shamda sariq sham qidiriladi (1 = faqat oxirgisi)

# --- Support ---
PIVOT_LEFT = 5
PIVOT_RIGHT = 5
CLUSTER_PCT = 0.6                                  # shu % ichidagi pivotlar bitta zona
MIN_TOUCHES = 2                                    # zona kamida necha marta tegilgan
SUP_TOL_PCT = {"15m": 0.5, "4h": 1.0, "1d": 1.5}   # narx zonadan shu % yuqorigacha bo'lsa "supportda"
SUP_BREAK_PCT = 0.3                                # zonadan shu % pastga tushsa, support sinadi

# --- Hajm va sdelka ---
SURGE_TF = "15m"
SURGE_BASE = 20       # oxirgi 20 sham o'rtachasi bilan solishtiramiz
VOL_X = 2.0           # hajm kamida shuncha baravar
TRD_X = 2.0           # sdelkalar soni kamida shuncha baravar

# --- Alert sharti ---
MIN_YELLOW_TF = 1     # kamida nechta taymfreymda sariq sham
MIN_SUPPORT_TF = 1    # kamida nechta taymfreymda support
REQUIRE_SURGE = True  # hajm+sdelka oshishi shart

DEBUG = False         # True: har coin uchun whale qiymatlarini chiqaradi (kalibrlash uchun)

# Chiqarib tashlanadigan coinlar (o'zingizdagi eski ro'yxatni shu yerga qo'shing)
STABLES = {"USDC", "FDUSD", "TUSD", "USDP", "DAI", "USDE", "BUSD", "EUR", "AEUR", "XUSD", "USD1", "PYUSD", "RLUSD", "UST", "USTC", "EURI", "BFUSD"}
STOCKS = {"TSLA", "AAPL", "NVDA", "COIN", "MSTR", "HOOD", "CRCL", "AMZN", "GOOGL", "META", "MSFT"}
EXCLUDE = STABLES | STOCKS

# Binance public ma'lumot manzillari (Colab AQSh serverida api.binance.com bloklanishi mumkin)
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


BOT_TOKEN = _secret("TG_BOT_TOKEN")
CHAT_ID = _secret("TG_CHAT_ID")


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
    df = df[df["close_time"] < now_ms].reset_index(drop=True)  # faqat YOPILGAN shamlar
    return df


# =========================== INDIKATOR: WHALE PUMP ===========================
def xsa(src, length, wei):
    """TDX SMA(X,N,M) - Pine'dagi xsa funksiyasining aynan nusxasi."""
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
    var1[0] = np.nan                                   # xrf(low,1) = oldingi sham low'i
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
    """Qaytaradi: (supportdami, zona narxi, tegishlar soni, zonadan masofa %)"""
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


def fmt(r):
    ys = "".join(f"{tf} " for tf in TFS if r["yellow"].get(tf)) or "-"
    ss = "".join(f"{tf} " for tf in TFS if r["support"].get(tf, (False,))[0]) or "-"
    rsi4 = r["rsi"].get("4h")
    return (f"*{r['base']}/USDT*  (ball {r['score']})\n"
            f"🟡 sariq: {ys}\n"
            f"🟢 support: {ss}\n"
            f"⚡ hajm x{r.get('vol_x', 0):.1f} | sdelka x{r.get('trd_x', 0):.1f}\n"
            f"RSI 4h: {rsi4:.0f}" if rsi4 is not None else f"*{r['base']}/USDT*")


def telegram(text):
    if not BOT_TOKEN or not CHAT_ID:
        print("[Telegram sozlanmagan] \n" + text)
        return
    for i in range(0, len(text), 3800):
        try:
            r = requests.post(f"https://api.telegram.org/bot{BOT_TOKEN}/sendMessage",
                              json={"chat_id": CHAT_ID, "text": text[i:i + 3800], "parse_mode": "Markdown"}, timeout=15)
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


alerted = set()  # (symbol, 15m sham vaqti) - bir xil signalni qayta yubormaslik


def cycle():
    results = run_scan()
    print_table(results)
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


def sleep_to_next_15m():
    now = time.time()
    nxt = (int(now // 900) + 1) * 900 + 8   # sham yopilgandan 8 soniya keyin
    time.sleep(max(nxt - now, 1))


if __name__ == "__main__":
    cycle()                 # darrov bir marta
    if os.getenv("RUN_ONCE"):   # GitHub Actions: bir marta ishlab to'xtaydi
        raise SystemExit(0)
    while True:             # keyin har 15 minutda
        sleep_to_next_15m()
        try:
            cycle()
        except Exception as e:
            print("Sikl xatosi:", e)
