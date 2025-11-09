# bot.py
# Tyre Pyrolysis Assistant (Group-ready)
# Feed -> plan + expected oil %, Actuals -> gap + refined prediction
# Requirements: python-telegram-bot>=21.0, pandas, numpy, openpyxl

import os, re, logging
from typing import Dict, Tuple, List
import numpy as np
import pandas as pd
from telegram import Update
from telegram.ext import Application, CommandHandler, MessageHandler, ContextTypes, filters

# ── Logging ────────────────────────────────────────────────────────────────────
logging.basicConfig(level=logging.INFO, format="%(asctime)s | %(levelname)s | %(name)s | %(message)s")
log = logging.getLogger("pyro_bot")

# ── Environment ────────────────────────────────────────────────────────────────
TELEGRAM_BOT_TOKEN = os.environ.get("TELEGRAM_BOT_TOKEN", "")
REPORT_PATH = os.environ.get("REPORT_PATH", "pyrolysis_feed_temp_ZONE_TIME_report.xlsx")
ENV_ALLOWED = os.environ.get("ALLOWED_CHAT_ID", "").strip()
ALLOWED_CHAT_ID = int(ENV_ALLOWED) if ENV_ALLOWED else None  # set in Railway

# ── Help text ─────────────────────────────────────────────────────────────────
HELP = (
    "Commands:\n"
    "/predict <feed>  → plan + expected oil %\n"
    "/plan <feed>     → plan only\n"
    "/actual <zones>  → compare vs plan + refined oil %\n"
    "/reload          → reload Excel rules\n\n"
    "Examples:\n"
    "/predict Radial=5115 Nylon=600 Chips=3465 Powder=1490\n"
    "/actual 50-200=02:55, 200-300=01:10, 300-400=02:55, 400-450=01:20, 450-480=00:20, 300-400 separator=02:45"
)

# ── Plant priors (you can tune) ───────────────────────────────────────────────
# Oil potential by feed type (weighted base, from your history)
OIL_POTENTIAL = {
    "radial": 47.0,
    "nylon":  43.0,
    "chips":  45.0,
    "powder": 41.0,
    "kachra": 38.0,
    "others": 42.0,
}

# Zone effect on oil (delta vs. plan) in **% oil per 30 minutes** (empirical)
# Positive = more time raises oil; negative = more time reduces oil (over-cracking).
EFFECT_PER_30_MIN = {
    "50-200 reactor":   +0.2,   # smooth devolatilization helps a bit
    "200-300 reactor":  +0.5,
    "300-400 reactor":  +0.9,   # main oil window (strongest)
    "400-450 reactor":  -0.4,   # too long -> cracking
    "450-480 reactor":  -0.8,
    "480-500 reactor":  -1.0,
    "500-520 reactor":  -1.0,
    "300-400 separator":+0.4,   # better condensation improves liquid take
}
# Small overall uplift when you follow the recommended plan cleanly
PLAN_UPLIFT_PERCENT = 2.0  # +2.0% typical bump when near plan

# ── Utils ─────────────────────────────────────────────────────────────────────
def to_hhmmss(minutes: float) -> str:
    if minutes is None or (isinstance(minutes, float) and np.isnan(minutes)): return "-"
    total = int(round(minutes * 60))
    h, m = divmod(total, 60)
    return f"{h:02d}:{m:02d}:00"

def hhmmss_to_minutes(s: str) -> float:
    s = s.strip()
    if ":" in s:
        parts = [float(x) for x in s.split(":")]
        if len(parts) == 3: h, m, sec = parts; return h*60 + m + sec/60
        if len(parts) == 2: m, sec = parts;  return m + sec/60
    return float(s)

def normalize_zone_key(s: str) -> str:
    x = s.strip().lower()
    x = x.replace("°c","").replace("temp","").replace("zone","")
    x = x.replace("minutes","").replace("mins","").replace("min","")
    x = re.sub(r"\s+"," ", x).strip()
    which = "separator" if "separator" in x else "reactor"
    m = re.search(r"(\d{2,3}\s*[-–]\s*\d{2,3})", x)
    if m:
        window = m.group(1).replace(" ", "").replace("–","-")
    else:
        nums = re.findall(r"\d+", x)
        window = f"{nums[0]}-{nums[1]}" if len(nums)>=2 else x
    return f"{window} {which}"

def parse_keyvals(text: str) -> Dict[str, float]:
    parts = re.split(r"[,\n]+|\s{2,}", text.strip())
    out = {}
    for p in parts:
        if "=" not in p: continue
        k, v = p.split("=", 1)
        out[k.strip()] = v.strip()
    return out

# ── Engine (Excel loader) ─────────────────────────────────────────────────────
class RecoEngine:
    SHEET = "ZoneTime_Recommendations"

    def __init__(self, path: str):
        self.path = path
        self.zone_map: Dict[str, float] = {}
        self._load()

    def _load(self):
        self.zone_map.clear()
        try:
            df = pd.read_excel(self.path, sheet_name=self.SHEET)
            for _, row in df.iterrows():
                feat = str(row.get("zone_time_feature", "")).strip()
                sug  = row.get("suggested_minutes", np.nan)
                if not feat: continue
                low = feat.lower()
                m = re.search(r"(\d{2,3}\s*[-–]\s*\d{2,3})", low)
                if not m: continue
                rng = m.group(1).replace(" ", "").replace("–","-")
                which = "separator" if "separator" in low else "reactor"
                self.zone_map[f"{rng} {which}"] = float(sug) if pd.notna(sug) else np.nan
            logging.info("Loaded %d zone rules from %s", len(self.zone_map), self.path)
        except Exception:
            logging.exception("Failed to read Excel; using safe defaults")
            self.zone_map.update({
                "50-200 reactor": 165.0,
                "200-300 reactor": 65.0,
                "300-400 reactor": 170.0,
                "400-450 reactor": 75.0,
                "450-480 reactor": 20.0,
                "480-500 reactor": 0.0,
                "500-520 reactor": 0.0,
                "300-400 separator": 160.0,
            })

    def reload(self) -> int:
        self._load()
        return len(self.zone_map)

    # Plan is currently global (you can extend to presets later)
    def plan_for_feed(self, feed_line: str) -> Dict[str, float]:
        return dict(self.zone_map)

    # Parse helpers
    def parse_feed(self, feed_line: str) -> Dict[str, float]:
        txt = feed_line.replace("Feed:", "").replace("/predict", "").replace("/plan","").strip()
        kv = parse_keyvals(txt)
        out = {}
        for k, v in kv.items():
            name = k.strip().lower()
            m = re.search(r"([0-9]*\.?[0-9]+)\s*([tTkKgG])?", v)
            if not m: continue
            val = float(m.group(1))
            unit = (m.group(2) or "").lower()
            if unit == "t": val *= 1000.0
            # normalize keys
            name = {"chip":"chips","other":"others"}.get(name, name)
            out[name] = out.get(name, 0.0) + val
        return out

    def parse_actuals(self, actual_line: str) -> Dict[str, float]:
        txt = actual_line.replace("Actual:", "").replace("/actual", "").strip()
        kv = parse_keyvals(txt)
        out = {}
        for k, v in kv.items():
            key = normalize_zone_key(k)
            try:
                minutes = hhmmss_to_minutes(v) if ":" in v else float(v)
                out[key] = minutes
            except: pass
        return out

# ── Prediction math ───────────────────────────────────────────────────────────
def feed_weighted_potential(feed_kg: Dict[str, float]) -> float:
    tot = sum(feed_kg.values()) or 1.0
    oil = 0.0
    for k, kg in feed_kg.items():
        base = OIL_POTENTIAL.get(k.lower(), OIL_POTENTIAL["others"])
        oil += (kg/tot) * base
    return float(oil)

def plan_efficiency_bonus(plan: Dict[str, float]) -> float:
    """Bonus applied when you intend to follow the recommended plan."""
    # Keep this simple/constant; you can make it smarter later.
    return PLAN_UPLIFT_PERCENT

def delta_oil_from_deltas(deltas_min: Dict[str, float]) -> float:
    """Sum of (delta minutes / 30) * effect_per_30min across zones."""
    total = 0.0
    for k, dmin in deltas_min.items():
        if k in EFFECT_PER_30_MIN and not (isinstance(dmin, float) and np.isnan(dmin)):
            total += (dmin / 30.0) * EFFECT_PER_30_MIN[k]
    return float(total)

# ── Bot runtime state ─────────────────────────────────────────────────────────
engine = RecoEngine(REPORT_PATH)
latest_plan: Dict[int, Dict[str, float]] = {}
latest_feed: Dict[int, Dict[str, float]] = {}

def format_plan(plan: Dict[str, float]) -> str:
    def order(k):
        m = re.match(r"(\d{2,3})-(\d{2,3})", k)
        a = int(m.group(1)) if m else 0; b = int(m.group(2)) if m else 0
        return (a,b, "separator" in k)
    lines = ["Recommended zone minutes:"]
    for k in sorted(plan.keys(), key=order):
        lines.append(f"{k}: {to_hhmmss(plan[k])}")
    return "\n".join(lines)

def is_allowed(update: Update) -> bool:
    if ALLOWED_CHAT_ID is None:
        return True
    return update.effective_chat.id == ALLOWED_CHAT_ID or update.effective_chat.id > 0  # allow private chat too

async def guard(update: Update) -> bool:
    if is_allowed(update): return True
    await update.message.reply_text("Please use me inside the official plant group.")
    return False

# ── Handlers ──────────────────────────────────────────────────────────────────
async def start(update: Update, ctx: ContextTypes.DEFAULT_TYPE):
    if not await guard(update): return
    await update.message.reply_text("PyroVision Assistant ready ✅\n\n"+HELP)

async def help_cmd(update: Update, ctx: ContextTypes.DEFAULT_TYPE):
    if not await guard(update): return
    await update.message.reply_text(HELP)

async def reload_cmd(update: Update, ctx: ContextTypes.DEFAULT_TYPE):
    if not await guard(update): return
    n = engine.reload()
    await update.message.reply_text(f"Reloaded Excel recommendations ✅ ({n} rules)")

async def plan_cmd(update: Update, ctx: ContextTypes.DEFAULT_TYPE):
    if not await guard(update): return
    feed = engine.parse_feed(update.message.text)
    plan = engine.plan_for_feed(update.message.text)
    latest_plan[update.effective_chat.id] = plan
    latest_feed[update.effective_chat.id] = feed
    await update.message.reply_text(format_plan(plan))

async def predict_cmd(update: Update, ctx: ContextTypes.DEFAULT_TYPE):
    if not await guard(update): return
    # 1) Parse feed, build plan
    feed = engine.parse_feed(update.message.text)
    if not feed:
        await update.message.reply_text("I couldn’t read your feed. Try:\n/predict Radial=5115 Nylon=600 Chips=3465 Powder=1490")
        return
    plan = engine.plan_for_feed(update.message.text)
    latest_plan[update.effective_chat.id] = plan
    latest_feed[update.effective_chat.id] = feed

    # 2) Base oil from feed composition
    base = feed_weighted_potential(feed)
    # 3) Plan bonus (assuming you’ll follow the plan)
    bonus = plan_efficiency_bonus(plan)
    pred = np.clip(base + bonus, 0, 100)

    # Build reply
    tot = sum(feed.values()) or 1.0
    order_keys = ["radial","nylon","chips","powder","kachra","others"]
    feed_lines = [f"• {k.title():<7} {feed.get(k,0):.0f} kg ({(feed.get(k,0)/tot*100):.1f}%)" for k in order_keys if k in feed]
    lines = [
        "📦 Feed mix:",
        *feed_lines,
        "",
        format_plan(plan),
        "",
        f"🎯 Expected Oil Yield (if you follow this plan): ~{pred:.1f}%  (base {base:.1f}% + {bonus:.1f}% plan bonus)"
    ]
    await update.message.reply_text("\n".join(lines))

async def actual_cmd(update: Update, ctx: ContextTypes.DEFAULT_TYPE):
    if not await guard(update): return
    chat_id = update.effective_chat.id
    plan = latest_plan.get(chat_id, engine.plan_for_feed("Feed:"))
    feed = latest_feed.get(chat_id, {})
    actual_map = engine.parse_actuals(update.message.text)
    # deltas
    deltas = {}
    for k, tgt in plan.items():
        if k in actual_map and not (isinstance(tgt, float) and np.isnan(tgt)):
            deltas[k] = actual_map[k] - tgt
    # text block
    lines = ["Δ vs plan (minutes):"]
    for k in sorted(deltas.keys()):
        lines.append(f"{k}: {deltas[k]:+.0f}")

    # tips
    tips = []
    for k, d in deltas.items():
        if abs(d) >= 10:
            tips.append(("reduce" if d>0 else "increase") + f" {k} by ~{abs(d):.0f} min")
    if not tips:
        tips = ["Near-optimal vs plan. Keep the same profile."]
    lines.append("\nRecommendations:")
    for t in tips: lines.append(f"• {t}")

    # oil prediction refinement from actuals
    base = feed_weighted_potential(feed) if feed else 0.0
    bonus = plan_efficiency_bonus(plan)
    delta_oil = delta_oil_from_deltas(deltas)
    refined = np.clip(base + bonus + delta_oil, 0, 100)
    lines.append(f"\n🔎 Refined Oil Yield (from your actual zones): ~{refined:.1f}%  (base {base:.1f}% + {bonus:.1f}% plan + {delta_oil:+.1f}% from deviations)")

    await update.message.reply_text("\n".join(lines))

async def text_router(update: Update, ctx: ContextTypes.DEFAULT_TYPE):
    if not await guard(update): return
    t = (update.message.text or "").strip()
    if t.lower().startswith("feed:"):
        await predict_cmd(update, ctx)   # treat bare 'Feed:' as predict
    elif t.lower().startswith("actual:"):
        await actual_cmd(update, ctx)
    else:
        await update.message.reply_text("I didn’t understand.\n\n"+HELP)

# ── Main ──────────────────────────────────────────────────────────────────────
def main():
    if not TELEGRAM_BOT_TOKEN:
        print("ERROR: set TELEGRAM_BOT_TOKEN env var.")
        return
    app = Application.builder().token(TELEGRAM_BOT_TOKEN).build()
    app.add_handler(CommandHandler("start", start))
    app.add_handler(CommandHandler("help", help_cmd))
    app.add_handler(CommandHandler("reload", reload_cmd))
    app.add_handler(CommandHandler("plan", plan_cmd))
    app.add_handler(CommandHandler("predict", predict_cmd))
    app.add_handler(CommandHandler("actual", actual_cmd))
    app.add_handler(MessageHandler(filters.TEXT & (~filters.COMMAND), text_router))
    log.info("Bot starting…")
    app.run_polling(allowed_updates=Update.ALL_TYPES)

if __name__ == "__main__":
    main()
