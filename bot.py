# =========================================================
# PyroVision Assistant — full bot.py
# - Chips=0.46, Powder=0.53 oil fractions (can override via Excel "YieldWeights")
# - Confidence up to 0.95 based on recent MAE + mix similarity
# - Daily summary 21:35 IST; 12h actual reminder (hourly ping) until entered
# =========================================================

import os, io, re, json, math, logging
from datetime import datetime, timedelta, time as dtime, timezone
from zoneinfo import ZoneInfo

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from apscheduler.schedulers.asyncio import AsyncIOScheduler

from telegram import Update, InputFile
from telegram.constants import ParseMode
from telegram.ext import (
    Application, CommandHandler, MessageHandler, ContextTypes, filters
)

LOG = logging.getLogger("pyro_bot")
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)s | %(name)s | %(message)s"
)

# -------- Configuration via env --------
TOKEN = os.getenv("TELEGRAM_BOT_TOKEN", "")
SUMMARY_CHAT_ID = int(os.getenv("SUMMARY_CHAT_ID", "0") or "0")
REPORT_PATH = os.getenv("REPORT_PATH", "pyrolysis_feed_temp_ZONE_TIME_report.xlsx")
BOT_TZ = ZoneInfo("Asia/Kolkata")  # IST

try:
    MACHINE_MAP = json.loads(os.getenv("MACHINE_MAP", "{}"))
except Exception:
    MACHINE_MAP = {}

STATE_PATH = "bot_state.json"

# -------- Persistent state --------
state = {
    # Oil fractions (base assumptions). Can be overridden by Excel "YieldWeights".
    "weights": {
        "radial": 0.44,
        "nylon":  0.42,
        "chips":  0.46,   # updated
        "powder": 0.53,   # updated
        "kachra": 0.40,
        "others": 0.40,
    },

    # last feed/plan per machine
    "latest_feed": {},       # chat_id -> {ts, batch, operator, feed, plan}
    "last_actual_ts": {},    # chat_id -> iso ts when /actual logged

    # reminder bookkeeping: key(chat:batch) -> {"chat_id":..,"batch":..,"due":iso}
    "reminders": {},

    # --- for confidence calibration ---
    "mix_mean": None,        # running mean of normalized mix
    "errors": []             # recent abs prediction errors in %, capped 30
}

# -------------------- Utilities --------------------
def save_state():
    try:
        with open(STATE_PATH, "w", encoding="utf-8") as f:
            json.dump(state, f, ensure_ascii=False, indent=2)
    except Exception as e:
        LOG.warning(f"⚠️ Could not save state: {e}")

def load_state():
    if os.path.exists(STATE_PATH):
        try:
            with open(STATE_PATH, "r", encoding="utf-8") as f:
                obj = json.load(f)
            state.update(obj)
            LOG.info("✅ State loaded.")
        except Exception as e:
            LOG.warning(f"⚠️ Could not load state: {e}")
    else:
        LOG.info("ℹ️ No state file found; starting fresh.")

    # ensure new keys exist
    state.setdefault("mix_mean", None)
    state.setdefault("errors", [])

def to_hhmmss(minutes):
    if minutes is None or (isinstance(minutes, float) and math.isnan(minutes)):
        return "00:00:00"
    total = int(round(float(minutes) * 60))
    h, rem = divmod(total, 3600)
    m, s = divmod(rem, 60)
    return f"{h:02d}:{m:02d}:{s:02d}"

def hhmmss_to_minutes(s: str) -> float:
    s = (s or "").strip()
    if ":" not in s:
        return float(s or 0.0)
    parts = [int(x) for x in s.split(":")]
    if len(parts) == 3:
        h, m, se = parts
        return h*60 + m + se/60
    if len(parts) == 2:
        m, se = parts
        return m + se/60
    return 0.0

def machine_label(chat_id: int) -> str:
    return MACHINE_MAP.get(str(chat_id)) or MACHINE_MAP.get(str(int(chat_id))) or str(chat_id)

def _norm_date(s: str | None) -> str:
    """Formats to 'DD-Mon-YYYY (Weekday)' in IST if possible; fallback to today."""
    if not s:
        now = datetime.now(BOT_TZ)
        return now.strftime("%d-%b-%Y (%A)")
    s2 = s.strip()
    fmts = ("%d-%m-%Y","%d.%m.%Y","%d/%m/%Y","%d-%m-%y","%d.%m.%y","%d/%m/%y","%Y-%m-%d")
    for fmt in fmts:
        try:
            dt = datetime.strptime(s2, fmt).replace(tzinfo=BOT_TZ)
            return dt.strftime("%d-%b-%Y (%A)")
        except Exception:
            continue
    try:
        dt = datetime.fromisoformat(s2)
        if dt.tzinfo is None:
            dt = dt.replace(tzinfo=BOT_TZ)
        return dt.strftime("%d-%b-%Y (%A)")
    except Exception:
        return datetime.now(BOT_TZ).strftime("%d-%b-%Y (%A)")

# -------------------- Excel rules --------------------
def load_zone_rules(path=REPORT_PATH, sheet="ZoneTime_Recommendations"):
    rules = {}
    try:
        df = pd.read_excel(path, sheet_name=sheet)
        for _, row in df.iterrows():
            feat = str(row.get("zone_time_feature", "")).strip().lower()
            mins = row.get("suggested_minutes", np.nan)
            if not feat or pd.isna(mins):
                continue
            m = re.search(r"(\d{2,3})\s*[-–]\s*(\d{2,3})", feat)
            if not m:
                continue
            w = f"{m.group(1)}-{m.group(2)}"
            which = "separator" if "separator" in feat else "reactor"
            rules[f"{w} {which}"] = float(mins)
        LOG.info(f"Loaded {len(rules)} zone rules from Excel.")
    except Exception as e:
        LOG.warning(f"Excel rules load failed: {e}")
        rules = {
            "50-200 reactor": 165.0, "200-300 reactor": 70.0, "300-400 reactor": 185.0,
            "300-400 separator": 175.0, "400-450 reactor": 75.0, "450-480 reactor": 20.0,
            "480-500 reactor": 0.0, "500-520 reactor": 0.0
        }
    return rules

ZONE_RULES = load_zone_rules()

def maybe_load_yield_weights():
    """Optionally override weights from Excel sheet 'YieldWeights' with columns:
       component, weight  (component in {radial,nylon,chips,powder,kachra,others}, weight 0..1)
    """
    try:
        df = pd.read_excel(REPORT_PATH, sheet_name="YieldWeights")
        comp = df["component"].astype(str).str.lower().str.strip()
        val  = df["weight"].astype(float)
        for k, v in zip(comp, val):
            if k in state["weights"] and 0.2 <= float(v) <= 0.8:
                state["weights"][k] = float(v)
        LOG.info("YieldWeights sheet applied.")
    except Exception:
        # optional; ignore missing
        pass

# -------------------- Parsing --------------------
def parse_feed(text: str) -> dict:
    """
    Accepts:
      Feed: Radial=5.1T, Nylon=0.6T, Chips=3.4T, Powder=1.5T, batch=92, operator=Ravi, date=09-11-2025
    T/MT/KG supported; returns kg internally (+ batch/operator/date if provided)
    """
    t = re.sub(r"^/?(plan|predict)?\s*feed\s*:\s*|^feed\s*:\s*", "", text, flags=re.I).strip()
    data = {}
    for part in re.split(r"[,\n;]+", t):
        p = part.strip()
        if not p:
            continue
        if "=" in p:
            k, v = [x.strip() for x in p.split("=", 1)]
        else:
            m = re.match(r"([A-Za-z]+)\s+([\d.]+)", p)
            if not m:
                continue
            k, v = m.group(1), m.group(2)
        lk = k.lower()
        if lk in ("batch","operator","date","machine"):
            data[lk] = v
            continue
        # units
        v2 = v.lower().strip()
        if v2.endswith(("t","ton","tons","mt")):
            valkg = float(re.sub(r"[^\d.]", "", v2)) * 1000.0
        elif v2.endswith(("kg","kgs")):
            valkg = float(re.sub(r"[^\d.]", "", v2))
        else:
            valkg = float(re.sub(r"[^\d.]", "", v2))
        data[lk] = valkg
    return data

def parse_actual(text: str) -> dict:
    """
    Actual: 50-200=01:14, 200-300=01:06, 300-400=02:07, 400-450=01:10, 450-480=00:32, oil=40.73, batch=66
    """
    t = re.sub(r"^/?actual\s*:?","", text, flags=re.I).strip()
    out = {}
    for chunk in re.split(r"[;,]+", t):
        if "=" not in chunk:
            continue
        k, v = [x.strip() for x in chunk.split("=", 1)]
        lk = k.lower()
        if re.match(r"\d{2,3}\s*-\s*\d{2,3}", lk):
            zone = re.sub(r"\s+", "", lk)  # "50-200"
            out[zone] = hhmmss_to_minutes(v)
        elif lk in ("oil","oil%","oilpct","oil_pct"):
            out["oil"] = float(re.sub(r"[^\d.]", "", v) or 0.0)
        elif lk == "batch":
            out["batch"] = v
    return out

# -------------------- Planning + Prediction --------------------
class RecoEngine:
    def __init__(self, base_rules: dict):
        self.defaults = dict(base_rules)

    def plan(self, feed: dict) -> dict:
        # Start from defaults and apply light composition nudges.
        plan = dict(self.defaults)
        total = sum(feed.get(k,0.0) for k in ("radial","nylon","chips","powder","kachra","others"))
        if total > 0:
            rr = feed.get("radial",0.0)/total
            cr = feed.get("chips",0.0)/total
            nr = feed.get("nylon",0.0)/total
            # emphasise 300-400 windows for heavier mixes
            for k in list(plan.keys()):
                if k.startswith("300-400"):
                    plan[k] = max(60.0, plan[k]*(1.00 + 0.20*rr + 0.10*cr - 0.08*nr))
                if k.startswith("200-300"):
                    plan[k] = max(45.0, plan[k]*(1.00 + 0.08*rr - 0.05*nr))
        return plan

ENGINE = RecoEngine(ZONE_RULES)

def _normalize_mix(feed: dict) -> dict:
    keys = ("radial","nylon","chips","powder","kachra","others")
    total = sum(feed.get(k,0.0) for k in keys)
    if total <= 0:
        return {k:0.0 for k in keys}
    return {k: feed.get(k,0.0)/total for k in keys}

def _update_mix_mean(mix_norm: dict):
    alpha = 0.10
    if not state.get("mix_mean"):
        state["mix_mean"] = dict(mix_norm)
        return
    mm = state["mix_mean"]
    for k,v in mix_norm.items():
        mm[k] = (1-alpha)*mm.get(k,0.0) + alpha*v

def _recent_mae() -> float:
    errs = state.get("errors") or []
    if not errs:
        return 3.0
    return float(np.mean(errs))

def predict_yield(feed: dict) -> tuple[float, float]:
    keys = ("radial","nylon","chips","powder","kachra","others")
    total = sum(feed.get(k,0.0) for k in keys)
    if total <= 0:
        return (0.0, 0.60)

    w = state["weights"]
    pred = 0.0
    for k in keys:
        pred += (feed.get(k,0.0)/total) * (w[k]*100.0)
    pred = float(np.clip(pred, 30.0, 60.0))

    # Confidence up to 0.95: penalize recent MAE and mix distance
    mae = _recent_mae()                 # %
    mae_penalty = np.clip(mae/6.0, 0.0, 1.0) * 0.28

    mix_norm = _normalize_mix(feed)
    mm = state.get("mix_mean")
    if mm:
        l1 = sum(abs(mix_norm[k]-mm.get(k,0.0)) for k in mix_norm)  # 0..2
    else:
        l1 = 0.4
    sim_penalty = np.clip(l1/0.8, 0.0, 1.0) * 0.22

    conf = 0.95 - mae_penalty - sim_penalty
    conf = float(np.clip(conf, 0.60, 0.95))
    return (round(pred,2), round(conf,2))

def learn_from_actual(feed: dict, actual_oil_pct: float):
    keys = ("radial","nylon","chips","powder","kachra","others")
    total = sum(feed.get(k,0.0) for k in keys)
    if total > 0:
        pred,_ = predict_yield(feed)
        err = (actual_oil_pct - pred)/100.0
        for k in keys:
            share = feed.get(k,0.0)/total
            state["weights"][k] += 0.01 * err * share  # tiny learning
        # keep sane bounds
        for k in state["weights"]:
            state["weights"][k] = float(np.clip(state["weights"][k], 0.30, 0.65))

    mix_norm = _normalize_mix(feed)
    _update_mix_mean(mix_norm)

    # track errors (cap 30)
    err_abs = abs(actual_oil_pct - predict_yield(feed)[0])
    state["errors"].append(float(err_abs))
    if len(state["errors"]) > 30:
        state["errors"] = state["errors"][-30:]
    save_state()

# -------------------- Rendering helpers --------------------
def pretty_plan(plan: dict) -> str:
    def keyer(x):
        m = re.match(r"(\d{2,3})-(\d{2,3})", x)
        if not m: return (999,999,x)
        return (int(m.group(1)), int(m.group(2)), x)
    lines = []
    for z in sorted(plan.keys(), key=keyer):
        lines.append(f"{z}: {to_hhmmss(plan[z])}")
    return "\n".join(lines)

def bar_plan_vs_actual(plan: dict, actual: dict | None) -> io.BytesIO:
    zones = [z for z in sorted(plan.keys(), key=lambda s: int(s.split("-")[0]))]
    pmins = [plan[z] for z in zones]
    amins = []
    if actual:
        amap = {k.replace(" ",""):v for k,v in actual.items() if "-" in k or re.match(r"\d{2,3}-\d{2,3}", k)}
        for z in zones:
            amins.append(amap.get(z.replace(" ",""), np.nan))
    x = np.arange(len(zones))
    fig, ax = plt.subplots(figsize=(9.5,3.5), dpi=150)
    width = 0.38
    ax.bar(x - width/2, pmins, width, label="Plan (min)")
    if actual:
        ax.bar(x + width/2, amins, width, label="Actual (min)")
    ax.set_xticks(x, zones, rotation=25)
    ax.set_ylabel("Minutes")
    ax.legend()
    ax.grid(True, axis="y", alpha=0.25)
    buf = io.BytesIO()
    plt.tight_layout()
    plt.savefig(buf, format="png")
    plt.close(fig)
    buf.seek(0)
    return buf

# -------------------- AI Recommendations (oil yield) --------------------
def oil_yield_reco(pred: float, actual: float, plan: dict, actual_zones: dict) -> list[str]:
    """Return bullet tips aimed at closing oil deficit."""
    delta = actual - pred  # negative if underperformed
    tips = []

    # Compute zone deviations (actual - plan) for guidance
    dev = {}
    for k,pmin in plan.items():
        key = k.replace(" ","")
        if key in actual_zones:
            dev[k] = actual_zones[key] - pmin

    # Heuristics:
    if delta < -1.5:  # below predicted
        # emphasize condensation & full conversion
        tips.append("Increase *300–400 separator* by ~15–25 min (more condensation).")
        tips.append("Extend *300–400 reactor* by ~20–30 min for heavier mix completion.")
        tips.append("Slightly extend *200–300 reactor* by ~10–15 min (stabilize ramp).")
        tips.append("Check line losses & condenser ΔT; avoid sharp temp spikes.")
    elif delta < -0.5:
        tips.append("Slightly extend *300–400 separator* by ~10–15 min; watch for over-cracking.")
        tips.append("Add ~10–15 min in *300–400 reactor* if drip reduces early.")
    else:
        tips.append("Yield on target; maintain smooth ramp and steady 300–400 recovery.")
    return tips

# -------------------- Scheduler jobs --------------------
def kkey(chat_id: int, batch: str | int | None) -> str:
    return f"{chat_id}:{batch or ''}"

async def reminder_tick(app: Application):
    """Runs every hour: ping any batch whose due time passed and not cleared yet."""
    now = datetime.now(BOT_TZ)
    expired = []
    for key, rec in state["reminders"].items():
        due = datetime.fromisoformat(rec["due"])
        if due.tzinfo is None:
            due = due.replace(tzinfo=BOT_TZ)
        if now >= due:
            cid = rec["chat_id"]
            bno = rec["batch"]
            txt = (f"⚠️ *Reminder:* Actual data not entered yet for *Batch {bno}* "
                   f"on *{machine_label(cid)}*. Please send:\n"
                   "`Actual: 50-200=hh:mm, 200-300=hh:mm, 300-400=hh:mm, 400-450=hh:mm, 450-480=hh:mm, oil=xx.x; batch=..`")
            try:
                await app.bot.send_message(cid, txt, parse_mode=ParseMode.MARKDOWN)
                if SUMMARY_CHAT_ID:
                    await app.bot.send_message(SUMMARY_CHAT_ID, f"🔔 {txt}", parse_mode=ParseMode.MARKDOWN)
            except Exception as e:
                LOG.warning(f"reminder send failed: {e}")
            # re-schedule next hour
            new_due = now + timedelta(hours=1)
            state["reminders"][key]["due"] = new_due.isoformat()
    save_state()

async def daily_summary_job(app: Application):
    now = datetime.now(BOT_TZ)
    lines = [f"📊 *Daily Summary* — {now.strftime('%d-%b-%Y (%A) %H:%M')} IST"]
    for cid_str, label in MACHINE_MAP.items():
        lf = state["latest_feed"].get(cid_str)
        status = "Idle"
        if lf:
            lfts = datetime.fromisoformat(lf["ts"])
            if lfts.tzinfo is None:
                lfts = lfts.replace(tzinfo=BOT_TZ)
            hrs = (now - lfts).total_seconds()/3600
            last_act_iso = state["last_actual_ts"].get(cid_str)
            completed = False
            if last_act_iso:
                lat = datetime.fromisoformat(last_act_iso)
                if lat.tzinfo is None: lat = lat.replace(tzinfo=BOT_TZ)
                completed = lat >= lfts
            if hrs <= 12 and not completed:
                status = f"Running (batch {lf.get('batch','?')})"
            elif completed:
                status = f"Completed (batch {lf.get('batch','?')})"
            else:
                status = "Idle"
        lines.append(f"• {label}: {status}")
    if SUMMARY_CHAT_ID:
        await app.bot.send_message(SUMMARY_CHAT_ID, "\n".join(lines), parse_mode=ParseMode.MARKDOWN)

# -------------------- Handlers --------------------
HELP = (
    "*Commands*\n"
    "• Send *Feed:* `Radial=5.1T, Nylon=0.6T, Chips=3.4T, Powder=1.5T, batch=92, operator=Ravi, date=09-11-2025`\n"
    "  → Plan + Predicted oil (also posted in Summary).\n"
    "• Send *Actual:* `50-200=01:14, 200-300=01:06, 300-400=02:07, 400-450=01:10, 450-480=00:32, oil=40.7; batch=92`\n"
    "  → Deviation + *Oil-yield recommendations* + chart (and the model learns).\n"
    "• `/status` → Machine status (in Summary it lists all machines)\n"
    "• `/reload` → Reload Excel rules / weights\n"
    "• `/id` → Show this chat’s id and label\n"
)

async def cmd_start(update: Update, context: ContextTypes.DEFAULT_TYPE):
    await update.message.reply_text("PyroVision Assistant ready ✅\n\n" + HELP, parse_mode=ParseMode.MARKDOWN)

async def cmd_help(update: Update, context: ContextTypes.DEFAULT_TYPE):
    await update.message.reply_text(HELP, parse_mode=ParseMode.MARKDOWN)

async def cmd_id(update: Update, context: ContextTypes.DEFAULT_TYPE):
    cid = update.effective_chat.id
    await update.message.reply_text(
        f"chat_id: `{cid}`\nname/title: *{machine_label(cid)}*",
        parse_mode=ParseMode.MARKDOWN
    )

async def cmd_reload(update: Update, context: ContextTypes.DEFAULT_TYPE):
    global ZONE_RULES, ENGINE
    ZONE_RULES = load_zone_rules()
    ENGINE = RecoEngine(ZONE_RULES)
    maybe_load_yield_weights()  # allow Excel to override chips/powder etc.
    save_state()
    await update.message.reply_text("🔁 Reloaded Excel rules (and YieldWeights if present).")

async def cmd_status(update: Update, context: ContextTypes.DEFAULT_TYPE):
    now = datetime.now(BOT_TZ)
    if SUMMARY_CHAT_ID and update.effective_chat.id == SUMMARY_CHAT_ID:
        lines = ["🟢 *Machine Status*"]
        for cid_str, label in MACHINE_MAP.items():
            lf = state["latest_feed"].get(cid_str)
            status = "Idle"
            if lf:
                lfts = datetime.fromisoformat(lf["ts"])
                if lfts.tzinfo is None: lfts = lfts.replace(tzinfo=BOT_TZ)
                hrs = (now - lfts).total_seconds()/3600
                last_act_iso = state["last_actual_ts"].get(cid_str)
                completed = False
               
