# =========================================================
# PyroVision Assistant — full bot.py (stable)
# - Chips=0.46, Powder=0.53 oil fractions (overridable via Excel 'YieldWeights')
# - Confidence up to 0.95 based on recent MAE + mix similarity
# - Daily summary 21:35 IST; 12h actual reminder then hourly ping until entered
# - Robust env parsing; safe state; no "no running event loop" usage
# =========================================================

import os, io, re, json, math, logging, sys, traceback
from datetime import datetime, timedelta, time as dtime
from zoneinfo import ZoneInfo

import numpy as np
import pandas as pd

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from apscheduler.schedulers.asyncio import AsyncIOScheduler
from apscheduler.triggers.cron import CronTrigger
from apscheduler.triggers.date import DateTrigger
from apscheduler.triggers.interval import IntervalTrigger

from telegram import Update, InputFile
from telegram.constants import ParseMode
from telegram.ext import Application, CommandHandler, MessageHandler, ContextTypes, filters

# ---------------- logging ----------------
LOG = logging.getLogger("pyro_bot")
logging.basicConfig(level=logging.INFO, format="%(asctime)s | %(levelname)s | %(name)s | %(message)s")

# ---------------- configuration (env) ----------------
def _get_token() -> str:
    tok = os.getenv("TELEGRAM_BOT_TOKEN", "").strip()
    if not tok or tok.startswith("123456"):
        raise RuntimeError("❌ Missing/placeholder TELEGRAM_BOT_TOKEN")
    return tok

def _get_summary_id():
    raw = (os.getenv("SUMMARY_CHAT_ID", "") or "").strip()
    if not raw:
        return None
    try:
        return int(raw)
    except Exception:
        LOG.error("Invalid SUMMARY_CHAT_ID %r, ignoring.", raw)
        return None

def _get_machine_map() -> dict:
    raw = (os.getenv("MACHINE_MAP", "") or "").strip()
    if not raw:
        LOG.warning("MACHINE_MAP not set; using empty mapping.")
        return {}
    try:
        mp = json.loads(raw)
        if not isinstance(mp, dict):
            raise ValueError("MACHINE_MAP must be JSON object")
        return {str(k): str(v) for k, v in mp.items()}
    except Exception as e:
        LOG.error("MACHINE_MAP parse error: %s. Raw: %r", e, raw)
        return {}

TOKEN = _get_token()
SUMMARY_CHAT_ID = _get_summary_id()
MACHINE_MAP = _get_machine_map()
REPORT_PATH = os.getenv("REPORT_PATH", "pyrolysis_feed_temp_ZONE_TIME_report.xlsx").strip()
BOT_TZ = ZoneInfo("Asia/Kolkata")  # IST

# ---------------- persistent state ----------------
STATE_PATH = "bot_state.json"
state = {
    "weights": {  # base; can be overridden by Excel 'YieldWeights'
        "radial": 0.44,
        "nylon":  0.42,
        "chips":  0.46,  # as discussed
        "powder": 0.53,  # as discussed
        "kachra": 0.40,
        "others": 0.40,
    },
    "latest_feed": {},     # chat_id -> {ts, batch, operator, date, feed, plan}
    "last_actual_ts": {},  # chat_id -> iso str
    "reminders": {},       # key(chat:batch) -> {chat_id,batch,due_iso}
    "mix_mean": None,      # running mean of normalized mix
    "errors": []           # recent abs prediction errors in %, capped 30
}

def save_state():
    try:
        with open(STATE_PATH, "w", encoding="utf-8") as f:
            json.dump(state, f, ensure_ascii=False, indent=2)
    except Exception as e:
        LOG.warning("Could not save state: %s", e)

def load_state():
    if os.path.exists(STATE_PATH):
        try:
            with open(STATE_PATH, "r", encoding="utf-8") as f:
                obj = json.load(f)
            state.update(obj)
            LOG.info("✅ State loaded.")
        except Exception as e:
            LOG.warning("⚠️ Could not load state: %s", e)
    else:
        LOG.info("ℹ️ No state file found; starting fresh.")
    state.setdefault("mix_mean", None)
    state.setdefault("errors", [])

# ---------------- excel: zone rules + yield weights ----------------
def load_zone_rules(path=REPORT_PATH, sheet="ZoneTime_Recommendations") -> dict:
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
        LOG.info("Loaded %d zone rules from Excel.", len(rules))
    except Exception as e:
        LOG.warning("Excel rules load failed: %s", e)
        rules = {
            "50-200 reactor": 165.0, "200-300 reactor": 70.0, "300-400 reactor": 185.0,
            "300-400 separator": 175.0, "400-450 reactor": 75.0, "450-480 reactor": 20.0,
            "480-500 reactor": 0.0, "500-520 reactor": 0.0
        }
    return rules

def maybe_load_yield_weights():
    try:
        df = pd.read_excel(REPORT_PATH, sheet_name="YieldWeights")
        comp = df["component"].astype(str).str.lower().str.strip()
        val  = df["weight"].astype(float)
        for k, v in zip(comp, val):
            if k in state["weights"] and 0.20 <= float(v) <= 0.80:
                state["weights"][k] = float(v)
        LOG.info("YieldWeights sheet applied.")
    except Exception:
        pass  # optional

ZONE_RULES = load_zone_rules()
maybe_load_yield_weights()

# ---------------- helpers ----------------
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
        if dt.tzinfo is None: dt = dt.replace(tzinfo=BOT_TZ)
        return dt.strftime("%d-%b-%Y (%A)")
    except Exception:
        return datetime.now(BOT_TZ).strftime("%d-%b-%Y (%A)")

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

# ---------------- parser ----------------
def parse_feed(text: str) -> dict:
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
            if not m: continue
            k, v = m.group(1), m.group(2)
        lk = k.lower()
        if lk in ("batch","operator","date","machine"):
            data[lk] = v
            continue
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
    t = re.sub(r"^/?actual\s*:?","", text, flags=re.I).strip()
    out = {}
    for chunk in re.split(r"[;,]+", t):
        if "=" not in chunk: continue
        k, v = [x.strip() for x in chunk.split("=", 1)]
        lk = k.lower()
        if re.match(r"\d{2,3}\s*-\s*\d{2,3}", lk):
            zone = re.sub(r"\s+", "", lk)
            out[zone] = hhmmss_to_minutes(v)
        elif lk in ("oil","oil%","oilpct","oil_pct"):
            out["oil"] = float(re.sub(r"[^\d.]", "", v) or 0.0)
        elif lk == "batch":
            out["batch"] = v
    return out

# ---------------- planning + prediction ----------------
class RecoEngine:
    def __init__(self, base_rules: dict):
        self.defaults = dict(base_rules)

    def plan(self, feed: dict) -> dict:
        plan = dict(self.defaults)
        total = sum(feed.get(k,0.0) for k in ("radial","nylon","chips","powder","kachra","others"))
        if total > 0:
            rr = feed.get("radial",0.0)/total
            cr = feed.get("chips",0.0)/total
            nr = feed.get("nylon",0.0)/total
            for k in list(plan.keys()):
                if k.startswith("300-400"):
                    plan[k] = max(60.0, plan[k]*(1.00 + 0.20*rr + 0.10*cr - 0.08*nr))
                if k.startswith("200-300"):
                    plan[k] = max(45.0, plan[k]*(1.00 + 0.08*rr - 0.05*nr))
        return plan

ENGINE = RecoEngine(ZONE_RULES)

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

    mae = _recent_mae()
    mae_penalty = np.clip(mae/6.0, 0.0, 1.0) * 0.28

    mix_norm = _normalize_mix(feed)
    mm = state.get("mix_mean")
    if mm:
        l1 = sum(abs(mix_norm[k]-mm.get(k,0.0)) for k in mix_norm)
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
            state["weights"][k] += 0.01 * err * share
        for k in state["weights"]:
            state["weights"][k] = float(np.clip(state["weights"][k], 0.30, 0.65))

    mix_norm = _normalize_mix(feed)
    _update_mix_mean(mix_norm)

    err_abs = abs(actual_oil_pct - predict_yield(feed)[0])
    state["errors"].append(float(err_abs))
    if len(state["errors"]) > 30:
        state["errors"] = state["errors"][-30:]
    save_state()

# ---------------- rendering ----------------
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

def oil_yield_reco(pred: float, actual: float, plan: dict, actual_zones: dict) -> list[str]:
    tips = []
    delta = actual - pred  # negative if under
    dev = {}
    for k,pmin in plan.items():
        key = k.replace(" ","")
        if key in actual_zones:
            dev[k] = actual_zones[key] - pmin
    # Use deviations to tailor tips
    if delta < -1.5:
        tips.append("Increase *300–400 separator* by ~15–25 min (improve condensation).")
        tips.append("Extend *300–400 reactor* by ~20–30 min to ensure full wax-to-oil cracking (avoid over-crack).")
        if dev.get("200-300 reactor",0) < -10:
            tips.append("You were short in *200–300 reactor*; add ~10–15 min for a gentler ramp.")
        tips.append("Check condenser ΔT and line losses; keep vapors ~<80°C at outlet.")
    elif delta < -0.5:
        tips.append("Add ~10–15 min in *300–400 separator*; watch for over-cracking signs.")
        if dev.get("300-400 reactor",0) < -10:
            tips.append("Slightly extend *300–400 reactor* (~10–15 min).")
        tips.append("Maintain steady heating; avoid spikes >10°C/min in 280–380°C band.")
    else:
        tips.append("Yield near/on target. Maintain current curve and smooth 300–400 recovery.")
    return tips

# ---------------- scheduler jobs ----------------
def kkey(chat_id: int, batch: str | int | None) -> str:
    return f"{chat_id}:{batch or ''}"

async def reminder_ping(app: Application, chat_id: int, batch: str):
    txt = (f"⚠️ *Reminder:* Actual not entered for *Batch {batch}* on *{machine_label(chat_id)}*.\n"
           "Please send:\n"
           "`Actual: 50-200=hh:mm, 200-300=hh:mm, 300-400=hh:mm, 400-450=hh:mm, 450-480=hh:mm, oil=xx.x; batch=..`")
    try:
        await app.bot.send_message(chat_id, txt, parse_mode=ParseMode.MARKDOWN)
        if SUMMARY_CHAT_ID:
            await app.bot.send_message(SUMMARY_CHAT_ID, f"🔔 {txt}", parse_mode=ParseMode.MARKDOWN)
    except Exception as e:
        LOG.warning("reminder send failed: %s", e)

def schedule_initial_reminder(sched: AsyncIOScheduler, app: Application, chat_id: int, batch: str):
    # one-shot at +12h
    due_dt = datetime.now(BOT_TZ) + timedelta(hours=12)
    sched.add_job(reminder_ping, DateTrigger(run_date=due_dt), args=[app, chat_id, batch],
                  id=f"rem1:{chat_id}:{batch}", replace_existing=True)

def schedule_hourly_reminders(sched: AsyncIOScheduler, app: Application, chat_id: int, batch: str):
    # hourly after first ping; we create it when first ping fires by text_router logic too (simple: schedule immediately)
    sched.add_job(reminder_ping, IntervalTrigger(hours=1, timezone=BOT_TZ),
                  args=[app, chat_id, batch], id=f"remH:{chat_id}:{batch}", replace_existing=True)

def cancel_reminders(sched: AsyncIOScheduler, chat_id: int, batch: str):
    for prefix in ("rem1", "remH"):
        jid = f"{prefix}:{chat_id}:{batch}"
        try:
            sched.remove_job(jid)
        except Exception:
            pass

async def daily_summary_job(app: Application):
    now = datetime.now(BOT_TZ)
    lines = [f"📊 *Daily Summary* — {now.strftime('%d-%b-%Y (%A) %H:%M')} IST"]
    for cid_str, label in MACHINE_MAP.items():
        lf = state["latest_feed"].get(cid_str)
        status = "Idle"
        if lf:
            lfts = datetime.fromisoformat(lf["ts"])
            if lfts.tzinfo is None: lfts = lfts.replace(tzinfo=BOT_TZ)
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
                status = f"Idle (last batch {lf.get('batch','?')})"
        lines.append(f"• {label}: {status}")
    if SUMMARY_CHAT_ID:
        await app.bot.send_message(SUMMARY_CHAT_ID, "\n".join(lines), parse_mode=ParseMode.MARKDOWN)

# ---------------- handlers ----------------
HELP = (
    "*Commands*\n"
    "• Send *Feed:* `Radial=5.1T, Nylon=0.6T, Chips=3.4T, Powder=1.5T, batch=92, operator=Ravi, date=09-11-2025`\n"
    "  → Plan + Predicted oil (also posted in Summary). Reminder in 12h; then hourly until Actual arrives.\n"
    "• Send *Actual:* `50-200=01:14, 200-300=01:06, 300-400=02:07, 400-450=01:10, 450-480=00:32, oil=40.7; batch=92`\n"
    "  → Deviation + *Oil-yield recommendations* + chart (and the model learns).\n"
    "• `/status` → Machine status (in Summary it lists all machines)\n"
    "• `/reload` → Reload Excel rules / weights\n"
    "• `/id` → Show this chat’s id and label\n"
)

def pretty_header(label: str, batch: str, operator: str, date_text: str) -> str:
    return f"🏷️ *{label}* — *Batch* {batch}, *Operator* {operator}, *Date* {date_text}"

async def cmd_start(update: Update, context: ContextTypes.DEFAULT_TYPE):
    await update.message.reply_text("PyroVision Assistant ready ✅\n\n" + HELP, parse_mode=ParseMode.MARKDOWN)

async def cmd_help(update: Update, context: ContextTypes.DEFAULT_TYPE):
    await update.message.reply_text(HELP, parse_mode=ParseMode.MARKDOWN)

async def cmd_id(update: Update, context: ContextTypes.DEFAULT_TYPE):
    cid = update.effective_chat.id
    await update.message.reply_text(f"chat_id: `{cid}`\nname/title: *{machine_label(cid)}*", parse_mode=ParseMode.MARKDOWN)

async def cmd_reload(update: Update, context: ContextTypes.DEFAULT_TYPE):
    global ZONE_RULES, ENGINE
    ZONE_RULES = load_zone_rules()
    ENGINE = RecoEngine(ZONE_RULES)
    maybe_load_yield_weights()
    save_state()
    await update.message.reply_text("🔁 Reloaded Excel rules (and YieldWeights if present).")

async def cmd_status(update: Update, context: ContextTypes.DEFAULT_TYPE):
    now = datetime.now(BOT_TZ)
    if SUMMARY_CHAT_ID and update.effective_chat.id == SUMMARY_CHAT_ID:
        lines = ["🟢 *Machine Status*"]
        for cid_str, label in MACHINE_MAP.items():
            lf = state["latest_feed"].get(cid_str)
            st = "Idle"
            if lf:
                lfts = datetime.fromisoformat(lf["ts"])
                if lfts.tzinfo is None: lfts = lfts.replace(tzinfo=BOT_TZ)
                hrs = (now - lfts).total_seconds()/3600
                last_act_iso = state["last_actual_ts"].get(cid_str)
                completed = False
                if last_act_iso:
                    lat = datetime.fromisoformat(last_act_iso)
                    if lat.tzinfo is None: lat = lat.replace(tzinfo=BOT_TZ)
                    completed = lat >= lfts
                if hrs <= 12 and not completed:
                    st = f"Running (batch {lf.get('batch','?')})"
                elif completed:
                    dtxt = datetime.fromisoformat(last_act_iso).strftime("%d-%b-%Y (%A)") if last_act_iso else ""
                    st = f"Completed — Last batch {lf.get('batch','?')} on {dtxt}"
                else:
                    st = "Idle"
            lines.append(f"• {label}: {st}")
        await update.message.reply_text("\n".join(lines), parse_mode=ParseMode.MARKDOWN)
    else:
        cid = str(update.effective_chat.id)
        label = machine_label(update.effective_chat.id)
        lf = state["latest_feed"].get(cid)
        st = "Idle"
        if lf:
            lfts = datetime.fromisoformat(lf["ts"])
            if lfts.tzinfo is None: lfts = lfts.replace(tzinfo=BOT_TZ)
            hrs = (now - lfts).total_seconds()/3600
            last_act_iso = state["last_actual_ts"].get(cid)
            completed = False
            if last_act_iso:
                lat = datetime.fromisoformat(last_act_iso)
                if lat.tzinfo is None: lat = lat.replace(tzinfo=BOT_TZ)
                completed = lat >= lfts
            if hrs <= 12 and not completed:
                st = f"Running (batch {lf.get('batch','?')})"
            elif completed:
                dtxt = lat.strftime("%d-%b-%Y (%A)")
                st = f"Completed — Last batch {lf.get('batch','?')} on {dtxt}"
            else:
                st = "Idle"
        await update.message.reply_text(f"{label}: {st}", parse_mode=ParseMode.MARKDOWN)

def remember_feed(chat_id: int, feed: dict, plan: dict):
    now = datetime.now(BOT_TZ)
    entry = {
        "ts": now.isoformat(),
        "batch": str(feed.get("batch","")),
        "operator": str(feed.get("operator","")),
        "date": _norm_date(feed.get("date")),
        "feed": feed, "plan": plan
    }
    state["latest_feed"][str(chat_id)] = entry
    save_state()
    return entry

async def plan_flow(update: Update, context: ContextTypes.DEFAULT_TYPE, sched: AsyncIOScheduler):
    chat_id = update.effective_chat.id
    feed = parse_feed(update.message.text)
    label = machine_label(chat_id)
    plan = ENGINE.plan(feed)
    pred, conf = predict_yield(feed)
    batch = feed.get("batch","?")
    operator = feed.get("operator","?")
    date_text = _norm_date(feed.get("date"))

    entry = remember_feed(chat_id, feed, plan)

    # schedule reminders
    if batch:
        schedule_initial_reminder(sched, context.application, chat_id, str(batch))
        schedule_hourly_reminders(sched, context.application, chat_id, str(batch))

    msg = []
    msg.append(pretty_header(label, batch, operator, date_text))
    msg.append(f"🛢️ Feed: Radial {feed.get('radial',0)/1000:.2f}T, Nylon {feed.get('nylon',0)/1000:.2f}T, Chips {feed.get('chips',0)/1000:.2f}T, Powder {feed.get('powder',0)/1000:.2f}T")
    msg.append(f"📈 Predicted Oil: *{pred:.2f}%*  (confidence {conf:.2f})")
    msg.append("\n*Recommended zone minutes (plan):*\n" + pretty_plan(plan))
    text = "\n".join(msg)

    await update.message.reply_text(text, parse_mode=ParseMode.MARKDOWN)
    if SUMMARY_CHAT_ID:
        try:
            await context.application.bot.send_message(SUMMARY_CHAT_ID, text, parse_mode=ParseMode.MARKDOWN)
        except Exception as e:
            LOG.warning("Summary send failed: %s", e)

async def actual_flow(update: Update, context: ContextTypes.DEFAULT_TYPE):
    chat_id = update.effective_chat.id
    actual = parse_actual(update.message.text)
    lf = state["latest_feed"].get(str(chat_id))
    if not lf:
        await update.message.reply_text("I don't find a recorded Feed for this machine. Send the Feed first.")
        return
    plan = lf.get("plan", {})
    deltas = {}
    for z,pmin in plan.items():
        zkey = z.replace(" ","")
        if zkey in actual:
            deltas[z] = actual[zkey] - pmin

    # oil-learning if provided
    extra = []
    if "oil" in actual:
        learn_from_actual(lf.get("feed",{}), float(actual["oil"]))
        state["last_actual_ts"][str(chat_id)] = datetime.now(BOT_TZ).isoformat()
        save_state()
        # cancel reminders for this batch
        if lf.get("batch"):
            cancel_reminders(context.application.job_queue.scheduler, chat_id, str(lf["batch"]))

        # oil yield recommendations
        pred, conf = predict_yield(lf.get("feed",{}))
        tips = oil_yield_reco(pred, float(actual["oil"]), plan, actual)
        extra.append(f"\n🛢️ Predicted vs Actual Oil: *{pred:.2f}%* → *{actual['oil']:.2f}%* (conf {conf:.2f})")
        extra.append("🛠️ *Recommendations to lift oil yield*: \n• " + "\n• ".join(tips))

    # build deviation text
    lines = ["📊 *Deviation vs plan (min):*"]
    for z in sorted(deltas.keys(), key=lambda s: int(s.split("-")[0])):
        lines.append(f"{z}: {deltas[z]:+.0f}")
    if not deltas:
        lines.append("_(no matching zones found in your message)_")
    text = "\n".join(lines + extra)

    await update.message.reply_text(text, parse_mode=ParseMode.MARKDOWN)

    # chart
    buf = bar_plan_vs_actual(plan, {k:v for k,v in actual.items() if "-" in k})
    try:
        await context.bot.send_photo(chat_id, photo=InputFile(buf, filename="plan_vs_actual.png"))
    except Exception as e:
        LOG.warning("chart send failed: %s", e)

    # post summary copy
    if SUMMARY_CHAT_ID:
        try:
            await context.application.bot.send_message(SUMMARY_CHAT_ID,
                f"🧾 Actual logged for *{machine_label(chat_id)}* (batch {lf.get('batch','?')}).\n" + text,
                parse_mode=ParseMode.MARKDOWN)
        except Exception as e:
            LOG.warning("Summary send failed: %s", e)

# ---------------- text router ----------------
async def text_router(update: Update, context: ContextTypes.DEFAULT_TYPE):
    txt = (update.message.text or "").strip()
    if re.match(r"(?i)^feed\s*:", txt):
        await plan_flow(update, context, context.application.job_queue.scheduler)
    elif re.match(r"(?i)^actual\s*:", txt):
        await actual_flow(update, context)
    elif re.match(r"(?i)^status$", txt):
        await cmd_status(update, context)
    else:
        await update.message.reply_text("I didn’t understand. Type /help for examples.")

# ---------------- main ----------------
def main():
    LOG.info("🚀 Starting PyroVision Assistant…")
    load_state()
    LOG.info("Config → REPORT_PATH=%s | SUMMARY_CHAT_ID=%s | MACHINE_MAP=%s", REPORT_PATH, SUMMARY_CHAT_ID, MACHINE_MAP)

    app = Application.builder().token(TOKEN).build()

    # Handlers
    app.add_handler(CommandHandler("start", cmd_start))
    app.add_handler(CommandHandler("help", cmd_help))
    app.add_handler(CommandHandler("id", cmd_id))
    app.add_handler(CommandHandler("reload", cmd_reload))
    app.add_handler(CommandHandler("status", cmd_status))
    app.add_handler(MessageHandler(filters.TEXT & ~filters.COMMAND, text_router))

    # Scheduler (uses PTB's built-in AsyncIOScheduler)
    sched: AsyncIOScheduler = app.job_queue.scheduler

    # Daily summary at 21:35 IST
    sched.add_job(
        daily_summary_job,
        trigger=CronTrigger(hour=21, minute=35, timezone=BOT_TZ),
        args=[app],
        id="daily_summary",
        replace_existing=True,
    )

    LOG.info("✅ Bot ready. Polling…")
    app.run_polling()

if __name__ == "__main__":
    try:
        main()
    except Exception as e:
        LOG.error("Fatal error: %s", e)
        traceback.print_exc()
        sys.exit(1)
