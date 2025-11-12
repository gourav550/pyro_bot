# ---------- PyroVision Assistant: full bot.py (with color-coded confidence) ----------
import os, io, re, json, math, logging
from datetime import datetime, timedelta, timezone
from zoneinfo import ZoneInfo

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from apscheduler.schedulers.asyncio import AsyncIOScheduler

from telegram import Update, InputFile
from telegram.constants import ParseMode
from telegram.ext import Application, CommandHandler, MessageHandler, ContextTypes, filters

# ---------------- logging ----------------
LOG = logging.getLogger("pyro_bot")
logging.basicConfig(level=logging.INFO, format="%(asctime)s | %(levelname)s | %(name)s | %(message)s")

# ---------------- config (env) ----------------
TOKEN = os.getenv("TELEGRAM_BOT_TOKEN", "")
SUMMARY_CHAT_ID = int(os.getenv("SUMMARY_CHAT_ID", "0") or "0")
REPORT_PATH = os.getenv("REPORT_PATH", "pyrolysis_feed_temp_ZONE_TIME_report.xlsx")
BOT_TZ = ZoneInfo("Asia/Kolkata")
try:
    MACHINE_MAP = json.loads(os.getenv("MACHINE_MAP", "{}"))
except Exception:
    MACHINE_MAP = {}

STATE_PATH = "bot_state.json"

# ---------------- confidence presentation config ----------------
# change thresholds here if you want different coloring
CONF_THRESH = {"green": 0.90, "yellow": 0.80}
CONF_BAR_BLOCKS = 5

# ---------------- initial persistent state ----------------
state = {
    "weights": {
        "radial": 0.44,
        "nylon": 0.42,
        "chips": 0.46,
        "powder": 0.53,
        "kachra": 0.40,
        "others": 0.40,
    },
    "latest_feed": {},
    "last_actual_ts": {},
    "reminders": {},
    "mix_mean": None,
    "errors": []
}

# ---------------- helpers: save/load state ----------------
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
            LOG.warning("Could not load state: %s", e)
    else:
        LOG.info("No previous state file found.")

    state.setdefault("mix_mean", None)
    state.setdefault("errors", [])

# ---------------- date / time helpers ----------------
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
        h,m,se = parts
        return h*60 + m + se/60.0
    if len(parts) == 2:
        m,se = parts
        return m + se/60.0
    return 0.0

def machine_label(chat_id: int) -> str:
    return MACHINE_MAP.get(str(chat_id)) or MACHINE_MAP.get(str(int(chat_id))) or str(chat_id)

def format_date_with_weekday(iso_or_none: str|None=None) -> str:
    if not iso_or_none:
        now = datetime.now(BOT_TZ)
        return now.strftime("%d-%b-%Y (%A)")
    try:
        dt = datetime.fromisoformat(iso_or_none)
        if dt.tzinfo is None:
            dt = dt.replace(tzinfo=BOT_TZ)
        return dt.strftime("%d-%b-%Y (%A)")
    except Exception:
        return datetime.now(BOT_TZ).strftime("%d-%b-%Y (%A)")

# ---------------- excel rules loader ----------------
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
            window = f"{m.group(1)}-{m.group(2)}"
            which = "separator" if "separator" in feat else "reactor"
            rules[f"{window} {which}"] = float(mins)
        LOG.info("Loaded %d zone rules from Excel.", len(rules))
    except Exception as e:
        LOG.warning("Excel load failed: %s. Using fallback defaults.", e)
        rules = {
            "50-200 reactor": 165.0, "200-300 reactor": 70.0, "300-400 reactor": 185.0,
            "300-400 separator": 175.0, "400-450 reactor": 75.0, "450-480 reactor": 20.0,
            "480-500 reactor": 0.0, "500-520 reactor": 0.0
        }
    return rules

ZONE_RULES = load_zone_rules()

def maybe_load_yield_weights():
    """Try to override state['weights'] from Excel sheet 'YieldWeights'."""
    try:
        df = pd.read_excel(REPORT_PATH, sheet_name="YieldWeights")
        comp = df["component"].astype(str).str.lower().str.strip()
        val = df["weight"].astype(float)
        for k,v in zip(comp, val):
            if k in state["weights"] and 0.2 <= float(v) <= 0.8:
                state["weights"][k] = float(v)
        LOG.info("YieldWeights applied from Excel.")
    except Exception:
        pass

# ---------------- parsing feed & actual ----------------
def parse_feed(text: str) -> dict:
    t = re.sub(r"^/?(plan|predict)?\s*feed\s*:\s*|^feed\s*:\s*", "", text, flags=re.I).strip()
    data = {}
    for part in re.split(r"[,\n;]+", t):
        p = part.strip()
        if not p: continue
        if "=" in p:
            k,v = [x.strip() for x in p.split("=",1)]
        else:
            m = re.match(r"([A-Za-z]+)\s+([\d.]+)", p)
            if not m: continue
            k,v = m.group(1), m.group(2)
        lk = k.lower()
        if lk in ("batch","operator","date","machine"):
            data[lk] = v
            continue
        v2 = v.lower()
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
        k,v = [x.strip() for x in chunk.split("=",1)]
        lk = k.lower()
        if re.match(r"\d{2,3}\s*-\s*\d{2,3}", lk):
            key = re.sub(r"\s+","", lk)
            out[key] = hhmmss_to_minutes(v)
        elif lk in ("oil","oil%","oilpct","oil_pct"):
            out["oil"] = float(re.sub(r"[^\d.]", "", v) or 0.0)
        elif lk == "batch":
            out["batch"] = v
    return out

# ---------------- planning & prediction engine ----------------
class RecoEngine:
    def __init__(self, rules: dict):
        self.defaults = dict(rules)

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

# ---------------- presentation: confidence bar & emoji ----------------
def render_confidence(conf: float) -> str:
    pct = int(round(conf*100))
    # emoji
    if conf >= CONF_THRESH["green"]:
        emoji = "🟢"
    elif conf >= CONF_THRESH["yellow"]:
        emoji = "🟡"
    else:
        emoji = "🔴"
    # bar
    filled = int(round(conf * CONF_BAR_BLOCKS))
    filled = max(0, min(CONF_BAR_BLOCKS, filled))
    bar = "▮" * filled + "▯" * (CONF_BAR_BLOCKS - filled)
    return f"{emoji} Confidence: {pct}% {bar}"

# ---------------- chart helpers ----------------
def chart_from_plan_vs_actual(plan: dict, actual: dict|None=None) -> io.BytesIO:
    zones = [z for z in sorted(plan.keys(), key=lambda s: int(re.match(r"(\d{2,3})", s).group(1)))]
    pmins = [plan[z] for z in zones]
    amins = []
    if actual:
        amap = {k.replace(" ",""):v for k,v in actual.items() if "-" in k}
        for z in zones:
            amins.append(amap.get(z.replace(" ",""), np.nan))
    x = np.arange(len(zones))
    fig, ax = plt.subplots(figsize=(9,3.2), dpi=140)
    width = 0.36
    ax.bar(x - width/2, pmins, width, label="Plan (min)")
    if actual:
        ax.bar(x + width/2, amins, width, label="Actual (min)")
    ax.set_xticks(x, zones, rotation=25)
    ax.set_ylabel("Minutes")
    ax.legend()
    ax.grid(True, axis="y", alpha=0.2)
    buf = io.BytesIO()
    plt.tight_layout()
    plt.savefig(buf, format="png")
    plt.close(fig)
    buf.seek(0)
    return buf

# ---------------- oil-yield recommendations ----------------
def oil_yield_reco(pred: float, actual: float, plan: dict, actual_zones: dict) -> list:
    delta = actual - pred
    tips = []
    dev = {}
    for k,pmin in plan.items():
        key = k.replace(" ","")
        if key in actual_zones:
            dev[k] = actual_zones[key] - pmin
    if delta < -1.5:
        tips.append("Increase *300–400 separator* by ~15–25 min (more condensation).")
        tips.append("Extend *300–400 reactor* by ~20–30 min for heavier mix completion.")
        tips.append("Slightly extend *200–300 reactor* by ~10–15 min (stabilize ramp).")
        tips.append("Check condenser ΔT & line losses; avoid sharp temp spikes.")
    elif delta < -0.5:
        tips.append("Slightly extend *300–400 separator* by ~10–15 min; watch for over-cracking.")
        tips.append("Add ~10–15 min in *300–400 reactor* if drip reduces early.")
    else:
        tips.append("Yield on target; maintain steady ramp and condensation.")
    return tips

# ---------------- scheduling: reminders & daily summary ----------------
def batch_key(chat_id: int, batch: str|int|None) -> str:
    return f"{chat_id}:{batch or ''}"

async def reminder_tick(app: Application):
    now = datetime.now(BOT_TZ)
    to_update = False
    for key,rec in list(state["reminders"].items()):
        due = datetime.fromisoformat(rec["due"])
        if due.tzinfo is None: due = due.replace(tzinfo=BOT_TZ)
        if now >= due:
            cid = rec["chat_id"]
            bno = rec["batch"]
            txt = (f"⚠️ *Reminder:* Actual data not entered yet for *Batch {bno}* on *{machine_label(cid)}*.\n"
                   "Please send:\n`Actual: 50-200=hh:mm, 200-300=hh:mm, 300-400=hh:mm, 400-450=hh:mm, 450-480=hh:mm, oil=xx.x; batch=..`")
            try:
                await app.bot.send_message(cid, txt, parse_mode=ParseMode.MARKDOWN)
                if SUMMARY_CHAT_ID:
                    await app.bot.send_message(SUMMARY_CHAT_ID, f"🔔 {txt}", parse_mode=ParseMode.MARKDOWN)
            except Exception as e:
                LOG.warning("reminder send failed: %s", e)
            # reschedule 1 hour later until cleared
            state["reminders"][key]["due"] = (now + timedelta(hours=1)).isoformat()
            to_update = True
    if to_update:
        save_state()

async def daily_summary_job(app: Application):
    now = datetime.now(BOT_TZ)
    lines = [f"📊 *Daily Summary* — {now.strftime('%d-%b-%Y (%A) %H:%M')} IST"]
    for cid_str,label in MACHINE_MAP.items():
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
                status = "Idle"
        lines.append(f"• {label}: {status}")
    if SUMMARY_CHAT_ID:
        try:
            await app.bot.send_message(SUMMARY_CHAT_ID, "\n".join(lines), parse_mode=ParseMode.MARKDOWN)
        except Exception as e:
            LOG.warning("daily summary send failed: %s", e)

# ---------------- command handlers ----------------
HELP = (
    "*Commands*\n"
    "• Send *Feed:* `Radial=5.1T, Nylon=0.6T, Chips=3.4T, Powder=1.5T, batch=92, operator=Ravi, date=09-11-2025`\n"
    "  → Plan + Predicted oil (also posted in Summary).\n"
    "• Send *Actual:* `50-200=01:14, 200-300=01:06, 300-400=02:07, 400-450=01:10, 450-480=00:32, oil=40.7; batch=92`\n"
    "  → Deviation + Oil-yield recommendations + chart\n"
    "• `/status` → Machine status (in Summary it lists all machines)\n"
    "• `/reload` → Reload Excel rules / weights\n"
    "• `/id` → Show this chat’s id and label\n"
    "• `what feed: ...` → Suggest a feed mix to maximize oil (try `what feed: total=10T`)\n"
)

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
    await update.message.reply_text("🔁 Reloaded Excel rules and yield weights (if present).")

async def cmd_status(update: Update, context: ContextTypes.DEFAULT_TYPE):
    now = datetime.now(BOT_TZ)
    if SUMMARY_CHAT_ID and update.effective_chat.id == SUMMARY_CHAT_ID:
        lines = ["🟢 *Machine Status*"]
        for cid_str,label in MACHINE_MAP.items():
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
                    status = "Idle"
            lines.append(f"• {label}: {status}")
        await update.message.reply_text("\n".join(lines), parse_mode=ParseMode.MARKDOWN)
    else:
        cid = str(update.effective_chat.id)
        label = machine_label(update.effective_chat.id)
        lf = state["latest_feed"].get(cid)
        st = "Idle"
        if lf:
            lfts = datetime.fromisoformat(lf["ts"])
            if lfts.tzinfo is None: lfts = lfts.replace(tzinfo=BOT_TZ)
            hrs = (datetime.now(BOT_TZ) - lfts).total_seconds()/3600
            last_act_iso = state["last_actual_ts"].get(cid)
            completed = False
            if last_act_iso:
                lat = datetime.fromisoformat(last_act_iso)
                if lat.tzinfo is None: lat = lat.replace(tzinfo=BOT_TZ)
                completed = lat >= lfts
            if hrs <= 12 and not completed:
                st = f"Running (batch {lf.get('batch','?')})"
            elif completed:
                st = f"Completed (batch {lf.get('batch','?')})"
            else:
                st = "Idle"
        await update.message.reply_text(f"{label}: {st}")

# ---------------- text routing: feed / actual / what feed ----------------
def pretty_plan(plan: dict) -> str:
    def keyer(x):
        m = re.match(r"(\d{2,3})-(\d{2,3})", x)
        if not m: return (999,999,x)
        return (int(m.group(1)), int(m.group(2)), x)
    lines = []
    for z in sorted(plan.keys(), key=keyer):
        lines.append(f"{z}: {to_hhmmss(plan[z])}")
    return "\n".join(lines)

def route_summary_post(context: ContextTypes.DEFAULT_TYPE, text: str):
    if SUMMARY_CHAT_ID:
        try:
            context.create_task(context.bot.send_message(SUMMARY_CHAT_ID, text, parse_mode=ParseMode.MARKDOWN))
        except Exception:
            pass

def remember_feed(chat_id: int, feed: dict, plan: dict):
    now = datetime.now(BOT_TZ)
    entry = {"ts": now.isoformat(), "batch": str(feed.get("batch","")), "operator": str(feed.get("operator","")), "feed": feed, "plan": plan}
    state["latest_feed"][str(chat_id)] = entry
    batch = entry["batch"]
    if batch:
        key = batch_key(chat_id, batch)
        state["reminders"][key] = {"chat_id": chat_id, "batch": batch, "due": (now + timedelta(hours=12)).isoformat()}
    save_state()

async def handle_feed(update: Update, context: ContextTypes.DEFAULT_TYPE, text: str, predict_only=False):
    chat_id = update.effective_chat.id
    feed = parse_feed(text)
    maybe_load_yield_weights()
    plan = ENGINE.plan(feed)
    pred, conf = predict_yield(feed)
    label = machine_label(chat_id)
    batch = feed.get("batch","?")
    oper = feed.get("operator","?")
    remember_feed(chat_id, feed, plan)
    conf_str = render_confidence(conf)
    msg = []
    msg.append(f"🏷️ *{label}* — *Batch* {batch}, *Operator* {oper}")
    msg.append(f"🛢️ Feed: Radial {feed.get('radial',0)/1000:.2f}T, Nylon {feed.get('nylon',0)/1000:.2f}T, Chips {feed.get('chips',0)/1000:.2f}T, Powder {feed.get('powder',0)/1000:.2f}T")
    msg.append(f"📈 *Predicted Oil:* *{pred:.2f}%*  — {conf_str}")
    if not predict_only:
        msg.append("\n*Recommended zone minutes:*\n" + pretty_plan(plan))
    text_out = "\n".join(msg)
    await update.message.reply_text(text_out, parse_mode=ParseMode.MARKDOWN)
    route_summary_post(context, text_out)

async def handle_actual(update: Update, context: ContextTypes.DEFAULT_TYPE):
    chat_id = update.effective_chat.id
    actual = parse_actual(update.message.text)
    lf = state["latest_feed"].get(str(chat_id))
    if not lf:
        await update.message.reply_text("No recorded Feed for this machine. Send the Feed first.")
        return
    plan = lf.get("plan", {})
    deltas = {}
    tips = []
    for z,pmin in plan.items():
        zkey = z.replace(" ","")
        if zkey in actual:
            am = actual[zkey]
            deltas[z] = am - pmin
    if "oil" in actual:
        learn_from_actual(lf.get("feed",{}), float(actual["oil"]))
        state["last_actual_ts"][str(chat_id)] = datetime.now(BOT_TZ).isoformat()
        key = batch_key(chat_id, lf.get("batch"))
        if key in state["reminders"]:
            del state["reminders"][key]
        save_state()
    pred,conf = predict_yield(lf.get("feed",{}))
    # Build deviation message showing minutes and hh:mm sign-aware
    def fmt_delta_minutes(m):
        sign = "-" if m < 0 else "+"
        mm = abs(int(round(m)))
        hh = mm // 60
        rem = mm % 60
        return f"{sign}{mm} min ({sign}{hh}:{rem:02d})"
    lines = [f"📊 *Deviation vs plan (min & hh:mm):*"]
    for z in sorted(deltas.keys(), key=lambda s: int(re.match(r"(\d{2,3})", s).group(1))):
        lines.append(f"{z}: {fmt_delta_minutes(deltas[z])}")
    oil_msg = ""
    if "oil" in actual:
        actual_oil = float(actual["oil"])
        diff = actual_oil - pred
        conf_str = render_confidence(conf)
        oil_msg = f"\n🛢️ Predicted oil: *{pred:.2f}%*  — {conf_str}\n✅ Actual oil: *{actual_oil:.2f}%*  (Δ {diff:+.2f}%)"
        # get recommendations
        tips = oil_yield_reco(pred, actual_oil, plan, actual)
    if tips:
        lines.append("\n🛠️ Recommendations:")
        for t in tips:
            lines.append(f"• {t}")
    else:
        lines.append("\n✅ Near-optimal execution vs plan.")
    await update.message.reply_text("\n".join(lines), parse_mode=ParseMode.MARKDOWN)
    # send chart
    try:
        buf = chart_from_plan_vs_actual(plan, actual)
        await context.bot.send_photo(chat_id, photo=InputFile(buf, filename="plan_vs_actual.png"))
    except Exception as e:
        LOG.warning("chart send failed: %s", e)
    # also post to summary
    route_summary_post(context, f"🧾 Actual logged for {machine_label(chat_id)} (batch {lf.get('batch','?')}).")

async def handle_whatfeed(update: Update, context: ContextTypes.DEFAULT_TYPE, text: str):
    # Simple 'what feed' handler: accept "what feed: total=10T" or empty -> produce a recommended split
    t = re.sub(r"^/?what\s*feed\s*:\s*|^what\s*feed\s*:\s*","", text, flags=re.I).strip()
    total_t = 10.0  # default tonnes
    # parse simple total= or constraints (not heavy parser)
    m = re.search(r"total\s*=\s*([\d\.]+)", t)
    if m:
        total_t = float(m.group(1))
    # heuristic recommend high powder/chips ratio for higher oil:
    # propose 40% radial, 10% nylon, 30% chips, 20% powder (by mass)
    rec = {"radial": 0.40, "nylon": 0.10, "chips": 0.30, "powder": 0.20, "kachra": 0.0, "others": 0.0}
    feed_kg = {k: rec[k]*total_t*1000.0 for k in rec}
    pred, conf = predict_yield(feed_kg)
    conf_str = render_confidence(conf)
    msg = []
    msg.append(f"🔎 *What feed* recommendation for total {total_t:.2f}T:")
    msg.append("• Suggested composition (T): " + ", ".join(f"{k.capitalize()} {feed_kg[k]/1000.0:.2f}T" for k in ("radial","nylon","chips","powder")))
    msg.append(f"• Predicted Oil: *{pred:.2f}%* — {conf_str}")
    plan = ENGINE.plan(feed_kg)
    msg.append("\n*Recommended zone minutes:*\n" + pretty_plan(plan))
    await update.message.reply_text("\n".join(msg), parse_mode=ParseMode.MARKDOWN)
    route_summary_post(context, "\n".join(msg))

async def text_router(update: Update, context: ContextTypes.DEFAULT_TYPE):
    text = (update.message.text or "").strip()
    if not text:
        await update.message.reply_text("Empty message. Type /help.")
        return
    if re.match(r"(?i)^feed\s*:", text):
        await handle_feed(update, context, text, predict_only=False)
    elif re.match(r"(?i)^what\s*feed\s*:", text):
        await handle_whatfeed(update, context, text)
    elif re.match(r"(?i)^actual\s*:", text):
        await handle_actual(update, context)
    elif re.match(r"(?i)^whatif\s*feed\s*:", text) or re.match(r"(?i)^whatif\s*:", text):
        # treat as predict only
        await handle_feed(update, context, text, predict_only=True)
    else:
        await update.message.reply_text("I didn't understand that. Type /help for examples.")

# ---------------- main: setup app, scheduler, jobs ----------------
def build_app():
    if not TOKEN:
        raise RuntimeError("TELEGRAM_BOT_TOKEN not set.")
    app = Application.builder().token(TOKEN).build()
    app.add_handler(CommandHandler("start", cmd_start))
    app.add_handler(CommandHandler("help", cmd_help))
    app.add_handler(CommandHandler("reload", cmd_reload))
    app.add_handler(CommandHandler("status", cmd_status))
    app.add_handler(CommandHandler("id", cmd_id))
    app.add_handler(MessageHandler(filters.TEXT & ~filters.COMMAND, text_router))
    return app

def schedule_jobs(app: Application):
    sched = AsyncIOScheduler(timezone=str(BOT_TZ))
    # hourly reminder tick (runs every hour at :00)
    sched.add_job(lambda: app.create_task(reminder_tick(app)), "cron", minute=0)
    # daily summary at 21:35 IST
    sched.add_job(lambda: app.create_task(daily_summary_job(app)), "cron", hour=21, minute=35)
    sched.start()

def main():
    LOG.info("Loaded zone rules from Excel.")
    load_state()
    maybe_load_yield_weights()
    app = build_app()
    schedule_jobs(app)
    LOG.info("🚀 PyroVision Assistant running...")
    app.run_polling()

if __name__ == "__main__":
    main()
# ---------- end of bot.py ----------
