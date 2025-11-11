# ==============================================
# PyroVision Assistant — full bot.py (PTB v21.x)
# ==============================================
import os, re, io, json, math, logging
from datetime import datetime, timedelta, time as dt_time, timezone
from zoneinfo import ZoneInfo

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

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

# ---------- Config via ENV ----------
BOT_TOKEN      = os.environ.get("TELEGRAM_BOT_TOKEN", "")
SUMMARY_CHAT_ID= int(os.environ.get("SUMMARY_CHAT_ID", "0") or "0")
REPORT_PATH    = os.environ.get("REPORT_PATH", "pyrolysis_feed_temp_ZONE_TIME_report.xlsx")

try:
    MACHINE_MAP = json.loads(os.environ.get("MACHINE_MAP", "{}") or "{}")
except Exception:
    MACHINE_MAP = {}

BOT_TZ = ZoneInfo("Asia/Kolkata")  # IST

# ---------- State (json on disk) ----------
STATE_PATH = "bot_state.json"
state = {
    "weights": {"radial": .44, "nylon": .42, "chips": .44, "powder": .43, "kachra": .40, "others": .40},
    "latest_feed": {},      # chat_id -> {ts, batch, operator, date, feed, plan}
    "last_actual_ts": {},   # chat_id -> iso dt
    "reminders": {}         # key(chat:batch) -> {chat_id,batch,next_iso,active:1}
}

def load_state():
    global state
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

def save_state():
    try:
        with open(STATE_PATH, "w", encoding="utf-8") as f:
            json.dump(state, f, ensure_ascii=False, indent=2)
    except Exception as e:
        LOG.warning(f"⚠️ Could not save state: {e}")

# ---------- Helpers ----------
def to_hhmmss(minutes: float | int | None) -> str:
    if minutes is None:
        return "-"
    try:
        m = float(minutes)
        if math.isnan(m):
            return "-"
    except Exception:
        return "-"
    total = int(round(m * 60))
    h, rem = divmod(total, 3600)
    mm, ss = divmod(rem, 60)
    return f"{h:02d}:{mm:02d}:{ss:02d}"

def hhmmss_to_minutes(s: str) -> float:
    s = (s or "").strip()
    if ":" not in s:
        # bare minutes
        try:
            return float(re.sub(r"[^\d.]", "", s))
        except Exception:
            return 0.0
    parts = [int(x) for x in s.split(":")]
    if len(parts) == 3:
        h, m, sec = parts
        return h*60 + m + sec/60
    if len(parts) == 2:
        m, sec = parts
        return m + sec/60
    return 0.0

def _norm_key(k: str) -> str:
    return re.sub(r"[^a-z0-9\-]", "", k.lower())

def _machine_label(chat_id: int) -> str:
    return MACHINE_MAP.get(str(chat_id), str(chat_id))

def _norm_date(s: str | None) -> str:
    """Return '10-Nov-2025 (Monday)' in IST, accepting many inputs."""
    if not s:
        now = datetime.now(BOT_TZ)
        return now.strftime("%d-%b-%Y (%A)")
    s = s.strip()
    # try common d/m/Y variants
    fmts = ["%d-%m-%Y", "%d.%m.%Y", "%d/%m/%Y", "%d-%b-%Y", "%d/%b/%Y", "%Y-%m-%d"]
    for fmt in fmts:
        try:
            dt = datetime.strptime(s, fmt).replace(tzinfo=BOT_TZ)
            return dt.strftime("%d-%b-%Y (%A)")
        except Exception:
            continue
    try:
        dt = datetime.fromisoformat(s)
        if dt.tzinfo is None:
            dt = dt.replace(tzinfo=BOT_TZ)
        return dt.astimezone(BOT_TZ).strftime("%d-%b-%Y (%A)")
    except Exception:
        # fallback: today
        return datetime.now(BOT_TZ).strftime("%d-%b-%Y (%A)")

# ---------- Excel zone rules ----------
def load_zone_rules(path: str) -> dict:
    rules = {}
    try:
        df = pd.read_excel(path, sheet_name="ZoneTime_Recommendations")
        for _, r in df.iterrows():
            feat = str(r.get("zone_time_feature", "")).lower()
            mins = r.get("suggested_minutes", np.nan)
            if not feat or pd.isna(mins):
                continue
            m = re.search(r"(\d{2,3})\s*[-–]\s*(\d{2,3})", feat)
            if not m:
                continue
            win = f"{m.group(1)}-{m.group(2)}"
            which = "separator" if "separator" in feat else "reactor"
            rules[f"{win} {which}"] = float(mins)
        LOG.info(f"Loaded {len(rules)} zone rules from Excel.")
    except Exception as e:
        LOG.warning(f"Excel load failed ({e}); using defaults.")
        rules = {
            "50-200 reactor": 165,
            "200-300 reactor": 68,
            "300-400 reactor": 185,
            "300-400 separator": 175,
            "400-450 reactor": 75,
            "450-480 reactor": 20,
            "480-500 reactor": 0,
            "500-520 reactor": 0
        }
    return rules

ZONE_RULES = load_zone_rules(REPORT_PATH)

def plan_for_feed(feed: dict) -> dict:
    """Start with rules, nudge 300–400 & 200–300 by composition."""
    plan = dict(ZONE_RULES)
    total = sum(feed.get(k, 0.0) for k in ("radial","nylon","chips","powder","kachra","others"))
    if total > 0:
        radial = feed.get("radial",0)/total
        chips  = feed.get("chips",0)/total
        nylon  = feed.get("nylon",0)/total
        # heavier → extend mid band; lighter nylon → reduce slightly
        adj_mid = 1.0 + 0.15*(radial + 0.5*chips - 0.6*nylon)
        if "300-400 reactor" in plan:
            plan["300-400 reactor"] = max(60, plan["300-400 reactor"]*adj_mid)
        if "300-400 separator" in plan:
            plan["300-400 separator"] = max(30, plan["300-400 separator"]*(0.9 + 0.2*(radial+chips)))
        if "200-300 reactor" in plan:
            plan["200-300 reactor"] = max(45, plan["200-300 reactor"]*(1.0 + 0.08*radial - 0.05*nylon))
    return plan

def predict_yield(feed: dict) -> tuple[float,float]:
    total = sum(feed.get(k,0.0) for k in ("radial","nylon","chips","powder","kachra","others"))
    if total <= 0:
        return (0.0, 0.6)
    w = state["weights"]
    base = 0.0
    for k, kg in feed.items():
        if k in w and total>0:
            base += (kg/total) * (w[k]*100.0)
    pred = max(30.0, min(55.0, base))
    spread = np.std([feed.get(x,0.0)/total for x in ("radial","nylon","chips","powder")])
    conf = max(0.60, min(0.92, 0.95 - 0.8*spread))
    return (round(pred,2), round(conf,2))

def learn_from_actual(feed: dict, actual_oil_pct: float):
    total = sum(feed.get(k,0.0) for k in ("radial","nylon","chips","powder","kachra","others"))
    if total <= 0:
        return
    pred,_ = predict_yield(feed)
    err = (actual_oil_pct - pred)/100.0
    for k in ("radial","nylon","chips","powder","kachra","others"):
        share = feed.get(k,0.0)/total
        state["weights"][k] += 0.01 * err * share
    save_state()

# ---------- Parsing ----------
def parse_feed_text(text: str) -> dict:
    # Accept variants: "/plan Feed: ...", "Feed: ...", "feed: ..."
    t = re.sub(r"(?i)^/?plan\s*", "", text).strip()
    t = re.sub(r"(?i)^feed\s*:\s*", "", t).strip()
    data = {}
    for part in re.split(r"[,\n;]+", t):
        if not part.strip():
            continue
        if "=" in part:
            k,v = [x.strip() for x in part.split("=",1)]
        else:
            m = re.match(r"([A-Za-z]+)\s+([\d\.]+)", part.strip())
            if not m:
                continue
            k,v = m.group(1), m.group(2)
        key = _norm_key(k)
        if key in ("batch","operator","date"):
            data[key] = v
            continue
        # units
        v = v.replace("ton","T").replace("mt","T")
        mT = re.match(r"([\d\.]+)\s*[tT]$", v)
        mK = re.match(r"([\d\.]+)\s*[kK][gG]?$", v)
        if mT:
            val = float(mT.group(1))*1000.0
        elif mK:
            val = float(mK.group(1))
        else:
            val = float(re.sub(r"[^\d\.]","", v) or 0.0)
        # map keys
        alias = {
            "radial":"radial","nylon":"nylon","chips":"chips","powder":"powder",
            "kachra":"kachra","others":"others"
        }
        data[alias.get(key,key)] = val
    return data

def parse_actual_text(text: str) -> dict:
    # Accept "/actual ..." or "actual: ..."
    t = re.sub(r"(?i)^/?actual\s*:?", "", text).strip()
    out = {}
    for part in re.split(r"[;,]+", t):
        part = part.strip()
        if not part:
            continue
        if "=" not in part:
            continue
        k,v = [x.strip() for x in part.split("=",1)]
        lk = _norm_key(k)
        if lk in ("oil","oilpct","oilpercent","oilpercentage"):
            out["oil"] = float(re.sub(r"[^\d\.]","", v) or 0.0)
        elif re.match(r"\d{2,3}\s*-\s*\d{2,3}", k):
            zkey = re.sub(r"\s+","", k) + " reactor"
            # If user typed just "300-400=01:20", treat as reactor zone
            out[zkey] = hhmmss_to_minutes(v)
        elif re.match(r"\d{2,3}\s*-\s*\d{2,3}\s*separator", k, re.I):
            zkey = re.sub(r"\s+"," ", k.lower())
            out[zkey] = hhmmss_to_minutes(v)
        elif lk == "batch":
            out["batch"] = v
    return out

# ---------- Rendering ----------
def _sorted_zones(keys):
    def kfun(s: str):
        m = re.match(r"(\d{2,3})-(\d{2,3})", s)
        if not m: return (999,999, s)
        return (int(m.group(1)), int(m.group(2)), s)
    return sorted(keys, key=kfun)

def pretty_plan(plan: dict) -> str:
    lines = []
    for z in _sorted_zones(plan.keys()):
        lines.append(f"{z}: {to_hhmmss(plan[z])}")
    return "\n".join(lines)

def plan_vs_actual_chart(plan: dict, actual: dict | None) -> io.BytesIO:
    zones = _sorted_zones(plan.keys())
    x = np.arange(len(zones))
    pmins = [plan[z] for z in zones]
    amins = [(actual.get(z, np.nan) if actual else np.nan) for z in zones]

    fig, ax = plt.subplots(figsize=(9.5, 3.6), dpi=160)
    width = 0.37
    ax.bar(x - width/2, pmins, width, label="Plan (min)")
    if actual:
        ax.bar(x + width/2, amins, width, label="Actual (min)")
    ax.set_xticks(x, zones, rotation=25)
    ax.set_ylabel("Minutes")
    ax.legend()
    ax.grid(alpha=0.25, axis="y")
    buf = io.BytesIO()
    plt.tight_layout()
    plt.savefig(buf, format="png")
    plt.close(fig)
    buf.seek(0)
    return buf

def _mixed_actual_reply(label: str, batch: str, operator: str, date_str: str,
                        oil_actual: float, oil_pred: float, deltas: dict,
                        oil_recos: list[str], quick_recos: list[str]) -> str:
    # grade by oil delta
    d = oil_actual - oil_pred
    if d >= -1.0:
        grade = "B"
        bullet = "🟡"
    elif d >= -4.0:
        grade = "C"
        bullet = "🟠"
    else:
        grade = "D"
        bullet = "🔴"

    # deltas
    delta_lines = []
    for z in _sorted_zones(deltas.keys()):
        delta_lines.append(f"• {z}: {deltas[z]:+d}")

    oil_lines = "\n".join([f"• {x}" for x in oil_recos]) if oil_recos else "• Keep separator knock-out stable; check condenser ΔT."
    quick_lines = "\n".join([f"• {x}" for x in quick_recos]) if quick_recos else "• Minor trims only."

    msg = (
f"📇 {label} • Batch {batch} • Operator: {operator}\n"
f"📅 {date_str}\n\n"
f"{bullet} Oil vs Predicted: actual {oil_actual:.2f}% (pred {oil_pred:.2f}%, Δ = {d:+.2f}%) → Grade {grade}\n"
f"🧰 Oil yield recommendations:\n{oil_lines}\n\n"
f"📊 Deviation vs plan (min):\n" + "\n".join(delta_lines) + "\n\n"
f"🛠️ Quick zone moves:\n{quick_lines}"
    )
    return msg

# ---------- Reminder + Summary ----------
def _bkey(chat_id: int, batch: str | int | None) -> str:
    return f"{chat_id}:{batch or ''}"

async def _reminder_once(context: ContextTypes.DEFAULT_TYPE):
    data = context.job.data or {}
    chat_id = data.get("chat_id")
    batch   = data.get("batch")
    label   = _machine_label(chat_id)
    txt = (f"⚠️ *Reminder:* Actual data not entered yet for *Batch {batch}* on *{label}*.\n"
           f"Please send:\n`Actual: 50-200=hh:mm, 200-300=hh:mm, 300-400=hh:mm, 300-400 separator=hh:mm, 400-450=hh:mm, 450-480=hh:mm; oil=xx.x; batch={batch}`")
    await context.bot.send_message(chat_id, txt, parse_mode=ParseMode.MARKDOWN)
    if SUMMARY_CHAT_ID:
        await context.bot.send_message(SUMMARY_CHAT_ID, f"🔔 {txt}", parse_mode=ParseMode.MARKDOWN)

def schedule_reminder(jobq, chat_id: int, batch: str):
    # after 12h, then hourly
    due = datetime.now(BOT_TZ) + timedelta(hours=12)
    name = f"reminder:{_bkey(chat_id, batch)}"
    jobq.run_repeating(
        _reminder_once,
        first=due,
        interval=3600,   # hourly
        name=name,
        data={"chat_id": chat_id, "batch": batch}
    )

def cancel_reminder(jobq, chat_id: int, batch: str):
    name = f"reminder:{_bkey(chat_id, batch)}"
    for job in jobq.get_jobs_by_name(name):
        job.schedule_removal()

async def _daily_summary(context: ContextTypes.DEFAULT_TYPE):
    now = datetime.now(BOT_TZ)
    lines = [f"📊 *Daily Summary* — {now.strftime('%d-%b %H:%M')} (IST)"]
    for cid_str, label in MACHINE_MAP.items():
        lf = state["latest_feed"].get(cid_str)
        status = "Idle"
        if lf:
            feed_ts = datetime.fromisoformat(lf["ts"]).astimezone(BOT_TZ)
            hrs = (now - feed_ts).total_seconds()/3600
            last_act_iso = state["last_actual_ts"].get(cid_str)
            completed = False
            if last_act_iso:
                completed = datetime.fromisoformat(last_act_iso) >= datetime.fromisoformat(lf["ts"])
            if hrs <= 12 and not completed:
                status = f"Running (batch {lf.get('batch','?')})"
            elif completed:
                status = f"Completed — Batch {lf.get('batch','?')} on {lf.get('date','')}"
            else:
                status = "Idle"
        lines.append(f"• {label}: {status}")
    if SUMMARY_CHAT_ID:
        await context.bot.send_message(SUMMARY_CHAT_ID, "\n".join(lines), parse_mode=ParseMode.MARKDOWN)

def schedule_daily(jobq):
    # 21:35 IST
    jobq.run_daily(
        _daily_summary,
        time=dt_time(hour=21, minute=35, tzinfo=BOT_TZ),
        name="daily_summary"
    )

# ---------- Handlers ----------
HELP = (
"*Commands*\n"
"• Send `Feed: Radial=5.1T, Nylon=0.6T, Chips=3.4T, Powder=1.5T, batch=92, operator=Ravi, date=10-11-2025` → Plan + Predicted oil (also posted in Summary).\n"
"• Send `Actual: 50-200=01:10, 200-300=00:45, 300-400=01:20, 300-400 separator=00:40, 400-450=00:22, 450-480=00:10; oil=46.2; batch=92` → Deviation + Oil recommendations + Chart (and the model learns).\n"
"• `/status` → Machine status (or all, if used in Summary group).\n"
"• `/reload` → Reload Excel rules.\n"
"• `/id` → Show this chat’s id and label.\n"
)

async def cmd_start(update: Update, context: ContextTypes.DEFAULT_TYPE):
    await update.message.reply_text("PyroVision Assistant ready ✅\n\n"+HELP, parse_mode=ParseMode.MARKDOWN)

async def cmd_help(update: Update, context: ContextTypes.DEFAULT_TYPE):
    await update.message.reply_text(HELP, parse_mode=ParseMode.MARKDOWN)

async def cmd_id(update: Update, context: ContextTypes.DEFAULT_TYPE):
    cid = update.effective_chat.id
    await update.message.reply_text(f"chat_id: `{cid}`\nname/title: *{_machine_label(cid)}*", parse_mode=ParseMode.MARKDOWN)

async def cmd_reload(update: Update, context: ContextTypes.DEFAULT_TYPE):
    global ZONE_RULES
    ZONE_RULES = load_zone_rules(REPORT_PATH)
    await update.message.reply_text("🔁 Reloaded recommendations from Excel.")

def _remember_feed(jobq, chat_id: int, feed: dict, plan: dict):
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

    if entry["batch"]:
        schedule_reminder(jobq, chat_id, entry["batch"])

async def send_plan(update: Update, context: ContextTypes.DEFAULT_TYPE, feed_text: str):
    chat_id = update.effective_chat.id
    feed = parse_feed_text(feed_text)
    plan = plan_for_feed(feed)
    pred, conf = predict_yield(feed)

    label = _machine_label(chat_id)
    batch = feed.get("batch","?")
    oper  = feed.get("operator","?")
    date_str = _norm_date(feed.get("date"))

    _remember_feed(context.job_queue, chat_id, feed, plan)

    head = (
        f"📒 *{label}* — *Batch* {batch}, *Operator* {oper}\n"
        f"• *Date* {date_str}\n\n"
        f"🛢️ *Feed:* Radial {feed.get('radial',0)/1000:.2f}T, Nylon {feed.get('nylon',0)/1000:.2f}T, "
        f"Chips {feed.get('chips',0)/1000:.2f}T, Powder {feed.get('powder',0)/1000:.2f}T\n"
        f"📈 *Predicted Oil:* *{pred:.2f}%*  _(confidence {conf:.2f})_\n\n"
        f"*Recommended zone minutes (plan):*\n{pretty_plan(plan)}"
    )
    await update.message.reply_text(head, parse_mode=ParseMode.MARKDOWN)
    if SUMMARY_CHAT_ID:
        await context.bot.send_message(SUMMARY_CHAT_ID, head, parse_mode=ParseMode.MARKDOWN)

async def cmd_status(update: Update, context: ContextTypes.DEFAULT_TYPE):
    now = datetime.now(BOT_TZ)
    if SUMMARY_CHAT_ID and update.effective_chat.id == SUMMARY_CHAT_ID:
        lines = ["🟢 *Machine Status*"]
        for cid_str,label in MACHINE_MAP.items():
            lf = state["latest_feed"].get(cid_str)
            st = "Idle"
            if lf:
                feed_ts = datetime.fromisoformat(lf["ts"]).astimezone(BOT_TZ)
                hrs = (now - feed_ts).total_seconds()/3600
                last_act_iso = state["last_actual_ts"].get(cid_str)
                completed = False
                if last_act_iso:
                    completed = datetime.fromisoformat(last_act_iso) >= datetime.fromisoformat(lf["ts"])
                if hrs <= 12 and not completed:
                    st = f"Running (batch {lf.get('batch','?')})"
                elif completed:
                    st = f"Completed — Batch {lf.get('batch','?')} on {lf.get('date','')}"
            lines.append(f"• {label}: {st}")
        await update.message.reply_text("\n".join(lines), parse_mode=ParseMode.MARKDOWN)
        return

    cid = str(update.effective_chat.id)
    label = _machine_label(update.effective_chat.id)
    lf = state["latest_feed"].get(cid)
    st = "Idle"
    if lf:
        feed_ts = datetime.fromisoformat(lf["ts"]).astimezone(BOT_TZ)
        hrs = (now - feed_ts).total_seconds()/3600
        last_act_iso = state["last_actual_ts"].get(cid)
        completed = last_act_iso and (datetime.fromisoformat(last_act_iso) >= datetime.fromisoformat(lf["ts"]))
        if hrs <= 12 and not completed:
            st = f"Running (batch {lf.get('batch','?')})"
        elif completed:
            st = f"Completed — Batch {lf.get('batch','?')} on {lf.get('date','')}"
    await update.message.reply_text(f"{label}: {st}")

async def cmd_actual(update: Update, context: ContextTypes.DEFAULT_TYPE, actual_text: str):
    chat_id = update.effective_chat.id
    cid = str(chat_id)
    lf = state["latest_feed"].get(cid)
    if not lf:
        await update.message.reply_text("No recorded Feed found for this machine. Send the Feed first.")
        return

    actual = parse_actual_text(actual_text)
    plan = lf.get("plan", {})
    # build deltas
    deltas = {}
    for z, pmin in plan.items():
        # actual keys may omit 'reactor' suffix for main zones—we added default earlier
        am = actual.get(z, np.nan)
        if isinstance(am, float) and not math.isnan(am):
            deltas[z] = int(round(am - pmin))

    # oil learning
    oil_actual = float(actual.get("oil", 0.0))
    pred, _ = predict_yield(lf.get("feed", {}))
    if "oil" in actual:
        learn_from_actual(lf.get("feed", {}), oil_actual)
        state["last_actual_ts"][cid] = datetime.now(BOT_TZ).isoformat()
        save_state()
        # stop reminders
        if lf.get("batch"):
            cancel_reminder(context.job_queue, chat_id, lf["batch"])

    # quick zone recos from deltas (significant ±)
    quick = []
    for z in _sorted_zones(deltas.keys()):
        dv = deltas[z]
        if abs(dv) >= 10:
            if dv > 0:
                quick.append(f"reduce *{z}* by ~{abs(dv)} min")
            else:
                quick.append(f"increase *{z}* by ~{abs(dv)} min")

    # oil-specific recos based on under-yield vs prediction
    oil_recos = []
    gap = oil_actual - pred  # negative if under
    if gap < -1.0:
        oil_recos = [
            "Extend *300–400 reactor* by ~20–30 min for heavier mix completion.",
            "Increase *300–400 separator* by ~15–20 min for stronger condensation.",
            "Slightly extend *200–300 reactor* by ~10–15 min to stabilize ramp.",
            "Check line losses & condenser ΔT; avoid sharp heat spikes."
        ]
    elif gap > 1.0:
        oil_recos = [
            "Good yield vs model — consider trimming tails in *450–480* by 5–10 min to save fuel.",
            "Keep separator steady; avoid over-condensing water."
        ]

    label = _machine_label(chat_id)
    msg = _mixed_actual_reply(
        label=label,
        batch=lf.get("batch","?"),
        operator=lf.get("operator","?"),
        date_str=lf.get("date", _norm_date(None)),
        oil_actual=oil_actual,
        oil_pred=pred,
        deltas=deltas,
        oil_recos=oil_recos,
        quick_recos=quick
    )
    await update.message.reply_text(msg, parse_mode=ParseMode.MARKDOWN)

    # chart
    buf = plan_vs_actual_chart(plan, {k: plan.get(k) + deltas.get(k, np.nan) if k in deltas else np.nan for k in plan})
    await context.bot.send_photo(chat_id, InputFile(buf, filename="plan_vs_actual.png"))

    # also ping Summary group
    if SUMMARY_CHAT_ID:
        await context.bot.send_message(SUMMARY_CHAT_ID, f"🧾 Actual logged for *{label}* (batch {lf.get('batch','?')}).", parse_mode=ParseMode.MARKDOWN)

async def text_router(update: Update, context: ContextTypes.DEFAULT_TYPE):
    text = (update.message.text or "").strip()
    # normalize for matching
    low = text.lower()

    if low.startswith("/plan") or low.startswith("plan ") or low.startswith("feed:") or low.startswith("feed "):
        await send_plan(update, context, text)
        return

    if low.startswith("/actual") or low.startswith("actual") or low.startswith(" actual"):
        await cmd_actual(update, context, text)
        return

    if low == "status" or low == "/status":
        await cmd_status(update, context)
        return

    await update.message.reply_text("I didn’t understand.\n\n"+HELP, parse_mode=ParseMode.MARKDOWN)

# ---------- Main ----------
def main():
    LOG.info("🚀 Starting PyroVision Assistant…")
    load_state()
    if not BOT_TOKEN:
        raise RuntimeError("TELEGRAM_BOT_TOKEN is missing.")

    app = Application.builder().token(BOT_TOKEN).build()

    # commands
    app.add_handler(CommandHandler("start", cmd_start))
    app.add_handler(CommandHandler("help",  cmd_help))
    app.add_handler(CommandHandler("id",    cmd_id))
    app.add_handler(CommandHandler("reload",cmd_reload))
    app.add_handler(CommandHandler("status",cmd_status))

    # text router (Feed/Actual/Status plain messages)
    app.add_handler(MessageHandler(filters.TEXT & ~filters.COMMAND, text_router))

    # jobs
    schedule_daily(app.job_queue)

    LOG.info("✅ Bot ready. Polling…")
    app.run_polling(drop_pending_updates=True)

if __name__ == "__main__":
    main()
