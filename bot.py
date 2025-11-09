# ==========================================
#  PyroVision Assistant  —  Railway version
#  PTB v21.x  |  Python 3.11
# ==========================================

import os, re, io, json, logging, asyncio, math
from datetime import datetime, timedelta, time, timezone
from zoneinfo import ZoneInfo

import numpy as np
import pandas as pd
from telegram import Update, InputFile
from telegram.constants import ParseMode
from telegram.ext import (
    Application, CommandHandler, MessageHandler, ContextTypes, filters,
)

# ---------- logging ----------
LOG = logging.getLogger("pyro_bot")
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)s | %(name)s | %(message)s"
)

# ---------- config & env ----------
TELEGRAM_BOT_TOKEN = os.environ.get("TELEGRAM_BOT_TOKEN", "").strip()
REPORT_PATH = os.environ.get("REPORT_PATH", "pyrolysis_feed_temp_ZONE_TIME_report.xlsx")
ZONE_SHEET = os.environ.get("ZONE_SHEET", "ZoneTime_Recommendations")

# TZ: Asia/Kolkata (IST)
BOT_TZ = ZoneInfo("Asia/Kolkata")

# Group mapping (defaults to your IDs; can be overridden via env)
DEFAULT_MACHINE_MAP = {
    "-4923260341": "Machine 1296 (R-1)",
    "-4859362706": "Machine 1297 (R-2)",
    "-4921356716": "Machine 1298 (R-3)",
    "-4628429766": "Machine 1299 (R-4)",
}
DEFAULT_SUMMARY_CHAT_ID = "-4865673071"

MACHINE_MAP = json.loads(os.environ.get("MACHINE_MAP", json.dumps(DEFAULT_MACHINE_MAP)))
SUMMARY_CHAT_ID = os.environ.get("SUMMARY_CHAT_ID", DEFAULT_SUMMARY_CHAT_ID)
try:
    SUMMARY_CHAT_ID = int(SUMMARY_CHAT_ID)
except Exception:
    SUMMARY_CHAT_ID = None

# ---------- persistence ----------
STATE_FILE = "bot_state.json"
state = {
    "weights": {"radial": .47, "nylon": .42, "chips": .44, "powder": .43, "kachra": .40, "others": .40},
    "latest_feed": {},          # chat_id -> {ts, batch, operator, feed, plan}
    "last_actual_ts": {},       # chat_id -> iso
    "reminders": {}             # key(chat:batch) -> {chat_id, batch}
}

def load_state():
    global state
    try:
        if os.path.exists(STATE_FILE):
            with open(STATE_FILE, "r", encoding="utf-8") as f:
                obj = json.load(f)
            # shallow merge (to keep defaults if keys missing)
            for k,v in obj.items():
                state[k] = v
            LOG.info("✅ State loaded.")
        else:
            LOG.info("ℹ️ No state file found; starting fresh.")
    except Exception as e:
        LOG.warning(f"⚠️ Could not load state: {e}")

def save_state():
    try:
        with open(STATE_FILE, "w", encoding="utf-8") as f:
            json.dump(state, f, ensure_ascii=False, indent=2)
    except Exception as e:
        LOG.warning(f"⚠️ Could not save state: {e}")

# ---------- utilities ----------
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
        return float(re.sub(r"[^\d.]", "", s) or 0.0)
    parts = [int(p) for p in s.split(":")]
    if len(parts) == 3:
        h, m, sec = parts
        return h * 60 + m + sec / 60
    if len(parts) == 2:
        m, sec = parts
        return m + sec / 60
    return 0.0

def norm_key(k: str) -> str:
    return re.sub(r"\s+", "", (k or "").strip().lower())

def machine_label(chat_id: int) -> str:
    # try string & int versions
    return MACHINE_MAP.get(str(chat_id)) or MACHINE_MAP.get(str(int(chat_id))) or str(chat_id)

def batch_key(chat_id: int, batch: str | int | None) -> str:
    return f"{chat_id}:{batch or ''}"

# ---------- zone rules ----------
def load_zone_rules(path=REPORT_PATH, sheet=ZONE_SHEET) -> dict[str, float]:
    rules: dict[str, float] = {}
    try:
        df = pd.read_excel(path, sheet_name=sheet)
        for _, row in df.iterrows():
            feat = str(row.get("zone_time_feature", "")).strip().lower()
            mins = row.get("suggested_minutes", np.nan)
            if not feat or pd.isna(mins):
                continue
            m = re.search(r"(\d{2,3}\s*[-–]\s*\d{2,3})", feat)
            if not m:
                continue
            window = m.group(1).replace(" ", "").replace("–", "-")
            which = "separator" if "separator" in feat else "reactor"
            rules[f"{window} {which}"] = float(mins)
        LOG.info("Loaded %d zone rules from Excel.", len(rules))
    except Exception as e:
        LOG.warning("⚠️ Excel load failed (%s). Using conservative defaults.", e)
        rules.update({
            "50-200 reactor": 60,
            "200-300 reactor": 60,
            "300-400 reactor": 70,
            "400-450 reactor": 80,
            "450-480 reactor": 25,
            "480-500 reactor": 15,
            "300-400 separator": 30,
        })
    return rules

ZONE_RULES = load_zone_rules()

# ---------- parsing ----------
def parse_feed(text: str) -> dict:
    """
    Feed: Radial=5.1T, Nylon=0.6T, Chips=3.4T, Powder=1.5T, batch=92, operator=Ravi
    """
    data: dict = {}
    t = re.sub(r"^/?(plan|predict|whatif)?\s*feed\s*:\s*|^feed\s*:\s*", "", text, flags=re.I).strip()
    for part in re.split(r"[,\n;]+", t):
        if not part.strip():
            continue
        if "=" in part:
            k, v = [x.strip() for x in part.split("=", 1)]
        else:
            m = re.match(r"([A-Za-z]+)\s+([\d.]+)", part.strip())
            if not m:
                continue
            k, v = m.group(1), m.group(2)
        key = norm_key(k)

        if key in ("batch", "operator", "machine", "date"):
            data[key] = v
            continue

        v2 = v.replace("ton", "T").replace("mt", "T").replace("kg", "K")
        mT = re.match(r"([\d.]+)\s*[tT]$", v2)
        mK = re.match(r"([\d.]+)\s*[kK]", v2)
        if mT:
            val = float(mT.group(1)) * 1000.0
        elif mK:
            val = float(mK.group(1))
        else:
            val = float(re.sub(r"[^\d.]", "", v2) or 0.0)
        data[key] = val

    mapping = {
        "radial": "radial", "nylon": "nylon", "chips": "chips",
        "powder": "powder", "kachra": "kachra", "others": "others"
    }
    return {mapping.get(k, k): v for k, v in data.items()}

def parse_actual(text: str) -> dict:
    """
    Actual: 50-200=01:10, 200-300=00:45, 300-400=01:20, 400-450=00:22; oil=46.2; batch=92
    """
    t = re.sub(r"^/?actual\s*:?","", text, flags=re.I).strip()
    out: dict = {}
    for chunk in re.split(r"[;,]+", t):
        if "=" not in chunk:
            continue
        k, v = [x.strip() for x in chunk.split("=", 1)]
        lk = norm_key(k)
        if lk in ("oil", "oil%"):
            out["oil"] = float(re.sub(r"[^\d.]", "", v) or 0.0)
        elif re.match(r"\d{2,3}\s*-\s*\d{2,3}", lk):
            out[lk.replace(" ", "")] = hhmmss_to_minutes(v)
        elif lk == "batch":
            out["batch"] = v
    return out

# ---------- planning + tiny learner ----------
class RecoEngine:
    def __init__(self, rules: dict[str, float]):
        self.rules = rules

    def reload(self):
        self.rules = load_zone_rules()

    def plan(self, feed: dict) -> dict[str, float]:
        plan = dict(self.rules)
        total = sum(feed.get(k, 0.0) for k in ("radial","nylon","chips","powder","kachra","others"))
        if total > 0:
            radial_ratio = feed.get("radial",0.0)/total
            chips_ratio  = feed.get("chips",0.0)/total
            nylon_ratio  = feed.get("nylon",0.0)/total
            adj = 1.0 + 0.15*(radial_ratio + 0.5*chips_ratio - 0.6*nylon_ratio)
            for k in list(plan.keys()):
                if k.startswith("300-400"):
                    plan[k] = max(20, plan[k]*adj)
                if k.startswith("200-300"):
                    plan[k] = max(15, plan[k]*(1.0 + 0.1*radial_ratio - 0.05*nylon_ratio))
        return plan

engine = RecoEngine(ZONE_RULES)

def predict_yield(feed: dict) -> tuple[float, float]:
    total = sum(feed.get(k,0.0) for k in ("radial","nylon","chips","powder","kachra","others"))
    if total <= 0:
        return (0.0, 0.55)
    w = state["weights"]
    base = 0.0
    for k, kg in feed.items():
        if k in w:
            base += (kg/total) * (w[k]*100.0)
    pred = max(30.0, min(55.0, base))
    ratio_spread = np.std([feed.get(x,0.0)/total for x in ("radial","nylon","chips","powder")])
    conf = max(0.55, min(0.95, 0.95 - 0.8*ratio_spread))
    return (round(pred,2), round(conf,2))

def learn_from_actual(feed: dict, actual_oil: float):
    total = sum(feed.get(k,0.0) for k in ("radial","nylon","chips","powder","kachra","others"))
    if total <= 0:
        return
    pred,_ = predict_yield(feed)
    err = (actual_oil - pred)/100.0
    for k in ("radial","nylon","chips","powder","kachra","others"):
        share = feed.get(k,0.0)/total
        state["weights"][k] += 0.01 * err * share
    save_state()

# ---------- formatting ----------
def pretty_plan(plan: dict[str, float]) -> str:
    def keyer(x: str):
        m = re.match(r"(\d{2,3})-(\d{2,3})", x)
        return (int(m.group(1)), int(m.group(2))) if m else (999, 999)
    lines = []
    for z in sorted(plan.keys(), key=keyer):
        lines.append(f"{z}: {to_hhmmss(plan[z])}")
    return "\n".join(lines)

# ---------- jobs (reminders + summary) ----------
async def reminder_job(context: ContextTypes.DEFAULT_TYPE):
    data = context.job.data or {}
    chat_id = data.get("chat_id")
    batch = data.get("batch")
    key = batch_key(chat_id, batch)
    if key in state["reminders"]:
        txt = (f"⚠️ *Reminder:* Actual data not entered yet for *Batch {batch}* "
               f"on *{machine_label(chat_id)}*. Please send `Actual: ...`")
        await context.bot.send_message(chat_id, txt, parse_mode=ParseMode.MARKDOWN)
        if SUMMARY_CHAT_ID:
            await context.bot.send_message(SUMMARY_CHAT_ID, f"🔔 {txt}", parse_mode=ParseMode.MARKDOWN)
        # keep repeating hourly until cleared

async def daily_summary_job(context: ContextTypes.DEFAULT_TYPE):
    now = datetime.now(BOT_TZ)
    lines = [f"📊 *Daily Summary* — {now.strftime('%d-%b %H:%M')} IST"]
    for cid_str, label in MACHINE_MAP.items():
        lf = state["latest_feed"].get(cid_str)
        status = "Idle"
        if lf:
            last_feed_ts = datetime.fromisoformat(lf["ts"])
            hrs = (now - last_feed_ts.astimezone(BOT_TZ)).total_seconds()/3600
            last_act_iso = state["last_actual_ts"].get(cid_str)
            completed = False
            if last_act_iso:
                last_act_ts = datetime.fromisoformat(last_act_iso)
                completed = last_act_ts >= last_feed_ts
            if hrs <= 11 and not completed:
                status = f"Running (batch {lf.get('batch','?')})"
            elif completed or hrs > 11:
                status = f"Completed (batch {lf.get('batch','?')})"
        lines.append(f"• {label}: {status}")
    if SUMMARY_CHAT_ID:
        await context.bot.send_message(SUMMARY_CHAT_ID, "\n".join(lines), parse_mode=ParseMode.MARKDOWN)

def schedule_jobs(app: Application):
    # daily summary at 21:35 IST
    if app.job_queue is None:
        LOG.warning("No JobQueue available. Install PTB with [job-queue].")
        return
    app.job_queue.run_daily(
        daily_summary_job,
        time=time(21, 35, tzinfo=BOT_TZ),
        name="daily_summary"
    )

# ---------- command handlers ----------
HELP_TEXT = (
    "*Commands*\n"
    "• `Feed: Radial=5.1T, Nylon=0.6T, Chips=3.4T, Powder=1.5T, batch=92, operator=Ravi` → Plan + Predicted Oil\n"
    "• `/whatif Feed: …` → Predicted Oil + Confidence (no plan)\n"
    "• `Actual: 50-200=1:10, 200-300=0:45, 300-400=1:20, 400-450=0:22; oil=46.2; batch=92` → Compare & Learn\n"
    "• `/status` → Status of this machine (or all in summary group)\n"
    "• `/reload` → Reload Excel rules\n"
    "• `/help` → Show commands\n"
)

async def start_cmd(update: Update, context: ContextTypes.DEFAULT_TYPE):
    await update.message.reply_text("PyroVision Assistant ready ✅\n\n" + HELP_TEXT, parse_mode=ParseMode.MARKDOWN)

async def help_cmd(update: Update, context: ContextTypes.DEFAULT_TYPE):
    await update.message.reply_text(HELP_TEXT, parse_mode=ParseMode.MARKDOWN)

async def reload_cmd(update: Update, context: ContextTypes.DEFAULT_TYPE):
    engine.reload()
    await update.message.reply_text("🔁 Rules reloaded from Excel.")

async def status_cmd(update: Update, context: ContextTypes.DEFAULT_TYPE):
    cid = update.effective_chat.id
    now = datetime.now(BOT_TZ)

    if SUMMARY_CHAT_ID and cid == SUMMARY_CHAT_ID:
        lines = ["🟢 *Machine Status*"]
        for cid_str, label in MACHINE_MAP.items():
            lf = state["latest_feed"].get(cid_str)
            st = "Idle"
            if lf:
                last_feed_ts = datetime.fromisoformat(lf["ts"])
                hrs = (now - last_feed_ts.astimezone(BOT_TZ)).total_seconds()/3600
                last_act_iso = state["last_actual_ts"].get(cid_str)
                completed = last_act_iso and (datetime.fromisoformat(last_act_iso) >= last_feed_ts)
                if hrs <= 11 and not completed:
                    st = f"Running (batch {lf.get('batch','?')})"
                elif completed:
                    st = f"Completed (batch {lf.get('batch','?')})"
                elif hrs > 12:
                    st = "Overdue (no actual)"
            lines.append(f"• {label}: {st}")
        await update.message.reply_text("\n".join(lines), parse_mode=ParseMode.MARKDOWN)
        return

    label = machine_label(cid)
    lf = state["latest_feed"].get(str(cid))
    st = "Idle"
    if lf:
        last_feed_ts = datetime.fromisoformat(lf["ts"])
        hrs = (now - last_feed_ts.astimezone(BOT_TZ)).total_seconds()/3600
        last_act_iso = state["last_actual_ts"].get(str(cid))
        completed = last_act_iso and (datetime.fromisoformat(last_act_iso) >= last_feed_ts)
        if hrs <= 11 and not completed:
            st = f"Running (batch {lf.get('batch','?')})"
        elif completed:
            st = f"Completed (batch {lf.get('batch','?')})"
        elif hrs > 12:
            st = "Overdue (no actual)"
    await update.message.reply_text(f"{label}: {st}")

def remember_feed(chat_id: int, feed: dict, plan: dict):
    now = datetime.now(BOT_TZ)
    entry = {
        "ts": now.isoformat(),
        "batch": str(feed.get("batch","")),
        "operator": str(feed.get("operator","")),
        "feed": feed,
        "plan": plan,
    }
    state["latest_feed"][str(chat_id)] = entry
    save_state()

async def plan_core(update: Update, context: ContextTypes.DEFAULT_TYPE, feed_text: str, predict_only: bool):
    chat_id = update.effective_chat.id
    label = machine_label(chat_id)

    feed = parse_feed(feed_text)
    plan = engine.plan(feed)
    pred, conf = predict_yield(feed)
    batch = feed.get("batch","?")
    oper  = feed.get("operator","?")

    # remember & schedule reminder (+12h first, then hourly)
    remember_feed(chat_id, feed, plan)
    if context.job_queue:
        first_due = datetime.now(BOT_TZ) + timedelta(hours=12)
        key = batch_key(chat_id, batch)
        state["reminders"][key] = {"chat_id": chat_id, "batch": batch}
        save_state()
        context.job_queue.run_repeating(
            reminder_job,
            interval=3600,                  # hourly repeats
            first=first_due,
            name=f"reminder:{key}",
            data={"chat_id": chat_id, "batch": batch}
        )

    lines = [
        f"🏷️ *{label}* — *Batch* {batch}, *Operator* {oper}",
        f"🛢️ Feed: Radial {feed.get('radial',0)/1000:.2f}T, Nylon {feed.get('nylon',0)/1000:.2f}T, Chips {feed.get('chips',0)/1000:.2f}T, Powder {feed.get('powder',0)/1000:.2f}T",
        f"📈 Predicted Oil: *{pred:.2f}%*  (confidence {conf:.2f})",
    ]
    if not predict_only:
        lines.append("\n*Recommended zone minutes (plan):*\n" + pretty_plan(plan))
    msg = "\n".join(lines)

    await update.message.reply_text(msg, parse_mode=ParseMode.MARKDOWN)
    if SUMMARY_CHAT_ID:
        await context.bot.send_message(SUMMARY_CHAT_ID, msg, parse_mode=ParseMode.MARKDOWN)

async def plan_cmd(update: Update, context: ContextTypes.DEFAULT_TYPE):
    await plan_core(update, context, update.message.text, predict_only=False)

async def whatif_cmd(update: Update, context: ContextTypes.DEFAULT_TYPE):
    t = re.sub(r"^/?whatif\s*", "", update.message.text, flags=re.I)
    await plan_core(update, context, t, predict_only=True)

async def actual_cmd(update: Update, context: ContextTypes.DEFAULT_TYPE):
    chat_id = update.effective_chat.id
    label = machine_label(chat_id)
    actual = parse_actual(update.message.text)

    last = state["latest_feed"].get(str(chat_id))
    if not last:
        await update.message.reply_text("I don't find a recorded Feed for this machine. Send the Feed first.")
        return

    plan = last.get("plan", {})
    deltas: dict[str, float] = {}
    tips: list[str] = []

    for z, pmin in plan.items():
        zkey = z.replace(" ", "")
        if zkey in actual:
            am = actual[zkey]
            deltas[z] = am - pmin
            if abs(deltas[z]) >= 5:
                tips.append(f"• {'reduce' if deltas[z] > 0 else 'increase'} {z} by ~{abs(deltas[z]):.0f} min")

    if "oil" in actual:
        learn_from_actual(last.get("feed", {}), float(actual["oil"]))
        state["last_actual_ts"][str(chat_id)] = datetime.now(BOT_TZ).isoformat()
        # clear repeating reminder
        key = batch_key(chat_id, last.get("batch"))
        if key in state["reminders"]:
            del state["reminders"][key]
        save_state()

    lines = ["📊 Deviation vs plan (min):"]
    for z in sorted(deltas.keys(), key=lambda s: int(s.split("-")[0])):
        lines.append(f"{z}: {deltas[z]:+.0f}")
    if tips:
        lines.append("\n🛠️ Recommendations:")
        lines.extend(tips)
    else:
        lines.append("\n✅ Near-optimal execution vs plan.")

    await update.message.reply_text("\n".join(lines))
    if SUMMARY_CHAT_ID:
        await context.bot.send_message(SUMMARY_CHAT_ID, f"🧾 Actual logged for {label}.", parse_mode=ParseMode.MARKDOWN)

async def text_router(update: Update, context: ContextTypes.DEFAULT_TYPE):
    text = (update.message.text or "").strip()
    if re.match(r"(?i)^feed\s*:", text):
        await plan_cmd(update, context)
    elif re.match(r"(?i)^actual\s*:", text):
        await actual_cmd(update, context)
    elif re.match(r"(?i)^status$", text):
        await status_cmd(update, context)
    elif re.match(r"(?i)^/whatif", text):
        await whatif_cmd(update, context)
    else:
        await update.message.reply_text("I didn’t understand. Type /help for examples or send a `Feed:` / `Actual:` line.")

# ---------- main ----------
def main():
    if not TELEGRAM_BOT_TOKEN:
        raise RuntimeError("Missing TELEGRAM_BOT_TOKEN env var!")

    load_state()

    app = Application.builder().token(TELEGRAM_BOT_TOKEN).build()

    # Commands
    app.add_handler(CommandHandler("start", start_cmd))
    app.add_handler(CommandHandler("help", help_cmd))
    app.add_handler(CommandHandler("reload", reload_cmd))
    app.add_handler(CommandHandler("status", status_cmd))
    app.add_handler(CommandHandler("plan", plan_cmd))
    app.add_handler(CommandHandler("whatif", whatif_cmd))
    app.add_handler(CommandHandler("actual", actual_cmd))

    # Free text router
    app.add_handler(MessageHandler(filters.TEXT & ~filters.COMMAND, text_router))

    # Jobs
    schedule_jobs(app)

    LOG.info("✅ Bot ready. Polling…")
    app.run_polling()

if __name__ == "__main__":
    main()
