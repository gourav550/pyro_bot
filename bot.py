import os, json, re, io, math, logging
from datetime import datetime, timedelta
from typing import Dict, Tuple
import pandas as pd
import numpy as np

from telegram import Update
from telegram.ext import (
    Application, CommandHandler, MessageHandler, ContextTypes, filters
)

# Optional scheduler for daily summary & reminders
from apscheduler.schedulers.background import BackgroundScheduler
import zoneinfo

# =============== BASIC CONFIG =================
LOG = logging.getLogger("pyro_bot")
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)s | %(name)s | %(message)s"
)

BOT_TZ = zoneinfo.ZoneInfo("Asia/Kolkata")  # IST (Kolkata)

TELEGRAM_BOT_TOKEN = os.environ.get("TELEGRAM_BOT_TOKEN", "")
REPORT_PATH = os.environ.get("REPORT_PATH", "pyrolysis_feed_temp_ZONE_TIME_report.xlsx")
ZONE_SHEET = "ZoneTime_Recommendations"

SUMMARY_CHAT_ID_ENV = os.environ.get("SUMMARY_CHAT_ID", "").strip()
SUMMARY_CHAT_ID = int(SUMMARY_CHAT_ID_ENV) if SUMMARY_CHAT_ID_ENV else None

MACHINE_MAP_ENV = os.environ.get("MACHINE_MAP", "").strip()
try:
    MACHINE_MAP: Dict[str, str] = json.loads(MACHINE_MAP_ENV) if MACHINE_MAP_ENV else {}
except Exception:
    MACHINE_MAP = {}

STATE_FILE = "bot_state.json"
state = {
    "weights": {  # crude per-material baseline (fractional yield contribution)
        "radial": .47, "nylon": .42, "chips": .44, "powder": .43, "kachra": .40, "others": .40
    },
    "latest_feed": {},      # chat_id -> {ts, batch, operator, date_str, feed, plan}
    "last_actual_ts": {},   # chat_id -> iso ts
    "reminders": {}         # key(chat:batch) -> {"chat_id","batch","due_iso"}
}

scheduler: BackgroundScheduler | None = None


# =============== STATE HANDLING =================
def load_state():
    global state
    try:
        if os.path.exists(STATE_FILE):
            with open(STATE_FILE, "r", encoding="utf-8") as f:
                obj = json.load(f)
            # shallow merge to keep defaults if keys missing
            for k, v in obj.items():
                state[k] = v
            LOG.info("✅ State loaded successfully.")
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


# =============== DATE & TIME HELPERS =================
def norm_date_with_weekday(s: str | None) -> str:
    """Return date string like '09-Nov-2025 (Sunday)'.
       Accepts several formats; if None/empty, use 'now' in IST."""
    if not s:
        now = datetime.now(BOT_TZ)
        return now.strftime("%d-%b-%Y (%A)")
    s = s.strip()
    for fmt in ("%d-%m-%Y", "%d.%m.%Y", "%d/%m/%Y", "%d-%m-%y", "%d.%m.%y", "%d/%m/%y", "%Y-%m-%d"):
        try:
            dt = datetime.strptime(s, fmt)
            return dt.strftime("%d-%b-%Y (%A)")
        except Exception:
            pass
    try:
        dt = datetime.fromisoformat(s)
        return dt.strftime("%d-%b-%Y (%A)")
    except Exception:
        # if unknown, just return raw
        return s


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
    parts = [int(p) for p in re.split(r":", s)]
    if len(parts) == 3:
        h, m, sec = parts
        return h * 60 + m + sec / 60
    if len(parts) == 2:
        m, sec = parts
        return m + sec / 60
    return 0.0


def norm_key(k: str) -> str:
    return re.sub(r"\s+", "", k.strip().lower())


def machine_label(chat_id: int) -> str:
    return MACHINE_MAP.get(str(chat_id)) or MACHINE_MAP.get(str(int(chat_id))) or f"{chat_id}"


def route_summary(context: ContextTypes.DEFAULT_TYPE, text: str):
    if SUMMARY_CHAT_ID:
        context.application.create_task(
            context.bot.send_message(SUMMARY_CHAT_ID, text, parse_mode="Markdown")
        )


def batch_key(chat_id: int, batch: str | int | None) -> str:
    return f"{chat_id}:{batch or ''}"


# =============== EXCEL ZONE RULES =================
def load_zone_rules(path=REPORT_PATH, sheet=ZONE_SHEET) -> Dict[str, float]:
    rules: Dict[str, float] = {}
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
        LOG.warning("⚠️ Failed to load Excel rules (%s). Using safe defaults.", e)
        rules = {
            "50-200 reactor": 60,
            "200-300 reactor": 60,
            "300-400 reactor": 70,
            "400-450 reactor": 80,
            "450-480 reactor": 25,
            "480-500 reactor": 15,
            "300-400 separator": 30,
        }
    return rules


ZONE_RULES = load_zone_rules()


class RecoEngine:
    def __init__(self, report_path: str):
        self.report_path = report_path
        self.zone_defaults: Dict[str, float] = {}
        self._load()

    def _load(self):
        self.zone_defaults = load_zone_rules(self.report_path)

    def plan(self, feed: Dict[str, float]) -> Dict[str, float]:
        # Base plan from defaults
        plan = dict(self.zone_defaults)
        # Light heuristics: more radial/chips → extend 300-400; more nylon → shorten a bit
        total = sum(feed.get(k, 0.0) for k in ("radial", "nylon", "chips", "powder", "kachra", "others"))
        if total > 0:
            radial = feed.get("radial", 0.0) / total
            chips = feed.get("chips", 0.0) / total
            nylon = feed.get("nylon", 0.0) / total
            adj_300_400 = 1.0 + 0.15 * (radial + 0.5 * chips - 0.6 * nylon)
            adj_200_300 = 1.0 + 0.10 * radial - 0.05 * nylon
            for k in list(plan.keys()):
                if k.startswith("300-400"):
                    plan[k] = max(20, plan[k] * adj_300_400)
                elif k.startswith("200-300"):
                    plan[k] = max(15, plan[k] * adj_200_300)
        return plan


engine = RecoEngine(REPORT_PATH)


# =============== PARSERS =================
def parse_feed(text: str) -> Dict[str, float | str]:
    # Accept: Feed: Radial=5.1T, Nylon=0.6T, Chips=3.4T, Powder=1.5T, batch=92, operator=Ravi, date=09-11-2025
    t = re.sub(r"^/?(plan|predict)?\s*feed\s*:|^feed\s*:", "", text, flags=re.I).strip()
    data: Dict[str, float | str] = {}
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
        lk = norm_key(k)

        # text fields
        if lk in ("batch", "operator", "machine", "date"):
            data[lk] = v
            continue

        # numeric units
        vv = v.replace("ton", "T").replace("mt", "T").replace("kg", "K")
        mT = re.match(r"([\d.]+)\s*[tT]$", vv)
        mK = re.match(r"([\d.]+)\s*[kK]$", vv)
        if mT:
            val = float(mT.group(1)) * 1000.0
        elif mK:
            val = float(mK.group(1))
        else:
            val = float(re.sub(r"[^\d.]", "", vv) or 0.0)
        data[lk] = val

    # Normalize names
    mapping = {"radial": "radial", "nylon": "nylon", "chips": "chips", "powder": "powder", "kachra": "kachra", "others": "others"}
    out: Dict[str, float | str] = {}
    for k, v in data.items():
        out[mapping.get(k, k)] = v
    return out


def parse_actual(text: str) -> Dict[str, float | str]:
    # Accept: Actual: 50-200=01:10, 200-300=00:45, 300-400=01:20, 400-450=00:22; oil=46.2; batch=92
    t = re.sub(r"^/?actual\s*:", "", text, flags=re.I).strip()
    out: Dict[str, float | str] = {}
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


# =============== PREDICTION + LEARNING =================
def predict_yield(feed: Dict[str, float | str]) -> Tuple[float, float]:
    total = sum(float(feed.get(k, 0.0)) for k in ("radial", "nylon", "chips", "powder", "kachra", "others"))
    if total <= 0:
        return (0.0, 0.6)
    w = state["weights"]
    base = 0.0
    for k in ("radial", "nylon", "chips", "powder", "kachra", "others"):
        kg = float(feed.get(k, 0.0))
        base += (kg / total) * (w[k] * 100.0)
    pred = max(30.0, min(55.0, base))
    ratios = [float(feed.get(x, 0.0)) / total for x in ("radial", "nylon", "chips", "powder")]
    conf = max(0.55, min(0.95, 0.95 - 0.8 * float(np.std(ratios))))
    return (round(pred, 2), round(conf, 2))


def learn_from_actual(feed: Dict[str, float | str], actual_oil: float):
    total = sum(float(feed.get(k, 0.0)) for k in ("radial", "nylon", "chips", "powder", "kachra", "others"))
    if total <= 0:
        return
    pred, _ = predict_yield(feed)
    err = (actual_oil - pred) / 100.0
    for k in ("radial", "nylon", "chips", "powder", "kachra", "others"):
        share = float(feed.get(k, 0.0)) / total
        state["weights"][k] += 0.01 * err * share
    save_state()


def pretty_plan(plan: Dict[str, float]) -> str:
    def keyer(z):
        m = re.match(r"(\d{2,3})-(\d{2,3})", z)
        return (int(m.group(1)), int(m.group(2))) if m else (999, 999)
    lines = []
    for z in sorted(plan.keys(), key=keyer):
        lines.append(f"{z}: {to_hhmmss(plan[z])}")
    return "\n".join(lines)


# =============== REMINDERS & DAILY SUMMARY =================
async def reminder_ping(context: ContextTypes.DEFAULT_TYPE, chat_id: int, batch: str):
    label = machine_label(chat_id)
    txt = f"⚠️ *Reminder:* Actual data not entered yet for *Batch {batch}* on *{label}*. Please send `Actual:`."
    await context.bot.send_message(chat_id, txt, parse_mode="Markdown")
    route_summary(context, f"🔔 {txt}")


def schedule_reminder(app: Application, chat_id: int, batch: str):
    """First reminder at +12h; then hourly until actual is received."""
    global scheduler
    key = batch_key(chat_id, batch)
    due = datetime.now(BOT_TZ) + timedelta(hours=12)
    state["reminders"][key] = {"chat_id": chat_id, "batch": batch, "due": due.isoformat()}
    save_state()

    async def _reminder_job():
        # runs each hour; only pings if key still pending AND due has passed
        if key not in state["reminders"]:
            return
        due_iso = state["reminders"][key]["due"]
        due_dt = datetime.fromisoformat(due_iso)
        if datetime.now(BOT_TZ) >= due_dt:
            await reminder_ping(app.bot, chat_id, batch)  # type: ignore

    # APScheduler: run every hour (start now; job decides if due)
    scheduler.add_job(lambda: app.create_task(_reminder_job()), "interval", minutes=60, id=f"rem:{key}", replace_existing=True)


async def daily_summary_job(context: ContextTypes.DEFAULT_TYPE):
    now = datetime.now(BOT_TZ)
    title = now.strftime("%d-%b-%Y (%A)")
    lines = [f"📊 *Daily Summary* — {title} (IST)"]
    for cid_str, label in MACHINE_MAP.items():
        cid = int(cid_str)
        lf = state["latest_feed"].get(cid_str)
        status = "🟡 Idle"
        if lf:
            last_feed_ts = datetime.fromisoformat(lf["ts"])
            hrs = (now - last_feed_ts.astimezone(BOT_TZ)).total_seconds() / 3600
            last_act_iso = state["last_actual_ts"].get(cid_str)
            completed = False
            if last_act_iso:
                completed = datetime.fromisoformat(last_act_iso) >= last_feed_ts
            if hrs <= 11 and not completed:
                status = f"🟢 Running (Batch {lf.get('batch','?')})"
            elif completed or hrs > 11:
                status = f"🔵 Completed (Batch {lf.get('batch','?')})"
        lines.append(f"• {label}: {status}")

    msg = "\n".join(lines)
    if SUMMARY_CHAT_ID:
        await context.bot.send_message(SUMMARY_CHAT_ID, msg, parse_mode="Markdown")


def schedule_daily_summary(app: Application):
    global scheduler
    if scheduler is None:
        scheduler = BackgroundScheduler(timezone=str(BOT_TZ))
        scheduler.start()

    # 21:35 IST every day
    scheduler.add_job(
        lambda: app.create_task(daily_summary_job(app)),  # type: ignore
        "cron", hour=21, minute=35, id="daily_summary", replace_existing=True
    )
    LOG.info("🕤 Daily summary scheduled at 9:35 PM IST.")


# =============== TELEGRAM HANDLERS =================
HELP_TEXT = (
    "*Commands & Examples*\n"
    "• Send a *Feed* (works in machine groups):\n"
    "`Feed: Radial=5.10T, Nylon=0.60T, Chips=3.40T, Powder=1.50T, batch=92, operator=Inglesh, date=09-11-2025`\n"
    "→ Plan (minutes) + Predicted Oil% (with confidence) + reminder\n\n"
    "• Log *Actual* (same machine group):\n"
    "`Actual: 50-200=01:10, 200-300=00:45, 300-400=01:20, 400-450=00:22; oil=46.2; batch=92`\n"
    "→ Compares vs plan, learns, clears reminder\n\n"
    "• `/status` – status of this machine (or all machines in Summary group)\n"
    "• `/reload` – reload Excel rules\n"
    "• `/id` – show chat id & label\n"
    "• `/help` – this help\n"
)

async def cmd_start(update: Update, context: ContextTypes.DEFAULT_TYPE):
    await update.message.reply_text("PyroVision Assistant ready ✅\n\n" + HELP_TEXT, parse_mode="Markdown")

async def cmd_help(update: Update, context: ContextTypes.DEFAULT_TYPE):
    await update.message.reply_text(HELP_TEXT, parse_mode="Markdown")

async def cmd_id(update: Update, context: ContextTypes.DEFAULT_TYPE):
    cid = update.effective_chat.id
    label = machine_label(cid)
    await update.message.reply_text(f"chat_id: `{cid}`\nname/title: *{label}*", parse_mode="Markdown")

async def cmd_reload(update: Update, context: ContextTypes.DEFAULT_TYPE):
    engine._load()
    await update.message.reply_text("🔁 Reloaded recommendations from Excel.")
    route_summary(context, f"🔁 Rules reloaded by *{machine_label(update.effective_chat.id)}*")

def remember_feed(chat_id: int, feed: Dict[str, float | str], plan: Dict[str, float]):
    now = datetime.now(BOT_TZ)
    entry = {
        "ts": now.isoformat(),
        "batch": str(feed.get("batch", "")),
        "operator": str(feed.get("operator", "")),
        "date_str": norm_date_with_weekday(str(feed.get("date") or "")),
        "feed": feed, "plan": plan
    }
    state["latest_feed"][str(chat_id)] = entry
    save_state()
    if entry["batch"]:
        schedule_reminder(context_application, chat_id, entry["batch"])  # scheduled after app exists

async def plan_core(update: Update, context: ContextTypes.DEFAULT_TYPE, feed_text: str):
    chat_id = update.effective_chat.id
    label = machine_label(chat_id)
    feed = parse_feed(feed_text)
    plan = engine.plan(feed)
    pred, conf = predict_yield(feed)
    date_str = norm_date_with_weekday(feed.get("date") if isinstance(feed.get("date"), str) else None)
    batch = feed.get("batch", "?")
    oper = feed.get("operator", "?")

    remember_feed(chat_id, feed, plan)

    msg = []
    msg.append(f"📒 *{label}* — *Batch* {batch}, *Operator* {oper}")
    msg.append(f"• *Date* {date_str}")
    msg.append(
        "🛢️ Feed: "
        f"Radial {float(feed.get('radial',0))/1000:.2f}T, "
        f"Nylon {float(feed.get('nylon',0))/1000:.2f}T, "
        f"Chips {float(feed.get('chips',0))/1000:.2f}T, "
        f"Powder {float(feed.get('powder',0))/1000:.2f}T"
    )
    msg.append(f"📈 Predicted Oil: *{pred:.2f}%*  (confidence {conf:.2f})")
    msg.append("\n*Recommended zone minutes (plan):*\n" + pretty_plan(plan))

    text = "\n".join(msg)
    await update.message.reply_text(text, parse_mode="Markdown")
    route_summary(context, text)

async def cmd_plan(update: Update, context: ContextTypes.DEFAULT_TYPE):
    await plan_core(update, context, update.message.text)

async def cmd_actual(update: Update, context: ContextTypes.DEFAULT_TYPE):
    chat_id = update.effective_chat.id
    lf = state["latest_feed"].get(str(chat_id))
    if not lf:
        await update.message.reply_text("I don’t find a recorded Feed for this machine. Send the Feed first.")
        return

    actual = parse_actual(update.message.text)
    plan = lf.get("plan", {})
    deltas = {}
    tips = []
    for z, pmin in plan.items():
        zkey = z.replace(" ", "")
        if zkey in actual:
            am = float(actual[zkey])
            deltas[z] = am - pmin
            if abs(deltas[z]) >= 5:
                tips.append(("reduce" if deltas[z] > 0 else "increase", z, abs(deltas[z])))

    if "oil" in actual:
        learn_from_actual(lf.get("feed", {}), float(actual["oil"]))
        state["last_actual_ts"][str(chat_id)] = datetime.now(BOT_TZ).isoformat()
        # clear reminder
        key = batch_key(chat_id, lf.get("batch"))
        if key in state["reminders"]:
            del state["reminders"][key]
        save_state()

    lines = ["📊 Deviation vs plan (min):"]
    for z in sorted(deltas.keys(), key=lambda s: int(s.split("-")[0])):
        lines.append(f"{z}: {deltas[z]:+.0f}")
    if tips:
        lines.append("\n🛠️ Recommendations:")
        for d, z, mins in tips:
            lines.append(f"• {d} {z} by ~{mins:.0f} min")
    else:
        lines.append("\n✅ Near-optimal execution vs plan.")
    await update.message.reply_text("\n".join(lines))

    route_summary(context, f"🧾 Actual logged for *{machine_label(chat_id)}* (batch {lf.get('batch','?')}).")

async def cmd_status(update: Update, context: ContextTypes.DEFAULT_TYPE):
    now = datetime.now(BOT_TZ)
    if SUMMARY_CHAT_ID and update.effective_chat.id == SUMMARY_CHAT_ID:
        lines = ["🟢 *Machine Status*"]
        for cid_str, label in MACHINE_MAP.items():
            lf = state["latest_feed"].get(cid_str)
            st = "🟡 Idle"
            if lf:
                last_feed_ts = datetime.fromisoformat(lf["ts"])
                hrs = (now - last_feed_ts.astimezone(BOT_TZ)).total_seconds() / 3600
                last_act_iso = state["last_actual_ts"].get(cid_str)
                completed = last_act_iso and (datetime.fromisoformat(last_act_iso) >= last_feed_ts)
                if hrs <= 11 and not completed:
                    st = f"🟢 Running (Batch {lf.get('batch','?')})"
                elif completed or hrs > 11:
                    st = f"🔵 Completed (Batch {lf.get('batch','?')})"
            lines.append(f"• {label}: {st}")
        await update.message.reply_text("\n".join(lines), parse_mode="Markdown")
    else:
        cid = str(update.effective_chat.id)
        label = machine_label(update.effective_chat.id)
        lf = state["latest_feed"].get(cid)
        st = "🟡 Idle"
        if lf:
            last_feed_ts = datetime.fromisoformat(lf["ts"])
            hrs = (now - last_feed_ts.astimezone(BOT_TZ)).total_seconds() / 3600
            last_act_iso = state["last_actual_ts"].get(cid)
            completed = last_act_iso and (datetime.fromisoformat(last_act_iso) >= last_feed_ts)
            if hrs <= 11 and not completed:
                st = f"🟢 Running (Batch {lf.get('batch','?')})"
            elif completed or hrs > 11:
                st = f"🔵 Completed (Batch {lf.get('batch','?')})"
        await update.message.reply_text(f"{label}: {st}")

async def text_router(update: Update, context: ContextTypes.DEFAULT_TYPE):
    text = (update.message.text or "").strip()
    if re.match(r"(?i)^feed\s*:", text):
        await plan_core(update, context, text)
    elif re.match(r"(?i)^actual\s*:", text):
        await cmd_actual(update, context)
    elif re.match(r"(?i)^status$", text):
        await cmd_status(update, context)
    else:
        await update.message.reply_text("I didn’t understand. Type /help for examples.")


# =============== MAIN =================
context_application: Application  # will be set in main()

def main():
    global context_application, scheduler
    LOG.info("🚀 Bot starting…")
    load_state()

    if not TELEGRAM_BOT_TOKEN:
        raise RuntimeError("❌ TELEGRAM_BOT_TOKEN missing!")

    app = Application.builder().token(TELEGRAM_BOT_TOKEN).build()
    context_application = app

    # Handlers
    app.add_handler(CommandHandler("start", cmd_start))
    app.add_handler(CommandHandler("help", cmd_help))
    app.add_handler(CommandHandler("id", cmd_id))
    app.add_handler(CommandHandler("plan", cmd_plan))
    app.add_handler(CommandHandler("actual", cmd_actual))
    app.add_handler(CommandHandler("reload", cmd_reload))
    app.add_handler(CommandHandler("status", cmd_status))
    app.add_handler(MessageHandler(filters.TEXT & ~filters.COMMAND, text_router))

    # Scheduler for daily summary & reminders
    schedule_daily_summary(app)

    LOG.info("✅ Bot ready. Polling…")
    app.run_polling()


if __name__ == "__main__":
    main()
