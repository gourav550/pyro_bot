# ======================================================
#  PyroVision Assistant – Railway / Telegram (PTB v21)
# ======================================================

import os, json, re, io, math, logging
from datetime import datetime, timedelta, time
from zoneinfo import ZoneInfo

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from telegram import Update, InputFile
from telegram.constants import ParseMode
from telegram.ext import (
    Application, CommandHandler, MessageHandler,
    ContextTypes, filters
)

LOG = logging.getLogger("pyro_bot")
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)s | %(name)s | %(message)s"
)

# ---------- ENV ----------
TOKEN = os.environ.get("TELEGRAM_BOT_TOKEN", "").strip()
REPORT_PATH = os.environ.get("REPORT_PATH", "pyrolysis_feed_temp_ZONE_TIME_report.xlsx")
ZONE_SHEET = os.environ.get("ZONE_SHEET", "ZoneTime_Recommendations")
SUMMARY_CHAT_ID = os.environ.get("SUMMARY_CHAT_ID", "").strip() or None  # e.g. "-4865673071"
BOT_TZ = ZoneInfo(os.environ.get("TIMEZONE", "Asia/Kolkata"))

# MACHINE_MAP is a JSON string: {"-4923260341":"Machine 1296 (R-1)", ...}
try:
    MACHINE_MAP = json.loads(os.environ.get("MACHINE_MAP", "{}"))
except Exception:
    MACHINE_MAP = {}

STATE_FILE = "bot_state.json"

# ---------- RUNTIME STATE ----------
# Weights roughly reflect base oil % contribution by mix; they will learn from /actual.
state = {
    "weights": {"radial": 0.45, "nylon": 0.42, "chips": 0.46, "powder": 0.53, "kachra": 0.40, "others": 0.40},
    "latest_feed": {},      # chat_id(str) -> {ts, date_str, batch, operator, feed, plan}
    "last_actual_ts": {},   # chat_id(str) -> iso
    "reminders": {}         # key(chat_id:batch) -> True
}

# ---------- PERSIST ----------
def load_state():
    global state
    try:
        if os.path.exists(STATE_FILE):
            with open(STATE_FILE, "r", encoding="utf-8") as f:
                obj = json.load(f)
            for k, v in obj.items():
                state[k] = v
            LOG.info("ℹ️ State loaded.")
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

# ---------- UTIL ----------
def to_hhmmss(minutes: float | int | None) -> str:
    if minutes is None:
        return "-"
    try:
        if isinstance(minutes, float) and math.isnan(minutes):
            return "-"
        total = int(round(float(minutes) * 60))
        h, rem = divmod(total, 3600)
        m, s = divmod(rem, 60)
        return f"{h:02d}:{m:02d}:{s:02d}"
    except Exception:
        return "-"

def hhmmss_to_minutes(s: str) -> float:
    s = (s or "").strip()
    if ":" not in s:
        return float(s or 0.0)
    parts = [int(x) for x in s.split(":")]
    if len(parts) == 3:
        h, m, sec = parts
        return h * 60 + m + sec / 60
    if len(parts) == 2:
        m, sec = parts
        return m + sec / 60
    return 0.0

def norm_key(s: str) -> str:
    return re.sub(r"\s+", " ", s.strip().lower())

def machine_label(chat_id: int) -> str:
    return MACHINE_MAP.get(str(chat_id)) or f"{chat_id}"

def _norm_date(s: str | None) -> str:
    """
    Normalize various date formats and include weekday.
    Fallback: now in BOT_TZ.
    """
    if not s:
        now = datetime.now(BOT_TZ)
        return now.strftime("%d-%b-%Y (%A)")
    s = s.strip()
    fmts = ("%d-%m-%Y", "%d.%m.%Y", "%d/%m/%Y", "%d-%m-%y", "%d.%m.%y", "%d/%m/%y",
            "%Y-%m-%d", "%Y/%m/%d")
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
        return dt.strftime("%d-%b-%Y (%A)")
    except Exception:
        return s

def batch_key(chat_id: int, batch: str | int | None) -> str:
    return f"{chat_id}:{batch or ''}"

def route_summary(context: ContextTypes.DEFAULT_TYPE, text: str):
    if SUMMARY_CHAT_ID:
        try:
            context.bot.send_message(chat_id=int(SUMMARY_CHAT_ID), text=text, parse_mode=ParseMode.MARKDOWN)
        except Exception:
            pass

# ---------- RULES (Excel) ----------
def load_zone_rules(path=REPORT_PATH, sheet=ZONE_SHEET) -> dict[str, float]:
    rules = {}
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
        LOG.info(f"Loaded {len(rules)} zone rules from Excel.")
    except Exception as e:
        LOG.warning(f"⚠️ Failed to load Excel rules ({e}); using defaults.")
        rules = {
            "50-200 reactor": 165,   # minutes defaults similar to your operations
            "200-300 reactor": 68,
            "300-400 reactor": 185,
            "300-400 separator": 175,
            "400-450 reactor": 75,
            "450-480 reactor": 20,
            "480-500 reactor": 0,
            "500-520 reactor": 0
        }
    return rules

ZONE_RULES = load_zone_rules()

class RecoEngine:
    def __init__(self, path: str):
        self.path = path
        self.zone_defaults = dict(ZONE_RULES)

    def reload(self):
        self.zone_defaults = load_zone_rules(self.path)

    def plan(self, feed: dict) -> dict[str, float]:
        plan = dict(self.zone_defaults)
        total = sum(feed.get(k, 0.0) for k in ("radial", "nylon", "chips", "powder", "kachra", "others"))
        if total > 0:
            r = feed.get("radial", 0.0) / total
            c = feed.get("chips", 0.0) / total
            n = feed.get("nylon", 0.0) / total
            adj = 1.0 + 0.15 * (r + 0.5 * c - 0.6 * n)
            for k in list(plan.keys()):
                if k.startswith("300-400"):
                    plan[k] = max(20, plan[k] * adj)
                if k.startswith("200-300"):
                    plan[k] = max(15, plan[k] * (1.0 + 0.1 * r - 0.05 * n))
        return plan

engine = RecoEngine(REPORT_PATH)

# ---------- PREDICT + LEARN ----------
def predict_yield(feed: dict) -> tuple[float, float]:
    total = sum(feed.get(k, 0.0) for k in ("radial", "nylon", "chips", "powder", "kachra", "others"))
    if total <= 0:
        return (0.0, 0.55)
    base = 0.0
    for k, kg in feed.items():
        if k in state["weights"]:
            base += (kg / total) * (state["weights"][k] * 100.0)
    pred = max(30.0, min(55.0, base))
    ratio_spread = np.std([feed.get(x, 0.0) / total for x in ("radial", "nylon", "chips", "powder")])
    conf = max(0.55, min(0.95, 0.95 - 0.8 * ratio_spread))
    return (round(pred, 2), round(conf, 2))

def learn_from_actual(feed: dict, actual_oil: float):
    total = sum(feed.get(k, 0.0) for k in ("radial", "nylon", "chips", "powder", "kachra", "others"))
    if total <= 0:
        return
    pred, _ = predict_yield(feed)
    err = (actual_oil - pred) / 100.0
    for k in ("radial", "nylon", "chips", "powder", "kachra", "others"):
        share = feed.get(k, 0.0) / total
        state["weights"][k] += 0.01 * err * share
    save_state()

# ---------- PARSERS ----------
def parse_feed(text: str) -> dict:
    # Accept: "Feed: Radial=5.1T, Nylon=0.6T, Chips=3.4T, Powder=1.5T, batch=92, operator=Ravi, date=09-11-2025"
    data = {}
    t = re.sub(r"^/?(plan|predict)?\s*feed\s*:\s*|^feed\s*:\s*", "", text, flags=re.I).strip()
    for part in re.split(r"[,\n;]+", t):
        if not part.strip():
            continue
        if "=" in part:
            k, v = [x.strip() for x in part.split("=", 1)]
        else:
            m = re.match(r"([A-Za-z]+)\s+([\d\.]+)", part.strip())
            if not m:
                continue
            k, v = m.group(1), m.group(2)
        key = norm_key(k)
        if key in ("batch", "operator", "machine", "date"):
            data[key] = v
            continue
        v2 = v.replace("ton", "T").replace("mt", "T").replace("kg", "K")
        mT = re.match(r"([\d\.]+)\s*[tT]$", v2)
        mK = re.match(r"([\d\.]+)\s*[kK][gG]?$", v2)
        if mT:
            val = float(mT.group(1)) * 1000.0
        elif mK:
            val = float(mK.group(1))
        else:
            val = float(re.sub(r"[^\d\.]", "", v2) or 0.0)
        data[key] = val

    mapping = {"radial": "radial", "nylon": "nylon", "chips": "chips", "powder": "powder",
               "kachra": "kachra", "others": "others"}
    out = {mapping.get(k, k): v for k, v in data.items()}
    return out

def parse_actual(text: str) -> dict:
    # "Actual: 50-200=02:57, 200-300=01:06, 300-400=03:18, 400-450=01:10, 450-480=00:32, oil=38.27, batch=66"
    t = re.sub(r"^/?actual\s*:\s*", "", text, flags=re.I).strip()
    out = {}
    for chunk in re.split(r"[;,]+", t):
        if "=" not in chunk:
            continue
        k, v = [x.strip() for x in chunk.split("=", 1)]
        lk = norm_key(k)
        if lk in ("oil", "oil%"):
            out["oil"] = float(re.sub(r"[^\d\.]", "", v) or 0.0)
        elif re.match(r"\d{2,3}\s*-\s*\d{2,3}", lk):
            zkey = lk.replace(" ", "")
            out[zkey] = hhmmss_to_minutes(v)
        elif lk == "batch":
            out["batch"] = v
    return out

# ---------- RENDER ----------
def pretty_plan(plan: dict[str, float]) -> str:
    def k(x):
        m = re.match(r"(\d{2,3})-(\d{2,3})", x)
        if not m:
            return (999, 999, x)
        return (int(m.group(1)), int(m.group(2)), x)
    lines = []
    for z in sorted(plan.keys(), key=k):
        lines.append(f"{z}: {to_hhmmss(plan[z])}")
    return "\n".join(lines)

def chart_plan_vs_actual(plan: dict[str, float], actual_map: dict[str, float] | None) -> io.BytesIO:
    zones = [z for z in sorted(plan.keys(), key=lambda s: int(s.split("-")[0]))]
    pmins = [plan[z] for z in zones]
    amins = [actual_map.get(z.replace(" ", ""), np.nan) if actual_map else np.nan for z in zones]
    x = np.arange(len(zones))
    width = 0.38
    fig, ax = plt.subplots(figsize=(9.5, 3.2), dpi=180)
    ax.bar(x - width/2, pmins, width, label="Plan (min)")
    if actual_map:
        ax.bar(x + width/2, amins, width, label="Actual (min)")
    ax.set_xticks(x, zones, rotation=22, ha="right")
    ax.set_ylabel("Minutes")
    ax.legend()
    ax.grid(True, axis="y", alpha=0.25)
    buf = io.BytesIO()
    plt.tight_layout()
    plt.savefig(buf, format="png")
    plt.close(fig)
    buf.seek(0)
    return buf

# ---------- JOBS ----------
async def daily_summary_job(context: ContextTypes.DEFAULT_TYPE):
    now = datetime.now(BOT_TZ)
    lines = [f"📊 *Daily Summary — {now.strftime('%d-%b-%Y (%A)')}*  _IST_"]
    for cid_str, label in MACHINE_MAP.items():
        lf = state["latest_feed"].get(cid_str)
        st = "Idle"
        if lf:
            feed_ts = datetime.fromisoformat(lf["ts"])
            hrs = (now - feed_ts.astimezone(BOT_TZ)).total_seconds() / 3600
            completed = False
            last_iso = state["last_actual_ts"].get(cid_str)
            if last_iso:
                completed = datetime.fromisoformat(last_iso) >= feed_ts
            if hrs <= 12 and not completed:
                st = f"Running (batch {lf.get('batch','?')})"
            else:
                dstr = _norm_date(lf.get("date_str"))
                st = f"Completed — last batch {lf.get('batch','?')} on {dstr}"
        lines.append(f"• {label}: {st}")
    msg = "\n".join(lines)
    if SUMMARY_CHAT_ID:
        await context.bot.send_message(int(SUMMARY_CHAT_ID), msg, parse_mode=ParseMode.MARKDOWN)

async def reminder_job(context: ContextTypes.DEFAULT_TYPE):
    data = context.job.data or {}
    chat_id = data.get("chat_id")
    batch = data.get("batch")
    key = batch_key(chat_id, batch)
    if key not in state["reminders"]:
        # already cleared – stop future repeats
        context.job.schedule_removal()
        return
    label = machine_label(chat_id)
    txt = (f"⚠️ *Reminder:* Actual data not entered yet for *Batch {batch}* on *{label}*.\n"
           f"Please send `Actual: 50-200=hh:mm, 200-300=..., ..., oil=xx.x; batch={batch}`")
    await context.bot.send_message(chat_id, txt, parse_mode=ParseMode.MARKDOWN)
    if SUMMARY_CHAT_ID:
        await context.bot.send_message(int(SUMMARY_CHAT_ID), f"🔔 {txt}", parse_mode=ParseMode.MARKDOWN)

def schedule_daily_summary(app: Application):
    # 21:35 IST daily
    app.job_queue.run_daily(
        daily_summary_job,
        time=time(hour=21, minute=35, tzinfo=BOT_TZ),
        name="daily_summary"
    )

def schedule_reminder(app: Application, chat_id: int, batch: str):
    name = f"reminder:{chat_id}:{batch}"
    # remove any existing with same name
    for j in app.job_queue.get_jobs_by_name(name):
        j.schedule_removal()
    # start after 12h, then hourly
    app.job_queue.run_repeating(
        reminder_job,
        interval=3600,
        first=timedelta(hours=12),
        name=name,
        data={"chat_id": chat_id, "batch": batch},
    )

# ---------- COMMANDS & HANDLERS ----------
HELP = (
    "*Commands*\n"
    "• Send `Feed: Radial=5.10T, Nylon=0.60T, Chips=3.40T, Powder=1.50T, batch=66, operator=Ravi, date=09-11-2025`.\n"
    "  → Plan + Predicted oil (+ fan-out to Summary).\n"
    "• `Actual: 50-200=02:57, 200-300=01:06, 300-400=03:18, 400-450=01:10, 450-480=00:32, oil=38.2, batch=66`.\n"
    "  → Deviations + tips + bar chart (+ learns & clears reminder).\n"
    "• `/status` (in any group) → Machine status; in Summary it lists all.\n"
    "• `/reload` → Reloads Excel rules.\n"
    "• `/id` → Prints this chat’s id and label.\n"
)

async def cmd_start(update: Update, context: ContextTypes.DEFAULT_TYPE):
    await update.message.reply_text("PyroVision Assistant ready ✅\n\n" + HELP, parse_mode=ParseMode.MARKDOWN)

async def cmd_help(update: Update, context: ContextTypes.DEFAULT_TYPE):
    await update.message.reply_text(HELP, parse_mode=ParseMode.MARKDOWN)

async def cmd_id(update: Update, context: ContextTypes.DEFAULT_TYPE):
    cid = update.effective_chat.id
    await update.message.reply_text(f"chat_id: `{cid}`\nname/title: *{machine_label(cid)}*", parse_mode=ParseMode.MARKDOWN)

async def cmd_reload(update: Update, context: ContextTypes.DEFAULT_TYPE):
    engine.reload()
    await update.message.reply_text("🔁 Reloaded recommendations from Excel.")
    route_summary(context, f"🔁 Rules reloaded by *{machine_label(update.effective_chat.id)}*")

def remember_feed(app: Application, chat_id: int, feed: dict, plan: dict):
    now = datetime.now(BOT_TZ)
    entry = {
        "ts": now.isoformat(),
        "date_str": _norm_date(feed.get("date")),
        "batch": str(feed.get("batch", "")),
        "operator": str(feed.get("operator", "")),
        "feed": feed,
        "plan": plan
    }
    state["latest_feed"][str(chat_id)] = entry
    save_state()
    # schedule reminder loop if batch present
    if entry["batch"]:
        key = batch_key(chat_id, entry["batch"])
        state["reminders"][key] = True
        save_state()
        schedule_reminder(app, chat_id, entry["batch"])

async def plan_core(update: Update, context: ContextTypes.DEFAULT_TYPE, feed_text: str):
    chat_id = update.effective_chat.id
    feed = parse_feed(feed_text)
    plan = engine.plan(feed)
    pred, conf = predict_yield(feed)
    label = machine_label(chat_id)
    batch = feed.get("batch", "?")
    oper = feed.get("operator", "?")
    dstr = _norm_date(feed.get("date"))

    remember_feed(context.application, chat_id, feed, plan)

    msg = []
    msg.append(f"📒 *{label}* — *Batch* {batch}, *Operator* {oper}")
    msg.append(f"• *Date* {dstr}")
    msg.append(f"🛢️ Feed: Radial {feed.get('radial',0)/1000:.2f}T, Nylon {feed.get('nylon',0)/1000:.2f}T, "
               f"Chips {feed.get('chips',0)/1000:.2f}T, Powder {feed.get('powder',0)/1000:.2f}T")
    msg.append(f"📈 Predicted Oil: *{pred:.2f}%*  (confidence {conf:.2f})")
    msg.append("\n*Recommended zone minutes (plan):*\n" + pretty_plan(plan))
    text = "\n".join(msg)

    await update.message.reply_text(text, parse_mode=ParseMode.MARKDOWN)
    route_summary(context, text)

async def cmd_plan(update: Update, context: ContextTypes.DEFAULT_TYPE):
    await plan_core(update, context, update.message.text)

async def cmd_actual(update: Update, context: ContextTypes.DEFAULT_TYPE):
    chat_id = update.effective_chat.id
    actual = parse_actual(update.message.text)
    lf = state["latest_feed"].get(str(chat_id))
    if not lf:
        await update.message.reply_text("I don't find a recorded Feed for this machine. Send the Feed first.")
        return
    plan = lf.get("plan", {})

    deltas, tips = {}, []
    for z, pmin in plan.items():
        zkey = z.replace(" ", "")
        if zkey in actual:
            am = actual[zkey]
            deltas[z] = am - pmin
            if abs(deltas[z]) >= 5:
                tips.append(("reduce" if deltas[z] > 0 else "increase", z, abs(deltas[z])))

    # learning & reminder clear
    if "oil" in actual:
        learn_from_actual(lf.get("feed", {}), float(actual["oil"]))
        state["last_actual_ts"][str(chat_id)] = datetime.now(BOT_TZ).isoformat()
        key = batch_key(chat_id, lf.get("batch"))
        if key in state["reminders"]:
            del state["reminders"][key]
        # cancel repeating job for this batch
        for j in context.application.job_queue.get_jobs_by_name(f"reminder:{chat_id}:{lf.get('batch')}"):
            j.schedule_removal()
        save_state()

    # reply deviations
    label = machine_label(chat_id)
    lines = [f"📎 *{label}*  •  {lf.get('date_str','')}  |  *Batch* {lf.get('batch','?')}  |  *Operator* {lf.get('operator','?')}",
             "📊 *Deviation vs plan (min):*"]
    for z in sorted(deltas.keys(), key=lambda s: int(s.split("-")[0])):
        lines.append(f"{z}: {deltas[z]:+.0f}")
    if tips:
        lines.append("\n🛠️ *Recommendations:*")
        for d, z, mins in tips:
            lines.append(f"• {d} {z} by ~{mins:.0f} min")
    else:
        lines.append("\n✅ Near-optimal execution vs plan.")
    text = "\n".join(lines)
    await update.message.reply_text(text, parse_mode=ParseMode.MARKDOWN)

    # bar chart (plan vs actual)
    buf = chart_plan_vs_actual(plan, {k: v for k, v in actual.items() if "-" in k})
    await context.bot.send_photo(chat_id, photo=InputFile(buf, filename="plan_vs_actual.png"))

    # summary notice
    if SUMMARY_CHAT_ID:
        await context.bot.send_message(int(SUMMARY_CHAT_ID), f"🧾 Actual logged for *{label}* (batch {lf.get('batch','?')}).",
                                       parse_mode=ParseMode.MARKDOWN)

async def cmd_status(update: Update, context: ContextTypes.DEFAULT_TYPE):
    now = datetime.now(BOT_TZ)
    if SUMMARY_CHAT_ID and str(update.effective_chat.id) == str(SUMMARY_CHAT_ID):
        lines = ["🟢 *Machine Status*"]
        for cid_str, label in MACHINE_MAP.items():
            lf = state["latest_feed"].get(cid_str)
            st = "Idle"
            if lf:
                feed_ts = datetime.fromisoformat(lf["ts"])
                hrs = (now - feed_ts.astimezone(BOT_TZ)).total_seconds() / 3600
                last_iso = state["last_actual_ts"].get(cid_str)
                completed = last_iso and (datetime.fromisoformat(last_iso) >= feed_ts)
                if hrs <= 12 and not completed:
                    st = f"Running (batch {lf.get('batch','?')})"
                else:
                    dstr = _norm_date(lf.get("date_str"))
                    st = f"Completed — last batch {lf.get('batch','?')} on {dstr}"
            lines.append(f"• {label}: {st}")
        await update.message.reply_text("\n".join(lines), parse_mode=ParseMode.MARKDOWN)
    else:
        cid_str = str(update.effective_chat.id)
        label = machine_label(update.effective_chat.id)
        lf = state["latest_feed"].get(cid_str)
        st = "Idle"
        if lf:
            feed_ts = datetime.fromisoformat(lf["ts"])
            hrs = (now - feed_ts.astimezone(BOT_TZ)).total_seconds() / 3600
            last_iso = state["last_actual_ts"].get(cid_str)
            completed = last_iso and (datetime.fromisoformat(last_iso) >= feed_ts)
            if hrs <= 12 and not completed:
                st = f"Running (batch {lf.get('batch','?')})"
            else:
                dstr = _norm_date(lf.get("date_str"))
                st = f"Completed — last batch {lf.get('batch','?')} on {dstr}"
        await update.message.reply_text(f"{label}: {st}", parse_mode=ParseMode.MARKDOWN)

async def text_router(update: Update, context: ContextTypes.DEFAULT_TYPE):
    text = (update.message.text or "").strip()
    if re.match(r"(?i)^feed\s*:", text):
        await plan_core(update, context, text)
    elif re.match(r"(?i)^actual\s*:", text):
        await cmd_actual(update, context)
    elif re.match(r"(?i)^status$", text):
        await cmd_status(update, context)
    else:
        await update.message.reply_text("I didn’t understand.\n\n" + HELP, parse_mode=ParseMode.MARKDOWN)

# ---------- MAIN ----------
def main():
    if not TOKEN:
        raise RuntimeError("❌ Set TELEGRAM_BOT_TOKEN env var.")
    load_state()

    app = Application.builder().token(TOKEN).build()

    # Handlers
    app.add_handler(CommandHandler("start", cmd_start))
    app.add_handler(CommandHandler("help", cmd_help))
    app.add_handler(CommandHandler("id", cmd_id))
    app.add_handler(CommandHandler("reload", cmd_reload))
    app.add_handler(CommandHandler("plan", cmd_plan))
    app.add_handler(CommandHandler("actual", cmd_actual))
    app.add_handler(CommandHandler("status", cmd_status))
    app.add_handler(MessageHandler(filters.TEXT & ~filters.COMMAND, text_router))

    # Jobs
    schedule_daily_summary(app)

    LOG.info("✅ Bot ready. Polling…")
    app.run_polling()

if __name__ == "__main__":
    main()
