# ======================================================
#  PyroVision Assistant — Telegram + Excel + Scheduler
#  - PTB v21.x (async)
#  - JobQueue used for reminders & daily summary
#  - Excel-driven zone rules + simple yield predictor
# ======================================================

from __future__ import annotations

import os, json, re, io, math, logging
from datetime import datetime, timedelta, time
from zoneinfo import ZoneInfo

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from telegram import Update, InputFile
from telegram.ext import (
    Application, CommandHandler, MessageHandler, CallbackContext,
    ContextTypes, filters
)

LOG = logging.getLogger("pyro_bot")
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)s | %(name)s | %(message)s"
)

# ---------- Config (env vars with sensible fallbacks) ----------
TELEGRAM_BOT_TOKEN = os.environ.get("TELEGRAM_BOT_TOKEN", "").strip()

BOT_TZ = ZoneInfo("Asia/Kolkata")  # Kolkata IST

REPORT_PATH = os.environ.get("REPORT_PATH", "pyrolysis_feed_temp_ZONE_TIME_report.xlsx")
ZONE_SHEET  = os.environ.get("ZONE_SHEET", "ZoneTime_Recommendations")

SUMMARY_CHAT_ID = int(os.environ.get("SUMMARY_CHAT_ID", "-4865673071") or 0)

# MACHINE_MAP env should be JSON: {"-4923260341": "Machine 1296 (R-1)", ...}
_default_machine_map = {
    "-4923260341": "Machine 1296 (R-1)",
    "-4859362706": "Machine 1297 (R-2)",
    "-4921356716": "Machine 1298 (R-3)",
    "-4628429766": "Machine 1299 (R-4)",
}
try:
    MACHINE_MAP: dict[str, str] = json.loads(os.environ.get("MACHINE_MAP", "{}")) or _default_machine_map
except Exception:
    MACHINE_MAP = _default_machine_map

# ---------- Persistent state ----------
STATE_FILE = "bot_state.json"
state = {
    "weights":   {"radial": .47, "nylon": .42, "chips": .46, "powder": .53, "kachra": .40, "others": .40},
    "latest_feed": {},      # chat_id -> {ts, batch, operator, date_str, feed, plan}
    "last_actual_ts": {},   # chat_id -> iso
    "reminders": {}         # "<chat_id>:<batch>" -> {"chat_id": int, "batch": str}
}

def load_state():
    global state
    try:
        if os.path.exists(STATE_FILE):
            with open(STATE_FILE, "r", encoding="utf-8") as f:
                obj = json.load(f)
            # shallow merge to keep new defaults
            for k, v in obj.items():
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

# ---------- Utilities ----------
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
    parts = [int(x) for x in s.split(":")]
    if len(parts) == 3:
        h, m, sec = parts
        return h * 60 + m + sec / 60.0
    if len(parts) == 2:
        m, sec = parts
        return m + sec / 60.0
    return 0.0

def norm_key(s: str) -> str:
    return re.sub(r"\s+", " ", (s or "").strip().lower())

def _norm_date(s: str | None) -> str:
    """Normalize date strings to 'DD-Mon-YYYY (Weekday)' in IST."""
    if not s:
        now = datetime.now(BOT_TZ)
        return now.strftime("%d-%b-%Y (%A)")
    s2 = s.strip()
    fmts = ("%d-%m-%Y", "%d.%m.%Y", "%d/%m/%Y", "%d-%m-%y", "%d.%m.%y", "%d/%m/%y",
            "%Y-%m-%d", "%d %b %Y", "%d-%b-%Y")
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
        return s2

def machine_label(chat_id: int) -> str:
    return MACHINE_MAP.get(str(chat_id)) or str(chat_id)

def batch_key(chat_id: int, batch: str | int | None) -> str:
    return f"{chat_id}:{batch or ''}"

# ---------- Excel-driven zone rules ----------
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
        LOG.warning("Excel load failed (%s). Using conservative defaults.", e)
        rules = {
            "50-200 reactor": 60,
            "200-300 reactor": 60,
            "300-400 reactor": 75,
            "400-450 reactor": 70,
            "450-480 reactor": 25,
            "480-500 reactor": 15,
            "300-400 separator": 30,
        }
    return rules

ZONE_RULES = load_zone_rules()

# ---------- Parsers ----------
def parse_feed(text: str) -> dict:
    """
    Accepts:
      Feed: Radial=5.1T, Nylon=0.6T, Chips=3.4T, Powder=1.5T, batch=92, operator=Ravi, date=09-11-2025
    Returns kg numbers and metadata keys (batch, operator, date_str).
    """
    data: dict = {}
    body = re.sub(r"^/?(plan|predict|whatif)?\s*feed\s*:\s*|^feed\s*:\s*", "", text, flags=re.I).strip()
    for part in re.split(r"[,\n;]+", body):
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
        if lk in ("batch", "operator", "machine", "date"):
            data[lk] = v
            continue
        v2 = v.replace("ton", "T").replace("mt", "T").replace("kg", "K")
        mT = re.match(r"([\d.]+)\s*[tT]$", v2)
        mK = re.match(r"([\d.]+)\s*[kK]$", v2)
        if mT:
            val = float(mT.group(1)) * 1000.0
        elif mK:
            val = float(mK.group(1))
        else:
            val = float(re.sub(r"[^\d.]", "", v2) or 0.0)
        data[lk] = val
    # normalize keys
    mapping = {
        "radial": "radial", "nylon": "nylon", "chips": "chips", "powder": "powder",
        "kachra": "kachra", "others": "others"
    }
    out = {mapping.get(k, k): v for k, v in data.items()}
    # normalize date string now for display
    if "date" in out:
        out["date_str"] = _norm_date(str(out["date"]))
    else:
        out["date_str"] = _norm_date(None)
    return out

def parse_actual(text: str) -> dict:
    """
    Accepts:
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

# ---------- Engine ----------
class RecoEngine:
    def __init__(self, report_path: str):
        self.report_path = report_path
        self.zone_defaults: dict[str, float] = {}
        self._load()

    def _load(self):
        self.zone_defaults = load_zone_rules(self.report_path, ZONE_SHEET)

    def plan(self, feed: dict) -> dict[str, float]:
        plan = dict(self.zone_defaults)
        # light composition adjustments
        total = sum(feed.get(k, 0.0) for k in ("radial", "nylon", "chips", "powder", "kachra", "others"))
        if total > 0:
            radial = feed.get("radial", 0.0) / total
            chips  = feed.get("chips", 0.0) / total
            nylon  = feed.get("nylon", 0.0) / total
            adj = 1.0 + 0.15 * (radial + 0.5 * chips - 0.6 * nylon)
            for k in list(plan.keys()):
                if k.startswith("300-400"):
                    plan[k] = max(20, plan[k] * adj)
                if k.startswith("200-300"):
                    plan[k] = max(15, plan[k] * (1.0 + 0.1 * radial - 0.05 * nylon))
        return plan

engine = RecoEngine(REPORT_PATH)

def predict_yield(feed: dict) -> tuple[float, float]:
    total = sum(feed.get(k, 0.0) for k in ("radial", "nylon", "chips", "powder", "kachra", "others"))
    if total <= 0:
        return (0.0, 0.6)
    w = state["weights"]
    base = 0.0
    for k, kg in feed.items():
        if k in w:
            base += (kg / total) * (w[k] * 100.0)
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

# ---------- Rendering ----------
def pretty_plan(plan: dict[str, float]) -> str:
    def k(x: str):
        m = re.match(r"(\d{2,3})-(\d{2,3})", x)
        if not m:
            return (999, 999, x)
        return (int(m.group(1)), int(m.group(2)), x)
    lines = []
    for z in sorted(plan.keys(), key=k):
        lines.append(f"{z}: {to_hhmmss(plan[z])}")
    return "\n".join(lines)

def plan_vs_actual_chart(plan: dict[str, float], actual: dict[str, float] | None = None) -> io.BytesIO:
    zones = [z for z in sorted(plan.keys(), key=lambda s: int(s.split("-")[0]))]
    pmins = [plan[z] for z in zones]
    amins = [actual.get(z.replace(" ", ""), np.nan) if actual else np.nan for z in zones]
    x = np.arange(len(zones))
    width = 0.38
    fig, ax = plt.subplots(figsize=(10, 3), dpi=160)
    ax.bar(x - width / 2, pmins, width, label="Plan (min)")
    if actual:
        ax.bar(x + width / 2, amins, width, label="Actual (min)")
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

# ---------- Jobs (reminders & daily summary) ----------
async def reminder_job(context: ContextTypes.DEFAULT_TYPE):
    data = context.job.data or {}
    chat_id = data.get("chat_id")
    batch   = data.get("batch")
    key = batch_key(chat_id, batch)

    # If still pending, ping and requeue hourly
    if key in state["reminders"]:
        txt = (f"⚠️ *Reminder:* Actual data not entered yet for *Batch {batch}* "
               f"on *{machine_label(chat_id)}*. Please send `/actual …`.")
        await context.bot.send_message(chat_id, txt, parse_mode="Markdown")
        if SUMMARY_CHAT_ID:
            await context.bot.send_message(SUMMARY_CHAT_ID, f"🔔 {txt}", parse_mode="Markdown")

        # Re-schedule next reminder in 1 hour
        context.job_queue.run_once(
            reminder_job,
            when=timedelta(hours=1),
            name=f"reminder:{key}",
            data={"chat_id": chat_id, "batch": batch}
        )

async def daily_summary_job(context: ContextTypes.DEFAULT_TYPE):
    now = datetime.now(BOT_TZ)
    header = f"📊 *Daily Summary — {now.strftime('%d-%b-%Y (%A) %H:%M IST')}*"
    lines = [header, ""]
    for cid_str, label in MACHINE_MAP.items():
        cid = int(cid_str)
        lf = state["latest_feed"].get(cid_str)
        status = "Idle"
        if lf:
            last_feed_ts = datetime.fromisoformat(lf["ts"])
            hrs = (now - last_feed_ts.astimezone(BOT_TZ)).total_seconds() / 3600
            last_act_iso = state["last_actual_ts"].get(cid_str)
            completed = False
            if last_act_iso:
                completed = datetime.fromisoformat(last_act_iso) >= last_feed_ts
            if hrs <= 11 and not completed:
                status = f"Running (batch {lf.get('batch','?')})"
            elif completed or hrs > 11:
                status = f"Completed (batch {lf.get('batch','?')})"
        lines.append(f"• {label}: {status}")
    msg = "\n".join(lines)
    if SUMMARY_CHAT_ID:
        await context.bot.send_message(SUMMARY_CHAT_ID, msg, parse_mode="Markdown")

def schedule_daily_summary(app: Application):
    # 21:35 IST daily
    app.job_queue.run_daily(
        daily_summary_job,
        time=time(21, 35, tzinfo=BOT_TZ),
        name="daily_summary"
    )

# ---------- Memory helpers ----------
def remember_feed(app: Application, chat_id: int, feed: dict, plan: dict):
    now = datetime.now(BOT_TZ)
    entry = {
        "ts": now.isoformat(),
        "batch": str(feed.get("batch", "")),
        "operator": str(feed.get("operator", "")),
        "date_str": str(feed.get("date_str", _norm_date(None))),
        "feed": feed,
        "plan": plan
    }
    state["latest_feed"][str(chat_id)] = entry

    # Schedule 12h reminder if batch present
    batch = entry["batch"]
    if batch:
        key = batch_key(chat_id, batch)
        state["reminders"][key] = {"chat_id": chat_id, "batch": batch}
        app.job_queue.run_once(
            reminder_job,
            when=now + timedelta(hours=12),
            name=f"reminder:{key}",
            data={"chat_id": chat_id, "batch": batch}
        )
    save_state()

# ---------- Commands ----------
HELP = (
    "*Commands*\n"
    "• Send `Feed: Radial=5.10T, Nylon=0.60T, Chips=3.40T, Powder=1.50T, batch=92, operator=Inglesh, date=09-11-2025` → Plan + predicted oil\n"
    "• `/whatif Feed: …` → Predicted oil only\n"
    "• `/actual Actual: 50-200=1:10, 200-300=0:45, 300-400=1:20, 400-450=0:22; oil=46.2; batch=92` → Compare & learn\n"
    "• `/chart` → Plan vs Actual chart (for last batch)\n"
    "• `/status` → Machine status (or all machines in summary group)\n"
    "• `/reload` → Reload Excel rules\n"
    "• `/id` → Show this chat id and name\n"
)

async def start_cmd(update: Update, context: ContextTypes.DEFAULT_TYPE):
    await update.message.reply_text("PyroVision Assistant ready ✅\n\n" + HELP, parse_mode="Markdown")

async def help_cmd(update: Update, context: ContextTypes.DEFAULT_TYPE):
    await update.message.reply_text(HELP, parse_mode="Markdown")

async def id_cmd(update: Update, context: ContextTypes.DEFAULT_TYPE):
    cid = update.effective_chat.id
    title = update.effective_chat.title or update.effective_chat.full_name
    await update.message.reply_text(f"chat_id: `{cid}`\nname/title: *{title}*", parse_mode="Markdown")

async def reload_cmd(update: Update, context: ContextTypes.DEFAULT_TYPE):
    engine._load()
    await update.message.reply_text("🔁 Reloaded recommendations from Excel.")

async def status_cmd(update: Update, context: ContextTypes.DEFAULT_TYPE):
    now = datetime.now(BOT_TZ)
    if SUMMARY_CHAT_ID and update.effective_chat.id == SUMMARY_CHAT_ID:
        lines = ["🟢 *Machine Status*"]
        for cid_str, label in MACHINE_MAP.items():
            lf = state["latest_feed"].get(cid_str)
            st = "Idle"
            if lf:
                last_feed_ts = datetime.fromisoformat(lf["ts"])
                hrs = (now - last_feed_ts.astimezone(BOT_TZ)).total_seconds() / 3600
                last_act_iso = state["last_actual_ts"].get(cid_str)
                completed = last_act_iso and (datetime.fromisoformat(last_act_iso) >= last_feed_ts)
                if hrs <= 11 and not completed:
                    st = f"Running (batch {lf.get('batch','?')})"
                elif completed:
                    st = f"Completed (batch {lf.get('batch','?')})"
                elif hrs > 12:
                    st = "Overdue (no actual)"
            lines.append(f"• {label}: {st}")
        await update.message.reply_text("\n".join(lines), parse_mode="Markdown")
    else:
        cid = str(update.effective_chat.id)
        label = machine_label(update.effective_chat.id)
        lf = state["latest_feed"].get(cid)
        st = "Idle"
        if lf:
            last_feed_ts = datetime.fromisoformat(lf["ts"])
            hrs = (now - last_feed_ts.astimezone(BOT_TZ)).total_seconds() / 3600
            last_act_iso = state["last_actual_ts"].get(cid)
            completed = last_act_iso and (datetime.fromisoformat(last_act_iso) >= last_feed_ts)
            if hrs <= 11 and not completed:
                st = f"Running (batch {lf.get('batch','?')})"
            elif completed:
                st = f"Completed (batch {lf.get('batch','?')})"
            elif hrs > 12:
                st = "Overdue (no actual)"
        await update.message.reply_text(f"{label}: {st}")

def _plan_header(label: str, feed: dict, pred: float, conf: float, batch: str, oper: str, date_str: str) -> str:
    return (
        f"🗂️ *{label}* — *Batch* {batch}, *Operator* {oper}\n"
        f"• *Date* {date_str}\n\n"
        f"🛢️  *Feed:* Radial {feed.get('radial',0)/1000:.2f}T, Nylon {feed.get('nylon',0)/1000:.2f}T, "
        f"Chips {feed.get('chips',0)/1000:.2f}T, Powder {feed.get('powder',0)/1000:.2f}T\n"
        f"📈  *Predicted Oil:* *{pred:.2f}%*  (confidence {conf:.2f})\n"
    )

async def plan_core(update: Update, context: ContextTypes.DEFAULT_TYPE, feed_text: str, predict_only=False):
    chat_id = update.effective_chat.id
    label = machine_label(chat_id)
    feed = parse_feed(feed_text)
    plan = engine.plan(feed)
    pred, conf = predict_yield(feed)
    batch = str(feed.get("batch", "?"))
    oper  = str(feed.get("operator", "?"))
    date_str = str(feed.get("date_str", _norm_date(None)))

    # Remember + schedule reminder
    remember_feed(context.application, chat_id, feed, plan)

    msg = [_plan_header(label, feed, pred, conf, batch, oper, date_str)]
    if not predict_only:
        msg.append("*Recommended zone minutes (plan):*\n" + pretty_plan(plan))
    text = "\n".join(msg)
    await update.message.reply_text(text, parse_mode="Markdown")
    if SUMMARY_CHAT_ID:
        await context.bot.send_message(SUMMARY_CHAT_ID, text, parse_mode="Markdown")

async def plan_cmd(update: Update, context: ContextTypes.DEFAULT_TYPE):
    await plan_core(update, context, update.message.text, predict_only=False)

async def whatif_cmd(update: Update, context: ContextTypes.DEFAULT_TYPE):
    t = re.sub(r"^/?whatif\s*","", update.message.text, flags=re.I)
    await plan_core(update, context, t, predict_only=True)

async def actual_cmd(update: Update, context: ContextTypes.DEFAULT_TYPE):
    chat_id = update.effective_chat.id
    lf = state["latest_feed"].get(str(chat_id))
    if not lf:
        await update.message.reply_text("No recent Feed found for this machine. Send the Feed first.")
        return
    plan = lf.get("plan", {})
    actual = parse_actual(update.message.text)

    deltas, tips = {}, []
    for z, pmin in plan.items():
        key = z.replace(" ", "")
        if key in actual:
            am = actual[key]
            deltas[z] = am - pmin
            if abs(deltas[z]) >= 5:
                tips.append(("reduce" if deltas[z] > 0 else "increase", z, abs(deltas[z])))

    if "oil" in actual:
        learn_from_actual(lf.get("feed", {}), float(actual["oil"]))
        state["last_actual_ts"][str(chat_id)] = datetime.now(BOT_TZ).isoformat()
        # clear any reminder for this batch
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

    # chart
    buf = plan_vs_actual_chart(plan, actual)
    await context.bot.send_photo(chat_id, photo=InputFile(buf, filename="plan_vs_actual.png"))
    if SUMMARY_CHAT_ID:
        await context.bot.send_message(SUMMARY_CHAT_ID, f"🧾 Actual logged for {machine_label(chat_id)} (batch {lf.get('batch','?')}).")

async def chart_cmd(update: Update, context: ContextTypes.DEFAULT_TYPE):
    lf = state["latest_feed"].get(str(update.effective_chat.id))
    if not lf:
        await update.message.reply_text("No recent plan to chart. Send a Feed first.")
        return
    plan = lf.get("plan", {})
    buf = plan_vs_actual_chart(plan, None)
    await context.bot.send_photo(update.effective_chat.id, photo=InputFile(buf, filename="plan_chart.png"))

async def text_router(update: Update, context: ContextTypes.DEFAULT_TYPE):
    txt = (update.message.text or "").strip()
    if re.match(r"(?i)^feed\s*:", txt):
        await plan_core(update, context, txt, predict_only=False)
    elif re.match(r"(?i)^actual\s*:", txt):
        await actual_cmd(update, context)
    elif re.fullmatch(r"(?i)status", txt):
        await status_cmd(update, context)
    else:
        await update.message.reply_text("I didn’t understand. Type /help for examples.")

# ---------- Main ----------
def main():
    if not TELEGRAM_BOT_TOKEN:
        raise RuntimeError("Missing TELEGRAM_BOT_TOKEN env var.")
    LOG.info("🚀 Starting PyroVision Assistant…")

    load_state()

    app = Application.builder().token(TELEGRAM_BOT_TOKEN).build()

    # Handlers
    app.add_handler(CommandHandler("start", start_cmd))
    app.add_handler(CommandHandler("help", help_cmd))
    app.add_handler(CommandHandler("plan", plan_cmd))
    app.add_handler(CommandHandler("whatif", whatif_cmd))
    app.add_handler(CommandHandler("actual", actual_cmd))
    app.add_handler(CommandHandler("chart", chart_cmd))
    app.add_handler(CommandHandler("status", status_cmd))
    app.add_handler(CommandHandler("reload", reload_cmd))
    app.add_handler(CommandHandler("id", id_cmd))
    app.add_handler(MessageHandler(filters.TEXT & ~filters.COMMAND, text_router))

    # Daily summary at 21:35 IST
    schedule_daily_summary(app)

    LOG.info("✅ Bot ready. Polling…")
    app.run_polling(close_loop=False)

if __name__ == "__main__":
    main()
