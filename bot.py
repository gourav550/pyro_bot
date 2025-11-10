# ======================================================
#  PyroVision Assistant  —  full working bot.py
#  Features:
#   • Feed -> plan + predicted oil (+weekday date)
#   • Actual -> deviations + recommendations + chart
#   • Status (machine / summary) with 12h Running window
#   • 12h reminders (hourly repeats) + daily summary 21:35 IST
# ======================================================

import os, json, re, io, math, asyncio, logging
from datetime import datetime, timedelta
from zoneinfo import ZoneInfo

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from telegram import Update, InputFile
from telegram.ext import (
    Application, CommandHandler, MessageHandler, ContextTypes, filters
)

# -------------------- Logging --------------------
LOG = logging.getLogger("pyro_bot")
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)s | %(name)s | %(message)s"
)

# -------------------- Config --------------------
TELEGRAM_BOT_TOKEN = os.environ.get("TELEGRAM_BOT_TOKEN", "").strip()
REPORT_PATH = os.environ.get("REPORT_PATH", "pyrolysis_feed_temp_ZONE_TIME_report.xlsx")
ZONE_SHEET = "ZoneTime_Recommendations"

BOT_TZ = ZoneInfo(os.environ.get("BOT_TZ", "Asia/Kolkata"))

SUMMARY_CHAT_ID_ENV = os.environ.get("SUMMARY_CHAT_ID", "").strip()
SUMMARY_CHAT_ID = int(SUMMARY_CHAT_ID_ENV) if SUMMARY_CHAT_ID_ENV else None

try:
    MACHINE_MAP = json.loads(os.environ.get("MACHINE_MAP", "{}"))
except Exception:
    MACHINE_MAP = {}

STATE_FILE = "bot_state.json"

# -------------------- Runtime state --------------------
state = {
    "weights": {   # simple, learnable oil weights per kg share → %
        "radial": .45, "nylon": .42, "chips": .46, "powder": .53, "kachra": .40, "others": .40
    },
    "latest_feed": {},     # chat_id -> {ts, batch, operator, feed, plan}
    "last_actual_ts": {},  # chat_id -> iso
    "reminders": {}        # key(chat:batch) -> {"chat_id","batch","due_iso"}
}

def load_state():
    global state
    try:
        if os.path.exists(STATE_FILE):
            with open(STATE_FILE, "r", encoding="utf-8") as f:
                obj = json.load(f)
            # shallow merge to keep defaults for new keys
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

# -------------------- Utilities --------------------
def to_hhmmss(minutes: float | int | None) -> str:
    try:
        if minutes is None or (isinstance(minutes, float) and math.isnan(minutes)):
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
        return float(re.sub(r"[^\d.]", "", s) or 0.0)
    parts = [int(x) for x in s.split(":")]
    if len(parts) == 3:
        h, m, sec = parts
        return h * 60 + m + sec / 60
    if len(parts) == 2:
        m, sec = parts
        return m + sec / 60
    return 0.0

def norm_key(k: str) -> str:
    return re.sub(r"\W+", "", (k or "").strip().lower())

def _norm_date(s: str | None) -> str:
    """Normalize date to 'DD-Mon-YYYY (Weekday)'. Uses BOT_TZ."""
    now = datetime.now(BOT_TZ)
    if not s:
        return now.strftime("%d-%b-%Y (%A)")
    s = s.strip()
    for fmt in ("%d-%m-%Y", "%d.%m.%Y", "%d/%m/%Y", "%d-%m-%y", "%d.%m.%y", "%d/%m/%y", "%Y-%m-%d"):
        try:
            dt = datetime.strptime(s, fmt).replace(tzinfo=BOT_TZ)
            return dt.strftime("%d-%b-%Y (%A)")
        except Exception:
            continue
    try:
        dt = datetime.fromisoformat(s)
        return dt.astimezone(BOT_TZ).strftime("%d-%b-%Y (%A)")
    except Exception:
        return now.strftime("%d-%b-%Y (%A)")

# -------------------- Excel rules --------------------
def load_zone_rules(path=REPORT_PATH, sheet=ZONE_SHEET):
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
        LOG.warning(f"⚠️ Failed to load Excel rules: {e}")
        rules = {
            "50-200 reactor": 165,  # fallback minutes
            "200-300 reactor": 68,
            "300-400 reactor": 183,
            "300-400 separator": 173,
            "400-450 reactor": 75,
            "450-480 reactor": 20,
            "480-500 reactor": 0,
            "500-520 reactor": 0
        }
    return rules

ZONE_RULES = load_zone_rules()

class RecoEngine:
    def __init__(self, report_path: str):
        self.report_path = report_path
        self.zone_defaults: dict[str, float] = {}
        self._load()
    def _load(self):
        self.zone_defaults = load_zone_rules(self.report_path, ZONE_SHEET)
    def plan(self, feed: dict) -> dict[str, float]:
        # Start from defaults, then small composition tweaks
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

# -------------------- Parsing --------------------
def parse_feed(text: str) -> dict:
    """
    Accept:
      Feed: Radial=5.1T, Nylon=0.6T, Chips=3.4T, Powder=1.5T, batch=92, operator=Ravi, date=10-11-2025
    """
    data = {}
    t = re.sub(r"^/?(plan|predict)?\s*feed\s*:\s*|^feed\s*:\s*", "", text, flags=re.I).strip()
    for part in re.split(r"[,\n;]+", t):
        if not part.strip():
            continue
        if "=" in part:
            key, val = [x.strip() for x in part.split("=", 1)]
        else:
            m = re.match(r"([A-Za-z]+)\s+([\d.]+)", part.strip())
            if not m:
                continue
            key, val = m.group(1), m.group(2)
        k = norm_key(key)
        if k in ("batch", "operator", "date"):
            data[k] = val
            continue
        # numeric w/ units
        v = val.replace("ton", "T").replace("mt", "T").replace("kg", "K")
        mT = re.match(r"([\d.]+)\s*[tT]$", v)
        mK = re.match(r"([\d.]+)\s*[kK]$", v)
        if mT:
            data[k] = float(mT.group(1)) * 1000.0
        elif mK:
            data[k] = float(mK.group(1))
        else:
            data[k] = float(re.sub(r"[^\d.]", "", v) or 0.0)
    # normalize keys
    mapping = {"radial":"radial","nylon":"nylon","chips":"chips","powder":"powder","kachra":"kachra","others":"others"}
    return {mapping.get(k, k): v for k, v in data.items()}

def parse_actual(text: str) -> dict:
    """
    Accept:
      Actual: 50-200=02:57, 200-300=01:06, 300-400=03:18, 400-450=01:10, 450-480=00:32, oil=38.27, batch=66
    """
    t = re.sub(r"^/?actual\s*:\s*", "", text, flags=re.I).strip()
    out = {}
    for chunk in re.split(r"[;,]+", t):
        if "=" not in chunk:
            continue
        k, v = [x.strip() for x in chunk.split("=", 1)]
        lk = norm_key(k)
        if lk in ("oil", "oilpercent", "oilyield"):
            out["oil"] = float(re.sub(r"[^\d.]", "", v) or 0.0)
        elif re.match(r"\d{2,3}\s*-\s*\d{2,3}", k):
            out[k.replace(" ", "")] = hhmmss_to_minutes(v)
        elif lk == "batch":
            out["batch"] = v
    return out

# -------------------- Prediction & learning --------------------
def predict_yield(feed: dict) -> tuple[float, float]:
    total = sum(feed.get(k, 0.0) for k in ("radial","nylon","chips","powder","kachra","others"))
    if total <= 0:
        return (0.0, 0.6)
    w = state["weights"]
    base = 0.0
    for k, kg in feed.items():
        if k in w:
            base += (kg / total) * (w[k] * 100.0)
    pred = max(30.0, min(55.0, base))
    ratio_spread = np.std([feed.get(x, 0.0) / total for x in ("radial","nylon","chips","powder")])
    conf = max(0.55, min(0.95, 0.95 - 0.8 * ratio_spread))
    return (round(pred, 2), round(conf, 2))

def learn_from_actual(feed: dict, oil_actual: float):
    total = sum(feed.get(k, 0.0) for k in ("radial","nylon","chips","powder","kachra","others"))
    if total <= 0:
        return
    pred, _ = predict_yield(feed)
    err = (oil_actual - pred) / 100.0
    for k in ("radial","nylon","chips","powder","kachra","others"):
        share = feed.get(k, 0.0) / total
        state["weights"][k] += 0.01 * err * share
    save_state()

# -------------------- Rendering helpers --------------------
def pretty_plan(plan: dict[str, float]) -> str:
    def k(x):
        m = re.match(r"(\d{2,3})-(\d{2,3})", x)
        if not m: return (999, 999, x)
        return (int(m.group(1)), int(m.group(2)), x)
    lines = []
    for z in sorted(plan.keys(), key=k):
        lines.append(f"{z}: {to_hhmmss(plan[z])}")
    return "\n".join(lines)

def route_summary(context: ContextTypes.DEFAULT_TYPE, text: str):
    if SUMMARY_CHAT_ID:
        asyncio.create_task(context.bot.send_message(SUMMARY_CHAT_ID, text, parse_mode="Markdown"))

def machine_label(chat_id: int) -> str:
    return MACHINE_MAP.get(str(chat_id)) or MACHINE_MAP.get(str(int(chat_id))) or str(chat_id)

def chart_from_plan_vs_actual(plan: dict[str, float], actual: dict[str, float] | None = None) -> io.BytesIO:
    zones = [z for z in sorted(plan.keys(), key=lambda s: int(s.split("-")[0]))]
    pmins = [plan[z] for z in zones]
    amins = [actual.get(z.replace(" ", ""), np.nan) if actual else np.nan for z in zones]
    x = np.arange(len(zones))
    width = 0.38
    fig, ax = plt.subplots(figsize=(7.8, 2.6), dpi=180)
    ax.bar(x - width/2, pmins, width, label="Plan (min)")
    if actual:
        ax.bar(x + width/2, amins, width, label="Actual (min)")
    ax.set_xticks(x, zones, rotation=25, ha="right")
    ax.set_ylabel("Minutes")
    ax.legend()
    ax.grid(True, axis="y", alpha=0.25)
    buf = io.BytesIO()
    plt.tight_layout()
    plt.savefig(buf, format="png")
    plt.close(fig)
    buf.seek(0)
    return buf

# -------------------- Recommendation builder --------------------
def make_recommendations(plan: dict[str, float],
                         actual_map: dict[str, float],
                         oil_pred: float | None = None,
                         oil_actual: float | None = None) -> tuple[list[str], dict[str, float]]:
    tips = []
    deltas = {}
    for z, pmin in plan.items():
        key = z.replace(" ", "")
        if key in actual_map:
            d = actual_map[key] - pmin
            deltas[z] = d
            if abs(d) >= 5:
                weight = 1.2 if z.startswith("300-400") else (1.1 if z.startswith("200-300") else 1.0)
                mins = round(abs(d) * weight)
                direction = "reduce" if d > 0 else "increase"
                tips.append(f"• {direction} {z} by ~{mins} min")
    if oil_actual is not None and oil_pred is not None:
        gap = round(oil_actual - oil_pred, 2)
        if gap < -0.5:
            tips.insert(0, "• Oil below expected → shift time from 50–200/400–450 into **200–300** and **300–400** next run.")
        elif gap > 0.5:
            tips.insert(0, "• Oil above expected → keep this profile; only minor smoothing.")
    if not tips:
        tips = ["• Near-optimal execution vs plan."]
    return tips, deltas

# -------------------- Reminders & daily summary (APScheduler-free lightweight) --------------------
# We implement simple tick-based checks inside polling cycle via asyncio tasks.

async def reminder_tick(app: Application):
    """Check every minute which reminders are due; send and re-queue hourly until actual arrives."""
    while True:
        try:
            now = datetime.now(BOT_TZ)
            due_keys = []
            for key, payload in list(state["reminders"].items()):
                due = datetime.fromisoformat(payload["due"]).astimezone(BOT_TZ)
                if now >= due:
                    chat_id = int(payload["chat_id"])
                    batch = payload["batch"]
                    text = (f"⚠️ *Reminder:* Actual data not entered yet for *Batch {batch}* "
                            f"on *{machine_label(chat_id)}*. Please send `Actual: ...`.")
                    await app.bot.send_message(chat_id, text, parse_mode="Markdown")
                    if SUMMARY_CHAT_ID:
                        await app.bot.send_message(SUMMARY_CHAT_ID, f"🔔 {text}", parse_mode="Markdown")
                    # re-schedule 1h later
                    payload["due"] = (now + timedelta(hours=1)).isoformat()
            save_state()
        except Exception as e:
            LOG.warning(f"reminder_tick error: {e}")
        await asyncio.sleep(60)  # every minute

async def daily_summary_tick(app: Application):
    """At 21:35 local time post a daily summary once per day."""
    last_day = None
    while True:
        try:
            now = datetime.now(BOT_TZ)
            if now.hour == 21 and now.minute == 35:
                day_key = now.strftime("%Y-%m-%d")
                if day_key != last_day and SUMMARY_CHAT_ID:
                    lines = [f"📊 *Daily Summary* — {now.strftime('%d-%b-%Y (%A) %H:%M')}"]
                    for cid_str, label in MACHINE_MAP.items():
                        lf = state["latest_feed"].get(cid_str)
                        if not lf:
                            lines.append(f"• {label}: Idle")
                            continue
                        feed_ts = datetime.fromisoformat(lf["ts"]).astimezone(BOT_TZ)
                        batch = lf.get("batch", "?")
                        last_act_iso = state["last_actual_ts"].get(cid_str)
                        completed = False
                        if last_act_iso:
                            completed = datetime.fromisoformat(last_act_iso).astimezone(BOT_TZ) >= feed_ts
                        hours = (now - feed_ts).total_seconds()/3600.0
                        if not completed and hours <= 12:
                            lines.append(f"• {label}: Running (batch {batch})")
                        else:
                            lines.append(f"• {label}: Completed (batch {batch}, Date {feed_ts.strftime('%d-%b-%Y (%A)')})")
                    await app.bot.send_message(SUMMARY_CHAT_ID, "\n".join(lines), parse_mode="Markdown")
                    last_day = day_key
        except Exception as e:
            LOG.warning(f"daily_summary_tick error: {e}")
        await asyncio.sleep(30)  # check twice a minute

def batch_key(chat_id: int, batch: str | int | None) -> str:
    return f"{chat_id}:{batch or ''}"

def remember_feed(chat_id: int, feed: dict, plan: dict):
    now = datetime.now(BOT_TZ)
    entry = {
        "ts": now.isoformat(),
        "batch": str(feed.get("batch", "")),
        "operator": str(feed.get("operator", "")),
        "feed": feed,
        "plan": plan
    }
    state["latest_feed"][str(chat_id)] = entry
    batch = entry["batch"]
    if batch:
        key = batch_key(chat_id, batch)
        # first reminder at +12h
        state["reminders"][key] = {"chat_id": chat_id, "batch": batch, "due": (now + timedelta(hours=12)).isoformat()}
    save_state()

# -------------------- Commands --------------------
HELP = (
    "*Commands*\n"
    "• `Feed: Radial=5.1T, Nylon=0.6T, Chips=3.4T, Powder=1.5T, batch=92, operator=Ravi` → Plan + Predicted Oil\n"
    "• `/actual Actual: 50-200=2:57, 200-300=1:06, 300-400=3:18, 400-450=1:10; oil=45.2; batch=92` → Compare & Learn\n"
    "• `/status` → Status of current machine (or all machines in Summary group)\n"
    "• `/reload` → Reload Excel rules\n"
)

async def start(update: Update, context: ContextTypes.DEFAULT_TYPE):
    await update.message.reply_text("PyroVision Assistant ready ✅\n\n" + HELP, parse_mode="Markdown")

async def help_cmd(update: Update, context: ContextTypes.DEFAULT_TYPE):
    await update.message.reply_text(HELP, parse_mode="Markdown")

async def reload_cmd(update: Update, context: ContextTypes.DEFAULT_TYPE):
    engine._load()
    await update.message.reply_text("🔁 Reloaded recommendations from Excel.")
    route_summary(context, f"🔁 Rules reloaded by {machine_label(update.effective_chat.id)}")

async def plan_cmd_core(update: Update, context: ContextTypes.DEFAULT_TYPE, feed_text: str):
    chat_id = update.effective_chat.id
    feed = parse_feed(feed_text)

    plan = engine.plan(feed)
    pred, conf = predict_yield(feed)
    label = machine_label(chat_id)
    batch = feed.get("batch", "?")
    oper = feed.get("operator", "?")
    date_str = _norm_date(feed.get("date"))

    remember_feed(chat_id, feed, plan)

    msg = []
    msg.append(f"📒 *{label}* — *Batch* {batch}, *Operator* {oper}")
    msg.append(f"• *Date* {date_str}")
    msg.append(f"🛢️ Feed: Radial {feed.get('radial',0)/1000:.2f}T, Nylon {feed.get('nylon',0)/1000:.2f}T, "
               f"Chips {feed.get('chips',0)/1000:.2f}T, Powder {feed.get('powder',0)/1000:.2f}T")
    msg.append(f"📈 Predicted Oil: *{pred:.2f}%*  (confidence {conf:.2f})")
    msg.append("\n*Recommended zone minutes (plan):*\n" + pretty_plan(plan))
    text = "\n".join(msg)

    await update.message.reply_text(text, parse_mode="Markdown")
    route_summary(context, text)

async def plan_cmd(update: Update, context: ContextTypes.DEFAULT_TYPE):
    await plan_cmd_core(update, context, update.message.text)

async def actual_cmd(update: Update, context: ContextTypes.DEFAULT_TYPE):
    chat_id = update.effective_chat.id
    cid_str = str(chat_id)
    actual = parse_actual(update.message.text)

    lf = state["latest_feed"].get(cid_str)
    if not lf:
        await update.message.reply_text("I don’t find a recorded Feed for this machine. Send the Feed first.")
        return

    plan = lf.get("plan", {})
    oil_actual = actual.get("oil")
    oil_pred, _ = predict_yield(lf.get("feed", {}))

    tips, deltas = make_recommendations(plan, actual, oil_pred=oil_pred, oil_actual=oil_actual)

    # Mark actual arrived
    state["last_actual_ts"][cid_str] = datetime.now(BOT_TZ).isoformat()
    key = batch_key(chat_id, lf.get("batch"))
    if key in state["reminders"]:
        del state["reminders"][key]
    save_state()

    head = f"🧾 *Actual logged* for {machine_label(chat_id)} (batch {lf.get('batch','?')})."
    lines = [head, "", "*Deviation vs plan (min):*"]
    for z in sorted(deltas.keys(), key=lambda s: int(s.split("-")[0])):
        lines.append(f"{z}: {deltas[z]:+.0f}")
    if oil_actual is not None:
        lines.append(f"\nOil entered: *{float(oil_actual):.2f}%* (vs model {oil_pred:.2f}%)")
    lines.append("\n*Recommendations:*")
    lines += tips
    msg_text = "\n".join(lines)

    await update.message.reply_text(msg_text, parse_mode="Markdown")
    buf = chart_from_plan_vs_actual(plan, {k.replace(" ",""): v for k, v in actual.items() if "-" in k})
    await context.bot.send_photo(chat_id, photo=InputFile(buf, filename="plan_vs_actual.png"))

    if SUMMARY_CHAT_ID:
        await context.bot.send_message(SUMMARY_CHAT_ID, msg_text, parse_mode="Markdown")
        buf2 = chart_from_plan_vs_actual(plan, {k.replace(" ",""): v for k, v in actual.items() if "-" in k})
        await context.bot.send_photo(SUMMARY_CHAT_ID, photo=InputFile(buf2, filename="plan_vs_actual.png"))

async def status_cmd(update: Update, context: ContextTypes.DEFAULT_TYPE):
    now = datetime.now(BOT_TZ)

    def line_for(cid_str: str):
        label = MACHINE_MAP.get(cid_str, cid_str)
        lf = state["latest_feed"].get(cid_str)
        if not lf:
            return f"• {label}: Idle"
        feed_ts = datetime.fromisoformat(lf["ts"]).astimezone(BOT_TZ)
        batch = lf.get("batch", "?")
        last_act_iso = state["last_actual_ts"].get(cid_str)
        completed = False
        if last_act_iso:
            completed = datetime.fromisoformat(last_act_iso).astimezone(BOT_TZ) >= feed_ts
        hours = (now - feed_ts).total_seconds() / 3600.0
        if not completed and hours <= 12:
            return f"• {label}: Running (batch {batch})"
        return f"• {label}: Completed (batch {batch}, Date {feed_ts.strftime('%d-%b-%Y (%A)')})"

    if SUMMARY_CHAT_ID and update.effective_chat.id == SUMMARY_CHAT_ID:
        lines = ["🟢 *Machine Status*"]
        for cid in MACHINE_MAP.keys():
            lines.append(line_for(cid))
        await update.message.reply_text("\n".join(lines), parse_mode="Markdown")
    else:
        cid_str = str(update.effective_chat.id)
        await update.message.reply_text(line_for(cid_str), parse_mode="Markdown")

async def text_router(update: Update, context: ContextTypes.DEFAULT_TYPE):
    t = (update.message.text or "").strip()
    if re.match(r"(?i)^feed\s*:", t):
        await plan_cmd_core(update, context, t)
    elif re.match(r"(?i)^actual\s*:", t):
        await actual_cmd(update, context)
    elif re.match(r"(?i)^status$", t):
        await status_cmd(update, context)
    else:
        await update.message.reply_text("I didn’t understand. Type /help for examples.")

# -------------------- Main --------------------
def main():
    if not TELEGRAM_BOT_TOKEN:
        raise RuntimeError("Missing TELEGRAM_BOT_TOKEN")
    load_state()

    app = Application.builder().token(TELEGRAM_BOT_TOKEN).build()

    app.add_handler(CommandHandler("start", start))
    app.add_handler(CommandHandler("help", help_cmd))
    app.add_handler(CommandHandler("plan", plan_cmd))
    app.add_handler(CommandHandler("actual", actual_cmd))
    app.add_handler(CommandHandler("reload", reload_cmd))
    app.add_handler(CommandHandler("status", status_cmd))
    app.add_handler(MessageHandler(filters.TEXT & ~filters.COMMAND, text_router))

    # Background “ticks” for reminders and daily summary
    app.create_task(reminder_tick(app))
    app.create_task(daily_summary_tick(app))

    LOG.info("✅ Bot ready. Polling…")
    app.run_polling()

if __name__ == "__main__":
    main()
