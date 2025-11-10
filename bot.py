# =========================================================
# PyroVision Assistant — Telegram + Excel + AI heuristics
# =========================================================

import os, io, re, json, math, logging
from datetime import datetime, timedelta, time, timezone
from zoneinfo import ZoneInfo

import numpy as np
import pandas as pd

import matplotlib
matplotlib.use("Agg")  # headless render
import matplotlib.pyplot as plt

from telegram import Update, InputFile
from telegram.constants import ParseMode
from telegram.ext import (
    Application, CommandHandler, MessageHandler, ContextTypes, filters
)

# -------------------- Logging --------------------
LOG = logging.getLogger("pyro_bot")
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)s | %(name)s | %(message)s"
)

# -------------------- Config ---------------------
BOT_TZ = ZoneInfo(os.environ.get("BOT_TZ", "Asia/Kolkata"))  # IST default

TELEGRAM_BOT_TOKEN = os.environ.get("TELEGRAM_BOT_TOKEN", "")
REPORT_PATH        = os.environ.get("REPORT_PATH", "pyrolysis_feed_temp_ZONE_TIME_report.xlsx")
ZONE_SHEET         = os.environ.get("ZONE_SHEET", "ZoneTime_Recommendations")

# Summary group (single chat id)
SUMMARY_CHAT_ID = None
_raw_sum = os.environ.get("SUMMARY_CHAT_ID", "").strip()
if _raw_sum:
    try:
        SUMMARY_CHAT_ID = int(_raw_sum)
    except:
        pass

# Machine map: JSON {"-1001":"Machine 1296 (R-1)", ...}
MACHINE_MAP: dict[str, str] = {}
try:
    _raw = os.environ.get("MACHINE_MAP", "").strip()
    if _raw:
        MACHINE_MAP = json.loads(_raw)
except Exception as e:
    LOG.warning("Could not parse MACHINE_MAP env var: %s", e)
if not MACHINE_MAP:
    # safe default; still works, prints id when missing
    MACHINE_MAP = {}

# -------------------- State ----------------------
STATE_FILE = "bot_state.json"
state = {
    "weights": {"radial": .46, "nylon": .40, "chips": .46, "powder": .53, "kachra": .38, "others": .40},
    "latest_feed": {},      # chat_id -> {ts, batch, operator, feed, plan}
    "last_actual_ts": {},   # chat_id -> iso
    "reminders": {}         # key(chat:batch) -> {"chat_id": int, "batch": str, "due": iso, "last_ping": iso}
}

def load_state():
    global state
    try:
        if os.path.exists(STATE_FILE):
            with open(STATE_FILE, "r", encoding="utf-8") as f:
                obj = json.load(f)
            # shallow merge to keep defaults if keys missing
            for k, v in obj.items():
                state[k] = v
            LOG.info("Loaded state from disk.")
        else:
            LOG.info("ℹ️ No state file found; starting fresh.")
    except Exception as e:
        LOG.warning("Could not load state: %s", e)

def save_state():
    try:
        with open(STATE_FILE, "w", encoding="utf-8") as f:
            json.dump(state, f, ensure_ascii=False, indent=2)
    except Exception as e:
        LOG.warning("Could not save state: %s", e)

# -------------------- Utilities ------------------
def to_hhmmss(minutes: float | int | None) -> str:
    if minutes is None or (isinstance(minutes, float) and math.isnan(minutes)):
        return "00:00:00"
    total = int(round(float(minutes) * 60))
    h, rem = divmod(total, 3600)
    m, s = divmod(rem, 60)
    return f"{h:02d}:{m:02d}:{s:02d}"

def hhmmss_to_minutes(s: str) -> float:
    s = (s or "").strip()
    if ":" not in s:
        try:
            return float(s)
        except:
            return 0.0
    parts = [int(x) for x in s.split(":")]
    if len(parts) == 3:
        h, m, sec = parts
        return h * 60 + m + sec / 60
    if len(parts) == 2:
        m, sec = parts
        return m + sec / 60
    return 0.0

def machine_label(chat_id: int) -> str:
    s = str(chat_id)
    return MACHINE_MAP.get(s) or MACHINE_MAP.get(str(int(chat_id))) or f"{chat_id}"

def nice_date(d: datetime | None = None) -> str:
    d = d or datetime.now(BOT_TZ)
    return d.strftime("%d-%b-%Y (%A)")

def batch_key(chat_id: int, batch: str | int | None) -> str:
    return f"{chat_id}:{batch or ''}"

# -------------------- Excel rules ----------------
def load_zone_rules(path=REPORT_PATH, sheet=ZONE_SHEET) -> dict[str, float]:
    rules: dict[str, float] = {}
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
        LOG.warning("Excel rules load failed (%s) — using safe defaults.", e)
        rules = {
            "50-200 reactor": 165,
            "200-300 reactor": 68,
            "300-400 reactor": 186,
            "300-400 separator": 175,
            "400-450 reactor": 75,
            "450-480 reactor": 20,
            "480-500 reactor": 0,
            "500-520 reactor": 0,
        }
    return rules

ZONE_RULES = load_zone_rules()

# -------------------- Parsing --------------------
def norm_key(s: str) -> str:
    return re.sub(r"[^a-z0-9\-]", "", s.lower())

def parse_feed(text: str) -> dict:
    """
    Accept both:
      Feed: Radial=5.10T, Nylon=0.60T, Chips=3.40T, Powder=1.50T, batch=66, operator=Ravi
      /feed: ...
    Units: T, t, Ton, MT, kg supported. Stored internally in kg.
    """
    t = re.sub(r"(?i)^/?feed\s*:\s*", "", text).strip()
    out = {}
    for part in re.split(r"[,\n;]+", t):
        if not part.strip():
            continue
        if "=" in part:
            k, v = [x.strip() for x in part.split("=", 1)]
        else:
            m = re.match(r"([A-Za-z\-]+)\s+([\d\.]+[a-zA-Z]*)", part)
            if not m:
                continue
            k, v = m.group(1), m.group(2)
        lk = norm_key(k)
        # fields that stay as text
        if lk in ("batch", "operator", "date"):
            out[lk] = v
            continue
        # numeric with units
        v_clean = v.strip()
        mT = re.match(r"([\d\.]+)\s*[tT](on|onne|on)?|([\d\.]+)\s*[mM][tT]", v_clean)
        mK = re.match(r"([\d\.]+)\s*[kK][gG]?$", v_clean)
        if mT:
            num = float(re.sub(r"[^\d\.]", "", v_clean))
            out[lk] = num * 1000.0
        elif mK:
            num = float(re.sub(r"[^\d\.]", "", v_clean))
            out[lk] = num
        else:
            try:
                out[lk] = float(re.sub(r"[^\d\.]", "", v_clean))
            except:
                pass
    return out

def parse_actual(text: str) -> dict:
    """
    Actual: 50-200=01:10, 200-300=00:45, 300-400=01:20, 400-450=00:22, 450-480=00:20, oil=46.2, batch=66
    """
    t = re.sub(r"(?i)^/?actual\s*:\s*", "", text).strip()
    out = {}
    for chunk in re.split(r"[;,]+", t):
        if "=" not in chunk:
            continue
        k, v = [x.strip() for x in chunk.split("=", 1)]
        lk = norm_key(k)
        if lk in ("oil", "oilpercent", "oilpct", "oilperc"):
            out["oil"] = float(re.sub(r"[^\d\.]", "", v) or 0.0)
        elif re.match(r"^\d{2,3}\s*-\s*\d{2,3}$", k.strip()):
            out[k.replace(" ", "")] = hhmmss_to_minutes(v)
        elif lk == "batch":
            out["batch"] = v
    return out

# -------------------- Engine ---------------------
class RecoEngine:
    def __init__(self, rules: dict[str, float]):
        self.rules = dict(rules)

    def reload(self):
        self.rules = load_zone_rules()

    def plan(self, feed: dict) -> dict[str, float]:
        # Base from rules
        pl = dict(self.rules)
        total = sum(feed.get(k, 0.0) for k in ("radial", "nylon", "chips", "powder", "kachra", "others"))
        if total > 0:
            radial = feed.get("radial", 0.0) / total
            chips  = feed.get("chips", 0.0) / total
            nylon  = feed.get("nylon", 0.0) / total
            # gentle adjustments
            for k in list(pl.keys()):
                if k.startswith("300-400"):
                    pl[k] = max(20, pl[k] * (1.0 + 0.15 * (radial + 0.5 * chips - 0.6 * nylon)))
                if k.startswith("200-300"):
                    pl[k] = max(15, pl[k] * (1.0 + 0.10 * radial - 0.05 * nylon))
        return pl

ENGINE = RecoEngine(ZONE_RULES)

def predict_yield(feed: dict) -> tuple[float, float]:
    total = sum(feed.get(k, 0.0) for k in ("radial", "nylon", "chips", "powder", "kachra", "others"))
    if total <= 0:
        return (0.0, 0.55)
    w = state["weights"]
    base = 0.0
    for k in w:
        share = feed.get(k, 0.0) / total
        base += share * (w[k] * 100.0)
    # bounds
    pred = max(30.0, min(55.0, base))
    # confidence by composition spread
    arr = [feed.get(x, 0.0) / total for x in ("radial", "nylon", "chips", "powder")]
    conf = max(0.60, min(0.95, 0.95 - 0.8 * np.std(arr)))
    return (round(pred, 2), round(conf, 2))

def learn_from_actual(feed: dict, oil_pct: float):
    total = sum(feed.get(k, 0.0) for k in ("radial", "nylon", "chips", "powder", "kachra", "others"))
    if total <= 0:
        return
    pred, _ = predict_yield(feed)
    err = (oil_pct - pred) / 100.0
    for k in state["weights"]:
        share = feed.get(k, 0.0) / total
        state["weights"][k] += 0.01 * err * share
    save_state()

# -------------------- Rendering ------------------
def pretty_plan(plan: dict[str, float]) -> str:
    def keyer(s: str):
        m = re.match(r"(\d{2,3})-", s)
        return int(m.group(1)) if m else 999
    lines = []
    for z in sorted(plan.keys(), key=keyer):
        lines.append(f"{z}: {to_hhmmss(plan[z])}")
    return "\n".join(lines)

def plan_vs_actual_chart(plan: dict[str, float], actual: dict[str, float] | None) -> io.BytesIO:
    zones = [z for z in sorted(plan.keys(), key=lambda s: int(s.split("-")[0]))]
    pmins = [plan[z] for z in zones]
    amins = [actual.get(z.replace(" ", ""), np.nan) if actual else np.nan for z in zones]

    fig, ax = plt.subplots(figsize=(12, 3), dpi=180)
    x = np.arange(len(zones))
    width = 0.38
    ax.bar(x - width / 2, pmins, width, label="Plan (min)")
    if actual:
        ax.bar(x + width / 2, amins, width, label="Actual (min)")
    ax.set_xticks(x, zones, rotation=25, ha="right")
    ax.set_ylabel("Minutes")
    ax.legend(loc="upper right")
    ax.grid(True, axis="y", alpha=0.2)
    buf = io.BytesIO()
    plt.tight_layout()
    plt.savefig(buf, format="png")
    plt.close(fig)
    buf.seek(0)
    return buf

# -------------------- Bot replies ----------------
HELP = (
    "*Commands*\n"
    "• Send `Feed: Radial=5.10T, Nylon=0.60T, Chips=3.40T, Powder=1.50T, batch=66, operator=Ravi`  \n"
    "  ↳ Plan + Predicted oil (+ fan-out to Summary).\n"
    "• Send `Actual: 50-200=01:10, 200-300=00:45, 300-400=01:20, 400-450=00:22, 450-480=00:20, oil=46.2, batch=66`  \n"
    "  ↳ Compare + tips + chart (+ learns & clears reminder).\n"
    "• `/status`  ↳ Machine status (in Summary group it lists all).\n"
    "• `/reload`  ↳ Reload Excel rules.\n"
    "• `/id`      ↳ Prints this chat’s id and label.\n"
    "\n"
    "You can also use `/feed:` and `/actual:` (slash forms)."
)

async def cmd_start(update: Update, context: ContextTypes.DEFAULT_TYPE):
    await update.message.reply_text("PyroVision Assistant ready ✅\n\n" + HELP, parse_mode=ParseMode.MARKDOWN)

async def cmd_help(update: Update, context: ContextTypes.DEFAULT_TYPE):
    await update.message.reply_text(HELP, parse_mode=ParseMode.MARKDOWN)

async def cmd_reload(update: Update, context: ContextTypes.DEFAULT_TYPE):
    ENGINE.reload()
    await update.message.reply_text("🔁 Reloaded recommendations from Excel.")

async def cmd_id(update: Update, context: ContextTypes.DEFAULT_TYPE):
    cid = update.effective_chat.id
    await update.message.reply_text(f"chat_id: `{cid}`\nname/title: *{machine_label(cid)}*",
                                    parse_mode=ParseMode.MARKDOWN)

def remember_feed(chat_id: int, feed: dict, plan: dict):
    now = datetime.now(BOT_TZ)
    entry = {
        "ts": now.isoformat(),
        "batch": str(feed.get("batch", "")),
        "operator": str(feed.get("operator", "")),
        "feed": feed,
        "plan": plan,
    }
    state["latest_feed"][str(chat_id)] = entry

    # 12h reminder schedule seed
    if entry["batch"]:
        key = batch_key(chat_id, entry["batch"])
        state["reminders"][key] = {
            "chat_id": chat_id,
            "batch": entry["batch"],
            "due": (now + timedelta(hours=12)).isoformat(),
            "last_ping": None
        }
    save_state()

def delta_tips(plan: dict[str, float], actual_min_by_zonekey: dict[str, float]) -> tuple[dict, list]:
    deltas = {}
    tips = []
    for z, pmin in plan.items():
        zkey = z.replace(" ", "")
        if zkey in actual_min_by_zonekey:
            am = actual_min_by_zonekey[zkey]
            diff = am - pmin
            deltas[z] = diff
            if abs(diff) >= 5:
                tips.append(("reduce" if diff > 0 else "increase", z, abs(diff)))
    return deltas, tips

async def plan_core(update: Update, context: ContextTypes.DEFAULT_TYPE, feed_text: str):
    chat_id = update.effective_chat.id
    feed = parse_feed(feed_text)
    plan = ENGINE.plan(feed)
    pred, conf = predict_yield(feed)

    label = machine_label(chat_id)
    batch = feed.get("batch", "?")
    oper = feed.get("operator", "?")
    today = nice_date()

    remember_feed(chat_id, feed, plan)

    msg = []
    msg.append(f"📘 *{label}* — *Batch* {batch}, *Operator* {oper}")
    msg.append(f"• *Date* {today}")
    msg.append(
        "🛢️ *Feed:* "
        f"Radial {feed.get('radial',0)/1000:.2f}T, Nylon {feed.get('nylon',0)/1000:.2f}T, "
        f"Chips {feed.get('chips',0)/1000:.2f}T, Powder {feed.get('powder',0)/1000:.2f}T"
    )
    msg.append(f"📈 *Predicted Oil:* *{pred:.2f}%*  (confidence {conf:.2f})")
    msg.append("\n*Recommended zone minutes (plan):*\n" + pretty_plan(plan))
    text = "\n".join(msg)

    await update.message.reply_text(text, parse_mode=ParseMode.MARKDOWN)
    if SUMMARY_CHAT_ID:
        await context.bot.send_message(SUMMARY_CHAT_ID, text, parse_mode=ParseMode.MARKDOWN)

async def cmd_actual(update: Update, context: ContextTypes.DEFAULT_TYPE):
    chat_id = update.effective_chat.id
    actual = parse_actual(update.message.text)
    lf = state["latest_feed"].get(str(chat_id))
    if not lf:
        await update.message.reply_text("No recorded Feed for this machine. Send the Feed first.")
        return

    plan = lf.get("plan", {})
    deltas, tips = delta_tips(plan, actual)

    lines = []
    lines.append("📊 *Deviation vs plan (min):*")
    for z in sorted(deltas.keys(), key=lambda s: int(s.split("-")[0])):
        lines.append(f"{z}: {deltas[z]:+.0f}")
    if tips:
        lines.append("\n🛠️ *Recommendations:*")
        for d, z, mins in tips:
            lines.append(f"• {d} {z} by ~{mins:.0f} min")
    else:
        lines.append("\n✅ Near-optimal execution vs plan.")

    # learn + clear reminder
    if "oil" in actual:
        learn_from_actual(lf.get("feed", {}), float(actual["oil"]))
    state["last_actual_ts"][str(chat_id)] = datetime.now(BOT_TZ).isoformat()
    key = batch_key(chat_id, lf.get("batch"))
    if key in state["reminders"]:
        del state["reminders"][key]
    save_state()

    # send text + chart
    await update.message.reply_text("\n".join(lines), parse_mode=ParseMode.MARKDOWN)
    buf = plan_vs_actual_chart(plan, actual)
    await context.bot.send_photo(chat_id, InputFile(buf, filename="plan_vs_actual.png"))

    # fan out to summary
    if SUMMARY_CHAT_ID:
        await context.bot.send_message(
            SUMMARY_CHAT_ID,
            f"🧾 Actual logged for *{machine_label(chat_id)}* (batch {lf.get('batch','?')}).",
            parse_mode=ParseMode.MARKDOWN
        )

async def cmd_status(update: Update, context: ContextTypes.DEFAULT_TYPE):
    now = datetime.now(BOT_TZ)

    # Summary chat → list all
    if SUMMARY_CHAT_ID and update.effective_chat.id == SUMMARY_CHAT_ID:
        lines = ["🟢 *Machine Status*"]
        for cid_str, label in MACHINE_MAP.items():
            lf = state["latest_feed"].get(cid_str)
            st = "Idle"
            if lf:
                last_feed_ts = datetime.fromisoformat(lf["ts"]).astimezone(BOT_TZ)
                hrs = (now - last_feed_ts).total_seconds() / 3600
                last_act_iso = state["last_actual_ts"].get(cid_str)
                completed = False
                if last_act_iso:
                    completed = datetime.fromisoformat(last_act_iso) >= last_feed_ts
                if hrs <= 11 and not completed:
                    st = f"Running (batch {lf.get('batch','?')})"
                elif completed or hrs > 11:
                    st = f"Completed (batch {lf.get('batch','?')})"
            lines.append(f"• {label}: {st}")
        await update.message.reply_text("\n".join(lines), parse_mode=ParseMode.MARKDOWN)
        return

    # Machine chat → this one
    cid = str(update.effective_chat.id)
    label = machine_label(update.effective_chat.id)
    lf = state["latest_feed"].get(cid)
    st = "Idle"
    if lf:
        last_feed_ts = datetime.fromisoformat(lf["ts"]).astimezone(BOT_TZ)
        hrs = (now - last_feed_ts).total_seconds() / 3600
        last_act_iso = state["last_actual_ts"].get(cid)
        completed = last_act_iso and (datetime.fromisoformat(last_act_iso) >= last_feed_ts)
        if hrs <= 11 and not completed:
            st = f"Running (batch {lf.get('batch','?')})"
        elif completed or hrs > 11:
            st = f"Completed (batch {lf.get('batch','?')})"
    await update.message.reply_text(f"{label}: {st}")

# -------------------- Jobs -----------------------
async def reminder_tick(context: ContextTypes.DEFAULT_TYPE):
    """
    Runs every 5 minutes.
    If a reminder is due (>=12h since feed) and last ping >1h ago (or never), ping machine & summary.
    """
    now = datetime.now(BOT_TZ)
    to_save = False
    for key, rec in list(state["reminders"].items()):
        chat_id = rec.get("chat_id")
        batch = rec.get("batch")
        due = datetime.fromisoformat(rec["due"]).astimezone(BOT_TZ)
        last = datetime.fromisoformat(rec["last_ping"]).astimezone(BOT_TZ) if rec.get("last_ping") else None

        # if actual already logged, skip & delete
        lf = state["latest_feed"].get(str(chat_id))
        last_act = state["last_actual_ts"].get(str(chat_id))
        if last_act and lf and datetime.fromisoformat(last_act) >= datetime.fromisoformat(lf["ts"]):
            del state["reminders"][key]
            to_save = True
            continue

        if now >= due and (last is None or (now - last) >= timedelta(hours=1)):
            txt = (f"⚠️ *Reminder:* Actual data not entered yet for *Batch {batch}* "
                   f"on *{machine_label(chat_id)}*. Please send `Actual: ...`")
            await context.bot.send_message(chat_id, txt, parse_mode=ParseMode.MARKDOWN)
            if SUMMARY_CHAT_ID:
                await context.bot.send_message(SUMMARY_CHAT_ID, "🔔 " + txt, parse_mode=ParseMode.MARKDOWN)
            state["reminders"][key]["last_ping"] = now.isoformat()
            to_save = True
    if to_save:
        save_state()

async def daily_summary(context: ContextTypes.DEFAULT_TYPE):
    now = datetime.now(BOT_TZ)
    title = f"📊 *Daily Summary* — {now.strftime('%d-%b-%Y (%A) %H:%M')} IST"
    lines = [title]
    for cid_str, label in MACHINE_MAP.items():
        lf = state["latest_feed"].get(cid_str)
        status = "Idle"
        if lf:
            last_feed_ts = datetime.fromisoformat(lf["ts"]).astimezone(BOT_TZ)
            hrs = (now - last_feed_ts).total_seconds() / 3600
            last_act_iso = state["last_actual_ts"].get(cid_str)
            completed = False
            if last_act_iso:
                completed = datetime.fromisoformat(last_act_iso) >= last_feed_ts
            if hrs <= 11 and not completed:
                status = f"Running (batch {lf.get('batch','?')})"
            elif completed or hrs > 11:
                status = f"Completed (batch {lf.get('batch','?')})"
        lines.append(f"• {label}: {status}")
    if SUMMARY_CHAT_ID:
        await context.bot.send_message(SUMMARY_CHAT_ID, "\n".join(lines), parse_mode=ParseMode.MARKDOWN)

def schedule_jobs(app: Application):
    # Reminder loop every 5 minutes
    app.job_queue.run_repeating(reminder_tick, interval=300, first=60)

    # Daily summary 21:35 IST
    hh, mm = 21, 35
    app.job_queue.run_daily(
        daily_summary,
        time=time(hour=hh, minute=mm, tzinfo=BOT_TZ),
        name="daily_summary"
    )

# -------------------- Text router ----------------
async def text_router(update: Update, context: ContextTypes.DEFAULT_TYPE):
    text = (update.message.text or "").strip()

    # Allow both "Feed:" and "/feed:", "Actual:" and "/actual:"
    if re.match(r"(?i)^/?feed\s*:", text):
        await plan_core(update, context, text)
    elif re.match(r"(?i)^/?actual\s*:", text):
        await cmd_actual(update, context)
    elif re.match(r"(?i)^status$", text):
        await cmd_status(update, context)
    else:
        await update.message.reply_text("I didn’t understand.\n\n" + HELP, parse_mode=ParseMode.MARKDOWN)

# -------------------- Main -----------------------
def main():
    LOG.info("Starting PyroVision Assistant…")
    if not TELEGRAM_BOT_TOKEN:
        raise RuntimeError("Missing TELEGRAM_BOT_TOKEN env var.")

    load_state()

    app = Application.builder().token(TELEGRAM_BOT_TOKEN).build()

    # commands
    app.add_handler(CommandHandler("start", cmd_start))
    app.add_handler(CommandHandler("help", cmd_help))
    app.add_handler(CommandHandler("reload", cmd_reload))
    app.add_handler(CommandHandler("status", cmd_status))
    app.add_handler(CommandHandler("id", cmd_id))

    # text
    app.add_handler(MessageHandler(filters.TEXT & ~filters.COMMAND, text_router))

    # schedule jobs via JobQueue (safe for PTB 21.x)
    schedule_jobs(app)

    LOG.info("✅ Bot ready. Polling…")
    app.run_polling(drop_pending_updates=True)

if __name__ == "__main__":
    main()
