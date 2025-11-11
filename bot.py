# ======================================================
#  PyroVision Assistant — Telegram + Excel + AI (v2)
#  - Feed  → Plan + Predicted Oil (+ summary fan-out)
#  - Actual → Deviations + Oil tips + Grade + Chart
#  - Auto-learning on every Actual
#  - 12h reminder + hourly nudges until Actual arrives
#  - Daily summary 21:35 IST
#  - Robust parsing & clear status rules
# ======================================================

import os, re, io, json, math, logging
from datetime import datetime, timedelta, time
from zoneinfo import ZoneInfo

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from telegram import Update, InputFile
from telegram.ext import (
    Application, CommandHandler, MessageHandler,
    ContextTypes, filters
)

# ── Logging ───────────────────────────────────────────
LOG = logging.getLogger("pyro_bot")
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)s | %(name)s | %(message)s"
)

# ── Config via env ────────────────────────────────────
BOT_TZ = ZoneInfo("Asia/Kolkata")
TELEGRAM_BOT_TOKEN = os.environ.get("TELEGRAM_BOT_TOKEN", "")
REPORT_PATH = os.environ.get("REPORT_PATH", "pyrolysis_feed_temp_ZONE_TIME_report.xlsx")
ZONE_SHEET = os.environ.get("ZONE_SHEET", "ZoneTime_Recommendations")

# Summary group (optional)
SUMMARY_CHAT_ID = int(os.environ.get("SUMMARY_CHAT_ID", "0") or "0")

# Machine map JSON: {"-100111": "Machine 1296 (R-1)", ...}
try:
    MACHINE_MAP = json.loads(os.environ.get("MACHINE_MAP", "{}"))
except Exception:
    MACHINE_MAP = {}

# ── Persistent state on disk ─────────────────────────
STATE_FILE = "bot_state.json"
state = {
    "weights": {  # per-kg base oil fraction priors (learned)
        "radial": 0.46, "nylon": 0.42, "chips": 0.46, "powder": 0.53,
        "kachra": 0.40, "others": 0.40
    },
    "latest_feed": {},     # chat_id(str) -> {ts, batch, operator, date_str, feed, plan}
    "last_actual_ts": {},  # chat_id(str) -> iso timestamp
    "reminders": {}        # key(chat:batch) -> {"chat_id": int, "batch": str}
}

def load_state():
    global state
    try:
        if os.path.exists(STATE_FILE):
            with open(STATE_FILE, "r", encoding="utf-8") as f:
                obj = json.load(f)
            # merge shallowly to keep defaults if keys missing
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

# ── Excel zone rules ─────────────────────────────────
def load_zone_rules(path=REPORT_PATH, sheet=ZONE_SHEET):
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
        LOG.info(f"Loaded {len(rules)} zone rules from Excel.")
    except Exception as e:
        LOG.warning(f"⚠️ Failed to load Excel rules: {e} — using safe defaults.")
        rules = {
            "50-200 reactor": 165, "200-300 reactor": 68, "300-400 reactor": 185,
            "300-400 separator": 175, "400-450 reactor": 75, "450-480 reactor": 20,
            "480-500 reactor": 0, "500-520 reactor": 0
        }
    return rules

ZONE_RULES = load_zone_rules()

# ── Small helpers ────────────────────────────────────
def klabel(chat_id: int) -> str:
    return MACHINE_MAP.get(str(chat_id)) or f"{chat_id}"

def to_hhmmss(minutes: float | int | None) -> str:
    if minutes is None or (isinstance(minutes, float) and math.isnan(minutes)):
        return "00:00:00"
    total = int(round(float(minutes) * 60))
    h, rem = divmod(total, 3600)
    m, s = divmod(rem, 60)
    return f"{h:02d}:{m:02d}:{s:02d}"

def hhmmss_to_minutes(s: str) -> float:
    s = s.strip()
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

def norm_key(s: str) -> str:
    return re.sub(r"[^a-z0-9\-]", "", s.lower())

def _norm_date(s: str | None) -> str:
    """Normalize various date inputs and include weekday, IST."""
    if not s:
        now = datetime.now(BOT_TZ)
        return now.strftime("%d-%b-%Y (%A)")
    s = s.strip()
    # common formats
    for fmt in ("%d-%m-%Y", "%d.%m.%Y", "%d/%m/%Y", "%d-%b-%Y", "%d.%b.%Y", "%d/%b/%Y",
                "%Y-%m-%d", "%Y/%m/%d"):
        try:
            dt = datetime.strptime(s, fmt).replace(tzinfo=BOT_TZ)
            return dt.strftime("%d-%b-%Y (%A)")
        except Exception:
            continue
    # try ISO
    try:
        dt = datetime.fromisoformat(s)
        if dt.tzinfo is None:
            dt = dt.replace(tzinfo=BOT_TZ)
        return dt.astimezone(BOT_TZ).strftime("%d-%b-%Y (%A)")
    except Exception:
        return s

def batch_key(chat_id: int, batch: str | int | None) -> str:
    return f"{chat_id}:{batch or ''}"

def pretty_plan(plan: dict[str, float]) -> str:
    def keyer(z):  # sort by start temp
        m = re.match(r"(\d{2,3})-", z)
        return int(m.group(1)) if m else 999
    lines = []
    for z in sorted(plan.keys(), key=keyer):
        lines.append(f"{z}: {to_hhmmss(plan[z])}")
    return "\n".join(lines)

# ── Parsing ──────────────────────────────────────────
def parse_feed(text: str) -> dict:
    # Accept "Feed: Radial=5.1T, Nylon=0.6T, Chips=3.4T, Powder=1.5T, batch=92, operator=Ravi, date=.."
    t = re.sub(r"^/?(plan\s+)?feed\s*:\s*", "", text, flags=re.I).strip()
    out = {}
    for part in re.split(r"[,\n;]+", t):
        part = part.strip()
        if not part:
            continue
        if "=" in part:
            k, v = [x.strip() for x in part.split("=", 1)]
        else:
            m = re.match(r"([A-Za-z\-]+)\s+([\d.]+)\s*([tTkKgG]*)", part)
            if not m:
                continue
            k = m.group(1)
            v = f"{m.group(2)}{m.group(3)}"
        lk = norm_key(k)
        if lk in ("operator", "batch", "date"):
            out[lk] = v
            continue
        # value → kg
        v = v.replace("ton", "T")
        mT = re.match(r"([\d.]+)\s*[tT]\b", v)
        mK = re.match(r"([\d.]+)\s*[kK][gG]?\b", v)
        if mT:
            val = float(mT.group(1)) * 1000.0
        elif mK:
            val = float(mK.group(1))
        else:
            val = float(re.sub(r"[^\d.]", "", v) or 0.0)
        keymap = {
            "radial": "radial", "nylon": "nylon", "chips": "chips", "powder": "powder",
            "kachra": "kachra", "others": "others"
        }
        out[keymap.get(lk, lk)] = val
    # normalize date string for display
    if "date" in out:
        out["date"] = _norm_date(str(out["date"]))
    else:
        out["date"] = _norm_date(None)
    return out

def parse_actual(text: str) -> dict:
    # “Actual: 50-200=01:10, 200-300=00:45, 300-400=01:20, 400-450=00:22; oil=46.2; batch=92”
    t = re.sub(r"^/?actual\s*:\s*", "", text, flags=re.I).strip()
    out = {}
    for chunk in re.split(r"[;,]+", t):
        if "=" not in chunk:
            continue
        k, v = [x.strip() for x in chunk.split("=", 1)]
        lk = norm_key(k)
        if lk in ("oil", "oilyield", "oilpercent", "oilpct"):
            out["oil"] = float(re.sub(r"[^\d.]", "", v) or 0.0)
        elif lk == "batch":
            out["batch"] = v
        elif re.match(r"\d{2,3}\s*-\s*\d{2,3}", k):
            zkey = re.sub(r"\s+", "", k)
            out[zkey] = hhmmss_to_minutes(v)
    return out

# ── Recommendation Engine ───────────────────────────
class Reco:
    def __init__(self, base_rules: dict[str, float]):
        self.rules = base_rules.copy()

    def plan(self, feed: dict) -> dict[str, float]:
        plan = self.rules.copy()
        total = sum(feed.get(k, 0.0) for k in ("radial", "nylon", "chips", "powder", "kachra", "others"))
        if total > 0:
            radial = feed.get("radial", 0.0) / total
            nylon  = feed.get("nylon", 0.0) / total
            chips  = feed.get("chips", 0.0) / total
            # Nudge 300–400 zones (reactor + separator) by composition
            adj = 1.0 + 0.18 * (radial + 0.6 * chips - 0.6 * nylon)
            for z in list(plan.keys()):
                if z.startswith("300-400"):
                    plan[z] = max(20, plan[z] * adj)
                elif z.startswith("200-300"):
                    plan[z] = max(15, plan[z] * (1.0 + 0.10 * radial - 0.05 * nylon))
        return plan

ENGINE = Reco(ZONE_RULES)

# ── Yield model (tiny & online-learned) ──────────────
def predict_yield(feed: dict) -> tuple[float, float]:
    total = sum(feed.get(k, 0.0) for k in ("radial", "nylon", "chips", "powder", "kachra", "others"))
    if total <= 0:
        return (0.0, 0.60)
    w = state["weights"]
    base = 0.0
    for k, kg in feed.items():
        if k in w:
            base += (kg / total) * (w[k] * 100.0)
    # Guard rails
    pred = max(30.0, min(55.0, base))
    # Confidence: balanced mixes → higher
    ratios = [feed.get(x, 0.0) / total for x in ("radial", "nylon", "chips", "powder") if total > 0]
    conf = max(0.60, min(0.95, 0.95 - 0.8 * np.std(ratios)))
    return (round(pred, 2), round(conf, 2))

def learn_from_actual(feed: dict, actual_oil_pct: float):
    total = sum(feed.get(k, 0.0) for k in ("radial", "nylon", "chips", "powder", "kachra", "others"))
    if total <= 0:
        return
    pred, _ = predict_yield(feed)
    err = (actual_oil_pct - pred) / 100.0
    for k in ("radial", "nylon", "chips", "powder", "kachra", "others"):
        share = feed.get(k, 0.0) / total
        state["weights"][k] += 0.01 * err * share  # gentle learning
    save_state()

def grade_from_deviation(oil_delta: float) -> tuple[str, str]:
    """Return emoji + grade string based on oil delta vs prediction."""
    if oil_delta >= +1.0:
        return ("🟢", "A (above prediction)")
    if oil_delta >= -1.0:
        return ("🟡", "B (near prediction)")
    return ("🔴", "C (below prediction)")

def oil_tips(feed: dict, plan: dict, actual: dict, actual_oil: float, pred_oil: float) -> list[str]:
    """Actionable hints: look at key zones & separator."""
    tips = []
    # Compare minutes vs plan for major drivers
    def get(z): return actual.get(z.replace(" ", ""), None)
    d_300r = (get("300-400 reactor") or 0) - plan.get("300-400 reactor", 0)
    d_300s = (get("300-400 separator") or 0) - plan.get("300-400 separator", 0)
    d_200r = (get("200-300 reactor") or 0) - plan.get("200-300 reactor", 0)

    oil_delta = actual_oil - pred_oil

    if oil_delta < -1.0:
        # Below predicted → likely under-condensation or over-cracking
        if d_300s < -10:
            tips.append("Increase *300–400 separator* by ~15–25 min (more condensation).")
        if d_300r < -15:
            tips.append("Extend *300–400 reactor* by ~20–30 min for heavier mix completion.")
        if d_200r < -10:
            tips.append("Slightly extend *200–300 reactor* by ~10–15 min (stabilize ramp).")
        tips.append("Check line losses & condenser ∆T; avoid sharp temp spikes.")
    elif oil_delta > +1.0:
        tips.append("Oil above prediction — consider trimming *300–400 reactor* by ~10–20 min next time to reduce cycle time.")
    else:
        tips.append("Near prediction — fine tune separator by ±10 min to chase +0.5–1.0% oil.")

    return tips

# ── Chart builders ───────────────────────────────────
def chart_plan_vs_actual(plan: dict[str, float], actual: dict[str, float] | None = None) -> io.BytesIO:
    zones = []
    for z in sorted(plan.keys(), key=lambda s: int(s.split("-")[0])):
        zones.append(z)
    pmins = [plan[z] for z in zones]
    amins = [ (actual or {}).get(z.replace(" ", ""), np.nan) for z in zones ]

    x = np.arange(len(zones))
    fig, ax = plt.subplots(figsize=(10, 3), dpi=170)
    ax.bar(x, pmins, width=0.35, label="Plan (min)")
    if actual:
        ax.bar(x + 0.35, amins, width=0.35, label="Actual (min)")
    ax.set_xticks(x + (0.35/2 if actual else 0), zones, rotation=22)
    ax.set_ylabel("Minutes")
    ax.legend()
    ax.grid(True, axis="y", alpha=0.25)
    buf = io.BytesIO()
    plt.tight_layout()
    plt.savefig(buf, format="png")
    plt.close(fig)
    buf.seek(0)
    return buf

# ── Messaging helpers ────────────────────────────────
async def fanout_summary(context: ContextTypes.DEFAULT_TYPE, text: str):
    if SUMMARY_CHAT_ID:
        try:
            await context.bot.send_message(SUMMARY_CHAT_ID, text, parse_mode="Markdown")
        except Exception:
            pass

# ── Handlers ─────────────────────────────────────────
HELP = (
    "*Commands*\n"
    "• Send *Feed:* `Feed: Radial=5.1T, Nylon=0.6T, Chips=3.4T, Powder=1.5T, batch=92, operator=Ravi, date=09-11-2025`\n"
    "  → Plan + Predicted oil (+ posted in Summary).\n"
    "• Send *Actual:* `Actual: 50-200=01:14, 200-300=01:06, 300-400=02:07, 400-450=01:10, 450-480=00:32, oil=40.7, batch=92`\n"
    "  → Deviations + *Recommendations* + chart (+ model learns).\n"
    "• `/status` → Machine status (in Summary it lists all machines).\n"
    "• `/reload` → Reload Excel rules.\n"
    "• `/id` → Show this chat’s id and label.\n"
)

async def start_cmd(update: Update, context: ContextTypes.DEFAULT_TYPE):
    await update.message.reply_text("PyroVision Assistant ready ✅\n\n" + HELP, parse_mode="Markdown")

async def help_cmd(update: Update, context: ContextTypes.DEFAULT_TYPE):
    await update.message.reply_text(HELP, parse_mode="Markdown")

async def id_cmd(update: Update, context: ContextTypes.DEFAULT_TYPE):
    cid = update.effective_chat.id
    await update.message.reply_text(f"chat_id: `{cid}`\nname/title: *{klabel(cid)}*", parse_mode="Markdown")

async def reload_cmd(update: Update, context: ContextTypes.DEFAULT_TYPE):
    global ZONE_RULES, ENGINE
    ZONE_RULES = load_zone_rules()
    ENGINE = Reco(ZONE_RULES)
    await update.message.reply_text("🔁 Reloaded Excel rules.")

def remember_feed(chat_id: int, feed: dict, plan: dict):
    entry = {
        "ts": datetime.now(BOT_TZ).isoformat(),
        "batch": str(feed.get("batch", "")),
        "operator": str(feed.get("operator", "")),
        "date_str": feed.get("date", _norm_date(None)),
        "feed": feed,
        "plan": plan
    }
    state["latest_feed"][str(chat_id)] = entry
    save_state()

def schedule_reminders(jobq, chat_id: int, batch: str):
    """12h reminder, then hourly nudges until cleared by /actual."""
    # First cancel any existing for same key
    key = f"rem:{chat_id}:{batch}"
    for j in jobq.jobs():
        if j.name == key or j.name.startswith(key + ":nudge"):
            j.schedule_removal()

    # 12h one-shot
    jobq.run_once(
        reminder_once,
        when=timedelta(hours=12),
        data={"chat_id": chat_id, "batch": batch},
        name=key
    )
    # Hourly repeating (starts after 13h)
    jobq.run_repeating(
        reminder_nudge,
        interval=3600,
        first=timedelta(hours=13),
        data={"chat_id": chat_id, "batch": batch},
        name=key + ":nudge"
    )

async def reminder_once(context: ContextTypes.DEFAULT_TYPE):
    chat_id = context.job.data["chat_id"]
    batch = context.job.data["batch"]
    txt = (f"⚠️ *Reminder:* Actual data not entered yet for *Batch {batch}* "
           f"on *{klabel(chat_id)}*. Please send `Actual: …`")
    await context.bot.send_message(chat_id, txt, parse_mode="Markdown")
    await fanout_summary(context, f"🔔 {txt}")

async def reminder_nudge(context: ContextTypes.DEFAULT_TYPE):
    chat_id = context.job.data["chat_id"]
    batch = context.job.data["batch"]
    # stop nudging if actual already logged
    lf = state["latest_feed"].get(str(chat_id))
    if not lf or str(lf.get("batch")) != str(batch):
        context.job.schedule_removal()
        return
    last_act = state["last_actual_ts"].get(str(chat_id))
    if last_act and datetime.fromisoformat(last_act) >= datetime.fromisoformat(lf["ts"]):
        context.job.schedule_removal()
        return
    txt = (f"⏰ *Hourly nudge:* Still waiting for *Actual* of Batch {batch} on *{klabel(chat_id)}*.")
    await context.bot.send_message(chat_id, txt, parse_mode="Markdown")
    await fanout_summary(context, f"🔔 {txt}")

def status_line_for_chat(cid: int) -> str:
    now = datetime.now(BOT_TZ)
    lf = state["latest_feed"].get(str(cid))
    if not lf:
        return f"{klabel(cid)}: Idle"
    feed_ts = datetime.fromisoformat(lf["ts"])
    since = (now - feed_ts).total_seconds() / 3600.0
    last_act_iso = state["last_actual_ts"].get(str(cid))
    completed = False
    if last_act_iso:
        completed = datetime.fromisoformat(last_act_iso) >= feed_ts
    if not completed and since <= 12.0:
        return f"{klabel(cid)}: Running (batch {lf.get('batch','?')})"
    # completed or >12h
    date_str = lf.get("date_str", _norm_date(None))
    return f"{klabel(cid)}: Completed (last batch {lf.get('batch','?')} on {date_str})"

async def status_cmd(update: Update, context: ContextTypes.DEFAULT_TYPE):
    if SUMMARY_CHAT_ID and update.effective_chat.id == SUMMARY_CHAT_ID:
        lines = ["🟢 *Machine Status*"]
        for cid_str in MACHINE_MAP.keys():
            lines.append("• " + status_line_for_chat(int(cid_str)))
        await update.message.reply_text("\n".join(lines), parse_mode="Markdown")
    else:
        await update.message.reply_text(status_line_for_chat(update.effective_chat.id), parse_mode="Markdown")

async def plan_from_feed(update: Update, context: ContextTypes.DEFAULT_TYPE, feed_text: str):
    chat_id = update.effective_chat.id
    feed = parse_feed(feed_text)
    plan = ENGINE.plan(feed)
    remember_feed(chat_id, feed, plan)

    pred, conf = predict_yield(feed)
    label = klabel(chat_id)
    batch = str(feed.get("batch", "—"))
    oper  = str(feed.get("operator", "—"))
    date_str = feed.get("date", _norm_date(None))

    msg = []
    msg.append(f"📒 *{label}* — *Batch* {batch}, *Operator* {oper}")
    msg.append(f"• *Date* {date_str}")
    msg.append(f"🛢️ *Feed:* Radial {feed.get('radial',0)/1000:.2f}T, Nylon {feed.get('nylon',0)/1000:.2f}T, "
               f"Chips {feed.get('chips',0)/1000:.2f}T, Powder {feed.get('powder',0)/1000:.2f}T")
    msg.append(f"📈 *Predicted Oil:* *{pred:.2f}%*  (confidence {conf:.2f})")
    msg.append("\n*Recommended zone minutes (plan):*\n" + pretty_plan(plan))
    text = "\n".join(msg)

    await update.message.reply_text(text, parse_mode="Markdown")
    await fanout_summary(context, text)

    # Schedule reminders if batch present
    if str(feed.get("batch", "")).strip():
        schedule_reminders(context.job_queue, chat_id, str(feed["batch"]))

async def actual_from_text(update: Update, context: ContextTypes.DEFAULT_TYPE, actual_text: str):
    chat_id = update.effective_chat.id
    lf = state["latest_feed"].get(str(chat_id))
    if not lf:
        await update.message.reply_text("I don't see a recent *Feed* for this machine. Send `Feed: …` first.", parse_mode="Markdown")
        return

    actual = parse_actual(actual_text)
    plan = lf.get("plan", {})
    feed = lf.get("feed", {})
    # collect deviations
    deltas = {}
    for z, pmin in plan.items():
        zkey = z.replace(" ", "")
        if zkey in actual:
            deltas[z] = actual[zkey] - pmin

    # Learn from oil % (if given)
    lines = []
    oil_line = None
    if "oil" in actual:
        pred, _ = predict_yield(feed)
        learn_from_actual(feed, float(actual["oil"]))
        state["last_actual_ts"][str(chat_id)] = datetime.now(BOT_TZ).isoformat()
        save_state()

        oil_delta = float(actual["oil"]) - pred
        emoji, grade = grade_from_deviation(oil_delta)
        tips = oil_tips(feed, plan, actual, float(actual["oil"]), pred)

        oil_line = (f"{emoji} *Oil vs Predicted:* actual *{actual['oil']:.2f}%* "
                    f"(pred *{pred:.2f}%*, Δ = {oil_delta:+.2f}%) → *Grade {grade}*")
        lines.append(oil_line)
        if tips:
            lines.append("🔧 *Oil yield recommendations:*")
            for t in tips:
                lines.append(f"• {t}")

    # Deviation table
    lines.append("\n📊 *Deviation vs plan (min):*")
    for z in sorted(deltas.keys(), key=lambda s: int(s.split("-")[0])):
        lines.append(f"{z}: {deltas[z]:+.0f}")

    if not oil_line:
        lines.append("\nℹ️ Add `oil=..` in Actual to get yield grade & tips.")

    txt = "\n".join(lines)
    await update.message.reply_text(txt, parse_mode="Markdown")

    # Chart
    buf = chart_plan_vs_actual(plan, {k.replace(" ", ""): v for k, v in actual.items() if "-" in k})
    await context.bot.send_photo(chat_id, InputFile(buf, filename="plan_vs_actual.png"))

    # Notify summary
    await fanout_summary(context, f"🧾 Actual logged for *{klabel(chat_id)}* (batch {lf.get('batch','?')}).")

# Text router (accept plain "Feed:" and "Actual:")
async def text_router(update: Update, context: ContextTypes.DEFAULT_TYPE):
    text = (update.message.text or "").strip()
    if re.match(r"(?i)^feed\s*:", text) or re.match(r"(?i)^plan\s*feed\s*:", text):
        await plan_from_feed(update, context, text)
    elif re.match(r"(?i)^actual\s*:", text):
        await actual_from_text(update, context, text)
    elif re.match(r"(?i)^status$", text):
        await status_cmd(update, context)
    elif re.match(r"(?i)^/id$", text):
        await id_cmd(update, context)
    elif re.match(r"(?i)^/reload$", text):
        await reload_cmd(update, context)
    elif re.match(r"(?i)^/help$", text) or re.match(r"(?i)^help$", text):
        await help_cmd(update, context)
    else:
        await update.message.reply_text("I didn’t understand.\n\n" + HELP, parse_mode="Markdown")

# ── Daily summary 21:35 IST ─────────────────────────
async def daily_summary(context: ContextTypes.DEFAULT_TYPE):
    now = datetime.now(BOT_TZ)
    title = now.strftime("🧾 *Daily Summary* — %d-%b-%Y (%A) · %H:%M IST")
    lines = [title]
    for cid_str, label in MACHINE_MAP.items():
        lines.append("• " + status_line_for_chat(int(cid_str)))
    if SUMMARY_CHAT_ID:
        await context.bot.send_message(SUMMARY_CHAT_ID, "\n".join(lines), parse_mode="Markdown")

def schedule_daily_summary(app: Application):
    # Every day at 21:35 IST
    app.job_queue.run_daily(
        daily_summary,
        time=time(hour=21, minute=35, tzinfo=BOT_TZ),
        name="daily_summary"
    )

# ── Main ─────────────────────────────────────────────
def main():
    LOG.info("🚀 Starting PyroVision Assistant…")
    if not TELEGRAM_BOT_TOKEN:
        raise RuntimeError("Missing TELEGRAM_BOT_TOKEN")
    load_state()

    app = Application.builder().token(TELEGRAM_BOT_TOKEN).build()

    # Commands
    app.add_handler(CommandHandler("start", start_cmd))
    app.add_handler(CommandHandler("help", help_cmd))
    app.add_handler(CommandHandler("reload", reload_cmd))
    app.add_handler(CommandHandler("status", status_cmd))
    app.add_handler(CommandHandler("id", id_cmd))

    # Text router for plain messages
    app.add_handler(MessageHandler(filters.TEXT & ~filters.COMMAND, text_router))

    # Jobs
    schedule_daily_summary(app)

    LOG.info("✅ Bot ready. Polling…")
    app.run_polling(close_loop=False)

if __name__ == "__main__":
    main()
