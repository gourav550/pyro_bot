# ======================================================
#  PYROVISION BOT – CLEAN HEADER (NO INDENTATION ERRORS)
# ======================================================

import os, json, re, logging
from datetime import time
import pandas as pd
import numpy as np
from telegram import Update
from telegram.ext import Application, CommandHandler, MessageHandler, ContextTypes, filters

LOG = logging.getLogger("pyro_bot")
logging.basicConfig(level=logging.INFO, format="%(asctime)s | %(levelname)s | %(name)s | %(message)s")

TELEGRAM_BOT_TOKEN = os.environ.get("TELEGRAM_BOT_TOKEN", "")
REPORT_PATH = os.environ.get("REPORT_PATH", "pyrolysis_feed_temp_ZONE_TIME_report.xlsx")
ZONE_SHEET = "ZoneTime_Recommendations"

STATE_FILE = "bot_state.json"
state = {}

# ------------------ STATE HANDLING ------------------
def load_state():
    """Load persistent state from disk (if any)."""
    global state
    try:
        if os.path.exists(STATE_FILE):
            with open(STATE_FILE, "r", encoding="utf-8") as f:
                state = json.load(f)
            LOG.info("✅ State loaded successfully.")
        else:
            LOG.info("ℹ️ No state file found, starting fresh.")
    except Exception as e:
        LOG.warning(f"⚠️ Could not load state: {e}")

def save_state():
    """Save runtime state to disk."""
    try:
        with open(STATE_FILE, "w", encoding="utf-8") as f:
            json.dump(state, f, ensure_ascii=False, indent=2)
    except Exception as e:
        LOG.warning(f"⚠️ Could not save state: {e}")

# ------------------ ZONE RULES ------------------
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
        LOG.info(f"Loaded {len(rules)} zone rules.")
    except Exception as e:
        LOG.warning(f"⚠️ Failed to load Excel rules: {e}")
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

# ------------------ TELEGRAM CORE ------------------
async def start(update: Update, context: ContextTypes.DEFAULT_TYPE):
    await update.message.reply_text("✅ PyroVision Bot ready.\nType /help for commands.")

async def help_command(update: Update, context: ContextTypes.DEFAULT_TYPE):
    await update.message.reply_text("/plan /actual /reload /status /id")

async def text_router(update: Update, context: ContextTypes.DEFAULT_TYPE):
    await update.message.reply_text("Message received, processing...")

# ------------------ MAIN FUNCTION ------------------
def main():
    LOG.info("🚀 Starting PyroVision Assistant…")
    load_state()

    if not TELEGRAM_BOT_TOKEN:
        raise RuntimeError("❌ Missing TELEGRAM_BOT_TOKEN environment variable!")

    app = Application.builder().token(TELEGRAM_BOT_TOKEN).build()
    app.add_handler(CommandHandler("start", start))
    app.add_handler(CommandHandler("help", help_command))
    app.add_handler(MessageHandler(filters.TEXT & ~filters.COMMAND, text_router))

    LOG.info("✅ Application started successfully.")
    app.run_polling()

if __name__ == "__main__":
    main()

 to_hhmmss(minutes: float|int|None) -> str:
    if minutes is None or (isinstance(minutes,float) and math.isnan(minutes)):
        return "-"
    total = int(round(float(minutes)*60))
    h, rem = divmod(total, 3600)
    m, s  = divmod(rem, 60)
    return f"{h:02d}:{m:02d}:{s:02d}"

def hhmmss_to_minutes(s: str) -> float:
    s = s.strip()
    if ":" not in s:
        return float(s)
    parts = list(map(int, s.split(":")))
    if len(parts)==3:
        h,m,sec = parts
        return h*60 + m + sec/60
    if len(parts)==2:
        m,sec = parts
        return m + sec/60
    return 0.0

def parse_feed(text: str) -> dict:
    # Accept “Feed: Radial=5.1T, Nylon=0.6T, Chips=3.4T, Powder=1.5T, batch=92, operator=Ravi”
    data = {}
    t = re.sub(r"^/?(plan|predict)?\s*feed\s*:\s*|^feed\s*:\s*", "", text, flags=re.I).strip()
    for part in re.split(r"[,\n;]+", t):
        if not part.strip(): continue
        if "=" in part:
            k,v = [x.strip() for x in part.split("=",1)]
        else:
            # "Radial 5T" form
            m = re.match(r"([A-Za-z]+)\s+([\d\.]+)", part.strip())
            if not m: continue
            k,v = m.group(1), m.group(2)
        key = norm_key(k)
        # handle units
        v = v.replace("ton","T").replace("mt","T").replace("kg","K").replace("t","T")
        if key in ("batch","operator","machine","date"):
            data[key] = v
            continue
        # numeric
        mT = re.match(r"([\d\.]+)\s*[tT]$", v)
        mK = re.match(r"([\d\.]+)\s*[kK][gG]?$", v)
        if mT:
            val = float(mT.group(1))*1000.0
        elif mK:
            val = float(mK.group(1))
        else:
            val = float(re.sub(r"[^\d\.]","", v) or 0.0)
            # assume kilograms if big, else tonnes? keep as kg
        data[key] = val
    # normalize keys
    mapping = {
        "radial":"radial","nylon":"nylon","chips":"chips","powder":"powder",
        "kachra":"kachra","others":"others"
    }
    out = {mapping.get(k,k):v for k,v in data.items()}
    return out

def parse_actual(text: str) -> dict:
    # “Actual: 50-200=01:10, 200-300=00:45, 300-400=01:20, 400-450=00:22; oil=46.2; batch=88”
    t = re.sub(r"^/?actual\s*:?","", text, flags=re.I).strip()
    out = {}
    for chunk in re.split(r"[;,]+", t):
        if "=" not in chunk: continue
        k,v = [x.strip() for x in chunk.split("=",1)]
        lk = norm_key(k)
        if lk=="oil" or lk=="oil%":
            out["oil"] = float(re.sub(r"[^\d\.]","", v) or 0.0)
        elif re.match(r"\d{2,3}\s*-\s*\d{2,3}", lk):
            # zone
            out[lk.replace(" ","")] = hhmmss_to_minutes(v)
        elif lk=="batch":
            out["batch"] = v
    return out

# ── Recommendation Engine (Excel-driven + simple heuristics) ──────────────────
class RecoEngine:
    def __init__(self, report_path: str):
        self.report_path = report_path
        self.zone_defaults: dict[str,float] = {}   # "50-200 reactor" etc.
        self._load()

    def _load(self):
        self.zone_defaults.clear()
        try:
            df = pd.read_excel(self.report_path, sheet_name="ZoneTime_Recommendations")
            for _, r in df.iterrows():
                feat = str(r.get("zone_time_feature","")).lower()
                mins = r.get("suggested_minutes", np.nan)
                if not feat or pd.isna(mins): continue
                m = re.search(r"(\d{2,3}\s*[-–]\s*\d{2,3})", feat)
                if not m: continue
                window = m.group(1).replace(" ","").replace("–","-")
                which = "separator" if "separator" in feat else "reactor"
                self.zone_defaults[f"{window} {which}"] = float(mins)
            log.info("Loaded %d zone rules from %s", len(self.zone_defaults), self.report_path)
        except Exception as e:
            log.warning("Excel load failed (%s). Falling back to conservative defaults.", e)
            self.zone_defaults.update({
                "50-200 reactor": 60, "200-300 reactor": 60, "300-400 reactor": 75,
                "400-450 reactor": 70, "450-480 reactor": 25, "480-500 reactor": 15,
                "300-400 separator": 30
            })

    def plan(self, feed: dict) -> dict[str,float]:
        # base plan from defaults
        plan = dict(self.zone_defaults)
        # very light adjustment from composition (more radial/chips → extend 300–400)
        total = sum(feed.get(k,0.0) for k in ("radial","nylon","chips","powder","kachra","others"))
        if total > 0:
            radial_ratio = feed.get("radial",0.0)/total
            chips_ratio  = feed.get("chips",0.0)/total
            nylon_ratio  = feed.get("nylon",0.0)/total
            # ±15% window on key zones
            adj = 1.0 + 0.15*(radial_ratio + 0.5*chips_ratio - 0.6*nylon_ratio)
            for k in list(plan.keys()):
                if k.startswith("300-400"):
                    plan[k] = max(20, plan[k]*adj)
                if k.startswith("200-300"):
                    plan[k] = max(15, plan[k]*(1.0 + 0.1*radial_ratio - 0.05*nylon_ratio))
        return plan

engine = RecoEngine(REPORT_PATH)

# ── Yield prediction (very simple, learns from /actual) ───────────────────────
def predict_yield(feed: dict) -> tuple[float,float]:
    """Return (predicted_oil_percent, confidence 0..1)."""
    total = sum(feed.get(k,0.0) for k in ("radial","nylon","chips","powder","kachra","others"))
    if total <= 0: return (0.0, 0.2)
    w = state["weights"]
    # weighted kg → % (divide by total kg)
    base = 0.0
    for k,kg in feed.items():
        if k in w:
            base += (kg/total) * (w[k]*100.0)  # convert to %
    # clamp reasonable range
    pred = max(30.0, min(55.0, base))
    # confidence grows with how similar this mix is to what we’ve seen
    # (proxy: more total kg + not too extreme ratios)
    ratio_spread = np.std([feed.get(x,0.0)/total for x in ("radial","nylon","chips","powder")])
    conf = max(0.55, min(0.95, 0.95 - 0.8*ratio_spread))
    return (round(pred,2), round(conf,2))

def learn_from_actual(feed: dict, actual_oil: float):
    """Tiny learning step: move weights toward explaining the actual."""
    total = sum(feed.get(k,0.0) for k in ("radial","nylon","chips","powder","kachra","others"))
    if total <= 0: return
    pred,_ = predict_yield(feed)
    err = (actual_oil - pred)/100.0
    # nudge weights in the direction of error, proportional to share
    for k in ("radial","nylon","chips","powder","kachra","others"):
        share = feed.get(k,0.0)/total
        state["weights"][k] += 0.01 * err * share   # small learning rate
    save_state()

# ── Rendering helpers ─────────────────────────────────────────────────────────
def pretty_plan(plan: dict[str,float]) -> str:
    def k(x):
        m = re.match(r"(\d{2,3})-(\d{2,3})", x)
        if not m: return (999,999,x)
        return (int(m.group(1)), int(m.group(2)), x)
    lines = []
    for z in sorted(plan.keys(), key=k):
        lines.append(f"{z}: {to_hhmmss(plan[z])}")
    return "\n".join(lines)

def route_summary(context: ContextTypes.DEFAULT_TYPE, text: str):
    if SUMMARY_CHAT_ID:
        asyncio.create_task(context.bot.send_message(SUMMARY_CHAT_ID, text))

def machine_label(chat_id: int) -> str:
    return MACHINE_MAP.get(str(chat_id)) or MACHINE_MAP.get(str(int(chat_id))) or f"{chat_id}"

# ── Graph generation ──────────────────────────────────────────────────────────
def chart_from_csv(df: pd.DataFrame) -> io.BytesIO:
    # Expect columns: ts (ISO or time), Tr, Ts, Pr, Ps (any subset ok)
    fig, ax1 = plt.subplots(figsize=(12,4), dpi=160)
    t = pd.to_datetime(df.get("ts") or df.index)
    if "Tr" in df: ax1.plot(t, df["Tr"], label="Temperature (Tr)")
    if "Ts" in df: ax1.plot(t, df["Ts"], label="Temperature (Ts)")
    ax1.set_ylabel("°C")
    ax2 = ax1.twinx()
    if "Pr" in df: ax2.plot(t, df["Pr"], linestyle="--", label="Pressure (Pr)", alpha=0.6)
    if "Ps" in df: ax2.plot(t, df["Ps"], linestyle="--", label="Pressure (Ps)", alpha=0.6)
    ax2.set_ylabel("Bar")
    ax1.set_xlabel("Time")
    lines, labels = [], []
    for ax in (ax1, ax2):
        l = ax.get_lines()
        lines += l
        labels += [ln.get_label() for ln in l]
    ax1.legend(lines, labels, loc="upper left")
    ax1.grid(True, alpha=0.2)
    buf = io.BytesIO()
    plt.tight_layout()
    plt.savefig(buf, format="png")
    plt.close(fig)
    buf.seek(0)
    return buf

def chart_from_plan_vs_actual(plan: dict[str,float], actual: dict[str,float]|None=None) -> io.BytesIO:
    zones = []
    for z in sorted(plan.keys(), key=lambda s: int(s.split("-")[0])):
        zones.append(z)
    pmins = [plan[z] for z in zones]
    amins = [actual.get(z, np.nan) if actual else np.nan for z in zones]
    x = np.arange(len(zones))
    width = 0.38
    fig, ax = plt.subplots(figsize=(10,3), dpi=160)
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

# ── Job helpers (reminders & daily summary) ───────────────────────────────────
def batch_key(chat_id: int, batch: str|int|None) -> str:
    return f"{chat_id}:{batch or ''}"

async def reminder_job(context: ContextTypes.DEFAULT_TYPE):
    data = context.job.data or {}
    chat_id = data.get("chat_id")
    batch   = data.get("batch")
    key = batch_key(chat_id, batch)
    # If still pending, ping machine + summary
    if key in state["reminders"]:
        txt = (f"⚠️ *Reminder:* Actual data not entered yet for *Batch {batch}* "
               f"on *{machine_label(chat_id)}*. Please send `/actual …`.")
        await context.bot.send_message(chat_id, txt, parse_mode=ParseMode.MARKDOWN)
        route_summary(context, f"🔔 {txt}")
        # keep it (operator may enter later); or remove after one reminder:
        # del state["reminders"][key]; save_state()

async def daily_summary_job(context: ContextTypes.DEFAULT_TYPE):
    now = datetime.now(BOT_TZ)
    lines = [f"📊 *Daily Summary* — {now.strftime('%d-%b %H:%M')} (IST)"]
    for cid_str,label in MACHINE_MAP.items():
        cid = int(cid_str)
        lf = state["latest_feed"].get(cid_str)
        status = "Idle"
        if lf:
            last_feed_ts = datetime.fromisoformat(lf["ts"])
            hrs = (now - last_feed_ts.astimezone(BOT_TZ)).total_seconds()/3600
            # Completed if >11h since feed or last actual is after feed
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
    msg = "\n".join(lines)
    if SUMMARY_CHAT_ID:
        await context.bot.send_message(SUMMARY_CHAT_ID, msg, parse_mode=ParseMode.MARKDOWN)

def schedule_daily_summary(app):
    print("⏰ Daily summary scheduling is disabled (JobQueue not available).")


# ── Handlers ──────────────────────────────────────────────────────────────────
HELP = (
    "*Commands*\n"
    "• `Feed: Radial=5.1T, Nylon=0.6T, Chips=3.4T, Powder=1.5T, batch=92, operator=Ravi` → Plan + Predicted Oil\n"
    "• `/whatif Feed: …` → Predicted Oil + Confidence (no plan)\n"
    "• `/actual Actual: 50-200=1:10, 200-300=0:45, 300-400=1:20, 400-450=0:22; oil=46.2; batch=92` → Compare & Learn\n"
    "• `/chart` with CSV (ts,Tr,Ts,Pr,Ps) attached → Temperature/Pressure chart\n"
    "• `/chart planned` (or after /actual) → Plan vs Actual bar chart image\n"
    "• `/status` → Shows status of all machines (in Summary group) or current machine\n"
    "• `/reload` → Reload Excel rules\n"
)

async def start(update: Update, context: ContextTypes.DEFAULT_TYPE):
    await update.message.reply_text("PyroVision Assistant ready ✅\n\n" + HELP, parse_mode=ParseMode.MARKDOWN)

async def help_cmd(update: Update, context: ContextTypes.DEFAULT_TYPE):
    await update.message.reply_text(HELP, parse_mode=ParseMode.MARKDOWN)

async def reload_cmd(update: Update, context: ContextTypes.DEFAULT_TYPE):
    engine._load()
    await update.message.reply_text("🔁 Reloaded recommendations from Excel.")
    route_summary(context, f"🔁 Rules reloaded by {machine_label(update.effective_chat.id)}")

def remember_feed(chat_id: int, feed: dict, plan: dict):
    now = datetime.now(BOT_TZ)
    entry = {
        "ts": now.isoformat(),
        "batch": str(feed.get("batch","")),
        "operator": str(feed.get("operator","")),
        "feed": feed, "plan": plan
    }
    state["latest_feed"][str(chat_id)] = entry
    # schedule reminder if batch present
    batch = entry["batch"]
    if batch:
        key = batch_key(chat_id, batch)
        state["reminders"][key] = {"chat_id": chat_id, "batch": batch, "due": (now+timedelta(hours=12)).isoformat()}
    save_state()

async def plan_cmd_core(update: Update, context: ContextTypes.DEFAULT_TYPE, feed_text: str, predict_only=False):
    chat_id = update.effective_chat.id
    feed = parse_feed(feed_text)
    plan = engine.plan(feed)
    pred, conf = predict_yield(feed)
    label = machine_label(chat_id)
    batch = feed.get("batch","?")
    oper  = feed.get("operator","?")
    # Remember feed (and set reminder)
    remember_feed(chat_id, feed, plan)

    msg = []
    msg.append(f"🏷️ *{label}* — *Batch* {batch}, *Operator* {oper}")
    msg.append(f"🛢️ Feed: Radial {feed.get('radial',0)/1000:.2f}T, Nylon {feed.get('nylon',0)/1000:.2f}T, Chips {feed.get('chips',0)/1000:.2f}T, Powder {feed.get('powder',0)/1000:.2f}T")
    msg.append(f"📈 Predicted Oil: *{pred:.2f}%*  (confidence {conf:.2f})")
    if not predict_only:
        msg.append("\n*Recommended zone minutes (plan):*\n" + pretty_plan(plan))
    text = "\n".join(msg)
    await update.message.reply_text(text, parse_mode=ParseMode.MARKDOWN)
    route_summary(context, text)

async def plan_cmd(update: Update, context: ContextTypes.DEFAULT_TYPE):
    await plan_cmd_core(update, context, update.message.text, predict_only=False)

async def whatif_cmd(update: Update, context: ContextTypes.DEFAULT_TYPE):
    t = re.sub(r"^/?whatif\s*","", update.message.text, flags=re.I)
    await plan_cmd_core(update, context, t, predict_only=True)

async def actual_cmd(update: Update, context: ContextTypes.DEFAULT_TYPE):
    chat_id = update.effective_chat.id
    actual = parse_actual(update.message.text)
    # get last plan for this machine
    lf = state["latest_feed"].get(str(chat_id))
    if not lf:
        await update.message.reply_text("I don't find a recorded Feed for this machine. Send the Feed first.")
        return
    plan = lf.get("plan", {})
    deltas = {}
    tips = []
    for z,pmin in plan.items():
        zkey = z.replace(" ","")
        if zkey in actual:
            am = actual[zkey]
            deltas[z] = am - pmin
            if abs(deltas[z]) >= 5:
                tips.append(("reduce" if deltas[z]>0 else "increase", z, abs(deltas[z])))
    # learning if oil provided
    if "oil" in actual:
        learn_from_actual(lf.get("feed",{}), float(actual["oil"]))
        state["last_actual_ts"][str(chat_id)] = datetime.now(BOT_TZ).isoformat()
        # clear reminder if any
        key = batch_key(chat_id, lf.get("batch"))
        if key in state["reminders"]:
            del state["reminders"][key]
        save_state()

    # reply
    lines = [f"📊 Deviation vs plan (min):"]
    for z in sorted(deltas.keys(), key=lambda s: int(s.split("-")[0])):
        lines.append(f"{z}: {deltas[z]:+.0f}")
    if tips:
        lines.append("\n🛠️ Recommendations:")
        for d,z,mins in tips:
            lines.append(f"• {d} {z} by ~{mins:.0f} min")
    else:
        lines.append("\n✅ Near-optimal execution vs plan.")
    await update.message.reply_text("\n".join(lines))

    # also send plan vs actual bar chart
    buf = chart_from_plan_vs_actual(plan, {k.replace(" ",""):v for k,v in actual.items() if "-" in k})
    await context.bot.send_photo(chat_id, photo=InputFile(buf, filename="plan_vs_actual.png"))
    route_summary(context, f"🧾 Actual logged for {machine_label(chat_id)} (batch {lf.get('batch','?')}).")

async def status_cmd(update: Update, context: ContextTypes.DEFAULT_TYPE):
    now = datetime.now(BOT_TZ)
    # if in summary group → show all. otherwise show current machine
    if SUMMARY_CHAT_ID and update.effective_chat.id == SUMMARY_CHAT_ID:
        lines = ["🟢 *Machine Status*"]
        for cid_str,label in MACHINE_MAP.items():
            lf = state["latest_feed"].get(cid_str)
            st = "Idle"
            if lf:
                last_feed_ts = datetime.fromisoformat(lf["ts"])
                hrs = (now - last_feed_ts.astimezone(BOT_TZ)).total_seconds()/3600
                last_act_iso = state["last_actual_ts"].get(cid_str)
                completed = False
                if last_act_iso:
                    completed = datetime.fromisoformat(last_act_iso) >= last_feed_ts
                if hrs <= 11 and not completed:
                    st = f"Running (batch {lf.get('batch','?')})"
                elif completed:
                    st = f"Completed (batch {lf.get('batch','?')})"
                elif hrs > 12:
                    st = f"Overdue (no actual)"
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
            last_feed_ts = datetime.fromisoformat(lf["ts"])
            hrs = (now - last_feed_ts.astimezone(BOT_TZ)).total_seconds()/3600
            last_act_iso = state["last_actual_ts"].get(cid)
            completed = last_act_iso and (datetime.fromisoformat(last_act_iso) >= last_feed_ts)
            if hrs <= 11 and not completed:
                st = f"Running (batch {lf.get('batch','?')})"
            elif completed:
                st = f"Completed (batch {lf.get('batch','?')})"
            elif hrs > 12:
                st = "Overdue (no actual)"
        await update.message.reply_text(f"{label}: {st}")

async def chart_cmd(update: Update, context: ContextTypes.DEFAULT_TYPE):
    # If CSV attached → temp/pressure chart; else, plan vs actual bar chart for last batch
    doc = update.message.document
    if doc and (doc.file_name.lower().endswith(".csv")):
        f = await doc.get_file()
        bio = io.BytesIO()
        await f.download_to_memory(out=bio)
        bio.seek(0)
        df = pd.read_csv(bio)
        buf = chart_from_csv(df)
        await context.bot.send_photo(update.effective_chat.id, InputFile(buf, filename="tp_chart.png"))
        return
    # no CSV → show last plan (and actual if exists)
    lf = state["latest_feed"].get(str(update.effective_chat.id))
    if not lf:
        await update.message.reply_text("No recent plan to chart. Send a Feed first, or attach a CSV log.")
        return
    plan = lf.get("plan",{})
    # try to find last actual deltas (not stored); render plan only
    buf = chart_from_plan_vs_actual(plan, None)
    await context.bot.send_photo(update.effective_chat.id, InputFile(buf, filename="plan_chart.png"))

async def text_router(update: Update, context: ContextTypes.DEFAULT_TYPE):
    text = (update.message.text or "").strip()
    if re.match(r"(?i)^feed\s*:", text):
        await plan_cmd_core(update, context, text, predict_only=False)
        # schedule reminder job in 12h (if batch present)
        lf = state["latest_feed"].get(str(update.effective_chat.id))
        if lf and lf.get("batch"):
            due = datetime.now(BOT_TZ) + timedelta(hours=12)
            context.job_queue.run_once(
                reminder_job,
                when=due.astimezone(BOT_TZ),
                name=f"reminder:{batch_key(update.effective_chat.id, lf['batch'])}",
                data={"chat_id": update.effective_chat.id, "batch": lf["batch"]}
            )
    elif re.match(r"(?i)^actual\s*:", text):
        await actual_cmd(update, context)
    elif re.match(r"(?i)^status$", text):
        await status_cmd(update, context)
    else:
        await update.message.reply_text("I didn’t understand. Type /help for examples.")

# ── Main ──────────────────────────────────────────────────────────────────────
def main():
    print("🚀 Starting PyroVision Assistant...")
    load_state()

    app = Application.builder().token(TELEGRAM_BOT_TOKEN).build()

    app.add_handler(CommandHandler("start", start))
    app.add_handler(CommandHandler("help", help_command))
    app.add_handler(CommandHandler("plan", plan_feed))
    app.add_handler(CommandHandler("actual", actual_output))
    app.add_handler(CommandHandler("reload", reload_excel))
    app.add_handler(CommandHandler("status", status_report))
    app.add_handler(MessageHandler(filters.TEXT & ~filters.COMMAND, text_router))

    schedule_daily_summary(app)

    print("✅ Bot ready. Running polling loop.")
    app.run_polling()


