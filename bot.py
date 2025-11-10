# ======================================================
#  PyroVision Assistant – with Oil-Yield Recommendations
#  (paste entire file as bot.py)
# ======================================================

import os, re, io, json, math, logging
from datetime import datetime, timedelta, timezone
from typing import Dict, Tuple, List

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from telegram import Update, InputFile
from telegram.ext import (
    Application, CommandHandler, MessageHandler, ContextTypes, filters
)

# -------------------- Config & Logging --------------------
LOG = logging.getLogger("pyro_bot")
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)s | %(name)s | %(message)s"
)

BOT_TZ  = timezone(timedelta(hours=5, minutes=30))  # IST
TOKEN   = os.getenv("TELEGRAM_BOT_TOKEN", "")
REPORT  = os.getenv("REPORT_PATH", "pyrolysis_feed_temp_ZONE_TIME_report.xlsx")
ZONE_SHEET = os.getenv("ZONE_SHEET", "ZoneTime_Recommendations")

SUMMARY_CHAT_ID = int(os.getenv("SUMMARY_CHAT_ID", "0") or 0)
try:
    MACHINE_MAP: Dict[str, str] = json.loads(os.getenv("MACHINE_MAP", "{}"))
except Exception:
    MACHINE_MAP = {}

STATE_FILE = "bot_state.json"

# -------------------- Persistent state --------------------
state = {
    "weights": {  # rough starting oil potential by category (fraction of feed mass)
        "radial": 0.46, "nylon": 0.41, "chips": 0.46, "powder": 0.53, "kachra": 0.40, "others": 0.40
    },
    "latest_feed": {},      # chat_id -> {ts,batch,operator,feed,plan,date}
    "last_actual_ts": {},   # chat_id -> iso
}

def load_state():
    global state
    try:
        if os.path.exists(STATE_FILE):
            with open(STATE_FILE, "r", encoding="utf-8") as f:
                obj = json.load(f)
            # shallow merge to keep keys if code updated
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

# -------------------- Excel rules --------------------
def load_zone_rules(path=REPORT, sheet=ZONE_SHEET) -> Dict[str,float]:
    rules: Dict[str,float] = {}
    try:
        df = pd.read_excel(path, sheet_name=sheet)
        for _, row in df.iterrows():
            feat = str(row.get("zone_time_feature", "")).strip().lower()
            mins = row.get("suggested_minutes", np.nan)
            if not feat or pd.isna(mins):
                continue
            m = re.search(r"(\d{2,3})\s*[–-]\s*(\d{2,3})", feat)
            if not m:
                continue
            window = f"{m.group(1)}-{m.group(2)}"
            which  = "separator" if "separator" in feat else "reactor"
            rules[f"{window} {which}"] = float(mins)
        LOG.info("Loaded %d zone rules from Excel.", len(rules))
    except Exception as e:
        LOG.warning("Excel load failed (%s). Using defaults.", e)
        rules.update({
            "50-200 reactor": 165,  # minutes; conservative baselines
            "200-300 reactor": 70,
            "300-400 reactor": 185,
            "300-400 separator": 170,
            "400-450 reactor": 75,
            "450-480 reactor": 20,
            "480-500 reactor": 0,
            "500-520 reactor": 0,
        })
    return rules

ZONE_RULES = load_zone_rules()

# -------------------- Utilities & parsing --------------------
def to_hhmmss(minutes: float|int|None) -> str:
    if minutes is None or (isinstance(minutes, float) and math.isnan(minutes)):
        return "-"
    total = int(round(float(minutes) * 60))
    h, r  = divmod(total, 3600)
    m, s  = divmod(r, 60)
    return f"{h:02d}:{m:02d}:{s:02d}"

def hhmmss_to_minutes(s: str) -> float:
    s = s.strip()
    if ":" not in s:
        return float(re.sub(r"[^\d.]", "", s) or 0.0)
    p = list(map(int, s.split(":")))
    if len(p) == 3:
        h, m, sec = p;  return h*60 + m + sec/60
    if len(p) == 2:
        m, sec = p;     return m + sec/60
    return 0.0

def _norm_key(k: str) -> str:
    return re.sub(r"[^a-z0-9\-]", "", k.lower())

def _norm_date(s: str|None) -> str:
    """Return 'dd-Mon-YYYY (Weekday)' in IST. Accepts common dd/mm/yy etc. If empty: now."""
    if not s:
        now = datetime.now(BOT_TZ);  return now.strftime("%d-%b-%Y (%A)")
    s = s.strip()
    for fmt in ("%d-%m-%Y", "%d.%m.%Y", "%d/%m/%Y", "%d-%m-%y", "%d.%m.%y", "%d/%m/%y", "%Y-%m-%d"):
        try:
            dt = datetime.strptime(s, fmt).replace(tzinfo=BOT_TZ)
            return dt.strftime("%d-%b-%Y (%A)")
        except Exception:
            continue
    try:
        dt = datetime.fromisoformat(s)
        if dt.tzinfo is None: dt = dt.replace(tzinfo=BOT_TZ)
        return dt.astimezone(BOT_TZ).strftime("%d-%b-%Y (%A)")
    except Exception:
        return s

def parse_feed(text: str) -> Dict[str, float|str]:
    """
    Accepts:
      Feed: Radial=5.1T, Nylon=0.6T, Chips=3.4T, Powder=1.5T, batch=92, operator=Ravi, date=09-11-2025
    """
    payload = re.sub(r"^/?(plan|predict)?\s*feed\s*:\s*|^feed\s*:\s*", "", text, flags=re.I).strip()
    out: Dict[str, float|str] = {}
    for part in re.split(r"[,\n;]+", payload):
        part = part.strip()
        if not part: continue
        if "=" in part:
            k, v = [x.strip() for x in part.split("=", 1)]
        else:
            m = re.match(r"([A-Za-z]+)\s+([\d.]+)", part)
            if not m: continue
            k, v = m.group(1), m.group(2)
        k = _norm_key(k)

        if k in ("batch","operator","date","machine"):
            out[k] = v
            continue

        v = v.replace("ton", "T").replace("mt","T").replace("kg","K")
        mT = re.match(r"([\d.]+)\s*[tT]$", v)
        mK = re.match(r"([\d.]+)\s*[kK]$", v)
        if mT: val = float(mT.group(1)) * 1000.0
        elif mK: val = float(mK.group(1))
        else:   val = float(re.sub(r"[^\d.]", "", v) or 0.0)
        out[k] = val

    # standardize keys
    rename = {"radial":"radial","nylon":"nylon","chips":"chips","powder":"powder","kachra":"kachra","others":"others"}
    return {rename.get(k, k): v for k, v in out.items()}

def parse_actual(text: str) -> Dict[str, float|str]:
    """
    Actual: 50-200=01:14, 200-300=01:06, 300-400=02:07, 400-450=01:10, 450-480=00:32, oil=40.73, batch=66
    """
    t = re.sub(r"^/?actual\s*:\s*", "", text, flags=re.I).strip()
    out: Dict[str, float|str] = {}
    for chunk in re.split(r"[;,]+", t):
        chunk = chunk.strip()
        if not chunk or "=" not in chunk: continue
        k, v = [x.strip() for x in chunk.split("=", 1)]
        lk = _norm_key(k)
        if lk in ("oil","oil%"):
            out["oil"] = float(re.sub(r"[^\d.]", "", v) or 0.0)
        elif lk == "batch":
            out["batch"] = v
        elif re.match(r"\d{2,3}-\d{2,3}", lk):
            out[lk] = hhmmss_to_minutes(v)
    return out

def machine_label(chat_id: int) -> str:
    return MACHINE_MAP.get(str(chat_id)) or MACHINE_MAP.get(str(int(chat_id))) or str(chat_id)

def pretty_plan(plan: Dict[str,float]) -> str:
    def keyfn(s: str): m=re.match(r"(\d{2,3})", s); return (int(m.group(1)) if m else 999, s)
    lines = []
    for z in sorted(plan.keys(), key=keyfn):
        lines.append(f"{z}: {to_hhmmss(plan[z])}")
    return "\n".join(lines)

# -------------------- Planning & yield model --------------------
class RecoEngine:
    def __init__(self, path: str):
        self.path = path
        self.zone_defaults: Dict[str,float] = {}
        self._load()

    def _load(self):
        self.zone_defaults = load_zone_rules(self.path, ZONE_SHEET)

    def plan(self, feed: Dict[str, float|str]) -> Dict[str,float]:
        plan = dict(self.zone_defaults)
        total = sum(float(feed.get(k,0) or 0) for k in ("radial","nylon","chips","powder","kachra","others"))
        if total > 0:
            r = float(feed.get("radial",0))/total
            c = float(feed.get("chips",0))/total
            n = float(feed.get("nylon",0))/total
            # Gentle adjustments based on composition
            for k in list(plan.keys()):
                if k.startswith("300-400"):
                    plan[k] = max(20, plan[k] * (1.0 + 0.15*(r + 0.5*c - 0.6*n)))
                if k.startswith("200-300"):
                    plan[k] = max(15, plan[k] * (1.0 + 0.10*r - 0.05*n))
        return plan

engine = RecoEngine(REPORT)

def predict_yield(feed: Dict[str, float|str]) -> Tuple[float,float]:
    """Return (predicted oil %, confidence 0..1)."""
    total = sum(float(feed.get(k,0) or 0) for k in ("radial","nylon","chips","powder","kachra","others"))
    if total <= 0:
        return (0.0, 0.6)
    w = state["weights"]
    pred = 0.0
    for k in ("radial","nylon","chips","powder","kachra","others"):
        kg = float(feed.get(k,0) or 0)
        pred += (kg/total) * (w.get(k,0)*100.0)
    pred = max(30.0, min(55.0, pred))
    # Confidence proxy – more balanced mix => higher
    ratio_spread = np.std([ (float(feed.get(x,0) or 0)/total) for x in ("radial","nylon","chips","powder") ])
    conf = max(0.60, min(0.95, 0.95 - 0.8*ratio_spread))
    return (round(pred,2), round(conf,2))

def learn_from_actual(feed: Dict[str,float|str], actual_oil: float):
    total = sum(float(feed.get(k,0) or 0) for k in ("radial","nylon","chips","powder","kachra","others"))
    if total <= 0: return
    pred,_ = predict_yield(feed)
    err = (actual_oil - pred)/100.0
    for k in ("radial","nylon","chips","powder","kachra","others"):
        share = float(feed.get(k,0) or 0)/total
        state["weights"][k] = float(state["weights"].get(k,0)) + 0.01 * err * share
    save_state()

# -------------------- Deviation & recommendations --------------------
def delta_tips(plan: Dict[str,float], actual_min_by_zonekey: Dict[str,float]) -> Tuple[Dict[str,float], List[str]]:
    """
    Compare actual vs plan and produce text tips with graded strength.
    """
    deltas, tips = {}, []
    def level(absmin: float) -> str:
        return "slightly " if absmin < 20 else "moderately " if absmin < 40 else "significantly "

    for z, pmin in plan.items():
        key = z.replace(" ", "")
        if key not in actual_min_by_zonekey:
            continue
        am   = actual_min_by_zonekey[key]
        diff = am - pmin
        deltas[z] = diff
        if abs(diff) >= 10:
            tips.append(f"• {level(abs(diff))}{'reduce' if diff>0 else 'increase'} *{z}* by ~{abs(diff):.0f} min")
    return deltas, tips

def oil_yield_reco(feed: Dict[str,float|str], predicted: float, actual_oil: float|None, deltas: Dict[str,float]) -> List[str]:
    """
    Heuristic recommendations to move oil yield toward predicted/target.
    Uses zone deviations & simple chemistry rules-of-thumb.
    """
    adv: List[str] = []
    # If actual_oil is provided, compare; else provide target guidance vs predicted.
    if actual_oil is not None:
        gap = actual_oil - predicted
        if gap >= 1.0:
            adv.append(f"✅ Oil yield is *{gap:+.2f}%* vs predicted — great job. Keep this profile.")
            return adv
        elif gap <= -1.0:
            adv.append(f"⚠️ Oil below model by *{gap:+.2f}%*. Suggestions:")
    else:
        adv.append(f"🎯 To reach ~*{predicted:.2f}%* oil, consider:")
        gap = None

    # Use deviations to decide levers
    early_cut  = (deltas.get("50-200 reactor",0) +
                  deltas.get("200-300 reactor",0) +
                  deltas.get("300-400 reactor",0))
    late_over  = max(0, deltas.get("400-450 reactor",0)) + max(0, deltas.get("450-480 reactor",0))

    if early_cut < -30:  # ran shorter than plan in early zones
        adv.append("• Extend *50-200 / 200-300 / 300-400 reactor* windows to improve primary vaporization.")
    if late_over > 20:   # stayed longer at high temp than plan
        adv.append("• Trim *400-450 / 450-480 reactor* to reduce over-cracking to gas.")
    if "300-400 separator" in deltas and deltas["300-400 separator"] < -20:
        adv.append("• Increase *300-400 separator* residence to capture heavy vapors.")
    if not adv or len(adv)==(1 if gap and gap<=-1 else 0):
        adv.append("• Keep ramps steady; avoid spikes; prioritize stable 300–380 °C vapor draw.")
    return adv

# -------------------- Chart --------------------
def chart_plan_vs_actual(plan: Dict[str,float], actual: Dict[str,float]|None=None) -> io.BytesIO:
    zones = [z for z in sorted(plan.keys(), key=lambda s: int(s.split("-")[0]))]
    pmins = [plan[z] for z in zones]
    amins = [actual.get(z.replace(" ",""), np.nan) if actual else np.nan for z in zones]
    x = np.arange(len(zones))
    width = 0.38

    fig, ax = plt.subplots(figsize=(10,3), dpi=200)
    ax.bar(x - width/2, pmins, width, label="Plan (min)")
    if actual:
        ax.bar(x + width/2, amins, width, label="Actual (min)")
    ax.set_xticks(x, zones, rotation=18, ha="right")
    ax.set_ylabel("Minutes")
    ax.legend()
    ax.grid(True, axis="y", alpha=0.25)
    buf = io.BytesIO()
    plt.tight_layout()
    plt.savefig(buf, format="png")
    plt.close(fig)
    buf.seek(0)
    return buf

# -------------------- Bot replies --------------------
HELP = (
    "*Commands*\n"
    "• Send **Feed:** `Feed: Radial=5.1T, Nylon=0.6T, Chips=3.4T, Powder=1.5T, batch=92, operator=Ravi, date=09-11-2025`\n"
    "  → Plan + Predicted oil (also posted in Summary).\n"
    "• Send **Actual:** `Actual: 50-200=01:14, 200-300=01:06, 300-400=02:07, 400-450=01:10, 450-480=00:32, oil=40.7; batch=92`\n"
    "  → Deviations + *Recommendations* + chart (and the model learns).\n"
    "• `/status` → Machine status (or all in the Summary group)\n"
    "• `/reload` → Reload Excel rules\n"
    "• `/id` → Show this chat’s id and label\n"
)

async def cmd_start(update: Update, context: ContextTypes.DEFAULT_TYPE):
    await update.message.reply_text("PyroVision Assistant ready ✅\n\n" + HELP, parse_mode="Markdown")

async def cmd_help(update: Update, context: ContextTypes.DEFAULT_TYPE):
    await update.message.reply_text(HELP, parse_mode="Markdown")

async def cmd_reload(update: Update, context: ContextTypes.DEFAULT_TYPE):
    engine._load()
    await update.message.reply_text("🔁 Reloaded recommendations from Excel.")
    if SUMMARY_CHAT_ID:
        await context.bot.send_message(SUMMARY_CHAT_ID, f"🔁 Rules reloaded by {machine_label(update.effective_chat.id)}")

async def cmd_id(update: Update, context: ContextTypes.DEFAULT_TYPE):
    cid = update.effective_chat.id
    await update.message.reply_text(f"chat_id: `{cid}`\nname/title: *{machine_label(cid)}*", parse_mode="Markdown")

async def cmd_status(update: Update, context: ContextTypes.DEFAULT_TYPE):
    now = datetime.now(BOT_TZ)
    if SUMMARY_CHAT_ID and update.effective_chat.id == SUMMARY_CHAT_ID:
        lines = ["🟢 *Machine Status*"]
        for cid_str, label in MACHINE_MAP.items():
            lf = state["latest_feed"].get(cid_str)
            st = "Idle"
            if lf:
                feed_ts = datetime.fromisoformat(lf["ts"]).astimezone(BOT_TZ)
                hours   = (now - feed_ts).total_seconds()/3600
                last_act_iso = state["last_actual_ts"].get(cid_str)
                completed = False
                if last_act_iso:
                    completed = datetime.fromisoformat(last_act_iso) >= datetime.fromisoformat(lf["ts"])
                if hours <= 12 and not completed:
                    st = f"Running (batch {lf.get('batch','?')})"
                elif completed or hours > 12:
                    st = f"Completed (batch {lf.get('batch','?')}, {lf.get('date','')})"
            lines.append(f"• {label}: {st}")
        await update.message.reply_text("\n".join(lines), parse_mode="Markdown")
    else:
        cid = str(update.effective_chat.id)
        label = machine_label(update.effective_chat.id)
        lf = state["latest_feed"].get(cid)
        st = "Idle"
        if lf:
            feed_ts = datetime.fromisoformat(lf["ts"]).astimezone(BOT_TZ)
            hours   = (now - feed_ts).total_seconds()/3600
            last_act_iso = state["last_actual_ts"].get(cid)
            completed = last_act_iso and (datetime.fromisoformat(last_act_iso) >= datetime.fromisoformat(lf["ts"]))
            if hours <= 12 and not completed:
                st = f"Running (batch {lf.get('batch','?')})"
            elif completed or hours > 12:
                st = f"Completed (batch {lf.get('batch','?')}, {lf.get('date','')})"
        await update.message.reply_text(f"{label}: {st}")

# --------------- Feed & Actual processing ---------------
def remember_feed(chat_id: int, feed: Dict[str,float|str], plan: Dict[str,float]):
    now = datetime.now(BOT_TZ)
    state["latest_feed"][str(chat_id)] = {
        "ts": now.isoformat(),
        "date": _norm_date(str(feed.get("date",""))),
        "batch": str(feed.get("batch","")),
        "operator": str(feed.get("operator","")),
        "feed": feed,
        "plan": plan,
    }
    save_state()

async def plan_core(update: Update, context: ContextTypes.DEFAULT_TYPE, feed_text: str, predict_only=False):
    chat_id = update.effective_chat.id
    label   = machine_label(chat_id)
    feed    = parse_feed(feed_text)
    plan    = engine.plan(feed)
    pred, conf = predict_yield(feed)
    remember_feed(chat_id, feed, plan)

    date_str = _norm_date(str(feed.get("date","")))
    batch = feed.get("batch","?")
    oper  = feed.get("operator","?")

    msg = []
    msg.append(f"📒 *{label}* — *Batch* {batch}, *Operator* {oper}")
    msg.append(f"• *Date* {date_str}")
    msg.append(f"🛢️ *Feed:* Radial {float(feed.get('radial',0))/1000:.2f}T, Nylon {float(feed.get('nylon',0))/1000:.2f}T, Chips {float(feed.get('chips',0))/1000:.2f}T, Powder {float(feed.get('powder',0))/1000:.2f}T")
    msg.append(f"📈 *Predicted Oil:* *{pred:.2f}%*  (confidence {conf:.2f})")
    if not predict_only:
        msg.append("\n*Recommended zone minutes (plan):*\n" + pretty_plan(plan))
    text = "\n".join(msg)

    await update.message.reply_text(text, parse_mode="Markdown")
    if SUMMARY_CHAT_ID:
        await context.bot.send_message(SUMMARY_CHAT_ID, text, parse_mode="Markdown")

async def cmd_plan(update: Update, context: ContextTypes.DEFAULT_TYPE):
    await plan_core(update, context, update.message.text, predict_only=False)

async def cmd_whatif(update: Update, context: ContextTypes.DEFAULT_TYPE):
    t = re.sub(r"^/?whatif\s*", "", update.message.text, flags=re.I)
    await plan_core(update, context, t, predict_only=True)

async def cmd_actual(update: Update, context: ContextTypes.DEFAULT_TYPE):
    chat_id = update.effective_chat.id
    cid = str(chat_id)
    lf = state["latest_feed"].get(cid)
    if not lf:
        await update.message.reply_text("I can't find a recent Feed for this machine. Send the Feed first.")
        return

    actual_raw = parse_actual(update.message.text)
    plan = lf.get("plan", {})
    # build "actual minutes by zone key"
    actual_min = {k: float(v) for k,v in actual_raw.items() if re.match(r"\d{2,3}-\d{2,3}", k)}

    # deviations + tips
    deltas, zone_tips = delta_tips(plan, actual_min)

    # oil recommendations
    feed = lf.get("feed", {})
    predicted, _ = predict_yield(feed)
    actual_oil = float(actual_raw.get("oil")) if "oil" in actual_raw else None
    oil_tips   = oil_yield_reco(feed, predicted, actual_oil, deltas)

    # learning
    if actual_oil is not None:
        learn_from_actual(feed, actual_oil)
        state["last_actual_ts"][cid] = datetime.now(BOT_TZ).isoformat()
        save_state()

    # Compose reply
    lines = ["📊 *Deviation vs plan (min):*"]
    for z in sorted(deltas.keys(), key=lambda s: int(s.split("-")[0])):
        lines.append(f"{z}: {deltas[z]:+.0f}")

    if zone_tips:
        lines.append("\n🛠️ *Zone Recommendations:*")
        lines.extend(zone_tips)
    else:
        lines.append("\n✅ Near-optimal execution vs plan.")

    lines.append("\n🛢️ *Oil-Yield Guidance:*")
    lines.extend(oil_tips)

    # Send text
    await update.message.reply_text("\n".join(lines), parse_mode="Markdown")

    # Send chart
    buf = chart_plan_vs_actual(plan, actual_min)
    await context.bot.send_photo(chat_id, photo=InputFile(buf, filename="plan_vs_actual.png"))

    # Fan-out notice to Summary
    if SUMMARY_CHAT_ID:
        await context.bot.send_message(
            SUMMARY_CHAT_ID,
            f"🧾 Actual logged for *{machine_label(chat_id)}* (batch {lf.get('batch','?')}).",
            parse_mode="Markdown"
        )

# -------------------- Text Router --------------------
async def router(update: Update, context: ContextTypes.DEFAULT_TYPE):
    text = (update.message.text or "").strip()
    if re.match(r"(?i)^feed\s*:", text):
        await plan_core(update, context, text, predict_only=False)
    elif re.match(r"(?i)^actual\s*:", text):
        await cmd_actual(update, context)
    elif re.match(r"(?i)^status$", text):
        await cmd_status(update, context)
    else:
        await update.message.reply_text("I didn’t understand.\n\n" + HELP, parse_mode="Markdown")

# -------------------- Main --------------------
def main():
    if not TOKEN:
        raise RuntimeError("❌ TELEGRAM_BOT_TOKEN env var is missing.")
    load_state()

    app = Application.builder().token(TOKEN).build()
    app.add_handler(CommandHandler("start",  cmd_start))
    app.add_handler(CommandHandler("help",   cmd_help))
    app.add_handler(CommandHandler("reload", cmd_reload))
    app.add_handler(CommandHandler("status", cmd_status))
    app.add_handler(CommandHandler("id",     cmd_id))
    app.add_handler(CommandHandler("plan",   cmd_plan))
    app.add_handler(CommandHandler("whatif", cmd_whatif))
    app.add_handler(CommandHandler("actual", cmd_actual))
    app.add_handler(MessageHandler(filters.TEXT & ~filters.COMMAND, router))

    LOG.info("✅ Bot ready. Polling…")
    app.run_polling()

if __name__ == "__main__":
    main()
