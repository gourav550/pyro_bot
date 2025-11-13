# bot.py -- PyroVision Assistant (fixed, full)
# Ensure env vars: TELEGRAM_BOT_TOKEN (required), SUMMARY_CHAT_ID (optional), REPORT_PATH (optional), MACHINE_MAP (optional JSON)

import os
import io
import re
import json
import math
import logging
import asyncio
from datetime import datetime, timedelta
from zoneinfo import ZoneInfo

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from apscheduler.schedulers.asyncio import AsyncIOScheduler
from telegram import InputFile
from telegram.constants import ParseMode
from telegram.ext import ApplicationBuilder, CommandHandler, MessageHandler, ContextTypes, filters

# ---- logging ----
LOG = logging.getLogger("pyro_bot")
logging.basicConfig(level=logging.INFO, format="%(asctime)s | %(levelname)s | %(name)s | %(message)s")

# ---- config via env ----
TOKEN = os.getenv("TELEGRAM_BOT_TOKEN", "")
SUMMARY_CHAT_ID = int(os.getenv("SUMMARY_CHAT_ID", "0") or "0")
REPORT_PATH = os.getenv("REPORT_PATH", "pyrolysis_feed_temp_ZONE_TIME_report.xlsx")
BOT_TZ = ZoneInfo(os.getenv("BOT_TZ", "Asia/Kolkata"))

try:
    MACHINE_MAP = json.loads(os.getenv("MACHINE_MAP", "{}"))
except Exception:
    MACHINE_MAP = {}

STATE_PATH = "bot_state.json"

# ---- persistent state ----
state = {
    "weights": {
        "radial": 0.44,
        "nylon": 0.42,
        "chips": 0.46,
        "powder": 0.53,
        "kachra": 0.40,
        "others": 0.40,
    },
    "latest_feed": {},    # chat_id -> feed record
    "last_actual_ts": {}, # chat_id -> iso ts
    "reminders": {},      # "chat:batch" -> {"chat_id":..., "batch":..., "due":iso}
    "mix_mean": None,
    "errors": []
}

# ---- I/O helpers ----
def save_state():
    try:
        with open(STATE_PATH, "w", encoding="utf-8") as f:
            json.dump(state, f, ensure_ascii=False, indent=2)
    except Exception as e:
        LOG.warning("Could not save state: %s", e)

def load_state():
    if os.path.exists(STATE_PATH):
        try:
            with open(STATE_PATH, "r", encoding="utf-8") as f:
                obj = json.load(f)
            state.update(obj)
            LOG.info("✅ State loaded.")
        except Exception as e:
            LOG.warning("Could not load state: %s", e)
    else:
        LOG.info("ℹ️ No state file; starting fresh.")
    state.setdefault("mix_mean", None)
    state.setdefault("errors", [])

# ---- formatting helpers ----
def to_hhmmss(minutes):
    try:
        if minutes is None or (isinstance(minutes, float) and math.isnan(minutes)):
            return "00:00:00"
        total = int(round(float(minutes) * 60))
        sign = "-" if total < 0 else ""
        total = abs(total)
        h, rem = divmod(total, 3600)
        m, s = divmod(rem, 60)
        return f"{sign}{h:02d}:{m:02d}:{s:02d}"
    except Exception:
        return "00:00:00"

def minutes_to_hhmm(minutes):
    try:
        m = int(round(minutes))
        sign = "-" if m < 0 else ""
        m = abs(m)
        h, rem = divmod(m, 60)
        return f"{sign}{h}:{rem:02d}"
    except Exception:
        return "0:00"

def hhmmss_to_minutes(s: str) -> float:
    s = (s or "").strip()
    if ":" not in s:
        try:
            return float(s or 0.0)
        except Exception:
            return 0.0
    parts = [int(x) for x in s.split(":")]
    if len(parts) == 3:
        h, m, se = parts
        return h*60 + m + se/60
    if len(parts) == 2:
        m, se = parts
        return m + se/60
    return 0.0

def machine_label(chat_id):
    return MACHINE_MAP.get(str(chat_id)) or MACHINE_MAP.get(str(int(chat_id))) or str(chat_id)

def _norm_date(s: str | None) -> str:
    if not s:
        now = datetime.now(BOT_TZ)
        return now.strftime("%d-%b-%Y (%A)")
    s2 = s.strip()
    fmts = ("%d-%m-%Y","%d.%m.%Y","%d/%m/%Y","%d-%m-%y","%d.%m.%y","%d/%m/%y","%Y-%m-%d")
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
        return datetime.now(BOT_TZ).strftime("%d-%b-%Y (%A)")

# ---- Excel rules ----
def load_zone_rules(path=REPORT_PATH, sheet="ZoneTime_Recommendations"):
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
            w = f"{m.group(1)}-{m.group(2)}"
            which = "separator" if "separator" in feat else "reactor"
            rules[f"{w} {which}"] = float(mins)
        LOG.info("Loaded %d zone rules from Excel.", len(rules))
    except Exception as e:
        LOG.warning("Excel rules load failed: %s; using defaults.", e)
        rules = {
            "50-200 reactor": 165.0, "200-300 reactor": 70.0, "300-400 reactor": 185.0,
            "300-400 separator": 175.0, "400-450 reactor": 75.0, "450-480 reactor": 20.0,
            "480-500 reactor": 0.0, "500-520 reactor": 0.0
        }
    return rules

ZONE_RULES = load_zone_rules()

def maybe_load_yield_weights(path=REPORT_PATH, sheet="YieldWeights"):
    try:
        df = pd.read_excel(path, sheet_name=sheet)
        comp = df["component"].astype(str).str.lower().str.strip()
        val  = df["weight"].astype(float)
        for k, v in zip(comp, val):
            if k in state["weights"] and 0.2 <= float(v) <= 0.8:
                state["weights"][k] = float(v)
        LOG.info("YieldWeights applied from Excel.")
    except Exception:
        pass

# ---- parsing ----
def parse_feed(text: str) -> dict:
    t = re.sub(r"^/?(plan|predict)?\s*feed\s*:\s*|^feed\s*:\s*", "", text, flags=re.I).strip()
    data = {}
    for part in re.split(r"[,\n;]+", t):
        p = part.strip()
        if not p:
            continue
        if "=" in p:
            k, v = [x.strip() for x in p.split("=", 1)]
        else:
            m = re.match(r"([A-Za-z]+)\s+([\d.]+)", p)
            if not m:
                continue
            k, v = m.group(1), m.group(2)
        lk = k.lower()
        if lk in ("batch","operator","date","machine"):
            data[lk] = v
            continue
        v2 = v.lower().strip()
        if any(v2.endswith(s) for s in ("t","ton","tons","mt")):
            valkg = float(re.sub(r"[^\d.]", "", v2)) * 1000.0
        elif any(v2.endswith(s) for s in ("kg","kgs")):
            valkg = float(re.sub(r"[^\d.]", "", v2))
        else:
            valkg = float(re.sub(r"[^\d.]", "", v2) or 0.0)
        data[lk] = valkg
    return data

def parse_actual(text: str) -> dict:
    t = re.sub(r"^/?actual\s*:?","", text, flags=re.I).strip()
    out = {}
    for chunk in re.split(r"[;,]+", t):
        if "=" not in chunk:
            continue
        k, v = [x.strip() for x in chunk.split("=", 1)]
        lk = k.lower()
        if re.match(r"\d{2,3}\s*-\s*\d{2,3}", lk):
            zone = re.sub(r"\s+", "", lk)
            out[zone] = hhmmss_to_minutes(v) if ":" in v else float(re.sub(r"[^\d.]", "", v) or 0.0)
        elif lk in ("oil","oil%","oilpct","oil_pct"):
            out["oil"] = float(re.sub(r"[^\d.]", "", v) or 0.0)
        elif lk == "batch":
            out["batch"] = v
    return out

# ---- engine ----
class RecoEngine:
    def __init__(self, base_rules):
        self.defaults = dict(base_rules)
    def plan(self, feed):
        plan = dict(self.defaults)
        total = sum(feed.get(k,0.0) for k in ("radial","nylon","chips","powder","kachra","others"))
        if total > 0:
            rr = feed.get("radial",0.0)/total
            cr = feed.get("chips",0.0)/total
            nr = feed.get("nylon",0.0)/total
            for k in list(plan.keys()):
                if k.startswith("300-400"):
                    plan[k] = max(60.0, plan[k]*(1.00 + 0.18*rr + 0.08*cr - 0.06*nr))
                if k.startswith("200-300"):
                    plan[k] = max(45.0, plan[k]*(1.00 + 0.06*rr - 0.04*nr))
        return plan

ENGINE = RecoEngine(ZONE_RULES)

def _normalize_mix(feed):
    keys = ("radial","nylon","chips","powder","kachra","others")
    total = sum(feed.get(k,0.0) for k in keys)
    if total <= 0:
        return {k:0.0 for k in keys}
    return {k: feed.get(k,0.0)/total for k in keys}

def _update_mix_mean(mix_norm):
    alpha = 0.10
    if not state.get("mix_mean"):
        state["mix_mean"] = dict(mix_norm)
        return
    mm = state["mix_mean"]
    for k,v in mix_norm.items():
        mm[k] = (1-alpha)*mm.get(k,0.0) + alpha*v

def _recent_mae():
    errs = state.get("errors") or []
    if not errs:
        return 2.5
    return float(np.mean(errs))

def predict_yield(feed):
    keys = ("radial","nylon","chips","powder","kachra","others")
    total = sum(feed.get(k,0.0) for k in keys)
    if total <= 0:
        return (0.0, 0.60)
    w = state["weights"]
    pred = 0.0
    for k in keys:
        pred += (feed.get(k,0.0)/total) * (w[k]*100.0)
    pred = float(np.clip(pred, 30.0, 60.0))
    mae = _recent_mae()
    mae_penalty = np.clip(mae/6.0, 0.0, 1.0) * 0.20
    mix_norm = _normalize_mix(feed)
    mm = state.get("mix_mean")
    if mm:
        l1 = sum(abs(mix_norm[k]-mm.get(k,0.0)) for k in mix_norm)
    else:
        l1 = 0.3
    sim_penalty = np.clip(l1/0.8, 0.0, 1.0) * 0.15
    conf = 0.95 - mae_penalty - sim_penalty
    conf = float(np.clip(conf, 0.60, 0.95))
    return (round(pred,2), round(conf,2))

def learn_from_actual(feed, actual_oil_pct):
    keys = ("radial","nylon","chips","powder","kachra","others")
    total = sum(feed.get(k,0.0) for k in keys)
    if total > 0:
        pred,_ = predict_yield(feed)
        err = (actual_oil_pct - pred)/100.0
        for k in keys:
            share = feed.get(k,0.0)/total
            state["weights"][k] += 0.007 * err * share
        for k in state["weights"]:
            state["weights"][k] = float(np.clip(state["weights"][k], 0.30, 0.65))
    mix_norm = _normalize_mix(feed)
    _update_mix_mean(mix_norm)
    err_abs = abs(actual_oil_pct - predict_yield(feed)[0])
    state["errors"].append(float(err_abs))
    if len(state["errors"]) > 40:
        state["errors"] = state["errors"][-40:]
    save_state()

# ---- rendering ----
def pretty_plan(plan):
    def keyer(x):
        m = re.match(r"(\d{2,3})-(\d{2,3})", x)
        if not m: return (999,999,x)
        return (int(m.group(1)), int(m.group(2)), x)
    lines = []
    for z in sorted(plan.keys(), key=keyer):
        lines.append(f"{z}: {to_hhmmss(plan[z])}")
    return "\n".join(lines)

def bar_plan_vs_actual(plan, actual):
    zones = [z for z in sorted(plan.keys(), key=lambda s: int(s.split("-")[0]))]
    pmins = [plan[z] for z in zones]
    amins = []
    amap = {}
    if actual:
        amap = {k.replace(" ",""):v for k,v in actual.items() if "-" in k or re.match(r"\d{2,3}-\d{2,3}", k)}
    for z in zones:
        amins.append(amap.get(z.replace(" ",""), np.nan))
    x = np.arange(len(zones))
    fig, ax = plt.subplots(figsize=(10,3.5), dpi=140)
    width = 0.38
    ax.bar(x - width/2, pmins, width, label="Plan (min)")
    if actual:
        ax.bar(x + width/2, amins, width, label="Actual (min)")
    ax.set_xticks(x, zones, rotation=30)
    ax.set_ylabel("Minutes")
    ax.legend()
    ax.grid(True, axis="y", alpha=0.25)
    buf = io.BytesIO()
    plt.tight_layout()
    plt.savefig(buf, format="png")
    plt.close(fig)
    buf.seek(0)
    return buf

# ---- heuristics ----
def oil_yield_reco(pred, actual, plan, actual_zones):
    delta = actual - pred
    tips = []
    if delta < -2.0:
        tips.append("Increase *300–400 separator* by ~15–25 min (more condensation).")
        tips.append("Extend *300–400 reactor* by ~20–30 min for heavier mix completion.")
        tips.append("Slightly extend *200–300 reactor* by ~10–15 min (stabilize ramp).")
        tips.append("Check line losses & condenser ΔT; avoid sharp temp spikes.")
    elif delta < -0.7:
        tips.append("Slightly extend *300–400 separator* by ~10–15 min; monitor condensables.")
        tips.append("Add ~10–15 min to *300–400 reactor* if drip ends early.")
    else:
        tips.append("Yield on or near target; maintain steady ramp and 300–400 recovery.")
    return tips

# ---- scheduler jobs ----
def kkey(chat_id, batch):
    return f"{chat_id}:{batch or ''}"

async def reminder_tick(app):
    now = datetime.now(BOT_TZ)
    for key, rec in list(state["reminders"].items()):
        try:
            due = datetime.fromisoformat(rec["due"])
            if due.tzinfo is None:
                due = due.replace(tzinfo=BOT_TZ)
            if now >= due:
                cid = rec["chat_id"]
                bno = rec["batch"]
                txt = (f"⚠️ *Reminder:* Actual data not entered for *Batch {bno}* on *{machine_label(cid)}*.\n"
                       "Please send:\n"
                       "`Actual: 50-200=hh:mm, 200-300=hh:mm, 300-400=hh:mm, 400-450=hh:mm, 450-480=hh:mm, oil=xx.x; batch=..`")
                try:
                    await app.bot.send_message(cid, txt, parse_mode=ParseMode.MARKDOWN)
                    if SUMMARY_CHAT_ID:
                        await app.bot.send_message(SUMMARY_CHAT_ID, f"🔔 Reminder sent to {machine_label(cid)} for batch {bno}", parse_mode=ParseMode.MARKDOWN)
                except Exception as e:
                    LOG.warning("Reminder send failed: %s", e)
                # schedule next hourly ping
                state["reminders"][key]["due"] = (now + timedelta(hours=1)).isoformat()
        except Exception as e:
            LOG.warning("reminder_tick parse failed for %s: %s", key, e)
    save_state()

async def daily_summary_job(app):
    now = datetime.now(BOT_TZ)
    lines = [f"📊 *Daily Summary* — {now.strftime('%d-%b-%Y (%A) %H:%M')} IST"]
    for cid_str, label in MACHINE_MAP.items():
        lf = state["latest_feed"].get(cid_str)
        status = "Idle"
        if lf:
            try:
                lfts = datetime.fromisoformat(lf["ts"])
                if lfts.tzinfo is None: lfts = lfts.replace(tzinfo=BOT_TZ)
                hrs = (now - lfts).total_seconds()/3600
                last_act_iso = state["last_actual_ts"].get(cid_str)
                completed = False
                if last_act_iso:
                    lat = datetime.fromisoformat(last_act_iso)
                    if lat.tzinfo is None: lat = lat.replace(tzinfo=BOT_TZ)
                    completed = lat >= lfts
                if hrs <= 12 and not completed:
                    status = f"Running (batch {lf.get('batch','?')})"
                elif completed:
                    # show completed with date & weekday
                    comp_date = _norm_date(lf.get("date"))
                    status = f"Completed (batch {lf.get('batch','?')}) — {comp_date}"
                else:
                    status = "Idle"
            except Exception:
                status = "Idle"
        lines.append(f"• {label}: {status}")
    if SUMMARY_CHAT_ID:
        try:
            await app.bot.send_message(SUMMARY_CHAT_ID, "\n".join(lines), parse_mode=ParseMode.MARKDOWN)
        except Exception as e:
            LOG.warning("Daily summary send failed: %s", e)

# ---- handlers ----
HELP = (
    "*Commands*\n"
    "• Send *Feed:* `Feed: Radial=5.1T, Nylon=0.6T, Chips=3.4T, Powder=1.5T, batch=92, operator=Ravi, date=09-11-2025`\n"
    "  → Plan + Predicted oil.\n"
    "• Send *Actual:* `Actual: 50-200=01:14, 200-300=01:06, 300-400=02:07, 400-450=01:10, 450-480=00:32, oil=40.7; batch=92`\n"
    "  → Deviation + Oil recommendations + chart.\n"
    "• `/status` → Machine status (or all in Summary chat)\n"
    "• `/reload` → Reload Excel rules / weights\n"
    "• `/id` → Show this chat’s id and label\n"
    "• `what feed: target=50.0` → Suggest feed to aim for target oil%\n"
)

async def cmd_start(update: "Update", context: ContextTypes.DEFAULT_TYPE):
    await update.message.reply_text("PyroVision Assistant ready ✅\n\n" + HELP, parse_mode=ParseMode.MARKDOWN)

async def cmd_help(update: "Update", context: ContextTypes.DEFAULT_TYPE):
    await update.message.reply_text(HELP, parse_mode=ParseMode.MARKDOWN)

async def cmd_id(update, context):
    cid = update.effective_chat.id
    await update.message.reply_text(f"chat_id: `{cid}`\nname/title: *{machine_label(cid)}*", parse_mode=ParseMode.MARKDOWN)

async def cmd_reload(update, context):
    global ZONE_RULES, ENGINE
    ZONE_RULES = load_zone_rules()
    ENGINE = RecoEngine(ZONE_RULES)
    maybe_load_yield_weights()
    save_state()
    await update.message.reply_text("🔁 Reloaded Excel rules and YieldWeights (if present).")

async def cmd_status(update, context):
    now = datetime.now(BOT_TZ)
    if SUMMARY_CHAT_ID and update.effective_chat.id == SUMMARY_CHAT_ID:
        lines = ["🟢 *Machine Status*"]
        for cid_str, label in MACHINE_MAP.items():
            lf = state["latest_feed"].get(cid_str)
            status = "Idle"
            if lf:
                try:
                    lfts = datetime.fromisoformat(lf["ts"])
                    if lfts.tzinfo is None: lfts = lfts.replace(tzinfo=BOT_TZ)
                    hrs = (now - lfts).total_seconds()/3600
                    last_act_iso = state["last_actual_ts"].get(cid_str)
                    completed = False
                    if last_act_iso:
                        lat = datetime.fromisoformat(last_act_iso)
                        if lat.tzinfo is None: lat = lat.replace(tzinfo=BOT_TZ)
                        completed = lat >= lfts
                    if hrs <= 12 and not completed:
                        status = f"Running (batch {lf.get('batch','?')})"
                    elif completed:
                        status = f"Completed (batch {lf.get('batch','?')})"
                    else:
                        status = "Idle"
                except Exception:
                    status = "Idle"
            lines.append(f"• {label}: {status}")
        await update.message.reply_text("\n".join(lines), parse_mode=ParseMode.MARKDOWN)
        return
    cid_str = str(update.effective_chat.id)
    lf = state["latest_feed"].get(cid_str)
    if not lf:
        await update.message.reply_text("No recent feed recorded for this chat.")
        return
    feed = lf.get("feed", {})
    pred, conf = lf.get("pred"), lf.get("conf")
    keys = ("radial","nylon","chips","powder")
    feed_line = ", ".join([f"{k.title()} {round(feed.get(k,0)/1000,3)}T" for k in keys if feed.get(k,0)])
    txt = (f"• *Machine:* {machine_label(cid_str)}\n"
           f"• *Batch:* {lf.get('batch','?')}\n"
           f"• Date: {_norm_date(lf.get('date'))}\n\n"
           f"Feed: {feed_line}\n"
           f"Predicted Oil: *{pred}%* (conf {conf})")
    await update.message.reply_text(txt, parse_mode=ParseMode.MARKDOWN)

async def handle_message(update, context):
    txt = (update.message.text or "").strip()
    if re.match(r"^/?feed\s*:", txt, flags=re.I) or re.search(r"\bfeed\s*:", txt, flags=re.I):
        await handle_feed(update, context); return
    if re.match(r"^/?actual\s*:", txt, flags=re.I) or re.search(r"^actual\s*:", txt, flags=re.I):
        await handle_actual(update, context); return
    if re.match(r"^what\s+feed\s*:", txt, flags=re.I):
        await handle_what_feed(update, context); return
    await update.message.reply_text("I didn't understand.\n\n" + HELP, parse_mode=ParseMode.MARKDOWN)

async def handle_feed(update, context):
    txt = update.message.text
    feed = parse_feed(txt)
    keys_all = ("radial","nylon","chips","powder","kachra","others")
    for k in keys_all:
        feed.setdefault(k, 0.0)
    plan = ENGINE.plan(feed)
    pred, conf = predict_yield(feed)
    cid = str(update.effective_chat.id)
    batch = feed.get("batch") or ""
    ts = datetime.now(BOT_TZ).isoformat()
    state["latest_feed"][cid] = {
        "ts": ts,
        "batch": batch,
        "operator": feed.get("operator",""),
        "date": feed.get("date") or datetime.now(BOT_TZ).strftime("%d-%m-%Y"),
        "feed": feed,
        "plan": plan,
        "pred": pred,
        "conf": conf
    }
    # schedule reminder in 12 hours
    reminder_key = kkey(cid, batch)
    due = (datetime.now(BOT_TZ) + timedelta(hours=12)).isoformat()
    state["reminders"][reminder_key] = {"chat_id": int(cid), "batch": batch, "due": due}
    save_state()
    lines = []
    lines.append(f"📦 *Machine {machine_label(cid)}* — Batch {batch}, Operator {feed.get('operator','')}")
    lines.append(f"• Date {_norm_date(feed.get('date'))}")
    keys = ("radial","nylon","chips","powder")
    feed_line = " • Feed: " + ", ".join([f"{k.title()} {round(feed.get(k,0)/1000,3)}T" for k in keys if feed.get(k,0)])
    lines.append(feed_line)
    lines.append(f"📈 Predicted Oil: *{pred}%* (confidence {conf})\n")
    lines.append("*Recommended zone minutes (plan):*")
    lines.append(pretty_plan(plan))
    await update.message.reply_text("\n".join(lines), parse_mode=ParseMode.MARKDOWN)

async def handle_actual(update, context):
    txt = update.message.text
    actual = parse_actual(txt)
    cid = str(update.effective_chat.id)
    batch = actual.get("batch") or state.get("latest_feed",{}).get(cid,{}).get("batch","")
    state["last_actual_ts"][cid] = datetime.now(BOT_TZ).isoformat()
    # clear reminders for this batch
    remk = kkey(cid, batch)
    if remk in state["reminders"]:
        try: del state["reminders"][remk]
        except Exception: pass
    save_state()
    lf = state["latest_feed"].get(cid)
    plan = lf.get("plan") if lf else ZONE_RULES
    pred, conf = (lf.get("pred"), lf.get("conf")) if lf else predict_yield({})
    actual_oil = actual.get("oil", None)

    out_lines = []
    out_lines.append(f"🧾 *Machine {machine_label(cid)} • Batch {batch}*")
    out_lines.append(f"• { _norm_date(lf.get('date') if lf else None) }")
    if actual_oil is not None:
        out_lines.append(f"\n🛢️ *Predicted vs Actual Oil:* {pred}% → {actual_oil}% (conf {conf})")

    zone_entries = []
    for z,pmin in plan.items():
        key = z.replace(" ","")
        m = re.match(r"(\d{2,3}-\d{2,3})", z)
        kshort = m.group(1) if m else z
        if key in actual:
            a_min = actual[key]
            dev_min = a_min - pmin
            zone_entries.append((z, dev_min))
        elif kshort in actual:
            a_min = actual[kshort]
            dev_min = a_min - pmin
            zone_entries.append((z, dev_min))

    if zone_entries:
        out_lines.append("\n📊 Deviation vs plan (min & hh:mm):")
        for z, dev in zone_entries:
            mm = int(round(dev))
            hhmm = minutes_to_hhmm(mm)
            out_lines.append(f"• {z}: {mm:+d} min ({hhmm})")
    else:
        out_lines.append("\n📊 Deviation vs plan (min):\n(no matching zones found in your message)")

    if actual_oil is not None:
        tips = oil_yield_reco(pred, actual_oil, plan, actual)
        out_lines.append("\n🛠️ Recommendations to lift oil yield:")
        for t in tips:
            out_lines.append(f"• {t}")
        if lf and lf.get("feed"):
            try:
                learn_from_actual(lf["feed"], actual_oil)
            except Exception as e:
                LOG.warning("Learning failed: %s", e)

    if zone_entries:
        out_lines.append("\n🔧 Quick zone moves (mins):")
        for z, dev in zone_entries:
            delta = -int(round(dev))
            if abs(delta) < 3:
                continue
            sign = "+" if delta>0 else ""
            out_lines.append(f"• {z}: {sign}{delta} min")
    save_state()

    # attach chart
    try:
        buf = bar_plan_vs_actual(plan, actual)
        msg_txt = "\n".join(out_lines)
        await update.message.reply_text(msg_txt, parse_mode=ParseMode.MARKDOWN)
        # send image
        buf.seek(0)
        await update.message.reply_photo(photo=InputFile(buf, filename="plan_vs_actual.png"))
    except Exception as e:
        LOG.warning("chart send failed: %s", e)
        await update.message.reply_text("\n".join(out_lines), parse_mode=ParseMode.MARKDOWN)

async def handle_what_feed(update, context):
    txt = update.message.text or ""
    m = re.search(r"target\s*=\s*([\d.]+)", txt)
    if not m:
        await update.message.reply_text("Please specify a `target=XX.X` oil% (e.g. `what feed: target=50.0`)", parse_mode=ParseMode.MARKDOWN)
        return
    target = float(m.group(1))
    base = {"radial":4500.0, "nylon":500.0, "chips":3500.0, "powder":1500.0, "kachra":0.0, "others":0.0}
    keys = ("radial","nylon","chips","powder","kachra","others")
    candidates = []
    for factor_pow in [1.0, 1.2, 1.5]:
        for reduce_radial in [1.0, 0.9, 0.8]:
            f = dict(base)
            f["powder"] *= factor_pow
            f["radial"] *= reduce_radial
            pred,_ = predict_yield(f)
            candidates.append((pred, f))
    candidates = sorted(candidates, key=lambda x: (abs(x[0]-target), -x[0]))
    top = candidates[:3]
    lines = [f"🔍 *Feed suggestions to aim for {target}% oil:*"]
    for pred, f in top:
        line = f"• Pred {pred}% → Feed: " + ", ".join([f"{k}:{round(f[k]/1000,3)}T" for k in keys if f[k]>0])
        lines.append(line)
    lines.append("\nAlso use engine plan with proposed feed to get recommended zones.")
    await update.message.reply_text("\n".join(lines), parse_mode=ParseMode.MARKDOWN)

# ---- main ----
async def main():
    if not TOKEN:
        LOG.error("Missing TELEGRAM_BOT_TOKEN environment variable.")
        return

    load_state()
    maybe_load_yield_weights()
    global ZONE_RULES, ENGINE
    ZONE_RULES = load_zone_rules()
    ENGINE = RecoEngine(ZONE_RULES)

    app = ApplicationBuilder().token(TOKEN).build()

    app.add_handler(CommandHandler("start", cmd_start))
    app.add_handler(CommandHandler("help", cmd_help))
    app.add_handler(CommandHandler("id", cmd_id))
    app.add_handler(CommandHandler("reload", cmd_reload))
    app.add_handler(CommandHandler("status", cmd_status))
    app.add_handler(MessageHandler(filters.TEXT & ~filters.COMMAND, handle_message))

    # Scheduler: schedule coroutine functions directly (pass app as arg)
    scheduler = AsyncIOScheduler(timezone=BOT_TZ)
    # reminder_tick every hour (start ~10s after launch)
    scheduler.add_job(reminder_tick, 'interval', hours=1, next_run_time=datetime.now(BOT_TZ)+timedelta(seconds=10), args=[app])
    # daily summary at 21:35 IST
    scheduler.add_job(daily_summary_job, 'cron', hour=21, minute=35, args=[app])
    scheduler.start()

    LOG.info("✅ PyroVision Assistant running (initialize & start)...")

    # Instead of await app.run_polling() which may cause "event loop already running" in some hosts,
    # we manually initialize/start and then wait on an asyncio Event. This avoids nested run_until_complete
    # calls and allows the process to be run in environments where an event loop already exists.
    await app.initialize()
    await app.start()

    stop_event = asyncio.Event()
    try:
        # keep running until stop_event is set or KeyboardInterrupt
        await stop_event.wait()
    except (KeyboardInterrupt, asyncio.CancelledError):
        LOG.info("Shutdown requested, stopping...")
    finally:
        try:
            await app.stop()
        except Exception as e:
            LOG.warning("Error during app.stop(): %s", e)
        try:
            await app.shutdown()
        except Exception as e:
            LOG.warning("Error during app.shutdown(): %s", e)

# safe startup wrapper -> works when an event loop already exists (e.g., in some containers)
def _start_bot_safely():
    try:
        loop = asyncio.get_running_loop()
    except RuntimeError:
        loop = None

    if loop and loop.is_running():
        LOG.info("Detected running event loop — scheduling bot as a task.")
        # schedule main as a task on existing loop
        asyncio.create_task(main())
    else:
        LOG.info("No running event loop — launching bot with asyncio.run().")
        try:
            asyncio.run(main())
        except KeyboardInterrupt:
            LOG.info("Shutting down...")

if __name__ == "__main__":
    _start_bot_safely()
