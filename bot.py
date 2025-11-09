# bot.py
# Telegram bot for tyre-pyrolysis: Feed -> zone plan (minutes); Actual -> compare vs plan.
# Now supports multi-group routing (machine groups -> summary group) with metadata.
# Requires: python-telegram-bot >= 21, pandas, numpy, openpyxl

import os
import re
import json
import logging
from datetime import datetime, timezone, timedelta

import numpy as np
import pandas as pd
from telegram import Update
from telegram.ext import Application, CommandHandler, MessageHandler, filters, ContextTypes

# ===================== CONFIG =====================

TELEGRAM_BOT_TOKEN = os.environ.get("TELEGRAM_BOT_TOKEN", "")
REPORT_PATH = os.environ.get("REPORT_PATH", "pyrolysis_feed_temp_ZONE_TIME_report.xlsx")
ZONE_SHEET = os.environ.get("ZONE_SHEET", "ZoneTime_Recommendations")

# Summary group chat id (post copies here)
SUMMARY_CHAT_ID = os.environ.get("SUMMARY_CHAT_ID", "")  # e.g. "-1001234567890"

# Map of machine group chat_id -> machine label.
# Set env MACHINE_MAP to a JSON string, e.g.:
# {"-100111...":"Machine 1296 (R-1)","-100222...":"Machine 1297 (R-2)","-100333...":"Machine 1298 (R-3)","-100444...":"Machine 1299 (R-4)"}
MACHINE_MAP = {}
_raw_map = os.environ.get("MACHINE_MAP", "")
if _raw_map:
    try:
        MACHINE_MAP = json.loads(_raw_map)
    except Exception:
        pass

# If you prefer, you can hardcode while testing:
# MACHINE_MAP = {
#     "-100111...": "Machine 1296 (R-1)",
#     "-100222...": "Machine 1297 (R-2)",
#     "-100333...": "Machine 1298 (R-3)",
#     "-100444...": "Machine 1299 (R-4)",
# }

# Optional: set your Indian time (UTC+5:30) for stamps
IST = timezone(timedelta(hours=5, minutes=30))

# ===================== UTILITIES =====================

def normalize_zone_key(s: str) -> str:
    s = s.strip().lower()
    s = s.replace("°c", "").replace("c", "")
    s = s.replace("temp", "").replace("zone", "").replace("minutes", "").replace("mins", "").replace("min", "")
    s = s.replace("reactor", "reactor").replace("separator", "separator")
    s = re.sub(r"\s+", " ", s).strip()
    if "separator" in s:
        return re.sub(r"[^0-9\-]", "", s.split("separator")[0]).strip() + " separator"
    elif "reactor" in s:
        return re.sub(r"[^0-9\-]", "", s.split("reactor")[0]).strip() + " reactor"
    else:
        return re.sub(r"[^0-9\-]", "", s) + " reactor"

def to_hhmmss(minutes: float) -> str:
    if minutes is None or np.isnan(minutes):
        return "-"
    total_seconds = int(round(minutes * 60))
    h = total_seconds // 3600
    m = (total_seconds % 3600) // 60
    s = total_seconds % 60
    return f"{h:02d}:{m:02d}:{s:02d}"

def hhmmss_to_minutes(s: str) -> float:
    s = s.strip()
    if not s or s in ("0", "00:00", "00:00:00"):
        return 0.0
    parts = s.split(":")
    if len(parts) == 3:
        h, m, sec = map(float, parts)
        return h*60 + m + sec/60.0
    elif len(parts) == 2:
        m, sec = map(float, parts)
        return m + sec/60.0
    else:
        return float(s)

def fmt_machine(chat_id: int) -> str:
    return MACHINE_MAP.get(str(chat_id), f"Chat {chat_id}")

def extract_metadata(text: str, fallback_machine: str) -> dict:
    """
    Parse optional metadata from the user's line:
      ... ; operator=Name ; date=2025-11-09 ; machine=R-2
    Case-insensitive; separators can be ';' or newline.
    If not present, auto-fill.
    """
    meta = {"operator": None, "date": None, "machine": None}
    pieces = re.split(r"[;\n]+", text)
    for p in pieces:
        kv = p.split("=", 1)
        if len(kv) != 2:
            continue
        k = kv[0].strip().lower()
        v = kv[1].strip()
        if k in ("operator", "op", "operater"):
            meta["operator"] = v
        elif k in ("date", "dt"):
            meta["date"] = v
        elif k in ("machine", "machine_no", "machine_name"):
            meta["machine"] = v

    if not meta["date"]:
        meta["date"] = datetime.now(IST).strftime("%Y-%m-%d")
    if not meta["machine"]:
        meta["machine"] = fallback_machine
    return meta

def banner(title: str, meta: dict) -> str:
    parts = [f"**{title}**"]
    if meta.get("machine"):
        parts.append(f"• Machine: {meta['machine']}")
    if meta.get("operator"):
        parts.append(f"• Operator: {meta['operator']}")
    if meta.get("date"):
        parts.append(f"• Date: {meta['date']}")
    return " | ".join(parts)

# ===================== RECOMMENDER =====================

class RecoEngine:
    def __init__(self, report_path: str):
        self.report_path = report_path
        self.zone_map = {}
        self._load_report()

    def _load_report(self):
        self.zone_map.clear()
        try:
            df = pd.read_excel(self.report_path, sheet_name=ZONE_SHEET)
            for _, row in df.iterrows():
                feat = str(row.get("zone_time_feature", "")).strip()
                sug = row.get("suggested_minutes", np.nan)
                if not feat:
                    continue
                low_feat = feat.lower()
                m = re.search(r"(\d{2,3}\s*[-–]\s*\d{2,3})", low_feat)
                if not m:
                    continue
                window = m.group(1)
                window = window.replace(" ", "").replace("–", "-")
                which = "separator" if "separator" in low_feat else "reactor"
                key = f"{window} {which}"
                self.zone_map[key] = float(sug) if pd.notna(sug) else np.nan
        except Exception:
            # conservative defaults (minutes)
            self.zone_map.update({
                "50-200 reactor": 120.0,
                "200-300 reactor": 75.0,
                "300-400 reactor": 180.0,
                "400-450 reactor": 40.0,
                "450-480 reactor": 20.0,
                "500-520 reactor": 10.0,
                "300-360 separator": 35.0,
            })

    def plan_from_feed(self, feed_text: str) -> dict:
        _ = self._parse_feed(feed_text)
        return dict(self.zone_map)

    def compare_actuals(self, actual_text: str, plan_minutes: dict):
        actual_map = self._parse_actuals(actual_text)
        tips = []
        deltas = {}
        for k, target_min in plan_minutes.items():
            if k not in actual_map or np.isnan(target_min):
                continue
            actual_min = actual_map[k]
            deltas[k] = actual_min - target_min
            if abs(deltas[k]) >= 5.0:
                direction = "reduce" if deltas[k] > 0 else "increase"
                tips.append(f"{direction} {k} by ~{abs(deltas[k]):.0f} min")
        if not tips:
            tips = ["Near-optimal vs plan. Keep the same profile."]
        return deltas, tips

    def _parse_feed(self, text: str) -> dict:
        pairs = re.split(r"[,\n]+", text.replace("Feed:", "").strip())
        out = {}
        for p in pairs:
            m = re.search(r"([A-Za-z\s/()]+)\s*=\s*([0-9.]+)\s*([Tt]|[Kk][Gg])?", p.strip())
            if m:
                name = m.group(1).strip().lower()
                val = float(m.group(2))
                unit = (m.group(3) or "").lower()
                if unit == "t":
                    val = val * 1000.0
                out[name] = val
        return out

    def _parse_actuals(self, text: str) -> dict:
        text = text.replace("Actual:", "").strip()
        pieces = re.split(r"[,\n]+", text)
        out = {}
        for p in pieces:
            if "=" not in p:
                continue
            left, right = p.split("=", 1)
            key = normalize_zone_key(left)
            val_text = right.strip()
            try:
                minutes = hhmmss_to_minutes(val_text) if ":" in val_text else float(val_text)
                out[key] = minutes
            except Exception:
                continue
        return out

engine = RecoEngine(REPORT_PATH)
latest_plan_cache = {}  # chat_id -> plan dict

def format_plan(plan: dict) -> str:
    def key_order(k):
        m = re.match(r"(\d{2,3})-(\d{2,3})", k)
        a = int(m.group(1)) if m else 0
        b = int(m.group(2)) if m else 0
        return (a, b, ("separator" in k))
    lines = []
    for z in sorted(plan.keys(), key=key_order):
        lines.append(f"{z}: {to_hhmmss(plan[z])}")
    return "\n".join(lines)

async def send_both(context: ContextTypes.DEFAULT_TYPE, origin_chat: int, text: str):
    """Send to origin chat; if SUMMARY_CHAT_ID set and differs, also send to summary."""
    await context.bot.send_message(chat_id=origin_chat, text=text, parse_mode="Markdown")
    if SUMMARY_CHAT_ID and str(origin_chat) != str(SUMMARY_CHAT_ID):
        try:
            await context.bot.send_message(chat_id=int(SUMMARY_CHAT_ID), text=text, parse_mode="Markdown")
        except Exception as e:
            logging.warning(f"Failed to send to summary: {e}")

# ===================== HANDLERS =====================

async def start(update: Update, context: ContextTypes.DEFAULT_TYPE):
    mlabel = fmt_machine(update.effective_chat.id)
    msg = (
        f"Hi! I’m your Excel + AI assistant for pyrolysis.\n"
        f"Detected room → *{mlabel}*\n\n"
        "Use:\n"
        "• `Feed: Nylon=2.0T, Radial=5.0T, Chips=3.5T, Powder=1.0T; operator=Rahul; date=2025-11-09`\n"
        "    → I’ll reply with *zone minutes* (here + summary).\n"
        "• `Actual: 50-200=01:30:00, 200-300=01:10:00, 300-400=02:45:00` (meta optional)\n"
        "    → I’ll compare to plan and give *what to change next time*.\n\n"
        "Commands: /plan, /actual, /reload, /help"
    )
    await update.message.reply_text(msg, parse_mode="Markdown")

async def help_cmd(update: Update, context: ContextTypes.DEFAULT_TYPE):
    await start(update, context)

async def plan_cmd(update: Update, context: ContextTypes.DEFAULT_TYPE):
    chat_id = update.effective_chat.id
    mlabel = fmt_machine(chat_id)
    raw = update.message.text
    # Support both "/plan ..." and plain "Feed: ..."
    payload = raw.replace("/plan", "Feed:").strip()
    meta = extract_metadata(payload, mlabel)

    plan = engine.plan_from_feed(payload)
    latest_plan_cache[chat_id] = plan

    header = banner("Recommended zone minutes", meta)
    body = format_plan(plan)
    await send_both(context, chat_id, f"{header}\n{body}")

async def actual_cmd(update: Update, context: ContextTypes.DEFAULT_TYPE):
    chat_id = update.effective_chat.id
    mlabel = fmt_machine(chat_id)
    raw = update.message.text
    payload = raw.replace("/actual", "Actual:").strip()
    meta = extract_metadata(payload, mlabel)

    plan = latest_plan_cache.get(chat_id, engine.plan_from_feed("Feed:"))
    deltas, tips = engine.compare_actuals(payload, plan)
    if not deltas:
        await update.message.reply_text(
            "Couldn’t read your actuals.\nExample:\n"
            "Actual: 50-200=01:10:00, 200-300=00:45:00, 300-400=01:20:00",
        )
        return

    header = banner("Deviation vs plan (min)", meta)
    lines = [header]
    for k in sorted(deltas.keys()):
        lines.append(f"{k}: {deltas[k]:+.0f}")
    lines.append("\n*Recommendations:*")
    for t in tips:
        lines.append(f"• {t}")

    await send_both(context, chat_id, "\n".join(lines))

async def reload_cmd(update: Update, context: ContextTypes.DEFAULT_TYPE):
    engine._load_report()
    await update.message.reply_text("Reloaded recommendations from report ✅")

async def text_router(update: Update, context: ContextTypes.DEFAULT_TYPE):
    chat_id = update.effective_chat.id
    mlabel = fmt_machine(chat_id)
    text = (update.message.text or "").strip()
    if text.lower().startswith("feed:"):
        meta = extract_metadata(text, mlabel)
        plan = engine.plan_from_feed(text)
        latest_plan_cache[chat_id] = plan
        header = banner("Recommended zone minutes", meta)
        await send_both(context, chat_id, f"{header}\n{format_plan(plan)}")
    elif text.lower().startswith("actual:"):
        meta = extract_metadata(text, mlabel)
        plan = latest_plan_cache.get(chat_id, engine.plan_from_feed("Feed:"))
        deltas, tips = engine.compare_actuals(text, plan)
        if not deltas:
            await update.message.reply_text(
                "Couldn’t read your actuals.\nExample:\n"
                "Actual: 50-200=01:10:00, 200-300=00:45:00, 300-400=01:20:00"
            )
            return
        header = banner("Deviation vs plan (min)", meta)
        lines = [header]
        for k in sorted(deltas.keys()):
            lines.append(f"{k}: {deltas[k]:+.0f}")
        lines.append("\n*Recommendations:*")
        for t in tips:
            lines.append(f"• {t}")
        await send_both(context, chat_id, "\n".join(lines))
    else:
        await help_cmd(update, context)

# ===================== MAIN =====================

def main():
    logging.basicConfig(level=logging.INFO)
    if not TELEGRAM_BOT_TOKEN:
        print("ERROR: Set TELEGRAM_BOT_TOKEN env var.")
        return

    app = Application.builder().token(TELEGRAM_BOT_TOKEN).build()
    app.add_handler(CommandHandler("start", start))
    app.add_handler(CommandHandler("help", help_cmd))
    app.add_handler(CommandHandler("plan", plan_cmd))
    app.add_handler(CommandHandler("actual", actual_cmd))
    app.add_handler(CommandHandler("reload", reload_cmd))
    app.add_handler(MessageHandler(filters.TEXT & (~filters.COMMAND), text_router))

    app.run_polling()

if __name__ == "__main__":
    main()
