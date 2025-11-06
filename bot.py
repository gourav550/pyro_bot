# bot.py
# Telegram bot: feed -> zone plan (minutes), then actuals -> gap + recommendations
# Requires: python-telegram-bot >= 20, pandas, numpy, openpyxl

import os
import re
import asyncio
import logging
from datetime import timedelta

import pandas as pd
import numpy as np
from telegram import Update
from telegram.ext import Application, CommandHandler, MessageHandler, filters, ContextTypes

# ==== CONFIG ====
TELEGRAM_BOT_TOKEN = os.environ.get("TELEGRAM_BOT_TOKEN", "YOUR_TELEGRAM_BOT_TOKEN_HERE")
REPORT_PATH = os.environ.get("REPORT_PATH", "pyrolysis_feed_temp_ZONE_TIME_report.xlsx")
ZONE_SHEET = "ZoneTime_Recommendations"  # sheet inside the report

# Map friendly zone keys users will send to the feature names in the report
# We'll normalize user inputs like "50-200", "50-200 reactor", etc.
def normalize_zone_key(s: str) -> str:
    s = s.strip().lower()
    s = s.replace("°c", "").replace("c", "")
    s = s.replace("temp", "").replace("zone", "").replace("minutes", "").replace("mins", "").replace("min", "")
    s = s.replace("reactor", "reactor").replace("separator", "separator")
    s = re.sub(r"\s+", " ", s).strip()
    # common forms: "50-200 reactor", "300-400 separator"
    # if separator/reactor not specified, default to reactor
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
        # allow pure minutes
        return float(s)

class RecoEngine:
    def __init__(self, report_path: str):
        self.report_path = report_path
        self.zone_map = {}  # key: "50-200 reactor" -> suggested_minutes(float)
        self.feed_influence = {}  # optional future use
        self._load_report()

    def _load_report(self):
        self.zone_map.clear()
        try:
            df = pd.read_excel(self.report_path, sheet_name=ZONE_SHEET)
            # Expect columns: zone_time_feature, suggested_minutes
            # zone_time_feature examples: "50-200 temp zone Reactor [min]"
            for _, row in df.iterrows():
                feat = str(row.get("zone_time_feature", "")).strip()
                sug = row.get("suggested_minutes", np.nan)
                if not feat:
                    continue
                # normalize to "50-200 reactor" / "50-200 separator"
                low_feat = feat.lower()
                # find window like "50-200"
                m = re.search(r"(\d{2,3}\s*[-–]\s*\d{2,3})", low_feat)
                if not m:
                    continue
                window = m.group(1)
                window = window.replace(" ", "").replace("–", "-")
                which = "separator" if "separator" in low_feat else "reactor"
                key = f"{window} {which}"
                self.zone_map[key] = float(sug) if pd.notna(sug) else np.nan
        except Exception as e:
            logging.exception("Failed to load report recommendations")
            # fallback: very conservative defaults in minutes
            defaults = {
                "50-200 reactor": 60.0,
                "200-300 reactor": 60.0,
                "300-400 reactor": 70.0,
                "400-450 reactor": 80.0,
                "450-480 reactor": 25.0,
                "480-500 reactor": 15.0,
                "300-400 separator": 30.0,
            }
            self.zone_map.update(defaults)

    def plan_from_feed(self, feed_text: str) -> dict:
        """
        For now, we use global suggested minutes per zone (data-driven from your report).
        Parsing feed keeps interface ready for future feed-aware tuning.
        """
        # parse feed pairs like "Nylon=4.5T" or "Radial=5500kg"
        _ = self._parse_feed(feed_text)
        # return minutes plan
        return dict(self.zone_map)

    def compare_actuals(self, actual_text: str, plan_minutes: dict) -> tuple[dict, list[str]]:
        """
        Compare Actual: "50-200=01:10:00, 200-300=00:45:00, 300-400=01:20:00, ..."
        to plan_minutes and create actionable tips.
        """
        actual_map = self._parse_actuals(actual_text)
        tips = []
        deltas = {}
        for k, target_min in plan_minutes.items():
            if k not in actual_map or np.isnan(target_min):
                continue
            actual_min = actual_map[k]
            deltas[k] = actual_min - target_min  # +ve = overshoot, -ve = under
            # rule-of-thumb thresholds
            if abs(deltas[k]) >= 5.0:
                direction = "reduce" if deltas[k] > 0 else "increase"
                tips.append(f"{direction} {k} by ~{abs(deltas[k]):.0f} min")
        if not tips:
            tips = ["Near-optimal execution vs plan. Keep the same profile."]
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
        # Accept forms like: "50-200=01:10:00", "50-200 reactor=70", "300-400 separator=00:30:00"
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
                if ":" in val_text:
                    minutes = hhmmss_to_minutes(val_text)
                else:
                    minutes = float(val_text)  # interpret as minutes
                out[key] = minutes
            except Exception:
                continue
        return out

# ==== Telegram handlers ====

engine = RecoEngine(REPORT_PATH)

HELP_TEXT = (
    "Send either:\n"
    "• Feed: Nylon=4.5T, Radial=5.5T, Kachra=0.6T, ...\n"
    "→ I’ll reply with zone plan (minutes) for Reactor & Separator.\n\n"
    "• Actual: 50-200=01:10:00, 200-300=00:45:00, 300-400=01:20:00, 400-450=01:00:00, 450-480=00:25:00, 480-500=00:10:00, 300-400 separator=00:30:00\n"
    "→ I’ll compare to plan and tell you what to change next time.\n\n"
    "Commands:\n"
    "/plan <feed line>   – Same as sending a Feed: message\n"
    "/actual <actuals>   – Same as sending an Actual: message\n"
    "/reload             – Reload recommendations from the Excel report\n"
)

latest_plan_cache = {}  # chat_id -> plan dict

async def start(update: Update, context: ContextTypes.DEFAULT_TYPE):
    await update.message.reply_text("Bot ready ✅\n\n" + HELP_TEXT)

async def help_cmd(update: Update, context: ContextTypes.DEFAULT_TYPE):
    await update.message.reply_text(HELP_TEXT)

def format_plan(plan: dict) -> str:
    # Sort by numeric range ascending and group reactor/separator
    def key_order(k):
        m = re.match(r"(\d{2,3})-(\d{2,3})", k)
        a = int(m.group(1)) if m else 0
        b = int(m.group(2)) if m else 0
        return (a, b, ("separator" in k))
    lines = []
    for z in sorted(plan.keys(), key=key_order):
        lines.append(f"{z}: {to_hhmmss(plan[z])}")
    return "\n".join(lines)

async def plan_cmd(update: Update, context: ContextTypes.DEFAULT_TYPE):
    text = update.message.text
    feed = text.replace("/plan", "Feed:").strip()
    plan = engine.plan_from_feed(feed)
    latest_plan_cache[update.effective_chat.id] = plan
    msg = "Recommended zone minutes (from data):\n" + format_plan(plan)
    await update.message.reply_text(msg)

async def actual_cmd(update: Update, context: ContextTypes.DEFAULT_TYPE):
    text = update.message.text
    actual = text.replace("/actual", "Actual:").strip()
    plan = latest_plan_cache.get(update.effective_chat.id, engine.plan_from_feed("Feed:"))
    deltas, tips = engine.compare_actuals(actual, plan)
    if not deltas:
        await update.message.reply_text("Couldn’t read your actuals. Example:\nActual: 50-200=01:10:00, 200-300=00:45:00, 300-400=01:20:00, 400-450=01:00:00, 450-480=00:25:00, 480-500=00:10:00")
        return
    # Build response
    lines = ["Deviation vs plan (min):"]
    for k in sorted(deltas.keys()):
        lines.append(f"{k}: {deltas[k]:+.0f}")
    lines.append("\nRecommendations:")
    for t in tips:
        lines.append(f"• {t}")
    await update.message.reply_text("\n".join(lines))

async def reload_cmd(update: Update, context: ContextTypes.DEFAULT_TYPE):
    engine._load_report()
    await update.message.reply_text("Reloaded recommendations from report ✅")

async def text_router(update: Update, context: ContextTypes.DEFAULT_TYPE):
    text = (update.message.text or "").strip()
    if text.lower().startswith("feed:"):
        plan = engine.plan_from_feed(text)
        latest_plan_cache[update.effective_chat.id] = plan
        await update.message.reply_text("Recommended zone minutes (from data):\n" + format_plan(plan))
    elif text.lower().startswith("actual:"):
        plan = latest_plan_cache.get(update.effective_chat.id, engine.plan_from_feed("Feed:"))
        deltas, tips = engine.compare_actuals(text, plan)
        if not deltas:
            await update.message.reply_text("Couldn’t read your actuals. Example:\nActual: 50-200=01:10:00, 200-300=00:45:00, 300-400=01:20:00, 400-450=01:00:00, 450-480=00:25:00, 480-500=00:10:00")
            return
        lines = ["Deviation vs plan (min):"]
        for k in sorted(deltas.keys()):
            lines.append(f"{k}: {deltas[k]:+.0f}")
        lines.append("\nRecommendations:")
        for t in tips:
            lines.append(f"• {t}")
        await update.message.reply_text("\n".join(lines))
    else:
        await update.message.reply_text("I didn’t understand. " + HELP_TEXT)

async def main():
    logging.basicConfig(level=logging.INFO)
    if TELEGRAM_BOT_TOKEN == "YOUR_TELEGRAM_BOT_TOKEN_HERE":
        print("ERROR: Set TELEGRAM_BOT_TOKEN env var or edit the file.")
        return
    app = Application.builder().token(TELEGRAM_BOT_TOKEN).build()
    app.add_handler(CommandHandler("start", start))
    app.add_handler(CommandHandler("help", help_cmd))
    app.add_handler(CommandHandler("plan", plan_cmd))
    app.add_handler(CommandHandler("actual", actual_cmd))
    app.add_handler(CommandHandler("reload", reload_cmd))
    app.add_handler(MessageHandler(filters.TEXT & (~filters.COMMAND), text_router))

    print("✅ Bot ready.")
    await app.run_polling()

if __name__ == "__main__":
    asyncio.run(main())
