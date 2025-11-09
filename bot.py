# bot.py
# Telegram bot for Tyre Pyrolysis: feed -> zone plan (minutes), actuals -> gap + recos
# Works in private chat AND (recommended) in ONE plant group by Chat ID.
# Requirements: python-telegram-bot>=21.0, pandas, numpy, openpyxl

import os
import re
import logging
from typing import Dict, List, Tuple

import numpy as np
import pandas as pd
from telegram import Update
from telegram.ext import (
    Application, CommandHandler, MessageHandler, ContextTypes, filters
)

# ── Logging ────────────────────────────────────────────────────────────────────
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)s | %(name)s | %(message)s"
)
log = logging.getLogger("pyro_bot")

# ── Config (env vars with safe defaults) ───────────────────────────────────────
TELEGRAM_BOT_TOKEN = os.environ.get("TELEGRAM_BOT_TOKEN", "")
REPORT_PATH = os.environ.get("REPORT_PATH", "pyrolysis_feed_temp_ZONE_TIME_report.xlsx")

# If you want to restrict the bot to ONE group, set ALLOWED_CHAT_ID here
# or via Railway env var. Leave as None to allow anywhere.
ENV_ALLOWED = os.environ.get("ALLOWED_CHAT_ID", "").strip()
ALLOWED_CHAT_ID = int(ENV_ALLOWED) if ENV_ALLOWED else -1001234567890  # <— REPLACE THIS

# messages visible to everyone
HELP_TEXT = (
    "Commands:\n"
    "/plan <feed line>  — same as sending a Feed: message\n"
    "/actual <actuals>  — same as sending an Actual: message\n"
    "/reload            — reload Excel recommendations\n\n"
    "Examples:\n"
    "Feed: Nylon=2.0T, Radial=5.0T, Chips=3.5T, Powder=0.6T\n"
    "Actual: 50-200=03:10:00, 200-300=01:00:00, 300-400=02:55:00, 400-450=01:10:00, 450-480=00:25:00\n"
)

# ── Utilities ──────────────────────────────────────────────────────────────────
def to_hhmmss(minutes: float) -> str:
    if minutes is None or (isinstance(minutes, float) and np.isnan(minutes)):
        return "-"
    total = int(round(minutes * 60))
    h, m = divmod(total, 60)
    return f"{h:02d}:{m:02d}:00"

def hhmmss_to_minutes(s: str) -> float:
    s = s.strip()
    if not s:
        return 0.0
    parts = s.split(":")
    if len(parts) == 3:
        h, m, sec = map(float, parts)
        return h * 60 + m + sec / 60.0
    if len(parts) == 2:
        m, sec = map(float, parts)
        return m + sec / 60.0
    return float(s)

def normalize_zone_key(s: str) -> str:
    """Map '50-200 reactor', '300-400 separator', etc."""
    x = s.strip().lower()
    x = x.replace("°c", "").replace("c", "")
    x = x.replace("temp", "").replace("zone", "")
    x = x.replace("minutes", "").replace("mins", "").replace("min", "")
    x = re.sub(r"\s+", " ", x).strip()
    which = "separator" if "separator" in x else "reactor"
    # keep only digits and dash around the window part
    m = re.search(r"(\d{2,3}\s*[-–]\s*\d{2,3})", x)
    if not m:
        # fallback: try to take first number block found
        nums = re.findall(r"\d+", x)
        if len(nums) >= 2:
            window = f"{nums[0]}-{nums[1]}"
        else:
            window = x
    else:
        window = m.group(1)
    window = window.replace(" ", "").replace("–", "-")
    return f"{window} {which}"

# ── Engine (loads Excel) ───────────────────────────────────────────────────────
class RecoEngine:
    """
    Reads the Excel sheet and builds a dict:
      key: '50-200 reactor' / '300-400 separator'
      val: suggested minutes (float)
    """
    SHEET = "ZoneTime_Recommendations"  # this sheet must exist in the report

    def __init__(self, path: str):
        self.path = path
        self.zone_map: Dict[str, float] = {}
        self._load()

    def _load(self):
        self.zone_map.clear()
        try:
            df = pd.read_excel(self.path, sheet_name=self.SHEET)
            # Expect columns: zone_time_feature, suggested_minutes
            # e.g. "50-200 temp zone Reactor [min]"
            for _, row in df.iterrows():
                feat = str(row.get("zone_time_feature", "")).strip()
                minutes = row.get("suggested_minutes", np.nan)
                if not feat:
                    continue
                low = feat.lower()
                m = re.search(r"(\d{2,3}\s*[-–]\s*\d{2,3})", low)
                if not m:
                    continue
                window = m.group(1).replace(" ", "").replace("–", "-")
                which = "separator" if "separator" in low else "reactor"
                key = f"{window} {which}"
                self.zone_map[key] = float(minutes) if pd.notna(minutes) else np.nan

            if not self.zone_map:
                raise ValueError("No rows parsed from Excel.")

            log.info("Loaded %d zone rules from %s", len(self.zone_map), self.path)

        except Exception as e:
            log.exception("Failed reading %s; using conservative defaults", self.path)
            # Safe defaults (minutes). Adjust these if you like.
            self.zone_map.update({
                "50-200 reactor": 160.0,
                "200-300 reactor": 60.0,
                "300-400 reactor": 175.0,
                "400-450 reactor": 65.0,
                "450-480 reactor": 25.0,
                "480-500 reactor": 12.0,
                "500-520 reactor": 12.0,
                "300-400 separator": 40.0,
            })

    def reload(self) -> int:
        self._load()
        return len(self.zone_map)

    # Currently plan is global; feed-aware tuning can be added later
    def plan_from_feed(self, feed_text: str) -> Dict[str, float]:
        _ = self._parse_feed(feed_text)
        return dict(self.zone_map)

    def compare_actuals(self, actual_text: str, plan_minutes: Dict[str, float]) -> Tuple[Dict[str, float], List[str]]:
        actual_map = self._parse_actuals(actual_text)
        deltas, tips = {}, []
        for k, tgt in plan_minutes.items():
            if k not in actual_map or (isinstance(tgt, float) and np.isnan(tgt)):
                continue
            act = actual_map[k]
            delta = act - tgt
            deltas[k] = delta
            if abs(delta) >= 5.0:
                tips.append(("reduce" if delta > 0 else "increase") + f" {k} by ~{abs(delta):.0f} min")
        if not tips:
            tips = ["Near-optimal execution vs plan. Keep the same profile."]
        return deltas, tips

    # Parsers
    def _parse_feed(self, text: str) -> Dict[str, float]:
        txt = text.replace("feed:", "").replace("Feed:", "").strip()
        pairs = re.split(r"[,\n]+", txt)
        out = {}
        for p in pairs:
            m = re.search(r"([A-Za-z\s/()]+)\s*=\s*([0-9.]+)\s*([Tt]|[Kk][Gg])?", p.strip())
            if not m:
                continue
            name = m.group(1).strip().lower()
            val = float(m.group(2))
            unit = (m.group(3) or "").lower()
            if unit == "t":
                val *= 1000.0
            out[name] = val
        return out

    def _parse_actuals(self, text: str) -> Dict[str, float]:
        t = text.replace("actual:", "").replace("Actual:", "").strip()
        out = {}
        for piece in re.split(r"[,\n]+", t):
            if "=" not in piece:
                continue
            left, right = piece.split("=", 1)
            key = normalize_zone_key(left)
            val = right.strip()
            try:
                minutes = hhmmss_to_minutes(val) if ":" in val else float(val)
                out[key] = minutes
            except Exception:
                continue
        return out

# ── Bot helpers ────────────────────────────────────────────────────────────────
engine = RecoEngine(REPORT_PATH)
latest_plan_cache: Dict[int, Dict[str, float]] = {}  # chat_id -> plan

def format_plan(plan: Dict[str, float]) -> str:
    def sort_key(k):
        m = re.match(r"(\d{2,3})-(\d{2,3})", k)
        a = int(m.group(1)) if m else 0
        b = int(m.group(2)) if m else 0
        return (a, b, "separator" in k)
    lines = ["Recommended zone minutes (from data):"]
    for k in sorted(plan.keys(), key=sort_key):
        lines.append(f"{k}: {to_hhmmss(plan[k])}")
    return "\n".join(lines)

def allowed_chat(update: Update) -> bool:
    """Return True if this chat is allowed (group restriction)."""
    if not ALLOWED_CHAT_ID:
        return True
    chat_id = update.effective_chat.id
    # Allow the set group, and allow private chats with the owner (optional)
    return chat_id == ALLOWED_CHAT_ID or chat_id > 0  # private chats have positive IDs

async def guard_group(update: Update) -> bool:
    """Politely redirect if used outside the group when restricted."""
    if allowed_chat(update):
        return True
    await update.message.reply_text("Please use me inside the official plant group.")
    return False

# ── Handlers ──────────────────────────────────────────────────────────────────
async def start(update: Update, ctx: ContextTypes.DEFAULT_TYPE):
    if not await guard_group(update): return
    await update.message.reply_text("PyroVision Assistant ready ✅\n\n" + HELP_TEXT)

async def help_cmd(update: Update, ctx: ContextTypes.DEFAULT_TYPE):
    if not await guard_group(update): return
    await update.message.reply_text(HELP_TEXT)

async def plan_cmd(update: Update, ctx: ContextTypes.DEFAULT_TYPE):
    if not await guard_group(update): return
    text = update.message.text
    feed_line = text.replace("/plan", "Feed:").strip()
    plan = engine.plan_from_feed(feed_line)
    latest_plan_cache[update.effective_chat.id] = plan
    await update.message.reply_text(format_plan(plan))

async def actual_cmd(update: Update, ctx: ContextTypes.DEFAULT_TYPE):
    if not await guard_group(update): return
    text = update.message.text
    actual_line = text.replace("/actual", "Actual:").strip()
    plan = latest_plan_cache.get(update.effective_chat.id, engine.plan_from_feed("Feed:"))
    deltas, tips = engine.compare_actuals(actual_line, plan)
    if not deltas:
        await update.message.reply_text(
            "Couldn’t read your actuals. Example:\n"
            "Actual: 50-200=01:10:00, 200-300=00:45:00, 300-400=01:20:00, 400-450=01:00:00, 450-480=00:25:00"
        )
        return
    lines = ["Deviation vs plan (minutes):"]
    for k in sorted(deltas.keys()):
        lines.append(f"{k}: {deltas[k]:+.0f}")
    lines.append("\nRecommendations:")
    for t in tips: lines.append(f"• {t}")
    await update.message.reply_text("\n".join(lines))

async def reload_cmd(update: Update, ctx: ContextTypes.DEFAULT_TYPE):
    if not await guard_group(update): return
    n = engine.reload()
    await update.message.reply_text(f"Reloaded recommendations from Excel ✅ ({n} rules)")

async def text_router(update: Update, ctx: ContextTypes.DEFAULT_TYPE):
    if not await guard_group(update): return
    text = (update.message.text or "").strip()
    if text.lower().startswith("feed:"):
        plan = engine.plan_from_feed(text)
        latest_plan_cache[update.effective_chat.id] = plan
        await update.message.reply_text(format_plan(plan))
    elif text.lower().startswith("actual:"):
        plan = latest_plan_cache.get(update.effective_chat.id, engine.plan_from_feed("Feed:"))
        deltas, tips = engine.compare_actuals(text, plan)
        if not deltas:
            await update.message.reply_text(
                "Couldn’t read your actuals. Example:\n"
                "Actual: 50-200=01:10:00, 200-300=00:45:00, 300-400=01:20:00, 400-450=01:00:00, 450-480=00:25:00"
            )
            return
        lines = ["Deviation vs plan (minutes):"]
        for k in sorted(deltas.keys()):
            lines.append(f"{k}: {deltas[k]:+.0f}")
        lines.append("\nRecommendations:")
        for t in tips: lines.append(f"• {t}")
        await update.message.reply_text("\n".join(lines))
    else:
        await update.message.reply_text("I didn’t understand.\n\n" + HELP_TEXT)

# ── Main ──────────────────────────────────────────────────────────────────────
def main():
    if not TELEGRAM_BOT_TOKEN:
        print("ERROR: set TELEGRAM_BOT_TOKEN environment variable.")
        return

    app = Application.builder().token(TELEGRAM_BOT_TOKEN).build()

    app.add_handler(CommandHandler("start", start))
    app.add_handler(CommandHandler("help", help_cmd))
    app.add_handler(CommandHandler("plan", plan_cmd))
    app.add_handler(CommandHandler("actual", actual_cmd))
    app.add_handler(CommandHandler("reload", reload_cmd))
    app.add_handler(MessageHandler(filters.TEXT & (~filters.COMMAND), text_router))

    log.info("Bot starting…")
    app.run_polling(allowed_updates=Update.ALL_TYPES)

if __name__ == "__main__":
    main()
