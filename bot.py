# bot.py
# PyroVision Assistant: Excel-driven zone plan + oil% prediction + adaptive learning
# Libraries: python-telegram-bot >= 21, pandas, numpy, openpyxl

import os
import re
import json
import logging
from pathlib import Path
from datetime import datetime
from typing import Dict, Tuple

import pandas as pd
import numpy as np
from telegram import Update
from telegram.ext import Application, CommandHandler, MessageHandler, ContextTypes, filters

# ========= ENV / CONFIG =========
TELEGRAM_BOT_TOKEN = os.environ.get("TELEGRAM_BOT_TOKEN", "")
REPORT_PATH        = os.environ.get("REPORT_PATH", "pyrolysis_feed_temp_ZONE_TIME_report.xlsx")
ZONE_SHEET         = os.environ.get("ZONE_SHEET", "ZoneTime_Recommendations")
SUMMARY_CHAT_ID    = os.environ.get("SUMMARY_CHAT_ID")  # e.g. "-1001234567890"
MACHINE_MAP_JSON   = os.environ.get("MACHINE_MAP", "{}") # {"-1001":"Machine 1296 (R-1)","-1002":"Machine 1297 (R-2)",...}
MODEL_STATE_PATH   = os.environ.get("MODEL_STATE_PATH", "model_state.json")

# ========= LOGGING =========
logger = logging.getLogger("pyro_bot")
logging.basicConfig(level=logging.INFO, format="%(asctime)s | %(levelname)s | %(name)s | %(message)s")

# ========= HELP =========
HELP_TEXT = (
    "Commands:\n"
    "/plan <feed>        → Make a plan and predict oil.\n"
    "/predict <feed>     → Predict oil % + confidence.\n"
    "/actual <actuals>   → Compare vs plan; add oil=45 to let AI learn.\n"
    "/reload             → Reload recommendations from Excel.\n\n"
    "Feed format:\n"
    "  Feed: Nylon=2.04T, Radial=4.80T, Chips=3.60T, Powder=0.575T, Kachra=0, Others=0\n"
    "  Optional tags: batch=92, operator=Ravi\n\n"
    "Actuals format:\n"
    "  Actual: 50-200=01:21, 200-300=01:50, 300-400=02:55, 400-450=00:22; oil=45.0; batch=92\n\n"
    "Tip: When you include oil=… in Actual, the model adapts and future predictions improve."
)

# ========= UTILS =========
def normalize_zone_key(s: str) -> str:
    """Map '50-200', '50-200 reactor', '50-200 separator' into canonical keys."""
    s = s.strip().lower()
    s = s.replace("°c", "").replace("c", "")
    s = s.replace("temp", "").replace("zone", "")
    s = s.replace("minutes", "").replace("mins", "").replace("min", "")
    s = s.replace("reactor", " reactor").replace("separator", " separator")
    s = re.sub(r"\s+", " ", s).strip()
    if "separator" in s:
        core = re.sub(r"[^0-9\-]", "", s.split("separator")[0]).strip()
        return f"{core} separator"
    if "reactor" in s:
        core = re.sub(r"[^0-9\-]", "", s.split("reactor")[0]).strip()
        return f"{core} reactor"
    # default to reactor when unspecified
    core = re.sub(r"[^0-9\-]", "", s)
    return f"{core} reactor"

def to_hhmmss(minutes: float) -> str:
    if minutes is None or (isinstance(minutes, float) and np.isnan(minutes)):
        return "-"
    total = int(round(float(minutes) * 60))
    h, m, s = total // 3600, (total % 3600) // 60, total % 60
    return f"{h:02d}:{m:02d}:{s:02d}"

def hhmmss_to_minutes(s: str) -> float:
    s = s.strip()
    if not s or s in ("0", "00:00", "00:00:00"):
        return 0.0
    parts = s.split(":")
    if len(parts) == 3:
        h, m, sec = map(float, parts)
        return h * 60 + m + sec / 60.0
    if len(parts) == 2:
        m, sec = map(float, parts)
        return m + sec / 60.0
    return float(s)

def parse_keyvals(text: str) -> Dict[str, str]:
    """Parse key=value pairs (batch=.., operator=.., oil=..)."""
    out = {}
    for kv in re.findall(r"(\b[a-zA-Z_]+)\s*=\s*([^,;]+)", text):
        out[kv[0].strip().lower()] = kv[1].strip()
    return out

# ========= ENGINE: Excel → zone map =========
class RecoEngine:
    def __init__(self, report_path: str, sheet_name: str):
        self.report_path = report_path
        self.sheet_name = sheet_name
        self.zone_map = {}
        self._load_report()

    def _load_report(self):
        self.zone_map.clear()
        try:
            df = pd.read_excel(self.report_path, sheet_name=self.sheet_name)
            # expect columns: zone_time_feature, suggested_minutes
            for _, row in df.iterrows():
                feat = str(row.get("zone_time_feature", "")).strip()
                sug  = row.get("suggested_minutes", np.nan)
                if not feat:
                    continue
                low = feat.lower()
                m = re.search(r"(\d{2,3}\s*[-–]\s*\d{2,3})", low)
                if not m:
                    continue
                window = m.group(1).replace(" ", "").replace("–", "-")
                key = f"{window} {'separator' if 'separator' in low else 'reactor'}"
                self.zone_map[key] = float(sug) if pd.notna(sug) else np.nan
            logger.info("Loaded %d zone rules from %s", len(self.zone_map), self.report_path)
        except Exception as e:
            logger.exception("Failed to load Excel; using safe defaults")
            self.zone_map.update({
                "50-200 reactor": 60.0, "200-300 reactor": 60.0,
                "300-400 reactor": 70.0, "400-450 reactor": 40.0,
                "480-500 reactor": 12.0, "500-520 reactor": 10.0,
                "300-400 separator": 30.0,
            })

    def plan_from_feed(self, feed_text: str) -> Dict[str, float]:
        _ = self._parse_feed(feed_text)  # future: make plan depend on feed
        return dict(self.zone_map)

    def compare_actuals(self, actual_text: str, plan_minutes: Dict[str, float]) -> Tuple[Dict[str, float], list]:
        actual_map = self._parse_actuals(actual_text)
        tips, deltas = [], {}
        for k, target_min in plan_minutes.items():
            if k not in actual_map or (isinstance(target_min, float) and np.isnan(target_min)):
                continue
            a = actual_map[k]
            deltas[k] = a - target_min
            if abs(deltas[k]) >= 5:
                tips.append(("reduce" if deltas[k] > 0 else "increase") + f" {k} by ~{abs(deltas[k]):.0f} min")
        if not tips:
            tips = ["Near-optimal execution vs plan. Keep the same profile."]
        return deltas, tips

    def _parse_feed(self, text: str) -> Dict[str, float]:
        # returns kg map
        text = text.replace("Feed:", "").strip()
        parts = re.split(r"[,\n]+", text)
        out = {}
        for p in parts:
            m = re.search(r"([A-Za-z\s/()]+)\s*=\s*([0-9.]+)\s*([Tt]|[Kk][Gg])?", p.strip())
            if m:
                name = m.group(1).strip().lower()
                val = float(m.group(2))
                unit = (m.group(3) or "").lower()
                if unit == "t":
                    val *= 1000.0
                out[name] = val
        return out

    def _parse_actuals(self, text: str) -> Dict[str, float]:
        text = text.replace("Actual:", "").strip()
        out = {}
        for p in re.split(r"[,\n;]+", text):
            if "=" not in p:
                continue
            left, right = p.split("=", 1)
            key = normalize_zone_key(left)
            val = right.strip()
            try:
                minutes = hhmmss_to_minutes(val) if ":" in val else float(val)
                out[key] = minutes
            except Exception:
                continue
        return out

engine = RecoEngine(REPORT_PATH, ZONE_SHEET)

# ========= LIGHT OIL% MODEL (linear + online learning) =========
FEATURE_NAMES = [
    "bias",
    "f_nylon", "f_radial", "f_chips", "f_powder", "f_others",
    "z_50_200", "z_200_300", "z_300_400", "z_400_450",
]

def _empty_state():
    return {
        "weights": {
            "bias": 44.0,
            "f_nylon":   -2.0,
            "f_radial":   2.0,
            "f_chips":    1.0,
            "f_powder":  -1.0,
            "f_others":   0.0,
            "z_50_200":   0.10,
            "z_200_300":  0.05,
            "z_300_400":  0.08,
            "z_400_450":  0.04,
        },
        "rmse": 4.0,
        "seen": 0
    }

def _load_state():
    p = Path(MODEL_STATE_PATH)
    if p.exists():
        try:
            return json.loads(p.read_text())
        except Exception:
            pass
    return _empty_state()

def _save_state(state):
    Path(MODEL_STATE_PATH).write_text(json.dumps(state))

MODEL = _load_state()

def _extract_feed_fractions(feed_map: Dict[str, float]) -> Dict[str, float]:
    keys = ["nylon","radial","chips","powder","others","kachra"]
    total = sum(feed_map.get(k, 0.0) for k in keys)
    if total <= 0:
        return dict(f_nylon=0,f_radial=0,f_chips=0,f_powder=0,f_others=0)
    return dict(
        f_nylon  = feed_map.get("nylon",0.0)/total,
        f_radial = feed_map.get("radial",0.0)/total,
        f_chips  = feed_map.get("chips",0.0)/total,
        f_powder = feed_map.get("powder",0.0)/total,
        f_others = (feed_map.get("others",0.0)+feed_map.get("kachra",0.0))/total,
    )

def _predict_oil_pct(feed_map: Dict[str,float], minutes: Dict[str,float]|None) -> Tuple[float,float]:
    w = MODEL["weights"]
    f = _extract_feed_fractions(feed_map)
    z = dict(z_50_200=0,z_200_300=0,z_300_400=0,z_400_450=0)
    if minutes:
        z["z_50_200"]  = float(minutes.get("50-200 reactor", 0))
        z["z_200_300"] = float(minutes.get("200-300 reactor", 0))
        z["z_300_400"] = float(minutes.get("300-400 reactor", 0))
        z["z_400_450"] = float(minutes.get("400-450 reactor", 0))
    y = (w["bias"] + w["f_nylon"]*f["f_nylon"] + w["f_radial"]*f["f_radial"]
         + w["f_chips"]*f["f_chips"] + w["f_powder"]*f["f_powder"] + w["f_others"]*f["f_others"]
         + w["z_50_200"]*z["z_50_200"] + w["z_200_300"]*z["z_200_300"]
         + w["z_300_400"]*z["z_300_400"] + w["z_400_450"]*z["z_400_450"])
    y = max(30.0, min(55.0, y))
    rmse = max(1e-6, float(MODEL.get("rmse", 4.0)))
    conf = max(0.1, min(0.95, 1.0/(1.0 + rmse/3.0)))
    return (round(y,2), round(conf,2))

def _sgd_update(feed_map: Dict[str,float], actual_minutes: Dict[str,float]|None, oil_pct: float):
    alpha = 0.003
    w = MODEL["weights"]
    f = _extract_feed_fractions(feed_map)
    z = dict(z_50_200=0,z_200_300=0,z_300_400=0,z_400_450=0)
    if actual_minutes:
        z["z_50_200"]  = float(actual_minutes.get("50-200 reactor", 0))
        z["z_200_300"] = float(actual_minutes.get("200-300 reactor", 0))
        z["z_300_400"] = float(actual_minutes.get("300-400 reactor", 0))
        z["z_400_450"] = float(actual_minutes.get("400-450 reactor", 0))
    y_hat, _ = _predict_oil_pct(feed_map, actual_minutes)
    err = (oil_pct - y_hat)

    w["bias"]        += alpha * err
    w["f_nylon"]     += alpha * err * f["f_nylon"]
    w["f_radial"]    += alpha * err * f["f_radial"]
    w["f_chips"]     += alpha * err * f["f_chips"]
    w["f_powder"]    += alpha * err * f["f_powder"]
    w["f_others"]    += alpha * err * f["f_others"]
    w["z_50_200"]    += alpha * err * z["z_50_200"]
    w["z_200_300"]   += alpha * err * z["z_200_300"]
    w["z_300_400"]   += alpha * err * z["z_300_400"]
    w["z_400_450"]   += alpha * err * z["z_400_450"]

    n = int(MODEL.get("seen",0)) + 1
    old_rmse = MODEL.get("rmse", 4.0)
    new_rmse = ((old_rmse**2 * (n-1) + err**2)/n) ** 0.5
    MODEL["rmse"], MODEL["seen"] = float(new_rmse), n
    _save_state(MODEL)

# ========= GROUP / CONTEXT =========
try:
    MACHINE_MAP: Dict[str, str] = json.loads(MACHINE_MAP_JSON)
except Exception:
    MACHINE_MAP = {}

def machine_label(chat_id: int) -> str:
    return MACHINE_MAP.get(str(chat_id), f"Chat {chat_id}")

def header_line(chat_id: int, extra: Dict[str,str]) -> str:
    lab = machine_label(chat_id)
    bt  = extra.get("batch")
    op  = extra.get("operator")
    stamp = datetime.now().strftime("%d-%b %H:%M")
    parts = [f"🧪 *{lab}* · {stamp}"]
    if bt: parts.append(f"Batch {bt}")
    if op: parts.append(f"Operator {op}")
    return " | ".join(parts)

async def send_to_summary(context: ContextTypes.DEFAULT_TYPE, text: str):
    if SUMMARY_CHAT_ID:
        try:
            await context.bot.send_message(chat_id=int(SUMMARY_CHAT_ID), text=text, parse_mode="Markdown", disable_web_page_preview=True)
        except Exception as e:
            logger.warning("Summary send failed: %s", e)

# ========= PLAN / PREDICT / ACTUAL HANDLERS =========
latest_plan_cache: Dict[int, Dict[str,float]] = {}
latest_feed_cache: Dict[int, Dict[str,float]] = {}

def format_plan(plan: Dict[str,float]) -> str:
    def key_order(k):
        m = re.match(r"(\d{2,3})-(\d{2,3})", k)
        a = int(m.group(1)) if m else 0
        b = int(m.group(2)) if m else 0
        return (a, b, ("separator" in k))
    lines = []
    for z in sorted(plan.keys(), key=key_order):
        lines.append(f"{z}: {to_hhmmss(plan[z])}")
    return "\n".join(lines)

async def start_cmd(update: Update, _: ContextTypes.DEFAULT_TYPE):
    await update.message.reply_text("PyroVision Assistant ready ✅\n\n"+HELP_TEXT)

async def help_cmd(update: Update, _: ContextTypes.DEFAULT_TYPE):
    await update.message.reply_text(HELP_TEXT)

async def reload_cmd(update: Update, _: ContextTypes.DEFAULT_TYPE):
    engine._load_report()
    await update.message.reply_text("Reloaded recommendations from Excel ✅")

async def plan_cmd(update: Update, context: ContextTypes.DEFAULT_TYPE):
    text = update.message.text
    feed_text = text.replace("/plan", "Feed:").strip()
    plan = engine.plan_from_feed(feed_text)
    feed_map = engine._parse_feed(feed_text)
    latest_plan_cache[update.effective_chat.id] = plan
    latest_feed_cache[update.effective_chat.id] = feed_map

    pred, conf = _predict_oil_pct(feed_map, plan)

    meta = parse_keyvals(text)
    hdr = header_line(update.effective_chat.id, meta)
    msg = (f"{hdr}\n\n*Recommended zone minutes (from data):*\n"
           f"{format_plan(plan)}\n\n"
           f"*Predicted oil:* {pred:.2f}%  _(confidence {conf:.2f})_")
    await update.message.reply_text(msg, parse_mode="Markdown", disable_web_page_preview=True)

    # also send to summary
    await send_to_summary(context, f"From {machine_label(update.effective_chat.id)}:\n\n{msg}")

async def predict_cmd(update: Update, context: ContextTypes.DEFAULT_TYPE):
    text = update.message.text
    feed_text = text.replace("/predict", "Feed:").strip()
    plan = engine.plan_from_feed(feed_text)
    feed_map = engine._parse_feed(feed_text)
    pred, conf = _predict_oil_pct(feed_map, plan)

    meta = parse_keyvals(text)
    hdr = header_line(update.effective_chat.id, meta)
    msg = (f"{hdr}\n\n*Predicted oil:* {pred:.2f}%  _(confidence {conf:.2f})_\n\n"
           f"*Plan used (from data):*\n{format_plan(plan)}")
    await update.message.reply_text(msg, parse_mode="Markdown", disable_web_page_preview=True)

async def actual_cmd(update: Update, context: ContextTypes.DEFAULT_TYPE):
    text = update.message.text
    actual_text = text.replace("/actual", "Actual:").strip()
    plan = latest_plan_cache.get(update.effective_chat.id, engine.plan_from_feed("Feed:"))
    deltas, tips = engine.compare_actuals(actual_text, plan)

    # Learn if oil= provided
    oil_pct = None
    m = re.search(r"oil\s*=\s*([0-9]+(?:\.[0-9]+)?)\s*%?", text, flags=re.I)
    if m:
        oil_pct = float(m.group(1))
        feed_map = latest_feed_cache.get(update.effective_chat.id, engine._parse_feed("Feed:"))
        _sgd_update(feed_map, engine._parse_actuals(actual_text), oil_pct)

    meta = parse_keyvals(text)
    hdr = header_line(update.effective_chat.id, meta)

    lines = [f"{hdr}", "*Deviation vs plan (min):*"]
    if not deltas:
        lines.append("Couldn’t read your actuals.\nExample:\nActual: 50-200=01:10:00, 200-300=00:45:00, 300-400=01:20:00, 400-450=01:00:00")
    else:
        for k in sorted(deltas.keys()):
            lines.append(f"{k}: {deltas[k]:+.0f}")
        lines.append("\n*Recommendations:*")
        for t in tips:
            lines.append(f"• {t}")
    if oil_pct is not None:
        lines.append(f"\nThanks! Learned from oil={oil_pct:.2f}%.")
    msg = "\n".join(lines)

    await update.message.reply_text(msg, parse_mode="Markdown", disable_web_page_preview=True)
    await send_to_summary(context, f"From {machine_label(update.effective_chat.id)}:\n\n{msg}")

async def text_router(update: Update, context: ContextTypes.DEFAULT_TYPE):
    text = (update.message.text or "").strip()
    low = text.lower()
    if low.startswith("feed:"):
        # treat as /plan
        plan = engine.plan_from_feed(text)
        feed_map = engine._parse_feed(text)
        latest_plan_cache[update.effective_chat.id] = plan
        latest_feed_cache[update.effective_chat.id] = feed_map

        pred, conf = _predict_oil_pct(feed_map, plan)
        meta = parse_keyvals(text)
        hdr = header_line(update.effective_chat.id, meta)
        msg = (f"{hdr}\n\n*Recommended zone minutes (from data):*\n"
               f"{format_plan(plan)}\n\n"
               f"*Predicted oil:* {pred:.2f}%  _(confidence {conf:.2f})_")
        await update.message.reply_text(msg, parse_mode="Markdown", disable_web_page_preview=True)
        await send_to_summary(context, f"From {machine_label(update.effective_chat.id)}:\n\n{msg}")

    elif low.startswith("actual:"):
        await actual_cmd(update, context)
    else:
        await update.message.reply_text("I didn’t understand.\n\n" + HELP_TEXT)

# ========= MAIN =========
def main():
    if not TELEGRAM_BOT_TOKEN:
        print("ERROR: Set TELEGRAM_BOT_TOKEN env var.")
        return

    app = Application.builder().token(TELEGRAM_BOT_TOKEN).build()
    app.add_handler(CommandHandler("start", start_cmd))
    app.add_handler(CommandHandler("help", help_cmd))
    app.add_handler(CommandHandler("reload", reload_cmd))
    app.add_handler(CommandHandler("plan", plan_cmd))
    app.add_handler(CommandHandler("predict", predict_cmd))
    app.add_handler(CommandHandler("actual", actual_cmd))
    app.add_handler(MessageHandler(filters.TEXT & (~filters.COMMAND), text_router))

    logger.info("Bot starting…")
    app.run_polling(allowed_updates=Update.ALL_TYPES)

if __name__ == "__main__":
    main()
