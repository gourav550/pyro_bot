# bot.py
# Telegram bot: feed -> zone plan (minutes), then actuals -> gap + recommendations
# Requires: python-telegram-bot >= 21, pandas, numpy, openpyxl

import os, re, logging
import numpy as np
import pandas as pd
from telegram import Update
from telegram.ext import Application, CommandHandler, MessageHandler, filters, ContextTypes

# ==== CONFIG ====
TELEGRAM_BOT_TOKEN = os.environ.get("TELEGRAM_BOT_TOKEN", "123456789:xxxxxxxxxxxxxx")
REPORT_PATH = os.environ.get("REPORT_PATH", "pyrolysis_feed_temp_ZONE_TIME_report.xlsx")
ZONE_SHEET = "ZoneTime_Recommendations"
PRESET_SHEET = "Presets_ZoneTimes"
RULES_SHEET = "Preset_Rules"

def normalize_zone_key(s: str) -> str:
    s = s.strip().lower()
    s = s.replace("°c","").replace("c","")
    s = s.replace("temp","").replace("zone","").replace("minutes","").replace("mins","").replace("min","")
    s = s.replace("reactor","reactor").replace("separator","separator")
    s = re.sub(r"\s+"," ",s).strip()
    if "separator" in s:
        return re.sub(r"[^0-9\-]","", s.split("separator")[0]).strip() + " separator"
    elif "reactor" in s:
        return re.sub(r"[^0-9\-]","", s.split("reactor")[0]).strip() + " reactor"
    else:
        return re.sub(r"[^0-9\-]","", s) + " reactor"

def to_hhmmss(minutes: float) -> str:
    if minutes is None or np.isnan(minutes):
        return "-"
    total = int(round(minutes*60))
    h, m, s = total//3600, (total%3600)//60, total%60
    return f"{h:02d}:{m:02d}:{s:02d}"

def hhmmss_to_minutes(s: str) -> float:
    s = s.strip()
    if not s or s in ("0","00:00","00:00:00"): return 0.0
    parts = s.split(":")
    if len(parts)==3:
        h,m,sec = map(float, parts); return h*60+m+sec/60.0
    if len(parts)==2:
        m,sec = map(float, parts);  return m+sec/60.0
    return float(s)

class RecoEngine:
    def __init__(self, report_path: str):
        self.report_path = report_path
        self.zone_map = {}
        self.presets = {}
        self.rules = []
        self._load_report()

    def _load_report(self):
        self.zone_map.clear(); self.presets.clear(); self.rules.clear()
        # Defaults
        try:
            df = pd.read_excel(self.report_path, sheet_name=ZONE_SHEET)
            for _, r in df.iterrows():
                feat = str(r.get("zone_time_feature","")).strip()
                sug  = r.get("suggested_minutes", np.nan)
                if not feat: continue
                low  = feat.lower()
                m = re.search(r"(\d{2,3}\s*[-–]\s*\d{2,3})", low)
                if not m: continue
                window = m.group(1).replace(" ","").replace("–","-")
                which  = "separator" if "separator" in low else "reactor"
                key = f"{window} {which}"
                self.zone_map[key] = float(sug) if pd.notna(sug) else np.nan
        except Exception:
            # Safe fallbacks
            self.zone_map.update({
                "50-200 reactor":165,
                "200-300 reactor":65,
                "300-400 reactor":170,
                "400-450 reactor":75,
                "450-480 reactor":20,
                "480-500 reactor":0,
                "500-520 reactor":0,
                "300-400 separator":160,
            })

        # Presets (optional)
        try:
            pf = pd.read_excel(self.report_path, sheet_name=PRESET_SHEET)
            for (profile), g in pf.groupby("profile"):
                mp = {}
                for _, r in g.iterrows():
                    feat = str(r["zone_time_feature"])
                    sug  = float(r["suggested_minutes"])
                    low  = feat.lower()
                    m = re.search(r"(\d{2,3}\s*[-–]\s*\d{2,3})", low)
                    if not m: continue
                    window = m.group(1).replace(" ","").replace("–","-")
                    which  = "separator" if "separator" in low else "reactor"
                    mp[f"{window} {which}"] = sug
                self.presets[profile] = mp
        except Exception:
            pass

        # Rules (optional)
        try:
            rf = pd.read_excel(self.report_path, sheet_name=RULES_SHEET)
            for _, r in rf.iterrows():
                self.rules.append((str(r["rule"]), str(r["profile"])))
        except Exception:
            pass

    def _parse_feed(self, text: str) -> dict:
        txt = text.replace("Feed:","").strip()
        out = {}
        for p in re.split(r"[,\n]+", txt):
            m = re.search(r"([A-Za-z\s/()]+)\s*=\s*([0-9.]+)\s*([Tt]|[Kk][Gg])?", p.strip())
            if m:
                name = m.group(1).strip().lower()
                val  = float(m.group(2))
                unit = (m.group(3) or "").lower()
                if unit=="t": val*=1000.0
                out[name] = val
        return out

    def pick_profile(self, feed_map: dict) -> str:
        total = sum(feed_map.values()) or 1.0
        pct = {k: 100.0*v/total for k,v in feed_map.items()}
        r = pct.get("radial",0); n = pct.get("nylon",0); c = pct.get("chips",0); p = pct.get("powder",0)
        # Rule table (simple)
        if r >= 40: return "Radial-heavy"
        if n >= 35: return "Nylon-heavy"
        if c >= 35: return "Chips-heavy"
        if p >= 15 and (r+n) < 60: return "Powder-heavy"
        return "Mixed"

    def plan_from_feed(self, feed_text: str) -> tuple[dict,str,dict]:
        feed = self._parse_feed(feed_text)
        profile = self.pick_profile(feed)
        # Use preset if available; else defaults
        plan = dict(self.presets.get(profile, self.zone_map))
        return plan, profile, feed

    def compare_actuals(self, actual_text: str, plan_minutes: dict) -> tuple[dict, list[str]]:
        actual = {}
        for p in re.split(r"[,\n]+", actual_text.replace("Actual:","").strip()):
            if "=" not in p: continue
            left, right = p.split("=",1)
            key = normalize_zone_key(left)
            try:
                minutes = hhmmss_to_minutes(right.strip()) if ":" in right else float(right)
                actual[key]=minutes
            except: pass

        tips, deltas = [], {}
        for k, target in plan_minutes.items():
            if k not in actual or np.isnan(target): continue
            d = actual[k] - target
            deltas[k]=d
            if abs(d)>=5:
                tips.append(("reduce" if d>0 else "increase")+f" {k} by ~{abs(d):.0f} min")
        if not tips: tips=["Near-optimal execution vs plan. Keep the same profile."]
        return deltas, tips

engine = RecoEngine(REPORT_PATH)

HELP = (
    "Send either:\n"
    "• Feed: Nylon=4.5T, Radial=5.5T, Chips=0.6T, Powder=0.5T\n"
    "  → I’ll reply with a zone plan & profile.\n\n"
    "• Actual: 50-200=01:10:00, 200-300=00:45:00, 300-400=01:20:00, 400-450=01:00:00, 450-480=00:25:00\n"
    "  → I’ll compare to plan and tell what to change next time.\n\n"
    "Commands:\n"
    "/plan <feed>   – Return a plan from feed\n"
    "/actual <vals> – Compare actuals vs plan\n"
    "/predict <feed> – Same as plan but shows profile & feed %\n"
    "/reload        – Reload the Excel recommendations\n"
)

latest_plan = {}

def format_plan(plan: dict)->str:
    def key_order(k):
        m=re.match(r"(\d{2,3})-(\d{2,3})",k); a=int(m.group(1)) if m else 0; b=int(m.group(2)) if m else 0
        return (a,b,("separator" in k))
    lines=[]
    for z in sorted(plan.keys(), key=key_order):
        lines.append(f"{z}: {to_hhmmss(plan[z])}")
    return "\n".join(lines)

async def start(update: Update, ctx: ContextTypes.DEFAULT_TYPE):
    await update.message.reply_text("Bot ready ✅\n\n"+HELP)

async def help_cmd(update: Update, ctx: ContextTypes.DEFAULT_TYPE):
    await update.message.reply_text(HELP)

async def plan_cmd(update: Update, ctx: ContextTypes.DEFAULT_TYPE):
    feed = update.message.text.replace("/plan","Feed:").strip()
    plan, profile, f = engine.plan_from_feed(feed)
    latest_plan[update.effective_chat.id] = plan
    await update.message.reply_text(
        f"Profile: {profile}\n"
        f"Plan (minutes):\n{format_plan(plan)}"
    )

async def predict_cmd(update: Update, ctx: ContextTypes.DEFAULT_TYPE):
    feed = engine._parse_feed(update.message.text.replace("/predict","").strip())
    plan, profile, f = engine.plan_from_feed("Feed:" + update.message.text.replace("/predict",""))
    total = sum(f.values()) or 1.0
    pct  = {k: 100*v/total for k,v in f.items()}
    latest_plan[update.effective_chat.id] = plan
    lines = [f"Selected profile: *{profile}*",
             "Feed %:",
             *(f"• {k.capitalize()}: {pct[k]:.1f}%" for k in ["radial","nylon","chips","powder"] if k in pct),
             "",
             "Recommended zone minutes:",
             format_plan(plan)]
    await update.message.reply_text("\n".join(lines), parse_mode=None)

async def actual_cmd(update: Update, ctx: ContextTypes.DEFAULT_TYPE):
    plan = latest_plan.get(update.effective_chat.id, engine.zone_map)
    deltas, tips = engine.compare_actuals(update.message.text, plan)
    if not deltas:
        await update.message.reply_text("Couldn’t read actuals. Example:\nActual: 50-200=01:10:00, 200-300=00:45:00, 300-400=01:20:00, 400-450=01:00:00, 450-480=00:25:00")
        return
    lines=["Deviation vs plan (min):"]
    for k in sorted(deltas.keys()):
        lines.append(f"{k}: {deltas[k]:+.0f}")
    lines.append("\nRecommendations:")
    for t in tips: lines.append("• "+t)
    await update.message.reply_text("\n".join(lines))

async def reload_cmd(update: Update, ctx: ContextTypes.DEFAULT_TYPE):
    engine._load_report()
    await update.message.reply_text("Reloaded recommendations from report ✅")

async def text_router(update: Update, ctx: ContextTypes.DEFAULT_TYPE):
    text=(update.message.text or "").strip().lower()
    if text.startswith("feed:"):
        plan, profile, f = engine.plan_from_feed(update.message.text)
        latest_plan[update.effective_chat.id]=plan
        await update.message.reply_text(f"Profile: {profile}\nPlan (minutes):\n{format_plan(plan)}")
    elif text.startswith("actual:"):
        plan=latest_plan.get(update.effective_chat.id, engine.zone_map)
        deltas,tips=engine.compare_actuals(update.message.text, plan)
        if not deltas:
            await update.message.reply_text("Couldn’t read actuals. Example:\nActual: 50-200=01:10:00, 200-300=00:45:00, 300-400=01:20:00, 400-450=01:00:00, 450-480=00:25:00")
            return
        lines=["Deviation vs plan (min):"]
        for k in sorted(deltas.keys()): lines.append(f"{k}: {deltas[k]:+.0f}")
        lines.append("\nRecommendations:")
        for t in tips: lines.append("• "+t)
        await update.message.reply_text("\n".join(lines))
    else:
        await update.message.reply_text("I didn’t understand.\n\n"+HELP)

def main():
    logging.basicConfig(level=logging.INFO)
    if TELEGRAM_BOT_TOKEN=="123456789:xxxxxxxxxxxxxx":
        print("ERROR: Set TELEGRAM_BOT_TOKEN env var or edit the file."); return

    app = Application.builder().token(TELEGRAM_BOT_TOKEN).build()
    app.add_handler(CommandHandler("start", start))
    app.add_handler(CommandHandler("help", help_cmd))
    app.add_handler(CommandHandler("plan", plan_cmd))
    app.add_handler(CommandHandler("predict", predict_cmd))
    app.add_handler(CommandHandler("actual", actual_cmd))
    app.add_handler(CommandHandler("reload", reload_cmd))
    app.add_handler(MessageHandler(filters.TEXT & (~filters.COMMAND), text_router))
    app.run_polling()

if __name__=="__main__":
    main()
