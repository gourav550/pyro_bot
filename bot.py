# =========================================================
# ✅ PyroVision Assistant – Final Build (Nov 2025)
# ---------------------------------------------------------
# - Learns yield weights (chips = 0.46, powder = 0.53)
# - Confidence up to 0.95 based on MAE + mix similarity
# - Fixed zone matching (“no matching zones” bug)
# - Mixed message format (predicted + oil reco + deviation chart)
# =========================================================

import os, io, re, json, math, logging
from datetime import datetime, timedelta
from zoneinfo import ZoneInfo
import numpy as np, pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from telegram import Update, InputFile
from telegram.constants import ParseMode
from telegram.ext import Application, CommandHandler, MessageHandler, ContextTypes, filters

# ---------- CONFIG ----------
LOG = logging.getLogger("pyro_bot")
logging.basicConfig(level=logging.INFO,
                    format="%(asctime)s | %(levelname)s | %(name)s | %(message)s")

TOKEN = os.getenv("TELEGRAM_BOT_TOKEN", "")
SUMMARY_CHAT_ID = int(os.getenv("SUMMARY_CHAT_ID", "0") or "0")
REPORT_PATH = os.getenv("REPORT_PATH", "pyrolysis_feed_temp_ZONE_TIME_report.xlsx")
BOT_TZ = ZoneInfo("Asia/Kolkata")

try:
    MACHINE_MAP = json.loads(os.getenv("MACHINE_MAP", "{}"))
except Exception:
    MACHINE_MAP = {}

STATE_PATH = "bot_state.json"

# ---------- STATE ----------
state = {
    "weights": {
        "radial": 0.44, "nylon": 0.42, "chips": 0.46, "powder": 0.53, "kachra": 0.40, "others": 0.40
    },
    "latest_feed": {}, "last_actual_ts": {}, "reminders": {},
    "mix_mean": None, "errors": []
}

def save_state():
    try:
        with open(STATE_PATH, "w", encoding="utf-8") as f:
            json.dump(state, f, ensure_ascii=False, indent=2)
    except Exception as e:
        LOG.warning(f"⚠️ Could not save state: {e}")

def load_state():
    if os.path.exists(STATE_PATH):
        try:
            with open(STATE_PATH, "r", encoding="utf-8") as f:
                state.update(json.load(f))
                LOG.info("✅ State loaded.")
        except Exception as e:
            LOG.warning(f"⚠️ Could not load state: {e}")
    state.setdefault("mix_mean", None)
    state.setdefault("errors", [])

# ---------- UTILS ----------
def _clean_zone_key(s: str) -> str:
    s = (s or "").strip().lower()
    s = re.sub(r"[–—−]", "-", s)
    s = re.sub(r"\s+", "", s)
    return s

def to_hhmmss(minutes):
    if minutes is None or math.isnan(float(minutes)): return "00:00:00"
    total = int(round(float(minutes)*60))
    h,m = divmod(total,60)
    return f"{h//60:02d}:{h%60:02d}:{m:02d}"

def hhmmss_to_minutes(s: str) -> float:
    s = (s or "").strip()
    if ":" not in s: return float(s or 0.0)
    parts=[int(x) for x in s.split(":")]
    if len(parts)==3: h,m,sec=parts; return h*60+m+sec/60
    if len(parts)==2: m,sec=parts; return m+sec/60
    return 0.0

def machine_label(cid:int)->str:
    return MACHINE_MAP.get(str(cid)) or MACHINE_MAP.get(str(int(cid))) or str(cid)

# ---------- EXCEL ----------
def load_zone_rules(path=REPORT_PATH, sheet="ZoneTime_Recommendations"):
    rules={}
    try:
        df=pd.read_excel(path,sheet_name=sheet)
        for _,r in df.iterrows():
            feat=str(r.get("zone_time_feature","")).strip().lower()
            mins=r.get("suggested_minutes",np.nan)
            if not feat or pd.isna(mins): continue
            m=re.search(r"(\d{2,3})\s*[-–]\s*(\d{2,3})",feat)
            if not m: continue
            key=f"{m.group(1)}-{m.group(2)} {'separator' if 'separator' in feat else 'reactor'}"
            rules[key]=float(mins)
        LOG.info(f"Loaded {len(rules)} zone rules from Excel.")
    except Exception as e:
        LOG.warning(f"Excel load failed: {e}")
        rules={
            "50-200 reactor":165,"200-300 reactor":70,"300-400 reactor":185,"300-400 separator":175,
            "400-450 reactor":75,"450-480 reactor":20,"480-500 reactor":0,"500-520 reactor":0
        }
    return rules

ZONE_RULES = load_zone_rules()

def maybe_load_yield_weights():
    try:
        df=pd.read_excel(REPORT_PATH,sheet_name="YieldWeights")
        for _,r in df.iterrows():
            k,v=r["component"].strip().lower(),float(r["weight"])
            if k in state["weights"]: state["weights"][k]=float(np.clip(v,0.3,0.65))
        LOG.info("YieldWeights sheet applied.")
    except Exception: pass

# ---------- PARSERS ----------
def parse_feed(txt:str)->dict:
    t=re.sub(r"^/?(plan|predict)?\s*feed\s*:?","",txt,flags=re.I).strip()
    data={}
    for part in re.split(r"[,;\n]+",t):
        if "=" not in part: continue
        k,v=[x.strip() for x in part.split("=",1)]
        lk=k.lower()
        if lk in ("batch","operator","date"): data[lk]=v; continue
        val=float(re.sub(r"[^\d.]", "", v))
        if v.lower().endswith(("t","ton","mt")): val*=1000
        data[lk]=val
    return data

def parse_actual(txt:str)->dict:
    t=re.sub(r"^/?actual\s*:?", "", txt, flags=re.I).strip()
    out={}
    for ch in re.split(r"[;,]+",t):
        if "=" not in ch: continue
        k,v=[x.strip() for x in ch.split("=",1)]
        lk=k.lower()
        if lk in ("oil","oil%","oilpct"): out["oil"]=float(re.sub(r"[^\d.]", "", v) or 0.0); continue
        if lk=="batch": out["batch"]=v; continue
        z=_clean_zone_key(lk)
        if re.match(r"^\d{2,3}-\d{2,3}$",z): out[z]=hhmmss_to_minutes(v)
    return out

# ---------- ENGINE ----------
class RecoEngine:
    def __init__(self,base): self.base=base
    def plan(self,feed):
        plan=dict(self.base)
        tot=sum(feed.get(k,0.0) for k in ("radial","nylon","chips","powder","kachra","others"))
        if tot>0:
            rr,cr,nr=feed.get("radial",0)/tot,feed.get("chips",0)/tot,feed.get("nylon",0)/tot
            for k in plan:
                if k.startswith("300-400"): plan[k]=max(60,plan[k]*(1+0.2*rr+0.1*cr-0.08*nr))
                if k.startswith("200-300"): plan[k]=max(45,plan[k]*(1+0.08*rr-0.05*nr))
        return plan

ENGINE=RecoEngine(ZONE_RULES)

# ---------- AI yield ----------
def _normalize_mix(feed):
    keys=("radial","nylon","chips","powder","kachra","others")
    tot=sum(feed.get(k,0.0) for k in keys)
    return {k:(feed.get(k,0)/tot if tot>0 else 0) for k in keys}

def _update_mix_mean(mix):
    α=0.1
    if not state.get("mix_mean"): state["mix_mean"]=dict(mix); return
    for k,v in mix.items(): state["mix_mean"][k]=(1-α)*state["mix_mean"].get(k,0)+α*v

def _recent_mae(): e=state.get("errors",[]); return float(np.mean(e) if e else 3.0)

def predict_yield(feed):
    w,state_w=state["weights"],state["weights"]
    tot=sum(feed.get(k,0) for k in w)
    if tot<=0: return (0.0,0.6)
    pred=sum((feed.get(k,0)/tot)*state_w[k]*100 for k in w)
    pred=float(np.clip(pred,30,60))
    mae=_recent_mae(); mae_pen=np.clip(mae/6,0,1)*0.28
    mix=_normalize_mix(feed); mm=state.get("mix_mean") or mix
    l1=sum(abs(mix[k]-mm.get(k,0)) for k in mix); sim_pen=np.clip(l1/0.8,0,1)*0.22
    conf=float(np.clip(0.95-mae_pen-sim_pen,0.6,0.95))
    return round(pred,2),round(conf,2)

def learn_from_actual(feed,oil):
    tot=sum(feed.get(k,0) for k in state["weights"])
    if tot>0:
        pred,_=predict_yield(feed); err=(oil-pred)/100
        for k in state["weights"]:
            share=feed.get(k,0)/tot; state["weights"][k]+=0.01*err*share
            state["weights"][k]=float(np.clip(state["weights"][k],0.3,0.65))
    _update_mix_mean(_normalize_mix(feed))
    state["errors"].append(abs(oil-predict_yield(feed)[0])); state["errors"]=state["errors"][-30:]
    save_state()

# ---------- Display ----------
def bar_plan_vs_actual(plan,actual):
    zones=sorted(plan.keys(),key=lambda s:int(s.split("-")[0]))
    p=[plan[z] for z in zones]; a=[]
    amap={_clean_zone_key(k):v for k,v in actual.items() if "-" in k}
    for z in zones: a.append(amap.get(_clean_zone_key(z.split()[0]),np.nan))
    x=np.arange(len(zones)); fig,ax=plt.subplots(figsize=(9,3.5),dpi=150)
    ax.bar(x-0.2,p,0.4,label="Plan"); ax.bar(x+0.2,a,0.4,label="Actual")
    ax.set_xticks(x,zones,rotation=25); ax.legend(); ax.grid(axis="y",alpha=0.3)
    buf=io.BytesIO(); plt.tight_layout(); plt.savefig(buf,format="png"); plt.close(fig); buf.seek(0)
    return buf

def oil_yield_reco(pred,actual):
    Δ=actual-pred; tips=[]
    if Δ<-1.5:
        tips+=["Increase *300–400 separator* by 15–25 min (condensation)",
               "Extend *300–400 reactor* by 20–30 min (mix completion)",
               "Slightly extend *200–300 reactor* by 10–15 min (stabilize ramp)",
               "Check line losses & ΔT; avoid spikes."]
    elif Δ<-0.5:
        tips+=["Slightly extend *300–400 separator* 10–15 min; watch over-crack",
               "Add 10–15 min in *300–400 reactor* if drip reduces early."]
    else:
        tips.append("✅ Yield on target – maintain smooth 300–400 ramp.")
    return tips

# ---------- COMMANDS ----------
HELP=(
"• *Feed:* Radial=5.1T, Nylon=0.6T, Chips=3.4T, Powder=1.5T, batch=92, operator=Ravi, date=09-11-2025\n"
"• *Actual:* 50-200=01:14, 200-300=01:06, 300-400=02:07, 400-450=01:10, 450-480=00:32, oil=40.7; batch=92\n"
"→ AI recommends and learns\n"
"/status  /reload  /id"
)

async def cmd_start(u,c): await u.message.reply_text("✅ PyroVision Assistant Ready\n\n"+HELP,parse_mode="Markdown")
async def cmd_help(u,c): await u.message.reply_text(HELP,parse_mode="Markdown")

async def cmd_reload(u,c):
    global ZONE_RULES,ENGINE
    ZONE_RULES=load_zone_rules(); ENGINE=RecoEngine(ZONE_RULES)
    maybe_load_yield_weights(); save_state()
    await u.message.reply_text("🔁 Reloaded Excel rules & weights.")

async def cmd_id(u,c):
    cid=u.effective_chat.id
    await u.message.reply_text(f"`{cid}` → *{machine_label(cid)}*",parse_mode="Markdown")

async def cmd_status(u,c):
    now=datetime.now(BOT_TZ); lines=["🟢 *Machine Status*"]
    for cid_str,label in MACHINE_MAP.items():
        lf=state["latest_feed"].get(cid_str); status="Idle"
        if lf:
            lfts=datetime.fromisoformat(lf["ts"])
            hrs=(now-lfts).total_seconds()/3600
            lat=state["last_actual_ts"].get(cid_str)
            completed=False
            if lat: completed=datetime.fromisoformat(lat)>=lfts
            if hrs<=12 and not completed: status=f"Running (batch {lf.get('batch','?')})"
            elif completed: status=f"Completed (batch {lf.get('batch','?')})"
        lines.append(f"• {label}: {status}")
    await u.message.reply_text("\n".join(lines),parse_mode="Markdown")

# ---------- MAIN HANDLER ----------
async def handle_msg(u,c):
    txt=u.message.text.strip()
    cid=str(u.effective_chat.id)
    if txt.lower().startswith("feed"):  # plan
        feed=parse_feed(txt)
        plan=ENGINE.plan(feed); pred,conf=predict_yield(feed)
        state["latest_feed"][cid]={"feed":feed,"plan":plan,"ts":datetime.now(BOT_TZ).isoformat(),
                                   "batch":feed.get("batch"),"operator":feed.get("operator")}
        save_state()
        lines=[f"🧠 *{machine_label(cid)}* — Batch {feed.get('batch','?')}, Operator {feed.get('operator','?')}",
               f"📅 Date {datetime.now(BOT_TZ):%d-%b-%Y (%A)}",
               f"🛢 Predicted Oil: {pred:.2f}% (conf {conf:.2f})",
               "\nRecommended zone minutes:"]
        for z,v in plan.items(): lines.append(f"{z}: {to_hhmmss(v)}")
        await u.message.reply_text("\n".join(lines),parse_mode="Markdown")
        return

    if txt.lower().startswith("actual"):
        actual=parse_actual(txt)
        lf=state["latest_feed"].get(cid)
        if not lf: await u.message.reply_text("⚠️ No feed found – send Feed first."); return
        plan=lf["plan"]; feed=lf["feed"]
        # normalize keys
        actual_z={_clean_zone_key(k):v for k,v in actual.items() if "-" in k}
        deltas={z:actual_z.get(_clean_zone_key(z.split()[0]),np.nan)-v for z,v in plan.items() if _clean_zone_key(z.split()[0]) in actual_z}
        if not deltas: await u.message.reply_text("📉 Deviation vs plan (min): (no matching zones found)",parse_mode="Markdown")
        else:
            lines=["📉 *Deviation vs plan (min):*"]
            for z,v in deltas.items(): lines.append(f"{z}: {int(v)}")
            await u.message.reply_text("\n".join(lines),parse_mode="Markdown")

        pred,_=predict_yield(feed); actual_oil=actual.get("oil",0)
        learn_from_actual(feed,actual_oil)
        tips=oil_yield_reco(pred,actual_oil)
        reco=[f"🛢 Predicted vs Actual Oil: {pred:.2f}% → {actual_oil:.2f}% (conf {_:.2f})",
              "🧩 *Recommendations to lift oil yield:*"]+[f"• {t}" for t in tips]
        await u.message.reply_text("\n".join(reco),parse_mode="Markdown")

        buf=bar_plan_vs_actual(plan,actual)
        await u.message.reply_photo(InputFile(buf,"chart.png"))

# ---------- APP ----------
def main():
    load_state()
    app = Application.builder().token(TOKEN).build()
    app.add_handler(CommandHandler("start", cmd_start))
    app.add_handler(CommandHandler("help", cmd_help))
    app.add_handler(CommandHandler("reload", cmd_reload))
    app.add_handler(CommandHandler("id", cmd_id))
    app.add_handler(CommandHandler("status", cmd_status))
    app.add_handler(MessageHandler(filters.TEXT & ~filters.COMMAND, handle_msg))

    LOG.info("🚀 PyroVision Assistant running...")
    app.run_polling()

if __name__ == "__main__":
    main()
