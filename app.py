# -*- coding: utf-8 -*-
# =====================================================================================
# Car Advisor – A/B Harness (Gemini "Tool" Prompt vs ChatGPT "User") + Resume Mode
# 50 שאלונים אמיתיים (סימולציה) • חיבור מלא לג'מיני ול־GPT • אפשרות להמשך מהרצה קיימת
# כולל:
# - דילוג על QIDs קיימים
# - Retry לכל API
# - סטטוס חי עם ספינר
# - Evaluator שנותן ציון, נימוק וסקירת per_car_review
# - שמירה מסודרת + צפייה בטבלאות ויצוא CSV
# =====================================================================================

import os, json, time, random, traceback
from datetime import datetime
from typing import Dict, Any, List, Optional

import streamlit as st
import pandas as pd
import google.generativeai as genai
from json_repair import repair_json

# OpenAI SDK
try:
    from openai import OpenAI
except Exception:
    OpenAI = None

# -------------------------------------------------------------------------------------
# CONFIG
# -------------------------------------------------------------------------------------
st.set_page_config(page_title="Car Advisor – Benchmark/Resume", page_icon="🚗", layout="wide")

GEMINI_MODEL = "gemini-2.5-flash"
OPENAI_MODEL = "gpt-4o-mini"

RUN_DIR = "runs"
os.makedirs(RUN_DIR, exist_ok=True)
ROWS_PATH = os.path.join(RUN_DIR, "ab_rows.json")
SUMMARY_PATH = os.path.join(RUN_DIR, "ab_summaries.json")
LAST_CSV_PATH = os.path.join(RUN_DIR, "ab_results_last.csv")

# Secrets
GEMINI_API_KEY = st.secrets.get("GEMINI_API_KEY", "")
OPENAI_API_KEY = st.secrets.get("OPENAI_API_KEY", "")

if not GEMINI_API_KEY:
    st.warning("⚠️ חסר GEMINI_API_KEY ב-.streamlit/secrets.toml")
if not OPENAI_API_KEY:
    st.warning("⚠️ חסר OPENAI_API_KEY ב-.streamlit/secrets.toml")

genai.configure(api_key=GEMINI_API_KEY)
gemini = genai.GenerativeModel(GEMINI_MODEL)
oa = OpenAI(api_key=OPENAI_API_KEY) if (OPENAI_API_KEY and OpenAI) else None

# -------------------------------------------------------------------------------------
# PROMPTS
# -------------------------------------------------------------------------------------
def build_gemini_prompt(profile: Dict[str, Any]) -> str:
    return f"""
Please recommend cars for an Israeli customer. Here is the user profile (JSON):
{json.dumps(profile, ensure_ascii=False, indent=2)}

Output requirements:
1) Return a SINGLE JSON object with fields: "search_performed", "search_queries", "recommended_cars".
2) search_performed: ALWAYS return True. You must use live web search (do not return False).
3) search_queries: ALWAYS return the actual queries you used.
4) recommended_cars: an array of 5–10 cars. EACH car MUST include:
   - brand, model, year, fuel, gear, turbo, engine_cc, price_range_nis
   - avg_fuel_consumption (+ fuel_method):
       * for non-EV: km per liter (number only)
       * for EV: kWh per 100 km (number only)
      **must return methods explanation only in hebrew** 
   - annual_fee (₪ per year, number only) + fee_method
   - reliability_score (1–10, number only) + reliability_method
   - maintenance_cost (₪/year, number only) + maintenance_method
   - safety_rating (1–10, number only) + safety_method
   - insurance_cost (₪/year, number only) + insurance_method
   - resale_value (1–10, number only) + resale_method
   - performance_score (1–10, number only) + performance_method
   - comfort_features (1–10, number only) + comfort_method
   - suitability (1–10, number only) + suitability_method
   - market_supply ("גבוה" / "בינוני" / "נמוך") + supply_method
   **must return methods explanation only in hebrew**
5) IMPORTANT: All scoring fields must be numbers only (except market_supply which is categorical).
6) IMPORTANT: Only return car models that are actually sold in Israel.
"""

USER_ONE_LINER = "תמליץ לי 5–10 רכבים על פי הצרכים שצוינו בפרופיל. תן נימוק ברור לכל דגם בעברית."
EVAL_PROMPT = """
דרג שתי תשובות (Gemini JSON ו-GPT Raw/JSON) עבור אותו פרופיל.
קריטריונים: רלוונטיות לצרכים, עומק והיגיון מקצועי, עקביות, בהירות, תועלת מעשית.
עבור על רשימת הרכבים, הוסף נימוקים קצרים (per_car_review), והחזר JSON בלבד:
{
  "gemini_score": 0-100,
  "gpt_score": 0-100,
  "winner": "Gemini"|"ChatGPT"|"Tie",
  "reason": "נימוק קצר בעברית",
  "per_car_review": [
    {"source": "Gemini", "car": "brand model year", "comment": "נימוק"},
    {"source": "ChatGPT", "car": "brand model year", "comment": "נימוק"}
  ]
}
"""

# -------------------------------------------------------------------------------------
# BUILD PROFILES (50)
# -------------------------------------------------------------------------------------
ENGINE = ["בנזין","דיזל","היברידי","פלאג-אין","חשמלי"]
GEAR = ["אוטומט","ידני","DCT/DSG","CVT"]
PRIMARY = ["אמינות","חיסכון בדלק","נוחות","ביצועים","שמירת ערך","בטיחות"]
BODY = ["האצ'בק","סדאן","קרוסאובר","סטיישן","מיני","SUV"]

def build_profile(i:int) -> Dict[str,Any]:
    random.seed(1000+i)
    return {
        "profile_id": f"Q{i:02d}",
        "budget_nis": random.choice([60000,80000,100000,120000,150000,180000,220000]),
        "family_size": random.choice([1,2,3,4,5]),
        "primary_use": random.choice(["נסיעות יומיומיות בעיר","נסיעות בין-עירוניות ארוכות","טיולים בסופי שבוע","נסיעות עבודה עם ציוד"]),
        "preferences": {
            "gearbox": random.choice(GEAR),
            "engine_type": random.choice(ENGINE),
            "body_style": random.choice(BODY),
            "priority_primary": random.choice(PRIMARY)
        }
    }

PROFILES = [build_profile(i+1) for i in range(50)]

# -------------------------------------------------------------------------------------
# UTILS
# -------------------------------------------------------------------------------------
def safe_json(txt:str)->dict:
    if not txt: return {}
    try:
        return json.loads(repair_json(txt))
    except Exception: return {}

def load_list(path):
    if os.path.exists(path):
        try: return json.load(open(path,"r",encoding="utf-8"))
        except Exception: return []
    return []

def append_item(path,item):
    data=load_list(path)
    data.append(item)
    json.dump(data,open(path,"w",encoding="utf-8"),ensure_ascii=False,indent=2)

def call_with_retry(fn,retries=3,backoff=1.5):
    last_err=None
    for i in range(retries):
        try: return fn()
        except Exception as e:
            last_err=e
            time.sleep(backoff*(i+1))
    raise last_err

# -------------------------------------------------------------------------------------
# MODEL CALLS
# -------------------------------------------------------------------------------------
def call_gemini(profile):
    def _do():
        resp = gemini.generate_content(build_gemini_prompt(profile), request_options={"timeout":180})
        return safe_json(resp.text)
    try: return call_with_retry(_do,3)
    except Exception:
        return {"_error":"Gemini failed","_trace":traceback.format_exc()}

def call_gpt(profile):
    if oa is None: return {"_raw":"No OpenAI","_json":{}}
    def _do():
        resp = oa.chat.completions.create(
            model=OPENAI_MODEL,
            messages=[{"role":"user","content":json.dumps(profile,ensure_ascii=False)},
                      {"role":"user","content":USER_ONE_LINER}],
            temperature=0.6
        )
        txt=resp.choices[0].message.content
        return {"_raw":txt,"_json":safe_json(txt)}
    try: return call_with_retry(_do,3)
    except Exception:
        return {"_raw":"GPT failed","_json":{}}

def call_eval(profile,gem,gpt):
    if oa is None:
        return {"gemini_score":0,"gpt_score":0,"winner":"Tie","reason":"Evaluator unavailable","per_car_review":[]}
    def _do():
        msgs=[
            {"role":"system","content":"Hebrew output only. Return JSON only."},
            {"role":"user","content":f"PROFILE:\n{json.dumps(profile,ensure_ascii=False,indent=2)}"},
            {"role":"user","content":f"GEMINI_JSON:\n{json.dumps(gem,ensure_ascii=False,indent=2)}"},
            {"role":"user","content":f"GPT_RAW:\n{gpt.get('_raw','')[:2000]}"},
            {"role":"user","content":EVAL_PROMPT}
        ]
        r=oa.chat.completions.create(model=OPENAI_MODEL,messages=msgs,temperature=0)
        return safe_json(r.choices[0].message.content)
    try: return call_with_retry(_do,2)
    except Exception:
        return {"gemini_score":0,"gpt_score":0,"winner":"Tie","reason":"Eval failed","per_car_review":[]}

# -------------------------------------------------------------------------------------
# MAIN UI
# -------------------------------------------------------------------------------------
st.title("🚗 Car Advisor – Benchmark + Resume")
st.caption("כולל מצב Resume, שמירה רציפה, צפייה בנתוני per_car_review, והורדות CSV")

uploaded = st.file_uploader("📤 העלה את ab_results.csv הקיים (אם יש):", type=["csv"])
run_btn = st.button("🚀 המשך או התחל הרצה")

if run_btn:
    existing_qids=set()
    if uploaded:
        df=pd.read_csv(uploaded)
        existing_qids=set(df["QID"].astype(str))
        st.info(f"נמצאו {len(existing_qids)} שאלונים קיימים. נמשיך רק את החסרים.")
    else:
        st.warning("לא הועלה קובץ, מתחילים מההתחלה.")
    profiles_left=[p for p in PROFILES if p["profile_id"] not in existing_qids]
    progress=st.progress(0)
    status=st.empty()
    run_rows=[]
    total=len(profiles_left)
    for idx,prof in enumerate(profiles_left,1):
        qid=prof["profile_id"]
        status.info(f"🔄 {qid}: Gemini עובד...")
        gem=call_gemini(prof)
        status.info(f"🤖 {qid}: GPT עובד...")
        gpt=call_gpt(prof)
        status.info(f"⚖️ {qid}: Evaluator...")
        ev=call_eval(prof,gem,gpt)
        entry={"ts":datetime.now().isoformat(),"profile":prof,"gemini":gem,"gpt":gpt,"eval":ev}
        append_item(ROWS_PATH,entry)
        run_rows.append(entry)
        progress.progress(idx/total)
    status.success("✅ הסתיים.")

    # --- TABLE ---
    table=pd.DataFrame([{
        "time":r["ts"][:19].replace("T"," "),
        "QID":r["profile"]["profile_id"],
        "תקציב":r["profile"]["budget_nis"],
        "שימוש":r["profile"]["primary_use"],
        "דלק":r["profile"]["preferences"]["engine_type"],
        "תיבה":r["profile"]["preferences"]["gearbox"],
        "Gemini score":r["eval"].get("gemini_score"),
        "GPT score":r["eval"].get("gpt_score"),
        "Winner":r["eval"].get("winner"),
        "Reason":r["eval"].get("reason","")
    } for r in run_rows])
    st.dataframe(table,use_container_width=True)

    # per_car_review viewer
    st.subheader("🔎 פירוט per_car_review (אם קיים)")
    rows=[]
    for r in run_rows:
        for pc in r["eval"].get("per_car_review",[]) or []:
            rows.append({
                "QID":r["profile"]["profile_id"],
                "Source":pc.get("source",""),
                "Car":pc.get("car",""),
                "Comment":pc.get("comment","")
            })
    if rows:
        st.dataframe(pd.DataFrame(rows),use_container_width=True)
    else:
        st.caption("אין נתוני per_car_review להצגה.")

    # CSV Export
    path=os.path.join(RUN_DIR,"ab_results_updated.csv")
    table.to_csv(path,index=False,encoding="utf-8-sig")
    with open(path,"rb") as f:
        st.download_button("⬇️ הורד CSV מעודכן",f,file_name="ab_results_updated.csv",mime="text/csv")

st.caption("© 2025 Car Advisor – כולל Resume, per_car_review viewer, Retry וסטטוס חי.")
