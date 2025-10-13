# -*- coding: utf-8 -*-
# =====================================================================================
# Car Advisor – A/B Harness (Gemini "Tool" Prompt vs ChatGPT "User")
# 50 שאלונים אמיתיים (סימולציה) • Gemini עם פרומפט מקצועי מחייב "חיפוש" • GPT רק עם הפרופיל
# כולל:
# - סטטוס חי (ספינר) לאיזה שאלון ואיזה כלי רץ עכשיו
# - Retry לכל API
# - Evaluator (GPT) שנותן ציון לכל מודל בכל שאלון + נימוק + מעבר על כל הרכבים
# - טבלת פלט אחודה מוכנה לניתוח + יצוא CSV
# - שמירה מסודרת לכל הרשומות (JSON) ולסיכומים
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
st.set_page_config(page_title="Car Advisor – A/B Impact Test", page_icon="🚗", layout="wide")

GEMINI_MODEL = "gemini-2.5-flash"
OPENAI_MODEL = "gpt-4o-mini"  # ניתן להחליף לפי הצורך

RUN_DIR = "runs"
os.makedirs(RUN_DIR, exist_ok=True)
ROWS_PATH = os.path.join(RUN_DIR, "ab_rows.json")        # כל הרשומות (פרופיל+פלטים+הערכה)
SUMMARY_PATH = os.path.join(RUN_DIR, "ab_summaries.json") # סיכומים
LAST_CSV_PATH = os.path.join(RUN_DIR, "ab_results_last.csv")

# Secrets
GEMINI_API_KEY = st.secrets.get("GEMINI_API_KEY", "")
OPENAI_API_KEY = st.secrets.get("OPENAI_API_KEY", "")

if not GEMINI_API_KEY:
    st.warning("⚠️ חסר GEMINI_API_KEY ב-.streamlit/secrets.toml")
if not OPENAI_API_KEY:
    st.warning("⚠️ חסר OPENAI_API_KEY ב-.streamlit/secrets.toml")

# Init clients
genai.configure(api_key=GEMINI_API_KEY)
gemini = genai.GenerativeModel(GEMINI_MODEL)
oa = OpenAI(api_key=OPENAI_API_KEY) if (OPENAI_API_KEY and OpenAI) else None

# -------------------------------------------------------------------------------------
# PROMPTS
# -------------------------------------------------------------------------------------
# פרומפט “כלי מקצועי” לג’מיני – EXACTLY לפי הדרישה שלך (כולל הדרישה ל-search_performed=True)
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

# פרומפט “משתמש רגיל” ל-GPT – רק הפרופיל + משפט בקשה קצר (לא תנאי מעבדה)
USER_ONE_LINER = "תמליץ לי 5–10 רכבים על פי הצרכים שצוינו בפרופיל. תן נימוק ברור לכל דגם בעברית."

# Evaluator: נותן ציון לכל מודל בכל שאלון, ומבקש לעבור גם על הרכבים
EVAL_PROMPT = """
דרג שתי תשובות (Gemini JSON ו-GPT Raw/JSON) עבור אותו פרופיל.
קריטריונים: רלוונטיות לצרכים, עומק והיגיון מקצועי, עקביות, בהירות, תועלת מעשית.
עבור כל אחת מהתשובות, עבור על רשימת הרכבים והערך כל דגם בקצרה (נימוק תמציתי).

החזר JSON בלבד:
{
  "gemini_score": 0,          // 0-100
  "gpt_score": 0,             // 0-100
  "winner": "Gemini" | "ChatGPT" | "Tie",
  "reason": "נימוק קצר בעברית",
  "per_car_review": [         // סקירה לכלי מנצח, דגם-דגם (אם אין, ריק)
    {
      "source": "Gemini" | "ChatGPT",
      "car": "brand model year",
      "comment": "נימוק קצר בעברית"
    }
  ]
}
"""

# -------------------------------------------------------------------------------------
# PROFILES (50)
# -------------------------------------------------------------------------------------
BRANDS = ["Toyota","Mazda","Hyundai","Kia","Honda","Subaru","Skoda","Seat","Volkswagen","Renault","Peugeot","Fiat","Alfa Romeo","BMW","Mercedes","Audi"]
BODY = ["האצ'בק","סדאן","קרוסאובר","סטיישן","מיני","SUV"]
ENGINE = ["בנזין","דיזל","היברידי","פלאג-אין","חשמלי"]
GEAR = ["אוטומט","ידני","DCT/DSG","CVT"]
PRIMARY = ["אמינות","חיסכון בדלק","נוחות","ביצועים","שמירת ערך","בטיחות"]

def build_profile(i:int) -> Dict[str,Any]:
    random.seed(1000+i)
    return {
        "profile_id": f"Q{i:02d}",
        "budget_nis": random.choice([60000, 80000, 100000, 120000, 150000, 180000, 220000]),
        "driver_age": random.choice([22,26,30,35,40,45,50]),
        "annual_km": random.choice([8000, 12000, 15000, 20000, 30000]),
        "family_size": random.choice([1,2,3,4,5]),
        "primary_use": random.choice([
            "נסיעות יומיומיות בעיר", "נסיעות בין-עירוניות ארוכות",
            "טיולים בסופי שבוע", "נסיעות עבודה עם ציוד", "נסיעות לעבודה במרכז"
        ]),
        "preferences": {
            "gearbox": random.choice(GEAR),
            "engine_type": random.choice(ENGINE),
            "body_style": random.choice(BODY),
            "priority_primary": random.choice(PRIMARY),
            "priority_secondary": random.sample(PRIMARY, k=2),
            "performance_importance": random.randint(1,5),
            "comfort_importance": random.randint(1,5),
            "reliability_importance": random.randint(1,5),
            "safety_importance": random.randint(1,5)
        },
        "must_haves": random.sample(["חיישני חניה","בלימה אוטונומית","בקרת שיוט אדפטיבית","מערכת בטיחות מתקדמת","תא מטען גדול","מסך גדול"], k=3),
        "nice_to_have": random.sample(["Sunroof","מערכת שמע פרימיום","טעינה אלחוטית","מושבים חשמליים","ג'אנטים"], k=2),
        "parking": random.choice(["עיר צפופה","פרבר","כפרי"]),
        "region": random.choice(["מרכז","שפלה","צפון","דרום","ירושלים","חוף"]),
        "risk_tolerance": random.choice(["נמוכה","בינונית","גבוהה"]),
        "notes": random.choice([
            "רוצה רכב חסכוני ואמין", "קודם היה רכב יקר בתחזוקה", "חניה צפופה ליד הבית",
            "צריך רכב למשפחה עם עגלה", "חשוב שמירת ערך"
        ])
    }

PROFILES: List[Dict[str,Any]] = [build_profile(i+1) for i in range(50)]

# -------------------------------------------------------------------------------------
# UTILS
# -------------------------------------------------------------------------------------
def safe_json(text: Optional[str]) -> Dict[str,Any]:
    if not text:
        return {}
    try:
        fixed = repair_json(text)
        return json.loads(fixed)
    except Exception:
        return {}

def load_list(path:str) -> List[Dict[str,Any]]:
    if os.path.exists(path):
        try:
            with open(path,"r",encoding="utf-8") as f:
                return json.load(f)
        except Exception:
            return []
    return []

def append_item(path:str, item:Dict[str,Any]):
    data = load_list(path)
    data.append(item)
    with open(path,"w",encoding="utf-8") as f:
        json.dump(data,f,ensure_ascii=False,indent=2)

def call_with_retry(fn, retries=3, backoff=1.5):
    last_err = None
    for i in range(retries):
        try:
            return fn()
        except Exception as e:
            last_err = e
            time.sleep(backoff*(i+1))
    raise last_err

# -------------------------------------------------------------------------------------
# MODEL CALLS
# -------------------------------------------------------------------------------------
def call_gemini(profile:Dict[str,Any], timeout=180) -> Dict[str,Any]:
    prompt = build_gemini_prompt(profile)
    def _do():
        resp = gemini.generate_content(prompt, request_options={"timeout": timeout})
        return safe_json(resp.text)
    try:
        return call_with_retry(_do, retries=3)
    except Exception:
        return {"_error": "Gemini call failed", "_trace": traceback.format_exc()}

def call_gpt_user(profile:Dict[str,Any], timeout=120) -> Dict[str,Any]:
    if oa is None:
        return {"_raw": "OpenAI client unavailable", "_json": {}}
    def _do():
        resp = oa.chat.completions.create(
            model=OPENAI_MODEL,
            messages=[
                {"role":"user","content": json.dumps(profile, ensure_ascii=False)},
                {"role":"user","content": USER_ONE_LINER}
            ],
            temperature=0.6,
            timeout=timeout
        )
        text = resp.choices[0].message.content
        parsed = safe_json(text)
        return {"_raw": text, "_json": parsed}
    try:
        return call_with_retry(_do, retries=3)
    except Exception:
        return {"_raw": "GPT call failed", "_json": {}, "_trace": traceback.format_exc()}

def call_evaluator(profile:Dict[str,Any], gem_json:Dict[str,Any], gpt_pack:Dict[str,Any]) -> Dict[str,Any]:
    if oa is None:
        return {"gemini_score":0,"gpt_score":0,"winner":"Tie","reason":"Evaluator unavailable","per_car_review":[]}
    def _do():
        msgs = [
            {"role":"system","content":"Hebrew output only. Return JSON only."},
            {"role":"user","content": f"PROFILE:\n{json.dumps(profile, ensure_ascii=False, indent=2)}"},
            {"role":"user","content": f"GEMINI_JSON:\n{json.dumps(gem_json, ensure_ascii=False, indent=2)}"},
            {"role":"user","content": f"GPT_RAW:\n{gpt_pack.get('_raw','')[:4000]}"},
            {"role":"user","content": f"GPT_JSON:\n{json.dumps(gpt_pack.get('_json',{}), ensure_ascii=False, indent=2)}"},
            {"role":"user","content": EVAL_PROMPT}
        ]
        resp = oa.chat.completions.create(
            model=OPENAI_MODEL,
            messages=msgs,
            temperature=0.0
        )
        return safe_json(resp.choices[0].message.content)
    try:
        return call_with_retry(_do, retries=2)
    except Exception:
        return {"gemini_score":0,"gpt_score":0,"winner":"Tie","reason":"Evaluation failed","per_car_review":[]}

# -------------------------------------------------------------------------------------
# TABLE BUILDERS
# -------------------------------------------------------------------------------------
def stringify_recs_from_gem(g:Dict[str,Any]) -> str:
    out=[]
    for it in (g.get("recommended_cars") or []):
        car = f"{it.get('brand','')} {it.get('model','')} {it.get('year','')}".strip()
        why = ""
        # נעדיף להצמיד את הסיבות מתוך השדות השונים הקיימים
        if it.get("performance_score") is not None or it.get("suitability") is not None:
            why = f"(perf {it.get('performance_score','?')}, suit {it.get('suitability','?')})"
        out.append(f"{car} {why}".strip())
    return " | ".join(out)

def stringify_recs_from_gptpack(p:Dict[str,Any]) -> str:
    j = p.get("_json") or {}
    # ננסה לחלץ שמות מתוך JSON אם קיים; אם לא – נציג raw עד 300 תווים
    cars=[]
    if isinstance(j, dict):
        for k,v in j.items():
            if isinstance(v, list):
                for it in v:
                    if isinstance(it, dict):
                        car = " ".join([str(it.get(x,"")) for x in ("brand","model","year")]).strip()
                        if car:
                            cars.append(car)
    if cars:
        return " | ".join(cars)
    raw = (p.get("_raw") or "").replace("\n"," ")[:300]
    return raw + ("..." if len(p.get("_raw",""))>300 else "")

def flatten_row(entry:Dict[str,Any]) -> Dict[str,Any]:
    prof = entry["profile"]
    gem  = entry["gemini"]
    gpt  = entry["gpt"]
    ev   = entry["eval"]

    return {
        "time": entry["ts"][:19].replace("T"," "),
        "QID": prof.get("profile_id",""),
        "תקציב": prof.get("budget_nis",""),
        "שימוש": prof.get("primary_use",""),
        "עדיפות": prof.get("preferences",{}).get("priority_primary",""),
        "דלק": prof.get("preferences",{}).get("engine_type",""),
        "תיבה": prof.get("preferences",{}).get("gearbox",""),
        "משפחה": prof.get("family_size",""),
        # Gemini
        "Gemini: search_performed": gem.get("search_performed",""),
        "Gemini: queries": " | ".join(gem.get("search_queries",[]) or []),
        "Gemini: recs": stringify_recs_from_gem(gem),
        # GPT
        "GPT: recs/RAW": stringify_recs_from_gptpack(gpt),
        # Evaluator
        "Eval: Gemini score": ev.get("gemini_score",""),
        "Eval: GPT score": ev.get("gpt_score",""),
        "Eval: Winner": ev.get("winner",""),
        "Eval: Reason": ev.get("reason",""),
    }

# -------------------------------------------------------------------------------------
# UI
# -------------------------------------------------------------------------------------
st.title("🚗 Car Advisor – מבחן אימפקט: כלי מקצועי מול צ'אט")
st.caption("50 שאלונים אמיתיים (סימולציה) • Gemini (פרומפט מקצועי) מול GPT (משתמש רגיל) • טבלת תוצאות וציון לכל שאלון")

with st.sidebar:
    st.markdown("### הגדרות הרצה")
    n = st.slider("מספר שאלונים", 10, 50, 50, step=5)
    seed = st.number_input("Seed לסדר הרצה", min_value=0, max_value=999999, value=42, step=1)
    run_btn = st.button("🚀 התחל הרצה רציפה")

prev = load_list(ROWS_PATH)
st.write(f"📁 רשומות שמורות: {len(prev)}")

status_box = st.empty()
progress = st.progress(0)

run_rows: List[Dict[str,Any]] = []

if run_btn:
    random.seed(seed)
    order = PROFILES[:]
    random.shuffle(order)
    order = order[:n]

    for i, prof in enumerate(order, start=1):
        # GEMINI
        status_box.info(f"⏳ סטטוס: שאלון {i}/{n} • כלי: Gemini • QID={prof['profile_id']}")
        with st.spinner(f"Gemini עובד על שאלון {i}/{n}..."):
            gem_json = call_gemini(prof)

        # GPT (USER)
        status_box.info(f"⏳ סטטוס: שאלון {i}/{n} • כלי: ChatGPT • QID={prof['profile_id']}")
        with st.spinner(f"ChatGPT עובד על שאלון {i}/{n}..."):
            gpt_pack = call_gpt_user(prof)

        # EVALUATOR
        status_box.info(f"⏳ סטטוס: הערכה • QID={prof['profile_id']}")
        with st.spinner(f"Evaluator מדרג את התוצאות לשאלון {i}/{n}..."):
            eval_obj = call_evaluator(prof, gem_json, gpt_pack)

        entry = {
            "ts": datetime.now().isoformat(),
            "profile": prof,
            "gemini": gem_json,
            "gpt": gpt_pack,
            "eval": eval_obj
        }
        append_item(ROWS_PATH, entry)
        run_rows.append(entry)

        progress.progress(i / n)

    # SUMMARY
    winners = {"Gemini":0,"ChatGPT":0,"Tie":0}
    for r in run_rows:
        w = r["eval"].get("winner","Tie")
        winners[w] = winners.get(w,0)+1
    summary = {
        "ts": datetime.now().isoformat(),
        "n": n,
        "winners": winners,
        "avg_scores": {
            "gemini": round(sum(r["eval"].get("gemini_score",0) for r in run_rows)/len(run_rows),2),
            "gpt": round(sum(r["eval"].get("gpt_score",0) for r in run_rows)/len(run_rows),2),
        }
    }
    append_item(SUMMARY_PATH, summary)
    status_box.success("✅ הרצה הושלמה. ראה טבלה למטה והורדות.")

# -------------------------------------------------------------------------------------
# TABLE RENDER + EXPORT
# -------------------------------------------------------------------------------------
st.divider()
st.subheader("📊 טבלת תוצאות – שאלון, המלצות, ציונים ומנצח")

display_rows = run_rows if run_rows else prev
if display_rows:
    table = pd.DataFrame([flatten_row(r) for r in display_rows])
    st.dataframe(table, use_container_width=True)

    # CSV Export
    table.to_csv(LAST_CSV_PATH, index=False, encoding="utf-8-sig")
    with open(LAST_CSV_PATH,"rb") as f:
        st.download_button("⬇️ הורד CSV של הטבלה", f, file_name="ab_results.csv", mime="text/csv")

    # Optional: הצג גם per_car_review אם קיים
    st.subheader("🔎 פירוט per_car_review (אם קיים מה-Evaluator)")
    reviews_rows=[]
    for r in display_rows:
        ev = r.get("eval",{})
        for pc in ev.get("per_car_review",[]) or []:
            reviews_rows.append({
                "QID": r["profile"].get("profile_id",""),
                "Source": pc.get("source",""),
                "Car": pc.get("car",""),
                "Comment": pc.get("comment","")
            })
    if reviews_rows:
        st.dataframe(pd.DataFrame(reviews_rows), use_container_width=True)
    else:
        st.caption("אין פריטי per_car_review להצגה.")

else:
    st.info("אין נתונים להצגה. הפעל הרצה.")

st.caption("© 2025 Car Advisor – A/B Impact Test. כולל סטטוס חי, Retry, הערכה והורדות CSV/JSON.")
