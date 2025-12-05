# -*- coding: utf-8 -*-
# =====================================================================================
# Car Advisor – Benchmark + Stress+++ v15 (Final Safe Version)
# - Recommender: Gemini (Grounding Enabled via Dict)
# - User Simulator: GPT-4o
# - Judge: GPT-4o
# =====================================================================================

import os, io, json, time, random, traceback, zipfile
from datetime import datetime
from typing import Dict, Any, List, Optional, Tuple
from concurrent.futures import ThreadPoolExecutor, as_completed

import streamlit as st
import pandas as pd
import numpy as np
import google.generativeai as genai
from json_repair import repair_json

# --- הסרנו את השורה הבעייתית של ה-Import ---

# OpenAI SDK (חובה עבור המשתמש והשופט)
try:
    from openai import OpenAI
except Exception:
    OpenAI = None

# -------------------------------------------------------------------------------------
# CONFIG
# -------------------------------------------------------------------------------------
st.set_page_config(page_title="Car Advisor – Benchmark / Stress+++ v15", page_icon="🚗", layout="wide")

# --- הגדרת המודלים ---

# 1. המודל שלנו (הממליץ)
# אם Gemini 3 Preview לא עובד לך, שנה ל: "gemini-1.5-pro"
GEMINI_RECOMMENDER_MODEL = "gemini-3-pro-preview" 

# 2. המודל המתחרה/משתמש
OPENAI_USER_MODEL = "gpt-4o"

# 3. המודל השופט
OPENAI_JUDGE_MODEL = "gpt-4o"

# נתיבי קבצים
RUN_DIR = "runs"
os.makedirs(RUN_DIR, exist_ok=True)
ROWS_PATH = os.path.join(RUN_DIR, "ab_rows.json")
PARTIAL_CSV_PATH = os.path.join(RUN_DIR, "ab_results_partial.csv")
FINAL_CSV_PATH = os.path.join(RUN_DIR, "ab_results_final.csv")
MERGED_CSV_PATH = os.path.join(RUN_DIR, "ab_results_merged.csv")

# Stress+++ Paths
STRESS_DIR = os.path.join(RUN_DIR, "stress_v15")
R1_DIR = os.path.join(STRESS_DIR, "round1")
R2_DIR = os.path.join(STRESS_DIR, "round2")
COMBINED_DIR = os.path.join(STRESS_DIR, "combined")
for d in [STRESS_DIR, R1_DIR, R2_DIR, COMBINED_DIR]:
    os.makedirs(d, exist_ok=True)

R1_ROWS_PATH     = os.path.join(R1_DIR, "stress_round1_rows.json")
R1_SUMMARY_CSV   = os.path.join(R1_DIR, "stress_round1_summary.csv")
R1_FAILS_CSV     = os.path.join(R1_DIR, "stress_round1_validation_failures.csv")
R1_ZIP_PATH      = os.path.join(R1_DIR, "stress_round1_bundle.zip")

R2_ROWS_PATH     = os.path.join(R2_DIR, "stress_round2_rows.json")
R2_SUMMARY_CSV   = os.path.join(R2_DIR, "stress_round2_summary.csv")
R2_FAILS_CSV     = os.path.join(R2_DIR, "stress_round2_validation_failures.csv")
R2_ZIP_PATH      = os.path.join(R2_DIR, "stress_round2_bundle.zip")

COMBINED_DIFFS_CSV  = os.path.join(COMBINED_DIR, "stress_combined_diffs.csv")
COMBINED_FAILS_CSV  = os.path.join(COMBINED_DIR, "stress_combined_validation_failures.csv")
COMBINED_ZIP_PATH   = os.path.join(COMBINED_DIR, "stress_combined_bundle.zip")

# Secrets
GEMINI_API_KEY = st.secrets.get("GEMINI_API_KEY", "") or os.getenv("GEMINI_API_KEY", "")
OPENAI_API_KEY = st.secrets.get("OPENAI_API_KEY", "") or os.getenv("OPENAI_API_KEY", "")

if not GEMINI_API_KEY:
    st.warning("⚠️ חסר GEMINI_API_KEY (secrets או env)")
if not OPENAI_API_KEY:
    st.warning("⚠️ חסר OPENAI_API_KEY (secrets או env)")

# Init clients
if GEMINI_API_KEY:
    genai.configure(api_key=GEMINI_API_KEY)
    
    # בדיקת גרסה כדי לוודא שהשרת לא מריץ משהו עתיק
    try:
        st.sidebar.caption(f"Google GenAI Version: {genai.__version__}")
    except:
        pass

    # לקוח למודל ההמלצות (Gemini)
    try:
        # ניסיון להגדיר כלי חיפוש בצורה שתעבוד ברוב הגרסאות
        # אנחנו משתמשים ברשימה של מילונים, שזו הדרך הסטנדרטית
        tools_config = [{'google_search': {}}]
        
        gemini_recommender = genai.GenerativeModel(
            GEMINI_RECOMMENDER_MODEL,
            generation_config={
                "temperature": 0.2,
                "top_p": 0.9,
                "top_k": 40,
            },
            tools=tools_config
        )
    except Exception as e:
        st.error(f"Error initializing Gemini: {e}")
        # במקרה חירום - נטען בלי כלים כדי שהאפליקציה לפחות תעלה
        gemini_recommender = genai.GenerativeModel(
            GEMINI_RECOMMENDER_MODEL
        )
else:
    gemini_recommender = None

# לקוח למודל ה-GPT
oa = OpenAI(api_key=OPENAI_API_KEY) if (OPENAI_API_KEY and OpenAI) else None

# -------------------------------------------------------------------------------------
# PROMPTS
# -------------------------------------------------------------------------------------
def build_gemini_prompt(profile: Dict[str, Any]) -> str:
    return f"""
Please recommend cars for an Israeli customer. Here is the user profile (JSON):
{json.dumps(profile, ensure_ascii=False, indent=2)}

You are in **STRESS+++ reality mode (90%)**.
Act as an **independent automotive data analyst** using live-market style reasoning for Israel.

🔴 **CRITICAL INSTRUCTION: USE GOOGLE SEARCH**
You MUST use the Google Search tool to verify current prices, availability, and trim levels in Israel for TODAY. 
Do not rely on outdated training data.

Hard constraints (MUST):
- Return only ONE top-level JSON object.
- Include all required fields for each car, numeric fields are pure numbers.
- Only models actually sold in Israel (high probability);

Output requirements:
1) Return a SINGLE JSON object with fields: "search_performed", "search_queries", "recommended_cars".
2) search_performed: ALWAYS return True.
3) search_queries: ALWAYS include the exact Hebrew queries you would run.
4) recommended_cars: an array of 5–10 cars. EACH car MUST include:
    - brand, model, year, fuel, gear, turbo, engine_cc, price_range_nis
    - avg_fuel_consumption (+ fuel_method)
    - annual_fee (₪/year) + fee_method
    - reliability_score (1–10) + reliability_method
    - maintenance_cost (₪/year) + maintenance_method
    - safety_rating (1–10) + safety_method
    - insurance_cost (₪/year) + insurance_method
    - resale_value (1–10) + resale_method
    - performance_score (1–10) + performance_method
    - comfort_features (1–10) + comfort_method
    - suitability (1–10) + suitability_method
    - market_supply ("גבוה" / "בינוני" / "נמוך") + supply_method
    - fit_score (0–100)
    - comparison_comment (Hebrew)
    - not_recommended_reason (Hebrew) — at least for one car
    **All explanations must be in Hebrew**

Reality additives:
- Prefer trims with high availability in second-hand market 2014–2020 if budget is low
- Penalize DSG issues where relevant
- Penalize inflated claims (e.g. EV savings ignoring battery degradation)
- For EV: gear must be automatic

Return only JSON.
"""

EVAL_PROMPT = """
אתה שופט מומחה להשוואת מערכות המלצה לרכב בישראל.
תפקידך לקבוע מי מהמודלים (Gemini או GPT) סיפק המלצה מדויקת יותר למציאות הישראלית.

השתמש בידע העדכני ביותר שיש לך כדי לאמת את הנתונים.
עליך לוודא:
1. האם הדגמים המומלצים אכן קיימים בשוק הישראלי בשנתון ובמחיר הנקובים?
2. האם רמות הגימור שהוצעו נמכרו בישראל?
3. האם נתוני צריכת הדלק/חשמל ריאליים?
הורד ניקוד משמעותי על "הזיות" (Hallucinations) או המצאת דגמים.

משקולות לשיפוט:
1. התאמה לצרכי המשתמש – 25%
2. עומק ניתוח מקצועי – 20%
3. בהירות ודיוק במבנה JSON – 20%
4. עקביות פנימית – 15%
5. רלוונטיות לשוק הישראלי (כולל אימות מחירים וזמינות) – 10%
6. איכות הנימוקים בעברית – 10%

פורמט פלט:
{
  "gemini_score": <0-100>,
  "gpt_score": <0-100>,
  "winner": "Gemini"|"ChatGPT"|"Tie",
  "reason": "נימוק קצר בעברית",
  "criteria_breakdown": {
    "fit_to_user": {"gemini":0-100, "gpt":0-100, "note":"עברית קצרה"},
    "technical_depth": {"gemini":0-100, "gpt":0-100, "note":"עברית קצרה"},
    "json_quality": {"gemini":0-100, "gpt":0-100, "note":"עברית קצרה"},
    "consistency": {"gemini":0-100, "gpt":0-100, "note":"עברית קצרה"},
    "israeli_relevance": {"gemini":0-100, "gpt":0-100, "note":"עברית קצרה"},
    "hebrew_quality": {"gemini":0-100, "gpt":0-100, "note":"עברית קצרה"}
  }
}
"""

USER_ONE_LINER = """
היי, אני לא ממש מבין ברכבים. קרא בבקשה את הפרופיל שלי למעלה והמלץ לי על 5–10 דגמים שמתאימים.
תכתוב בעברית פשוטה, בלי מקצועי מדי. תגיד לי למה כל דגם מתאים לי, ואל תציף בטבלאות.
תסביר גם חסרונות חשובים (ביטוח יקר, אמינות, DSG, ירידת ערך, צריכת דלק/חשמל אמיתית).
בסוף תן סיכום קצר – מה הכי מתאים לי ולמה.
"""

# -------------------------------------------------------------------------------------
# PROFILES GENERATION
# -------------------------------------------------------------------------------------
ENGINE = ["בנזין","דיזל","היברידי","פלאג-אין","חשמלי"]
GEAR = ["אוטומט","ידני","DCT/DSG","CVT"]
PRIMARY = ["אמינות","חיסכון בדלק","נוחות","ביצועים","שמירת ערך","בטיחות"]
BODY = ["האצ'בק","סדאן","קרוסאובר","סטיישן","מיני","SUV"]

def build_profile(i:int) -> Dict[str,Any]:
    random.seed(1000+i)
    return {
        "profile_id": f"Q{i:02d}",
        "budget_nis": random.choice([45000,60000,80000,100000,120000,150000,180000,220000,260000]),
        "family_size": random.choice([1,2,3,4,5]),
        "primary_use": random.choice([
            "נסיעות יומיומיות בעיר","נסיעות בין-עירוניות ארוכות",
            "טיולים בסופי שבוע","נסיעות עבודה עם ציוד","נסיעות לעבודה במרכז"
        ]),
        "preferences": {
            "gearbox": random.choice(GEAR),
            "engine_type": random.choice(ENGINE),
            "body_style": random.choice(BODY),
            "priority_primary": random.choice(PRIMARY),
            "priority_secondary": random.sample(PRIMARY, k=2)
        },
        "must_haves": random.sample(
            ["חיישני חניה","בלימה אוטונומית","בקרת שיוט אדפטיבית","מערכת בטיחות מתקדמת","תא מטען גדול","מסך גדול"], k=3
        ),
        "nice_to_have": random.sample(
            ["Sunroof","מערכת שמע פרימיום","טעינה אלחוטית","מושבים חשמליים","ג'אנטים"], k=2
        ),
        "parking": random.choice(["עיר צפופה","פרבר","כפרי"]),
        "region": random.choice(["מרכז","שפלה","צפון","דרום","ירושלים","חוף"]),
        "risk_tolerance": random.choice(["נמוכה","בינונית","גבוהה"])
    }

def build_extreme_profiles(n=12, noise_level: float = 0.9, add_soft_contradictions: bool = True):
    extremes = []
    for i in range(n):
        prof = build_profile(100 + i)
        if i == 0:
            prof["budget_nis"] = 15000; prof["preferences"]["engine_type"] = "חשמלי"
        elif i == 1:
            prof["budget_nis"] = 400000; prof["preferences"]["priority_primary"] = "ביצועים"
        elif i == 2:
            prof["family_size"] = 6; prof["must_haves"] = ["תא מטען גדול","מערכת בטיחות מתקדמת","בלימה אוטונומית"]
        elif i == 3:
            prof["preferences"]["gearbox"] = "ידני"; prof["preferences"]["engine_type"] = "דיזל"
        elif i == 4:
            prof["primary_use"] = "נסיעות בין-עירוניות ארוכות"; prof["budget_nis"] = 180000
        elif i == 5:
            prof["preferences"]["priority_primary"] = "אמינות"; prof["preferences"]["engine_type"] = "היברידי"
        elif i == 6:
            prof["preferences"]["priority_primary"] = "חיסכון בדלק"; prof["region"] = "צפון"
        elif i == 7:
            prof["preferences"]["priority_primary"] = "שמירת ערך"; prof["budget_nis"] = 50000
        elif i == 8:
            prof["risk_tolerance"] = "גבוהה"; prof["preferences"]["priority_primary"] = "ביצועים"
        elif i == 9:
            prof["risk_tolerance"] = "נמוכה"; prof["preferences"]["priority_primary"] = "אמינות"
        elif i == 10:
            prof["preferences"]["engine_type"] = "פלאג-אין"; prof["budget_nis"] = 250000
        elif i == 11:
            prof["budget_nis"] = 20000; prof["preferences"]["engine_type"] = "בנזין"
        prof["profile_id"] = f"X{i+1:02d}"

        if random.random() < noise_level:
            hints = ["העדפה להימנע מ-DSG ישנים", "טווחי מחירים בישראל עלו לאחרונה", "טעינות לילה זמינות בבניין", "נסיעות קצרות בפקקים", "שוק היד-2 חשוב"]
            prof["context_hints"] = random.sample(hints, k=min(3, len(hints)))

        if add_soft_contradictions and random.random() < noise_level:
            prof["soft_constraints"] = random.choice([
                "רוצה גם ביצועים חזקים וגם חיסכון גבוה",
                "תקציב נמוך אך מבקש אבזור פרימיום",
                "רוצה EV אבל חושש מעלויות ביטוח"
            ])
        extremes.append(prof)
    return extremes

# -------------------------------------------------------------------------------------
# UTILS
# -------------------------------------------------------------------------------------
def safe_json(text: Optional[str]) -> Dict[str,Any]:
    if not text: return {}
    try:
        fixed = repair_json(text)
        return json.loads(fixed)
    except: return {}

def load_list(path:str) -> List[Dict[str,Any]]:
    if os.path.exists(path):
        try:
            with open(path,"r",encoding="utf-8") as f: return json.load(f)
        except: return []
    return []

def save_list(path:str, data:List[Dict[str,Any]]):
    with open(path,"w",encoding="utf-8") as f: json.dump(data,f,ensure_ascii=False,indent=2)

def append_item(path:str, item:Dict[str,Any]):
    data = load_list(path)
    data.append(item)
    save_list(path, data)

def call_with_retry(fn, retries=3, backoff=1.5):
    last_err = None
    for i in range(retries):
        try: return fn()
        except Exception as e:
            last_err = e
            time.sleep(backoff*(i+1))
    raise last_err

def _is_num(x): return isinstance(x,(int,float)) and not isinstance(x,bool)

def make_zip(output_path: str, files: List[Tuple[str, str]]):
    try:
        with zipfile.ZipFile(output_path, "w", compression=zipfile.ZIP_DEFLATED) as zf:
            for fp, arc in files:
                if os.path.exists(fp): zf.write(fp, arc)
    except Exception as e: st.warning(f"ZIP Error: {e}")

# -------------------------------------------------------------------------------------
# VALIDATION
# -------------------------------------------------------------------------------------
REQ_NUM = ["reliability_score","maintenance_cost","safety_rating","insurance_cost","resale_value","performance_score","comfort_features","suitability","annual_fee","avg_fuel_consumption","fit_score"]
REQ_CAT = ["brand","model","year","fuel","gear","price_range_nis","market_supply"]

def validate_gemini_car(c: dict) -> list:
    issues = []
    for k in REQ_CAT + REQ_NUM:
        if k not in c: issues.append(f"missing field: {k}")
    if "year" in c and not _is_num(c["year"]): issues.append("year must be numeric")
    return issues

def validate_gemini_payload(gem_json: dict) -> dict:
    out = {"ok": True, "total_cars": 0, "cars_with_issues": 0, "issues": []}
    try:
        cars = (gem_json or {}).get("recommended_cars", [])
        out["total_cars"] = len(cars)
        for idx, c in enumerate(cars):
            if not isinstance(c, dict): continue
            errs = validate_gemini_car(c)
            if errs:
                out["cars_with_issues"] += 1
                out["issues"].append({"car_index": idx, "errors": errs})
        out["ok"] = (out["cars_with_issues"] == 0 and out["total_cars"] > 0)
    except Exception as e:
        out["ok"] = False
        out["issues"].append({"car_index": None, "errors": [f"validator exception: {e}"]})
    return out

def extract_car_tuples(gem_json: dict) -> set:
    s = set()
    try:
        for c in (gem_json or {}).get("recommended_cars", []) or []:
            b, m, y = str(c.get("brand","")).strip(), str(c.get("model","")).strip(), c.get("year", "")
            s.add((b, m, y))
    except: pass
    return s

def jaccard_overlap(a: set, b: set) -> float:
    if not a and not b: return 1.0
    if not a or not b: return 0.0
    return len(a & b) / float(len(a | b))

# -------------------------------------------------------------------------------------
# API CALLS
# -------------------------------------------------------------------------------------
def call_gemini(profile:Dict[str,Any], timeout=180) -> Dict[str,Any]:
    if gemini_recommender is None: return {"_error": "Gemini Recommender client unavailable"}
    prompt = build_gemini_prompt(profile)
    def _do():
        resp = gemini_recommender.generate_content(prompt, request_options={"timeout": timeout})
        text = resp.text if hasattr(resp, 'text') else ""
        return safe_json(text)
    try: return call_with_retry(_do, retries=3)
    except Exception: return {"_error": "Gemini call failed", "_trace": traceback.format_exc()}

def call_gpt_user(profile:Dict[str,Any], timeout=120) -> Dict[str,Any]:
    if oa is None: return {"_raw": "OpenAI client unavailable", "_json": {}}
    def _do():
        resp = oa.chat.completions.create(
            model=OPENAI_USER_MODEL,
            messages=[
                {"role":"user","content": f"User Profile:\n{json.dumps(profile, ensure_ascii=False)}"},
                {"role":"user","content": USER_ONE_LINER.strip()},
            ],
            temperature=0.9
        )
        text = resp.choices[0].message.content
        return {"_raw": text, "_json": safe_json(text)}
    try: return call_with_retry(_do, retries=3)
    except: return {"_raw": "GPT call failed", "_json": {}, "_trace": traceback.format_exc()}

def call_evaluator(profile:Dict[str,Any], gem_json:Dict[str,Any], gpt_pack:Dict[str,Any]) -> Dict[str,Any]:
    if oa is None: return {"gemini_score":0,"gpt_score":0,"winner":"Tie","reason":"No OpenAI","criteria_breakdown":{}}
    def _do():
        msgs = [
            {"role":"system","content":"Hebrew output only. Return JSON only."},
            {"role":"user","content": f"PROFILE:\n{json.dumps(profile, ensure_ascii=False)}"},
            {"role":"user","content": f"GEMINI_JSON:\n{json.dumps(gem_json, ensure_ascii=False)}"},
            {"role":"user","content": f"GPT_RAW:\n{(gpt_pack.get('_raw','') or '')[:2000]}"},
            {"role":"user","content": EVAL_PROMPT}
        ]
        resp = oa.chat.completions.create(model=OPENAI_JUDGE_MODEL, messages=msgs, temperature=0.0)
        return safe_json(resp.choices[0].message.content)
    try: return call_with_retry(_do, retries=2)
    except: return {"gemini_score":0,"gpt_score":0,"winner":"Tie","reason":"Eval Failed","criteria_breakdown":{}}

# -------------------------------------------------------------------------------------
# UI LOGIC
# -------------------------------------------------------------------------------------
def flatten_row(entry:Dict[str,Any]) -> Dict[str,Any]:
    prof = entry.get("profile",{})
    ev = entry.get("eval",{})
    gem_data = entry.get("gemini", {})
    queries = gem_data.get("search_queries", [])
    return {
        "time": entry.get("ts","")[:19].replace("T"," "),
        "QID": prof.get("profile_id",""),
        "תקציב": prof.get("budget_nis",""),
        "Gemini Search Count": len(queries) if queries else 0,
        "Eval: Gemini score": ev.get("gemini_score",""),
        "Eval: GPT score": ev.get("gpt_score",""),
        "Eval: Winner": ev.get("winner",""),
    }

def export_dataframe_now(rows:List[Dict[str,Any]], csv_path:str):
    df = pd.DataFrame([flatten_row(r) for r in rows])
    if len(df): df.to_csv(csv_path, index=False, encoding="utf-8-sig")
    return df

def merge_two_csvs(base_csv: Optional[pd.DataFrame], new_csv: pd.DataFrame) -> pd.DataFrame:
    if base_csv is None or base_csv.empty: return new_csv.copy()
    base = base_csv.drop_duplicates(subset=["QID"], keep="last")
    new = new_csv.drop_duplicates(subset=["QID"], keep="last")
    merged = pd.concat([base[~base["QID"].isin(new["QID"])], new], ignore_index=True)
    return merged.sort_values("QID").reset_index(drop=True)

st.title("🚗 Car Advisor – Gemini 3 Pro (Preview) vs GPT-4o")
st.caption("ממליץ: Gemini 3 Pro (עם חיפוש) | משתמש: GPT-4o | שופט: GPT-4o")

with st.sidebar:
    st.markdown("### ⚙️ Benchmark")
    batch_size = st.slider("Batch Size", 5, 50, 15, 5)
    seed = st.number_input("Seed", 0, 999999, 42)
    st.markdown("---")
    uploaded_prev = st.file_uploader("Upload previous CSV", type=["csv"])
    run_btn = st.button("🚀 Start Benchmark (Standard)")

def build_order_for_batch(done_qids:set, batch:int, seed_val:int)->List[Dict[str,Any]]:
    random.seed(seed_val)
    base_profiles = [build_profile(i+1) for i in range(50)]
    order = [p for p in base_profiles if p["profile_id"] not in done_qids]
    random.shuffle(order)
    return order[:batch]

if run_btn:
    completed_qids = set()
    base_df = None
    if uploaded_prev:
        try:
            base_df = pd.read_csv(uploaded_prev)
            completed_qids = set(base_df["QID"].astype(str))
            st.success(f"Loaded {len(completed_qids)} previous runs.")
        except: st.error("Error reading CSV")
    
    order = build_order_for_batch(completed_qids, batch_size, seed)
    if not order: st.success("Batch complete.")
    else:
        progress_bar = st.progress(0.0)
        t0 = time.perf_counter()
        batch_rows = []
        with ThreadPoolExecutor(max_workers=3) as pool:
            futures = []
            for idx, prof in enumerate(order, start=1):
                def run_one(p=prof, i=idx):
                    gjson = call_gemini(p)
                    gpt = call_gpt_user(p)
                    ev = call_evaluator(p, gjson, gpt)
                    return i, p, gjson, gpt, ev
                futures.append(pool.submit(run_one))
            
            done_count = 0
            for fut in as_completed(futures):
                idx, prof, gem, gpt, ev = fut.result()
                entry = {"ts": datetime.now().isoformat(), "profile": prof, "gemini": gem, "gpt": gpt, "eval": ev}
                append_item(ROWS_PATH, entry)
                batch_rows.append(entry)
                export_dataframe_now(batch_rows, PARTIAL_CSV_PATH)
                done_count += 1
                progress_bar.progress(done_count/len(order))
                
                queries = gem.get("search_queries", [])
                icon = f"🌐 ({len(queries)})" if queries else "🏠"
                st.write(f"• {prof['profile_id']} | Search: {icon} | Winner: {ev.get('winner','?')}")

        df_batch = export_dataframe_now(batch_rows, FINAL_CSV_PATH)
        st.success("Batch Done.")
        if base_df is not None and len(df_batch):
            merged = merge_two_csvs(base_df, df_batch)
            merged.to_csv(MERGED_CSV_PATH, index=False, encoding="utf-8-sig")
        st.dataframe(df_batch, use_container_width=True)

# Stress+++
st.markdown("---")
st.header("🧪 Stress+++ Mode (Gemini w/ Search)")
if "stress_stage" not in st.session_state: st.session_state.stress_stage = "idle"
if "stress_run1_data" not in st.session_state: st.session_state.stress_run1_data = None
if "stress_run2_data" not in st.session_state: st.session_state.stress_run2_data = None

def run_one_stress_round(run_no:int, profiles:List[Dict[str,Any]], out_rows:str, out_sum:str, out_fail:str, workers:int=3):
    rows, failures, run_summary = [], [], {}
    with ThreadPoolExecutor(max_workers=workers) as pool:
        futures = []
        for prof in profiles:
            def r(p=prof):
                gem = call_gemini(p)
                gpt = call_gpt_user(p)
                ev  = call_evaluator(p, gem, gpt)
                val = validate_gemini_payload(gem)
                return p, gem, gpt, ev, val
            futures.append(pool.submit(r))
        
        for i, fut in enumerate(as_completed(futures)):
            p, gem, gpt, ev, val = fut.result()
            if not val["ok"]:
                failures.append({"QID": p["profile_id"], "issues": json.dumps(val["issues"], ensure_ascii=False)})
            
            entry = {"ts": datetime.now().isoformat(), "run": run_no, "profile": p, "gemini": gem, "gpt": gpt, "eval": ev, "validation": val}
            append_item(out_rows, entry)
            rows.append(entry)
            
            queries = gem.get("search_queries", [])
            winner = ev.get("winner", "Tie")
            
            run_summary[p["profile_id"]] = {
                "gem_score": ev.get("gemini_score", 0),
                "gpt_score": ev.get("gpt_score", 0),
                "winner": winner,
                "cars_set": extract_car_tuples(gem),
                "val_ok": val["ok"],
                "search_count": len(queries)
            }
            
            search_icon = f"🌐 ({len(queries)})" if queries else "🏠"
            valid_icon = "✅" if val["ok"] else "❌"
            
            with st.expander(f"{i+1}. {p['profile_id']} | 🏆 {winner} | Search: {search_icon} | Valid: {valid_icon}"):
                st.markdown("### ⚖️ הכרעת השופט (GPT-4o)")
                c1, c2, c3 = st.columns([1,1,3])
                with c1: st.metric("Gemini 3 Score", ev.get("gemini_score", 0))
                with c2: st.metric("GPT-4o Score", ev.get("gpt_score", 0))
                with c3: st.info(ev.get('reason', 'N/A'))
                st.json(ev.get("criteria_breakdown", {}))
                st.divider()
                
                c_gem, c_gpt = st.columns(2)
                with c_gem:
                    st.subheader(f"🤖 Gemini 3 Pro ({len(gem.get('recommended_cars', []))} רכבים)")
                    if queries:
                        with st.popover("🔍 שאילתות חיפוש"): st.code("\n".join(queries))
                    
                    for c in gem.get("recommended_cars", []):
                        st.text(f"• {c.get('brand')} {c.get('model')} {c.get('year')} ({c.get('price_range_nis')}₪)")
                    with st.popover("📄 JSON מלא"): st.json(gem)
                
                with c_gpt:
                    st.subheader("👤 GPT-4o (User)")
                    st.text_area("תשובה", value=gpt.get("_raw", ""), height=200)
                
                st.divider()
                st.caption("📝 פרופיל:")
                st.json(p)

    df_sum = pd.DataFrame([{ "QID": k, "Winner": v["winner"], "Search Queries": v["search_count"] } for k,v in run_summary.items()])
    if len(df_sum): df_sum.to_csv(out_sum, index=False, encoding="utf-8-sig")
    if len(failures): pd.DataFrame(failures).to_csv(out_fail, index=False, encoding="utf-8-sig")
    return rows, df_sum, pd.DataFrame(failures), run_summary

if st.session_state.stress_stage == "idle":
    noise = st.slider("Noise", 0.0, 1.0, 0.9)
    if st.button("Start Round 1"):
        profs = build_extreme_profiles(noise_level=noise)
        r1_res = run_one_stress_round(1, profs, R1_ROWS_PATH, R1_SUMMARY_CSV, R1_FAILS_CSV)
        st.session_state.stress_profiles = profs
        st.session_state.stress_run1_data = r1_res[3]
        make_zip(R1_ZIP_PATH, [(R1_ROWS_PATH,"rows.json"), (R1_SUMMARY_CSV,"sum.csv")])
        st.session_state.stress_stage = "r1_done"
        st.rerun()

elif st.session_state.stress_stage == "r1_done":
    st.success("Round 1 Done.")
    with open(R1_ZIP_PATH, "rb") as f: st.download_button("Download Round 1 ZIP", f, file_name="r1.zip")
    if st.button("Start Round 2"):
        r2_res = run_one_stress_round(2, st.session_state.stress_profiles, R2_ROWS_PATH, R2_SUMMARY_CSV, R2_FAILS_CSV)
        st.session_state.stress_run2_data = r2_res[3]
        
        diffs = compute_combined_diffs(st.session_state.stress_profiles, st.session_state.stress_run1_data, st.session_state.stress_run2_data)
        diffs.to_csv(COMBINED_DIFFS_CSV, index=False, encoding="utf-8-sig")
        make_zip(COMBINED_ZIP_PATH, [(COMBINED_DIFFS_CSV, "diffs.csv")])
        
        st.session_state.stress_stage = "finished"
        st.rerun()

elif st.session_state.stress_stage == "finished":
    st.success("All Done.")
    with open(COMBINED_ZIP_PATH, "rb") as f: st.download_button("Download Combined ZIP", f, file_name="combined.zip")
    if st.button("Reset"):
        st.session_state.stress_stage = "idle"
        st.rerun()

st.caption("© 2025 Car Advisor (Gemini 3 Pro Preview)")
