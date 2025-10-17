# -*- coding: utf-8 -*-
# =====================================================================================
# Car Advisor – Benchmark + Stress++ v14 (Two-Phase with Midway Exports)
# - הכל נשמר מהקוד שלך: Benchmark A/B (Gemini vs GPT) + Evaluator
# - Stress Mode משודרג: 12 פרופילי קצה, שני סבבים; לאחר סבב 1 שמירה והורדה, ואז סבב 2
# - GPT שודרג ל-gpt-4o | Gemini שודרג ל-gemini-2.5-pro
# - תיקוני JSON, ולידציה קשיחה, מדדי עקביות ו-Overlap, ייצוא CSV/JSON מלא
# - Streamlit UI מלא עם מצב ביניים (stage) בין ריצות
# =====================================================================================

import os, json, time, random, traceback
from datetime import datetime
from typing import Dict, Any, List, Optional, Tuple

import streamlit as st
import pandas as pd
import numpy as np
import google.generativeai as genai
from json_repair import repair_json

# OpenAI SDK (אופציונלי)
try:
    from openai import OpenAI
except Exception:
    OpenAI = None

# -------------------------------------------------------------------------------------
# CONFIG
# -------------------------------------------------------------------------------------
st.set_page_config(page_title="Car Advisor – Benchmark / Stress++ v14", page_icon="🚗", layout="wide")

# מודלים
GEMINI_MODEL = "gemini-2.5-pro"   # פרו לאתגר ג'מיני
OPENAI_MODEL = "gpt-4o"           # המודל החזק הזמין

# נתיבי קבצים (Benchmark רגיל)
RUN_DIR = "runs"
os.makedirs(RUN_DIR, exist_ok=True)
ROWS_PATH = os.path.join(RUN_DIR, "ab_rows.json")
PARTIAL_CSV_PATH = os.path.join(RUN_DIR, "ab_results_partial.csv")
FINAL_CSV_PATH = os.path.join(RUN_DIR, "ab_results_final.csv")
MERGED_CSV_PATH = os.path.join(RUN_DIR, "ab_results_merged.csv")

# Stress++ – תיקיות מסודרות לשני סבבים + משולב
STRESS_DIR = os.path.join(RUN_DIR, "stress_v14")
R1_DIR = os.path.join(STRESS_DIR, "round1")
R2_DIR = os.path.join(STRESS_DIR, "round2")
COMBINED_DIR = os.path.join(STRESS_DIR, "combined")
for d in [STRESS_DIR, R1_DIR, R2_DIR, COMBINED_DIR]:
    os.makedirs(d, exist_ok=True)

# קבצי יצוא לסבב 1
R1_ROWS_PATH     = os.path.join(R1_DIR, "stress_round1_rows.json")
R1_SUMMARY_CSV   = os.path.join(R1_DIR, "stress_round1_summary.csv")
R1_FAILS_CSV     = os.path.join(R1_DIR, "stress_round1_validation_failures.csv")

# קבצי יצוא לסבב 2
R2_ROWS_PATH     = os.path.join(R2_DIR, "stress_round2_rows.json")
R2_SUMMARY_CSV   = os.path.join(R2_DIR, "stress_round2_summary.csv")
R2_FAILS_CSV     = os.path.join(R2_DIR, "stress_round2_validation_failures.csv")

# קבצים משולבים
COMBINED_DIFFS_CSV  = os.path.join(COMBINED_DIR, "stress_combined_diffs.csv")
COMBINED_FAILS_CSV  = os.path.join(COMBINED_DIR, "stress_combined_validation_failures.csv")

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
    gemini = genai.GenerativeModel(
        GEMINI_MODEL,
        generation_config={
            "temperature": 0.2,  # יציבות בין ריצות
            "top_p": 0.9,
            "top_k": 40,
        },
    )
else:
    gemini = None

oa = OpenAI(api_key=OPENAI_API_KEY) if (OPENAI_API_KEY and OpenAI) else None

# -------------------------------------------------------------------------------------
# PROMPTS (מקורי + Stress++)
# -------------------------------------------------------------------------------------
def build_gemini_prompt(profile: Dict[str, Any]) -> str:
    """
    גרסת Stress++ – שומרת על הפורמט והדרישות המקוריות שלך 1:1,
    ומוסיפה דרישות אנליטיות להשוואה והעמקה. מחזירה JSON יחיד בלבד.
    """
    return f"""
Please recommend cars for an Israeli customer. Here is the user profile (JSON):
{json.dumps(profile, ensure_ascii=False, indent=2)}

You are in **STRESS++ evaluation mode**.
Act as an **independent automotive data analyst** using live-market style reasoning for Israel.

Output requirements:
1) Return a SINGLE JSON object with fields: "search_performed", "search_queries", "recommended_cars".
2) search_performed: ALWAYS return True. You must simulate live web search (no refusals).
3) search_queries: ALWAYS include the exact queries you would use in Hebrew.
4) recommended_cars: an array of 5–10 cars. EACH car MUST include:
   - brand, model, year, fuel, gear, turbo, engine_cc, price_range_nis
   - avg_fuel_consumption (+ fuel_method):
       * for non-EV: km per liter (number only)
       * for EV: kWh per 100 km (number only)
      **must return methods explanation only in Hebrew**
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
   - fit_score (0–100, number only): הערכת התאמה כוללת ללקוח לפי השקלול הכולל.
   - comparison_comment: ניתוח השוואתי קצר מול מתחרים באותו תקציב (בעברית).
   - not_recommended_reason: אם הדגם פחות מומלץ – פרט למה.
   **All explanations must be in Hebrew.**

5) IMPORTANT:
   - Numeric fields must be numbers only (no text). Do not omit any field.
   - At least ONE car must include a genuine not_recommended_reason.
   - Only models actually sold in Israel.

6) Additional analytical requirements:
   - השווה בין שני הדגמים המובילים והסבר מי עדיף ולמה (בעברית).
   - פרט שיקולי תחזוקה/אמינות/ביצועים מנוגדים.
   - כלול לפחות רכב אחד שהוא "overrated" וכתוב למה הציבור טועה לגביו.

Return only JSON.
"""

USER_ONE_LINER = "תמליץ לי 5–10 רכבים על פי הצרכים שצוינו בפרופיל. תן נימוק ברור לכל דגם בעברית."

EVAL_PROMPT = """
דרג שתי תשובות (Gemini JSON ו-GPT Raw/JSON) עבור אותו פרופיל.
קריטריונים: רלוונטיות לצרכים, עומק והיגיון מקצועי, עקביות, בהירות ותועלת מעשית.
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
# PROFILES GENERATION
# -------------------------------------------------------------------------------------
ENGINE = ["בנזין","דיזל","היברידי","פלאג-אין","חשמלי"]
GEAR = ["אוטומט","ידני","DCT/DSG","CVT"]
PRIMARY = ["אמינות","חיסכון בדלק","נוחות","ביצועים","שמירת ערך","בטיחות"]
BODY = ["האצ'בק","סדאן","קרוסאובר","סטיישן","מיני","SUV"]

def build_profile(i:int) -> Dict[str,Any]:
    random.seed(1000+i)  # יציבות בין ריצות
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

def build_extreme_profiles(n=12):
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
        extremes.append(prof)
    return extremes

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

def save_list(path:str, data:List[Dict[str,Any]]):
    with open(path,"w",encoding="utf-8") as f:
        json.dump(data,f,ensure_ascii=False,indent=2)

def append_item(path:str, item:Dict[str,Any]):
    data = load_list(path)
    data.append(item)
    save_list(path, data)

def call_with_retry(fn, retries=3, backoff=1.5):
    last_err = None
    for i in range(retries):
        try:
            return fn()
        except Exception as e:
            last_err = e
            time.sleep(backoff*(i+1))
    raise last_err

def _is_num(x):
    return isinstance(x,(int,float)) and not isinstance(x,bool)

# -------------------------------------------------------------------------------------
# VALIDATION & OVERLAP
# -------------------------------------------------------------------------------------
REQ_NUM = [
    "reliability_score","maintenance_cost","safety_rating",
    "insurance_cost","resale_value","performance_score",
    "comfort_features","suitability","annual_fee","avg_fuel_consumption"
]
REQ_CAT = ["brand","model","year","fuel","gear","price_range_nis","market_supply"]

def validate_gemini_car(c: dict) -> list:
    issues = []
    for k in REQ_CAT + REQ_NUM:
        if k not in c:
            issues.append(f"missing field: {k}")
    if "year" in c and not _is_num(c["year"]):
        issues.append("year must be numeric")
    for k in ["reliability_score","safety_rating","resale_value","performance_score","comfort_features","suitability"]:
        if k in c:
            v = c[k]
            if not _is_num(v) or not (1 <= float(v) <= 10):
                issues.append(f"{k} must be number 1–10")
    for k in ["maintenance_cost","insurance_cost","annual_fee"]:
        if k in c:
            v = c[k]
            if not _is_num(v) or float(v) < 0:
                issues.append(f"{k} must be >=0")
    if "fuel" in c and "avg_fuel_consumption" in c:
        av = c["avg_fuel_consumption"]
        if not _is_num(av) or float(av) <= 0:
            issues.append("avg_fuel_consumption must be positive number")
    if str(c.get("fuel","")).lower() == "electric" and str(c.get("gear","")).lower() != "automatic":
        issues.append("EV must be automatic")
    if "market_supply" in c and str(c["market_supply"]) not in ["גבוה","בינוני","נמוך"]:
        issues.append("market_supply must be one of: גבוה/בינוני/נמוך")
    return issues

def validate_gemini_payload(gem_json: dict) -> dict:
    out = {"ok": True, "total_cars": 0, "cars_with_issues": 0, "issues": []}
    try:
        cars = (gem_json or {}).get("recommended_cars", [])
        out["total_cars"] = len(cars)
        for idx, c in enumerate(cars):
            if not isinstance(c, dict):
                out["issues"].append({"car_index": idx, "errors": ["car is not object"]})
                continue
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
            b = str(c.get("brand","")).strip()
            m = str(c.get("model","")).strip()
            y = c.get("year", "")
            y = int(y) if _is_num(y) else str(y).strip()
            s.add((b, m, y))
    except Exception:
        pass
    return s

def jaccard_overlap(a: set, b: set) -> float:
    if not a and not b:
        return 1.0
    if not a or not b:
        return 0.0
    return len(a & b) / float(len(a | b))

# -------------------------------------------------------------------------------------
# MODEL CALLS
# -------------------------------------------------------------------------------------
def call_gemini(profile:Dict[str,Any], timeout=180) -> Dict[str,Any]:
    if gemini is None:
        return {"_error": "Gemini client unavailable"}
    prompt = build_gemini_prompt(profile)
    def _do():
        resp = gemini.generate_content(prompt, request_options={"timeout": timeout})
        text = None
        try:
            text = resp.text
        except Exception:
            try:
                text = resp.candidates[0].content.parts[0].text
            except Exception:
                text = ""
        return safe_json(text)
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
            temperature=0.8,
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
            {"role":"user","content": f"GPT_RAW:\n{(gpt_pack.get('_raw','') or '')[:2000]}"},
            {"role":"user","content": EVAL_PROMPT}
        ]
        resp = oa.chat.completions.create(model=OPENAI_MODEL, messages=msgs, temperature=0.0)
        return safe_json(resp.choices[0].message.content)
    try:
        return call_with_retry(_do, retries=2)
    except Exception:
        return {"gemini_score":0,"gpt_score":0,"winner":"Tie","reason":"Evaluation failed","per_car_review":[]}

# -------------------------------------------------------------------------------------
# BENCHMARK (כללי, כמו אצלך)
# -------------------------------------------------------------------------------------
def flatten_row(entry:Dict[str,Any]) -> Dict[str,Any]:
    prof = entry.get("profile",{})
    ev   = entry.get("eval",{})
    return {
        "time": entry.get("ts","")[:19].replace("T"," "),
        "QID": prof.get("profile_id",""),
        "תקציב": prof.get("budget_nis",""),
        "שימוש": prof.get("primary_use",""),
        "עדיפות": (prof.get("preferences",{}) or {}).get("priority_primary",""),
        "דלק": (prof.get("preferences",{}) or {}).get("engine_type",""),
        "תיבה": (prof.get("preferences",{}) or {}).get("gearbox",""),
        "משפחה": prof.get("family_size",""),
        "Eval: Gemini score": ev.get("gemini_score",""),
        "Eval: GPT score": ev.get("gpt_score",""),
        "Eval: Winner": ev.get("winner",""),
        "Eval: Reason": ev.get("reason",""),
    }

def write_csv_now(rows:List[Dict[str,Any]], path:str):
    try:
        df = pd.DataFrame([flatten_row(r) for r in rows])
        if len(df):
            df.to_csv(path, index=False, encoding="utf-8-sig")
    except Exception as e:
        st.warning(f"שגיאה בשמירת CSV: {e}")

def merge_two_csvs(base_csv: Optional[pd.DataFrame], new_csv: pd.DataFrame) -> pd.DataFrame:
    if base_df is None or base_df.empty:
        return new_csv.copy()
    base = base_csv.drop_duplicates(subset=["QID"], keep="last")
    new  = new_csv.drop_duplicates(subset=["QID"], keep="last")
    merged = pd.concat([base[~base["QID"].isin(new["QID"])], new], ignore_index=True)
    return merged.sort_values("QID").reset_index(drop=True)

# -------------------------------------------------------------------------------------
# STRESS++ HELPERS – ריצה לסבב אחד בלבד (להפעלה כשלב)
# -------------------------------------------------------------------------------------
def run_one_stress_round(run_no:int, profiles:List[Dict[str,Any]], out_rows_path:str, out_summary_csv:str, out_fails_csv:str) -> Tuple[List[Dict[str,Any]], pd.DataFrame, pd.DataFrame, dict]:
    """
    מריץ סבב אחד (12 פרופילים), שומר לוג JSON, מחזיר:
    - rows: כל הרשומות
    - df_summary: טבלת סיכום (ציונים/מנצח/תקינות)
    - df_fails: כשלים בוולידציה
    - run_summary_data: dict לניתוח בשלב המשולב
    """
    rows = []
    failures = []
    run_summary = {}

    st.info(f"🔁 הרצה {run_no}/2…")
    for prof in profiles:
        qid = prof["profile_id"]

        gem = call_gemini(prof)
        gpt = call_gpt_user(prof)
        ev  = call_evaluator(prof, gem, gpt)

        val = validate_gemini_payload(gem)
        if not val["ok"]:
            failures.append({
                "QID": qid, "run": run_no,
                "total_cars": val["total_cars"],
                "cars_with_issues": val["cars_with_issues"],
                "issues": json.dumps(val["issues"], ensure_ascii=False)
            })

        entry = {"ts": datetime.now().isoformat(), "run": run_no, "profile": prof, "gemini": gem, "gpt": gpt, "eval": ev, "validation": val}
        append_item(out_rows_path, entry)
        rows.append(entry)

        run_summary[qid] = {
            "gem_score": ev.get("gemini_score", 0),
            "gpt_score": ev.get("gpt_score", 0),
            "winner": ev.get("winner", "Tie"),
            "cars_set": extract_car_tuples(gem),
            "val_ok": val["ok"]
        }

        st.write(f"• {qid} | Gemini={ev.get('gemini_score','?')} / GPT={ev.get('gpt_score','?')} | Winner={ev.get('winner','?')} | Valid={val['ok']}")

    # טבלת סיכום לסבב
    summary_rows = []
    for qid, r in run_summary.items():
        summary_rows.append({
            "QID": qid,
            "Gemini Score": r["gem_score"],
            "GPT Score": r["gpt_score"],
            "Winner": r["winner"],
            "Valid": "כן" if r["val_ok"] else "לא",
            "Cars Count": len(r["cars_set"])
        })
    df_summary = pd.DataFrame(summary_rows).sort_values("QID").reset_index(drop=True)
    df_fails = pd.DataFrame(failures)

    # שמירה
    try:
        if len(df_summary):
            df_summary.to_csv(out_summary_csv, index=False, encoding="utf-8-sig")
        if len(df_fails):
            df_fails.to_csv(out_fails_csv, index=False, encoding="utf-8-sig")
    except Exception as e:
        st.warning(f"שגיאה בשמירת קבצי הסבב {run_no}: {e}")

    return rows, df_summary, df_fails, run_summary

def compute_combined_diffs(profiles:List[Dict[str,Any]], r1:dict, r2:dict) -> pd.DataFrame:
    diffs_rows = []
    for prof in profiles:
        qid = prof["profile_id"]
        r1d = r1.get(qid, {})
        r2d = r2.get(qid, {})
        gdiff = abs(float(r1d.get("gem_score", 0)) - float(r2d.get("gem_score", 0)))
        tdiff = abs(float(r1d.get("gpt_score", 0)) - float(r2d.get("gpt_score", 0)))
        win_consistent = (str(r1d.get("winner","")) == str(r2d.get("winner","")))
        overlap = jaccard_overlap(r1d.get("cars_set", set()), r2d.get("cars_set", set()))
        both_valid = bool(r1d.get("val_ok", False) and r2d.get("val_ok", False))
        diffs_rows.append({
            "QID": qid,
            "Gemini Δ": gdiff,
            "GPT Δ": tdiff,
            "Winner R1": r1d.get("winner",""),
            "Winner R2": r2d.get("winner",""),
            "Winner Consistent": "כן" if win_consistent else "לא",
            "Cars Overlap (Jaccard)": round(overlap, 3),
            "Both Runs Valid": "כן" if both_valid else "לא"
        })
    return pd.DataFrame(diffs_rows).sort_values("QID").reset_index(drop=True)

# -------------------------------------------------------------------------------------
# UI – HEADER
# -------------------------------------------------------------------------------------
st.title("🚗 Car Advisor – Benchmark + Stress++ v14")
st.caption("A/B מלא + בדיקת עמידות דו־שלבית. אחרי סבב 1: הורדה והעלאה, ואז ממשיכים לסבב 2 ומייצרים קבצים משולבים.")

with st.sidebar:
    st.markdown("### ⚙️ הגדרות Benchmark")
    batch_size = st.slider("מספר שאלונים בסבב (להקטין עומס/Timeout)", min_value=5, max_value=50, value=15, step=5)
    seed = st.number_input("Seed לסדר ההרצה", min_value=0, max_value=999999, value=42, step=1)
    st.markdown("---")
    uploaded_prev = st.file_uploader("📤 העלה CSV קודם להמשך (אופציונלי)", type=["csv"])
    st.markdown("---")
    run_btn = st.button("🚀 התחל / המשך סבב Benchmark")

# -------------------------------------------------------------------------------------
# BENCHMARK RUN (כמו אצלך, ללא שינוי מהותי)
# -------------------------------------------------------------------------------------
ENGINE  # רק כדי שלא תהיה אזהרת שימוש

def build_order_for_batch(done_qids:set, batch:int, seed_val:int)->List[Dict[str,Any]]:
    random.seed(seed_val)
    base_profiles = [build_profile(i+1) for i in range(50)]
    order = [p for p in base_profiles if p["profile_id"] not in done_qids]
    random.shuffle(order)
    return order[:batch]

def export_dataframe_now(rows:List[Dict[str,Any]], csv_path:str):
    df = pd.DataFrame([flatten_row(r) for r in rows])
    if len(df):
        df.to_csv(csv_path, index=False, encoding="utf-8-sig")
        return df
    return pd.DataFrame()

if run_btn:
    completed_qids = set()
    base_df = None
    if uploaded_prev is not None:
        try:
            base_df = pd.read_csv(uploaded_prev)
            completed_qids = set(base_df["QID"].astype(str))
            st.success(f"✅ נטענו {len(completed_qids)} שאלונים מקובץ קודם – נמשיך משם.")
        except Exception as e:
            st.error(f"שגיאה בקריאת הקובץ שהועלה: {e}")
            base_df = None

    if not completed_qids and os.path.exists(PARTIAL_CSV_PATH):
        try:
            df_partial = pd.read_csv(PARTIAL_CSV_PATH)
            completed_qids = set(df_partial["QID"].astype(str))
            base_df = df_partial.copy()
            st.info(f"נמצאו על הדיסק {len(completed_qids)} שאלונים – ממשיכים מהם (Partial CSV).")
        except Exception as e:
            st.warning(f"לא הצלחתי לקרוא partial קיים: {e}")

    order = build_order_for_batch(completed_qids, batch_size, seed)
    total_in_batch = len(order)
    if total_in_batch == 0:
        st.success("אין מה להשלים – כל ה-50 כבר הושלמו בקובץ שהועלה/קיים.")
    else:
        phase_box = st.empty()
        progress_bar = st.progress(0.0)
        meta_cols = st.columns(3)
        info_box = st.empty()
        t0 = time.perf_counter()

        batch_rows: List[Dict[str,Any]] = []
        for idx, prof in enumerate(order, start=1):
            qid = prof["profile_id"]
            phase_box.info(f"🌀 [{idx}/{total_in_batch}] {qid} – Gemini עובד…")
            gem = call_gemini(prof)

            phase_box.info(f"🤖 [{idx}/{total_in_batch}] {qid} – ChatGPT עובד…")
            gpt = call_gpt_user(prof)

            phase_box.info(f"⚖️ [{idx}/{total_in_batch}] {qid} – Evaluator מדרג…")
            ev  = call_evaluator(prof, gem, gpt)

            entry = {"ts": datetime.now().isoformat(), "profile": prof, "gemini": gem, "gpt": gpt, "eval": ev}
            append_item(ROWS_PATH, entry)
            batch_rows.append(entry)

            export_dataframe_now(batch_rows, PARTIAL_CSV_PATH)

            done = idx
            left = total_in_batch - done
            elapsed = time.perf_counter() - t0
            eta = (elapsed/done)*left if done else 0.0
            with meta_cols[0]: st.metric("הושלמו בסבב זה", f"{done}/{total_in_batch}")
            with meta_cols[1]: st.metric("נשאר בסבב", f"{left}")
            with meta_cols[2]: st.metric("זמן/ETA (דק')", f"{elapsed/60:.1f} / {eta/60:.1f}")
            progress_bar.progress(done/total_in_batch)
            info_box.caption("💾 נשמר Partial אחרי כל שאלון. אם נסגר/נעצר – יש ממה להמשיך.")

        df_batch = export_dataframe_now(batch_rows, FINAL_CSV_PATH)
        st.success(f"✅ הסבב הושלם: {len(batch_rows)} שאלונים. נוצר Final CSV לסבב זה.")
        if base_df is not None and len(df_batch):
            merged = merge_two_csvs(base_df, df_batch)
            merged.to_csv(MERGED_CSV_PATH, index=False, encoding="utf-8-sig")
            st.info(f"🧩 בוצע Merge עם הקובץ הקודם. סה\"כ כעת: {len(merged)} רשומות ייחודיות.")
        st.dataframe(df_batch, use_container_width=True)

        c1, c2, c3 = st.columns(3)
        with c1:
            if os.path.exists(PARTIAL_CSV_PATH):
                with open(PARTIAL_CSV_PATH,"rb") as f:
                    st.download_button("⬇️ הורד Partial CSV (חי)", f, file_name="ab_results_partial.csv", mime="text/csv")
        with c2:
            if os.path.exists(FINAL_CSV_PATH):
                with open(FINAL_CSV_PATH,"rb") as f:
                    st.download_button("⬇️ הורד Final CSV (סבב זה)", f, file_name="ab_results_final.csv", mime="text/csv")
        with c3:
            if os.path.exists(MERGED_CSV_PATH):
                with open(MERGED_CSV_PATH,"rb") as f:
                    st.download_button("⬇️ הורד Merged CSV", f, file_name="ab_results_merged.csv", mime="text/csv")

        st.subheader("🔎 פירוט per_car_review (אם קיים)")
        rows_view=[]
        for r in batch_rows:
            for pc in r.get("eval",{}).get("per_car_review",[]) or []:
                rows_view.append({
                    "QID": r["profile"].get("profile_id",""),
                    "Source": pc.get("source",""),
                    "Car": pc.get("car",""),
                    "Comment": pc.get("comment","")
                })
        if rows_view:
            st.dataframe(pd.DataFrame(rows_view), use_container_width=True)
        else:
            st.caption("אין פריטי per_car_review להצגה עבור סבב זה.")

# -------------------------------------------------------------------------------------
# STAGEFUL STRESS++ (שני שלבים עם שמירה/הורדה באמצע)
# -------------------------------------------------------------------------------------
st.markdown("---")
st.header("🧪 Stress++ Mode (שני שלבים עם יצוא באמצע)")
st.caption("סבב 1 → שמירה והורדה → סבב 2 → שילוב תוצאות.")

if "stress_stage" not in st.session_state:
    st.session_state.stress_stage = "idle"     # idle | r1_done (await_r2) | finished
if "stress_profiles" not in st.session_state:
    st.session_state.stress_profiles = None
if "stress_run1_data" not in st.session_state:
    st.session_state.stress_run1_data = None   # dict (ל-hold נתוני ריצה 1 ל-combined)
if "stress_run2_data" not in st.session_state:
    st.session_state.stress_run2_data = None

# שלב 1 – הפעלה ראשונה
if st.session_state.stress_stage == "idle":
    if st.button("🧪 הפעל סבב 1 (12 פרופילי קצה)"):
        # איפוס קבצי round1
        try:
            if os.path.exists(R1_ROWS_PATH): os.remove(R1_ROWS_PATH)
        except Exception:
            pass

        profiles = build_extreme_profiles()
        rows, df_summary, df_fails, run1_summary_dict = run_one_stress_round(
            run_no=1,
            profiles=profiles,
            out_rows_path=R1_ROWS_PATH,
            out_summary_csv=R1_SUMMARY_CSV,
            out_fails_csv=R1_FAILS_CSV
        )

        # שמירה לזיכרון ה־session להמשך
        st.session_state.stress_profiles = profiles
        st.session_state.stress_run1_data = run1_summary_dict
        st.session_state.stress_stage = "r1_done"

# שלב ביניים – הורדה/העלאה של סבב 1, ואז המשך לסבב 2
elif st.session_state.stress_stage == "r1_done":
    st.success("✅ סבב 1 הושלם. ניתן להוריד את קבצי הסבב הראשון:")
    col1, col2, col3 = st.columns(3)
    with col1:
        if os.path.exists(R1_ROWS_PATH):
            with open(R1_ROWS_PATH, "rb") as f:
                st.download_button("⬇️ הורד Round 1 – rows.json", f, file_name="stress_round1_rows.json", mime="application/json")
    with col2:
        if os.path.exists(R1_SUMMARY_CSV):
            with open(R1_SUMMARY_CSV, "rb") as f:
                st.download_button("⬇️ הורד Round 1 – summary.csv", f, file_name="stress_round1_summary.csv", mime="text/csv")
    with col3:
        if os.path.exists(R1_FAILS_CSV):
            with open(R1_FAILS_CSV, "rb") as f:
                st.download_button("⬇️ הורד Round 1 – validation_failures.csv", f, file_name="stress_round1_validation_failures.csv", mime="text/csv")

    st.info("כעת ניתן להעלות/לשמור את הקבצים האלו. לאחר מכן המשך להפעלת סבב 2.")
    if st.button("▶️ המשך להרצה 2 (ולאחד תוצאות)"):
        # איפוס קבצי round2
        try:
            if os.path.exists(R2_ROWS_PATH): os.remove(R2_ROWS_PATH)
        except Exception:
            pass

        profiles = st.session_state.stress_profiles or build_extreme_profiles()
        rows2, df_summary2, df_fails2, run2_summary_dict = run_one_stress_round(
            run_no=2,
            profiles=profiles,
            out_rows_path=R2_ROWS_PATH,
            out_summary_csv=R2_SUMMARY_CSV,
            out_fails_csv=R2_FAILS_CSV
        )

        # שמירת נתוני ריצה 2
        st.session_state.stress_run2_data = run2_summary_dict

        # חישוב שילוב ופערים
        df_diffs = compute_combined_diffs(profiles, st.session_state.stress_run1_data, st.session_state.stress_run2_data)
        df_fails_combined = pd.DataFrame()
        # איחוד כשלים משני הסבבים אם קיימים
        if os.path.exists(R1_FAILS_CSV):
            try:
                df1f = pd.read_csv(R1_FAILS_CSV)
                df_fails_combined = pd.concat([df_fails_combined, df1f], ignore_index=True)
            except Exception:
                pass
        if os.path.exists(R2_FAILS_CSV):
            try:
                df2f = pd.read_csv(R2_FAILS_CSV)
                df_fails_combined = pd.concat([df_fails_combined, df2f], ignore_index=True)
            except Exception:
                pass

        # שמירה לקבצים משולבים
        try:
            if len(df_diffs):
                df_diffs.to_csv(COMBINED_DIFFS_CSV, index=False, encoding="utf-8-sig")
            if len(df_fails_combined):
                df_fails_combined.to_csv(COMBINED_FAILS_CSV, index=False, encoding="utf-8-sig")
        except Exception as e:
            st.warning(f"שגיאה בשמירת קבצי COMBINED: {e}")

        # הצגה + מדדים
        st.subheader("📊 פערי ציונים בין ריצות (Δ) ועקביות מנצח")
        if len(df_diffs):
            st.dataframe(df_diffs, use_container_width=True)
            avg_gem_delta = float(np.mean(df_diffs["Gemini Δ"])) if len(df_diffs) else 0.0
            avg_gpt_delta = float(np.mean(df_diffs["GPT Δ"])) if len(df_diffs) else 0.0
            avg_overlap   = float(np.mean(df_diffs["Cars Overlap (Jaccard)"])) if len(df_diffs) else 0.0
            winner_consistency_rate = (df_diffs["Winner Consistent"].eq("כן").mean()*100.0) if len(df_diffs) else 0.0
            both_valid_rate = (df_diffs["Both Runs Valid"].eq("כן").mean()*100.0) if len(df_diffs) else 0.0

            c1, c2, c3, c4, c5 = st.columns(5)
            with c1: st.metric("ממוצע Δ Gemini", f"{avg_gem_delta:.2f}")
            with c2: st.metric("ממוצע Δ GPT", f"{avg_gpt_delta:.2f}")
            with c3: st.metric("ממוצע Overlap", f"{avg_overlap:.2f}")
            with c4: st.metric("Winner Consistency", f"{winner_consistency_rate:.1f}%")
            with c5: st.metric("Valid בשתי הריצות", f"{both_valid_rate:.1f}%")

            st.markdown("### 📈 גרף Δ ציונים לפי QID")
            chart_delta = df_diffs[["QID","Gemini Δ","GPT Δ"]].set_index("QID")
            st.line_chart(chart_delta)
            st.markdown("### 📈 גרף יציבות רשימות (Cars Overlap)")
            chart_overlap = df_diffs[["QID","Cars Overlap (Jaccard)"]].set_index("QID")
            st.bar_chart(chart_overlap)
        else:
            st.caption("אין נתוני דלתא לתצוגה.")

        st.subheader("🧯 כשלים בוולידציה – משולב")
        if len(df_fails_combined):
            st.dataframe(df_fails_combined, use_container_width=True)
        else:
            st.caption("אין כשלים משולבים – כל הרכבים עברו את בדיקות הסכמה והטווחים.")

        # כפתורי הורדה למשולבים ולסבב 2
        st.markdown("### ⬇️ הורדת קבצי סבב 2 ומשולב")
        d1, d2, d3, d4, d5 = st.columns(5)
        with d1:
            if os.path.exists(R2_ROWS_PATH):
                with open(R2_ROWS_PATH, "rb") as f:
                    st.download_button("⬇️ Round 2 – rows.json", f, file_name="stress_round2_rows.json", mime="application/json")
        with d2:
            if os.path.exists(R2_SUMMARY_CSV):
                with open(R2_SUMMARY_CSV, "rb") as f:
                    st.download_button("⬇️ Round 2 – summary.csv", f, file_name="stress_round2_summary.csv", mime="text/csv")
        with d3:
            if os.path.exists(R2_FAILS_CSV):
                with open(R2_FAILS_CSV, "rb") as f:
                    st.download_button("⬇️ Round 2 – validation_failures.csv", f, file_name="stress_round2_validation_failures.csv", mime="text/csv")
        with d4:
            if os.path.exists(COMBINED_DIFFS_CSV):
                with open(COMBINED_DIFFS_CSV, "rb") as f:
                    st.download_button("⬇️ Combined – diffs.csv", f, file_name="stress_combined_diffs.csv", mime="text/csv")
        with d5:
            if os.path.exists(COMBINED_FAILS_CSV):
                with open(COMBINED_FAILS_CSV, "rb") as f:
                    st.download_button("⬇️ Combined – validation_failures.csv", f, file_name="stress_combined_validation_failures.csv", mime="text/csv")

        st.session_state.stress_stage = "finished"

# שלב סיום
elif st.session_state.stress_stage == "finished":
    st.success("🎉 שני הסבבים הושלמו ונוצרו קבצים משולבים. ניתן להריץ מחדש אם תרצה.")
    if st.button("🔄 איפוס מצב Stress++"):
        st.session_state.stress_stage = "idle"
        st.session_state.stress_profiles = None
        st.session_state.stress_run1_data = None
        st.session_state.stress_run2_data = None

# -------------------------------------------------------------------------------------
# FOOTER
# -------------------------------------------------------------------------------------
st.caption("© 2025 Car Advisor – Benchmark/Stress++ v14. שני שלבים עם יצוא ביניים, תיקוני JSON ולידציה קשיחה.")
