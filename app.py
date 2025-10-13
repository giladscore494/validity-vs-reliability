# -*- coding: utf-8 -*-
# =====================================================================================
# Car Advisor – A/B Harness (Gemini "Tool" Prompt vs ChatGPT "User") + Resume + AutoSave
# 50 שאלונים אמיתיים (סימולציה) • חיבור מלא לג'מיני ול־GPT • אפשרות להמשך מהרצה קיימת
# כולל:
# - דילוג על QIDs קיימים (Resume אמיתי מקובץ CSV קודם או מהרצה קודמת)
# - Auto-Save: עדכון CSV אחרי כל שאלון, כך שגם אם נעצר – יש קובץ מוכן להמשך
# - סטטוס חי: שלב נוכחי (Gemini/GPT/Evaluator), אחוזים, טיימר, כמה נשאר
# - Evaluator: ציון, נימוק וסקירת per_car_review
# - שמירה מסודרת ל-JSON (ab_rows.json) + יצוא CSV (ab_results_partial.csv/ab_results_final.csv)
# - Merge עם CSV קודם אם הועלה
# =====================================================================================

import os, json, time, random, traceback
from datetime import datetime
from typing import Dict, Any, List, Optional

import streamlit as st
import pandas as pd
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
st.set_page_config(page_title="Car Advisor – Benchmark / Resume", page_icon="🚗", layout="wide")

GEMINI_MODEL = "gemini-2.5-flash"
OPENAI_MODEL = "gpt-4o-mini"  # ניתן לשנות ל-gpt-4o וכו'

RUN_DIR = "runs"
os.makedirs(RUN_DIR, exist_ok=True)
ROWS_PATH = os.path.join(RUN_DIR, "ab_rows.json")               # יומן מלא של כל הרשומות
PARTIAL_CSV_PATH = os.path.join(RUN_DIR, "ab_results_partial.csv")  # נשמר "חי" אחרי כל שאלון
FINAL_CSV_PATH = os.path.join(RUN_DIR, "ab_results_final.csv")      # נשמר בסיום ריצה מלאה/בלוק
MERGED_CSV_PATH = os.path.join(RUN_DIR, "ab_results_merged.csv")    # מיזוג עם קובץ קודם (אם הועלה)

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
def build_gemini_prompt(profile: Dict[str, Any]) -> str:
    # זה הפרומפט "כלי מקצועי" – כולל חובת חיפוש ופורמט קשיח
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

# "משתמש רגיל" – GPT מקבל רק פרופיל ובקשה קצרה
USER_ONE_LINER = "תמליץ לי 5–10 רכבים על פי הצרכים שצוינו בפרופיל. תן נימוק ברור לכל דגם בעברית."

# Evaluator – ציון לכל מודל, מנצח, נימוק, וסקירת per_car_review
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
# BUILD 50 PROFILES (סט קבוע/רנדומלי אך דטרמיניסטי לפי seed)
# -------------------------------------------------------------------------------------
ENGINE = ["בנזין","דיזל","היברידי","פלאג-אין","חשמלי"]
GEAR = ["אוטומט","ידני","DCT/DSG","CVT"]
PRIMARY = ["אמינות","חיסכון בדלק","נוחות","ביצועים","שמירת ערך","בטיחות"]
BODY = ["האצ'בק","סדאן","קרוסאובר","סטיישן","מיני","SUV"]

def build_profile(i:int) -> Dict[str,Any]:
    random.seed(1000+i)  # כדי שיהיה יציב בין ריצות
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

def flatten_row(entry:Dict[str,Any]) -> Dict[str,Any]:
    """שורה לטבלה הסופית/ביניים – קריא וברור כמו שביקשת."""
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
    """כתיבה מיידית של CSV מכל הרשומות שבזיכרון (לנפילות / עצירות לא צפויות)."""
    try:
        df = pd.DataFrame([flatten_row(r) for r in rows])
        if len(df):
            df.to_csv(path, index=False, encoding="utf-8-sig")
    except Exception as e:
        st.warning(f"שגיאה בשמירת CSV: {e}")

def merge_two_csvs(base_csv: Optional[pd.DataFrame], new_csv: pd.DataFrame) -> pd.DataFrame:
    """מיזוג לפי QID (לוקח את החדש במקרה כפילות)."""
    if base_csv is None or base_csv.empty:
        return new_csv.copy()
    base = base_csv.drop_duplicates(subset=["QID"], keep="last")
    new  = new_csv.drop_duplicates(subset=["QID"], keep="last")
    merged = pd.concat([base[~base["QID"].isin(new["QID"])], new], ignore_index=True)
    return merged.sort_values("QID").reset_index(drop=True)

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
            temperature=0.6
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
# UI – HEADER
# -------------------------------------------------------------------------------------
st.title("🚗 Car Advisor – Benchmark + Resume + Auto-Save")
st.caption("מריץ עד 50 שאלונים בסבבים, שומר אוטומטית אחרי כל שאלון, וממשיך מהרצה קודמת אם הועלה CSV.")

with st.sidebar:
    st.markdown("### ⚙️ הגדרות")
    batch_size = st.slider("מספר שאלונים בסבב (להקטין עומס/Timeout)", min_value=5, max_value=50, value=15, step=5)
    seed = st.number_input("Seed לסדר ההרצה", min_value=0, max_value=999999, value=42, step=1)
    st.markdown("---")
    uploaded_prev = st.file_uploader("📤 העלה CSV קודם להמשך (אופציונלי)", type=["csv"])
    st.markdown("---")
    run_btn = st.button("🚀 התחל / המשך סבב")

# כפתור הורדת CSV קיים גם בלי להריץ
st.markdown("#### ⬇️ הורדות קיימות")
cols = st.columns(3)
with cols[0]:
    if os.path.exists(PARTIAL_CSV_PATH):
        with open(PARTIAL_CSV_PATH,"rb") as f:
            st.download_button("הורד Partial CSV (חי)", f, file_name="ab_results_partial.csv", mime="text/csv")
    else:
        st.caption("אין Partial CSV עדיין.")
with cols[1]:
    if os.path.exists(FINAL_CSV_PATH):
        with open(FINAL_CSV_PATH,"rb") as f:
            st.download_button("הורד Final CSV (סבב קודם)", f, file_name="ab_results_final.csv", mime="text/csv")
    else:
        st.caption("אין Final CSV עדיין.")
with cols[2]:
    if os.path.exists(MERGED_CSV_PATH):
        with open(MERGED_CSV_PATH,"rb") as f:
            st.download_button("הורד Merged CSV", f, file_name="ab_results_merged.csv", mime="text/csv")
    else:
        st.caption("אין Merged CSV עדיין.")

# -------------------------------------------------------------------------------------
# MAIN RUN
# -------------------------------------------------------------------------------------
def build_order_for_batch(done_qids:set, batch:int, seed_val:int)->List[Dict[str,Any]]:
    random.seed(seed_val)
    order = [p for p in PROFILES if p["profile_id"] not in done_qids]
    random.shuffle(order)
    return order[:batch]

def export_dataframe_now(rows:List[Dict[str,Any]], csv_path:str):
    df = pd.DataFrame([flatten_row(r) for r in rows])
    if len(df):
        df.to_csv(csv_path, index=False, encoding="utf-8-sig")
        return df
    return pd.DataFrame()

if run_btn:
    # 1) Existing completed QIDs (from uploaded CSV or from partial on disk)
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

    # אם אין קובץ שהועלה אבל יש partial קיים על הדיסק – נטען ממנו (Resume אוטומטי)
    if not completed_qids and os.path.exists(PARTIAL_CSV_PATH):
        try:
            df_partial = pd.read_csv(PARTIAL_CSV_PATH)
            completed_qids = set(df_partial["QID"].astype(str))
            base_df = df_partial.copy()
            st.info(f"נמצאו על הדיסק {len(completed_qids)} שאלונים – ממשיכים מהם (Partial CSV).")
        except Exception as e:
            st.warning(f"לא הצלחתי לקרוא partial קיים: {e}")

    # 2) Build order for this batch
    order = build_order_for_batch(completed_qids, batch_size, seed)
    total_in_batch = len(order)
    if total_in_batch == 0:
        st.success("אין מה להשלים – כל ה-50 כבר הושלמו בקובץ שהועלה/קיים.")
    else:
        # Widgets for live status
        phase_box = st.empty()
        progress_bar = st.progress(0.0)
        meta_cols = st.columns(3)
        info_box = st.empty()
        t0 = time.perf_counter()

        batch_rows: List[Dict[str,Any]] = []  # נשמור את הסבב הנוכחי
        # 3) Run loop (Gemini -> GPT -> Evaluator), Auto-Save after each questionnaire
        for idx, prof in enumerate(order, start=1):
            qid = prof["profile_id"]
            # Gemini
            phase_box.info(f"🌀 [{idx}/{total_in_batch}] {qid} – Gemini עובד…")
            gem = call_gemini(prof)

            # GPT
            phase_box.info(f"🤖 [{idx}/{total_in_batch}] {qid} – ChatGPT עובד…")
            gpt = call_gpt_user(prof)

            # Evaluator
            phase_box.info(f"⚖️ [{idx}/{total_in_batch}] {qid} – Evaluator מדרג…")
            ev = call_evaluator(prof, gem, gpt)

            # append to rows.json and to in-memory list
            entry = {
                "ts": datetime.now().isoformat(),
                "profile": prof,
                "gemini": gem,
                "gpt": gpt,
                "eval": ev
            }
            append_item(ROWS_PATH, entry)
            batch_rows.append(entry)

            # Auto-save: write partial CSV "live" after each questionnaire
            export_dataframe_now(batch_rows, PARTIAL_CSV_PATH)

            # Live progress + meta
            done = idx
            left = total_in_batch - done
            elapsed = time.perf_counter() - t0
            eta = (elapsed/done)*left if done else 0.0
            with meta_cols[0]:
                st.metric("הושלמו בסבב זה", f"{done}/{total_in_batch}")
            with meta_cols[1]:
                st.metric("נשאר בסבב", f"{left}")
            with meta_cols[2]:
                st.metric("זמן/ETA (דק')", f"{elapsed/60:.1f} / {eta/60:.1f}")
            progress_bar.progress(done/total_in_batch)
            info_box.caption("💾 נשמר קובץ partial אחרי כל שאלון. אם נסגר/נעצר – יש ממה להמשיך.")

        # 4) End of this batch – finalize CSV for this batch
        df_batch = export_dataframe_now(batch_rows, FINAL_CSV_PATH)
        st.success(f"✅ הסבב הושלם: {len(batch_rows)} שאלונים. נוצר Final CSV לסבב זה.")
        if base_df is not None and len(df_batch):
            merged = merge_two_csvs(base_df, df_batch)
            merged.to_csv(MERGED_CSV_PATH, index=False, encoding="utf-8-sig")
            st.info(f"🧩 בוצע Merge עם הקובץ הקודם. סה\"כ כעת: {len(merged)} רשומות ייחודיות.")
        st.dataframe(df_batch, use_container_width=True)

        # Download buttons
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

        # per_car_review – viewer
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
# FOOTER
# -------------------------------------------------------------------------------------
st.caption("© 2025 Car Advisor – Benchmark/Resume/Auto-Save. שומר אחרי כל שאלון, מציג סטטוס חי, ונתמך ב-Resume מלא.")
