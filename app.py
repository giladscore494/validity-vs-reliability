# -*- coding: utf-8 -*-
# =====================================================================================
# Car Advisor – Batch A/B Harness (Tool vs Chat)
# 50 שאלונים אמיתיים (סימולציה), Gemini (פרומפט מקצועי) מול ChatGPT (פרופיל בלבד)
# פלט: טבלה מלאה + ציון לכל מודל + מנצח לכל שאלון, שמירה ל-JSON ויצוא CSV
# =====================================================================================

import streamlit as st
import os, json, time, random, traceback
from datetime import datetime
from typing import List, Dict, Any, Optional
import pandas as pd

# --- APIs ---
import google.generativeai as genai
from json_repair import repair_json
try:
    from openai import OpenAI
except Exception:
    OpenAI = None

st.set_page_config(page_title="Car Advisor – A/B Reliability Harness", page_icon="🚗", layout="wide")

# ==========================
# CONFIG & SECRETS
# ==========================
GEMINI_MODEL = "gemini-2.5-flash"
OPENAI_MODEL = "gpt-4o-mini"   # עדכן אם תרצה
RUN_DIR = "runs"
os.makedirs(RUN_DIR, exist_ok=True)
RUN_ROWS_FILE = os.path.join(RUN_DIR, "batch_rows.json")
RUN_SUMMARY_FILE = os.path.join(RUN_DIR, "batch_summary.json")

GEMINI_API_KEY = st.secrets.get("GEMINI_API_KEY", "")
OPENAI_API_KEY = st.secrets.get("OPENAI_API_KEY", "")

if not GEMINI_API_KEY:
    st.warning("⚠️ חסר GEMINI_API_KEY ב-secrets.")
if not OPENAI_API_KEY:
    st.warning("⚠️ חסר OPENAI_API_KEY ב-secrets.")

genai.configure(api_key=GEMINI_API_KEY)
gemini_client = genai.GenerativeModel(GEMINI_MODEL)
oa_client = OpenAI(api_key=OPENAI_API_KEY) if (OPENAI_API_KEY and OpenAI) else None

# ==========================
# PROMPTS
# ==========================
# זהה לעקרונות המקצועיים בקוד שלך: החזר JSON מלא בלבד
GEMINI_SYSTEM_PROMPT = """אתה אנליסט מומחה לשוק הרכב הישראלי. קבל פרופיל שאלון של משתמש, נתח לפי שלבים, יישם 18 מקרי קצה, והחזר JSON בלבד בפורמט שנדרש. אל תחזיר טקסט מחוץ ל-JSON."""
GEMINI_LONG_RULES = """
--- שלבי ניתוח (מקוצר לניסוי, תואם עקרונות הכלי) ---
1) פענוח פרטי הצורך מתוך הפרופיל (תקציב, שימוש, דרישות, אילוצים).
2) ניתוח שוק/קטגוריה/מתחרים והמלצה מותאמת.
3) 18 מקרי קצה (כפי שמפורט בקוד המקורי שלך) – יש ליישם.
4) סכם ב-short_verdict בעברית רהוטה.

--- נוסחת ציון (0–100) ---
מחיר מול שוק 25% • תחזוקה/מצב 25% • אמינות דגם 20% • גיל/ק״מ 15% • אמינות מוכר 10% • ביקוש 5%

החזר JSON בלבד:
{
  "from_ad": {"brand":"", "model":"", "year":0, "mileage_km":0, "price_nis":0},
  "deal_score": 0,
  "classification": "",
  "short_verdict": "",
  "key_reasons": [],
  "user_info": {"reliability_summary":"", "maintenance_tips":[], "common_faults":[], "market_context":""},
  "recommendations": [{"car":"", "why":""}]
}
"""

# ChatGPT – משתמש רגיל: מקבל רק הפרופיל + משפט בקשה. בלי כבלים/סכמה. לא תנאי מעבדה.
USER_ONE_LINER = "תמליץ לי 5 רכבים על פי הצרכים שצוינו בפרופיל."

# Evaluator – נותן ציון לכל מודל ובוחר מנצח עם נימוק
EVAL_PROMPT = """
אנא דרג שתי תשובות (Gemini JSON ו-GPT Raw/JSON) עבור אותו פרופיל.
קריטריונים: רלוונטיות לצרכים, עומק והיגיון מקצועי, עקביות, בהירות, תועלת מעשית.
החזר JSON בלבד:
{
  "gemini_score": 0,   // 0-100
  "gpt_score": 0,      // 0-100
  "winner": "Gemini" | "ChatGPT" | "Tie",
  "reason": "נימוק קצר בעברית"
}
"""

# ==========================
# DATA: 50 QUESTIONNAIRES
# ==========================
# "בול כמו שהמשתמש ממלא": פרופיל מלא, ללא מודעה – שימושים/עדיפויות/אילוצים.
# מגוון רחב ע"י רנדום דטרמיניסטי (seed), אבל הפורמט קבוע וברור.
def build_questionnaire(i: int) -> Dict[str, Any]:
    random.seed(1000 + i)
    budget = random.choice([45000, 60000, 75000, 90000, 110000, 130000, 160000, 200000])
    age = random.choice([22, 27, 31, 36, 42, 48])
    annual_km = random.choice([8000, 12000, 15000, 20000, 30000])
    family = random.choice([1, 2, 3, 4, 5])
    primary_use = random.choice([
        "נסיעות יומיומיות בעיר", "נסיעות בין-עירוניות ארוכות", "השכרה קלה למשפחה",
        "טיולים בסופי שבוע", "נסיעות עבודה עם ציוד"
    ])
    gearbox = random.choice(["אוטומט", "ידני", "DCT/DSG", "CVT"])
    engine = random.choice(["בנזין", "דיזל", "היברידי", "פלאג-אין", "חשמלי"])
    body = random.choice(["האצ'בק", "סדאן", "קרוסאובר", "סטיישן", "מיני"])
    pri = random.choice(["אמינות", "חיסכון בדלק", "נוחות", "ביצועים", "שמירת ערך", "בטיחות"])
    secondary = random.sample(["אמינות", "חיסכון בדלק", "נוחות", "ביצועים", "שמירת ערך", "בטיחות"], k=2)
    musts = random.sample(["בקרת שיוט אדפטיבית", "בלימה אוטונומית", "מסך גדול", "תא מטען גדול", "חיישני חניה", "מושב נהג חשמלי", "מערכת בטיחות מתקדמת"], k=3)
    nice = random.sample(["Sunroof", "מערכת שמע פרימיום", "טעינת אלחוט", "עור", "ג'אנטים"], k=2)
    parking = random.choice(["עיר צפופה", "פרבר", "כפרי"])
    region = random.choice(["מרכז", "שפלה", "צפון", "דרום", "ירושלים", "חוף"])
    risk = random.choice(["נמוכה", "בינונית", "גבוהה"])

    return {
        "profile_id": f"Q{i:02d}",
        "budget_nis": budget,
        "driver_age": age,
        "annual_km": annual_km,
        "family_size": family,
        "primary_use": primary_use,
        "preferences": {
            "gearbox": gearbox,
            "engine_type": engine,
            "body_style": body,
            "priority_primary": pri,
            "priority_secondary": secondary,
            "performance_importance": random.randint(1,5),
            "comfort_importance": random.randint(1,5),
            "reliability_importance": random.randint(1,5),
            "safety_importance": random.randint(1,5)
        },
        "must_haves": musts,
        "nice_to_have": nice,
        "parking": parking,
        "region": region,
        "risk_tolerance": risk,
        "notes": random.choice([
            "נסיעות יומיומיות + שני ילדים קטנים", "נהיגה בין-עירונית 60% מהזמן",
            "רכב קודם היה יקר בתחזוקה", "חניה צפופה ליד הבית", "רוצה רכב שומר ערך"
        ])
    }

QUESTIONNAIRES: List[Dict[str, Any]] = [build_questionnaire(i+1) for i in range(50)]

# ==========================
# UTILS
# ==========================
def safe_parse_json(text: Optional[str]) -> Dict[str, Any]:
    if not text:
        return {}
    try:
        fixed = repair_json(text)
        return json.loads(fixed)
    except Exception:
        return {}

def load_list(path: str) -> List[Dict[str, Any]]:
    if os.path.exists(path):
        try:
            with open(path, "r", encoding="utf-8") as f:
                return json.load(f)
        except Exception:
            return []
    return []

def append_row(path: str, row: Dict[str, Any]):
    data = load_list(path)
    data.append(row)
    with open(path, "w", encoding="utf-8") as f:
        json.dump(data, f, ensure_ascii=False, indent=2)

def call_with_retry(fn, retries=3, base_sleep=1.2):
    last_err = None
    for i in range(retries):
        try:
            return fn()
        except Exception as e:
            last_err = e
            time.sleep(base_sleep * (i+1))
    raise last_err

# ==========================
# MODEL CALLS
# ==========================
def run_gemini(profile: Dict[str, Any]) -> Dict[str, Any]:
    def _call():
        resp = gemini_client.generate_content(
            [GEMINI_SYSTEM_PROMPT, f"פרופיל שאלון משתמש (JSON):\n{json.dumps(profile, ensure_ascii=False, indent=2)}\n{GEMINI_LONG_RULES}"],
            request_options={"timeout": 120}
        )
        return safe_parse_json(resp.text)
    try:
        return call_with_retry(_call, retries=3)
    except Exception:
        return {"_error": traceback.format_exc()}

def run_chat_gpt(profile: Dict[str, Any]) -> Dict[str, Any]:
    if oa_client is None:
        return {"_raw": "OpenAI client unavailable.", "_json": {}}
    def _call():
        resp = oa_client.chat.completions.create(
            model=OPENAI_MODEL,
            messages=[
                {"role": "user", "content": json.dumps(profile, ensure_ascii=False)},
                {"role": "user", "content": USER_ONE_LINER}
            ],
            temperature=0.6,   # יותר "משתמש", לא תנאי מעבדה
        )
        text = resp.choices[0].message.content
        parsed = safe_parse_json(text)  # אם יחזיר JSON – נקלוט. אם לא – נשמור raw בלבד.
        return {"_raw": text, "_json": parsed}
    try:
        return call_with_retry(_call, retries=3)
    except Exception:
        return {"_raw": traceback.format_exc(), "_json": {}}

def run_evaluator(profile: Dict[str, Any], gem_json: Dict[str, Any], gpt_pack: Dict[str, Any]) -> Dict[str, Any]:
    if oa_client is None:
        return {"gemini_score": 0, "gpt_score": 0, "winner": "Tie", "reason": "Evaluator unavailable."}
    def _call():
        payload = [
            {"role": "system", "content": "You are a strict evaluator. Hebrew output only, JSON only."},
            {"role": "user", "content": f"PROFILE:\n{json.dumps(profile, ensure_ascii=False, indent=2)}"},
            {"role": "user", "content": f"GEMINI_JSON:\n{json.dumps(gem_json, ensure_ascii=False, indent=2)}"},
            {"role": "user", "content": f"GPT_RAW:\n{gpt_pack.get('_raw','')}"},
            {"role": "user", "content": f"GPT_JSON:\n{json.dumps(gpt_pack.get('_json', {}), ensure_ascii=False, indent=2)}"},
            {"role": "user", "content": EVAL_PROMPT},
        ]
        resp = oa_client.chat.completions.create(
            model=OPENAI_MODEL,
            messages=payload,
            temperature=0.0
        )
        return safe_parse_json(resp.choices[0].message.content)
    try:
        return call_with_retry(_call, retries=2)
    except Exception:
        return {"gemini_score": 0, "gpt_score": 0, "winner": "Tie", "reason": "Evaluation failed."}

# ==========================
# UI
# ==========================
st.title("🚗 Car Advisor – מבחן אימפקט: כלי מקצועי (Gemini) מול צ'אט (ChatGPT)")
st.caption("50 שאלונים ריאליים • Gemini עם פרומפט מקצועי • GPT רק עם פרופיל • טבלת פלט מלאה + ציוני Evaluator")

with st.sidebar:
    st.markdown("### הגדרות")
    n = st.slider("מספר שאלונים להרצה", 10, 50, 50, step=5)
    seed = st.number_input("Seed לרנדום (לסדר ההרצה בלבד)", min_value=0, max_value=999999, value=42, step=1)
    run_btn = st.button("🚀 התחל הרצה רציפה")

# היסטוריה קצרה
prev_rows = load_list(RUN_ROWS_FILE)
st.write(f"📁 רשומות קיימות: {len(prev_rows)}")
if prev_rows:
    st.dataframe(pd.DataFrame([{
        "time": r.get("ts","")[:19].replace("T"," "),
        "qid": r.get("profile",{}).get("profile_id",""),
        "winner": r.get("eval",{}).get("winner",""),
        "gemini_score": r.get("eval",{}).get("gemini_score",""),
        "gpt_score": r.get("eval",{}).get("gpt_score",""),
    } for r in prev_rows][-100:]), use_container_width=True)

status_box = st.empty()
progress = st.progress(0)

# תצוגת תוצאות ריצה זו
run_rows: List[Dict[str, Any]] = []

if run_btn:
    random.seed(seed)
    chosen = QUESTIONNAIRES[:]
    random.shuffle(chosen)
    chosen = chosen[:n]

    for idx, prof in enumerate(chosen, start=1):
        with st.spinner(f"🔄 מריץ שאלון {idx}/{n} – Gemini…"):
            status_box.info(f"⏳ סטטוס: שאלון {idx}/{n} • כלי: Gemini • QID={prof['profile_id']}")
            gem_json = run_gemini(prof)

        with st.spinner(f"🔄 מריץ שאלון {idx}/{n} – ChatGPT…"):
            status_box.info(f"⏳ סטטוס: שאלון {idx}/{n} • כלי: ChatGPT • QID={prof['profile_id']}")
            gpt_pack = run_chat_gpt(prof)

        with st.spinner(f"🔍 מעריך תוצאות לשאלון {idx}/{n}…"):
            status_box.info(f"⏳ סטטוס: הערכה • QID={prof['profile_id']}")
            eval_obj = run_evaluator(prof, gem_json, gpt_pack)

        row = {
            "ts": datetime.now().isoformat(),
            "profile": prof,
            "gemini_json": gem_json,       # כל מה שגימיני כתב – נכנס ישר לטבלה (JSON מלא)
            "gpt_raw": gpt_pack.get("_raw",""),   # מה שצ'אט כתב – טקסט גולמי
            "gpt_json": gpt_pack.get("_json",{}), # ואם היה JSON – נכניס גם אותו
            "eval": eval_obj
        }
        run_rows.append(row)
        append_row(RUN_ROWS_FILE, row)

        progress.progress(idx / n)

    # סיכום ריצה
    summary = {
        "ts": datetime.now().isoformat(),
        "n": n,
        "winners_count": {
            "Gemini": sum(1 for r in run_rows if r.get("eval",{}).get("winner") == "Gemini"),
            "ChatGPT": sum(1 for r in run_rows if r.get("eval",{}).get("winner") == "ChatGPT"),
            "Tie": sum(1 for r in run_rows if r.get("eval",{}).get("winner") == "Tie"),
        },
        "avg_scores": {
            "gemini": round(sum(r.get("eval",{}).get("gemini_score",0) for r in run_rows)/len(run_rows), 2),
            "gpt": round(sum(r.get("eval",{}).get("gpt_score",0) for r in run_rows)/len(run_rows), 2),
        }
    }
    append_row(RUN_SUMMARY_FILE, summary)
    status_box.success("✅ הרצה הושלמה. הטבלה המלאה מוכנה למטה.")

# ==========================
# TABLE RENDER (מוגמר)
# ==========================
st.divider()
st.subheader("📊 טבלת פלט – כל המידע והציונים")

def flatten_for_table(r: Dict[str, Any]) -> Dict[str, Any]:
    prof = r.get("profile", {})
    gem = r.get("gemini_json", {})
    gpt_raw = (r.get("gpt_raw") or "").strip()
    gpt_j = r.get("gpt_json", {})
    ev = r.get("eval", {})

    # מקבץ תמצית המלצות (אם קיימות)
    def recs_from_gem(x):
        out = []
        for it in (x.get("recommendations") or []):
            car = it.get("car","")
            why = it.get("why","")
            out.append(f"{car} – {why}")
        return " | ".join(out) if out else ""

    def recs_from_gpt_json(x):
        # נסה לקרוא מבנה נפוץ: רשימת רכבים עם "why"
        if not x:
            return ""
        candidates = []
        # חפש מפתחות אפשריים
        if isinstance(x, dict):
            for k, v in x.items():
                if isinstance(v, list):
                    for it in v:
                        if isinstance(it, dict):
                            car = it.get("car") or it.get("name") or it.get("model") or ""
                            why = it.get("why") or it.get("reason") or ""
                            if car or why:
                                candidates.append(f"{car} – {why}".strip(" –"))
        return " | ".join(candidates)

    return {
        "time": r.get("ts","")[:19].replace("T"," "),
        "QID": prof.get("profile_id",""),
        # תקציר שאלון
        "תקציב": prof.get("budget_nis",""),
        "שימוש עיקרי": prof.get("primary_use",""),
        "עדיפות ראשית": prof.get("preferences",{}).get("priority_primary",""),
        "דלק": prof.get("preferences",{}).get("engine_type",""),
        "תיבה": prof.get("preferences",{}).get("gearbox",""),
        "משפחה": prof.get("family_size",""),
        # פלט Gemini (כל ה-JSON נשמר בקובץ; בטבלה – תמצית קריאה)
        "Gemini: ציון": gem.get("deal_score",""),
        "Gemini: סיווג": gem.get("classification",""),
        "Gemini: פסקה": gem.get("short_verdict",""),
        "Gemini: המלצות": recs_from_gem(gem),
        # פלט GPT
        "GPT: טקסט גולמי": gpt_raw[:500] + ("..." if len(gpt_raw) > 500 else ""),
        "GPT: המלצות (אם JSON)": recs_from_gpt_json(gpt_j),
        # הערכה
        "Evaluator: Gemini score": ev.get("gemini_score",""),
        "Evaluator: GPT score": ev.get("gpt_score",""),
        "Evaluator: Winner": ev.get("winner",""),
        "Evaluator: Reason": ev.get("reason",""),
    }

# בנה טבלת ריצה אחרונה אם יש, אחרת מהקובץ
display_rows = run_rows if run_rows else prev_rows
if display_rows:
    table = pd.DataFrame([flatten_for_table(r) for r in display_rows])
    st.dataframe(table, use_container_width=True)
    # יצוא
    csv_bytes = table.to_csv(index=False).encode("utf-8-sig")
    st.download_button("⬇️ הורד CSV של התוצאות", csv_bytes, file_name="car_advisor_ab_results.csv", mime="text/csv")
else:
    st.info("אין נתונים להצגה עדיין. לחץ על 'התחל הרצה רציפה'.")


st.caption("© 2025 Car Advisor – A/B Reliability Harness. כולל ספינר סטטוס, Retry לכשלים, ושמירה לוקאלית של כל הפלטים.")
