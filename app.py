import re
import datetime
import html
from pathlib import Path
from typing import Dict, List, Tuple, Optional, Any

import joblib
import streamlit as st
from pypdf import PdfReader
from pypdf.errors import PdfReadError


# ──────────────────────────────────────────────
# PAGE CONFIG
# ──────────────────────────────────────────────
st.set_page_config(
    page_title="ResumeAI",
    page_icon="📄",
    layout="wide",
    initial_sidebar_state="expanded",
)


# ──────────────────────────────────────────────
# PATHS & MODEL
# ──────────────────────────────────────────────
BASE_DIR = Path(__file__).resolve().parent
MODEL_PATH = BASE_DIR / "resume_jd_model.pkl"
VECTORIZER_PATH = BASE_DIR / "resume_jd_vectorizer.pkl"


@st.cache_resource
def load_model() -> Tuple[Any, Any, Optional[str]]:
    try:
        if not MODEL_PATH.exists() or not VECTORIZER_PATH.exists():
            return None, None, "Model files not found."
        model = joblib.load(MODEL_PATH)
        vectorizer = joblib.load(VECTORIZER_PATH)
        return model, vectorizer, None
    except Exception as e:
        return None, None, str(e)


model, vectorizer, model_error = load_model()


# ──────────────────────────────────────────────
# DATA
# ──────────────────────────────────────────────
TECHNICAL_SKILLS = {
    "python", "java", "javascript", "typescript", "c++", "c#", "go", "rust",
    "html", "css", "sass", "sql", "react", "react.js", "react native", "next.js",
    "angular", "vue", "vue.js", "tailwind", "bootstrap", "redux", "jquery",
    "node.js", "express", "express.js", "flask", "django", "fastapi",
    "graphql", "rest api", "api", "mysql", "postgresql", "sqlite", "mongodb",
    "redis", "supabase", "firebase", "git", "github", "gitlab", "ci/cd",
    "docker", "kubernetes", "linux", "aws", "azure", "gcp", "numpy", "pandas",
    "scikit-learn", "matplotlib", "seaborn", "machine learning", "deep learning",
    "artificial intelligence", "natural language processing", "nlp",
    "computer vision", "cnn", "ann", "rnn", "tensorflow", "pytorch", "keras",
    "generative ai", "rag", "llm", "langchain", "streamlit", "gradio",
    "figma", "postman", "jira", "agile", "scrum", "unit testing", "jest", "pytest",
}

SOFT_SKILLS = {
    "communication", "leadership", "teamwork", "problem solving", "problem-solving",
    "organization", "time management", "adaptability", "creativity", "collaboration",
    "critical thinking", "presentation", "management", "attention to detail",
    "project management", "decision making", "interpersonal skills", "mentoring",
    "conflict resolution",
}

SECTION_PATTERNS = {
    "summary": ["summary", "professional summary", "profile", "objective", "career objective"],
    "education": ["education", "academic background", "academic qualifications"],
    "experience": ["experience", "work experience", "professional experience", "employment history"],
    "projects": ["projects", "academic projects", "personal projects", "project experience"],
    "skills": ["skills", "technical skills", "core skills", "competencies"],
    "certifications": ["certifications", "certificates", "licenses"],
    "volunteer": ["volunteer", "volunteering", "community involvement"],
    "achievements": ["achievements", "awards", "honors", "accomplishments"],
}

ACTION_VERBS = {
    "developed", "built", "created", "designed", "implemented", "integrated",
    "optimized", "managed", "led", "automated", "improved", "deployed",
    "configured", "analyzed", "tested", "delivered", "engineered", "maintained",
    "launched", "streamlined", "collaborated", "reduced", "increased", "achieved",
    "spearheaded", "architected", "refactored", "migrated", "scaled",
}

WEAK_PHRASES = {
    "hard working", "hardworking", "responsible for", "worked on", "helped with",
    "good communication", "team player", "quick learner", "passionate about",
    "looking for an opportunity", "seeking an opportunity", "duties included",
    "familiar with", "exposure to", "results-driven", "self-motivated",
}

SKILL_ALIASES = {
    "react.js": "react", "reactjs": "react", "node": "node.js", "nodejs": "node.js",
    "express.js": "express", "vue.js": "vue", "nextjs": "next.js",
    "ml": "machine learning", "dl": "deep learning", "ai": "artificial intelligence",
    "nlp": "natural language processing", "js": "javascript", "ts": "typescript",
    "postgres": "postgresql", "scikit learn": "scikit-learn", "rest": "rest api",
    "restful api": "rest api", "tailwind css": "tailwind", "tailwindcss": "tailwind",
    "generative artificial intelligence": "generative ai", "gen ai": "generative ai",
    "large language model": "llm", "large language models": "llm",
}


# ──────────────────────────────────────────────
# CSS
# ──────────────────────────────────────────────
st.markdown(
    """
    <style>
    @import url('https://fonts.googleapis.com/css2?family=Inter:wght@400;500;600;700;800&display=swap');
    * { font-family: 'Inter', sans-serif; }
    .stApp { background: #f6f8fb; color: #111827; }
    #MainMenu, footer { visibility: hidden; }
    header { background: transparent !important; box-shadow: none !important; }
    [data-testid="stToolbar"] { visibility: hidden; }
    [data-testid="stDecoration"] { display: none; }
    [data-testid="stAppViewContainer"] { background: #f6f8fb; }
    .block-container { max-width: 1320px; padding: 2.4rem 2.5rem 4rem 2.5rem; }
    [data-testid="stSidebar"] { background: #111827; border-right: 1px solid #1f2937; }
    [data-testid="stSidebar"] > div:first-child { padding: 1.7rem 1.15rem; }
    [data-testid="stSidebar"] * { color: #f9fafb; }
    .sidebar-brand { font-size: 24px; font-weight: 800; letter-spacing: -0.7px; margin-bottom: 3px; }
    .sidebar-subtitle { color: #9ca3af !important; font-size: 12px; line-height: 1.5; margin-bottom: 24px; }
    .sidebar-label { color: #9ca3af !important; font-size: 10px; font-weight: 700; text-transform: uppercase; letter-spacing: 1px; margin: 20px 0 8px 2px; }
    .model-status { border: 1px solid #374151; background: #1f2937; border-radius: 9px; padding: 9px 11px; margin-top: 16px; font-size: 11px; color: #d1d5db !important; }
    .status-dot { display: inline-block; width: 7px; height: 7px; border-radius: 50%; background: #10b981; margin-right: 7px; }
    .sidebar-note { color: #9ca3af !important; font-size: 11px; line-height: 1.55; margin-top: 20px; }
    .hero { position: relative; overflow: hidden; background: #ffffff; border: 1px solid #e5e7eb; border-radius: 18px; padding: 30px 32px; margin-bottom: 24px; box-shadow: 0 6px 20px rgba(17, 24, 39, 0.04); }
    .hero::after { content: ""; position: absolute; width: 170px; height: 170px; border-radius: 50%; background: #eff6ff; right: -55px; top: -70px; z-index: 0; }
    .hero-content { position: relative; z-index: 1; }
    .hero-kicker { color: #1a56db; font-size: 11px; font-weight: 800; letter-spacing: 1.2px; text-transform: uppercase; margin-bottom: 7px; }
    .hero-title { color: #111827; font-size: 32px; font-weight: 800; letter-spacing: -1.2px; margin: 0 0 8px 0; }
    .hero-text { color: #6b7280; font-size: 13px; line-height: 1.65; max-width: 720px; margin: 0; }
    .section-heading { color: #111827; font-size: 19px; font-weight: 800; letter-spacing: -0.4px; margin: 25px 0 12px 0; }
    .section-card { background: #ffffff; border: 1px solid #e5e7eb; border-radius: 14px; padding: 19px; box-shadow: 0 4px 16px rgba(17, 24, 39, 0.035); height: 100%; }
    .section-card-title { font-size: 13px; font-weight: 800; color: #111827; margin-bottom: 13px; }
    .score-card { background: #ffffff; border: 1px solid #e5e7eb; border-radius: 13px; padding: 17px 15px; text-align: center; min-height: 116px; box-shadow: 0 3px 12px rgba(17, 24, 39, 0.03); }
    .score-value { font-size: 28px; font-weight: 800; letter-spacing: -1px; line-height: 1.1; margin-bottom: 7px; }
    .score-label { color: #6b7280; font-size: 10px; font-weight: 700; text-transform: uppercase; letter-spacing: 0.6px; }
    .metric-card { background: #ffffff; border: 1px solid #e5e7eb; border-radius: 13px; padding: 17px; min-height: 91px; }
    .metric-value { color: #111827; font-size: 23px; font-weight: 800; margin-bottom: 4px; }
    .metric-label { color: #6b7280; font-size: 11px; font-weight: 600; }
    .tag-wrap { display: flex; flex-wrap: wrap; gap: 7px; }
    .tag { display: inline-flex; align-items: center; background: #f3f4f6; color: #374151; border: 1px solid #e5e7eb; border-radius: 7px; padding: 5px 9px; font-size: 11px; font-weight: 600; }
    .tag.match { background: #ecfdf5; color: #047857; border-color: #a7f3d0; }
    .tag.gap { background: #fef2f2; color: #b91c1c; border-color: #fecaca; }
    .list-item { display: flex; gap: 10px; align-items: flex-start; padding: 9px 0; border-bottom: 1px solid #f3f4f6; color: #374151; font-size: 12px; line-height: 1.55; }
    .list-item:last-child { border-bottom: none; }
    .positive-item::before { content: "✓"; color: #059669; font-weight: 800; flex-shrink: 0; }
    .negative-item::before { content: "!"; color: #dc2626; font-weight: 800; flex-shrink: 0; }
    .priority-item { border-left: 3px solid #1a56db; background: #f8fbff; border-radius: 0 9px 9px 0; padding: 11px 13px; margin-bottom: 8px; color: #374151; font-size: 12px; line-height: 1.5; }
    .empty-state { background: #f9fafb; border: 1px dashed #d1d5db; border-radius: 10px; padding: 15px; color: #6b7280; font-size: 12px; text-align: center; }
    .small-note { color: #6b7280; font-size: 10px; line-height: 1.55; margin-top: 8px; }
    .result-banner { background: #111827; border-radius: 14px; padding: 21px; color: #ffffff; margin-bottom: 16px; }
    .result-banner-title { font-size: 16px; font-weight: 800; margin-bottom: 5px; }
    .result-banner-text { color: #d1d5db; font-size: 11px; line-height: 1.55; }
    .result-number { font-size: 34px; font-weight: 800; line-height: 1; }
    .result-caption { color: #9ca3af; font-size: 10px; margin-top: 5px; }
    .stButton > button { width: 100%; border-radius: 9px; border: 1px solid #111827; background: #111827; color: #ffffff; font-weight: 700; font-size: 12px; min-height: 42px; }
    .stButton > button:hover { background: #1f2937; border-color: #1f2937; color: #ffffff; }
    .stDownloadButton > button { width: 100%; border-radius: 9px; background: #ffffff; border: 1px solid #d1d5db; color: #111827; font-weight: 700; font-size: 12px; min-height: 42px; }
    .stDownloadButton > button:hover { border-color: #9ca3af; color: #111827; }
/* ========== CLEAN FILE UPLOADER (Best version) ========== */
[data-testid="stFileUploader"] {
    background: transparent !important;
    padding: 0 !important;
}

[data-testid="stFileUploaderDropzone"] {
    border: 1.5px dashed #cbd5e1 !important;
    border-radius: 12px !important;
    background: #fafbfc !important;
    padding: 28px 20px !important;
    transition: all 0.2s ease;
    display: flex !important;
    flex-direction: column !important;
    align-items: center !important;
    justify-content: center !important;
    min-height: 110px !important;
}

[data-testid="stFileUploaderDropzone"]:hover {
    border-color: #94a3b8 !important;
    background: #f1f5f9 !important;
}

/* Hide the default long instruction text */
[data-testid="stFileUploaderDropzone"] > div > div > span,
[data-testid="stFileUploaderDropzone"] small,
[data-testid="stFileUploaderDropzoneInstructions"],
[data-testid="stFileUploaderDropzone"] p {
    display: none !important;
}

/* Style the Browse button nicely + center it */
[data-testid="stFileUploader"] button {
    background: #111827 !important;
    color: white !important;
    border: none !important;
    border-radius: 8px !important;
    font-weight: 600 !important;
    font-size: 13px !important;
    padding: 10px 24px !important;
    min-height: 40px !important;
    margin: 0 auto !important;
}

[data-testid="stFileUploader"] button:hover {
    background: #1f2937 !important;
    color: white !important;
}
    textarea { border-radius: 11px !important; border: 1px solid #d1d5db !important; background: #ffffff !important; color: #111827 !important; }
    textarea:focus { border-color: #1a56db !important; box-shadow: 0 0 0 1px #1a56db !important; }
    .stProgress > div > div { background: #1a56db; }
    [data-testid="stMetric"] { background: #ffffff; border: 1px solid #e5e7eb; border-radius: 12px; padding: 12px; }
    @media (max-width: 900px) { .block-container { padding: 1.5rem 1.2rem 3rem 1.2rem; } .hero { padding: 24px; } .hero-title { font-size: 27px; } }
    @media (max-width: 640px) { .hero-title { font-size: 24px; } .hero { padding: 20px; } }
    </style>
    """,
    unsafe_allow_html=True,
)


# ──────────────────────────────────────────────
# HELPERS
# ──────────────────────────────────────────────
def normalize(text: str) -> str:
    return re.sub(r"\s+", " ", text.lower()).strip()


def compile_keyword_pattern(keyword: str) -> re.Pattern:
    escaped = re.escape(keyword.lower()).replace(r"\ ", r"\s+")
    return re.compile(rf"(?<!\w){escaped}(?!\w)", re.IGNORECASE)


TECHNICAL_PATTERNS = {s: compile_keyword_pattern(s) for s in TECHNICAL_SKILLS}
SOFT_PATTERNS = {s: compile_keyword_pattern(s) for s in SOFT_SKILLS}
ACTION_PATTERNS = {v: compile_keyword_pattern(v) for v in ACTION_VERBS}


def normalize_skill(skill: str) -> str:
    return SKILL_ALIASES.get(normalize(skill), normalize(skill))


def extract_pdf_text(uploaded_file) -> Tuple[str, str]:
    try:
        uploaded_file.seek(0)
        reader = PdfReader(uploaded_file)

        if reader.is_encrypted:
            try:
                result = reader.decrypt("")
                if result == 0:
                    return "", "This PDF is password protected."
            except Exception:
                return "", "This PDF is password protected."

        pages = []
        for page in reader.pages:
            try:
                pages.append(page.extract_text() or "")
            except Exception:
                continue

        text = "\n".join(pages).strip()
        if not text:
            return "", "No readable text was found. This may be a scanned PDF."
        return text, ""
    except PdfReadError:
        return "", "The PDF could not be read. It may be corrupted."
    except Exception as e:
        return "", f"Could not process the PDF: {e}"


def extract_text_file(uploaded_file) -> Tuple[str, str]:
    try:
        uploaded_file.seek(0)
        content = uploaded_file.read()
        try:
            text = content.decode("utf-8")
        except UnicodeDecodeError:
            text = content.decode("latin-1", errors="ignore")
        text = text.strip()
        if not text:
            return "", "The text file is empty."
        return text, ""
    except Exception as e:
        return "", f"Could not process the text file: {e}"


def detect_contact(text: str) -> Dict[str, bool]:
    normalized = normalize(text)
    return {
        "Email": bool(re.search(r"\b[A-Z0-9._%+-]+@[A-Z0-9.-]+\.[A-Z]{2,}\b", text, re.I)),
        "Phone": bool(re.search(r"(?<!\d)(?:\+?\d[\d\s().-]{7,}\d)(?!\d)", text)),
        "LinkedIn": "linkedin.com" in normalized,
        "GitHub": "github.com" in normalized,
    }


def detect_sections(text: str) -> List[str]:
    normalized = normalize(text)
    found = []
    for section, patterns in SECTION_PATTERNS.items():
        for p in patterns:
            if re.search(rf"(?<!\w){re.escape(p)}(?!\w)", normalized):
                found.append(section)
                break
    return found


def detect_skills(text: str) -> Tuple[List[str], List[str]]:
    normalized = normalize(text)
    technical = sorted({normalize_skill(s) for s, pat in TECHNICAL_PATTERNS.items() if pat.search(normalized)})
    soft = sorted({normalize_skill(s) for s, pat in SOFT_PATTERNS.items() if pat.search(normalized)})
    return technical, soft


def detect_action_verbs(text: str) -> List[str]:
    normalized = normalize(text)
    return sorted({v for v, pat in ACTION_PATTERNS.items() if pat.search(normalized)})


def detect_weak_phrases(text: str) -> List[str]:
    normalized = normalize(text)
    return sorted({p for p in WEAK_PHRASES if p in normalized})


def detect_metrics(text: str) -> List[str]:
    patterns = [
        r"\b\d+(?:\.\d+)?%",
        r"\$\s?\d+(?:,\d{3})*(?:\.\d+)?",
        r"\b\d+(?:,\d{3})+\b",
        r"\b\d+(?:\.\d+)?\s?(?:users|clients|customers|projects|students|members|records|requests|downloads|hours|days|months|years)\b",
    ]
    found = []
    for p in patterns:
        found.extend(re.findall(p, text, re.I))
    return sorted(set(found))


def get_bullets(text: str) -> List[str]:
    bullets = []
    for line in text.splitlines():
        stripped = line.strip()
        if re.match(r"^[-•▪◦*]\s+", stripped) or re.match(r"^\d+[.)]\s+", stripped):
            bullets.append(stripped)
    return bullets


def get_score_color(score: float) -> str:
    if score >= 75:
        return "#059669"
    if score >= 50:
        return "#d97706"
    return "#dc2626"


# ──────────────────────────────────────────────
# SCORING
# ──────────────────────────────────────────────
def calculate_scores(text: str) -> Dict[str, Any]:
    words = len(re.findall(r"\b[\w+#.-]+\b", text))
    sections = detect_sections(text)
    contact = detect_contact(text)
    technical, soft = detect_skills(text)
    action_verbs = detect_action_verbs(text)
    weak_phrases = detect_weak_phrases(text)
    metrics = detect_metrics(text)
    bullets = get_bullets(text)

    section_score = min(100, round((len(sections) / 8) * 100 * 1.3))
    contact_score = round((sum(contact.values()) / max(len(contact), 1)) * 100)

    technical_count = min(len(technical), 12)
    soft_count = min(len(soft), 4)
    skill_score = round((technical_count / 12) * 75 + (soft_count / 4) * 25)

    action_score = min(100, len(action_verbs) * 10)
    evidence_score = min(100, len(metrics) * 18 + min(len(bullets), 6) * 7)

    content_score = round((section_score + skill_score + action_score + evidence_score) / 4)
    ats_score = round(section_score * 0.35 + contact_score * 0.20 + skill_score * 0.25 + evidence_score * 0.20)

    if words == 0:
        word_score = 0
    elif words < 250:
        word_score = 65
    elif words < 350:
        word_score = 80
    elif words > 1100:
        word_score = 75
    else:
        word_score = 100

    overall = round(ats_score * 0.35 + content_score * 0.35 + skill_score * 0.15 + evidence_score * 0.15)

    return {
        "overall": overall,
        "ats": ats_score,
        "content": content_score,
        "skills": skill_score,
        "evidence": evidence_score,
        "structure": section_score,
        "contact": contact_score,
        "action": action_score,
        "word_score": word_score,
        "words": words,
        "sections": sections,
        "contact_info": contact,
        "technical": technical,
        "soft": soft,
        "action_verbs": action_verbs,
        "weak_phrases": weak_phrases,
        "metrics": metrics,
        "bullets": bullets,
    }


def get_strengths(data: Dict) -> List[str]:
    strengths = []
    if data["structure"] >= 75:
        strengths.append("Your resume contains a strong set of standard sections.")
    if data["skills"] >= 70:
        strengths.append("A good range of technical and professional skills is detected.")
    if data["evidence"] >= 65:
        strengths.append("Your resume includes measurable evidence and bullet-based content.")
    if data["action"] >= 60:
        strengths.append("Strong action-oriented language is present.")
    if data["contact"] >= 75:
        strengths.append("Most important contact details are available.")
    if 350 <= data["words"] <= 1100:
        strengths.append("The resume has a reasonable amount of written content.")
    if not strengths:
        strengths.append("Your resume has a usable foundation that can be improved further.")
    return strengths


def get_issues(data: Dict) -> List[str]:
    issues = []
    if data["structure"] < 75:
        issues.append("Some standard resume sections may be missing.")
    if data["skills"] < 70:
        issues.append("The detected skill coverage could be stronger.")
    if data["evidence"] < 65:
        issues.append("More measurable results and specific achievements would strengthen the resume.")
    if data["action"] < 60:
        issues.append("More strong action verbs could improve bullet quality.")
    if data["contact"] < 75:
        issues.append("Some contact information appears to be missing.")
    if data["words"] < 250:
        issues.append("The resume may be too short to communicate enough relevant detail.")
    if data["words"] > 1100:
        issues.append("The resume may contain more content than necessary.")
    if data["weak_phrases"]:
        issues.append("Generic phrases were detected and could be replaced with specific evidence.")
    if not issues:
        issues.append("No major structural issues were detected.")
    return issues


def get_priorities(data: Dict) -> List[str]:
    priorities = []
    if data["skills"] < 70:
        priorities.append("Strengthen the skills section with technologies directly relevant to your target roles.")
    if data["evidence"] < 65:
        priorities.append("Add numbers, percentages, users, performance improvements, or other measurable outcomes.")
    if data["action"] < 60:
        priorities.append("Rewrite experience and project bullets with stronger action verbs.")
    if data["structure"] < 75:
        priorities.append("Add or improve standard resume sections such as Experience, Projects, Skills, and Education.")
    if data["contact"] < 75:
        priorities.append("Complete your professional contact information.")
    if data["weak_phrases"]:
        priorities.append("Replace generic phrases with concrete actions and outcomes.")
    if data["words"] > 1100:
        priorities.append("Remove repetitive or low-value content to improve focus.")
    if data["words"] < 250:
        priorities.append("Add relevant project, experience, or achievement details.")
    if not priorities:
        priorities.append("Keep refining the resume around the specific role you are targeting.")
    return priorities[:5]


def executive_summary(data: Dict) -> str:
    overall = data["overall"]
    if overall >= 75:
        opening = "Your resume has a strong overall structure and useful professional signals."
    elif overall >= 50:
        opening = "Your resume has a solid foundation with several areas that can be strengthened."
    else:
        opening = "Your resume has a basic foundation, but several areas need improvement."

    details = []
    details.append("skill coverage is good" if data["skills"] >= 70 else "skill coverage can be improved")
    details.append("measurable evidence is present" if data["evidence"] >= 65 else "more measurable evidence is needed")
    details.append("action-oriented language is being used" if data["action"] >= 60 else "stronger action-oriented language would help")
    return opening + " " + ", ".join(details) + "."


def build_report_text(data: Dict, source_name: str) -> str:
    lines = [
        "RESUMEAI — RESUME ANALYSIS REPORT",
        "=" * 42,
        f"Source: {source_name}",
        f"Generated: {datetime.datetime.now().strftime('%Y-%m-%d %H:%M')}",
        "",
        "SCORE OVERVIEW",
        "-" * 42,
        f"Overall Score: {data['overall']}/100",
        f"ATS Readiness: {data['ats']}/100",
        f"Content Quality: {data['content']}/100",
        f"Skills: {data['skills']}/100",
        f"Evidence: {data['evidence']}/100",
        f"Structure: {data['structure']}/100",
        f"Contact: {data['contact']}/100",
        f"Action Language: {data['action']}/100",
        "",
        "EXECUTIVE SUMMARY",
        "-" * 42,
        executive_summary(data),
        "",
        "RESUME METRICS",
        "-" * 42,
        f"Word Count: {data['words']}",
        f"Bullet Points: {len(data['bullets'])}",
        f"Sections Found: {len(data['sections'])}",
        f"Measurable Results: {len(data['metrics'])}",
        "",
        "CONTACT INFORMATION",
        "-" * 42,
    ]
    for k, v in data["contact_info"].items():
        lines.append(f"{k}: {'Found' if v else 'Missing'}")
    lines += [
        "",
        "TECHNICAL SKILLS",
        "-" * 42,
        ", ".join(data["technical"]) if data["technical"] else "None detected",
        "",
        "PROFESSIONAL SKILLS",
        "-" * 42,
        ", ".join(data["soft"]) if data["soft"] else "None detected",
        "",
        "STRENGTHS",
        "-" * 42,
    ]
    lines += [f"- {i}" for i in get_strengths(data)]
    lines += ["", "AREAS TO IMPROVE", "-" * 42]
    lines += [f"- {i}" for i in get_issues(data)]
    lines += ["", "PRIORITIES", "-" * 42]
    lines += [f"- {i}" for i in get_priorities(data)]
    if data["action_verbs"]:
        lines += ["", "ACTION VERBS", "-" * 42, ", ".join(data["action_verbs"])]
    if data["weak_phrases"]:
        lines += ["", "GENERIC PHRASES", "-" * 42, ", ".join(data["weak_phrases"])]
    lines.append("\nThis report is decision support, not a hiring decision.")
    return "\n".join(lines)


# ──────────────────────────────────────────────
# JOB MATCH
# ──────────────────────────────────────────────
def extract_job_skills(job_description: str) -> List[str]:
    technical, soft = detect_skills(job_description)
    return sorted(set(technical + soft))


def calculate_skill_gap(resume_text: str, job_description: str) -> Tuple[List[str], List[str]]:
    resume_skills = set(extract_job_skills(resume_text))
    job_skills = set(extract_job_skills(job_description))
    matching = sorted(job_skills.intersection(resume_skills))
    missing = sorted(job_skills.difference(resume_skills))
    return matching, missing


def predict_match(resume_text: str, job_description: str, model, vectorizer) -> Tuple[Optional[str], Optional[float]]:
    try:
        combined = resume_text + "\n" + job_description
        transformed = vectorizer.transform([combined])
        prediction = model.predict(transformed)[0]
        confidence = 0.0
        if hasattr(model, "predict_proba"):
            confidence = float(max(model.predict_proba(transformed)[0])) * 100
        return str(prediction), confidence
    except Exception:
        return None, None


# ──────────────────────────────────────────────
# RENDER HELPERS
# ──────────────────────────────────────────────
def render_score_card(label: str, score: float):
    color = get_score_color(score)
    st.markdown(
        f"""
        <div class="score-card">
            <div class="score-value" style="color:{color};">{score}</div>
            <div class="score-label">{html.escape(label)}</div>
        </div>
        """,
        unsafe_allow_html=True,
    )


def render_tags(items: List[str], tag_class: str = ""):
    if not items:
        st.markdown('<div class="empty-state">Nothing detected yet.</div>', unsafe_allow_html=True)
        return
    tags = "".join(f'<span class="tag {tag_class}">{html.escape(item)}</span>' for item in items)
    st.markdown(f'<div class="tag-wrap">{tags}</div>', unsafe_allow_html=True)


def render_list(items: List[str], positive: bool = True):
    if not items:
        st.markdown('<div class="empty-state">Nothing detected.</div>', unsafe_allow_html=True)
        return
    class_name = "positive-item" if positive else "negative-item"
    content = "".join(f'<div class="list-item {class_name}">{html.escape(item)}</div>' for item in items)
    st.markdown(content, unsafe_allow_html=True)


def render_priorities(items: List[str]):
    content = "".join(
        f'<div class="priority-item"><strong>{i}.</strong> {html.escape(item)}</div>'
        for i, item in enumerate(items, 1)
    )
    st.markdown(content, unsafe_allow_html=True)


# ──────────────────────────────────────────────
# SESSION STATE
# ──────────────────────────────────────────────
if "resume_analysis" not in st.session_state:
    st.session_state.resume_analysis = None
if "resume_text" not in st.session_state:
    st.session_state.resume_text = ""
if "resume_source_name" not in st.session_state:
    st.session_state.resume_source_name = ""
if "job_match" not in st.session_state:
    st.session_state.job_match = None


# ──────────────────────────────────────────────
# SIDEBAR
# ──────────────────────────────────────────────
with st.sidebar:
    st.markdown('<div class="sidebar-brand">📄 ResumeAI</div>', unsafe_allow_html=True)
    st.markdown(
        '<div class="sidebar-subtitle">Resume analysis and job matching in one place.</div>',
        unsafe_allow_html=True,
    )
    st.markdown('<div class="sidebar-label">Workspace</div>', unsafe_allow_html=True)

    nav_page = st.radio(
        "Navigation",
        ["Analyze Resume", "Job Match", "About"],
        label_visibility="collapsed",
    )

    st.markdown('<div class="sidebar-label">Status</div>', unsafe_allow_html=True)
    if model is not None and vectorizer is not None:
        st.markdown(
            '<div class="model-status"><span class="status-dot"></span>Matching system ready</div>',
            unsafe_allow_html=True,
        )
    else:
        st.markdown(
            '<div class="model-status">Matching system unavailable</div>',
            unsafe_allow_html=True,
        )

    st.markdown(
        '<div class="sidebar-note">'
        "Your resume is processed within the current session. "
        "Resume content is not stored by this interface."
        "</div>",
        unsafe_allow_html=True,
    )


# ──────────────────────────────────────────────
# PAGE: ANALYZE RESUME
# ──────────────────────────────────────────────
if nav_page == "Analyze Resume":
    st.markdown(
        """
        <div class="hero">
            <div class="hero-content">
                <div class="hero-kicker">Resume Analysis</div>
                <div class="hero-title">Analyze your resume</div>
                <p class="hero-text">
                    Review your resume for structure, skills, contact details,
                    action language, measurable evidence, and overall readiness.
                </p>
            </div>
        </div>
        """,
        unsafe_allow_html=True,
    )

    st.markdown('<div class="section-heading">Upload your resume</div>', unsafe_allow_html=True)

    input_col1, input_col2 = st.columns(2, gap="large")

    with input_col1:
        st.markdown('<div class="section-card">', unsafe_allow_html=True)
        st.markdown('<div class="section-card-title">Upload PDF or TXT</div>', unsafe_allow_html=True)

        uploaded_file = st.file_uploader(
            " ",
            type=["pdf", "txt"],
            label_visibility="collapsed",
            help=None,
            key="resume_uploader"
        )

        st.markdown('<div class="small-note">PDF and TXT files are supported.</div>', unsafe_allow_html=True)
        st.markdown("</div>", unsafe_allow_html=True)

    with input_col2:
        st.markdown('<div class="section-card">', unsafe_allow_html=True)
        st.markdown('<div class="section-card-title">Or paste resume text</div>', unsafe_allow_html=True)
        pasted_text = st.text_area(
            "Resume text",
            height=190,
            placeholder="Paste your resume content here...",
            label_visibility="collapsed"
        )
        st.markdown("</div>", unsafe_allow_html=True)

    st.write("")
    analyze_col, status_col = st.columns([1, 2])

    with analyze_col:
        analyze_clicked = st.button("Analyze Resume", use_container_width=True)

    with status_col:
        if st.session_state.resume_source_name:
            st.markdown(
                f'<div class="small-note" style="padding-top:11px;">'
                f'Current resume: <strong>{html.escape(st.session_state.resume_source_name)}</strong></div>',
                unsafe_allow_html=True,
            )

    if analyze_clicked:
        resume_text = ""
        source_name = ""

        if uploaded_file is not None:
            source_name = uploaded_file.name
            if uploaded_file.type == "application/pdf" or uploaded_file.name.lower().endswith(".pdf"):
                resume_text, error = extract_pdf_text(uploaded_file)
            else:
                resume_text, error = extract_text_file(uploaded_file)
            if error:
                st.error(error)
                resume_text = ""
        elif pasted_text.strip():
            resume_text = pasted_text.strip()
            source_name = "Pasted resume text"
        else:
            st.warning("Upload a resume or paste your resume text first.")

        if resume_text and len(resume_text) >= 30:
            with st.spinner("Analyzing resume..."):
                analysis = calculate_scores(resume_text)
            st.session_state.resume_analysis = analysis
            st.session_state.resume_text = resume_text
            st.session_state.resume_source_name = source_name
            st.session_state.job_match = None
            st.success("Resume analyzed successfully.")
        elif resume_text:
            st.error("The resume text is too short to analyze meaningfully.")

    if st.session_state.resume_analysis:
        data = st.session_state.resume_analysis

        st.markdown('<div class="section-heading">Score Overview</div>', unsafe_allow_html=True)

        score_row_1 = st.columns(4, gap="medium")
        with score_row_1[0]:
            render_score_card("Overall", data["overall"])
        with score_row_1[1]:
            render_score_card("ATS Readiness", data["ats"])
        with score_row_1[2]:
            render_score_card("Content Quality", data["content"])
        with score_row_1[3]:
            render_score_card("Skills", data["skills"])

        score_row_2 = st.columns(4, gap="medium")
        with score_row_2[0]:
            render_score_card("Evidence", data["evidence"])
        with score_row_2[1]:
            render_score_card("Structure", data["structure"])
        with score_row_2[2]:
            render_score_card("Contact", data["contact"])
        with score_row_2[3]:
            render_score_card("Action Language", data["action"])

        st.markdown('<div class="section-heading">Executive Summary</div>', unsafe_allow_html=True)
        st.markdown(
            f'<div class="section-card"><div style="color:#374151;font-size:13px;line-height:1.7;">{html.escape(executive_summary(data))}</div></div>',
            unsafe_allow_html=True,
        )

        st.markdown('<div class="section-heading">Resume Metrics</div>', unsafe_allow_html=True)
        metric_cols = st.columns(4, gap="medium")
        metrics_data = [
            ("Word Count", data["words"]),
            ("Bullet Points", len(data["bullets"])),
            ("Sections Found", len(data["sections"])),
            ("Measurable Results", len(data["metrics"])),
        ]
        for col, (label, value) in zip(metric_cols, metrics_data):
            with col:
                st.markdown(
                    f'<div class="metric-card"><div class="metric-value">{value}</div><div class="metric-label">{html.escape(label)}</div></div>',
                    unsafe_allow_html=True,
                )

        st.markdown('<div class="section-heading">Contact Information</div>', unsafe_allow_html=True)
        contact_cols = st.columns(4, gap="medium")
        for col, (key, value) in zip(contact_cols, data["contact_info"].items()):
            with col:
                color = "#059669" if value else "#dc2626"
                status = "Found" if value else "Missing"
                st.markdown(
                    f'<div class="metric-card"><div class="metric-value" style="color:{color};font-size:17px;">{status}</div><div class="metric-label">{html.escape(key)}</div></div>',
                    unsafe_allow_html=True,
                )

        skill_col1, skill_col2 = st.columns(2, gap="large")
        with skill_col1:
            st.markdown('<div class="section-heading">Technical Skills</div>', unsafe_allow_html=True)
            st.markdown('<div class="section-card">', unsafe_allow_html=True)
            render_tags(data["technical"])
            st.markdown("</div>", unsafe_allow_html=True)
        with skill_col2:
            st.markdown('<div class="section-heading">Professional Skills</div>', unsafe_allow_html=True)
            st.markdown('<div class="section-card">', unsafe_allow_html=True)
            render_tags(data["soft"])
            st.markdown("</div>", unsafe_allow_html=True)

        st.markdown('<div class="section-heading">Strengths & Areas to Improve</div>', unsafe_allow_html=True)
        strength_col, issue_col = st.columns(2, gap="large")
        with strength_col:
            st.markdown('<div class="section-card">', unsafe_allow_html=True)
            st.markdown('<div class="section-card-title">Strengths</div>', unsafe_allow_html=True)
            render_list(get_strengths(data), positive=True)
            st.markdown("</div>", unsafe_allow_html=True)
        with issue_col:
            st.markdown('<div class="section-card">', unsafe_allow_html=True)
            st.markdown('<div class="section-card-title">Areas to Improve</div>', unsafe_allow_html=True)
            render_list(get_issues(data), positive=False)
            st.markdown("</div>", unsafe_allow_html=True)

        st.markdown('<div class="section-heading">Priority Improvements</div>', unsafe_allow_html=True)
        st.markdown('<div class="section-card">', unsafe_allow_html=True)
        render_priorities(get_priorities(data))
        st.markdown("</div>", unsafe_allow_html=True)

        if data["action_verbs"] or data["weak_phrases"]:
            action_col, phrase_col = st.columns(2, gap="large")
            with action_col:
                if data["action_verbs"]:
                    st.markdown('<div class="section-heading">Action Verbs</div>', unsafe_allow_html=True)
                    st.markdown('<div class="section-card">', unsafe_allow_html=True)
                    render_tags(data["action_verbs"], "match")
                    st.markdown("</div>", unsafe_allow_html=True)
            with phrase_col:
                if data["weak_phrases"]:
                    st.markdown('<div class="section-heading">Generic Phrases</div>', unsafe_allow_html=True)
                    st.markdown('<div class="section-card">', unsafe_allow_html=True)
                    render_tags(data["weak_phrases"], "gap")
                    st.markdown("</div>", unsafe_allow_html=True)

        st.markdown('<div class="section-heading">Export</div>', unsafe_allow_html=True)
        report_text = build_report_text(data, st.session_state.resume_source_name)
        st.download_button(
            "Download Analysis Report",
            data=report_text,
            file_name="resumeai_analysis_report.txt",
            mime="text/plain",
            use_container_width=True,
        )


# ──────────────────────────────────────────────
# PAGE: JOB MATCH
# ──────────────────────────────────────────────
elif nav_page == "Job Match":
    st.markdown(
        """
        <div class="hero">
            <div class="hero-content">
                <div class="hero-kicker">Job Match</div>
                <div class="hero-title">Match against a job description</div>
                <p class="hero-text">
                    Compare your resume with a target role to identify
                    matching skills, missing requirements, and areas to improve.
                </p>
            </div>
        </div>
        """,
        unsafe_allow_html=True,
    )

    if not st.session_state.resume_text:
        st.markdown(
            """
            <div class="empty-state">
                Analyze your resume first, then return here to compare it
                with a job description.
            </div>
            """,
            unsafe_allow_html=True,
        )
    else:
        st.markdown('<div class="section-heading">Job Description</div>', unsafe_allow_html=True)
        st.markdown('<div class="section-card">', unsafe_allow_html=True)
        job_description = st.text_area(
            "Paste job description",
            height=270,
            placeholder="Paste the complete job description here...",
            label_visibility="collapsed",
        )
        st.markdown("</div>", unsafe_allow_html=True)

        st.write("")
        compare_clicked = st.button("Compare Resume", use_container_width=True)

        if compare_clicked:
            if not job_description.strip():
                st.warning("Paste a job description first.")
            else:
                with st.spinner("Comparing your resume with the role..."):
                    matching_skills, missing_skills = calculate_skill_gap(
                        st.session_state.resume_text, job_description
                    )
                    job_skills = extract_job_skills(job_description)
                    coverage = round((len(matching_skills) / len(job_skills)) * 100) if job_skills else 0

                    prediction = confidence = None
                    if model is not None and vectorizer is not None:
                        prediction, confidence = predict_match(
                            st.session_state.resume_text, job_description, model, vectorizer
                        )

                    st.session_state.job_match = {
                        "matching_skills": matching_skills,
                        "missing_skills": missing_skills,
                        "job_skills": job_skills,
                        "coverage": coverage,
                        "prediction": prediction,
                        "confidence": confidence,
                    }

        if st.session_state.job_match:
            match_data = st.session_state.job_match

            st.markdown('<div class="section-heading">Match Overview</div>', unsafe_allow_html=True)
            overview_col1, overview_col2, overview_col3 = st.columns(3, gap="medium")

            with overview_col1:
                st.markdown(
                    f'<div class="metric-card"><div class="metric-value" style="color:{get_score_color(match_data["coverage"])};">{match_data["coverage"]}%</div><div class="metric-label">Skill Coverage</div></div>',
                    unsafe_allow_html=True,
                )
            with overview_col2:
                st.markdown(
                    f'<div class="metric-card"><div class="metric-value" style="color:#059669;">{len(match_data["matching_skills"])}</div><div class="metric-label">Matching Skills</div></div>',
                    unsafe_allow_html=True,
                )
            with overview_col3:
                st.markdown(
                    f'<div class="metric-card"><div class="metric-value" style="color:#dc2626;">{len(match_data["missing_skills"])}</div><div class="metric-label">Skills Not Detected</div></div>',
                    unsafe_allow_html=True,
                )

            if match_data["prediction"] is not None:
                st.markdown('<div class="section-heading">Match Signal</div>', unsafe_allow_html=True)
                pred_text = normalize(str(match_data["prediction"]))
                positive = pred_text in {"match", "matched", "1", "true", "yes"} or "match" in pred_text
                title = "Your resume shows a strong match signal." if positive else "Your resume shows a lower match signal."

                confidence_html = ""
                if match_data["confidence"] is not None:
                    confidence_html = f"""
                        <div style="margin-top:14px;">
                            <div class="result-number">{round(match_data['confidence'])}%</div>
                            <div class="result-caption">Signal confidence</div>
                        </div>
                    """

                st.markdown(
                    f"""
                    <div class="result-banner">
                        <div class="result-banner-title">{html.escape(title)}</div>
                        <div class="result-banner-text">
                            This signal is based on the information provided in your resume and the job description.
                        </div>
                        {confidence_html}
                    </div>
                    """,
                    unsafe_allow_html=True,
                )

            skills_col1, skills_col2 = st.columns(2, gap="large")
            with skills_col1:
                st.markdown('<div class="section-heading">Matching Skills</div>', unsafe_allow_html=True)
                st.markdown('<div class="section-card">', unsafe_allow_html=True)
                render_tags(match_data["matching_skills"], "match")
                st.markdown("</div>", unsafe_allow_html=True)
            with skills_col2:
                st.markdown('<div class="section-heading">Skills to Consider</div>', unsafe_allow_html=True)
                st.markdown('<div class="section-card">', unsafe_allow_html=True)
                render_tags(match_data["missing_skills"], "gap")
                st.markdown("</div>", unsafe_allow_html=True)

            st.markdown('<div class="section-heading">Role Requirements Detected</div>', unsafe_allow_html=True)
            st.markdown('<div class="section-card">', unsafe_allow_html=True)
            render_tags(match_data["job_skills"])
            st.markdown("</div>", unsafe_allow_html=True)

            st.markdown('<div class="section-heading">What This Means</div>', unsafe_allow_html=True)
            if match_data["coverage"] >= 75:
                interpretation = "A large portion of the detected role-related skills are already present in your resume."
            elif match_data["coverage"] >= 50:
                interpretation = "Your resume covers several relevant requirements, but some role-related skills are not currently detected."
            else:
                interpretation = "The detected skill overlap is limited. Review the role requirements and make sure genuinely relevant experience and skills are represented in your resume."

            st.markdown(
                f'<div class="section-card"><div style="color:#374151;font-size:13px;line-height:1.7;">{html.escape(interpretation)}</div></div>',
                unsafe_allow_html=True,
            )

            st.markdown('<div class="section-heading">Improvement Plan</div>', unsafe_allow_html=True)
            improvement_items = []
            if match_data["missing_skills"]:
                improvement_items.append(
                    "Review the skills not detected above and add them only if you genuinely have the relevant knowledge or experience."
                )
            if match_data["matching_skills"]:
                improvement_items.append(
                    "Make your strongest matching skills visible in relevant project and experience bullets."
                )
            improvement_items.append(
                "Tailor your resume wording to the actual terminology used in the target job description."
            )
            improvement_items.append(
                "Support important skills with concrete projects, responsibilities, or measurable results."
            )

            st.markdown('<div class="section-card">', unsafe_allow_html=True)
            render_priorities(improvement_items)
            st.markdown("</div>", unsafe_allow_html=True)


# ──────────────────────────────────────────────
# PAGE: ABOUT
# ──────────────────────────────────────────────
elif nav_page == "About":
    st.markdown(
        """
        <div class="hero">
            <div class="hero-content">
                <div class="hero-kicker">About ResumeAI</div>
                <div class="hero-title">Understand your resume better.</div>
                <p class="hero-text">
                    ResumeAI helps you review your resume structure, skills,
                    evidence, and alignment with a target job description.
                </p>
            </div>
        </div>
        """,
        unsafe_allow_html=True,
    )

    about_col1, about_col2 = st.columns(2, gap="large")
    with about_col1:
        st.markdown(
            """
            <div class="section-card">
                <div class="section-card-title">Resume Analysis</div>
                <div style="color:#4b5563;font-size:12px;line-height:1.7;">
                    Review key resume signals including structure,
                    contact information, technical skills, professional
                    skills, action language, measurable evidence, and
                    content quality.
                </div>
            </div>
            """,
            unsafe_allow_html=True,
        )
    with about_col2:
        st.markdown(
            """
            <div class="section-card">
                <div class="section-card-title">Job Matching</div>
                <div style="color:#4b5563;font-size:12px;line-height:1.7;">
                    Compare your resume with a target job description
                    to identify relevant skill overlap and requirements
                    that may need more attention.
                </div>
            </div>
            """,
            unsafe_allow_html=True,
        )

    st.markdown('<div class="section-heading">How it works</div>', unsafe_allow_html=True)
    steps = [
        ("01", "Upload", "Upload a PDF/TXT resume or paste your resume text."),
        ("02", "Analyze", "ResumeAI checks important resume content and signals."),
        ("03", "Compare", "Add a job description to review role-specific alignment."),
        ("04", "Improve", "Use the findings to refine your resume for the target role."),
    ]
    step_cols = st.columns(4, gap="medium")
    for col, (number, title, description) in zip(step_cols, steps):
        with col:
            st.markdown(
                f"""
                <div class="section-card">
                    <div style="color:#1a56db;font-size:11px;font-weight:800;margin-bottom:9px;">{number}</div>
                    <div class="section-card-title">{title}</div>
                    <div style="color:#6b7280;font-size:11px;line-height:1.6;">{description}</div>
                </div>
                """,
                unsafe_allow_html=True,
            )

    st.markdown(
        '<div class="small-note" style="margin-top:22px;">'
        "ResumeAI provides analysis and decision support. Results should be "
        "reviewed alongside the actual job requirements and your own experience."
        "</div>",
        unsafe_allow_html=True,
    )