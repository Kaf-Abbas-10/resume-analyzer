"""
scorer.py

Deterministic resume scoring utilities. These functions are intentionally
simple and explainable so you can test logic locally without calling an LLM.

Main functions:
- score_resume_against_job(resume_text, job_description, required_skills=None)
    -> returns dict with match_score (0-100), components, and recommendations.

Notes:
- For best results, pass a domain-specific `required_skills` list.
- This module is purposely dependency-light (only uses stdlib).
"""

import re
import math
from collections import Counter
from difflib import get_close_matches
from typing import List, Dict, Any, Tuple

# ---------- Configuration / Defaults ----------
# Small default skill vocabulary. Extend this list per your domain.
DEFAULT_SKILLS = [
    "python", "java", "c++", "c", "javascript", "nodejs", "react", "angular",
    "fastapi", "django", "flask", "sql", "postgresql", "mysql", "mongodb",
    "docker", "kubernetes", "aws", "azure", "gcp", "rest", "graphql",
    "git", "linux", "tensorflow", "pytorch", "machine learning", "nlp",
    "data analysis", "pandas", "numpy", "multithreading", "parallel computing",
    "openmp", "mpi", "cuda"
]

# Fuzzy matching threshold for skill detection (0..1). Higher = more strict.
FUZZY_MATCH_CUTOFF = 0.8

# Weights for combining component scores into final match_score
WEIGHTS = {
    "skill": 0.55,
    "experience": 0.25,
    "formatting": 0.10,
    "keywords_coverage": 0.10
}


# ---------- Helper utilities ----------
def normalize_text(text: str) -> str:
    """Lowercase and normalize whitespace for simple matching."""
    if not isinstance(text, str):
        return ""
    return re.sub(r"\s+", " ", text.strip().lower())


def tokenize_words(text: str) -> List[str]:
    """Simple word tokenizer removing punctuation."""
    text = normalize_text(text)
    # keep words and dots (for versions like c++ we handle separately)
    words = re.findall(r"[a-z0-9\+\#\.]+", text)
    return words


def find_years_of_experience(resume_text: str) -> float:
    """
    Estimate years of experience from resume by:
    1) searching explicit patterns like "X years", "X+ years"
    2) falling back to earliest/latest year ranges found (e.g., 2016-2019)
    Returns a non-negative float (years).
    """
    t = resume_text

    # 1) explicit "X years" patterns
    pattern1 = re.compile(r"(\d+(?:\.\d+)?)\s*\+\s*years|\b(\d+(?:\.\d+)?)\s+years\b", re.IGNORECASE)
    matches = pattern1.findall(t)
    years = []
    for m in matches:
        # m may be tuple from groups; pick first non-empty
        val = next((g for g in m if g), None)
        try:
            if val:
                years.append(float(val))
        except ValueError:
            pass
    if years:
        # return max mentioned (safer)
        return max(years)

    # 2) detect year ranges like 2016-2019 or '2016 to 2019'
    years_found = re.findall(r"\b(19|20)\d{2}\b", t)
    years_found = [int(y) for y in years_found]
    if years_found:
        # If resume includes many years, try to approximate span
        if len(years_found) >= 2:
            span = max(years_found) - min(years_found)
            # If the span is unrealistic (<1), return 1
            return max(0.0, float(span))
        else:
            # single year present — can't infer experience length confidently
            return 0.0

    return 0.0


# ---------- Skill extraction & matching ----------
def extract_skills(resume_text: str, skill_vocabulary: List[str] = None) -> List[str]:
    """
    Find probable skills in resume_text by:
    - exact matches against vocabulary
    - fuzzy close matches using difflib.get_close_matches
    Returns a list of unique skills (lowercased).
    """
    skill_vocabulary = skill_vocabulary or DEFAULT_SKILLS
    text = normalize_text(resume_text)
    found = set()

    # Exact / substring matches (prefer longer tokens to avoid false positives)
    for skill in sorted(skill_vocabulary, key=lambda s: -len(s)):
        if skill in text:
            found.add(skill)

    # Token-level fuzzy matching: look for close tokens in resume words
    words = set(tokenize_words(text))
    for vocab_skill in skill_vocabulary:
        if vocab_skill in found:
            continue
        # check multi-word skill first
        if " " in vocab_skill:
            # already checked substring; skip
            continue
        # fuzzy match single token skills to words
        candidates = get_close_matches(vocab_skill, words, n=2, cutoff=FUZZY_MATCH_CUTOFF)
        if candidates:
            found.add(vocab_skill)

    return sorted(found)


def parse_required_skills_from_job(job_description: str) -> List[str]:
    """
    Extract probable required skills from the job description by
    matching against DEFAULT_SKILLS vocabulary. If none found, return
    first N nouns as fallback (not implemented here) — we keep it simple.
    """
    jd = normalize_text(job_description)
    required = []
    for skill in DEFAULT_SKILLS:
        if skill in jd:
            required.append(skill)
    # Deduplicate & return
    return sorted(set(required))


def compute_skill_match_score(resume_skills: List[str], required_skills: List[str]) -> Tuple[float, Dict[str, Any]]:
    """
    Returns skill_score (0..1) and a detail dict with matched/missing lists.
    Algorithm:
      - matched = intersection
      - partial credit if resume contains related skills (simple fuzzy)
      - skill_score = matched_count / required_count (clamped)
    If required_skills is empty, we compute a heuristic coverage score from resume_skills.
    """
    resume_set = set(resume_skills)
    required_set = set(required_skills)

    matched = sorted(list(resume_set & required_set))
    missing = sorted(list(required_set - resume_set))

    # Partial credit: if required empty, base on breadth of resume skills
    if not required_skills:
        # cap at 10 skills for scoring
        score = min(len(resume_skills) / 10.0, 1.0)
    else:
        score = len(matched) / max(len(required_skills), 1)

    # also compute related matches (fuzzy: partial overlaps like "sql" vs "postgresql")
    related = []
    for req in missing:
        for rs in resume_skills:
            # simple containment checks or short-token overlap
            if req in rs or rs in req or req.split()[0] == rs.split()[0]:
                related.append((req, rs))
                break

    detail = {
        "required": sorted(list(required_set)),
        "resume_skills": sorted(list(resume_set)),
        "matched": matched,
        "missing": missing,
        "related_matches": related,
        "skill_score_raw": score
    }
    return float(score), detail


# ---------- Formatting / ATS checks ----------
def check_ats_issues(resume_text: str) -> Dict[str, Any]:
    """
    Heuristic checks for common ATS issues.
    Returns a dict:
        - contact_present: bool
        - has_sections: bool (education/experience/skills)
        - bullet_density_ok: bool
        - suspicious_fonts_or_images: False (not implementable from text)
        - issues: list of issue strings
    """
    t = resume_text
    issues = []

    # contact info: look for email or phone
    email_present = bool(re.search(r"[a-z0-9.\-+_]+@[a-z0-9.\-+_]+\.[a-z]+", t, re.IGNORECASE))
    phone_present = bool(re.search(r"\b(\+?\d{1,3}[-.\s]?)?(\d{2,4}[-.\s]?){2,4}\d{2,4}\b", t))
    if not email_present:
        issues.append("Missing email address.")
    if not phone_present:
        issues.append("Missing phone number.")

    # section headings
    has_education = bool(re.search(r"\beducation\b", t))
    has_experience = bool(re.search(r"\bexperience\b", t))
    has_skills = bool(re.search(r"\bskills?\b", t))

    if not (has_education or has_experience or has_skills):
        issues.append("No clear section headings (Education / Experience / Skills).")

    # bullet check: many resumes use '-' or '•' or numbered lists
    bullets = len(re.findall(r"\n\s*[\-\•\*]\s+", resume_text))
    lines = len(re.findall(r"\n", resume_text)) + 1
    bullet_density = bullets / max(lines, 1)

    if bullet_density < 0.03:  # heuristic: fewer than ~3% lines as bullets
        issues.append("Low bullet density — consider using bullet points for achievements.")

    # length checks
    word_count = len(tokenize_words(resume_text))
    if word_count < 150:
        issues.append("Resume looks short (<150 words). Consider adding more detail.")
    if word_count > 3000:
        issues.append("Resume is very long (>3000 words). Consider trimming to 1–2 pages.")

    formatting_score = 1.0
    if issues:
        # scale down formatting score proportional to number of issues (clamped)
        formatting_score = max(0.0, 1.0 - 0.25 * len(issues))

    return {
        "contact_present": email_present and phone_present,
        "has_sections": has_education or has_experience or has_skills,
        "bullet_density": bullet_density,
        "word_count": word_count,
        "issues": issues,
        "formatting_score_raw": formatting_score
    }


# ---------- Keyword coverage ----------
def compute_keyword_coverage(resume_text: str, keywords: List[str]) -> Tuple[float, Dict[str, Any]]:
    """
    Compute how many of the important keywords (from job description or requirements)
    appear in the resume. Returns (coverage 0..1, detail).
    """
    if not keywords:
        return 0.0, {"keywords": [], "matched": [], "coverage": 0.0}

    text = normalize_text(resume_text)
    matched = []
    for kw in keywords:
        if normalize_text(kw) in text:
            matched.append(kw)
    coverage = len(matched) / len(keywords)
    return float(coverage), {"keywords": keywords, "matched": matched, "coverage": coverage}


# ---------- Main scoring function ----------
def score_resume_against_job(
    resume_text: str,
    job_description: str,
    required_skills: List[str] = None
) -> Dict[str, Any]:
    """
    Compose the final scoring result.

    Returns a dictionary with:
    - match_score (0..100)
    - components: skill_score, experience_score, formatting_score, keywords_coverage_score
    - details: skill_detail, experience_years, formatting_detail, keyword_detail
    - recommendations (list)
    """
    if resume_text is None:
        resume_text = ""
    if job_description is None:
        job_description = ""

    resume_text_norm = normalize_text(resume_text)
    job_description_norm = normalize_text(job_description)

    # 1) Skills
    resume_skills = extract_skills(resume_text, DEFAULT_SKILLS)
    required_skills_from_jd = required_skills or parse_required_skills_from_job(job_description)
    skill_score_raw, skill_detail = compute_skill_match_score(resume_skills, required_skills_from_jd)

    # 2) Experience
    years = find_years_of_experience(resume_text)
    # map years to a normalized 0..1 experience score (0-10 years mapped to 0..1)
    experience_score = min(years / 10.0, 1.0)

    # 3) Formatting / ATS issues
    formatting_detail = check_ats_issues(resume_text)
    formatting_score_raw = formatting_detail.get("formatting_score_raw", 0.0)

    # 4) Keyword coverage (use required_skills_from_jd + additional keywords found in JD)
    jd_keywords = required_skills_from_jd.copy()
    # add other tokens from JD that look like keywords (simple heuristic: top nouns/words)
    # For simplicity, take the top N most frequent non-stop tokens from JD
    jd_tokens = [w for w in tokenize_words(job_description_norm) if len(w) > 2]
    freq = Counter(jd_tokens)
    top_tokens = [w for w, _ in freq.most_common(20)]
    # combine and dedupe
    keywords = sorted(set(jd_keywords + top_tokens))
    keyword_coverage_score, keyword_detail = compute_keyword_coverage(resume_text, keywords)

    # Weighted aggregation
    weighted_skill = skill_score_raw * WEIGHTS["skill"]
    weighted_experience = experience_score * WEIGHTS["experience"]
    weighted_formatting = formatting_score_raw * WEIGHTS["formatting"]
    weighted_keywords = keyword_coverage_score * WEIGHTS["keywords_coverage"]

    overall_score = weighted_skill + weighted_experience + weighted_formatting + weighted_keywords
    # map to 0..100 scale
    match_score = round(max(0.0, min(overall_score, 1.0)) * 100)

    # recommendations (simple heuristics)
    recommendations = []
    # skill recommendations
    if required_skills_from_jd:
        missing = skill_detail.get("missing", [])
        if missing:
            recommendations.append(f"Add or highlight these required skills: {', '.join(missing)}.")
    else:
        if len(resume_skills) < 3:
            recommendations.append("Consider adding a dedicated 'Skills' section listing core technologies.")

    # experience recs
    if years < 1:
        recommendations.append("If you have relevant experience (projects, internships), mention durations explicitly (e.g., '6 months').")
    elif years >= 10:
        recommendations.append("Consider summarizing older roles to keep the resume concise.")

    # formatting recs
    if formatting_detail["issues"]:
        recommendations.extend(formatting_detail["issues"])

    # keyword recs
    if keyword_detail["coverage"] < 0.6:
        recommendations.append("Include more job-specific keywords from the job description (tools, frameworks, methodologies).")

    result = {
        "match_score": match_score,
        "components": {
            "skill_score_raw": round(skill_score_raw, 3),
            "experience_years": years,
            "experience_score": round(experience_score, 3),
            "formatting_score_raw": round(formatting_score_raw, 3),
            "keyword_coverage_score": round(keyword_coverage_score, 3),
            "weights_used": WEIGHTS
        },
        "details": {
            "skill_detail": skill_detail,
            "formatting_detail": formatting_detail,
            "keyword_detail": keyword_detail
        },
        "recommendations": recommendations
    }

    return result


# ---------- Example usage (can be removed in production) ----------
if __name__ == "__main__":
    demo_resume = """
    Kaf Abbas
    Email: kaf.abbas@example.com
    Phone: +92 300 1234567

    Summary
    Experienced Python developer with 2+ years building REST APIs using FastAPI and Flask.
    Worked with Docker, PostgreSQL and Git. Familiar with AWS and basic ML using tensorflow.

    Experience
    Software Engineer at ExampleCorp (2021-2023)
    - Built backend services using Python and FastAPI.
    - Containerized apps using Docker and deployed to AWS ECS.

    Education
    BSc Computer Science, FAST University
    """

    demo_jd = """
    We are hiring a Python Backend Developer.
    Required: Python, FastAPI, SQL, Docker, Git. Preferred: AWS, REST APIs.
    """

    out = score_resume_against_job(demo_resume, demo_jd)
    import json
    print(json.dumps(out, indent=2))
