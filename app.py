from extractor import extract_text_from_pdf
from agents.resume_agent import ResumeAnalyzerAgent
from scorer import score_resume_against_job


def run_resume_analysis(resume_path, job_description):
    resume_text = extract_text_from_pdf(resume_path)
    
    agent = ResumeAnalyzerAgent()
    result = agent.analyze(resume_text, job_description)
    
    return result

if __name__ == "__main__":
    job_desc = """
    We are hiring a Python Backend Developer.
    Skills: Python, FastAPI, SQL, REST APIs, Docker, Git.
    Minimum 1+ years experience.
    """

    output = run_resume_analysis("examples/k1.pdf", job_desc)
    print(output)
