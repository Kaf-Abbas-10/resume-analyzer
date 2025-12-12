import os
from langchain_groq import ChatGroq
from langchain_core.prompts import PromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain_core.runnables import RunnableSequence
from dotenv import load_dotenv

load_dotenv()
api_key = os.getenv("GROQ_API_KEY")
class ResumeAnalyzerAgent:
    def __init__(self):
        self.llm =     llm = ChatGroq(api_key=api_key, model="llama-3.1-8b-instant",
        temperature=0,  # Makes output deterministic
        seed=42)

        self.analysis_prompt = PromptTemplate(
            input_variables=["resume_text", "job_description"],
            template="""
        You are an ATS Resume Analysis Expert.

        Analyze the candidate's resume below:

        RESUME:
        {resume_text}

        JOB DESCRIPTION:
        {job_description}

        Return results in structured JSON with these keys:
        - match_score (0-100)
        - missing_skills
        - strengths
        - weaknesses
        - ats_issues
        - summary
        """
        )
        

        self.chain = (
            self.analysis_prompt | self.llm | StrOutputParser()
        )

    def analyze(self, resume_text: str, job_description: str) -> str:
        return self.chain.invoke({
            "resume_text": resume_text,
            "job_description": job_description
        })
