from flask import Flask, request, jsonify
from flask_cors import CORS
from werkzeug.utils import secure_filename
import os
import tempfile
from extractor import extract_text_from_pdf
from agents.resume_agent import ResumeAnalyzerAgent
from scorer import score_resume_against_job
import json

app = Flask(__name__)
CORS(app)  # Enable CORS for frontend communication

# Configuration
UPLOAD_FOLDER = tempfile.gettempdir()
ALLOWED_EXTENSIONS = {'pdf'}

app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER
app.config['MAX_CONTENT_LENGTH'] = 16 * 1024 * 1024  # 16MB max file size

def allowed_file(filename):
    return '.' in filename and filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS

@app.route('/health', methods=['GET'])
def health_check():
    return jsonify({'status': 'healthy'}), 200

@app.route('/analyze', methods=['POST'])
def analyze_resume():
    try:
        # Check if file is present
        if 'resume' not in request.files:
            return jsonify({'error': 'No resume file provided'}), 400
        
        file = request.files['resume']
        job_description = request.form.get('job_description', '')
        
        # Validate file
        if file.filename == '':
            return jsonify({'error': 'No file selected'}), 400
        
        if not allowed_file(file.filename):
            return jsonify({'error': 'Only PDF files are allowed'}), 400
        
        if not job_description.strip():
            return jsonify({'error': 'Job description is required'}), 400
        
        # Save file temporarily
        filename = secure_filename(file.filename)
        filepath = os.path.join(app.config['UPLOAD_FOLDER'], filename)
        file.save(filepath)
        
        try:
            # Extract text from PDF
            resume_text = extract_text_from_pdf(filepath)
            
            if not resume_text.strip():
                return jsonify({'error': 'Could not extract text from PDF. Please ensure it is not scanned/image-based.'}), 400
            
            # Use deterministic scorer for consistent results
            result = score_resume_against_job(resume_text, job_description)
            
            # Optionally, you can also get LLM analysis (but it may vary)
            # agent = ResumeAnalyzerAgent()
            # llm_result = agent.analyze(resume_text, job_description)
            # result['llm_analysis'] = llm_result
            
            return jsonify(result), 200
            
        finally:
            # Clean up temporary file
            if os.path.exists(filepath):
                os.remove(filepath)
    
    except Exception as e:
        print(f"Error analyzing resume: {str(e)}")
        return jsonify({'error': f'Analysis failed: {str(e)}'}), 500

@app.route('/analyze-with-llm', methods=['POST'])
def analyze_resume_with_llm():
    """Alternative endpoint that uses LLM for qualitative analysis"""
    try:
        if 'resume' not in request.files:
            return jsonify({'error': 'No resume file provided'}), 400
        
        file = request.files['resume']
        job_description = request.form.get('job_description', '')
        
        if file.filename == '' or not allowed_file(file.filename):
            return jsonify({'error': 'Invalid file'}), 400
        
        if not job_description.strip():
            return jsonify({'error': 'Job description is required'}), 400
        
        filename = secure_filename(file.filename)
        filepath = os.path.join(app.config['UPLOAD_FOLDER'], filename)
        file.save(filepath)
        
        try:
            resume_text = extract_text_from_pdf(filepath)
            
            if not resume_text.strip():
                return jsonify({'error': 'Could not extract text from PDF'}), 400
            
            # Get deterministic scoring
            score_result = score_resume_against_job(resume_text, job_description)
            
            # Get LLM analysis
            agent = ResumeAnalyzerAgent()
            llm_analysis = agent.analyze(resume_text, job_description)
            
            # Try to parse LLM output as JSON
            try:
                llm_data = json.loads(llm_analysis)
            except:
                llm_data = {"raw_analysis": llm_analysis}
            
            # Combine results
            result = {
                **score_result,
                'llm_analysis': llm_data
            }
            
            return jsonify(result), 200
            
        finally:
            if os.path.exists(filepath):
                os.remove(filepath)
    
    except Exception as e:
        print(f"Error analyzing resume: {str(e)}")
        return jsonify({'error': f'Analysis failed: {str(e)}'}), 500

if __name__ == '__main__':
    print("Starting ATS Resume Analyzer Server...")
    print("Server running at http://localhost:5000")
    print("API Endpoints:")
    print("  - POST /analyze : Deterministic scoring")
    print("  - POST /analyze-with-llm : Scoring + LLM analysis")
    print("  - GET /health : Health check")
    app.run(debug=True, port=5000)