"""Smoke test for the ATS pipeline"""
import requests
import json
import sys
import io

# Create a minimal test resume as a text file
resume_text = """
John Doe
Software Engineer

Summary
Experienced software engineer with 5 years of experience in full-stack development.

Experience
Senior Software Engineer at Google
January 2021 - Present
- Architected and developed microservices using Python, FastAPI, and Docker
- Implemented CI/CD pipelines using Jenkins and Kubernetes
- Optimized database queries in PostgreSQL reducing latency by 40%

Software Developer at Microsoft
June 2019 - December 2020
- Developed REST APIs using Node.js and Express
- Built frontend applications with React and TypeScript
- Managed deployments on AWS using Terraform

Education
Bachelor of Science in Computer Science
Stanford University, 2019

Projects
Open Source Contribution - Built a machine learning pipeline using TensorFlow and Scikit-learn
Personal Blog - Created a full-stack web application using Next.js and MongoDB

Skills
Python, JavaScript, TypeScript, React, Node.js, Docker, Kubernetes, AWS, PostgreSQL, MongoDB, Git
"""

jd_text = """
We are looking for a Senior Software Engineer with experience in:
- Python and FastAPI for backend development
- React and TypeScript for frontend
- Docker and Kubernetes for containerization
- AWS cloud services
- PostgreSQL and MongoDB databases
- CI/CD pipelines
- Microservices architecture
- Machine Learning basics (TensorFlow, scikit-learn)
"""

# Send to the ATS endpoint
url = "http://127.0.0.1:8080/ats/score"

# Create file-like objects
resume_file = ("test_resume.txt", resume_text.encode("utf-8"), "text/plain")

response = requests.post(
    url,
    files={"resume": resume_file},
    data={"jd_text": jd_text}
)

print(f"Status: {response.status_code}")

if response.status_code == 200:
    result = response.json()
    print(json.dumps(result, indent=2))
    
    # Validate key structure
    required_keys = [
        "ats_score", "technical_skills", "soft_skills_found",
        "action_verbs_count", "action_verbs_found", "metrics_breakdown",
        "semantic_similarity_score", "tfidf_analysis",
        "estimated_experience", "resume_sections", "contextual_skill_weights"
    ]
    
    print("\n--- KEY VALIDATION ---")
    for key in required_keys:
        status = "OK" if key in result else "MISSING"
        print(f"  {key}: {status}")
    
    print(f"\nATS Score: {result['ats_score']}")
    print(f"Matched Skills: {result['technical_skills']['matched']}")
    print(f"Missing Skills: {result['technical_skills']['missing']}")
    print(f"Semantic Score: {result.get('semantic_similarity_score', 'N/A')}")
    print(f"YoE: {result.get('estimated_experience', {}).get('total_yoe', 'N/A')}")
else:
    print(f"Error: {response.text}")
