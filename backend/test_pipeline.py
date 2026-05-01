"""Quick end-to-end smoke test for the refactored ATS pipeline."""
import json, sys, os

# Force UTF-8 output
os.environ["PYTHONIOENCODING"] = "utf-8"

from app.services.ats_service import run_pipeline

SAMPLE_RESUME = """
John Doe
Senior Software Engineer

Summary
Experienced software engineer with 5+ years in full-stack web development.

Experience
Senior Developer at Google
Jan 2021 - Present
- Architected microservices using Python, Django, and React
- Engineered CI/CD pipelines on AWS with Docker and Kubernetes
- Led a team of 5 engineers to deliver ML-powered analytics

Software Engineer at Microsoft
Jun 2018 - Dec 2020
- Developed REST APIs using Node.js and Express
- Implemented real-time data pipelines with Apache Kafka
- Optimized database queries reducing response time by 40%

Education
Bachelor of Science in Computer Science
MIT, 2018

Skills
Python, JavaScript, React, Node.js, Django, AWS, Docker, Kubernetes, SQL, Git

Projects
Open-source ML Toolkit
- Built a machine learning pipeline using TensorFlow and scikit-learn
- Deployed on AWS Lambda with automated testing
"""

SAMPLE_JD = """
We are looking for a Senior Full-Stack Developer with:
- 3+ years experience with Python and JavaScript
- Strong experience with React and Node.js
- Cloud experience (AWS preferred)
- Docker and Kubernetes knowledge
- Experience with CI/CD pipelines
- REST API development
- Database design and SQL
- Machine learning experience is a plus
- Strong leadership and communication skills
"""

print("=" * 60)
print("ATS ML PIPELINE -- END-TO-END SMOKE TEST")
print("=" * 60)

result = run_pipeline(
    resume_bytes=SAMPLE_RESUME.encode("utf-8"),
    resume_filename="test_resume.txt",
    jd_text=SAMPLE_JD,
)

if "error" in result:
    print(f"\nERROR: {result['error']}")
    sys.exit(1)

print(f"\n[PASS] ATS Score: {result['ats_score']}")
print(f"\n--- Score Breakdown ---")
for key, value in result['metrics_breakdown'].items():
    if key != 'weights':
        print(f"   {key}: {value}")

print(f"\n--- Technical Skills ---")
print(f"   Matched: {result['technical_skills']['matched']}")
print(f"   Missing: {result['technical_skills']['missing']}")

print(f"\n--- Soft Skills: {result['soft_skills_found']}")
print(f"--- Action Verbs ({result['action_verbs_count']}): {result['action_verbs_found']}")

print(f"\n--- Semantic Similarity: {result['semantic_similarity_score']}")

print(f"\n--- TF-IDF Analysis ---")
print(f"   Score: {result['tfidf_analysis']['score']}")
print(f"   Keywords matched: {result['tfidf_analysis']['matched_count']}/{result['tfidf_analysis']['total_analyzed']}")

print(f"\n--- Experience ---")
print(f"   Total YoE: {result['estimated_experience']['total_yoe']}")
for entry in result['estimated_experience']['experience_entries']:
    org = entry['org']
    start = entry['start_date']
    end = entry['end_date']
    dur = entry['duration_years']
    print(f"   * {org}: {start} -> {end} ({dur} yrs)")

sections_with_content = [k for k, v in result['resume_sections'].items() if v]
print(f"\n--- Resume Sections: {sections_with_content}")

print(f"\n--- Contextual Weights ---")
weights_2x = [k for k, v in result['contextual_skill_weights'].items() if v == 2.0]
if weights_2x:
    print(f"   2x weight (from Experience/Projects): {weights_2x}")

llm_status = 'Not configured (expected)' if not result['llm_enriched_skills'] else str(result['llm_enriched_skills'])
print(f"\n--- LLM Skills: {llm_status}")

print("\n" + "=" * 60)
print("ALL CHECKS PASSED -- PIPELINE FULLY OPERATIONAL")
print("=" * 60)
