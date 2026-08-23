import urllib.request, json, time

time.sleep(2)

payload = json.dumps({
    "verified_skills": ["react", "python"],
    "unverified_skills": ["docker", "postgresql", "redis", "celery"],
    "missing_skills": ["kubernetes", "aws", "terraform", "kafka"]
}).encode()

req = urllib.request.Request(
    "http://127.0.0.1:8080/api/mcq/generate",
    data=payload,
    headers={"Content-Type": "application/json"},
    method="POST"
)

try:
    with urllib.request.urlopen(req, timeout=90) as resp:
        body = json.loads(resp.read())
        total = body["total"]
        source = body["source"]
        questions = body["questions"]
        print(f"STATUS: 200 OK")
        print(f"Total questions: {total}")
        print(f"Source: {source}")
        for i, q in enumerate(questions, 1):
            tier = q["forensic_tier"]
            skill = q["skill"]
            question = q["question"][:70]
            print(f"  Q{i} [{tier}] ({skill}): {question}...")
except Exception as e:
    print(f"ERROR: {e}")
