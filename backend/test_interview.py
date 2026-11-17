import requests

print("Fetching questions...")
# Note: Using '+' for URL encoding just in case
get_response = requests.get("http://127.0.0.1:8080/api/interview/questions?role=backend+developer")
raw_data = get_response.json()

print("RAW RESPONSE FROM BACKEND:")
print(raw_data)
print("-" * 50)

# Safely extract the list of questions depending on how Claude structured the JSON
if isinstance(raw_data, dict) and "questions" in raw_data:
    questions = raw_data["questions"]
elif isinstance(raw_data, list):
    questions = raw_data
else:
    print("Error: Unrecognized JSON structure.")
    exit(1)

# Extract IDs (handling if the questions are dictionaries or something else)
try:
    question_ids = [str(q["id"]) for q in questions]
    print(f"Successfully extracted {len(question_ids)} question IDs: {question_ids}")
except TypeError as e:
    print("Failed to extract IDs. The questions list does not contain dictionaries with 'id' keys.")
    exit(1)

# 2. Simulate uploading the same test video 5 times
print("\nUploading 5 videos for evaluation (this will take a minute)...")
url = "http://127.0.0.1:8080/api/interview/submit"

# Adjust 'test_video.mp4' to match your actual file name
files = [('videos', (f'video{i}.mp4', open('test_video.mp4', 'rb'), 'video/mp4')) for i in range(len(question_ids))]
data = {'question_ids': question_ids}

submit_response = requests.post(url, files=files, data=data)
print("Status Code:", submit_response.status_code)
print("Evaluation Report:\n", submit_response.json())