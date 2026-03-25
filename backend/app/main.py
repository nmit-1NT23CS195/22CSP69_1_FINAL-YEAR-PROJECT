from fastapi import FastAPI

app = FastAPI(title="PlaceBuddy API")

@app.get("/")
def root():
    return {"message": "PlaceBuddy Backend Running 🚀"}