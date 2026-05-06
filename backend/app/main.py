from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

from app.api.routes import roles
from app.api.routes import resume, skills, gap, ats

app = FastAPI(
    title="PlaceBuddy ATS System",
    description="Intelligent Applicant Tracking System backend for the PlaceBuddy placement readiness platform.",
    version="1.0.0",
)

# ---------------------------------------------------------------------------
# CORS — allow the React/Vite dev server to call the API from the browser
# ---------------------------------------------------------------------------
app.add_middleware(
    CORSMiddleware,
    allow_origins=[
        "http://localhost:5173",      # Vite default
        "http://127.0.0.1:5173",
        "http://localhost:3000",      # CRA / Next.js default
    ],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# ---------------------------------------------------------------------------
# Routers
# ---------------------------------------------------------------------------
app.include_router(resume.router, prefix="/resume", tags=["Resume"])
app.include_router(skills.router, prefix="/skills", tags=["Skills"])
app.include_router(gap.router, prefix="/gap", tags=["Gap Analysis"])
app.include_router(ats.router, prefix="/ats", tags=["ATS Score"])
app.include_router(roles.router)