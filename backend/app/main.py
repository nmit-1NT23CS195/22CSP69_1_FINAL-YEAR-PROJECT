from fastapi import FastAPI
from app.api.routes import roles
from app.api.routes import resume, skills, gap, ats

app = FastAPI(title="ATS System")

app.include_router(resume.router, prefix="/resume", tags=["Resume"])
app.include_router(skills.router, prefix="/skills", tags=["Skills"])
app.include_router(gap.router, prefix="/gap", tags=["Gap Analysis"])
app.include_router(ats.router, prefix="/ats", tags=["ATS Score"])
app.include_router(roles.router)