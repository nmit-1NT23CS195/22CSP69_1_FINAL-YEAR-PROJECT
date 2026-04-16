from fastapi import APIRouter
from app.services.role_service import get_all_roles

router = APIRouter(prefix="/roles", tags=["Roles"])


@router.get("/")
def get_roles():
    return {
        "roles": get_all_roles()
    }