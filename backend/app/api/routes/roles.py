from typing import List

from fastapi import APIRouter

from app.services.role_service import get_all_roles

router = APIRouter(prefix="/roles", tags=["Roles"])


@router.get("/", response_model=List[str])
def get_roles() -> List[str]:
    """
    Return a flat list of all available job role names.
    The frontend can directly map over this list to populate a searchable dropdown.

    Example response:
        ["Backend Developer", "Data Scientist", "Frontend Engineer", ...]
    """
    return get_all_roles()