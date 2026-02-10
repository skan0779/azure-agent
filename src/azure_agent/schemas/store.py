from pydantic import BaseModel
from typing import Optional

class UserProfile(BaseModel):
    name: Optional[str] = None
    age: Optional[int] = None
    language: Optional[str] = "ko"