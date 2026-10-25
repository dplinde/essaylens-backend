from datetime import datetime

from pydantic import BaseModel, EmailStr, Field, field_validator, model_validator


class SignUp(BaseModel):
    full_name: str = Field(min_length=2, max_length=120)
    email: EmailStr
    password: str = Field(min_length=8, max_length=128)
    confirm_password: str

    @model_validator(mode="after")
    def match(self):
        if self.password != self.confirm_password:
            raise ValueError("Passwords do not match")
        return self


class Login(BaseModel):
    email: EmailStr
    password: str


class UserOut(BaseModel):
    user_id: int
    full_name: str
    email: EmailStr
    model_config = {"from_attributes": True}


class TokenOut(BaseModel):
    access_token: str
    token_type: str = "bearer"
    user: UserOut


class Criterion(BaseModel):
    name: str = Field(min_length=1, max_length=80)
    description: str = ""
    max_score: float = Field(gt=0, le=1000)
    weight: float = Field(gt=0, le=100)  # percent


class RubricIn(BaseModel):
    rubric_name: str = Field(min_length=1, max_length=120)
    scale_type: str = "points"  # points | likert
    criteria: list[Criterion] = Field(min_length=1, max_length=12)

    @field_validator("scale_type")
    @classmethod
    def _scale(cls, v):
        if v not in ("points", "likert"):
            raise ValueError("scale_type must be 'points' or 'likert'")
        return v

    @model_validator(mode="after")
    def weights_sum(self):
        total = sum(c.weight for c in self.criteria)
        if abs(total - 100) > 0.01:
            raise ValueError(f"Criterion weights must sum to 100 (got {total:g})")
        return self


class RubricOut(BaseModel):
    rubric_id: int
    rubric_name: str
    scale_type: str
    criteria: list[Criterion]
    total_weight: float
    version: int
    is_builtin: bool = False
    updated_at: datetime


class CriterionScore(BaseModel):
    name: str
    score: float
    max_score: float
    weight: float
    comment: str = ""


class ResultOut(BaseModel):
    overall_score: float
    criterion_scores: list[CriterionScore]
    strengths: list[str]
    weaknesses: list[str]
    suggestions: list[str]
    confidence: float | None = None
    graded_by: str
    graded_at: datetime


class SubmissionSummary(BaseModel):
    submission_id: int
    title: str
    essay_topic: str
    input_method: str
    version: int
    parent_id: int | None
    status: str
    is_final: bool
    overall_score: float | None
    submission_date: datetime


class SubmissionDetail(SubmissionSummary):
    transcribed_text: str
    htr_confidence: float | None
    rubric: RubricOut
    result: ResultOut | None


class ReviseIn(BaseModel):
    text: str = Field(min_length=50)
    style: str = "Detailed"


class UpdateMe(BaseModel):
    full_name: str = Field(min_length=2, max_length=120)


class ChangePassword(BaseModel):
    current_password: str
    new_password: str = Field(min_length=8, max_length=128)
