from datetime import datetime, timezone

from sqlalchemy import JSON, DateTime, Float, ForeignKey, Integer, String, Text
from sqlalchemy.orm import Mapped, mapped_column, relationship

from .db import Base


def now():
    return datetime.now(timezone.utc)


class User(Base):
    __tablename__ = "users"
    user_id: Mapped[int] = mapped_column(Integer, primary_key=True)
    full_name: Mapped[str] = mapped_column(String(120))
    email: Mapped[str] = mapped_column(String(255), unique=True, index=True)
    password_hash: Mapped[str] = mapped_column(String(255))
    created_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), default=now)
    last_login: Mapped[datetime | None] = mapped_column(DateTime(timezone=True), nullable=True)


class Rubric(Base):
    __tablename__ = "rubrics"
    rubric_id: Mapped[int] = mapped_column(Integer, primary_key=True)
    user_id: Mapped[int | None] = mapped_column(ForeignKey("users.user_id"), nullable=True)  # None = built-in
    rubric_name: Mapped[str] = mapped_column(String(120))
    scale_type: Mapped[str] = mapped_column(String(20), default="points")  # points | likert
    criteria: Mapped[list] = mapped_column(JSON)  # [{name, description, max_score, weight}]
    total_weight: Mapped[float] = mapped_column(Float, default=100.0)
    version: Mapped[int] = mapped_column(Integer, default=1)
    created_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), default=now)
    updated_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), default=now, onupdate=now)


class Submission(Base):
    __tablename__ = "submissions"
    submission_id: Mapped[int] = mapped_column(Integer, primary_key=True)
    user_id: Mapped[int] = mapped_column(ForeignKey("users.user_id"), index=True)
    title: Mapped[str] = mapped_column(String(200))
    essay_topic: Mapped[str] = mapped_column(Text)
    input_method: Mapped[str] = mapped_column(String(20))  # handwritten | digital | pasted
    original_file_path: Mapped[str | None] = mapped_column(String(500), nullable=True)
    transcribed_text: Mapped[str] = mapped_column(Text)  # encrypted at rest
    htr_confidence: Mapped[float | None] = mapped_column(Float, nullable=True)
    rubric_id: Mapped[int] = mapped_column(ForeignKey("rubrics.rubric_id"))
    # revision chain
    parent_id: Mapped[int | None] = mapped_column(ForeignKey("submissions.submission_id"), nullable=True)
    version: Mapped[int] = mapped_column(Integer, default=1)
    status: Mapped[str] = mapped_column(String(20), default="graded")  # pending|graded|revised|final
    is_final: Mapped[int] = mapped_column(Integer, default=0)
    submission_date: Mapped[datetime] = mapped_column(DateTime(timezone=True), default=now, index=True)

    result: Mapped["GradingResult"] = relationship(back_populates="submission", uselist=False, cascade="all, delete-orphan")
    rubric: Mapped[Rubric] = relationship()


class GradingResult(Base):
    __tablename__ = "grading_results"
    result_id: Mapped[int] = mapped_column(Integer, primary_key=True)
    submission_id: Mapped[int] = mapped_column(ForeignKey("submissions.submission_id"), unique=True)
    overall_score: Mapped[float] = mapped_column(Float)  # 0-100
    criterion_scores: Mapped[list] = mapped_column(JSON)  # [{name, score, max_score, weight, comment}]
    strengths: Mapped[list] = mapped_column(JSON)
    weaknesses: Mapped[list] = mapped_column(JSON)
    suggestions: Mapped[list] = mapped_column(JSON)
    confidence: Mapped[float | None] = mapped_column(Float, nullable=True)
    graded_by: Mapped[str] = mapped_column(String(60), default="llm")
    graded_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), default=now)

    submission: Mapped[Submission] = relationship(back_populates="result")
