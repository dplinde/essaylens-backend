import os
import uuid

from fastapi import APIRouter, Depends, File, Form, HTTPException, Query, UploadFile
from fastapi.responses import PlainTextResponse
from sqlalchemy import func, select
from sqlalchemy.orm import Session

from ..config import settings
from ..db import get_db
from ..deps import current_user
from ..models import GradingResult, Rubric, Submission, User
from ..schemas import ReviseIn, SubmissionDetail, SubmissionSummary
from ..security import decrypt_text, encrypt_text
from ..services import extract, grading, htr
from .rubrics import ensure_default, to_out

router = APIRouter(prefix="/essays", tags=["essays"])

MIN_WORDS = 50
IMAGE_EXT = (".jpg", ".jpeg", ".png", ".webp", ".bmp")


def _summary(s: Submission) -> SubmissionSummary:
    return SubmissionSummary(
        submission_id=s.submission_id, title=s.title, essay_topic=s.essay_topic, input_method=s.input_method,
        version=s.version, parent_id=s.parent_id, status=s.status, is_final=bool(s.is_final),
        overall_score=s.result.overall_score if s.result else None, submission_date=s.submission_date)


def _detail(s: Submission) -> SubmissionDetail:
    r = s.result
    return SubmissionDetail(
        **_summary(s).model_dump(), transcribed_text=decrypt_text(s.transcribed_text), htr_confidence=s.htr_confidence,
        rubric=to_out(s.rubric),
        result=None if not r else dict(overall_score=r.overall_score, criterion_scores=r.criterion_scores, strengths=r.strengths,
                                       weaknesses=r.weaknesses, suggestions=r.suggestions, confidence=r.confidence,
                                       graded_by=r.graded_by, graded_at=r.graded_at))


def _owned(db: Session, user: User, sid: int) -> Submission:
    s = db.get(Submission, sid)
    if not s or s.user_id != user.user_id:
        raise HTTPException(404, "Essay not found")
    return s


def _grade_and_store(db, sub: Submission, text: str, style: str = "Detailed"):
    rub = sub.rubric
    res = grading.grade_essay(sub.essay_topic, text, rub.criteria, rub.scale_type, style)
    db.add(GradingResult(submission_id=sub.submission_id, **res))
    sub.status = "graded"
    db.commit()
    db.refresh(sub)


@router.post("", response_model=SubmissionDetail, status_code=201)
async def submit_essay(
    topic: str = Form(..., min_length=3),
    title: str = Form(""),
    rubric_id: int | None = Form(None),
    text: str | None = Form(None),
    style: str = Form("Detailed"),
    file: UploadFile | None = File(None),
    user: User = Depends(current_user),
    db: Session = Depends(get_db),
):
    """Submit an essay as pasted text, a digital file (.docx/.pdf/.txt) or a photo of a handwritten essay."""
    rub = db.get(Rubric, rubric_id) if rubric_id else ensure_default(db)
    if not rub or (rub.user_id not in (None, user.user_id)):
        raise HTTPException(404, "Rubric not found")

    method, conf, path, body = "pasted", None, None, (text or "").strip()
    if file is not None:
        data = await file.read()
        if len(data) > settings.max_upload_mb * 1024 * 1024:
            raise HTTPException(413, f"File exceeds {settings.max_upload_mb} MB")
        name = (file.filename or "").lower()
        os.makedirs(settings.upload_dir, exist_ok=True)
        if name.endswith(IMAGE_EXT):
            method = "handwritten"
            try:
                body, conf, _ = htr.transcribe(data)
            except htr.InvalidSubmission as e:
                raise HTTPException(422, detail={"code": "invalid_submission", "message": str(e)})
            except RuntimeError as e:
                raise HTTPException(503, str(e))
            except ValueError as e:
                raise HTTPException(422, detail={"code": "invalid_submission", "message": str(e)})
        else:
            method = "digital"
            try:
                body = extract.extract_text(name, data)
            except extract.UnsupportedFile as e:
                raise HTTPException(415, str(e))
            except Exception:
                raise HTTPException(422, detail={"code": "invalid_submission", "message": "The file could not be read."})
        path = os.path.join(settings.upload_dir, f"{uuid.uuid4().hex}{os.path.splitext(name)[1]}")
        with open(path, "wb") as f:
            f.write(data)

    if len(body.split()) < MIN_WORDS:
        raise HTTPException(422, detail={"code": "invalid_submission", "message": f"Essay must be at least {MIN_WORDS} words."})

    sub = Submission(user_id=user.user_id, title=title.strip() or topic[:60], essay_topic=topic.strip(), input_method=method,
                     original_file_path=path, transcribed_text=encrypt_text(body), htr_confidence=conf, rubric_id=rub.rubric_id, status="pending")
    db.add(sub)
    db.commit()
    _grade_and_store(db, sub, body, style)
    return _detail(sub)


@router.get("", response_model=list[SubmissionSummary])
def history(q: str | None = None, min_score: float | None = None, max_score: float | None = None,
            latest_only: bool = False, limit: int = Query(50, le=200), offset: int = 0,
            user: User = Depends(current_user), db: Session = Depends(get_db)):
    stmt = select(Submission).where(Submission.user_id == user.user_id)
    if q:
        stmt = stmt.where(Submission.title.ilike(f"%{q}%") | Submission.essay_topic.ilike(f"%{q}%"))
    if latest_only:  # hide superseded versions
        child = select(Submission.parent_id).where(Submission.parent_id.is_not(None))
        stmt = stmt.where(Submission.submission_id.not_in(child))
    rows = db.scalars(stmt.order_by(Submission.submission_date.desc()).limit(limit).offset(offset)).all()
    out = [_summary(s) for s in rows]
    if min_score is not None:
        out = [o for o in out if o.overall_score is not None and o.overall_score >= min_score]
    if max_score is not None:
        out = [o for o in out if o.overall_score is not None and o.overall_score <= max_score]
    return out


@router.get("/{sid}", response_model=SubmissionDetail)
def detail(sid: int, user: User = Depends(current_user), db: Session = Depends(get_db)):
    return _detail(_owned(db, user, sid))


@router.post("/{sid}/revise", response_model=SubmissionDetail, status_code=201)
def revise(sid: int, body: ReviseIn, user: User = Depends(current_user), db: Session = Depends(get_db)):
    """Create a new version of an essay (revised text) and re-grade it."""
    parent = _owned(db, user, sid)
    if len(body.text.split()) < MIN_WORDS:
        raise HTTPException(422, detail={"code": "invalid_submission", "message": f"Essay must be at least {MIN_WORDS} words."})
    top = db.scalar(select(func.max(Submission.version)).where((Submission.submission_id == parent.submission_id) | (Submission.parent_id == parent.submission_id)))
    child = Submission(user_id=user.user_id, title=parent.title, essay_topic=parent.essay_topic, input_method="pasted",
                       transcribed_text=encrypt_text(body.text), rubric_id=parent.rubric_id, parent_id=parent.submission_id,
                       version=(top or parent.version) + 1, status="pending")
    parent.status = "revised"
    db.add(child)
    db.commit()
    _grade_and_store(db, child, body.text, body.style)
    return _detail(child)


@router.post("/{sid}/finalize", response_model=SubmissionSummary)
def finalize(sid: int, user: User = Depends(current_user), db: Session = Depends(get_db)):
    s = _owned(db, user, sid)
    s.is_final, s.status = 1, "final"
    db.commit()
    return _summary(s)


@router.get("/{sid}/compare/{other}")
def compare(sid: int, other: int, user: User = Depends(current_user), db: Session = Depends(get_db)):
    a, b = _owned(db, user, sid), _owned(db, user, other)
    if not (a.result and b.result):
        raise HTTPException(409, "Both essays must be graded")
    names = [c["name"] for c in a.result.criterion_scores]
    bs = {c["name"]: c for c in b.result.criterion_scores}
    rows = [{"criterion": c["name"], "from": c["score"], "to": bs.get(c["name"], {}).get("score"),
             "max": c["max_score"]} for c in a.result.criterion_scores if c["name"] in names]
    return {"from_id": sid, "to_id": other, "overall_from": a.result.overall_score, "overall_to": b.result.overall_score,
            "overall_delta": round(b.result.overall_score - a.result.overall_score, 1), "criteria": rows}


@router.get("/{sid}/export", response_class=PlainTextResponse)
def export(sid: int, user: User = Depends(current_user), db: Session = Depends(get_db)):
    s = _owned(db, user, sid)
    r = s.result
    lines = [f"EssayLens Report — {s.title}", f"Topic: {s.essay_topic}", f"Date: {s.submission_date:%b %d, %Y}  |  Version {s.version}",
             f"Rubric: {s.rubric.rubric_name}", "", f"GRADE: {r.overall_score:g}/100", "", "CRITERION BREAKDOWN:"]
    lines += [f"  {c['name']}: {c['score']:g}/{c['max_score']:g} ({c['weight']:g}%) — {c.get('comment','')}" for c in r.criterion_scores]
    for label, items, mark in (("STRENGTHS", r.strengths, "✓"), ("WEAKNESSES", r.weaknesses, "!"), ("ACTIONABLE FEEDBACK", r.suggestions, "→")):
        lines += ["", label + ":"] + [f"  {mark} {i}" for i in items]
    lines += ["", f"Graded by {r.graded_by}. AI-generated; provisional and for self-assessment."]
    return "\n".join(lines)


@router.delete("/{sid}", status_code=204)
def delete(sid: int, user: User = Depends(current_user), db: Session = Depends(get_db)):
    s = _owned(db, user, sid)
    for c in db.scalars(select(Submission).where(Submission.parent_id == sid)).all():
        c.parent_id = s.parent_id
    db.flush()
    if s.original_file_path and os.path.exists(s.original_file_path):
        os.remove(s.original_file_path)
    db.delete(s)
    db.commit()
