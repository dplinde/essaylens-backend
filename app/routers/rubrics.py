from fastapi import APIRouter, Depends, HTTPException
from sqlalchemy import or_, select
from sqlalchemy.orm import Session

from ..db import get_db
from ..deps import current_user
from ..models import Rubric, User
from ..schemas import RubricIn, RubricOut
from ..services.grading import DEFAULT_CRITERIA

router = APIRouter(prefix="/rubrics", tags=["rubrics"])


def to_out(r: Rubric) -> RubricOut:
    return RubricOut(rubric_id=r.rubric_id, rubric_name=r.rubric_name, scale_type=r.scale_type, criteria=r.criteria,
                     total_weight=r.total_weight, version=r.version, is_builtin=r.user_id is None, updated_at=r.updated_at)


def ensure_default(db: Session) -> Rubric:
    r = db.scalar(select(Rubric).where(Rubric.user_id.is_(None)))
    if not r:
        r = Rubric(user_id=None, rubric_name="Standard Academic Essay", scale_type="points", criteria=DEFAULT_CRITERIA, total_weight=100)
        db.add(r)
        db.commit()
    return r


@router.get("", response_model=list[RubricOut])
def list_rubrics(user: User = Depends(current_user), db: Session = Depends(get_db)):
    ensure_default(db)
    rows = db.scalars(select(Rubric).where(or_(Rubric.user_id.is_(None), Rubric.user_id == user.user_id)).order_by(Rubric.rubric_id)).all()
    return [to_out(r) for r in rows]


@router.post("", response_model=RubricOut, status_code=201)
def create(body: RubricIn, user: User = Depends(current_user), db: Session = Depends(get_db)):
    r = Rubric(user_id=user.user_id, rubric_name=body.rubric_name, scale_type=body.scale_type,
               criteria=[c.model_dump() for c in body.criteria], total_weight=100)
    db.add(r)
    db.commit()
    return to_out(r)


def _own(db, user, rid) -> Rubric:
    r = db.get(Rubric, rid)
    if not r or r.user_id != user.user_id:
        raise HTTPException(404, "Rubric not found (built-in rubrics are read-only)")
    return r


@router.put("/{rid}", response_model=RubricOut)
def update(rid: int, body: RubricIn, user: User = Depends(current_user), db: Session = Depends(get_db)):
    r = _own(db, user, rid)
    r.rubric_name, r.scale_type = body.rubric_name, body.scale_type
    r.criteria = [c.model_dump() for c in body.criteria]
    r.version += 1
    db.commit()
    return to_out(r)


@router.delete("/{rid}", status_code=204)
def delete(rid: int, user: User = Depends(current_user), db: Session = Depends(get_db)):
    from ..models import Submission

    r = _own(db, user, rid)
    if db.scalar(select(Submission.submission_id).where(Submission.rubric_id == rid).limit(1)):
        raise HTTPException(409, "Rubric is used by existing essays and cannot be deleted")
    db.delete(r)
    db.commit()
