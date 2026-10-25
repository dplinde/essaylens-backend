from collections import Counter, defaultdict

from fastapi import APIRouter, Depends
from sqlalchemy import select
from sqlalchemy.orm import Session

from ..db import get_db
from ..deps import current_user
from ..models import Submission, User

router = APIRouter(prefix="/analytics", tags=["analytics"])


@router.get("/summary")
def summary(user: User = Depends(current_user), db: Session = Depends(get_db)):
    subs = [s for s in db.scalars(select(Submission).where(Submission.user_id == user.user_id).order_by(Submission.submission_date)).all() if s.result]
    scores = [s.result.overall_score for s in subs]
    crit = defaultdict(list)
    for s in subs:
        for c in s.result.criterion_scores:
            crit[c["name"]].append(100 * c["score"] / c["max_score"])
    weak = Counter(w for s in subs for w in s.result.weaknesses)
    sugg = Counter(x for s in subs for x in s.result.suggestions)
    n = len(scores)
    half = max(1, n // 2)
    return {
        "essay_count": n,
        "average_score": round(sum(scores) / n, 2) if n else None,
        "best_score": max(scores) if scores else None,
        "improvement": round(sum(scores[-half:]) / len(scores[-half:]) - sum(scores[:half]) / len(scores[:half]), 1) if n >= 2 else None,
        "trend": [{"submission_id": s.submission_id, "title": s.title, "date": s.submission_date.isoformat(), "score": s.result.overall_score} for s in subs],
        "criterion_averages": {k: round(sum(v) / len(v), 1) for k, v in crit.items()},
        "common_weaknesses": [{"text": t, "count": c} for t, c in weak.most_common(5)],
        "common_suggestions": [{"text": t, "count": c} for t, c in sugg.most_common(5)],
    }
