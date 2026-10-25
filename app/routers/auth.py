from datetime import datetime, timezone

from fastapi import APIRouter, Depends, HTTPException
from sqlalchemy import select
from sqlalchemy.orm import Session

from ..db import get_db
from ..deps import current_user
from ..models import User
from ..schemas import ChangePassword, Login, SignUp, TokenOut, UpdateMe, UserOut
from ..security import create_token, hash_password, verify_password

router = APIRouter(prefix="/auth", tags=["auth"])


@router.post("/signup", response_model=TokenOut, status_code=201)
def signup(body: SignUp, db: Session = Depends(get_db)):
    email = body.email.lower()
    if db.scalar(select(User).where(User.email == email)):
        raise HTTPException(409, "An account with this email already exists")
    user = User(full_name=body.full_name.strip(), email=email, password_hash=hash_password(body.password))
    db.add(user)
    db.commit()
    return TokenOut(access_token=create_token(user.user_id), user=UserOut.model_validate(user))


@router.post("/login", response_model=TokenOut)
def login(body: Login, db: Session = Depends(get_db)):
    user = db.scalar(select(User).where(User.email == body.email.lower()))
    if not user or not verify_password(body.password, user.password_hash):
        raise HTTPException(401, "Incorrect email or password")
    user.last_login = datetime.now(timezone.utc)
    db.commit()
    return TokenOut(access_token=create_token(user.user_id), user=UserOut.model_validate(user))


@router.get("/me", response_model=UserOut)
def me(user: User = Depends(current_user)):
    return user


@router.put("/me", response_model=UserOut)
def update_me(body: UpdateMe, user: User = Depends(current_user), db: Session = Depends(get_db)):
    user.full_name = body.full_name.strip()
    db.commit()
    return user


@router.post("/change-password", status_code=204)
def change_password(body: ChangePassword, user: User = Depends(current_user), db: Session = Depends(get_db)):
    if not verify_password(body.current_password, user.password_hash):
        raise HTTPException(400, "Current password is incorrect")
    user.password_hash = hash_password(body.new_password)
    db.commit()


@router.delete("/me", status_code=204)
def delete_account(user: User = Depends(current_user), db: Session = Depends(get_db)):
    """RA 10173: right to erasure — removes the account and all essays, results and rubrics."""
    from ..models import Rubric, Submission

    for s in db.scalars(select(Submission).where(Submission.user_id == user.user_id)).all():
        s.parent_id = None
    db.flush()
    for s in db.scalars(select(Submission).where(Submission.user_id == user.user_id)).all():
        db.delete(s)
    for r in db.scalars(select(Rubric).where(Rubric.user_id == user.user_id)).all():
        db.delete(r)
    db.delete(user)
    db.commit()
