from contextlib import asynccontextmanager

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import RedirectResponse

from . import models  # noqa: F401
from .db import Base, SessionLocal, engine
from .routers import analytics, auth, essays, rubrics


@asynccontextmanager
async def lifespan(app: FastAPI):
    Base.metadata.create_all(engine)
    with SessionLocal() as db:
        rubrics.ensure_default(db)
    yield


app = FastAPI(title="EssayLens API", version="1.0.0", lifespan=lifespan,
              description="Rubric-guided essay grading and feedback. AI-generated grades are provisional.")
app.add_middleware(CORSMiddleware, allow_origins=["*"], allow_methods=["*"], allow_headers=["*"])
for r in (auth.router, rubrics.router, essays.router, analytics.router):
    app.include_router(r)


@app.get("/", include_in_schema=False)
def root():
    return RedirectResponse(url="/docs")


@app.get("/health")
def health():
    return {"status": "ok"}
