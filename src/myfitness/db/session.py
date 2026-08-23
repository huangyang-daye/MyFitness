from collections.abc import Generator
from contextlib import contextmanager

from sqlalchemy import create_engine
from sqlalchemy.orm import Session, sessionmaker

from myfitness.config import get_settings
from myfitness.db.models import Base, User

_engine = None
_SessionLocal = None


def get_engine():
    global _engine
    if _engine is None:
        settings = get_settings()
        _engine = create_engine(
            settings.database_url,
            pool_pre_ping=True,
            pool_recycle=3600,
            connect_args={"connect_timeout": settings.db_connect_timeout},
        )
    return _engine


def get_session_factory() -> sessionmaker[Session]:
    global _SessionLocal
    if _SessionLocal is None:
        _SessionLocal = sessionmaker(bind=get_engine(), autoflush=False, autocommit=False)
    return _SessionLocal


@contextmanager
def session_scope() -> Generator[Session, None, None]:
    session = get_session_factory()()
    try:
        yield session
        session.commit()
    except Exception:
        session.rollback()
        raise
    finally:
        session.close()


def get_or_create_default_user(session: Session, user_id: int = 1) -> User:
    user = session.get(User, user_id)
    if user is None:
        user = User(id=user_id, name="default", profile={})
        session.add(user)
        session.flush()
    return user


def init_db() -> None:
    """Create tables directly (dev helper). Prefer Alembic migrate in production."""
    Base.metadata.create_all(get_engine())


def ensure_default_user(user_id: int = 1) -> User:
    with session_scope() as session:
        return get_or_create_default_user(session, user_id)
