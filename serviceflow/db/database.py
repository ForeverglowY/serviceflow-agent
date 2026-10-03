from sqlalchemy import create_engine, text

from serviceflow.db.models import Base
from serviceflow.settings import settings

DATABASE_URL = (
    "postgresql+psycopg://"
    "agent:agent_password@localhost:5432/agent_study"
)

engine = create_engine(
    DATABASE_URL,
    echo=True,
    pool_timeout=settings.database_pool_timeout_seconds,
    connect_args={
        "connect_timeout": (
            settings.database_connect_timeout_seconds
        ),
        "options": (
            "-c statement_timeout="
            f"{settings.database_statement_timeout_ms}"
        ),
    },
)


def main() -> None:
    Base.metadata.create_all(engine)

    with engine.connect() as connection:
        result = connection.execute(
            text("SELECT current_database(), current_user")
        )

        print(result.one())


if __name__ == "__main__":
    main()
