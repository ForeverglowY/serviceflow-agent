from sqlalchemy import create_engine, text
from db_models import Base

DATABASE_URL = (
    "postgresql+psycopg://"
    "agent:agent_password@localhost:5432/agent_study"
)

engine = create_engine(DATABASE_URL, echo=True)


def main() -> None:
    Base.metadata.create_all(engine)

    with engine.connect() as connection:
        result = connection.execute(
            text("SELECT current_database(), current_user")
        )

        print(result.one())

if __name__ == "__main__":
    main()
