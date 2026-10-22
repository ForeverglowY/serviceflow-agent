"""显式初始化空演示库；不会代替旧数据库的ALTER迁移，不删除数据。"""
from sqlalchemy import text

from scripts.seed_data import main as seed_data
from serviceflow.db.database import engine
from serviceflow.db.models import Base


def main() -> None:
    with engine.begin() as connection:
        connection.execute(text("CREATE EXTENSION IF NOT EXISTS vector"))
        Base.metadata.create_all(connection)
    seed_data()
    print("业务表及演示数据初始化结束；旧表结构须事先显式执行迁移。")


if __name__ == "__main__":
    main()
