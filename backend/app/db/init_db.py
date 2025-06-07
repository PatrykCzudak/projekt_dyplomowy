import time
import sqlalchemy
from sqlalchemy.exc import OperationalError
from app.db.database import engine, Base
from app.models import models

def table_exists(table_name):
    inspector = sqlalchemy.inspect(engine)
    return table_name in inspector.get_table_names()

def init():
    retries = 10
    while retries > 0:
        try:
            print("Checking the database connection....")
            if table_exists("assets"):
                print("Removing existing tables...")
                Base.metadata.drop_all(bind=engine)
            print("Creating new tables...")
            Base.metadata.create_all(bind=engine)
            print("The database has been refreshed.")
            break
        except OperationalError as e:
            print(f"The database has not yet started, retrying... ({10 - retries + 1})")
            time.sleep(2)
            retries -= 1
    else:
        print("Failed to connect to the database.")
        raise RuntimeError("Could not connect to database after retries.")

if __name__ == "__main__":
    init()
