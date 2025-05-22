import time
import sqlalchemy
from sqlalchemy.exc import OperationalError
from app.db.database import engine, Base
from app.models import models  # <-- Ważne, żeby modele zostały zaimportowane

def table_exists(table_name):
    inspector = sqlalchemy.inspect(engine)
    return table_name in inspector.get_table_names()

def init():
    retries = 10
    while retries > 0:
        try:
            print("Sprawdzam połączenie z bazą danych...")
            # próba inspekcji (czy baza gotowa)
            if table_exists("assets"):
                print("Usuwam istniejące tabele...")
                Base.metadata.drop_all(bind=engine)
            print("Tworzę nowe tabele...")
            Base.metadata.create_all(bind=engine)
            print("Baza danych została odświeżona.")
            break
        except OperationalError as e:
            print(f"Baza jeszcze się nie uruchomiła, ponawiam... ({10 - retries + 1})")
            time.sleep(2)
            retries -= 1
    else:
        print("Nie udało się połączyć z bazą danych.")
        raise RuntimeError("Could not connect to database after retries.")

if __name__ == "__main__":
    init()
