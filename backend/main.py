
# ───────────────────────────────
#  Importy
# ───────────────────────────────
from fastapi import FastAPI, HTTPException
import uvicorn

app = FastAPI(
    title="Investment Portfolio API",
    version="0.1.0",
    docs_url="/docs",          # Swagger
    redoc_url="/redoc",        # ReDoc
)


# ───────────────────────────────
#  Root & Health-check
# ───────────────────────────────
@app.get("/", tags=["system"])
async def root():
    """Prosty endpoint testowy."""
    return {"message": "Investment Portfolio API is up 🟢"}


@app.get("/health", tags=["system"])
async def healthcheck():
    """Używane przez load-balancer / monitoring."""
    return {"status": "ok"}


# ───────────────────────────────
#  Autoryzacja (chyba że nie ma autoryzacji) xD
# ───────────────────────────────
@app.post("/auth/login", tags=["auth"])
async def login(username: str, password: str):
    """
    Zwraca fikcyjny token JWT – placeholder do późniejszej implementacji.
    """
    if username == "admin" and password == "password":
        return {"access_token": "fake-jwt-token", "token_type": "bearer"}
    raise HTTPException(status_code=401, detail="Invalid credentials")


# ───────────────────────────────
#  Portfel – podsumowanie (szkic)
# ───────────────────────────────
@app.get("/portfolio/summary", tags=["portfolio"])
async def portfolio_summary():
    """
    Zwraca przykładowe podsumowanie portfela.
    Właściwa logika zostanie dodana później.
    """
    return {
        "total_value": 0.0,
        "positions": [],
    }


# Uruchomienie przez `python backend/main.py` (gdybyś nie korzystał z Uvicorna CLI)
if __name__ == "__main__":
    import uvicorn
    uvicorn.run("main:app", host="0.0.0.0", port=8000, reload=True)