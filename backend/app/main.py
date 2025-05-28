from fastapi import FastAPI
from app.api.v1 import assets, transactions, portfolio
from fastapi.middleware.cors import CORSMiddleware
from app.api.v1 import risk
from app.api.v1 import optimization_endpoints 

app = FastAPI()

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"], 
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

app.include_router(assets.router)
app.include_router(transactions.router)
app.include_router(portfolio.router)
app.include_router(risk.router)
app.include_router(optimization_endpoints.router)

@app.get("/health")
async def read_health():
    return {"status": "ok"}



