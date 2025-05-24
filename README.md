### Aplikacja webowa do zarządzania portfelem inwestycyjnym i analizy ryzyka.

#### Zrealizowane funkcjonalności
Backend (FastAPI + SQLAlchemy + PostgreSQL)
 - Rejestracja aktywów i transakcji (automatyczne dodawanie aktywa jeśli nie istnieje)
 - Obliczanie i wyświetlanie stanu portfela w czasie rzeczywistym
 - Pobieranie danych historycznych z Yahoo Finance (via yfinance)
 - Moduł analizy ryzyka:
   - Klasyczne metody:
     - VaR (parametryczny)
     - VaR (historyczny)
     - Expected Shortfall
   - Metody AI:
     - Klasyfikacja poziomu ryzyka
     - Regresja wartości ryzyka
 - Endpointy REST API do pobierania wyników ryzyka dla aktywa i portfela
 - Obsługa CORS (React <-> FastAPI)

Frontend (React + Tailwind CSS + ApexCharts)
 - Dynamiczne dodawanie transakcji (z automatycznym przeliczaniem ceny z Yahoo)
 - Widok portfela z aktualną i historyczną wartością pozycji
 - Strona z wykresami historycznymi (wybór aktywa, zakres czasu, typ wykresu – linia/świeczki)
 - Widok "Risk" z zakładkami:
 - Ryzyko aktywa (AssetRisk)
 - Ryzyko portfela (PortfolioRisk)
 - Możliwość dodawania dynamicznych „kafelków” analizy i rozwijania wykresów

#### Rzeczy do zrobienia / Do poprawy
 - Obsługa błędów po stronie backendu i frontend
 - Estetyczne dopracowanie UI (np. animacje, kolory, przejścia)
 - Responsywność (dostosowanie do mobile)
 - Autoryzacja / zarządzanie kontami użytkowników (jeśli planowane)
 - Większa kontrola nad wyborem zakresu dat (na frontendzie)
 - Utrwalanie wyników analizy ryzyka (np. do bazy danych)
 - Jednostkowe testy backendu (Pytest)
 - Więcej modeli AI dla oceny ryzyka
 - Implementacja metod optymalizacji portfolio klasycznych (Markowitz) i AI
 - (Opcjonalnie) Tworzenie raportów przy pomocy LLM/SLM

### Jak uruchomić:
 - Backend:
   bash
   - docker compose up --build
 - Frontend:
   bash
   - npm instal && npm run dev

Frontend dostępny pod: http://localhost:3000
Backend (FastAPI docs): http://localhost:8000/docs