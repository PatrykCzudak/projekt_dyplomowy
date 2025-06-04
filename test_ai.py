import requests

def test_asset_risk_ai(base_url, symbol):
    endpoint = f"{base_url}/risk/asset/{symbol}/ai"
    try:
        response = requests.get(endpoint)
        response.raise_for_status()
        data = response.json()
        print(f"\n✅ Wynik dla {symbol.upper()}:")
        print(f"  - Kategoria ryzyka: {data.get('risk_category')}")
        print(f"  - Prawdopodobieństwo: {data.get('risk_prediction')}")
    except requests.exceptions.RequestException as e:
        print(f"\n❌ Błąd podczas zapytania dla {symbol.upper()}: {e}")
        if e.response is not None:
            print("Szczegóły błędu:")
            print(e.response.text)

def main():
    # Adres backendu FastAPI
    base_url = "http://localhost:8000"

    # Lista symboli do przetestowania
    symbols = ["AAPL", "MSFT", "NVDA", "GOOG", "MSFT"]

    print("🔍 Testuję endpoint /risk/asset/{symbol}/ai ...")
    for symbol in symbols:
        test_asset_risk_ai(base_url, symbol)

if __name__ == "__main__":
    main()
