import requests

def test_markowitz_ai_endpoint():
    print("\n=== Testing /optimize/markowitz-ai endpoint ===")
    url = "http://localhost:8000/optimize/markowitz-ai"
    params = {
        'gamma': 1.0,
        'period': '5y',
        'top_n': 2
    }

    try:
        response = requests.post(url, params=params)
        if response.status_code == 200:
            data = response.json()
            print("✅ Endpoint response:")
            for symbol, weight in data['weights'].items():
                print(f"   {symbol}: {weight:.4f}")
        else:
            print(f"❌ API Error: {response.status_code}")
            print(f"Details: {response.text}")
    except Exception as e:
        print(f"❌ Request Error: {e}")

if __name__ == "__main__":
    test_markowitz_ai_endpoint()
