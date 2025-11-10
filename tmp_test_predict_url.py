import requests, json
url = "http://127.0.0.1:8000/predict_url"
payload = {"url": "https://mashable.com/article/hamnet-advanced-screenings-focus-features"}   # replace with any article URL
r = requests.post(url, json=payload, timeout=30)
print(r.status_code)
try:
    print(json.dumps(r.json(), indent=2))
except Exception:
    print(r.text)