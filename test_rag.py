import requests

url = "http://127.0.0.1:8000/rag/ask"
data = {
    "question": "How do I wear the Gear VR headset?",
    "top_k": 3
}

response = requests.post(url, json=data)
print(response.json())