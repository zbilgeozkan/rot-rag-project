import requests

url = "http://localhost:8000/rag/ask"
data = {
    "question": "How do I wear the Gear VR headset?",
    "top_k": 3
}

response = requests.post(url, json=data)
print(response.json())