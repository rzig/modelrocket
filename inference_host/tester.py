import requests

res = requests.post("http://localhost:5002/inference", json={"token": "30fceeb2-99c0-4366-89ca-8946090059f8", "model": "79ba0bfa-9195-4900-a60d-6ff034614d83", "inputs": {"modelInput": [[1, 2, 3, 4], [1, 2, 3, 4]]}})
print(res.content)