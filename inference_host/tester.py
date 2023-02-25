import requests

res = requests.post("http://localhost:5002/inference", json={"model": "1234", "inputs": {"modelInput": [[1, 2, 3, 4], [1, 2, 3, 4]]}})
print(res.content)