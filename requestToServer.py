import requests
import predict
API_ENDPOINT = "http://localhost/firefighter/receive.php"

score = predict.score
class_name = predict.class_name

headers = {
    "Content-Type": "application/x-www-form-urlencoded"
}
values = {
    "score": score,
    "class_name": class_name
}

response = requests.post(API_ENDPOINT, data=values, headers=headers)
if response.ok:
    print("Upload completed successfully!")
    print(f"konten : {response.content}")
else:
    print("Something went wrong!")
