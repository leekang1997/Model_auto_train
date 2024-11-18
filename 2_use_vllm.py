import requests
import json

# 配置 API 调用地址和请求数据
api_url = "http://127.0.0.1:8000/v1/chat/completions"  # 根据实际服务地址调整
headers = {
    "Content-Type": "application/json",
    "Authorization": "Bearer YOUR_API_KEY"  # 如果不需要认证，可以忽略
}
payload = {
    "model": "Qwen2-7B-Instruct",
    "messages": [
        {"role": "system", "content": "You are a helpful assistant."},
        {"role": "user", "content": "请告诉我如何使用LLama Factory训练模型？"}
    ],
    "max_tokens": 200,
    "temperature": 0.7
}

# 调用模型服务
def call_model_service():
    try:
        print("正在调用模型服务...")
        response = requests.post(api_url, headers=headers, data=json.dumps(payload))
        if response.status_code == 200:
            result = response.json()
            print("调用成功，模型返回结果：")
            print(json.dumps(result, indent=4, ensure_ascii=False))
        else:
            print(f"调用失败，状态码: {response.status_code}")
            print(response.text)
    except Exception as e:
        print(f"调用模型服务时出错: {e}")

if __name__ == "__main__":
    call_model_service()
