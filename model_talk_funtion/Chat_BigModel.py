from openai import OpenAI
#import Data_info
#Qwen2-14B-Instruct

def chat_BigModel(message, history,model_info):
    """调用大模型并返回响应"""
    client = OpenAI(
        base_url='http://localhost:8009/v1',
        api_key='ollama',  # required, but unused
    )

    # 构造消息
    messages = []
    for hist in history:
        # 添加历史消息
        messages.append({"role": "user", "content": hist})

    # 添加当前消息
    messages.append({"role": "user", "content": message})

    # 调用模型
    response = client.chat.completions.create(
        model=model_info,
        messages=messages
    )
    #print(response)
    # 返回模型的回复
    return response.choices[0].message.content


print(chat_BigModel(' {时间}:2022-12-19 00:00:00+08:00 指标：给矿量瞬时值设定:350.000', '','Qwen2-0.5B-Instruct'))
'''
# 示例调用
history = []  # 这里应该是历史对话记录的列表
#print(chat_BigModel('C620-1型普通车床故障分析的目的是什么?', '','qwen2-72B-instruct-F16_20240816_F16_q4:latest'))
print(chat_BigModel('C620-1型普通车床故障分析的目的是什么?', history))
'''
'''try:
        parsed_data = json.loads(response.choices[0].message.content)
        print(parsed_data)
        return json.dumps(parsed_data)  # 返回JSON字符串
    except json.JSONDecodeError as err:
        print(f"解析错误: {err}")
        return response.choices[0].message.content  # 如果无法解析，则返回原始内容
'''
# 其他定义保持不变
