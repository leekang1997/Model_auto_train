#这里主要放数据库的配置信息
# 现有模型列表2024-10-29
#Qwen2-3B-Instruct
#Qwen2-14B-Instruct
Qwen2_3B_Instruct_model_url="http://localhost:8009/"
#Qwen2-0.5B-Instruct_url="http://localhost:8007/"
model='qwen2-7b:latest'
model_qwen72b_ins='qwen2-72B-instruct-F16_20240816_F16_q4:latest'
def client(model):
    client_chat='None'
    #if(model=='Qwen2-3B-Instruct'):
    #   client_chat = f'{Qwen2_3B_Instruct_model_url}v1'
    if(model=='Qwen2-0.5B-Instruct'):
        client_chat= f'http://localhost:8007/v1'
    elif(model=='Qwen2-3B-Instruct'):
        client_chat=f'{Qwen2_3B_Instruct_model_url}v1'

    return client_chat


