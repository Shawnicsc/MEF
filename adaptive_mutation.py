from jailbreak.datasets.jailbreak_datasets import JailbreakDataset
from jailbreak.models.wenxinyiyan_model import WenxinyiyanModel
from jailbreak.attacker.CodeChameleon_2024 import CodeChameleon
from jailbreak.models.kimi_model import KimiModel
from jailbreak.models.huggingface_model import from_pretrained
import torch
# 在导入部分添加 time 模块（如果尚未导入）
import time
import os
import logging

# 设置日志
logging.basicConfig(
    filename='/home/zhangsheng/wangwei/log_codeChameleon.log',
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)

# 百度智能云 
API_KEY = "nzQXeNrTcdwZ0zlQJ7WNz5XX"
SECRET_KEY = "MTvN1o6mcUjxsRFlh4rjPPAef1ZxwLFl"
# kimi api key
Kimi_API_KEY = "sk-ssxAWaNr52a9h15j3WU6ZwAEGtSfnWBXIW5NMtoOHeK54Nw6"
# 本地 llama-2 
MODEL_PATH = "/home/zhangsheng//qwen-7b-chat"
MODEL_PATH_2 = "/home/zhangsheng/Llama-2-7b-chat-hf"
MODEL_PATH_3 = "/home/zhangsheng/wangwei/vicuna-7b-v1.5"
MODEL_PATH_4 = "/home/zhangsheng/wangwei/Llama-2-13b-chat-hf"

# 加载本地模型
target_model = from_pretrained(
                        model_name_or_path=MODEL_PATH_2,
                        model_name="llama-2",
                        dtype=torch.float16)


# 加载评估模型
eval_model = WenxinyiyanModel(API_KEY,SECRET_KEY)


# 加载数据集
jailbreak_dataset = JailbreakDataset('./test_file/heldout_test.jsonl')

# 记录开始时间
start_time = time.time()

# 创建 CodeChameleon 攻击器
attacker = CodeChameleon(
    attack_model=None,
    target_model=target_model,   # 目标模型
    eval_model=eval_model,     # 评估模型
    jailbreak_datasets=jailbreak_dataset,  # 数据集
    log_time=True  # 启用时间记录
)

# 开始攻击
attacker.attack(save_path='/home/zhangsheng/wangwei/codeChameleon_attack_result.jsonl')

# 记录总耗时，转换为毫秒
total_time = (time.time() - start_time) * 1000
logging.info(f"总变异时间: {total_time:.2f} 毫秒")
print(f"总变异时间: {total_time:.2f} 毫秒")