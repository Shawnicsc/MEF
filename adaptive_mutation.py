from jailbreak.datasets.jailbreak_datasets import JailbreakDataset
from jailbreak.attacker.Adaptive_2025 import Adaptive
from jailbreak.models.kimi_model import KimiModel
from jailbreak.models.huggingface_model import from_pretrained
import torch
import time
import os
import logging

# 加载本地模型
target_model = from_pretrained(
                        model_name_or_path=MODEL_PATH_2,
                        model_name="llama-2",
                        dtype=torch.float16)


# 加载评估模型
eval_model = KimiModel(API_KEY,SECRET_KEY)


# 加载数据集
jailbreak_dataset = JailbreakDataset('./test_file/heldout_test.jsonl')

# 记录开始时间
start_time = time.time()

# 创建攻击器
attacker = Adaptive(
    attack_model=None,
    target_model=target_model, # 目标模型
    eval_model=eval_model,  # 评估模型
    jailbreak_datasets=jailbreak_dataset,  # 数据集
    log_time=True 
)

# 开始攻击
attacker.attack()

# 记录总耗时
total_time = (time.time() - start_time) * 1000
logging.info(f"总变异时间: {total_time:.2f} 毫秒")
print(f"总变异时间: {total_time:.2f} 毫秒")