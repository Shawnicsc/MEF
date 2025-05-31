"""
CodeChameleon Class
============================================
A novel framework for jailbreaking in LLMs based on
personalized encryption and decryption.

Paper title: CodeChameleon: Personalized Encryption Framework for Jailbreaking Large Language Models
arXiv Link: https://arxiv.org/abs/2402.16717
"""

import logging
import os.path
import time

# 设置日志配置，同时输出到控制台和文件
log_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'logs')
os.makedirs(log_dir, exist_ok=True)
log_file = os.path.join(log_dir, f'codechameleon_{time.strftime("%Y%m%d_%H%M%S")}.log')

# 创建日志处理器
file_handler = logging.FileHandler(log_file, encoding='utf-8')
console_handler = logging.StreamHandler()

# 设置日志格式
log_format = logging.Formatter('%(asctime)s - %(levelname)s - %(message)s')
file_handler.setFormatter(log_format)
console_handler.setFormatter(log_format)

# 配置根日志记录器
logger = logging.getLogger()
logger.setLevel(logging.INFO)
logger.addHandler(file_handler)
logger.addHandler(console_handler)

logging.info(f"日志将保存到: {log_file}")

from jailbreak.metrics.Evaluator import EvaluatorGenerativeJudge
from jailbreak.attacker import AttackerBase
from jailbreak.datasets import JailbreakDataset, Instance
from jailbreak.mutation.rule import *

__all__ = ['CodeChameleon']


class CodeChameleon(AttackerBase):
    r"""
    Implementation of CodeChameleon Jailbreak Challenges in Large Language Models
    """

    def __init__(self, attack_model, target_model, eval_model, jailbreak_datasets, log_time=False):
        # 保存现有的初始化代码
        self.attack_model = attack_model
        self.target_model = target_model
        self.eval_model = eval_model
        self.jailbreak_datasets = jailbreak_datasets
        # 添加时间记录标志
        self.log_time = log_time
        self.mutations = [
            BinaryTree(attr_name='query'),
            # Length(attr_name='query'),
            # Reverse(attr_name='query'),
            # OddEven(attr_name='query'),
        ]
        # 不需要评估器，仅保留代码但不使用
        self.evaluator = EvaluatorGenerativeJudge(eval_model)
        self.current_jailbreak = 0
        self.current_query = 0
        self.current_reject = 0

    def single_attack(self, instance: Instance) -> JailbreakDataset:
        r"""
        Single attack process with mutation-specific logging
        """
        instance_ds = JailbreakDataset([instance])
        source_instance_list = []
        updated_instance_list = []

        # Track mutations and log transformation results
        for mutation in self.mutations:
            mutation_name = mutation.__class__.__name__
            transformed_jailbreak_datasets = mutation(instance_ds)

            # Log mutation application and instance count
            logging.info(
                f"Applied {mutation_name} mutation. Generated {len(transformed_jailbreak_datasets)} instances.")

            # Add mutation metadata to instances
            for item in transformed_jailbreak_datasets:
                item.metadata = getattr(item, 'metadata', {})  # Initialize metadata if not exists
                item.metadata['applied_mutation'] = mutation_name
                source_instance_list.append(item)

        # 仅生成变异提示词，不执行攻击和评估
        for instance in source_instance_list:
            try:
                # 只生成格式化的提示词，不调用模型生成回答
                formatted_prompt = instance.jailbreak_prompt.format(
                    decryption_function=instance.decryption_function,
                    query=instance.query
                )
                
                # 不再调用模型生成回答
                answer = self.target_model.generate(formatted_prompt)
                instance.target_responses.append(answer)

                # 记录变异信息
                mutation_used = instance.metadata.get('applied_mutation', 'unknown')
                logging.info(
                    f"Mutation '{mutation_used}' processed\n"
                    f"Original query: {instance.query}\n"
                    f"Transformed query: {formatted_prompt}\n"
                    # f"Model response: {answer}\n"
                    "-------------------------"
                )

                updated_instance_list.append(instance)

            except Exception as e:
                logging.error(f"Error processing instance: {str(e)}")

        return JailbreakDataset(updated_instance_list)

    def get_mutated_prompts(self) -> list:
        r"""
        获取所有变异后的提示词
        
        :return: 包含所有变异后提示词的列表，包括攻击结果信息
        """
        mutated_prompts = []
        for instance in self.attack_results:
            formatted_prompt = instance.jailbreak_prompt.format(
                decryption_function=instance.decryption_function,
                query=instance.query
            )
            mutated_prompts.append({
                'original_query':None,
                'mutated_prompt': formatted_prompt,
                'decrypted_result':None,
                'isSuccess': None,
                # 'mutation_type': instance.metadata.get('applied_mutation', 'unknown'),
                # 'num_jailbreak': instance.num_jailbreak,
                # 'num_reject': instance.num_reject,
            })
        return mutated_prompts
        
    def save_mutated_prompts(self, save_path='mutated_prompts.csv'):
        """
        将所有变异后的提示词保存到一个文件中，使用CSV格式
        
        :param save_path: 保存文件的路径
        """
        import csv
        mutated_prompts = self.get_mutated_prompts()
        with open(save_path, 'w', encoding='utf-8', newline='') as f:
            fieldnames = ['original_query', 'mutated_prompt', 'decrypted_result', 'isSuccess']
            writer = csv.DictWriter(f, fieldnames=fieldnames)
            writer.writeheader()
            for prompt_data in mutated_prompts:
                writer.writerow(prompt_data)
        logging.info(f"变异后的提示词已保存至: {os.path.abspath(save_path)}")

    def attack(self, save_path = 'CodeChameleon_attack_result.jsonl'):
        r"""
        Execute attack with enhanced progress logging
        """
        # 记录开始时间
        if self.log_time:
            start_time = time.time()
            logging.info("开始变异攻击...")
            
        logging.info("Jailbreak started! Initializing mutations...")
        self.attack_results = JailbreakDataset([])
        try:
            for idx, instance in enumerate(self.jailbreak_datasets):
                # 记录每个样本的处理时间
                if self.log_time:
                    sample_start_time = time.time()
                    
                logging.info(f"Processing instance {idx + 1}/{len(self.jailbreak_datasets)}")
                results = self.single_attack(instance)
                for new_instance in results: 
                    self.attack_results.add(new_instance)
                    
                # 记录每个样本的处理时间，转换为毫秒
                if self.log_time:
                    sample_time = (time.time() - sample_start_time) * 1000
                    logging.info(f"样本 {idx+1} 变异时间: {sample_time:.2f} 毫秒")

            logging.info("Final evaluation starting...")
            self.evaluator(self.attack_results)
            self.update(self.attack_results)
            logging.info("Final evaluation completed.")
        
        except KeyboardInterrupt:
            logging.warning("Jailbreak interrupted by user! Partial results preserved.")

        # 保存变异后的提示词到文件
        #self.save_mutated_prompts()
        
        # 记录结束时间，转换为毫秒
        if self.log_time:
            end_time = time.time()
            total_time = (end_time - start_time) * 1000
            logging.info(f"变异攻击完成，总耗时: {total_time:.2f} 毫秒")
        
        # 不再需要记录和保存结果
        self.log()
        self.jailbreak_datasets.save_to_jsonl(save_path)
        logging.info(
            'Jailbreak result saved at {}!'.format(os.path.join(os.path.dirname(os.path.abspath(__file__)), save_path)))
        
        # 仅返回变异后的提示词
        #return self.attack_results, self.get_mutated_prompts()

    def update(self, Dataset: JailbreakDataset):
        r"""
        Update the state of the Jailbroken based on the evaluation results of Datasets.
        
        :param Dataset: The Dataset that is attacked.
        """
        # 不再需要更新状态
        for prompt_node in Dataset:
            self.current_jailbreak += prompt_node.num_jailbreak
            self.current_query += prompt_node.num_query
            self.current_reject += prompt_node.num_reject

    def log(self):
        r"""
        Report the attack results.
        """
        logging.info("======Jailbreak report:======")
        logging.info(f"Total queries: {self.current_query}")
        logging.info(f"Total jailbreak: {self.current_jailbreak}")
        logging.info(f"Total reject: {self.current_reject}")
        logging.info("========Report End===========")
