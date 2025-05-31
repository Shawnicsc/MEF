'''
GPTFuzzer Class
============================================
This Class achieves a jailbreak method describe in the paper below.
This part of code is based on the code from the paper.

Paper title: GPTFUZZER: Red Teaming Large Language Models with Auto-Generated Jailbreak Prompts

arXiv link: https://arxiv.org/pdf/2309.10253.pdf

Source repository: https://github.com/sherdencooper/GPTFuzz
'''
import logging
import random
import numpy as np
from tqdm import tqdm
import json  # 添加json库
import os  # 添加os库用于处理文件路径

from jailbreak.attacker.attacker_base import AttackerBase
from jailbreak.constraint import DeleteHarmLess
from jailbreak.datasets.instance import Instance
from jailbreak.metrics.Evaluator import EvaluatorClassificatonJudge
from jailbreak.seed import SeedTemplate
from jailbreak.selector.MCTSExploreSelectPolicy import MCTSExploreSelectPolicy
from jailbreak.datasets import JailbreakDataset
from jailbreak.mutation.generation import CrossOver, Expand, GenerateSimilar, Shorten, Rephrase


class GPTFuzzer(AttackerBase):
    """
    GPTFuzzer is a class for performing fuzzing attacks on LLM-based models.
    It utilizes mutator and selection policies to generate jailbreak prompts,
    aiming to find vulnerabilities in target models.
    """
    def __init__(self,attack_model, target_model, eval_model , jailbreak_datasets:JailbreakDataset = None, energy:int = 1,  max_query: int = 100, max_jailbreak: int = 100, max_reject: int = 100, max_iteration: int = 100, seeds_num = 76, template_file = None, output_file = "gptfuzzer_results.jsonl"):
        """
        Initialize the GPTFuzzer object with models, policies, and configurations.
        :param ~ModelBase attack_model: The model used to generate attack prompts.
        :param ~ModelBase target_model: The target GPT model being attacked.
        :param ~ModelBase eval_model: The model used for evaluation during attacks.
        :param ~JailbreakDataset jailbreak_datasets: Initial set of prompts for seed pool, if any.
        :param int max_query: Maximum query.
        :param int max_jailbreak: Maximum number of jailbroken issues.
        :param int max_reject: Maximum number of rejected issues.
        :param int max_iteration: Maximum iteration for mutate testing.
        :param str output_file: Path to save the results in JSONL format.
        """
        super().__init__(attack_model, target_model, eval_model, jailbreak_datasets)
        self.Questions = jailbreak_datasets
        self.Questions_length = len(self.Questions)
        self.initial_prompt_seed = SeedTemplate().new_seeds(seeds_num=seeds_num, prompt_usage='attack', method_list=['Gptfuzzer'],template_file=template_file)
        self.prompt_nodes = JailbreakDataset(
            [Instance(jailbreak_prompt= prompt) for prompt in self.initial_prompt_seed]
        )
        for i,instance in enumerate(self.prompt_nodes):
            instance.index = i
            instance.visited_num = 0
            instance.level = 0
        for i,instance in enumerate(self.Questions):
            instance.index = i
        self.initial_prompts_nodes = JailbreakDataset([instance for instance in self.prompt_nodes])

        self.current_query: int = 0
        self.current_jailbreak: int = 0
        self.current_reject: int = 0

        self.total_query = 0
        self.total_jailbreak = 0
        self.total_reject = 0

        self.current_iteration: int = 0

        self.max_query: int = max_query
        self.max_jailbreak: int = max_jailbreak
        self.max_reject: int = max_reject
        self.max_iteration: int = max_iteration
        self.energy: int = energy
        
        # 添加输出文件路径
        self.output_file = output_file
        # 初始化结果文件
        self._initialize_output_file()

        self.mutations = [
            CrossOver(self.attack_model, seed_pool=self.initial_prompts_nodes),
            Expand(self.attack_model),
            GenerateSimilar(self.attack_model),
            Shorten(self.attack_model),
            Rephrase(self.attack_model)
        ]
        self.select_policy = MCTSExploreSelectPolicy(self.prompt_nodes, self.initial_prompts_nodes, self.Questions)
        self.evaluator = EvaluatorClassificatonJudge(self.eval_model)
        self.constrainer = DeleteHarmLess(self.attack_model, prompt_pattern = '{jailbreak_prompt}', attr_name = ['jailbreak_prompt'])

        self.select_policy.initial()

    def _initialize_output_file(self):
        """初始化输出文件，创建文件并写入头部信息"""
        # 确保输出目录存在
        output_dir = os.path.dirname(self.output_file)
        if output_dir and not os.path.exists(output_dir):
            os.makedirs(output_dir)
            
        # 清空文件内容或创建新文件
        with open(self.output_file, 'w', encoding='utf-8') as f:
            pass
        
        logging.info(f"Results will be saved to {self.output_file}")

    def save_results(self, instances):
        """
        将实例结果保存为JSONL格式
        :param instances: 要保存的实例列表
        """
        with open(self.output_file, 'a', encoding='utf-8') as f:
            for instance in instances:
                # 确定是否成功（jailbreak）
                is_success = instance.num_jailbreak > 0
                
                # 构建要保存的数据
                result_data = {
                    "prompt": instance.jailbreak_prompt,
                    "query": instance.query,
                    "full_prompt": instance.jailbreak_prompt + instance.query if instance.query else instance.jailbreak_prompt,
                    "success": is_success,
                    "response": instance.target_responses[0] if instance.target_responses else "",
                    "iteration": self.current_iteration
                }
                
                # 写入JSONL格式
                f.write(json.dumps(result_data, ensure_ascii=False) + '\n')

    def attack(self):
        """
        Main loop for the fuzzing process, repeatedly selecting, mutating, evaluating, and updating.
        """
        logging.info("Fuzzing started!")
        self.attack_results = JailbreakDataset([])
        try:
            while not self.is_stop():
                seed_instance = self.select_policy.select()[0]
                mutated_results = self.single_attack(seed_instance)
                for instance in mutated_results:
                    instance.parents = [seed_instance]
                    instance.children = []
                    seed_instance.children.append(instance)
                    instance.index = len(self.prompt_nodes)
                    self.prompt_nodes.add(instance)

                for mutator_instance in mutated_results:
                    self.temp_results = JailbreakDataset([])
                    for query_instance in tqdm(self.Questions):
                        temp_instance = mutator_instance.copy()
                        temp_instance.target_responses = []
                        temp_instance.eval_results = []
                        temp_instance.query = query_instance.query
                        if '{query}' in temp_instance.jailbreak_prompt:
                            input_seed = temp_instance.jailbreak_prompt.replace('{query}',temp_instance.query)
                        else:
                            input_seed = temp_instance.jailbreak_prompt + temp_instance.query
                        response = self.target_model.generate(input_seed)
                        temp_instance.target_responses.append(response)
                        self.temp_results.add(temp_instance)
                    self.evaluator(self.temp_results)
                    mutator_instance.level = seed_instance.level + 1
                    mutator_instance.visited_num = 0

                    self.update(self.temp_results)
                    # 保存本次迭代的结果
                    self.save_results(self.temp_results)
                    for instance in self.temp_results:
                        self.attack_results.add(instance.copy())
                self.log()
        except KeyboardInterrupt:
            logging.info("Fuzzing interrupted by user!")
        self.jailbreak_datasets = self.attack_results
        logging.info(f"Fuzzing finished! Results saved to {self.output_file}")


    def single_attack(self, instance: Instance):
        """
        Perform an attack using a single query.
        :param ~Instance instance: The instance to be used in the attack. In gptfuzzer, the instance jailbreak_prompt is mutated by different methods.
        :return: ~JailbreakDataset: The response from the mutated query.
        """
        # 判断instance中有jailbreak_prompt
        assert instance.jailbreak_prompt is not None, 'A jailbreak prompt must be provided'
        instance = instance.copy()
        instance.parents = []
        instance.children = []
        mutator = random.choice(self.mutations)

        return_dataset = JailbreakDataset([])
        for i in range(self.energy):
            instance = mutator(JailbreakDataset([instance]))[0]
            if instance.query is not None:
                if '{query}' in instance.jailbreak_prompt:
                    input_seed = instance.jailbreak_prompt.format(query=instance.query)
                else:
                    input_seed = instance.jailbreak_prompt + instance.query
                response = self.target_model.generate(input_seed)
                instance.target_responses.append(response)
            instance.parents = []
            instance.children = []
            return_dataset.add(instance)
        return return_dataset

    def is_stop(self):
        """
         Check if the stopping criteria for fuzzing are met.
         :return bool: True if any stopping criteria is met, False otherwise.
         """
        checks = [
            ('max_query', 'total_query'),
            ('max_jailbreak', 'total_jailbreak'),
            ('max_reject', 'total_reject'),
            ('max_iteration', 'current_iteration'),
        ]
        return any(getattr(self, max_attr) != -1 and getattr(self, curr_attr) >= getattr(self, max_attr) for
                   max_attr, curr_attr in checks)


    def update(self, Dataset: JailbreakDataset):
        """
        Update the state of the fuzzer based on the evaluation results of prompt nodes.
        :param ~JailbreakDataset prompt_nodes: The prompt nodes that have been evaluated.
        """
        self.current_iteration += 1

        current_jailbreak = 0
        current_query = 0
        current_reject = 0
        for instance in Dataset:
            current_jailbreak += instance.num_jailbreak
            current_query += instance.num_query
            current_reject += instance.num_reject

            self.total_jailbreak += instance.num_jailbreak
            self.total_query += instance.num_query
            self.total_reject += instance.num_reject

        self.current_jailbreak = current_jailbreak
        self.current_query = current_query
        self.current_reject = current_reject

        self.select_policy.update(Dataset)

    def log(self):
        """
        The current attack status is displayed
        """
        logging.info(
            f"Iteration {self.current_iteration}: {self.current_jailbreak} jailbreaks, {self.current_reject} rejects, {self.current_query} queries")
        logging.info(
            f"Total: {self.total_jailbreak} jailbreaks, {self.total_reject} rejects, {self.total_query} queries")
        print('现在成功了： ',len(self.attack_results))
        print(f'结果已保存到: {self.output_file}')

