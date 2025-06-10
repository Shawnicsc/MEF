import json
import logging
from typing import List, Dict, Any
from dataclasses import dataclass, field

@dataclass
class DataInstance:
    """单个数据实例的数据类"""
    query: str
    jailbreak_prompt: str = None
    reference_responses: List[str] = field(default_factory=list)
    target_responses: List[str] = field(default_factory=list)
    eval_results: List[bool] = field(default_factory=list)
    metadata: Dict[str, Any] = field(default_factory=dict)

    def copy(self) -> 'DataInstance':
        return DataInstance(
            query=self.query,
            jailbreak_prompt=self.jailbreak_prompt,
            reference_responses=self.reference_responses.copy(),
            target_responses=self.target_responses.copy(),
            eval_results=self.eval_results.copy(),
            metadata=self.metadata.copy()
        )

class DatasetHandler:
    """数据集处理类"""
    
    def __init__(self):
        self.instances: List[DataInstance] = []
        
    def add_instance(self, instance: DataInstance) -> None:
        """添加单个实例到数据集"""
        self.instances.append(instance)
        
    def load_from_json(self, filepath: str) -> None:
        """从JSON文件加载数据集"""
        try:
            with open(filepath, 'r', encoding='utf-8') as f:
                data = json.load(f)
                for item in data:
                    self.instances.append(DataInstance(**item))
        except Exception as e:
            logging.error(f"加载数据集失败: {str(e)}")
            
    def save_to_json(self, filepath: str) -> None:
        """保存数据集到JSON文件"""
        try:
            data = [vars(instance) for instance in self.instances]
            with open(filepath, 'w', encoding='utf-8') as f:
                json.dump(data, f, ensure_ascii=False, indent=2)
        except Exception as e:
            logging.error(f"保存数据集失败: {str(e)}")
            
    def get_all_queries(self) -> List[str]:
        """获取所有查询文本"""
        return [instance.query for instance in self.instances]
    
    def get_instance_by_query(self, query: str) -> DataInstance:
        """根据查询文本获取实例"""
        for instance in self.instances:
            if instance.query == query:
                return instance
        return None