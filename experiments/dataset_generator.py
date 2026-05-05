"""
实验数据生成器
======================================================================

生成不同场景的测试数据集，包括：
- 简单一致性场景
- 中等一致性场景
- 低一致性场景
"""

import random
from typing import List, Dict


class DatasetGenerator:
    """生成多智能体语义共识的测试数据"""
    
    # 不同领域的假设、证据、推理、结论样本
    domains = {
        'finance': {
            'assumptions': [
                '假设股市处于上升趋势',
                '假设利率保持稳定',
                '假设市场情绪积极',
                '假设经济增长放缓',
                '假设流动性充足'
            ],
            'evidence': [
                '最近三个月上市公司盈利增长15%',
                '央行公布数据显示存款准备金率下降',
                '机构投资者净买入增加30%',
                'GDP增速下降到5.5%',
                '市场融资规模创新高'
            ],
            'inference': [
                '通过盈利数据推断市场向好',
                '流动性充足导致资金面宽松',
                '机构行为表明看好后市',
                '经济增速放缓但仍保持增长',
                '融资便利性提高促进投资'
            ],
            'conclusion': [
                '建议增加股票配置',
                '认为短期可继续持股',
                '判断市场将继续上涨',
                '建议控制风险敞口',
                '看好市场长期表现'
            ]
        },
        'medical': {
            'assumptions': [
                '假设患者基础代谢正常',
                '假设症状由炎症引起',
                '假设免疫系统功能正常',
                '假设无其他并发症',
                '假设患者遵从医嘱'
            ],
            'evidence': [
                '血液检查显示白细胞升高40%',
                '影像学检查发现局部浓聚',
                'CRP指标升高到80mg/L',
                '患者体温升至38.5摄氏度',
                '患者主诉疼痛加重'
            ],
            'inference': [
                '综合化验结果判断为细菌感染',
                '影像特征符合炎症表现',
                '炎症指标证实急性期',
                '症状与体征相符',
                '诊断方向明确为感染'
            ],
            'conclusion': [
                '建议使用广谱抗生素',
                '需要住院进一步观察',
                '预计7天内有效控制',
                '密切监测生命体征',
                '后期需做恢复期评估'
            ]
        },
        'legal': {
            'assumptions': [
                '假设当事人完全民事行为能力',
                '假设证据链完整可靠',
                '假设无时效期限问题',
                '假设合同条款清晰',
                '假设无其他法律冲突'
            ],
            'evidence': [
                '书面合同明确约定违约责任',
                '证人证言支持一方主张',
                '银行转账记录显示资金流向',
                '现场检验报告记录在案',
                '通话录音记录交易过程'
            ],
            'inference': [
                '根据合同条款判断被告违约',
                '证据链条完整指向事实',
                '财务数据证实经济损失',
                '技术鉴定报告支持诉讼请求',
                '程序合法性得以确认'
            ],
            'conclusion': [
                '判决支持原告诉讼请求',
                '责令被告承担赔偿责任',
                '确定赔偿额度为50万元',
                '要求支付迟延履行利息',
                '案件费用由被告负担'
            ]
        }
    }
    
    @staticmethod
    def generate_aeic_record(domain: str, seed: int = None) -> Dict:
        """生成单个AEIC记录"""
        if seed is not None:
            random.seed(seed)
        
        if domain not in DatasetGenerator.domains:
            domain = list(DatasetGenerator.domains.keys())[0]
        
        domain_data = DatasetGenerator.domains[domain]
        
        return {
            'assumptions': random.choice(domain_data['assumptions']),
            'evidence': random.choice(domain_data['evidence']),
            'inference': random.choice(domain_data['inference']),
            'conclusion': random.choice(domain_data['conclusion'])
        }
    
    @staticmethod
    def generate_identical_nodes(num_nodes: int, domain: str = 'finance') -> List[Dict]:
        """生成相同或极其相似的节点（高一致性场景）"""
        base_record = DatasetGenerator.generate_aeic_record(domain, seed=42)
        nodes = []
        
        for i in range(num_nodes):
            if i == 0:
                nodes.append(base_record)
            else:
                # 轻微变化（添加同义词或重述）
                record = base_record.copy()
                nodes.append(record)
        
        return nodes
    
    @staticmethod
    def generate_similar_nodes(num_nodes: int, domain: str = 'finance') -> List[Dict]:
        """生成相似的节点（中等一致性场景）"""
        nodes = []
        base_indices = list(range(len(DatasetGenerator.domains[domain]['assumptions'])))
        random.shuffle(base_indices)
        
        for i in range(num_nodes):
            idx = base_indices[i % len(base_indices)]
            domain_data = DatasetGenerator.domains[domain]
            
            # 从同一索引取相关内容
            record = {
                'assumptions': domain_data['assumptions'][idx],
                'evidence': domain_data['evidence'][(idx + 1) % len(domain_data['evidence'])],
                'inference': domain_data['inference'][(idx + 2) % len(domain_data['inference'])],
                'conclusion': domain_data['conclusion'][(idx + 3) % len(domain_data['conclusion'])]
            }
            nodes.append(record)
        
        return nodes
    
    @staticmethod
    def generate_diverse_nodes(num_nodes: int, domain: str = 'finance') -> List[Dict]:
        """生成多样化的节点（低一致性场景）"""
        nodes = []
        
        for i in range(num_nodes):
            # 完全随机选择
            record = DatasetGenerator.generate_aeic_record(domain, seed=None)
            nodes.append(record)
        
        return nodes
    
    @staticmethod
    def create_benchmark_dataset(num_tasks_per_scenario: int = 7) -> Dict[str, List[Dict]]:
        """创建基准数据集，包含所有场景和所有一致性水平"""
        dataset = {
            'high_consistency': [],
            'medium_consistency': [],
            'low_consistency': []
        }
        
        domains_list = list(DatasetGenerator.domains.keys())
        
        # 高一致性场景
        for i in range(num_tasks_per_scenario):
            domain = domains_list[i % len(domains_list)]
            nodes = DatasetGenerator.generate_identical_nodes(3, domain)
            dataset['high_consistency'].append({'nodes': nodes, 'domain': domain})
        
        # 中等一致性场景
        for i in range(num_tasks_per_scenario):
            domain = domains_list[i % len(domains_list)]
            nodes = DatasetGenerator.generate_similar_nodes(3, domain)
            dataset['medium_consistency'].append({'nodes': nodes, 'domain': domain})
        
        # 低一致性场景
        for i in range(num_tasks_per_scenario):
            domain = domains_list[i % len(domains_list)]
            nodes = DatasetGenerator.generate_diverse_nodes(3, domain)
            dataset['low_consistency'].append({'nodes': nodes, 'domain': domain})
        
        return dataset
    
    @staticmethod
    def create_simple_dataset(num_tasks: int = 21, num_nodes: int = 3) -> List[Dict]:
        """
        创建简单数据集（混合所有场景）
        
        Args:
            num_tasks: 任务数
            num_nodes: 每个任务的节点数
        
        Returns:
            任务数据集列表
        """
        dataset = []
        consistency_types = [
            DatasetGenerator.generate_identical_nodes,
            DatasetGenerator.generate_similar_nodes,
            DatasetGenerator.generate_diverse_nodes
        ]
        domains_list = list(DatasetGenerator.domains.keys())
        
        for i in range(num_tasks):
            consistency_fn = consistency_types[i % len(consistency_types)]
            domain = domains_list[i % len(domains_list)]
            
            nodes = consistency_fn(num_nodes, domain)
            dataset.append({
                'nodes': nodes,
                'domain': domain,
                'consistency_type': consistency_fn.__name__
            })
        
        return dataset


if __name__ == '__main__':
    print("Testing dataset generator...")
    
    # 测试简单数据集
    dataset = DatasetGenerator.create_simple_dataset(7)
    
    print(f"\nGenerated {len(dataset)} tasks")
    
    for i, task in enumerate(dataset[:2]):
        print(f"\nTask {i+1} ({task['consistency_type']}, domain: {task['domain']}):")
        for j, node in enumerate(task['nodes'][:2]):
            print(f"  Node {j+1}:")
            print(f"    Conclusion: {node['conclusion'][:30]}...")
    
    # 测试基准数据集
    benchmark = DatasetGenerator.create_benchmark_dataset(3)
    print(f"\n\nBenchmark dataset:")
    for scenario, tasks in benchmark.items():
        print(f"  {scenario}: {len(tasks)} tasks")
