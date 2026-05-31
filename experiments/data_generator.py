"""数据生成器：为4个实验生成多领域 AEIC 测试数据"""
import random
from typing import List, Dict

DOMAINS = {
    "finance": {
        "A": ["假设股市处于上升趋势", "假设利率保持稳定", "假设市场情绪积极",
              "假设经济增长放缓", "假设流动性充足", "假设通胀预期可控"],
        "E": ["最近三个月上市公司盈利增长15%", "央行数据显示存款准备金率下降",
              "机构投资者净买入增加30%", "GDP增速下降到5.5%", "市场融资规模创新高",
              "消费者信心指数回升至120"],
        "I": ["通过盈利数据推断市场向好", "流动性充足导致资金面宽松",
              "机构行为表明看好后市", "经济增速放缓但仍保持增长",
              "融资便利性提高促进投资", "多项指标交叉验证支持乐观判断"],
        "C": ["建议增加股票配置", "认为短期可继续持股", "判断市场将继续上涨",
              "建议控制风险敞口", "看好市场长期表现", "综合评估风险收益比良好"],
    },
    "medical": {
        "A": ["假设患者基础代谢正常", "假设症状由炎症引起", "假设免疫系统功能正常",
              "假设无其他并发症", "假设患者遵从医嘱", "假设药物无过敏反应"],
        "E": ["血液检查显示白细胞升高40%", "影像学检查发现局部浓聚",
              "CRP指标升高到80mg/L", "患者体温升至38.5摄氏度",
              "患者主诉疼痛加重", "血培养结果阳性"],
        "I": ["综合化验结果判断为细菌感染", "影像特征符合炎症表现",
              "炎症指标证实急性期", "症状与体征相符",
              "诊断方向明确为感染", "多项检查结果互相印证"],
        "C": ["建议使用广谱抗生素", "需要住院进一步观察", "预计7天内有效控制",
              "密切监测生命体征", "后期需做恢复期评估", "建议联合用药方案"],
    },
    "legal": {
        "A": ["假设合同条款合法有效", "假设双方具备完全民事行为能力",
              "假设证据链条完整", "假设适用中国合同法", "假设不存在欺诈情形"],
        "E": ["合同文本第三条明确约定违约金", "银行转账记录显示已支付定金",
              "公证书确认签字真实", "录音证据证明协商过程", "第三方鉴定报告确认损失"],
        "I": ["依据合同法第107条判断构成违约", "因果关系链条完整",
              "损失金额有明确证据支撑", "违约责任认定无争议", "法律适用无冲突"],
        "C": ["判决被告承担违约责任", "赔偿金额为合同约定数额", "驳回原告其他诉讼请求",
              "诉讼费用由被告承担", "建议调解结案"],
    },
}


def generate_dataset(num_tasks: int = 21, nodes_per_task: int = 3,
                     consistency: str = "high") -> List[Dict]:
    """
    生成实验数据集

    consistency: "high" / "medium" / "low" 控制节点间相似程度
    """
    random.seed(42)
    dataset = []
    domain_names = list(DOMAINS.keys())

    for t in range(num_tasks):
        domain = domain_names[t % len(domain_names)]
        data = DOMAINS[domain]
        nodes = []

        for n in range(nodes_per_task):
            if consistency == "high":
                # 高一致性：从相同子集选取
                a_idx = t % len(data["A"])
                e_indices = [(t + k) % len(data["E"]) for k in range(3)]
                i_idx = t % len(data["I"])
                c_idx = t % len(data["C"])
            elif consistency == "medium":
                # 中一致性：部分相同部分不同
                a_idx = (t + n) % len(data["A"])
                e_indices = [(t + n + k) % len(data["E"]) for k in range(3)]
                i_idx = t % len(data["I"])
                c_idx = (t + n) % len(data["C"])
            else:
                # 低一致性：随机选取
                a_idx = random.randint(0, len(data["A"]) - 1)
                e_indices = random.sample(range(len(data["E"])), 3)
                i_idx = random.randint(0, len(data["I"]) - 1)
                c_idx = random.randint(0, len(data["C"]) - 1)

            nodes.append({
                "A": data["A"][a_idx],
                "E": [data["E"][i] for i in e_indices],
                "I": data["I"][i_idx],
                "C": data["C"][c_idx],
            })

        dataset.append({"nodes": nodes, "domain": domain, "consistency": consistency})

    return dataset


def generate_diversity_data(topic: str, mode: str = "heterogeneous") -> List[Dict]:
    """为认知多样性实验生成数据"""
    if mode == "homogeneous":
        base = {
            "A": f"基于标准分析框架，{topic}的主要前提是现有趋势持续",
            "E": [f"统计数据表明{topic}趋势稳定", f"历史分析支持该判断", f"主流研究结论一致"],
            "I": f"基于线性外推和主流共识，{topic}的标准推论为正向发展",
            "C": f"综合分析认为{topic}前景积极",
        }
        return [
            {**base, "A": base["A"] + f"（分析师{i}确认）"} for i in range(3)
        ]
    else:
        return [
            {
                "A": f"从宏观经济视角看{topic}，假设参与者理性决策",
                "E": ["GDP增长率数据", "货币政策走向分析", "国际贸易格局变化"],
                "I": f"供需模型和乘数效应显示{topic}的经济影响深远",
                "C": f"从经济学角度判断{topic}将带来结构性变化",
            },
            {
                "A": f"从社会结构视角分析{topic}，关注群体行为",
                "E": ["社会调查数据", "人口结构变迁", "舆情趋势分析"],
                "I": f"社会网络效应表明{topic}具有快速扩散特征",
                "C": f"从社会学角度预测{topic}将重塑社会互动模式",
            },
            {
                "A": f"从技术可行性视角评估{topic}，假设技术持续演进",
                "E": ["技术成熟度曲线", "专利申请趋势", "产业路线图"],
                "I": f"技术扩散模型显示{topic}的技术拐点正在临近",
                "C": f"从技术角度判断{topic}的实现条件已基本成熟",
            },
        ]
