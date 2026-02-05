"""
混合语义引擎 - 去中心化场景的最佳实践
结合了本地BERT、领域知识库和MinHash的多层次语义理解

Architecture:
  Layer 1: BERT语义理解 (文本 → 向量 → 相似度)
  Layer 2: MinHash集合匹配 (证据集合 → Jaccard相似度)
  Layer 3: 领域知识增强 (规则 + 同义词库 → 域特定相似度)
  
优点:
  ✅ 完全本地化 - 模型缓存后零网络依赖
  ✅ 真实语义理解 - 识别同义词、近义词、相关概念
  ✅ 轻量级 - 80MB MiniLM模型 + 几KB词典
  ✅ CPU可运行 - 无需GPU
  ✅ 可复现性 - 所有Agent用同一模型
  ✅ 可解释性 - 清晰的分层决策链路
"""

import numpy as np
import json
import warnings
from typing import Dict, List, Tuple, Optional, Any
from pathlib import Path
from datasketch import MinHash
import jieba

# 忽略warnings避免输出冗余信息
warnings.filterwarnings('ignore')

# ==================== Layer 1: BERT语义引擎 ====================

class BertSemanticEngine:
    """
    基于预训练BERT的语义相似度计算
    使用sentence-transformers提供的MiniLM模型（轻量级）
    """
    
    def __init__(self, model_name: str = 'paraphrase-multilingual-MiniLM-L12-v2', 
                 device: str = 'cpu'):
        """
        初始化BERT引擎
        
        Args:
            model_name: 使用的预训练模型
            device: 运行设备 ('cpu' 或 'cuda')
        """
        try:
            from sentence_transformers import SentenceTransformer
            self.model = SentenceTransformer(model_name, device=device)
            self.initialized = True
            print(f"✓ BERT引擎初始化成功 (设备: {device})")
        except ImportError:
            print("⚠ sentence-transformers未安装，BERT功能不可用")
            print("  请运行: pip install sentence-transformers")
            self.initialized = False
            self.model = None
    
    def similarity(self, text1: str, text2: str) -> float:
        """
        计算两段文本的语义相似度
        
        Args:
            text1: 第一段文本
            text2: 第二段文本
            
        Returns:
            相似度分数 [0, 1]
        """
        if not self.initialized:
            return 0.0
        
        try:
            # 转换为字符串并编码
            text1_str = str(text1).strip()
            text2_str = str(text2).strip()
            
            if not text1_str or not text2_str:
                return 0.0
            
            # 编码为向量
            emb1 = self.model.encode(text1_str, convert_to_tensor=True)
            emb2 = self.model.encode(text2_str, convert_to_tensor=True)
            
            # 计算余弦相似度
            similarity = self._cosine_similarity(emb1, emb2)
            
            return float(max(0.0, min(1.0, similarity)))  # 归一化到[0, 1]
        
        except Exception as e:
            print(f"⚠ BERT相似度计算失败: {e}")
            return 0.0
    
    @staticmethod
    def _cosine_similarity(a, b) -> float:
        """计算余弦相似度"""
        norm_a = np.linalg.norm(a.cpu().numpy() if hasattr(a, 'cpu') else a)
        norm_b = np.linalg.norm(b.cpu().numpy() if hasattr(b, 'cpu') else b)
        
        if norm_a == 0 or norm_b == 0:
            return 0.0
        
        a_np = a.cpu().numpy() if hasattr(a, 'cpu') else a
        b_np = b.cpu().numpy() if hasattr(b, 'cpu') else b
        
        return float(np.dot(a_np, b_np) / (norm_a * norm_b))


# ==================== Layer 2: 领域知识库 ====================

class DomainKnowledgeBase:
    """
    金融领域的同义词库和规则引擎
    提供可解释的、可维护的领域特定语义判断
    """
    
    def __init__(self):
        """初始化领域知识库"""
        # 同义词组
        self.synonym_groups = {
            "批准": ["同意", "通过", "核准", "接受", "允许", "批复", "同意"],
            "拒绝": ["否决", "驳回", "不同意", "拒批", "否认", "反对"],
            "资产": ["财产", "资金", "财富", "资源", "家产", "物产"],
            "证明": ["凭证", "文件", "材料", "证件", "据凭", "证据"],
            "评估": ["审核", "审计", "核查", "验证", "检查", "鉴定"],
            "贷款": ["放款", "借款", "融资", "债权", "信贷"],
            "信用": ["信誉", "征信", "资信", "信用额度", "信用记录"],
            "高": ["优", "好", "强", "强劲", "出色"],
            "低": ["差", "弱", "不足", "欠缺"],
            "企业": ["公司", "商业", "机构", "单位", "组织"],
            "个人": ["个体", "人", "自然人"],
        }
        
        # 反义词对
        self.antonyms = {
            "批准": ["拒绝", "驳回"],
            "通过": ["否决"],
            "接受": ["拒绝"],
            "高": ["低"],
            "优": ["差"],
            "通过": ["否决"],
        }
        
        # 构建词汇索引
        self.word_to_group = {}
        for group_id, (key, synonyms) in enumerate(self.synonym_groups.items()):
            for word in [key] + synonyms:
                self.word_to_group[word] = group_id
    
    def similarity(self, text1: str, text2: str) -> float:
        """
        基于规则的语义相似度计算
        
        Args:
            text1: 第一段文本
            text2: 第二段文本
            
        Returns:
            相似度分数 [0, 1]
        """
        text1_str = str(text1).strip()
        text2_str = str(text2).strip()
        
        if not text1_str or not text2_str:
            return 0.0
        
        # 分词
        words1 = set(self._tokenize(text1_str))
        words2 = set(self._tokenize(text2_str))
        
        if not words1 or not words2:
            return 0.0
        
        # 1. 字面匹配
        literal_match = len(words1 & words2) / len(words1 | words2)
        
        # 2. 同义词匹配
        synonym_score = self._synonym_match(words1, words2)
        
        # 3. 反义词惩罚
        antonym_penalty = self._antonym_penalty(words1, words2)
        
        # 综合得分（同义词权重较高）
        final_score = (0.3 * literal_match + 0.7 * synonym_score) * (1 - antonym_penalty)
        
        return max(0.0, min(1.0, final_score))
    
    def _tokenize(self, text: str) -> List[str]:
        """分词处理"""
        try:
            words = jieba.cut(text)
            # 过滤空字符和停用词
            return [w for w in words if len(w.strip()) > 0 and w not in ['的', '了', '在', '是', '都']]
        except:
            # 如果jieba不可用，回退到字符级别
            return list(text)
    
    def _synonym_match(self, words1: set, words2: set) -> float:
        """计算同义词匹配度"""
        if not words1 or not words2:
            return 0.0
        
        matched_score = 0
        total_words = len(words1) + len(words2)
        
        for w1 in words1:
            for w2 in words2:
                if w1 == w2:
                    matched_score += 2.0  # 完全匹配
                elif self._are_synonyms(w1, w2):
                    matched_score += 1.5  # 同义词匹配
        
        return matched_score / total_words if total_words > 0 else 0.0
    
    def _are_synonyms(self, word1: str, word2: str) -> bool:
        """检查两个词是否同义"""
        group1 = self.word_to_group.get(word1)
        group2 = self.word_to_group.get(word2)
        return group1 is not None and group1 == group2
    
    def _antonym_penalty(self, words1: set, words2: set) -> float:
        """计算反义词惩罚系数"""
        for w1 in words1:
            for w2 in words2:
                if w2 in self.antonyms.get(w1, []):
                    return 0.8  # 存在反义词时大幅降低相似度
        return 0.0


# ==================== Layer 3: MinHash集合匹配 ====================

class SetMatchingEngine:
    """
    基于MinHash的证据集合匹配
    用于计算两个证据集合的Jaccard相似度
    """
    
    def __init__(self, num_perm: int = 128):
        """
        初始化集合匹配引擎
        
        Args:
            num_perm: MinHash排列数（越大越精确）
        """
        self.num_perm = num_perm
    
    def similarity(self, set1: Any, set2: Any) -> float:
        """
        计算两个集合的Jaccard相似度
        
        Args:
            set1: 第一个集合（可以是list、str或JSON格式的list字符串）
            set2: 第二个集合
            
        Returns:
            Jaccard相似度 [0, 1]
        """
        # 转换为list格式
        list1 = self._to_list(set1)
        list2 = self._to_list(set2)
        
        if not list1 or not list2:
            return 0.0
        
        # 创建MinHash对象
        m1 = MinHash(num_perm=self.num_perm)
        m2 = MinHash(num_perm=self.num_perm)
        
        # 添加元素
        for item in list1:
            m1.update(str(item).encode('utf-8'))
        
        for item in list2:
            m2.update(str(item).encode('utf-8'))
        
        # 计算Jaccard相似度
        return m1.jaccard(m2)
    
    @staticmethod
    def _to_list(obj: Any) -> List[str]:
        """将对象转换为list"""
        if isinstance(obj, list):
            return obj
        elif isinstance(obj, str):
            # 尝试解析JSON格式的list
            if obj.startswith('['):
                try:
                    return json.loads(obj)
                except:
                    return [obj]
            return [obj]
        else:
            return [obj]


# ==================== 混合语义引擎 ====================

class HybridSemanticEngine:
    """
    混合语义引擎 - 结合三层技术的最佳实践
    
    架构:
      文本对 → Layer1(BERT) → Layer2(MinHash) → Layer3(DomainKB) → 融合 → 最终相似度
    
    特点:
      - 完全本地化：无需网络连接，模型一次性下载
      - 多层次：结合向量语义、集合论和规则三种方法
      - 可扩展：易于添加新的语义层或调整权重
      - 可解释：提供分层决策和调试信息
    """
    
    def __init__(self, enable_bert: bool = True, device: str = 'cpu'):
        """
        初始化混合语义引擎
        
        Args:
            enable_bert: 是否启用BERT引擎（需要sentence-transformers）
            device: 设备选择 ('cpu' 或 'cuda')
        """
        # 初始化三层引擎
        self.bert_engine = BertSemanticEngine(device=device) if enable_bert else None
        self.domain_kb = DomainKnowledgeBase()
        self.set_matcher = SetMatchingEngine()
        
        # 默认权重配置
        self.weights = {
            'bert': 0.5,      # BERT语义理解权重
            'domain': 0.3,    # 领域知识权重
            'minhash': 0.2    # MinHash集合权重
        }
        
        print("✓ 混合语义引擎初始化成功")
        print(f"  - BERT引擎: {'已启用' if self.bert_engine and self.bert_engine.initialized else '禁用'}")
        print(f"  - 领域知识库: 已加载 ({len(self.domain_kb.synonym_groups)}个同义词组)")
        print(f"  - MinHash引擎: 已加载")
    
    def evaluate_game(self, row_a: Dict, row_b: Dict) -> Dict[str, Any]:
        """
        评估两个提议之间的语义相似度（博弈评分）
        
        这是主要的API - 替换原来的四个算子
        
        Args:
            row_a: 提议A (包含 assumptions, evidence, inference, conclusion)
            row_b: 提议B (同上)
            
        Returns:
            {
                'sim_a': 前提相似度,
                'sim_e': 证据相似度,
                'sim_i': 推理相似度,
                'sim_c': 结论相似度,
                'total_score': 综合得分,
                'utility': 博弈收益U,
                'method': 各项使用的方法,
                'debug': 调试信息
            }
        """
        # 提取各层信息
        assumptions_a = str(row_a.get('assumptions', ''))
        assumptions_b = str(row_b.get('assumptions', ''))
        
        evidence_a = row_a.get('evidence', [])
        evidence_b = row_b.get('evidence', [])
        
        inference_a = str(row_a.get('inference', ''))
        inference_b = str(row_b.get('inference', ''))
        
        conclusion_a = str(row_a.get('conclusion', ''))
        conclusion_b = str(row_b.get('conclusion', ''))
        
        # ===== Layer 1: 前提相似度 (BERT + Domain) =====
        sim_a_bert = self.bert_similarity(assumptions_a, assumptions_b)
        sim_a_domain = self.domain_kb.similarity(assumptions_a, assumptions_b)
        sim_a = max(sim_a_bert, sim_a_domain)  # 取较高值
        
        # ===== Layer 2: 证据相似度 (MinHash) =====
        sim_e = self.set_matcher.similarity(evidence_a, evidence_b)
        
        # ===== Layer 3: 推理相似度 (BERT + Domain) =====
        sim_i_bert = self.bert_similarity(inference_a, inference_b)
        sim_i_domain = self.domain_kb.similarity(inference_a, inference_b)
        sim_i = max(sim_i_bert, sim_i_domain)
        
        # ===== Layer 4: 结论相似度 (BERT + Domain) =====
        sim_c_bert = self.bert_similarity(conclusion_a, conclusion_b)
        sim_c_domain = self.domain_kb.similarity(conclusion_a, conclusion_b)
        sim_c = max(sim_c_bert, sim_c_domain)
        
        # ===== 融合得分 =====
        # 权重配置：前提0.2, 证据0.3, 推理0.2, 结论0.3
        layer_weights = {
            'a': 0.2,
            'e': 0.3,
            'i': 0.2,
            'c': 0.3
        }
        
        total_score = (
            sim_a * layer_weights['a'] +
            sim_e * layer_weights['e'] +
            sim_i * layer_weights['i'] +
            sim_c * layer_weights['c']
        )
        
        # ===== 博弈收益函数 =====
        R = 100  # 奖励
        C = 25   # 成本
        utility = (total_score * R) - C
        
        return {
            'sim_a': round(sim_a, 4),
            'sim_e': round(sim_e, 4),
            'sim_i': round(sim_i, 4),
            'sim_c': round(sim_c, 4),
            'total_score': round(total_score, 4),
            'utility': round(utility, 2),
            'method': {
                'assumptions': f"BERT:{sim_a_bert:.2f}, Domain:{sim_a_domain:.2f} → {sim_a:.2f}",
                'evidence': f"MinHash:{sim_e:.2f}",
                'inference': f"BERT:{sim_i_bert:.2f}, Domain:{sim_i_domain:.2f} → {sim_i:.2f}",
                'conclusion': f"BERT:{sim_c_bert:.2f}, Domain:{sim_c_domain:.2f} → {sim_c:.2f}"
            },
            'debug': {
                'bert_available': self.bert_engine is not None and self.bert_engine.initialized,
                'domain_kb_groups': len(self.domain_kb.synonym_groups),
                'minhash_perms': self.set_matcher.num_perm
            }
        }
    
    def bert_similarity(self, text1: str, text2: str) -> float:
        """计算BERT相似度"""
        if self.bert_engine and self.bert_engine.initialized:
            return self.bert_engine.similarity(text1, text2)
        return 0.0
    
    def domain_similarity(self, text1: str, text2: str) -> float:
        """计算领域知识相似度"""
        return self.domain_kb.similarity(text1, text2)
    
    def set_similarity(self, set1: Any, set2: Any) -> float:
        """计算集合相似度"""
        return self.set_matcher.similarity(set1, set2)
    
    def set_weights(self, bert_weight: float = None, domain_weight: float = None, 
                    minhash_weight: float = None):
        """
        调整三层权重配置
        
        Args:
            bert_weight: BERT层权重
            domain_weight: 领域知识层权重
            minhash_weight: MinHash层权重
        """
        if bert_weight is not None:
            self.weights['bert'] = bert_weight
        if domain_weight is not None:
            self.weights['domain'] = domain_weight
        if minhash_weight is not None:
            self.weights['minhash'] = minhash_weight
        
        # 归一化
        total = sum(self.weights.values())
        for key in self.weights:
            self.weights[key] /= total
    
    def add_domain_synonyms(self, synonyms_dict: Dict[str, List[str]]):
        """
        添加自定义同义词组
        
        Args:
            synonyms_dict: {关键词: [同义词列表]}
        """
        for key, words in synonyms_dict.items():
            if key not in self.domain_kb.synonym_groups:
                self.domain_kb.synonym_groups[key] = words
                # 重建索引
                for word in [key] + words:
                    self.domain_kb.word_to_group[word] = len(self.domain_kb.synonym_groups) - 1


# ==================== 测试与演示 ====================

if __name__ == "__main__":
    import json
    
    print("\n" + "="*60)
    print("混合语义引擎 - 测试演示")
    print("="*60 + "\n")
    
    # 初始化引擎
    engine = HybridSemanticEngine(enable_bert=True, device='cpu')
    
    # 测试数据
    test_cases = [
        {
            'name': '完全同义 - 批准贷款',
            'a': {
                'assumptions': '标准信用评估流程',
                'evidence': ['身份证', '营业执照'],
                'inference': '符合所有条件',
                'conclusion': '批准贷款申请'
            },
            'b': {
                'assumptions': '标准信用评估',
                'evidence': ['身份证', '营业执照'],
                'inference': '满足全部要求',
                'conclusion': '同意放款请求'
            }
        },
        {
            'name': '反义 - 批准vs拒绝',
            'a': {
                'assumptions': '基础审核标准',
                'evidence': ['身份证'],
                'inference': '基本符合',
                'conclusion': '批准申请'
            },
            'b': {
                'assumptions': '基础审核标准',
                'evidence': ['身份证'],
                'inference': '不符合要求',
                'conclusion': '拒绝申请'
            }
        },
        {
            'name': '无关 - 不同领域',
            'a': {
                'assumptions': '金融风险评估',
                'evidence': ['财务报表'],
                'inference': '经过验证',
                'conclusion': '批准融资'
            },
            'b': {
                'assumptions': '天气预报算法',
                'evidence': ['气象数据'],
                'inference': '已处理',
                'conclusion': '明天有雨'
            }
        }
    ]
    
    # 运行测试
    for test in test_cases:
        print(f"\n📋 {test['name']}")
        print("-" * 60)
        
        result = engine.evaluate_game(test['a'], test['b'])
        
        print(f"前提相似度 (sim_a): {result['sim_a']:.4f}")
        print(f"  {result['method']['assumptions']}")
        
        print(f"\n证据相似度 (sim_e): {result['sim_e']:.4f}")
        print(f"  {result['method']['evidence']}")
        
        print(f"\n推理相似度 (sim_i): {result['sim_i']:.4f}")
        print(f"  {result['method']['inference']}")
        
        print(f"\n结论相似度 (sim_c): {result['sim_c']:.4f}")
        print(f"  {result['method']['conclusion']}")
        
        print(f"\n综合得分: {result['total_score']:.4f}")
        print(f"博弈收益U: {result['utility']:.2f}")
        
        # 判定策略
        if result['utility'] > 55:
            strategy = "✅ ESS_Consensus (达成共识)"
        elif result['utility'] > 0:
            strategy = "⚠️  Audit_Required (需要审计)"
        else:
            strategy = "❌ Inconsistent_Reject (不一致，拒绝)"
        
        print(f"决策: {strategy}")
    
    print("\n" + "="*60)
    print("✓ 测试完成")
    print("="*60)
