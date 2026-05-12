import numpy as np
import jieba
import re
from itertools import combinations
from openai import OpenAI


class ConsistencyChecker:
    def __init__(self, api_key, base_url):
        self.client = OpenAI(api_key=api_key, base_url=base_url)
    #语义一致性
    def get_embedding(self, text):
        response = self.client.embeddings.create(
            model="text-embedding-v1",
            input=text
        )
        return np.array(response.data[0].embedding)

    def semantic_consistency(self, answers):
        embeddings = [self.get_embedding(a) for a in answers]
        sims = []
        for i, j in combinations(range(len(embeddings)), 2):
            sim = self.cosine_similarity(embeddings[i], embeddings[j])
            sims.append(sim)
        return float(np.mean(sims)) if sims else 1.0

    # 关键词一致性
    def extract_keywords(self, text, topk=10):
        words = jieba.lcut(text)
        words = [w for w in words if len(w) > 1]
        freq = {}
        for w in words:
            freq[w] = freq.get(w, 0) + 1
        sorted_words = sorted(freq.items(), key=lambda x: x[1], reverse=True)
        return set([w for w, _ in sorted_words[:topk]])

    def keyword_consistency(self, answers):
        keyword_sets = [self.extract_keywords(a) for a in answers]
        sims = []
        for i, j in combinations(range(len(keyword_sets)), 2):
            inter = len(keyword_sets[i] & keyword_sets[j])
            union = len(keyword_sets[i] | keyword_sets[j])
            sim = inter / union if union > 0 else 0
            sims.append(sim)
        return np.mean(sims) if sims else 1.0

    #事实一致性
    def extract_facts(self, text):
        years = re.findall(r'\d{4}年', text)
        return set(years)

    def fact_consistency(self, answers):
        fact_sets = [self.extract_facts(a) for a in answers]
        sims = []
        for i, j in combinations(range(len(fact_sets)), 2):
            inter = len(fact_sets[i] & fact_sets[j])
            union = len(fact_sets[i] | fact_sets[j])
            sim = inter / union if union > 0 else 0
            sims.append(sim)
        return np.mean(sims) if sims else 1.0

    # ----------------------
    # 总一致性（你要的加权公式）
    # consistency = 0.5*语义 + 0.3*关键词 + 0.2*事实
    # ----------------------
    def compute_consistency(self, answers):
        semantic_score = self.semantic_consistency(answers)
        keyword_score = self.keyword_consistency(answers)
        fact_score = self.fact_consistency(answers)

        # 严格按你给的权重计算
        total_consistency = (
            0.5 * semantic_score +
            0.3 * keyword_score +
            0.2 * fact_score
        )
        # 返回 总分数 + 三个维度分数（方便你看细节）
        return round(total_consistency, 4), round(semantic_score, 4), round(keyword_score, 4), round(fact_score, 4)

    # ----------------------
    # 选择最具代表性答案（保留你原来的）
    # ----------------------
    def select_representative(self, answers):
        embeddings = [self.get_embedding(a) for a in answers]
        scores = []
        for i in range(len(embeddings)):
            sim_sum = 0
            for j in range(len(embeddings)):
                if i != j:
                    sim_sum += self.cosine_similarity(embeddings[i], embeddings[j])
            scores.append(sim_sum)
        best_idx = np.argmax(scores)
        return answers[best_idx]

    # ----------------------
    # 工具方法
    # ----------------------
    @staticmethod
    def cosine_similarity(a, b):
        return np.dot(a, b) / (np.linalg.norm(a) * np.linalg.norm(b))