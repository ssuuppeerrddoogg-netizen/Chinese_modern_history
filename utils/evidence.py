import numpy as np
from openai import OpenAI

class EvidenceChecker:
    def __init__(self, api_key, base_url):
        self.client = OpenAI(api_key=api_key, base_url=base_url)
        self.MAX_LEN = 2000   # 根据你模型实际限制调整

    def get_embedding(self, text):
        if not isinstance(text, str):
            text = str(text)
        if text.strip() == "":
            text = "空文本"
        # 截断超长文本
        if len(text) > self.MAX_LEN:
            text = text[:self.MAX_LEN]
        response = self.client.embeddings.create(
            model="text-embedding-v1",
            input=text
        )
        return np.array(response.data[0].embedding)

    def compute_evidence_with_docs(self, final_answer, docs):
        final_emb = self.get_embedding(str(final_answer))
        sims = []
        for doc in docs:
            content = str(doc.page_content).strip()

            if not content:
                continue  # 跳过空内容

            emb = self.get_embedding(content)
            sim = self.cosine_similarity(final_emb, emb)
            sims.append(sim)

        if len(sims) == 0:
            return 0.0
        return float(np.max(sims)) #取最大相似度作为证据支持度
    @staticmethod
    def cosine_similarity(a, b):
        return np.dot(a, b) / (np.linalg.norm(a) * np.linalg.norm(b))