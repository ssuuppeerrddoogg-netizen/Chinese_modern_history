from openai import OpenAI

class CustomEmbedding:
    def __init__(self, api_key, base_url):
        self.client = OpenAI(
            api_key=api_key,
            base_url=base_url
        )

    def embed_documents(self, texts):
        embeddings = []
        MAX_LEN = 2000 
        for i, text in enumerate(texts):
            if not isinstance(text, str):
                text = str(text)
            if text.strip() == "":
                text = "空文本"
            # 截断过长文本
            if len(text) > MAX_LEN:
                print(f"⚠️ 文本块 {i} 长度 {len(text)} 超过 {MAX_LEN}，已截断至 {MAX_LEN}")
                text = text[:MAX_LEN]
            
            # 逐个调用API
            response = self.client.embeddings.create(
                model="text-embedding-v1",
                input=text
            )
            embeddings.append(response.data[0].embedding)
        return embeddings

    def embed_query(self, text):
        if not isinstance(text, str):
            text = str(text)
        if len(text) > 2000:
            text = text[:2000]
        response = self.client.embeddings.create(
            model="text-embedding-v1",
            input=text
        )
        return response.data[0].embedding
    
    # 添加 __call__ 方法，使对象可调用
    def __call__(self, text):
        return self.embed_query(text)