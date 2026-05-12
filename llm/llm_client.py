from openai import OpenAI

class LLMClient:
    def __init__(self, api_key, base_url):
        self.client = OpenAI(
            api_key=api_key,
            base_url=base_url
        )

    def chat(self, model_name, question, context=None, system_prompt="请简要回答以下问题："):
        try:
            if context:
                user_content = f"回答时参考以下内容：\n{context}\n\n问题：{question}"
            else:
                user_content = question

            response = self.client.chat.completions.create(
                model=model_name,
                messages=[
                    {"role": "system", "content": system_prompt},
                    {"role": "user", "content": user_content}
                ],
                stream=False,
                extra_body={"enable_thinking": False}
            )
            return response.choices[0].message.content.strip()
        except Exception as e:
            return f"❌ 调用失败: {str(e)}"        
        
