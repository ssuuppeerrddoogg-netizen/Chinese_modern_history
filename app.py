import streamlit as st
from llm.llm_client import LLMClient
from config.settings import API_KEY, BASE_URL, MODELS
from utils.consistency import ConsistencyChecker
from utils.evidence import EvidenceChecker
from rag.rag_pipeline import RAGPipeline
from utils.evidence import EvidenceChecker
from rag.timeline import build_timeline_context

st.set_page_config(page_title="LLM测试系统", layout="wide")

st.title("中国近代史大模型问答系统")

# 初始化客户端
llm = LLMClient(API_KEY, BASE_URL)
@st.cache_resource
def load_rag():
    return RAGPipeline(API_KEY, BASE_URL)
rag = load_rag()

# 侧边栏：模型选择
st.sidebar.header("参数设置")
model_choice = st.sidebar.selectbox(
    "模型选择",
    list(MODELS.keys())
)
rag_mode = st.sidebar.selectbox(
    "选择回答模式",
    ["直接生成（无RAG）", "普通RAG", "时间线RAG"]
)

model_name = MODELS[model_choice]
num_samples = st.sidebar.slider("自洽性检验采样次数", 1, 10, 5)
st.sidebar.caption("当采样次数=1时本系统将自动关闭自洽性检验")
enable_consistency = num_samples > 1

# 主界面
question = st.text_area("请输入你的问题：", height=100)

if st.button("提交问题"):
    if question.strip() == "":
        st.warning("请输入问题")
    else:
        context = None
        docs = None
        timeline_data = None

        docs_for_check = rag.retrieve(question, k=3)
        if rag_mode != "直接生成（无RAG）":
            docs = rag.retrieve(question, k=3)
            if rag_mode == "普通RAG":
                context = rag.build_context(docs)
            elif rag_mode == "时间线RAG":
                context, timeline_data = build_timeline_context(docs, question.split())

        answers=[]
        with st.spinner("模型正在思考/作答，请稍等..."):
            for i in range(num_samples):
                ans = llm.chat(model_name, question, context=context)
                answers.append(ans)

        if not enable_consistency:
            #只有1个样本，直接取第一个答案
            final_answer = answers[0]
            consistency_score = None
            evidence_score = None
            R = None
        else:
            #一致性计算
            checker = ConsistencyChecker(API_KEY, BASE_URL)
            total_consistency, semantic_score, keyword_score, fact_score = checker.compute_consistency(answers)
            consistency_score = total_consistency
            final_answer = checker.select_representative(answers)  

            #证据支持度计算
            evidence_checker = EvidenceChecker(API_KEY, BASE_URL)
            evidence_score = evidence_checker.compute_evidence_with_docs(final_answer, docs_for_check)    

            #风险值R计算
            alpha, beta = 0.6, 0.4
            if consistency_score is not None and evidence_score is not None:
                R = alpha * (1 - consistency_score) + beta * (1 - evidence_score)
            else:
                R = None


        #展示UI（优化版）
        st.subheader("📌 最终答案")
        st.success(final_answer)

        #一致性等级颜色显示
        st.subheader("📊 幻觉风险评估")
        col1, col2, col3 = st.columns(3)      

        # 一致性
        with col1:
            if enable_consistency:
                if consistency_score > 0.8:
                    label = "🟢 高"
                elif consistency_score > 0.5:
                    label = "🟡 中"
                else:
                    label = "🔴 低"

                st.metric("一致性", f"{consistency_score:.2f}", label)
            else:
                st.metric("一致性", "未启用")

        # 证据支持
        with col2:
            if evidence_score is not None:
                if evidence_score > 0.7:
                    label = "🟢 强"
                elif evidence_score > 0.4:
                    label = "🟡 中"
                else:
                    label = "🔴 弱"

                st.metric("证据支持", f"{evidence_score:.2f}", label)
            else:
                st.metric("证据支持", "未启用")

        # 风险值 R
        with col3:
            if R is not None:
                if R < 0.3:
                    label = "🟢 低风险"
                elif R < 0.6:
                    label = "🟡 中风险"
                else:
                    label = "🔴 高风险"

                st.metric("风险值 R", f"{R:.2f}", label)
            else:
                st.metric("风险值 R", "未计算")

        #若采用RAG，折叠面板展示检索证据
        if rag_mode != "直接生成（无RAG）":
            with st.expander("📂 查看检索证据"):
                for i, doc in enumerate(docs):
                    st.write(f"**证据{i+1}:**")
                    st.write(doc.page_content[:300] + "...")

        # timeline模式展示时间线分析
        if rag_mode == "时间线RAG":
            with st.expander("📅 时间线分析"):
                for item in timeline_data:
                    t = item["t"]
                    text = item["text"][:200]

                    if t:
                        y, md = t // 10000, t % 10000
                        m, d = md // 100, md % 100

                        if m == 0 and d == 0:
                            label = f"{y}年"
                        elif d == 0:
                            label = f"{y}年{m}月"
                        else:
                            label = f"{y}年{m}月{d}日"
                    else:
                        label = "无时间信息"

                    st.markdown(f"**🕒 {label}**")
                    st.write(text + "...")
                    st.divider()            

        #折叠面板展示所有答案
        with st.expander("查看所有候选答案"):
            for i, ans in enumerate(answers):
                st.write(f"**回答{i+1}:** {ans}")

#with st.expander("🔧 调试：查看所有文本块（仅开发用）"):
    #if st.checkbox("显示全部文本块", key="show_all_chunks"):
        # 注意：需要通过 rag.vectorstore 获取所有文档
        # 由于 FAISS 存储方式不同，可能需要迭代 index 或 docstore
        #all_docs = list(rag.vectorstore.docstore._dict.values())
        #for i, doc in enumerate(all_docs):
            #st.write(f"**Chunk {i+1}** (长度: {len(doc.page_content)}):")
            #st.write(doc.page_content[:300] + ("..." if len(doc.page_content) > 300 else ""))
            #st.divider()