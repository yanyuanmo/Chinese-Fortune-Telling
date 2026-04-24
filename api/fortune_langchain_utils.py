# api/fortune_langchain_utils.py
"""
LangChain utilities specialized for the Chinese Fortune Teller application.
Retrieval strategy: HyDE (Hypothetical Document Embedding) + BGE Cross-Encoder Rerank

Pipeline:
  1. HyDE: LLM generates a hypothetical classical Chinese passage from the question
  2. Wide recall: embed the hypothetical passage, retrieve top-k=15 candidates from ChromaDB
  3. BGE rerank: score each candidate against the *original* question, keep top_n=7
  4. Stuff + generate: feed top 7 chunks into the QA prompt
"""

from langchain_openai import ChatOpenAI
from langchain_core.output_parsers import StrOutputParser
from langchain_classic.chains import create_history_aware_retriever, create_retrieval_chain
from langchain_classic.chains.combine_documents import create_stuff_documents_chain
from langchain_core.runnables import RunnableLambda
from langchain_core.documents import Document
import logging
import os

from chroma_utils import get_vectorstore
from fortune_prompts import (
    fortune_contextualize_prompt,
    fortune_qa_prompt,
    birthday_analysis_prompt,
    yearly_forecast_prompt
)

output_parser = StrOutputParser()

# ── HyDE prompt (asks LLM to write a classical-style passage) ──────────────
HYDE_PROMPT = (
    "你是中国传统命理学专家。请根据以下问题，"
    "仿照古典命理文献（文言文/半文言文）的风格，"
    "写一段80-150字的原文片段，直接包含问题答案所涉及的术语和论述。"
    "只输出片段本身，不要标题、序号或解释。\n\n"
    "问题：{question}"
)

# Lazy-loaded BGE cross-encoder (avoid reloading on every request)
_bge_encoder = None

def _get_bge_encoder():
    global _bge_encoder
    if _bge_encoder is None:
        from langchain_community.cross_encoders import HuggingFaceCrossEncoder
        rerank_model = os.getenv("RERANK_MODEL", "BAAI/bge-reranker-base")
        logging.info(f"Loading BGE cross-encoder: {rerank_model}")
        _bge_encoder = HuggingFaceCrossEncoder(model_name=rerank_model)
    return _bge_encoder


def _build_hyde_rerank_retriever(llm, hyde_k: int = 15, top_n: int = 7):
    """
    Returns a RunnableLambda that:
      - Accepts a dict with key 'input' (the current question after history condensing)
      - Returns a list of Documents (top_n reranked)
    """
    vectorstore = get_vectorstore()

    def _retrieve(query) -> list[Document]:
        q = query.get("input", "") if isinstance(query, dict) else str(query)

        # Step 1: HyDE — generate hypothetical passage
        try:
            hyp_text = llm.invoke(HYDE_PROMPT.format(question=q)).content.strip()
        except Exception as e:
            logging.warning(f"HyDE generation failed, falling back to raw query: {e}")
            hyp_text = q

        # Step 2: Wide recall using hypothetical passage embedding
        candidates = vectorstore.similarity_search(hyp_text, k=hyde_k)

        if not candidates:
            return []

        # Step 3: BGE cross-encoder rerank using *original* question
        encoder = _get_bge_encoder()
        pairs  = [[q, doc.page_content] for doc in candidates]
        scores = encoder.score(pairs)
        ranked = sorted(zip(scores, candidates), key=lambda x: x[0], reverse=True)
        return [doc for _, doc in ranked[:top_n]]

    return RunnableLambda(_retrieve)


def get_fortune_chain(query_type: str = "general", model: str = "moonshot-v1-8k"):
    """
    Creates a HyDE + BGE-Rerank RAG chain for fortune telling.

    Args:
        query_type: 'general' | 'bazi' | 'forecast'
        model: Kimi model name

    Returns:
        A retrieval chain (history-aware)
    """
    llm = ChatOpenAI(
        model=model,
        openai_api_key=os.environ.get("MOONSHOT_API_KEY", ""),
        openai_api_base="https://api.moonshot.cn/v1",
        temperature=0.7,
        max_retries=2,
    )

    # History-aware retriever: condenses chat history + question into a standalone query,
    # then feeds that into our HyDE+Rerank retriever
    hyde_rerank_retriever = _build_hyde_rerank_retriever(llm, hyde_k=15, top_n=7)
    history_aware_retriever = create_history_aware_retriever(
        llm,
        hyde_rerank_retriever,
        fortune_contextualize_prompt
    )

    # QA prompt selection
    if query_type == "bazi":
        qa_prompt = birthday_analysis_prompt
    elif query_type == "forecast":
        qa_prompt = yearly_forecast_prompt
    else:
        qa_prompt = fortune_qa_prompt

    question_answer_chain = create_stuff_documents_chain(llm, qa_prompt)
    rag_chain = create_retrieval_chain(history_aware_retriever, question_answer_chain)
    return rag_chain
