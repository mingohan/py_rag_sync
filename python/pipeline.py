"""
RAG Query Pipeline
Hybrid Search（Dense + Sparse/BM25）+ RRF + AutoMerging + Re-ranking + Multi-query
使用 Qdrant + Vertex AI
"""
import threading
from llama_index.core import VectorStoreIndex, PromptTemplate, Settings
from llama_index.core.callbacks import CallbackManager
from llama_index.core.retrievers import QueryFusionRetriever
from llama_index.core.query_engine import RetrieverQueryEngine
from llama_index.core.base.embeddings.base import BaseEmbedding
from llama_index.core.llms import CustomLLM, CompletionResponse, LLMMetadata
from llama_index.core.llms.callbacks import llm_completion_callback
from llama_index.core.postprocessor.types import BaseNodePostprocessor
from llama_index.core.schema import NodeWithScore, QueryBundle
# Vertex import is deferred to build_llm() to avoid blocking on import
from llama_index.vector_stores.qdrant import QdrantVectorStore
from qdrant_client import QdrantClient, AsyncQdrantClient
from qdrant_client.models import Distance, VectorParams, SparseVectorParams, SparseIndexParams, PayloadSchemaType
from typing import List, Optional

from .config import get_settings

settings = get_settings()


def _build_langfuse_callback():
    """若 Langfuse key 有設定則回傳 callback，否則回傳 None"""
    if not (settings.langfuse_public_key and settings.langfuse_secret_key):
        return None
    try:
        from langfuse.llama_index import LlamaIndexCallbackHandler
        return LlamaIndexCallbackHandler(
            public_key=settings.langfuse_public_key,
            secret_key=settings.langfuse_secret_key,
            host=settings.langfuse_host,
        )
    except Exception as e:
        print(f"[langfuse] init failed, skipping: {e}")
        return None


def _init_vertex():
    import vertexai
    vertexai.init(
        project=settings.google_cloud_project,
        location=settings.google_cloud_location,
    )


def _get_credentials():
    import os
    from google.oauth2 import service_account
    sa_path = os.environ.get(
        "GOOGLE_APPLICATION_CREDENTIALS",
        settings.google_application_credentials,
    )
    return service_account.Credentials.from_service_account_file(
        sa_path,
        scopes=["https://www.googleapis.com/auth/cloud-platform"],
    )


class GeminiVertexEmbedding(BaseEmbedding):
    """Custom embedding class using google-genai with Vertex AI backend."""

    def _get_client(self):
        from google import genai
        from google.genai import types as genai_types  # noqa: F401
        creds = _get_credentials()
        return genai.Client(
            vertexai=True,
            project=settings.google_cloud_project,
            location=settings.google_cloud_location,
            credentials=creds,
        )

    def _embed(self, texts: List[str]) -> List[List[float]]:
        import time
        from google.genai import types as genai_types
        client = self._get_client()
        for attempt in range(5):
            try:
                result = client.models.embed_content(
                    model=settings.embedding_model,
                    contents=texts,
                    config=genai_types.EmbedContentConfig(
                        task_type="RETRIEVAL_QUERY",
                        output_dimensionality=settings.embedding_dimensions,
                    ),
                )
                return [e.values for e in result.embeddings]
            except Exception as e:
                if ("429" in str(e) or "Quota exceeded" in str(e) or "RESOURCE_EXHAUSTED" in str(e)) and attempt < 4:
                    wait = 60 * (attempt + 1)
                    print(f"  [Query Embedding] 429 limit hit. Waiting {wait}s before retry {attempt+1}/4...")
                    time.sleep(wait)
                    continue
                raise

    def _get_query_embedding(self, query: str) -> List[float]:
        return self._embed([query])[0]

    def _get_text_embedding(self, text: str) -> List[float]:
        import time
        from google.genai import types as genai_types
        client = self._get_client()
        for attempt in range(5):
            try:
                result = client.models.embed_content(
                    model=settings.embedding_model,
                    contents=[text],
                    config=genai_types.EmbedContentConfig(
                        task_type="RETRIEVAL_DOCUMENT",
                        output_dimensionality=settings.embedding_dimensions,
                    ),
                )
                return result.embeddings[0].values
            except Exception as e:
                if "429" in str(e) or "Quota exceeded" in str(e) or "RESOURCE_EXHAUSTED" in str(e):
                    if attempt < 4:
                        wait = 60 * (attempt + 1)
                        print(f"  [Embedding] 429 limit hit. Waiting {wait}s before retry {attempt+1}/4...")
                        time.sleep(wait)
                        continue
                raise

    def _get_text_embeddings(self, texts: List[str]) -> List[List[float]]:
        import time
        from google.genai import types as genai_types
        client = self._get_client()
        for attempt in range(5):
            try:
                result = client.models.embed_content(
                    model=settings.embedding_model,
                    contents=texts,
                    config=genai_types.EmbedContentConfig(
                        task_type="RETRIEVAL_DOCUMENT",
                        output_dimensionality=settings.embedding_dimensions,
                    ),
                )
                return [e.values for e in result.embeddings]
            except Exception as e:
                if "429" in str(e) or "Quota exceeded" in str(e) or "RESOURCE_EXHAUSTED" in str(e):
                    if attempt < 4:
                        wait = 30 * (attempt + 1)
                        print(f"  [Embedding Batch] 429 limit hit. Waiting {wait}s before retry {attempt+1}/4...")
                        time.sleep(wait)
                        continue
                raise

    async def _aget_query_embedding(self, query: str) -> List[float]:
        import asyncio
        return await asyncio.to_thread(self._get_query_embedding, query)

    async def _aget_text_embedding(self, text: str) -> List[float]:
        import asyncio
        return await asyncio.to_thread(self._get_text_embedding, text)


def build_embedding():
    return GeminiVertexEmbedding(embed_batch_size=10)


_genai_client_local = threading.local()


class _GenAIGlobalLLM(CustomLLM):
    """google-genai SDK wrapper，支援 location=global（給 gemini-3-flash-preview 用）"""

    model_name: str = "gemini-3-flash-preview"
    temperature: float = 0.1
    max_tokens_val: int = 8192

    @property
    def metadata(self) -> LLMMetadata:
        return LLMMetadata(
            model_name=self.model_name,
            context_window=1_000_000,
            num_output=self.max_tokens_val,
        )

    def _client(self):
        from google import genai
        # 每個 thread 持有獨立 client，避免 httpx 跨 thread 被關閉
        if not getattr(_genai_client_local, "client", None):
            _genai_client_local.client = genai.Client(
                vertexai=True,
                project=settings.google_cloud_project,
                location=settings.google_cloud_llm_location,
                credentials=_get_credentials(),
            )
        return _genai_client_local.client

    @llm_completion_callback()
    def complete(self, prompt: str, **kwargs) -> CompletionResponse:
        from google.genai import types as genai_types
        config = genai_types.GenerateContentConfig(
            temperature=self.temperature,
            max_output_tokens=self.max_tokens_val,
        )
        response = self._client().models.generate_content(
            model=self.model_name,
            contents=prompt,
            config=config,
        )
        return CompletionResponse(text=response.text)

    @llm_completion_callback()
    def stream_complete(self, prompt: str, **kwargs):
        from google.genai import types as genai_types
        config = genai_types.GenerateContentConfig(
            temperature=self.temperature,
            max_output_tokens=self.max_tokens_val,
        )
        text = ""
        for chunk in self._client().models.generate_content_stream(
            model=self.model_name,
            contents=prompt,
            config=config,
        ):
            delta = chunk.text or ""
            text += delta
            yield CompletionResponse(text=text, delta=delta)


def build_llm():
    llm_location = settings.google_cloud_llm_location
    if llm_location == "global":
        return _GenAIGlobalLLM(
            model_name=settings.llm_model,
            temperature=settings.llm_temperature,
            max_tokens_val=settings.llm_max_tokens,
        )
    import vertexai
    from llama_index.llms.vertex import Vertex
    vertexai.init(
        project=settings.google_cloud_project,
        location=llm_location,
    )
    return Vertex(
        model=settings.llm_model,
        project=settings.google_cloud_project,
        location=llm_location,
        credentials=_get_credentials(),
        temperature=settings.llm_temperature,
        max_tokens=settings.llm_max_tokens,
    )


def get_qdrant_client() -> QdrantClient:
    return QdrantClient(url=settings.qdrant_url)


def get_async_qdrant_client() -> AsyncQdrantClient:
    return AsyncQdrantClient(url=settings.qdrant_url)


def ensure_collection(client: QdrantClient):
    """Collection 不存在時自動建立（含 Dense + Sparse vector + payload index）"""
    collections = [c.name for c in client.get_collections().collections]
    if settings.qdrant_collection not in collections:
        client.create_collection(
            collection_name=settings.qdrant_collection,
            vectors_config={
                "dense": VectorParams(
                    size=settings.embedding_dimensions,
                    distance=Distance.COSINE,
                )
            },
            sparse_vectors_config={
                "sparse": SparseVectorParams(
                    index=SparseIndexParams(on_disk=False)
                )
            },
        )
        # Payload index 加速 filter 刪除（source_type、source_id）
        for field in ("source_type", "source_id"):
            client.create_payload_index(
                collection_name=settings.qdrant_collection,
                field_name=field,
                field_schema=PayloadSchemaType.KEYWORD,
            )


def _bm25_sparse_encoder():
    """使用 Qdrant/bm25（與 sync pipeline 一致），避免載入 SPLADE neural model（~600MB）。"""
    from llama_index.vector_stores.qdrant.utils import fastembed_sparse_encoder
    return fastembed_sparse_encoder(model_name="Qdrant/bm25")


def build_vector_store(client: QdrantClient, aclient: AsyncQdrantClient | None = None) -> QdrantVectorStore:
    sparse_fn = _bm25_sparse_encoder()
    return QdrantVectorStore(
        client=client,
        aclient=aclient,
        collection_name=settings.qdrant_collection,
        dense_vector_name="dense",
        sparse_vector_name="sparse",
        enable_hybrid=True,
        sparse_doc_fn=sparse_fn,
        sparse_query_fn=sparse_fn,
    )


def build_reranker():
    """Cohere 優先，沒有 API key 則 fallback 到 flashrank（本地）"""
    if settings.cohere_api_key:
        from llama_index.postprocessor.cohere_rerank import CohereRerank
        return CohereRerank(
            api_key=settings.cohere_api_key,
            top_n=settings.rerank_top_n,
            model=settings.cohere_rerank_model,
        )
    from flashrank import Ranker, RerankRequest

    class FlashrankRerank(BaseNodePostprocessor):
        def _postprocess_nodes(
            self,
            nodes: List[NodeWithScore],
            query_bundle: Optional[QueryBundle] = None,
        ) -> List[NodeWithScore]:
            ranker = Ranker()
            query = query_bundle.query_str if query_bundle else ""
            passages = [{"id": i, "text": n.text} for i, n in enumerate(nodes)]
            request = RerankRequest(query=query, passages=passages)
            results = ranker.rerank(request)
            id_to_node = {i: n for i, n in enumerate(nodes)}
            reranked = [id_to_node[r["id"]] for r in results[: settings.rerank_top_n]]
            return reranked

    return FlashrankRerank()


CITATION_QA_TEMPLATE = PromptTemplate(
    "以下是從知識庫中擷取的相關段落，每段前標有編號 [n]。\n"
    "請根據這些段落回答問題。回答時用 [n] 標示引用來源編號（例如 [1][3]）。\n"
    "只能引用以下段落中實際存在的編號，不可自行創造不存在的編號。\n"
    "如果段落中沒有足夠資訊，請直接說明。\n\n"
    "段落：\n"
    "---------------------\n"
    "{context_str}\n"
    "---------------------\n"
    "問題：{query_str}\n"
    "回答："
)


def build_shared_components():
    """
    建立共用元件（retriever + reranker），可被多個 engine 複用。
    回傳 (fusion_retriever, reranker)
    """
    # Langfuse callback（有 key 才啟用）
    callback_handlers = []
    langfuse_cb = _build_langfuse_callback()
    if langfuse_cb:
        callback_handlers.append(langfuse_cb)
    callback_manager = CallbackManager(callback_handlers)
    Settings.callback_manager = callback_manager
    # 必須在 QueryFusionRetriever 建立前設好 Settings.llm，
    # 否則 LlamaIndex 會 fallback 嘗試載入 OpenAI。
    Settings.llm = build_llm_with_temperature(settings.llm_temperature)

    client = get_qdrant_client()
    aclient = get_async_qdrant_client()
    ensure_collection(client)
    vector_store = build_vector_store(client, aclient)

    embed_model = build_embedding()
    index = VectorStoreIndex.from_vector_store(
        vector_store=vector_store,
        embed_model=embed_model,
    )

    retriever = index.as_retriever(
        similarity_top_k=settings.similarity_top_k,
        vector_store_query_mode="hybrid",
        alpha=settings.hybrid_alpha,
    )

    fusion_retriever = QueryFusionRetriever(
        retrievers=[retriever],
        similarity_top_k=settings.similarity_top_k,
        num_queries=settings.num_queries,
        mode="reciprocal_rerank",
        use_async=True,
        verbose=True,
    )

    reranker = build_reranker()
    return fusion_retriever, reranker


class _CitationNumberingPostprocessor(BaseNodePostprocessor):
    """在 reranker 之後給每個 node 加上 [n] 前綴，
    使 context_str 裡的編號與參考來源列表一致。"""

    def _postprocess_nodes(
        self,
        nodes: List[NodeWithScore],
        query_bundle: Optional[QueryBundle] = None,
    ) -> List[NodeWithScore]:
        for i, node in enumerate(nodes, start=1):
            node.node.text = f"[{i}] {node.node.text}"
        return nodes


def build_engine(retriever, reranker, temperature: float, streaming: bool) -> "RetrieverQueryEngine":
    """用指定 temperature 建立 engine，retriever / reranker 可共用。"""
    llm = build_llm_with_temperature(temperature)
    if streaming:
        Settings.llm = llm  # /ask endpoint 需要 Settings.llm for streaming
    return RetrieverQueryEngine.from_args(
        retriever=retriever,
        node_postprocessors=[reranker, _CitationNumberingPostprocessor()],
        llm=llm,
        text_qa_template=CITATION_QA_TEMPLATE,
        streaming=streaming,
    )


def build_llm_with_temperature(temperature: float):
    """用指定 temperature 建立 LLM instance。"""
    llm_location = settings.google_cloud_llm_location
    if llm_location == "global":
        return _GenAIGlobalLLM(
            model_name=settings.llm_model,
            temperature=temperature,
            max_tokens_val=settings.llm_max_tokens,
        )
    import vertexai
    from llama_index.llms.vertex import Vertex
    vertexai.init(
        project=settings.google_cloud_project,
        location=llm_location,
    )
    return Vertex(
        model=settings.llm_model,
        project=settings.google_cloud_project,
        location=llm_location,
        credentials=_get_credentials(),
        temperature=temperature,
        max_tokens=settings.llm_max_tokens,
    )


def build_query_engine_pair() -> tuple["RetrieverQueryEngine", "RetrieverQueryEngine"]:
    """
    向後相容：建立預設 temperature 的 streaming + sync engine pair。
    """
    fusion_retriever, reranker = build_shared_components()
    engine_streaming = build_engine(fusion_retriever, reranker, settings.llm_temperature, streaming=True)
    engine_sync = build_engine(fusion_retriever, reranker, settings.llm_temperature, streaming=False)
    Settings.llm = engine_streaming.llm
    return engine_streaming, engine_sync


def build_query_engine(streaming: bool = True) -> RetrieverQueryEngine:
    """向後相容包裝，只需要單一 engine 時使用。"""
    streaming_engine, sync_engine = build_query_engine_pair()
    return streaming_engine if streaming else sync_engine
