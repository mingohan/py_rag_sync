"""
Embedding：Vertex AI gemini-embedding-2-preview（via google-genai）
- Dense embedding：google-genai Vertex AI backend（RETRIEVAL_DOCUMENT）
- Sparse embedding：fastembed BM25（本地執行，不需 API）
"""
import time
import os
from google import genai
from google.genai import types as genai_types
from google.oauth2 import service_account
from fastembed import SparseTextEmbedding
from llama_index.core.schema import TextNode
from python.config import get_settings

settings = get_settings()

BATCH_SIZE = 10
MAX_RETRIES = 5
RETRY_DELAY = 60

# Model singleton（避免每次重新載入）
_sparse_model = None
_genai_client = None


def _get_sparse_model() -> SparseTextEmbedding:
    global _sparse_model
    if _sparse_model is None:
        _sparse_model = SparseTextEmbedding(model_name=settings.sparse_model)
    return _sparse_model


def _get_genai_client() -> genai.Client:
    global _genai_client
    if _genai_client is None:
        sa_path = os.environ.get(
            "GOOGLE_APPLICATION_CREDENTIALS",
            settings.google_application_credentials,
        )
        creds = service_account.Credentials.from_service_account_file(
            sa_path,
            scopes=["https://www.googleapis.com/auth/cloud-platform"],
        )
        _genai_client = genai.Client(
            vertexai=True,
            project=settings.google_cloud_project,
            location=settings.google_cloud_location,
            credentials=creds,
        )
    return _genai_client


def embed_nodes(nodes: list[TextNode]) -> list[TextNode]:
    """
    批次 embed，同時產生：
    - node.embedding（Dense，送 Vertex AI）
    - node.metadata["sparse_embedding"]（Sparse BM25，本地）
    """
    client = _get_genai_client()
    sparse_model = _get_sparse_model()

    for i in range(0, len(nodes), BATCH_SIZE):
        batch = nodes[i:i + BATCH_SIZE]
        texts = [n.text for n in batch]

        # Dense embedding（Vertex AI）
        dense_results = _embed_batch(client, texts, task_type="RETRIEVAL_DOCUMENT")

        # Sparse embedding（本地 BM25，不需要 API）
        sparse_results = list(sparse_model.embed(texts))

        for node, dense, sparse in zip(batch, dense_results, sparse_results):
            node.embedding = dense.values
            node.metadata["sparse_indices"] = sparse.indices.tolist()
            node.metadata["sparse_values"] = sparse.values.tolist()

        print(f"  embedded {min(i + BATCH_SIZE, len(nodes))}/{len(nodes)} nodes")

    return nodes


def embed_query(text: str) -> tuple[list[float], dict]:
    """查詢時同時產生 dense + sparse embedding"""
    client = _get_genai_client()
    sparse_model = _get_sparse_model()

    dense_results = _embed_batch(client, [text], task_type="RETRIEVAL_QUERY")
    sparse_result = list(sparse_model.embed([text]))[0]

    return (
        dense_results[0].values,
        {
            "indices": sparse_result.indices.tolist(),
            "values": sparse_result.values.tolist(),
        }
    )


def _embed_batch(client: genai.Client, texts: list[str], task_type: str):
    for attempt in range(MAX_RETRIES):
        try:
            result = client.models.embed_content(
                model=settings.embedding_model,
                contents=texts,
                config=genai_types.EmbedContentConfig(
                    task_type=task_type,
                    output_dimensionality=settings.embedding_dimensions,
                ),
            )
            return result.embeddings
        except Exception as e:
            if attempt < MAX_RETRIES - 1:
                print(f"  embed failed ({e}), retry {attempt + 1}...")
                wait = RETRY_DELAY * (attempt + 1)
                print(f"  waiting {wait}s before retry...")
                time.sleep(wait)
            else:
                raise
