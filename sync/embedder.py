"""
Embedding: Vertex AI gemini-embedding-2-preview (via google-genai)
- Dense embedding: GeminiVertexEmbedding from python.pipeline
- Sparse embedding: fastembed BM25 (local, no API needed)
"""
from fastembed import SparseTextEmbedding
from sync.models import Chunk
from python.config import get_settings

settings = get_settings()

BATCH_SIZE = 10

_sparse_model = None
_embedding_model = None


def _get_sparse_model() -> SparseTextEmbedding:
    global _sparse_model
    if _sparse_model is None:
        _sparse_model = SparseTextEmbedding(model_name=settings.sparse_model)
    return _sparse_model


def _get_embedding_model():
    from python.pipeline import GeminiVertexEmbedding
    global _embedding_model
    if _embedding_model is None:
        _embedding_model = GeminiVertexEmbedding(embed_batch_size=10)
    return _embedding_model


def embed_nodes(nodes: list[Chunk]) -> list[Chunk]:
    """
    Batch embed, producing both:
    - node.embedding (Dense, via Vertex AI)
    - node.metadata["sparse_indices"] / ["sparse_values"] (Sparse BM25, local)
    """
    emb = _get_embedding_model()
    sparse_model = _get_sparse_model()

    for i in range(0, len(nodes), BATCH_SIZE):
        batch = nodes[i:i + BATCH_SIZE]
        texts = [n.text for n in batch]

        dense_vecs = emb.get_text_embeddings(texts)
        sparse_results = list(sparse_model.embed(texts))

        for node, dense, sparse in zip(batch, dense_vecs, sparse_results):
            node.embedding = dense
            node.metadata["sparse_indices"] = sparse.indices.tolist()
            node.metadata["sparse_values"] = sparse.values.tolist()

        print(f"  embedded {min(i + BATCH_SIZE, len(nodes))}/{len(nodes)} nodes")

    return nodes


def embed_query(text: str) -> tuple[list[float], dict]:
    """Generate both dense and sparse embeddings for a query"""
    emb = _get_embedding_model()
    sparse_model = _get_sparse_model()

    dense = emb.get_query_embedding(text)
    sparse_result = list(sparse_model.embed([text]))[0]

    return (
        dense,
        {
            "indices": sparse_result.indices.tolist(),
            "values": sparse_result.values.tolist(),
        }
    )
