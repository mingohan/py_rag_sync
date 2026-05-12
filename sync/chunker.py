"""
Chunking strategies

Document-type (Drive / GitLab Wiki / PDF):
  SemanticSplitterNodeParser — finds semantic breakpoints via embedding similarity

Card-type (Redmine / GitLab Issues / Trello / Slack):
  No splitting — keeps full context intact

PDF:
  pymupdf4llm converts to Markdown → document-type flow
  Scanned PDF → Gemini Vision OCR → Markdown → document-type flow

Contextual Retrieval:
  After chunk_document / chunk_card, optionally call add_context_to_nodes()
  to prepend 50-100 token context to each chunk for improved search accuracy
"""
import hashlib
import re
import uuid
from llama_index.core.node_parser import SemanticSplitterNodeParser
from llama_index.core.schema import Document as LIDocument
from sync.models import SourceDocument, Chunk, DocumentSection

_SECTION_NS = uuid.UUID("a3e4d5c6-b7f8-4a9b-8c1d-2e3f4a5b6c7d")


def _section_to_node_id(section_id: str) -> str:
    """Convert string section_id to deterministic UUID for Qdrant"""
    return str(uuid.uuid5(_SECTION_NS, section_id))


MIN_CONTENT_LENGTH = 20


def is_low_quality(text: str) -> bool:
    stripped = text.strip()
    if len(stripped) < MIN_CONTENT_LENGTH:
        return True
    if re.fullmatch(r"[\d\s\-\—\=\*\_\|\.]+", stripped):
        return True
    return False


def chunk_document(doc: SourceDocument, embed_model) -> list[Chunk]:
    """Document-type: SemanticSplitterNodeParser splits at semantic breakpoints"""
    parser = SemanticSplitterNodeParser(
        embed_model=embed_model,
        buffer_size=1,
        breakpoint_percentile_threshold=85,
        embed_model_task_type="SEMANTIC_SIMILARITY",
    )
    li_doc = LIDocument(text=doc.text, metadata=doc.metadata)
    try:
        nodes = parser.get_nodes_from_documents([li_doc])
    except ValueError as e:
        # SemanticSplitter gets an empty/mismatched embedding from the API;
        # fall back to treating the whole document as one chunk.
        print(f"  [warn] semantic split failed ({e}), falling back to single chunk")
        return chunk_card(doc.text, doc.metadata)
    return [
        Chunk(node_id=n.node_id, text=n.text, metadata=dict(n.metadata))
        for n in nodes if not is_low_quality(n.text)
    ]


def chunk_pdf(file_path: str, metadata: dict, embed_model) -> list[Chunk]:
    """
    PDF: pymupdf4llm converts to Markdown then uses chunk_document.
    Scanned PDF (minimal text) → Gemini Vision OCR.
    """
    import pymupdf4llm

    md_text = pymupdf4llm.to_markdown(file_path, show_progress=False)

    if len(md_text.strip()) < 100:
        md_text = _ocr_pdf(file_path)

    doc = SourceDocument(text=md_text, metadata=metadata)
    return chunk_document(doc, embed_model)


def _get_genai_client():
    from google import genai
    from google.auth import default as google_auth_default
    from python.config import get_settings
    s = get_settings()
    creds, _ = google_auth_default(scopes=["https://www.googleapis.com/auth/cloud-platform"])
    return genai.Client(
        vertexai=True,
        project=s.google_cloud_project,
        location=s.google_cloud_llm_location,
        credentials=creds,
    )


def _ocr_pdf(file_path: str) -> str:
    import pymupdf
    import base64
    from google.genai import types as genai_types

    client = _get_genai_client()
    result_pages = []

    pdf = pymupdf.open(file_path)
    for page_num, page in enumerate(pdf):
        mat = pymupdf.Matrix(2.0, 2.0)
        pix = page.get_pixmap(matrix=mat)
        img_bytes = pix.tobytes("png")
        img_b64 = base64.b64encode(img_bytes).decode()

        from python.config import get_settings
        model = get_settings().llm_model
        response = client.models.generate_content(
            model=model,
            contents=[
                genai_types.Part.from_bytes(data=base64.b64decode(img_b64), mime_type="image/png"),
                "Transcribe all text content from this page completely, preserving paragraph structure. Convert tables to Markdown table format.",
            ],
        )
        result_pages.append(f"<!-- page {page_num + 1} -->\n{response.text}")

    pdf.close()
    return "\n\n".join(result_pages)


def chunk_card(content: str, metadata: dict) -> list[Chunk]:
    """Card-type (Redmine / GitLab Issues / Trello / Slack): no splitting"""
    if is_low_quality(content):
        return []
    node_id = str(uuid.uuid4())
    return [Chunk(node_id=node_id, text=content, metadata=dict(metadata))]


# Gemini embedding-2 limit is 8192 tokens.
# CJK ~2 chars/token → 8192 * 2 = ~16k chars; use 12k to stay safely under.
_MAX_SECTION_CHARS = 12000


def _split_long_text(text: str, max_chars: int) -> list[str]:
    """Split text at paragraph boundaries without exceeding max_chars."""
    paragraphs = text.split("\n\n")
    result = []
    current_parts: list[str] = []
    current_len = 0
    for para in paragraphs:
        para_len = len(para)
        if current_len + para_len > max_chars and current_parts:
            result.append("\n\n".join(current_parts))
            current_parts = [para]
            current_len = para_len
        else:
            current_parts.append(para)
            current_len += para_len + 2  # +2 for the "\n\n" separator
    if current_parts:
        result.append("\n\n".join(current_parts))
    return result or [text]


def chunk_sections(sections: list[DocumentSection]) -> list[Chunk]:
    """Section-level chunking with stable deterministic node IDs.

    Each section's metadata should already contain all doc-level fields
    (source_type, source_id, source_updated_at, etc.). This function adds
    section-specific fields (section_id, section_type, section_md5).

    Long sections (> _MAX_SECTION_CHARS) are split at paragraph boundaries into
    sub-chunks that share the same section_id and section_md5, so the diff logic
    in _sync_sections() treats them as a single logical unit.
    """
    chunks = []
    for section in sections:
        if is_low_quality(section.text):
            continue
        base_meta = {
            **section.metadata,
            "section_id": section.section_id,
            "section_type": section.section_type,
            "section_md5": section.md5,
        }
        if len(section.text) <= _MAX_SECTION_CHARS:
            chunks.append(Chunk(
                node_id=_section_to_node_id(section.section_id),
                text=section.text,
                metadata=base_meta,
            ))
        else:
            sub_texts = _split_long_text(section.text, _MAX_SECTION_CHARS)
            for i, sub_text in enumerate(sub_texts):
                if is_low_quality(sub_text):
                    continue
                chunks.append(Chunk(
                    # Sub-chunk IDs include index so each Qdrant point is unique,
                    # but section_id in metadata stays the same for group deletion.
                    node_id=_section_to_node_id(f"{section.section_id}::chunk_{i}"),
                    text=sub_text,
                    metadata=base_meta,
                ))
    return chunks


_CONTEXT_PROMPT = """\
Here is a complete document, followed by a chunk extracted from it.
In 1-2 sentences (50-100 tokens), describe this chunk's role and context within the overall document,
to help the search system find it more accurately. Output only the description, no prefix.

<document>
{doc_text}
</document>

<chunk>
{chunk_text}
</chunk>
"""


def add_context_to_nodes(
    nodes: list[Chunk],
    doc_text: str,
    chunk_context_cache: dict | None = None,
) -> list[Chunk]:
    """
    Contextual Retrieval: prepends context to each node.
    doc_text is the full document text before chunking.
    chunk_context_cache maps chunk_md5 → previously generated context_text to skip LLM calls.
    """
    from python.config import get_settings
    client = _get_genai_client()
    truncated_doc = doc_text[:30000] if len(doc_text) > 30000 else doc_text

    cache_hits = 0
    for node in nodes:
        chunk_md5 = hashlib.md5(node.text.encode()).hexdigest()
        node.metadata["chunk_md5"] = chunk_md5

        if chunk_context_cache is not None and chunk_md5 in chunk_context_cache:
            context = chunk_context_cache[chunk_md5]
            node.metadata["context_text"] = context
            node.text = f"{context}\n\n{node.text}"
            cache_hits += 1
            continue

        try:
            prompt = _CONTEXT_PROMPT.format(
                doc_text=truncated_doc,
                chunk_text=node.text,
            )
            response = client.models.generate_content(
                model=get_settings().llm_model,
                contents=prompt,
            )
            context = response.text.strip()
            if context:
                node.metadata["context_text"] = context
                node.text = f"{context}\n\n{node.text}"
        except Exception as e:
            print(f"  [contextual] failed for node {node.node_id}: {e}")

    if cache_hits:
        print(f"  [contextual] {cache_hits}/{len(nodes)} contexts reused from cache")
    return nodes