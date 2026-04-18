"""
Chunking 策略

文件類（Drive / GitLab Wiki / PDF）：
  SemanticSplitterNodeParser — 用 embedding 相似度找語意斷點，切出語意完整的段落

卡片類（Redmine / GitLab Issues / Trello / Slack）：
  整筆不切，保留完整脈絡

PDF：
  pymupdf4llm 先轉 Markdown → 走文件類流程
  掃描 PDF → Gemini Vision OCR → Markdown → 文件類流程

Contextual Retrieval（#10）：
  chunk_document / chunk_card 切好後，可選呼叫 add_context_to_nodes()
  為每個 chunk prepend 50-100 token 的背景說明，改善語意搜尋準確率
"""
import re
from llama_index.core.node_parser import SemanticSplitterNodeParser
from llama_index.core.schema import Document, TextNode


MIN_CONTENT_LENGTH = 20


def is_low_quality(text: str) -> bool:
    stripped = text.strip()
    if len(stripped) < MIN_CONTENT_LENGTH:
        return True
    if re.fullmatch(r"[\d\s\-\—\=\*\_\|\.]+", stripped):
        return True
    return False


def chunk_document(doc: Document, embed_model) -> list[TextNode]:
    """文件類：SemanticSplitterNodeParser 按語意斷點切割"""
    parser = SemanticSplitterNodeParser(
        embed_model=embed_model,
        buffer_size=1,  # 避免 buffer_size 過大導致 Token 暴增引發 Vertex AI 429 Quota Error
        breakpoint_percentile_threshold=85,  # 降低切點門檻 (從 95 -> 85)，產生較大、較完整的語意段落
        embed_model_task_type="SEMANTIC_SIMILARITY",
    )
    nodes = parser.get_nodes_from_documents([doc])
    return [n for n in nodes if not is_low_quality(n.text)]


def chunk_pdf(file_path: str, metadata: dict, embed_model) -> list[TextNode]:
    """
    PDF：pymupdf4llm 轉 Markdown 後走 chunk_document。
    掃描 PDF（文字量極少）→ Gemini Vision OCR。
    """
    import pymupdf4llm

    md_text = pymupdf4llm.to_markdown(file_path, show_progress=False)

    if len(md_text.strip()) < 100:
        md_text = _ocr_pdf(file_path)

    doc = Document(text=md_text, metadata=metadata)
    return chunk_document(doc, embed_model)


def _get_genai_client():
    import os
    from google import genai
    from google.oauth2 import service_account
    from python.config import get_settings
    s = get_settings()
    sa_path = os.environ.get("GOOGLE_APPLICATION_CREDENTIALS", s.google_application_credentials)
    creds = service_account.Credentials.from_service_account_file(
        sa_path,
        scopes=["https://www.googleapis.com/auth/cloud-platform"],
    )
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
                "請將這頁的所有文字內容完整轉錄，保留段落結構。如有表格請轉成 Markdown 表格格式。",
            ],
        )
        result_pages.append(f"<!-- page {page_num + 1} -->\n{response.text}")

    pdf.close()
    return "\n\n".join(result_pages)


def chunk_card(content: str, metadata: dict) -> list[TextNode]:
    """卡片類（Redmine / GitLab Issues / Trello / Slack）：整筆不切"""
    if is_low_quality(content):
        return []
    node = TextNode(text=content, metadata=metadata)
    return [node]


_CONTEXT_PROMPT = """\
以下是一份完整文件，之後是從中擷取的一個段落。
請用 1-2 句話（50-100 個 token）說明這個段落在整份文件中的角色與背景，
幫助搜尋系統更準確地找到它。只輸出說明本身，不要加任何前綴。

<document>
{doc_text}
</document>

<chunk>
{chunk_text}
</chunk>
"""


def add_context_to_nodes(nodes: list[TextNode], doc_text: str) -> list[TextNode]:
    """
    Contextual Retrieval：為每個 node prepend 背景說明。
    doc_text 是切割前的完整文件文字。
    """
    from python.config import get_settings
    client = _get_genai_client()
    # 大幅放寬限制，充分利用 Gemini 破百萬 Token 的 context window (30萬字元約為 10萬 Token)
    truncated_doc = doc_text[:300000] if len(doc_text) > 300000 else doc_text

    for node in nodes:
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
                node.text = f"{context}\n\n{node.text}"
        except Exception as e:
            print(f"  [contextual] failed for node {node.node_id}: {e}")

    return nodes
