import hashlib
from dataclasses import dataclass, field


def compute_md5(content: str) -> str:
    return hashlib.md5(content.encode()).hexdigest()


@dataclass
class DocumentSection:
    section_id: str      # stable ID, e.g. "redmine_issue_123::journal_456"
    section_type: str    # "header" | "description" | "journal" | "comment" | "checklist"
    text: str
    metadata: dict       # inherits doc-level metadata (source_type, source_id, etc.)
    md5: str             # compute_md5(text)


@dataclass
class SourceDocument:
    text: str
    metadata: dict = field(default_factory=dict)


@dataclass
class Chunk:
    node_id: str
    text: str
    embedding: list[float] | None = None
    metadata: dict = field(default_factory=dict)