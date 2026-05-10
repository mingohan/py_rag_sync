"""
Redmine Reader (custom, LlamaHub has no official support)
Logic corresponds to the original sync/redmine_client.rb

Exposes two APIs:
  fetch_redmine_lightweight()   — all issue ids + updated_on (no journals)
  fetch_redmine_issue(id)       — single issue with full details + sections
"""
import os
import httpx
from sync.models import SourceDocument, DocumentSection, compute_md5


def fetch_redmine_lightweight() -> list[dict]:
    """Fetch all issues without journals — returns list of {id, updated_on}"""
    base_url = os.environ["REDMINE_URL"].rstrip("/")
    api_key = os.environ["REDMINE_API_KEY"]
    headers = {"X-Redmine-API-Key": api_key}
    items = []
    offset = 0
    limit = 100
    while True:
        resp = httpx.get(
            f"{base_url}/issues.json",
            headers=headers,
            params={"limit": limit, "offset": offset, "status_id": "*"},
            timeout=30,
        )
        resp.raise_for_status()
        data = resp.json()
        issues = data.get("issues", [])
        if not issues:
            break
        for issue in issues:
            items.append({"id": issue["id"], "updated_on": issue.get("updated_on")})
        offset += limit
        if offset >= data.get("total_count", 0):
            break
    print(f"  [redmine] found {len(items)} issues (lightweight)")
    return items


def fetch_redmine_issue(issue_id: int) -> SourceDocument:
    """Fetch a single issue with full details including journals"""
    base_url = os.environ["REDMINE_URL"].rstrip("/")
    api_key = os.environ["REDMINE_API_KEY"]
    headers = {"X-Redmine-API-Key": api_key}
    resp = httpx.get(
        f"{base_url}/issues/{issue_id}.json",
        headers=headers,
        params={"include": "journals,watchers,relations,attachments"},
        timeout=30,
    )
    resp.raise_for_status()
    return _build_redmine_doc(resp.json()["issue"], base_url)


def _build_redmine_doc(full: dict, base_url: str) -> SourceDocument:
    source_id = f"redmine_issue_{full['id']}"

    # --- header: title + meta line + stable structured data ---
    header_parts = [f"# {full['subject']}"]

    meta = []
    if full.get("project"):
        meta.append(f"Project: {full['project']['name']}")
    if full.get("tracker"):
        meta.append(f"Tracker: {full['tracker']['name']}")
    if full.get("status"):
        closed = " (closed)" if full["status"].get("is_closed") else ""
        meta.append(f"Status: {full['status']['name']}{closed}")
    if full.get("priority"):
        meta.append(f"Priority: {full['priority']['name']}")
    if full.get("assigned_to"):
        meta.append(f"Assignee: {full['assigned_to']['name']}")
    if full.get("author"):
        meta.append(f"Author: {full['author']['name']}")
    if full.get("start_date"):
        meta.append(f"Start: {full['start_date']}")
    if full.get("due_date"):
        meta.append(f"Due: {full['due_date']}")
    if full.get("done_ratio") is not None:
        meta.append(f"Done: {full['done_ratio']}%")
    if full.get("estimated_hours") is not None:
        meta.append(f"Estimated: {full['estimated_hours']}h")
    if full.get("spent_hours") is not None:
        meta.append(f"Spent: {full['spent_hours']}h")
    if full.get("created_on"):
        meta.append(f"Created: {full['created_on'][:10]}")
    if full.get("updated_on"):
        meta.append(f"Updated: {full['updated_on'][:10]}")
    if full.get("closed_on"):
        meta.append(f"Closed: {full['closed_on'][:10]}")
    if meta:
        header_parts.append(" | ".join(meta))

    # Custom fields
    custom_fields = full.get("custom_fields", [])
    non_empty_cf = [cf for cf in custom_fields if cf.get("value") not in (None, "", [])]
    if non_empty_cf:
        header_parts.append("## Custom Fields")
        for cf in non_empty_cf:
            val = cf["value"]
            if isinstance(val, list):
                val = ", ".join(str(v) for v in val)
            header_parts.append(f"- {cf['name']}: {val}")

    # Watchers
    watchers = full.get("watchers", [])
    if watchers:
        names = ", ".join(w["name"] for w in watchers)
        header_parts.append(f"## Watchers\n{names}")

    # Relations
    relations = full.get("relations", [])
    if relations:
        header_parts.append("## Relations")
        for r in relations:
            other_id = r["issue_to_id"] if r["issue_id"] == full["id"] else r["issue_id"]
            rel_type = r.get("relation_type", "")
            delay = f" (delayed {r['delay']}d)" if r.get("delay") else ""
            header_parts.append(f"- {rel_type}: #{other_id}{delay}")

    # Attachments
    attachments = full.get("attachments", [])
    if attachments:
        header_parts.append("## Attachments")
        for att in attachments:
            name = att.get("filename", "")
            size = att.get("filesize", "")
            mime = att.get("content_type", "")
            url = att.get("content_url", "")
            author = att.get("author", {}).get("name", "")
            date = att.get("created_on", "")[:10] if att.get("created_on") else ""
            desc = att.get("description", "")
            line = f"- {name}"
            if mime:
                line += f" ({mime})"
            if size:
                line += f" {size}B"
            if author:
                line += f" by {author}"
            if date:
                line += f" [{date}]"
            if desc:
                line += f" — {desc}"
            if url:
                line += f" {url}"
            header_parts.append(line)

    header_text = "\n\n".join(header_parts)

    # --- description ---
    description_text = (full.get("description") or "").strip()

    # --- core text for context generation (stable content without journals) ---
    core_parts = [header_text]
    if description_text:
        core_parts.append(description_text)
    core_text = "\n\n".join(core_parts)

    # --- base metadata shared by all sections ---
    base_meta = {
        "source_type": "redmine",
        "source_id": source_id,
        "file_name": f"#{full['id']} {full['subject']}",
        "source_url": f"{base_url}/issues/{full['id']}",
        "source_updated_at": full.get("updated_on"),
        "file_updated_at": full.get("updated_on"),
        "project": full.get("project", {}).get("name", ""),
        "tracker": full.get("tracker", {}).get("name", ""),
        "status": full.get("status", {}).get("name", ""),
        "priority": full.get("priority", {}).get("name", ""),
        "assigned_to": full.get("assigned_to", {}).get("name", ""),
        "author": full.get("author", {}).get("name", ""),
        "start_date": full.get("start_date", "") or "",
        "due_date": full.get("due_date", "") or "",
        "done_ratio": str(full.get("done_ratio", "")),
        "created_on": full.get("created_on", "")[:10] if full.get("created_on") else "",
        "closed_on": full.get("closed_on", "")[:10] if full.get("closed_on") else "",
    }

    # --- build sections ---
    sections: list[DocumentSection] = []

    if len(header_text.strip()) >= 20:
        sections.append(DocumentSection(
            section_id=f"{source_id}::header",
            section_type="header",
            text=header_text,
            metadata=dict(base_meta),
            md5=compute_md5(header_text),
        ))

    if description_text and len(description_text.strip()) >= 20:
        sections.append(DocumentSection(
            section_id=f"{source_id}::description",
            section_type="description",
            text=description_text,
            metadata=dict(base_meta),
            md5=compute_md5(description_text),
        ))

    for j in full.get("journals", []):
        notes = (j.get("notes") or "").strip()
        if not notes or len(notes) < 5:
            continue
        author = j.get("user", {}).get("name", "Unknown")
        date = j.get("created_on", "")[:10] if j.get("created_on") else ""
        journal_text = f"**{author}** ({date})：{notes}"
        if len(journal_text.strip()) < 20:
            continue
        sections.append(DocumentSection(
            section_id=f"{source_id}::journal_{j['id']}",
            section_type="journal",
            text=journal_text,
            metadata=dict(base_meta),
            md5=compute_md5(journal_text),
        ))

    # --- full text for context generation ---
    full_parts = list(core_parts)
    for section in sections:
        if section.section_type == "journal":
            full_parts.append(section.text)
    full_text = "\n\n".join(full_parts)

    return SourceDocument(
        text=full_text,
        metadata={
            **base_meta,
            "sections": sections,
            "core_text": core_text,
        },
    )
