"""
Redmine Reader（自訂，LlamaHub 無官方支援）
邏輯對應原有 sync/redmine_client.rb
"""
import os
import httpx
from llama_index.core.schema import Document


def fetch_redmine_documents() -> list[Document]:
    base_url = os.environ["REDMINE_URL"].rstrip("/")
    api_key = os.environ["REDMINE_API_KEY"]
    headers = {"X-Redmine-API-Key": api_key}
    docs = []

    offset = 0
    limit = 100

    while True:
        resp = httpx.get(
            f"{base_url}/issues.json",
            headers=headers,
            params={
                "limit": limit,
                "offset": offset,
                "status_id": "*",
                "include": "journals,watchers,relations,attachments",
            },
            timeout=30,
        )
        resp.raise_for_status()
        data = resp.json()
        issues = data.get("issues", [])

        if not issues:
            break

        for full in issues:
            content_parts = [f"# {full['subject']}"]

            # --- meta line ---
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
                content_parts.append(" | ".join(meta))

            # --- description ---
            if (full.get("description") or "").strip():
                content_parts.append(full["description"])

            # --- custom fields ---
            custom_fields = full.get("custom_fields", [])
            non_empty_cf = [
                cf for cf in custom_fields
                if cf.get("value") not in (None, "", [])
            ]
            if non_empty_cf:
                content_parts.append("## 自訂欄位")
                for cf in non_empty_cf:
                    val = cf["value"]
                    if isinstance(val, list):
                        val = ", ".join(str(v) for v in val)
                    content_parts.append(f"- {cf['name']}: {val}")

            # --- watchers ---
            watchers = full.get("watchers", [])
            if watchers:
                names = ", ".join(w["name"] for w in watchers)
                content_parts.append(f"## 關注者\n{names}")

            # --- relations ---
            relations = full.get("relations", [])
            if relations:
                content_parts.append("## 關聯 Issue")
                for r in relations:
                    other_id = r["issue_to_id"] if r["issue_id"] == full["id"] else r["issue_id"]
                    rel_type = r.get("relation_type", "")
                    delay = f"（延遲 {r['delay']}d）" if r.get("delay") else ""
                    content_parts.append(f"- {rel_type}: #{other_id}{delay}")

            # --- attachments ---
            attachments = full.get("attachments", [])
            if attachments:
                content_parts.append("## 附件")
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
                    content_parts.append(line)

            # --- journals（留言 + 變更紀錄）---
            journals = full.get("journals", [])
            if journals:
                content_parts.append("## 留言與紀錄")
                for j in journals:
                    author = j.get("user", {}).get("name", "Unknown")
                    date = j.get("created_on", "")[:10] if j.get("created_on") else ""
                    notes = (j.get("notes") or "").strip()
                    details = j.get("details", [])

                    if notes:
                        content_parts.append(f"**{author}** ({date})：{notes}")

                    for d in details:
                        prop = d.get("property", "")
                        name = d.get("name", "")
                        old_val = d.get("old_value") or ""
                        new_val = d.get("new_value") or ""
                        if prop == "attr":
                            # 跳過 description diff（太長）
                            if name == "description":
                                content_parts.append(f"  ↳ [{author} {date}] 更新了 description")
                            else:
                                content_parts.append(f"  ↳ [{author} {date}] {name}: {old_val!r} → {new_val!r}")
                        elif prop == "cf":
                            content_parts.append(f"  ↳ [{author} {date}] 自訂欄位 {name}: {old_val!r} → {new_val!r}")
                        elif prop == "relation":
                            action = "新增" if not old_val else "移除"
                            content_parts.append(f"  ↳ [{author} {date}] {action}關聯 {name}: #{new_val or old_val}")
                        elif prop == "attachment":
                            content_parts.append(f"  ↳ [{author} {date}] 附件異動: {new_val or old_val}")

            doc = Document(
                text="\n\n".join(content_parts),
                metadata={
                    "source_type": "redmine",
                    "source_id": f"redmine_issue_{full['id']}",
                    "file_name": f"#{full['id']} {full['subject']}",
                    "source_url": f"{base_url}/issues/{full['id']}",
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
                },
                doc_id=f"redmine_{full['id']}",
            )
            docs.append(doc)

        offset += limit
        if offset >= data.get("total_count", 0):
            break

    print(f"  [redmine] fetched {len(docs)} issues")
    return docs
