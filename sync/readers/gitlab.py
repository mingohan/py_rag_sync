"""
GitLab Reader
Fetches Issues (with notes) and Wiki pages.
GITLAB_PROJECTS can be comma-separated namespace/project, e.g. "my-org/backend,my-org/frontend"

Issues use section-level embedding with timestamp pre-filtering.
Wiki pages use the existing full-document MD5 approach (no sections).
"""
import os
import httpx
from sync.models import SourceDocument, DocumentSection, compute_md5


def _get_client() -> tuple[str, dict]:
    base = os.environ["GITLAB_URL"].rstrip("/")
    token = os.environ["GITLAB_TOKEN"]
    return base, {"PRIVATE-TOKEN": token}


def _paginate(base: str, path: str, headers: dict, params: dict = {}) -> list[dict]:
    items = []
    page = 1
    while True:
        resp = httpx.get(
            f"{base}/api/v4/{path}",
            headers=headers,
            params={**params, "per_page": 100, "page": page},
            timeout=30,
        )
        resp.raise_for_status()
        batch = resp.json()
        if not batch:
            break
        items.extend(batch)
        if len(batch) < 100:
            break
        page += 1
    return items


def fetch_gitlab_issues_lightweight(project_path: str) -> list[dict]:
    """Fetch all issues (no notes) — returns list with updated_at for Layer 1 check"""
    base, headers = _get_client()
    encoded = project_path.replace("/", "%2F")
    issues = _paginate(base, f"projects/{encoded}/issues", headers, {"scope": "all", "state": "all"})
    result = []
    for issue in issues:
        result.append({
            "iid": issue["iid"],
            "updated_at": issue.get("updated_at"),
            "project_path": project_path,
            "issue": issue,  # full issue dict from list API (no notes)
        })
    return result


def fetch_gitlab_issue_full(project_path: str, issue: dict) -> SourceDocument:
    """Build full SourceDocument for an issue including notes"""
    base, headers = _get_client()
    encoded = project_path.replace("/", "%2F")
    iid = issue["iid"]

    # Fetch notes (comments)
    notes = []
    try:
        all_notes = _paginate(base, f"projects/{encoded}/issues/{iid}/notes", headers)
        notes = [n for n in all_notes if not n.get("system") and (n.get("body") or "").strip()]
    except Exception:
        pass

    return _build_gitlab_issue_doc(issue, notes, project_path, base)


def _build_gitlab_issue_doc(issue: dict, notes: list[dict], project_path: str, base: str) -> SourceDocument:
    source_id = f"gitlab_issue_{project_path}_{issue['iid']}"

    # --- header: title + meta line ---
    header_parts = [f"# {issue['title']}"]
    meta = []
    meta.append(f"#{issue['iid']}")
    if issue.get("state"):
        meta.append(f"State: {issue['state']}")
    if issue.get("issue_type"):
        meta.append(f"Type: {issue['issue_type']}")
    if issue.get("labels"):
        meta.append(f"Labels: {', '.join(issue['labels'])}")
    if issue.get("milestone"):
        meta.append(f"Milestone: {issue['milestone']['title']}")
    if issue.get("assignees"):
        meta.append(f"Assignees: {', '.join(a['name'] for a in issue['assignees'])}")
    if issue.get("author"):
        meta.append(f"Author: {issue['author']['name']}")
    if issue.get("due_date"):
        meta.append(f"Due: {issue['due_date']}")
    if issue.get("created_at"):
        meta.append(f"Created: {issue['created_at'][:10]}")
    if issue.get("updated_at"):
        meta.append(f"Updated: {issue['updated_at'][:10]}")
    if issue.get("closed_at"):
        meta.append(f"Closed: {issue['closed_at'][:10]}")
    if issue.get("closed_by"):
        meta.append(f"ClosedBy: {issue['closed_by']['name']}")
    if issue.get("confidential"):
        meta.append("Confidential: yes")
    ts = issue.get("time_stats", {})
    if ts.get("human_time_estimate"):
        meta.append(f"Estimate: {ts['human_time_estimate']}")
    if ts.get("human_total_time_spent"):
        meta.append(f"Spent: {ts['human_total_time_spent']}")
    tc = issue.get("task_completion_status", {})
    if tc.get("count", 0) > 0:
        meta.append(f"Tasks: {tc['completed_count']}/{tc['count']}")
    if meta:
        header_parts.append(" | ".join(meta))

    header_text = "\n\n".join(header_parts)

    # --- description ---
    description_text = (issue.get("description") or "").strip()

    # --- core text for context generation ---
    core_parts = [header_text]
    if description_text:
        core_parts.append(description_text)
    core_text = "\n\n".join(core_parts)

    # --- base metadata ---
    base_meta = {
        "source_type": "gitlab_issue",
        "source_id": source_id,
        "file_name": f"#{issue['iid']} {issue['title']}",
        "source_url": issue.get("web_url", ""),
        "source_updated_at": issue.get("updated_at"),
        "file_updated_at": issue.get("updated_at"),
        "state": issue.get("state", ""),
        "issue_type": issue.get("issue_type", ""),
        "labels": ", ".join(issue.get("labels", [])),
        "author": issue.get("author", {}).get("name", ""),
        "assignees": ", ".join(a["name"] for a in issue.get("assignees", [])),
        "milestone": issue.get("milestone", {}).get("title", "") if issue.get("milestone") else "",
        "due_date": issue.get("due_date", "") or "",
        "created_at": issue.get("created_at", "")[:10] if issue.get("created_at") else "",
        "closed_at": issue.get("closed_at", "")[:10] if issue.get("closed_at") else "",
        "project": project_path,
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

    for n in notes:
        author = n.get("author", {}).get("name", "Unknown")
        date = n.get("created_at", "")[:10] if n.get("created_at") else ""
        updated = n.get("updated_at", "")[:10] if n.get("updated_at") else ""
        date_str = date
        if updated and updated != date:
            date_str += f" (edited {updated})"
        note_text = f"**{author}** ({date_str})：{n['body']}"
        if len(note_text.strip()) < 20:
            continue
        sections.append(DocumentSection(
            section_id=f"{source_id}::comment_{n['id']}",
            section_type="comment",
            text=note_text,
            metadata=dict(base_meta),
            md5=compute_md5(note_text),
        ))

    # --- full text ---
    full_parts = list(core_parts)
    for section in sections:
        if section.section_type == "comment":
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


def fetch_gitlab_wiki_documents(project_path: str) -> list[SourceDocument]:
    """Wiki pages — full-document MD5 approach (no sections)"""
    base, headers = _get_client()
    encoded = project_path.replace("/", "%2F")
    pages = _paginate(base, f"projects/{encoded}/wikis", headers, {"with_content": 1})
    docs = []
    for page in pages:
        content = (page.get("content") or "").strip()
        if not content:
            continue
        slug = page.get("slug", "")
        title = page.get("title", slug)
        fmt = page.get("format", "markdown")
        project_url = f"{base}/{project_path}"

        parts = [f"# {title}"]
        if fmt and fmt != "markdown":
            parts.append(f"Format: {fmt}")
        parts.append(content)

        docs.append(SourceDocument(
            text="\n\n".join(parts),
            metadata={
                "source_type": "gitlab_wiki",
                "source_id": f"gitlab_wiki_{project_path}_{slug}",
                "file_name": title,
                "source_url": f"{project_url}/-/wikis/{slug}",
                "format": fmt,
                "project": project_path,
            },
        ))
    return docs


def fetch_gitlab_documents() -> list[SourceDocument]:
    """Legacy full-fetch function kept for backward compatibility"""
    base, headers = _get_client()
    projects = [p.strip() for p in os.environ.get("GITLAB_PROJECTS", "").split(",") if p.strip()]
    docs = []
    for project_path in projects:
        print(f"  [gitlab] project: {project_path}")
        try:
            items = fetch_gitlab_issues_lightweight(project_path)
            for item in items:
                try:
                    doc = fetch_gitlab_issue_full(project_path, item["issue"])
                    docs.append(doc)
                except Exception as e:
                    print(f"    issue {item['iid']} error: {e}")
            print(f"    issues: {len(items)}")
        except Exception as e:
            print(f"    issues error: {e}")
        try:
            wiki = fetch_gitlab_wiki_documents(project_path)
            print(f"    wiki: {len(wiki)}")
            docs.extend(wiki)
        except Exception as e:
            print(f"    wiki error: {e}")

    print(f"  [gitlab] fetched total {len(docs)} documents (issues + wikis)")
    return docs
