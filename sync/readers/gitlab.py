"""
GitLab Reader
抓取 Issues（含 notes）和 Wiki pages
GITLAB_PROJECTS 可以是逗號分隔的多個 namespace/project，e.g. "metis/nerv,foo/bar"
"""
import os
import httpx
from llama_index.core.schema import Document


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


def _fetch_issues(base: str, headers: dict, project_path: str) -> list[Document]:
    encoded = project_path.replace("/", "%2F")
    issues = _paginate(base, f"projects/{encoded}/issues", headers, {"scope": "all", "state": "all"})
    docs = []
    for issue in issues:
        parts = [f"# {issue['title']}"]

        # --- meta line ---
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

        # time stats
        ts = issue.get("time_stats", {})
        if ts.get("human_time_estimate"):
            meta.append(f"Estimate: {ts['human_time_estimate']}")
        if ts.get("human_total_time_spent"):
            meta.append(f"Spent: {ts['human_total_time_spent']}")

        # task checklist
        tc = issue.get("task_completion_status", {})
        if tc.get("count", 0) > 0:
            meta.append(f"Tasks: {tc['completed_count']}/{tc['count']}")

        if meta:
            parts.append(" | ".join(meta))

        # --- description ---
        if (issue.get("description") or "").strip():
            parts.append(issue["description"])

        # --- notes（留言，排除系統訊息）---
        try:
            notes = _paginate(base, f"projects/{encoded}/issues/{issue['iid']}/notes", headers)
            notes = [n for n in notes if not n.get("system") and (n.get("body") or "").strip()]
            if notes:
                parts.append("## 留言")
                for n in notes:
                    author = n.get("author", {}).get("name", "Unknown")
                    date = n.get("created_at", "")[:10] if n.get("created_at") else ""
                    updated = n.get("updated_at", "")[:10] if n.get("updated_at") else ""
                    date_str = date
                    if updated and updated != date:
                        date_str += f" (edited {updated})"
                    parts.append(f"**{author}** ({date_str})：{n['body']}")
        except Exception:
            pass

        docs.append(Document(
            text="\n\n".join(parts),
            metadata={
                "source_type": "gitlab_issue",
                "source_id": f"gitlab_issue_{project_path}_{issue['iid']}",
                "file_name": f"#{issue['iid']} {issue['title']}",
                "source_url": issue.get("web_url", ""),
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
            },
        ))
    return docs


def _fetch_wiki(base: str, headers: dict, project_path: str) -> list[Document]:
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

        docs.append(Document(
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


def fetch_gitlab_documents() -> list[Document]:
    base, headers = _get_client()
    projects = [p.strip() for p in os.environ.get("GITLAB_PROJECTS", "").split(",") if p.strip()]
    docs = []
    for project_path in projects:
        print(f"  [gitlab] project: {project_path}")
        try:
            issues = _fetch_issues(base, headers, project_path)
            print(f"    issues: {len(issues)}")
            docs.extend(issues)
        except Exception as e:
            print(f"    issues error: {e}")
        try:
            wiki = _fetch_wiki(base, headers, project_path)
            print(f"    wiki: {len(wiki)}")
            docs.extend(wiki)
        except Exception as e:
            print(f"    wiki error: {e}")
    
    print(f"  [gitlab] fetched total {len(docs)} documents (issues + wikis)")
    return docs
