"""
Asana Reader
Fetches tasks from specified projects (with comments, subtasks, custom fields)
ASANA_PROJECT_IDS is comma-separated project GIDs

Tasks use section-level embedding with timestamp pre-filtering.
"""
import os
import httpx
from sync.models import SourceDocument, DocumentSection, compute_md5


def _get_headers() -> dict:
    return {"Authorization": f"Bearer {os.environ['ASANA_ACCESS_TOKEN']}"}


def _get(path: str, params: dict = {}) -> dict:
    resp = httpx.get(
        f"https://app.asana.com/api/1.0/{path}",
        headers=_get_headers(),
        params=params,
        timeout=30,
    )
    resp.raise_for_status()
    return resp.json()


def _paginate(path: str, params: dict = {}) -> list[dict]:
    """Fetch all pages from an Asana paginated endpoint."""
    results = []
    current_params = {**params, "limit": 100}
    while True:
        data = _get(path, current_params)
        results.extend(data.get("data", []))
        next_page = data.get("next_page")
        if not next_page:
            break
        current_params = {**params, "limit": 100, "offset": next_page["offset"]}
    return results


def fetch_asana_lightweight() -> list[dict]:
    """
    Fetch all tasks from all projects with just gid + modified_at.
    Returns list of {id, modified_at, project_context} for Layer 1 check.
    """
    project_ids = [p.strip() for p in os.environ.get("ASANA_PROJECT_IDS", "").split(",") if p.strip()]
    items = []
    for project_gid in project_ids:
        try:
            project = _get(f"projects/{project_gid}", {"opt_fields": "gid,name,permalink_url"})["data"]
            project_context = {
                "project_gid": project_gid,
                "project_name": project.get("name", project_gid),
                "project_url": project.get("permalink_url", ""),
            }
            print(f"  [asana] project: {project_context['project_name']}")
            tasks = _paginate(f"projects/{project_gid}/tasks", {"opt_fields": "gid,modified_at"})
            for task in tasks:
                items.append({
                    "id": task["gid"],
                    "modified_at": task.get("modified_at"),
                    "project_context": project_context,
                })
        except Exception as e:
            print(f"  [asana] project {project_gid} lightweight error: {e}")
    print(f"  [asana] found {len(items)} tasks (lightweight)")
    return items


def fetch_asana_task_full(task_gid: str, project_context: dict) -> SourceDocument:
    """Fetch a single task with full details and build SourceDocument with sections."""
    task = _get(f"tasks/{task_gid}", {
        "opt_fields": (
            "gid,name,notes,completed,assignee.name,due_on,projects.name,"
            "custom_fields.name,custom_fields.display_value,"
            "modified_at,created_at,permalink_url,memberships.section.name,tags.name"
        ),
    })["data"]

    stories = _paginate(f"tasks/{task_gid}/stories", {
        "opt_fields": "gid,created_at,created_by.name,type,text",
    })
    comments = [s for s in stories if s.get("type") == "comment"]

    subtasks = _paginate(f"tasks/{task_gid}/subtasks", {
        "opt_fields": "gid,name,notes,completed,assignee.name,due_on",
    })

    return _build_asana_task_doc(task, comments, subtasks, project_context)


def _build_asana_task_doc(
    task: dict,
    comments: list[dict],
    subtasks: list[dict],
    project_context: dict,
) -> SourceDocument:
    project_name = project_context["project_name"]
    task_gid = task["gid"]
    source_id = f"asana_task_{task_gid}"

    assignee = (task.get("assignee") or {}).get("name", "")
    section_name = ""
    memberships = task.get("memberships") or []
    if memberships:
        section_name = ((memberships[0].get("section") or {}).get("name") or "")
    tags = [t["name"] for t in (task.get("tags") or []) if t.get("name")]

    # --- header: title + meta line ---
    header_parts = [f"# {task['name']}"]
    meta = [f"Project: {project_name}"]
    if section_name:
        meta.append(f"Section: {section_name}")
    if assignee:
        meta.append(f"Assignee: {assignee}")
    if task.get("due_on"):
        meta.append(f"Due: {task['due_on']}")
    meta.append("Status: completed" if task.get("completed") else "Status: active")
    if tags:
        meta.append(f"Tags: {', '.join(tags)}")
    if task.get("modified_at"):
        meta.append(f"LastModified: {task['modified_at'][:10]}")
    header_parts.append(" | ".join(meta))

    custom_fields = [cf for cf in (task.get("custom_fields") or []) if cf.get("display_value")]
    if custom_fields:
        header_parts.append("## Custom Fields")
        for cf in custom_fields:
            header_parts.append(f"- {cf['name']}: {cf['display_value']}")

    header_text = "\n\n".join(header_parts)

    # --- description ---
    description_text = (task.get("notes") or "").strip()

    # --- core text for context generation ---
    core_parts = [header_text]
    if description_text:
        core_parts.append(description_text)
    core_text = "\n\n".join(core_parts)

    # --- base metadata ---
    base_meta = {
        "source_type": "asana",
        "source_id": source_id,
        "file_name": f"{project_name} / {task['name']}",
        "source_url": task.get("permalink_url", ""),
        "source_updated_at": task.get("modified_at"),
        "project_name": project_name,
        "section_name": section_name,
        "assignee": assignee,
        "due_on": task.get("due_on", ""),
        "completed": str(task.get("completed", False)),
        "tags": ", ".join(tags),
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

    if subtasks:
        subtask_parts = ["## Subtasks"]
        for st in subtasks:
            checked = "x" if st.get("completed") else " "
            line = f"- [{checked}] {st['name']}"
            st_assignee = (st.get("assignee") or {}).get("name", "")
            if st_assignee:
                line += f" (@{st_assignee})"
            if st.get("due_on"):
                line += f" [due: {st['due_on']}]"
            subtask_parts.append(line)
            st_notes = (st.get("notes") or "").strip()
            if len(st_notes) >= 20:
                subtask_parts.append(f"  {st_notes}")
        subtask_text = "\n".join(subtask_parts)
        if len(subtask_text.strip()) >= 20:
            sections.append(DocumentSection(
                section_id=f"{source_id}::subtasks",
                section_type="subtasks",
                text=subtask_text,
                metadata=dict(base_meta),
                md5=compute_md5(subtask_text),
            ))

    for story in comments:
        author = (story.get("created_by") or {}).get("name", "Unknown")
        text = (story.get("text") or "").strip()
        date = story.get("created_at", "")[:10] if story.get("created_at") else ""
        if not text:
            continue
        comment_text = f"**{author}** ({date})：{text}"
        if len(comment_text.strip()) < 20:
            continue
        sections.append(DocumentSection(
            section_id=f"{source_id}::comment_{story['gid']}",
            section_type="comment",
            text=comment_text,
            metadata=dict(base_meta),
            md5=compute_md5(comment_text),
        ))

    # --- full text ---
    full_parts = list(core_parts)
    for section in sections:
        if section.section_type in ("subtasks", "comment"):
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


def fetch_asana_documents() -> list[SourceDocument]:
    """Legacy full-fetch function kept for backward compatibility"""
    items = fetch_asana_lightweight()
    docs = []
    for item in items:
        try:
            doc = fetch_asana_task_full(item["id"], item["project_context"])
            docs.append(doc)
        except Exception as e:
            print(f"  [asana] task {item['id']} error: {e}")
    return docs
