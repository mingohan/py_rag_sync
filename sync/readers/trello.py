"""
Trello Reader
Fetches cards from multiple boards (with checklists, comments, attachments, members)
TRELLO_BOARD_IDS is comma-separated

Cards use section-level embedding with timestamp pre-filtering.
"""
import os
import httpx
from sync.models import SourceDocument, DocumentSection, compute_md5


def _get_auth() -> dict:
    return {
        "key": os.environ["TRELLO_API_KEY"],
        "token": os.environ["TRELLO_TOKEN"],
    }


def _get(path: str, params: dict = {}) -> dict | list:
    auth = _get_auth()
    resp = httpx.get(
        f"https://api.trello.com/1/{path}",
        params={**auth, **params},
        timeout=30,
    )
    resp.raise_for_status()
    return resp.json()


def _fetch_board_context(board_id: str) -> dict:
    """Fetch board-level metadata needed to build card content"""
    board = _get(f"boards/{board_id}", {"fields": "name,shortUrl"})
    lists = _get(f"boards/{board_id}/lists", {"fields": "id,name"})
    members = _get(f"boards/{board_id}/members", {"fields": "id,fullName,username"})
    return {
        "board_id": board_id,
        "board_name": board.get("name", board_id),
        "board_url": board.get("shortUrl", ""),
        "list_names": {l["id"]: l["name"] for l in lists},
        "member_names": {m["id"]: m["fullName"] for m in members},
    }


def fetch_trello_lightweight() -> list[dict]:
    """
    Fetch all cards from all boards with just id + dateLastActivity.
    Returns list of {id, dateLastActivity, board_context} for Layer 1 check.
    """
    board_ids = [b.strip() for b in os.environ.get("TRELLO_BOARD_IDS", "").split(",") if b.strip()]
    items = []
    for board_id in board_ids:
        try:
            board_context = _fetch_board_context(board_id)
            print(f"  [trello] board: {board_context['board_name']}")
            cards = _get(f"boards/{board_id}/cards", {"fields": "id,dateLastActivity"})
            for card in cards:
                items.append({
                    "id": card["id"],
                    "dateLastActivity": card.get("dateLastActivity"),
                    "board_context": board_context,
                })
        except Exception as e:
            print(f"  [trello] board {board_id} lightweight error: {e}")
    print(f"  [trello] found {len(items)} cards (lightweight)")
    return items


def fetch_trello_card_full(card_id: str, board_context: dict) -> SourceDocument:
    """Fetch a single card with full details and build SourceDocument with sections"""
    card = _get(f"cards/{card_id}", {
        "fields": "id,name,desc,idList,shortUrl,labels,due,dueComplete,start,closed,dateLastActivity,idMembers,idShort,url,badges",
        "checklists": "all",
        "attachments": "true",
        "members": "true",
        "customFieldItems": "true",
    })

    comments = []
    if card.get("badges", {}).get("comments", 0) > 0:
        try:
            actions = _get(f"cards/{card_id}/actions", {"filter": "commentCard"})
            comments = [a for a in actions if a.get("type") == "commentCard"]
        except Exception:
            pass

    return _build_trello_card_doc(card, comments, board_context)


def _build_trello_card_doc(card: dict, comments: list[dict], board_context: dict) -> SourceDocument:
    board_name = board_context["board_name"]
    board_url = board_context["board_url"]
    list_names = board_context["list_names"]
    member_names = board_context["member_names"]
    source_id = f"trello_{card['id']}"
    list_name = list_names.get(card.get("idList", ""), "")
    assigned = [member_names.get(mid, mid) for mid in card.get("idMembers", [])]

    # --- header: title + meta line ---
    header_parts = [f"# {card['name']}"]
    meta = []
    if list_name:
        meta.append(f"List: {list_name}")
    if card.get("labels"):
        meta.append(f"Labels: {', '.join(l['name'] for l in card['labels'] if l.get('name'))}")
    if card.get("due"):
        due_str = card["due"][:10]
        complete = " (done)" if card.get("dueComplete") else ""
        meta.append(f"Due: {due_str}{complete}")
    if card.get("start"):
        meta.append(f"Start: {card['start'][:10]}")
    if card.get("closed"):
        meta.append("Archived: yes")
    if assigned:
        meta.append(f"Members: {', '.join(assigned)}")
    if meta:
        header_parts.append(" | ".join(meta))

    # Attachments in header (stable data)
    attachments = card.get("attachments", [])
    if attachments:
        header_parts.append("## Attachments")
        for att in attachments:
            name = att.get("name", "")
            url = att.get("url", "")
            mime = att.get("mimeType", "")
            date = att.get("date", "")[:10] if att.get("date") else ""
            line = f"- {name}"
            if mime:
                line += f" ({mime})"
            if date:
                line += f" [{date}]"
            if url:
                line += f" {url}"
            header_parts.append(line)

    # Custom fields in header
    custom_fields = card.get("customFieldItems", [])
    if custom_fields:
        header_parts.append("## Custom Fields")
        for cf in custom_fields:
            cf_name = cf.get("idCustomField", "")
            value = cf.get("value", {})
            val_str = next(iter(value.values()), "") if value else ""
            header_parts.append(f"- {cf_name}: {val_str}")

    header_text = "\n\n".join(header_parts)

    # --- description ---
    description_text = (card.get("desc") or "").strip()

    # --- core text for context generation ---
    core_parts = [header_text]
    if description_text:
        core_parts.append(description_text)
    core_text = "\n\n".join(core_parts)

    # --- base metadata ---
    base_meta = {
        "source_type": "trello",
        "source_id": source_id,
        "file_name": f"{board_name} / {card['name']}",
        "source_url": card.get("shortUrl", ""),
        "source_updated_at": card.get("dateLastActivity"),
        "board_name": board_name,
        "list_name": list_name,
        "members": ", ".join(assigned),
        "labels": ", ".join(l["name"] for l in card.get("labels", []) if l.get("name")),
        "due": card.get("due", "")[:10] if card.get("due") else "",
        "due_complete": str(card.get("dueComplete", False)),
        "start": card.get("start", "")[:10] if card.get("start") else "",
        "archived": str(card.get("closed", False)),
        "date_last_activity": card.get("dateLastActivity", "")[:10] if card.get("dateLastActivity") else "",
        "card_number": str(card.get("idShort", "")),
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

    for cl in card.get("checklists", []):
        cl_parts = [f"## {cl['name']}"]
        for item in cl.get("checkItems", []):
            checked = "x" if item.get("state") == "complete" else " "
            cl_parts.append(f"- [{checked}] {item['name']}")
        cl_text = "\n".join(cl_parts)
        if len(cl_text.strip()) < 20:
            continue
        sections.append(DocumentSection(
            section_id=f"{source_id}::checklist_{cl['id']}",
            section_type="checklist",
            text=cl_text,
            metadata=dict(base_meta),
            md5=compute_md5(cl_text),
        ))

    for c in comments:
        author = c.get("memberCreator", {}).get("fullName", "Unknown")
        text = c.get("data", {}).get("text", "")
        date = c.get("date", "")[:10] if c.get("date") else ""
        if not text.strip():
            continue
        comment_text = f"**{author}** ({date})：{text}"
        if len(comment_text.strip()) < 20:
            continue
        action_id = c.get("id", "")
        sections.append(DocumentSection(
            section_id=f"{source_id}::comment_{action_id}",
            section_type="comment",
            text=comment_text,
            metadata=dict(base_meta),
            md5=compute_md5(comment_text),
        ))

    # --- full text ---
    full_parts = list(core_parts)
    for section in sections:
        if section.section_type in ("checklist", "comment"):
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


def fetch_trello_documents() -> list[SourceDocument]:
    """Legacy full-fetch function kept for backward compatibility"""
    items = fetch_trello_lightweight()
    docs = []
    for item in items:
        try:
            doc = fetch_trello_card_full(item["id"], item["board_context"])
            docs.append(doc)
        except Exception as e:
            print(f"  [trello] card {item['id']} error: {e}")
    return docs
