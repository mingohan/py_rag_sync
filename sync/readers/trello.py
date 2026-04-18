"""
Trello Reader
抓取多個 board 的 cards（含 checklists、comments、attachments、members）
TRELLO_BOARD_IDS 逗號分隔
"""
import os
import httpx
from llama_index.core.schema import Document


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


def fetch_trello_documents() -> list[Document]:
    board_ids = [b.strip() for b in os.environ.get("TRELLO_BOARD_IDS", "").split(",") if b.strip()]
    docs = []

    for board_id in board_ids:
        try:
            board = _get(f"boards/{board_id}", {"fields": "name,shortUrl"})
            board_name = board.get("name", board_id)
            board_url = board.get("shortUrl", "")
            print(f"  [trello] board: {board_name}")

            # lists（欄位名稱）
            lists = _get(f"boards/{board_id}/lists", {"fields": "id,name"})
            list_names = {l["id"]: l["name"] for l in lists}

            # board members id -> fullName
            board_members = _get(f"boards/{board_id}/members", {"fields": "id,fullName,username"})
            member_names = {m["id"]: m["fullName"] for m in board_members}

            # cards（含 checklists、attachments、members、customFieldItems）
            # actions 不放在這裡，改為逐張卡片另外抓，避免部分看板 403
            cards = _get(f"boards/{board_id}/cards", {
                "fields": "id,name,desc,idList,shortUrl,labels,due,dueComplete,start,closed,dateLastActivity,idMembers,idShort,url,badges",
                "checklists": "all",
                "attachments": "true",
                "members": "true",
                "customFieldItems": "true",
            })

            for card in cards:
                parts = [f"# {card['name']}"]

                # --- meta line ---
                meta = []
                list_name = list_names.get(card.get("idList", ""), "")
                if list_name:
                    meta.append(f"List: {list_name}")
                if card.get("labels"):
                    meta.append(f"Labels: {', '.join(l['name'] for l in card['labels'] if l.get('name'))}")
                if card.get("due"):
                    due_str = card["due"][:10]
                    complete = " (完成)" if card.get("dueComplete") else ""
                    meta.append(f"Due: {due_str}{complete}")
                if card.get("start"):
                    meta.append(f"Start: {card['start'][:10]}")
                if card.get("closed"):
                    meta.append("Archived: yes")
                if card.get("dateLastActivity"):
                    meta.append(f"LastActivity: {card['dateLastActivity'][:10]}")
                # 負責人
                assigned = [member_names.get(mid, mid) for mid in card.get("idMembers", [])]
                if assigned:
                    meta.append(f"Members: {', '.join(assigned)}")
                if meta:
                    parts.append(" | ".join(meta))

                # --- description ---
                if (card.get("desc") or "").strip():
                    parts.append(card["desc"])

                # --- checklists ---
                for cl in card.get("checklists", []):
                    parts.append(f"## {cl['name']}")
                    for item in cl.get("checkItems", []):
                        checked = "x" if item.get("state") == "complete" else " "
                        parts.append(f"- [{checked}] {item['name']}")

                # --- attachments ---
                attachments = card.get("attachments", [])
                if attachments:
                    parts.append("## 附件")
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
                        parts.append(line)

                # --- custom fields ---
                custom_fields = card.get("customFieldItems", [])
                if custom_fields:
                    parts.append("## 自訂欄位")
                    for cf in custom_fields:
                        cf_name = cf.get("idCustomField", "")
                        value = cf.get("value", {})
                        # value 可能是 {text: ...} / {number: ...} / {checked: ...}
                        val_str = next(iter(value.values()), "") if value else ""
                        parts.append(f"- {cf_name}: {val_str}")

                # --- comments（只有 badges.comments > 0 才呼叫，避免 3000+ 次無謂請求）---
                if card.get("badges", {}).get("comments", 0) > 0:
                    try:
                        actions = _get(f"cards/{card['id']}/actions", {"filter": "commentCard"})
                        comments = [a for a in actions if a.get("type") == "commentCard"]
                        if comments:
                            parts.append("## 留言")
                            for c in comments:
                                author = c.get("memberCreator", {}).get("fullName", "Unknown")
                                text = c.get("data", {}).get("text", "")
                                date = c.get("date", "")[:10] if c.get("date") else ""
                                if text.strip():
                                    parts.append(f"**{author}** ({date})：{text}")
                    except Exception:
                        pass

                docs.append(Document(
                    text="\n\n".join(parts),
                    metadata={
                        "source_type": "trello",
                        "source_id": f"trello_{card['id']}",
                        "file_name": f"{board_name} / {card['name']}",
                        "source_url": card.get("shortUrl", ""),
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
                    },
                ))

        except Exception as e:
            print(f"  [trello] board {board_id} error: {e}")

    return docs
