"""
Sync Pipeline 主程式
執行：docker compose --profile sync run --rm py_sync
"""
import os
import hashlib
from concurrent.futures import ThreadPoolExecutor, as_completed
from datetime import datetime, timezone

from qdrant_client.models import PointStruct
from llama_index.core.schema import TextNode

from python.config import get_settings
from python.pipeline import get_qdrant_client, ensure_collection, build_vector_store, build_embedding
from .chunker import chunk_document, chunk_card, add_context_to_nodes
from .embedder import embed_nodes
from .readers.redmine import fetch_redmine_documents
from .readers.google_drive import fetch_google_drive_documents
from .readers.gitlab import fetch_gitlab_documents
from .readers.trello import fetch_trello_documents

settings = get_settings()


def compute_md5(content: str) -> str:
    return hashlib.md5(content.encode()).hexdigest()


def get_existing_source_ids(client) -> dict[str, set[str]]:
    """捲動取得所有 source_type → set[source_id]，用於增量同步與孤兒清除"""
    result_map: dict[str, set[str]] = {}
    offset = None
    while True:
        result, next_offset = client.scroll(
            collection_name=settings.qdrant_collection,
            with_payload=["source_type", "source_id"],
            with_vectors=False,
            limit=1000,
            offset=offset,
        )
        for point in result:
            source_type = point.payload.get("source_type")
            source_id = point.payload.get("source_id")
            if source_type and source_id:
                result_map.setdefault(source_type, set()).add(source_id)
        if next_offset is None:
            break
        offset = next_offset
    return result_map


def get_existing_md5s(client) -> dict[str, str]:
    """捲動取得所有 source_id → file_md5，用於增量同步"""
    md5s = {}
    offset = None
    while True:
        result, next_offset = client.scroll(
            collection_name=settings.qdrant_collection,
            with_payload=["source_id", "file_md5"],
            with_vectors=False,
            limit=1000,
            offset=offset,
        )
        for point in result:
            sid = point.payload.get("source_id")
            md5 = point.payload.get("file_md5")
            if sid and md5:
                md5s[sid] = md5
        if next_offset is None:
            break
        offset = next_offset
    return md5s


def delete_orphans(client, source_type: str, current_ids: set[str], existing_ids: set[str]):
    """刪除來源已消失的孤兒資料"""
    orphan_ids = existing_ids - current_ids
    for source_id in orphan_ids:
        print(f"  [orphan] deleting {source_type}/{source_id}")
        delete_source(client, source_type, source_id)
    if orphan_ids:
        print(f"  deleted {len(orphan_ids)} orphan(s) from {source_type}")


def delete_source(client, source_type: str, source_id: str):
    from qdrant_client.models import Filter, FieldCondition, MatchValue
    client.delete(
        collection_name=settings.qdrant_collection,
        points_selector=Filter(
            must=[
                FieldCondition(key="source_type", match=MatchValue(value=source_type)),
                FieldCondition(key="source_id", match=MatchValue(value=source_id)),
            ]
        ),
    )


def upsert_nodes(client, nodes: list[TextNode]):
    if not nodes:
        return
    points = []
    for node in nodes:
        sparse_indices = node.metadata.pop("sparse_indices", [])
        sparse_values = node.metadata.pop("sparse_values", [])
        points.append(PointStruct(
            id=node.node_id,
            vector={
                "dense": node.embedding,
                "sparse": {
                    "indices": sparse_indices,
                    "values": sparse_values,
                },
            },
            payload={
                **node.metadata,
                "text": node.text,
                "synced_at": datetime.now(timezone.utc).isoformat(),
            },
        ))
    client.upsert(collection_name=settings.qdrant_collection, points=points)


def sync_google_drive(client, existing_md5s: dict, embed_model) -> dict:
    print("\n[Google Drive] syncing...")
    stats = {"added": 0, "skipped": 0, "failed": 0, "current_ids": set()}

    docs = fetch_google_drive_documents()
    total = len(docs)

    for i, doc in enumerate(docs, 1):
        source_id = doc.metadata["source_id"]
        title = doc.metadata.get("file_name", source_id)
        stats["current_ids"].add(source_id)
        md5 = compute_md5(doc.text)
        if existing_md5s.get(source_id) == md5:
            stats["skipped"] += 1
            continue

        print(f"  [{i}/{total}] {title}")
        try:
            delete_source(client, "google_drive", source_id)
            doc.metadata["file_md5"] = md5
            nodes = chunk_document(doc, embed_model)
            nodes = add_context_to_nodes(nodes, doc.text)
            nodes = embed_nodes(nodes)
            upsert_nodes(client, nodes)
            stats["added"] += 1
        except Exception as e:
            print(f"  [error] google_drive/{source_id}: {e}")
            stats["failed"] += 1

    print(f"  added={stats['added']} skipped={stats['skipped']} failed={stats['failed']}")
    return stats


def sync_gitlab(client, existing_md5s: dict, embed_model) -> dict:
    print("\n[GitLab] syncing...")
    stats = {"added": 0, "skipped": 0, "failed": 0, "current_ids": set()}

    docs = fetch_gitlab_documents()
    total = len(docs)

    for i, doc in enumerate(docs, 1):
        source_id = doc.metadata["source_id"]
        source_type = doc.metadata["source_type"]
        title = doc.metadata.get("file_name", source_id)
        stats["current_ids"].add(source_id)
        md5 = compute_md5(doc.text)
        if existing_md5s.get(source_id) == md5:
            stats["skipped"] += 1
            continue

        print(f"  [{i}/{total}] {title}")
        try:
            delete_source(client, source_type, source_id)
            doc.metadata["file_md5"] = md5
            nodes = chunk_card(doc.text, doc.metadata)
            nodes = add_context_to_nodes(nodes, doc.text)
            nodes = embed_nodes(nodes)
            upsert_nodes(client, nodes)
            stats["added"] += 1
        except Exception as e:
            print(f"  [error] {source_type}/{source_id}: {e}")
            stats["failed"] += 1

    print(f"  added={stats['added']} skipped={stats['skipped']} failed={stats['failed']}")
    return stats


def sync_trello(client, existing_md5s: dict) -> dict:
    print("\n[Trello] syncing...")
    stats = {"added": 0, "skipped": 0, "failed": 0, "current_ids": set()}

    docs = fetch_trello_documents()
    total = len(docs)

    for i, doc in enumerate(docs, 1):
        source_id = doc.metadata["source_id"]
        title = doc.metadata.get("file_name", source_id)
        stats["current_ids"].add(source_id)
        md5 = compute_md5(doc.text)
        if existing_md5s.get(source_id) == md5:
            stats["skipped"] += 1
            continue

        print(f"  [{i}/{total}] {title}")
        try:
            delete_source(client, "trello", source_id)
            doc.metadata["file_md5"] = md5
            nodes = chunk_card(doc.text, doc.metadata)
            nodes = add_context_to_nodes(nodes, doc.text)
            nodes = embed_nodes(nodes)
            upsert_nodes(client, nodes)
            stats["added"] += 1
        except Exception as e:
            print(f"  [error] trello/{source_id}: {e}")
            stats["failed"] += 1

    print(f"  added={stats['added']} skipped={stats['skipped']} failed={stats['failed']}")
    return stats


def sync_redmine(client, existing_md5s: dict) -> dict:
    print("\n[Redmine] syncing...")
    stats = {"added": 0, "skipped": 0, "failed": 0, "current_ids": set()}

    docs = fetch_redmine_documents()
    total = len(docs)

    for i, doc in enumerate(docs, 1):
        source_id = doc.metadata["source_id"]
        title = doc.metadata.get("file_name", source_id)
        stats["current_ids"].add(source_id)
        md5 = compute_md5(doc.text)

        if existing_md5s.get(source_id) == md5:
            stats["skipped"] += 1
            continue

        print(f"  [{i}/{total}] {title}")
        try:
            delete_source(client, "redmine", source_id)
            doc.metadata["file_md5"] = md5
            nodes = chunk_card(doc.text, doc.metadata)
            nodes = add_context_to_nodes(nodes, doc.text)
            nodes = embed_nodes(nodes)
            upsert_nodes(client, nodes)
            stats["added"] += 1
        except Exception as e:
            print(f"  [error] redmine/{source_id}: {e}")
            stats["failed"] += 1

    print(f"  added={stats['added']} skipped={stats['skipped']} failed={stats['failed']}")
    return stats


def main():
    print(f"=== py_rag sync started at {datetime.now(timezone.utc).isoformat()} ===")

    client = get_qdrant_client()
    ensure_collection(client)
    existing_md5s = get_existing_md5s(client)
    existing_source_ids = get_existing_source_ids(client)

    # embed_model 只初始化一次，供所有需要 SemanticSplitter 的來源共用
    embed_model = build_embedding()

    # 各來源並行執行
    tasks = {"redmine": (sync_redmine, (client, existing_md5s))}
    if os.environ.get("GOOGLE_DRIVE_FOLDER_ID"):
        tasks["google_drive"] = (sync_google_drive, (client, existing_md5s, embed_model))
    else:
        print("\n[Google Drive] GOOGLE_DRIVE_FOLDER_ID not set, skipping")
    if os.environ.get("GITLAB_TOKEN"):
        tasks["gitlab"] = (sync_gitlab, (client, existing_md5s, embed_model))
    else:
        print("\n[GitLab] GITLAB_TOKEN not set, skipping")
    if os.environ.get("TRELLO_API_KEY"):
        tasks["trello"] = (sync_trello, (client, existing_md5s))
    else:
        print("\n[Trello] TRELLO_API_KEY not set, skipping")

    total_stats = {}
    with ThreadPoolExecutor(max_workers=len(tasks)) as executor:
        futures = {
            executor.submit(fn, *args): source
            for source, (fn, args) in tasks.items()
        }
        for future in as_completed(futures):
            source = futures[future]
            try:
                total_stats[source] = future.result()
            except Exception as e:
                print(f"  [error] {source} source failed entirely: {e}")

    # 孤兒清除：刪除來源已消失的資料
    print("\n[orphan cleanup]")
    for source_key, stats in total_stats.items():
        current_ids = stats.get("current_ids", set())
        if source_key == "gitlab":
            # gitlab 有兩種 source_type
            for st in ("gitlab_issue", "gitlab_wiki"):
                st_ids = {sid for sid in current_ids if sid.startswith(f"{st}_")}
                delete_orphans(client, st, st_ids, existing_source_ids.get(st, set()))
        else:
            existing_ids = existing_source_ids.get(source_key, set())
            delete_orphans(client, source_key, current_ids, existing_ids)


    # TODO: Slack
    # total_stats["slack"] = sync_slack(client, existing_md5s)

    print("\n=== sync complete ===")
    for source, stats in total_stats.items():
        print(f"  {source}: {stats}")

    _notify_slack(total_stats)


def _notify_slack(total_stats: dict):
    slack_token = settings.slack_bot_token
    slack_channel = os.environ.get("SLACK_SYNC_CHANNEL")
    if not slack_token or not slack_channel:
        return

    lines = [f"*py_rag sync 完成* ({datetime.now(timezone.utc).strftime('%Y-%m-%d %H:%M UTC')})"]
    for source, stats in total_stats.items():
        added = stats.get("added", 0)
        skipped = stats.get("skipped", 0)
        failed = stats.get("failed", 0)
        icon = ":warning:" if failed else ":white_check_mark:"
        lines.append(f"{icon} *{source}*: 新增 {added} | 略過 {skipped} | 失敗 {failed}")

    import httpx
    try:
        httpx.post(
            "https://slack.com/api/chat.postMessage",
            headers={"Authorization": f"Bearer {slack_token}"},
            json={"channel": slack_channel, "text": "\n".join(lines)},
            timeout=10,
        )
    except Exception as e:
        print(f"  [slack] notify failed: {e}")


if __name__ == "__main__":
    main()
