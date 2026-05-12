"""
Sync Pipeline — main entry point
Run: docker compose --profile sync run --rm py_sync
"""
import os
import hashlib
from concurrent.futures import ThreadPoolExecutor, as_completed
from datetime import datetime, timezone

from qdrant_client.models import Filter, FieldCondition, MatchValue, PointStruct
from sync.models import Chunk, DocumentSection

from python.config import get_settings
from python.pipeline import get_qdrant_client, ensure_collection
from python.pipeline import build_embedding
from .chunker import chunk_document, chunk_card, chunk_sections, add_context_to_nodes
from .embedder import embed_nodes
from .readers.redmine import fetch_redmine_lightweight, fetch_redmine_issue
from .readers.google_drive import list_google_drive_files, fetch_google_drive_file
from .readers.gitlab import fetch_gitlab_issues_lightweight, fetch_gitlab_issue_full, fetch_gitlab_wiki_documents
from .readers.asana import fetch_asana_lightweight, fetch_asana_task_full
from .readers.trello import fetch_trello_lightweight, fetch_trello_card_full

settings = get_settings()


def compute_md5(content: str) -> str:
    return hashlib.md5(content.encode()).hexdigest()


def get_existing_state(client) -> tuple[dict, dict, dict]:
    """
    Scroll all Qdrant points and return:
      doc_state:           source_id  → {updated_at, file_md5}
      section_state:       section_id → {section_md5}
      chunk_context_cache: chunk_md5  → context_text
    """
    doc_state: dict[str, dict] = {}
    section_state: dict[str, dict] = {}
    chunk_context_cache: dict[str, str] = {}
    offset = None
    while True:
        result, next_offset = client.scroll(
            collection_name=settings.qdrant_collection,
            with_payload=["source_id", "section_id", "section_md5", "source_updated_at", "file_md5",
                          "chunk_md5", "context_text"],
            with_vectors=False,
            limit=1000,
            offset=offset,
        )
        for point in result:
            source_id = point.payload.get("source_id")
            section_id = point.payload.get("section_id")
            section_md5 = point.payload.get("section_md5")
            updated_at = point.payload.get("source_updated_at")
            file_md5 = point.payload.get("file_md5")
            chunk_md5 = point.payload.get("chunk_md5")
            context_text = point.payload.get("context_text")

            if source_id and source_id not in doc_state:
                doc_state[source_id] = {
                    "updated_at": updated_at,
                    "file_md5": file_md5,
                }

            if section_id and section_md5:
                section_state[section_id] = {"section_md5": section_md5}

            if chunk_md5 and context_text:
                chunk_context_cache[chunk_md5] = context_text

        if next_offset is None:
            break
        offset = next_offset

    return doc_state, section_state, chunk_context_cache


def get_existing_source_ids(client) -> dict[str, set[str]]:
    """Scroll through all source_type → set[source_id] for orphan cleanup"""
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


def delete_orphans(client, source_type: str, current_ids: set[str], existing_ids: set[str]):
    """Delete orphaned data for sources that no longer exist"""
    orphan_ids = existing_ids - current_ids
    for source_id in orphan_ids:
        print(f"  [orphan] deleting {source_type}/{source_id}")
        delete_source(client, source_type, source_id)
    if orphan_ids:
        print(f"  deleted {len(orphan_ids)} orphan(s) from {source_type}")


def delete_source(client, source_type: str, source_id: str):
    client.delete(
        collection_name=settings.qdrant_collection,
        points_selector=Filter(
            must=[
                FieldCondition(key="source_type", match=MatchValue(value=source_type)),
                FieldCondition(key="source_id", match=MatchValue(value=source_id)),
            ]
        ),
    )


def delete_section(client, section_id: str):
    """Delete all Qdrant points belonging to a specific section"""
    client.delete(
        collection_name=settings.qdrant_collection,
        points_selector=Filter(
            must=[FieldCondition(key="section_id", match=MatchValue(value=section_id))]
        ),
    )


def update_source_updated_at(client, source_id: str, updated_at: str):
    """Update source_updated_at on all chunks for a source without re-embedding"""
    client.set_payload(
        collection_name=settings.qdrant_collection,
        payload={"source_updated_at": updated_at},
        points=Filter(
            must=[FieldCondition(key="source_id", match=MatchValue(value=source_id))]
        ),
    )


def upsert_nodes(client, nodes: list[Chunk]):
    if not nodes:
        return
    points = []
    for node in nodes:
        if node.embedding is None:
            print(f"  [warn] skipping node {node.node_id}: missing embedding")
            continue
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


def _sync_sections(
    client,
    source_id: str,
    source_type: str,
    sections: list[DocumentSection],
    doc_state: dict,
    section_state: dict,
    core_text: str,
    updated_at: str | None,
    stats: dict,
    chunk_context_cache: dict | None = None,
) -> bool:
    """
    Section-level diff and upsert. Returns True if any changes were made.

    Layer 2: compare each section's MD5 against section_state.
    Only embed sections whose MD5 has changed or are new.
    Delete sections that no longer exist in the source.
    """
    existing_doc = doc_state.get(source_id)
    new_section_ids = {s.section_id for s in sections}
    old_section_ids = {sid for sid in section_state if sid.startswith(source_id + "::")}

    # Clean up if: (a) brand-new source, or (b) old format with no section_id in Qdrant.
    # Old format: existing_doc is present but old_section_ids is empty (no section_id payload).
    is_old_format = existing_doc is not None and not old_section_ids
    if not existing_doc or is_old_format:
        delete_source(client, source_type, source_id)

    # Determine which sections need embedding
    changed_sections = []
    for section in sections:
        existing_md5 = section_state.get(section.section_id, {}).get("section_md5")
        if existing_md5 == section.md5:
            stats["skipped_sections"] += 1
            continue
        changed_sections.append(section)

    # Delete changed sections before re-upserting (avoid stale Qdrant points)
    for section in changed_sections:
        if section.section_id in section_state:
            delete_section(client, section.section_id)

    # Delete orphan sections (sections that disappeared from the source)
    dead_section_ids = old_section_ids - new_section_ids
    for dead_sid in dead_section_ids:
        delete_section(client, dead_sid)

    changed = bool(changed_sections or dead_section_ids)

    if changed_sections:
        nodes = chunk_sections(changed_sections)
        nodes = add_context_to_nodes(nodes, core_text, chunk_context_cache=chunk_context_cache)
        nodes = embed_nodes(nodes)
        upsert_nodes(client, nodes)
        stats["updated_sections"] += len(changed_sections)

    # Ensure source_updated_at is current on all unchanged sections
    if existing_doc and existing_doc.get("updated_at") != updated_at and updated_at:
        update_source_updated_at(client, source_id, updated_at)

    return changed


def sync_google_drive(client, doc_state: dict, section_state: dict, embed_model, chunk_context_cache: dict | None = None) -> dict:
    print("\n[Google Drive] syncing...")
    stats = {"added": 0, "skipped": 0, "failed": 0, "current_ids": set()}

    service, files = list_google_drive_files()
    total = len(files)
    pending_skips = 0

    for i, f in enumerate(files, 1):
        fid = f["id"]
        source_id = f"drive_{fid}"
        modified_time = f.get("modifiedTime", "")
        stats["current_ids"].add(source_id)

        existing = doc_state.get(source_id, {})

        # Layer 1: modifiedTime unchanged → skip download entirely
        if existing.get("updated_at") == modified_time and modified_time:
            stats["skipped"] += 1
            pending_skips += 1
            continue

        # Download content
        doc = fetch_google_drive_file(service, f)
        if not doc:
            continue

        # Layer 2: file MD5 unchanged → skip embedding (but update timestamp)
        md5 = compute_md5(doc.text)
        if existing.get("file_md5") == md5:
            if existing.get("updated_at") != modified_time and modified_time:
                update_source_updated_at(client, source_id, modified_time)
            stats["skipped"] += 1
            pending_skips += 1
            continue

        if pending_skips:
            print(f"  [{i}/{total}] skipped {pending_skips} (no change)")
            pending_skips = 0
        print(f"  [{i}/{total}] {doc.metadata.get('file_name', source_id)}")
        try:
            doc.metadata["file_md5"] = md5
            nodes = chunk_document(doc, embed_model)
            nodes = add_context_to_nodes(nodes, doc.text, chunk_context_cache=chunk_context_cache)
            nodes = embed_nodes(nodes)
            delete_source(client, "google_drive", source_id)
            upsert_nodes(client, nodes)
            stats["added"] += 1
        except Exception as e:
            print(f"  [error] google_drive/{source_id}: {e}")
            stats["failed"] += 1

    if pending_skips:
        print(f"  [{total}/{total}] skipped {pending_skips} (no change)")
    print(f"  added={stats['added']} skipped={stats['skipped']} failed={stats['failed']}")
    return stats


def sync_gitlab(client, doc_state: dict, section_state: dict, embed_model, chunk_context_cache: dict | None = None) -> dict:
    print("\n[GitLab] syncing...")
    stats = {
        "added": 0, "skipped": 0, "failed": 0,
        "updated_sections": 0, "skipped_sections": 0,
        "current_ids": set(),
    }

    projects = [p.strip() for p in os.environ.get("GITLAB_PROJECTS", "").split(",") if p.strip()]

    for project_path in projects:
        print(f"  [gitlab] project: {project_path}")

        # --- Issues (section-level) ---
        try:
            issue_items = fetch_gitlab_issues_lightweight(project_path)
            print(f"    issues: {len(issue_items)}")
            total = len(issue_items)
            pending_skips = 0

            for i, item in enumerate(issue_items, 1):
                source_id = f"gitlab_issue_{project_path}_{item['iid']}"
                stats["current_ids"].add(source_id)
                updated_at = item.get("updated_at")
                existing = doc_state.get(source_id, {})

                # Layer 1: timestamp check
                if existing.get("updated_at") == updated_at and updated_at:
                    stats["skipped"] += 1
                    pending_skips += 1
                    continue

                if pending_skips:
                    print(f"    [{i}/{total}] skipped {pending_skips} (no change)")
                    pending_skips = 0

                try:
                    doc = fetch_gitlab_issue_full(project_path, item["issue"])
                except Exception as e:
                    print(f"    [error] issue {item['iid']} fetch: {e}")
                    stats["failed"] += 1
                    continue

                sections = doc.metadata.get("sections", [])
                core_text = doc.metadata.get("core_text", doc.text)

                print(f"    [{i}/{total}] {doc.metadata.get('file_name', source_id)}")
                try:
                    changed = _sync_sections(
                        client, source_id, "gitlab_issue", sections,
                        doc_state, section_state, core_text, updated_at, stats,
                        chunk_context_cache=chunk_context_cache,
                    )
                    if changed or not existing:
                        stats["added"] += 1
                except Exception as e:
                    print(f"    [error] {source_id}: {e}")
                    stats["failed"] += 1

            if pending_skips:
                print(f"    [{total}/{total}] skipped {pending_skips} (no change)")

        except Exception as e:
            print(f"    issues error: {e}")

        # --- Wiki (full-document MD5, no sections) ---
        try:
            wiki_docs = fetch_gitlab_wiki_documents(project_path)
            print(f"    wiki: {len(wiki_docs)}")
            for doc in wiki_docs:
                source_id = doc.metadata["source_id"]
                stats["current_ids"].add(source_id)
                md5 = compute_md5(doc.text)
                existing = doc_state.get(source_id, {})
                if existing.get("file_md5") == md5:
                    stats["skipped"] += 1
                    continue
                print(f"    wiki: {doc.metadata.get('file_name', source_id)}")
                try:
                    delete_source(client, "gitlab_wiki", source_id)
                    doc.metadata["file_md5"] = md5
                    nodes = chunk_card(doc.text, doc.metadata)
                    nodes = add_context_to_nodes(nodes, doc.text, chunk_context_cache=chunk_context_cache)
                    nodes = embed_nodes(nodes)
                    upsert_nodes(client, nodes)
                    stats["added"] += 1
                except Exception as e:
                    print(f"    [error] {source_id}: {e}")
                    stats["failed"] += 1
        except Exception as e:
            print(f"    wiki error: {e}")

    print(f"  added={stats['added']} skipped={stats['skipped']} "
          f"updated_sections={stats['updated_sections']} skipped_sections={stats['skipped_sections']} "
          f"failed={stats['failed']}")
    return stats


def sync_asana(client, doc_state: dict, section_state: dict, chunk_context_cache: dict | None = None) -> dict:
    print("\n[Asana] syncing...")
    stats = {
        "added": 0, "skipped": 0, "failed": 0,
        "updated_sections": 0, "skipped_sections": 0,
        "current_ids": set(),
    }

    lightweight_items = fetch_asana_lightweight()
    total = len(lightweight_items)
    pending_skips = 0

    for i, item in enumerate(lightweight_items, 1):
        source_id = f"asana_task_{item['id']}"
        stats["current_ids"].add(source_id)
        updated_at = item.get("modified_at")
        existing = doc_state.get(source_id, {})

        # Layer 1: timestamp check
        if existing.get("updated_at") == updated_at and updated_at:
            stats["skipped"] += 1
            pending_skips += 1
            continue

        if pending_skips:
            print(f"  [{i}/{total}] skipped {pending_skips} (no change)")
            pending_skips = 0

        try:
            doc = fetch_asana_task_full(item["id"], item["project_context"])
        except Exception as e:
            print(f"  [error] asana/{source_id} fetch: {e}")
            stats["failed"] += 1
            continue

        sections = doc.metadata.get("sections", [])
        core_text = doc.metadata.get("core_text", doc.text)

        print(f"  [{i}/{total}] {doc.metadata.get('file_name', source_id)}")
        try:
            changed = _sync_sections(
                client, source_id, "asana", sections,
                doc_state, section_state, core_text, updated_at, stats,
                chunk_context_cache=chunk_context_cache,
            )
            if changed or not existing:
                stats["added"] += 1
        except Exception as e:
            print(f"  [error] {source_id}: {e}")
            stats["failed"] += 1

    if pending_skips:
        print(f"  [{total}/{total}] skipped {pending_skips} (no change)")
    print(f"  added={stats['added']} skipped={stats['skipped']} "
          f"updated_sections={stats['updated_sections']} skipped_sections={stats['skipped_sections']} "
          f"failed={stats['failed']}")
    return stats


def sync_trello(client, doc_state: dict, section_state: dict, chunk_context_cache: dict | None = None) -> dict:
    print("\n[Trello] syncing...")
    stats = {
        "added": 0, "skipped": 0, "failed": 0,
        "updated_sections": 0, "skipped_sections": 0,
        "current_ids": set(),
    }

    lightweight_items = fetch_trello_lightweight()
    total = len(lightweight_items)
    pending_skips = 0

    for i, item in enumerate(lightweight_items, 1):
        source_id = f"trello_{item['id']}"
        stats["current_ids"].add(source_id)
        updated_at = item.get("dateLastActivity")
        existing = doc_state.get(source_id, {})

        # Layer 1: timestamp check
        if existing.get("updated_at") == updated_at and updated_at:
            stats["skipped"] += 1
            pending_skips += 1
            continue

        if pending_skips:
            print(f"  [{i}/{total}] skipped {pending_skips} (no change)")
            pending_skips = 0

        try:
            doc = fetch_trello_card_full(item["id"], item["board_context"])
        except Exception as e:
            print(f"  [error] trello/{source_id} fetch: {e}")
            stats["failed"] += 1
            continue

        sections = doc.metadata.get("sections", [])
        core_text = doc.metadata.get("core_text", doc.text)

        print(f"  [{i}/{total}] {doc.metadata.get('file_name', source_id)}")
        try:
            changed = _sync_sections(
                client, source_id, "trello", sections,
                doc_state, section_state, core_text, updated_at, stats,
                chunk_context_cache=chunk_context_cache,
            )
            if changed or not existing:
                stats["added"] += 1
        except Exception as e:
            print(f"  [error] {source_id}: {e}")
            stats["failed"] += 1

    if pending_skips:
        print(f"  [{total}/{total}] skipped {pending_skips} (no change)")
    print(f"  added={stats['added']} skipped={stats['skipped']} "
          f"updated_sections={stats['updated_sections']} skipped_sections={stats['skipped_sections']} "
          f"failed={stats['failed']}")
    return stats


def sync_redmine(client, doc_state: dict, section_state: dict, chunk_context_cache: dict | None = None) -> dict:
    print("\n[Redmine] syncing...")
    stats = {
        "added": 0, "skipped": 0, "failed": 0,
        "updated_sections": 0, "skipped_sections": 0,
        "current_ids": set(),
    }

    lightweight_items = fetch_redmine_lightweight()
    total = len(lightweight_items)
    pending_skips = 0

    for i, item in enumerate(lightweight_items, 1):
        source_id = f"redmine_issue_{item['id']}"
        stats["current_ids"].add(source_id)
        updated_at = item.get("updated_on")
        existing = doc_state.get(source_id, {})

        # Layer 1: timestamp check
        if existing.get("updated_at") == updated_at and updated_at:
            stats["skipped"] += 1
            pending_skips += 1
            continue

        if pending_skips:
            print(f"  [{i}/{total}] skipped {pending_skips} (no change)")
            pending_skips = 0

        try:
            doc = fetch_redmine_issue(item["id"])
        except Exception as e:
            print(f"  [error] redmine/{source_id} fetch: {e}")
            stats["failed"] += 1
            continue

        sections = doc.metadata.get("sections", [])
        core_text = doc.metadata.get("core_text", doc.text)

        print(f"  [{i}/{total}] {doc.metadata.get('file_name', source_id)}")
        try:
            changed = _sync_sections(
                client, source_id, "redmine", sections,
                doc_state, section_state, core_text, updated_at, stats,
                chunk_context_cache=chunk_context_cache,
            )
            if changed or not existing:
                stats["added"] += 1
        except Exception as e:
            print(f"  [error] {source_id}: {e}")
            stats["failed"] += 1

    if pending_skips:
        print(f"  [{total}/{total}] skipped {pending_skips} (no change)")
    print(f"  added={stats['added']} skipped={stats['skipped']} "
          f"updated_sections={stats['updated_sections']} skipped_sections={stats['skipped_sections']} "
          f"failed={stats['failed']}")
    return stats


def main():
    print(f"=== py_rag sync started at {datetime.now(timezone.utc).isoformat()} ===")

    client = get_qdrant_client()
    ensure_collection(client)
    doc_state, section_state, chunk_context_cache = get_existing_state(client)
    existing_source_ids = get_existing_source_ids(client)

    embed_model = build_embedding()

    tasks = {"redmine": (sync_redmine, (client, doc_state, section_state, chunk_context_cache))}
    if os.environ.get("GOOGLE_DRIVE_FOLDER_ID"):
        tasks["google_drive"] = (sync_google_drive, (client, doc_state, section_state, embed_model, chunk_context_cache))
    else:
        print("\n[Google Drive] GOOGLE_DRIVE_FOLDER_ID not set, skipping")
    if os.environ.get("GITLAB_TOKEN"):
        tasks["gitlab"] = (sync_gitlab, (client, doc_state, section_state, embed_model, chunk_context_cache))
    else:
        print("\n[GitLab] GITLAB_TOKEN not set, skipping")
    # if os.environ.get("TRELLO_API_KEY"):
    #     tasks["trello"] = (sync_trello, (client, doc_state, section_state, chunk_context_cache))
    # else:
    #     print("\n[Trello] TRELLO_API_KEY not set, skipping")
    if os.environ.get("ASANA_ACCESS_TOKEN"):
        tasks["asana"] = (sync_asana, (client, doc_state, section_state, chunk_context_cache))
    else:
        print("\n[Asana] ASANA_ACCESS_TOKEN not set, skipping")

    if not tasks:
        print("\n[sync] no sources configured, exiting")
        return

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

    print("\n[orphan cleanup]")
    for source_key, stats in total_stats.items():
        current_ids = stats.get("current_ids", set())
        if source_key == "gitlab":
            for st in ("gitlab_issue", "gitlab_wiki"):
                st_ids = {sid for sid in current_ids if sid.startswith(f"{st}_")}
                delete_orphans(client, st, st_ids, existing_source_ids.get(st, set()))
        else:
            existing_ids = existing_source_ids.get(source_key, set())
            delete_orphans(client, source_key, current_ids, existing_ids)

    print("\n=== sync complete ===")
    for source, stats in total_stats.items():
        print(f"  {source}: {stats}")

    _notify_slack(total_stats)


def _notify_slack(total_stats: dict):
    slack_token = settings.slack_bot_token
    slack_channel = os.environ.get("SLACK_SYNC_CHANNEL")
    if not slack_token or not slack_channel:
        return

    lines = [f"*py_rag sync complete* ({datetime.now(timezone.utc).strftime('%Y-%m-%d %H:%M UTC')})"]
    for source, stats in total_stats.items():
        added = stats.get("added", 0)
        skipped = stats.get("skipped", 0)
        failed = stats.get("failed", 0)
        updated_sections = stats.get("updated_sections", 0)
        skipped_sections = stats.get("skipped_sections", 0)
        icon = ":warning:" if failed else ":white_check_mark:"
        line = f"{icon} *{source}*: added {added} | skipped {skipped} | failed {failed}"
        if updated_sections or skipped_sections:
            line += f" | sections updated {updated_sections} / skipped {skipped_sections}"
        lines.append(line)

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
