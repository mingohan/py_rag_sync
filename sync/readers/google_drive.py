"""
Google Drive Reader
直接用 googleapiclient 讀取 Shared Drive，繞過 LlamaIndex GoogleDriveReader
對 Shared Drive 的支援問題。
"""
import os
import io
import json
import tempfile
from google.oauth2.credentials import Credentials
from googleapiclient.discovery import build
from googleapiclient.http import MediaIoBaseDownload
from llama_index.core.schema import Document

_EXPORT_MIME = {
    "application/vnd.google-apps.document": (
        "application/vnd.openxmlformats-officedocument.wordprocessingml.document",
        ".docx",
    ),
    "application/vnd.google-apps.spreadsheet": (
        "application/vnd.openxmlformats-officedocument.spreadsheetml.sheet",
        ".xlsx",
    ),
    "application/vnd.google-apps.presentation": (
        "application/vnd.openxmlformats-officedocument.presentationml.presentation",
        ".pptx",
    ),
}

_EDITOR_URL = {
    "application/vnd.google-apps.document": "https://docs.google.com/document/d/{id}/edit",
    "application/vnd.google-apps.spreadsheet": "https://docs.google.com/spreadsheets/d/{id}/edit",
    "application/vnd.google-apps.presentation": "https://docs.google.com/presentation/d/{id}/edit",
}


def _build_service():
    token_path = os.environ.get("GOOGLE_DRIVE_TOKEN_PATH", "/app/credentials/token.json")
    with open(token_path) as f:
        info = json.load(f)
    creds = Credentials.from_authorized_user_info(info)
    return build("drive", "v3", credentials=creds)


def _resolve_shortcut(service, file: dict) -> dict | None:
    """將 shortcut 解析成目標檔案的 metadata，保留 shortcut 的 name。"""
    target_id = file.get("shortcutDetails", {}).get("targetId")
    target_mime = file.get("shortcutDetails", {}).get("targetMimeType")
    if not target_id:
        return None
    return {
        "id": target_id,
        "name": file["name"],  # 用捷徑的名稱
        "mimeType": target_mime,
        "modifiedTime": file.get("modifiedTime", ""),
    }


def _list_files(service, folder_id: str, _seen_ids: set | None = None) -> list[dict]:
    """遞迴列出資料夾內所有檔案（含子資料夾），自動 follow shortcut。"""
    if _seen_ids is None:
        _seen_ids = set()

    files = []
    page_token = None
    while True:
        resp = service.files().list(
            q=f"'{folder_id}' in parents and trashed=false",
            includeItemsFromAllDrives=True,
            supportsAllDrives=True,
            fields="nextPageToken, files(id, name, mimeType, modifiedTime, shortcutDetails)",
            pageToken=page_token,
        ).execute()
        for f in resp.get("files", []):
            if f["mimeType"] == "application/vnd.google-apps.folder":
                files.extend(_list_files(service, f["id"], _seen_ids))
            elif f["mimeType"] == "application/vnd.google-apps.shortcut":
                target = _resolve_shortcut(service, f)
                if target and target["id"] not in _seen_ids:
                    _seen_ids.add(target["id"])
                    if target["mimeType"] == "application/vnd.google-apps.folder":
                        files.extend(_list_files(service, target["id"], _seen_ids))
                    else:
                        files.append(target)
            else:
                if f["id"] not in _seen_ids:
                    _seen_ids.add(f["id"])
                    files.append(f)
        page_token = resp.get("nextPageToken")
        if not page_token:
            break
    return files


def _read_file_text(service, file: dict) -> str | None:
    """下載並取出文字內容"""
    mime = file["mimeType"]
    fid = file["id"]

    if mime in _EXPORT_MIME:
        export_mime, ext = _EXPORT_MIME[mime]
        req = service.files().export_media(fileId=fid, mimeType=export_mime)
    elif mime == "application/pdf" or mime.startswith("text/"):
        req = service.files().get_media(fileId=fid, supportsAllDrives=True)
    else:
        return None  # 不支援的格式跳過

    buf = io.BytesIO()
    downloader = MediaIoBaseDownload(buf, req)
    done = False
    while not done:
        _, done = downloader.next_chunk()
    buf.seek(0)

    if mime in _EXPORT_MIME:
        _, ext = _EXPORT_MIME[mime]
        suffix = ext
    elif mime == "application/pdf":
        suffix = ".pdf"
    else:
        suffix = ".txt"

    with tempfile.NamedTemporaryFile(suffix=suffix, delete=False) as tmp:
        tmp.write(buf.read())
        tmp_path = tmp.name

    try:
        if suffix == ".pdf":
            import pymupdf4llm
            return pymupdf4llm.to_markdown(tmp_path, show_progress=False)
        elif suffix in (".docx", ".pptx", ".xlsx"):
            from llama_index.core import SimpleDirectoryReader
            reader = SimpleDirectoryReader(input_files=[tmp_path])
            docs = reader.load_data()
            return "\n\n".join(d.text for d in docs)
        else:
            return buf.getvalue().decode("utf-8", errors="ignore")
    finally:
        os.unlink(tmp_path)


def fetch_google_drive_documents() -> list[Document]:
    folder_id = os.environ["GOOGLE_DRIVE_FOLDER_ID"]
    service = _build_service()

    files = _list_files(service, folder_id)
    print(f"  [google_drive] found {len(files)} files")

    docs = []
    for i, f in enumerate(files, 1):
        print(f"  [google_drive] ({i}/{len(files)}) {f['name']}")
        try:
            text = _read_file_text(service, f)
            if not text or not text.strip():
                continue
            fid = f["id"]
            mime = f["mimeType"]
            url_template = _EDITOR_URL.get(mime, "https://drive.google.com/file/d/{id}/view")
            doc = Document(
                text=text,
                metadata={
                    "source_type": "google_drive",
                    "source_id": f"drive_{fid}",
                    "file_name": f["name"],
                    "mime_type": mime,
                    "modified_time": f.get("modifiedTime", ""),
                    "source_url": url_template.format(id=fid),
                },
            )
            docs.append(doc)
        except Exception as e:
            print(f"  [google_drive] skip {f['name']}: {e}")

    return docs
