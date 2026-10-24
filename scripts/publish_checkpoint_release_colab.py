#!/usr/bin/env python3
"""Publish the epoch-5 Swin checkpoint as a public GitHub Release asset from Colab."""
from pathlib import Path
import hashlib
import os
import requests

REPOSITORY = "vishal122006/roadguard-swin-transformer"
RELEASE_TAG = "swin-epoch5"
ASSET_NAME = "best_coco_bbox_mAP_epoch_5.pth"
DEFAULT_CHECKPOINT = Path(
    "/content/drive/MyDrive/RoadGuard/work_dirs/"
    "roadguard_swin_tiny_faster_rcnn/best_coco_bbox_mAP_epoch_5.pth"
)

def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(8 * 1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()

def api_request(method: str, url: str, token: str, **kwargs):
    headers = kwargs.pop("headers", {})
    headers.update({
        "Authorization": f"Bearer {token}",
        "Accept": "application/vnd.github+json",
        "X-GitHub-Api-Version": "2022-11-28",
        "User-Agent": "RoadGuard-Swin-Checkpoint-Publisher",
    })
    response = requests.request(method, url, headers=headers, timeout=(30, 900), **kwargs)
    if not response.ok:
        raise RuntimeError(
            f"GitHub API returned HTTP {response.status_code}: {response.text[:1000]}. "
            f"Confirm GITHUB_TOKEN has Contents: Read and write access to {REPOSITORY}."
        )
    return response

def main():
    try:
        from google.colab import userdata
    except ImportError as exc:
        raise RuntimeError("Run this script in Google Colab.") from exc

    token = userdata.get("GITHUB_TOKEN")
    if not token:
        raise RuntimeError(
            "Colab secret GITHUB_TOKEN is missing or notebook access is disabled."
        )

    checkpoint = Path(
        os.getenv("ROADGUARD_CHECKPOINT_LOCAL_PATH", str(DEFAULT_CHECKPOINT))
    ).expanduser()
    if not checkpoint.is_file():
        raise FileNotFoundError(
            f"Checkpoint not found: {checkpoint}. Mount Google Drive and set "
            "ROADGUARD_CHECKPOINT_LOCAL_PATH if required."
        )
    if checkpoint.stat().st_size < 10 * 1024 * 1024:
        raise RuntimeError("Checkpoint is unexpectedly small; refusing to publish.")

    size = checkpoint.stat().st_size
    checksum = sha256_file(checkpoint)
    print(f"Checkpoint: {checkpoint.name}")
    print(f"Size: {size / 1024**2:.1f} MB")
    print(f"SHA-256: {checksum}")
    print("NOTICE: the following step makes the checkpoint publicly downloadable.")
    confirmation = input(
        "Type PUBLISH to upload weights to the PUBLIC GitHub release: "
    ).strip()
    if confirmation != "PUBLISH":
        print("Cancelled. No upload was performed.")
        return

    api = f"https://api.github.com/repos/{REPOSITORY}"
    headers = {
        "Authorization": f"Bearer {token}",
        "Accept": "application/vnd.github+json",
        "X-GitHub-Api-Version": "2022-11-28",
    }
    get_release = requests.get(
        f"{api}/releases/tags/{RELEASE_TAG}", headers=headers, timeout=30
    )
    if get_release.status_code == 404:
        get_release = api_request(
            "POST", f"{api}/releases", token,
            json={
                "tag_name": RELEASE_TAG,
                "name": "RoadGuard Swin epoch-5 checkpoint",
                "body": (
                    "Epoch-5 Swin Transformer Tiny + FPN + Faster R-CNN checkpoint. "
                    "Research use; see the README and experiment report for measured "
                    "results and limitations."
                ),
                "draft": False,
                "prerelease": True,
            },
        )
    elif not get_release.ok:
        raise RuntimeError(
            f"Cannot read GitHub release (HTTP {get_release.status_code}). "
            "Check that your token can access the new repository."
        )

    release = get_release.json()
    existing = next(
        (asset for asset in release.get("assets", []) if asset.get("name") == ASSET_NAME),
        None,
    )
    if existing:
        if existing.get("digest") == f"sha256:{checksum}":
            print("Identical checkpoint already exists:", existing.get("browser_download_url"))
            return
        raise RuntimeError(
            f"{ASSET_NAME} already exists but its checksum is not confirmed identical. "
            "Nothing was overwritten; inspect the release before retrying."
        )

    upload_url = release["upload_url"].split("{", 1)[0] + f"?name={ASSET_NAME}"
    with checkpoint.open("rb") as handle:
        upload = api_request(
            "POST", upload_url, token,
            headers={"Content-Type": "application/octet-stream"},
            data=handle,
        )
    asset = upload.json()
    print("\nPublished checkpoint asset:", asset.get("browser_download_url"))
    print("Streamlit app will download it automatically on first detection.")

if __name__ == "__main__":
    main()
