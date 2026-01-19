"""Discord Webhook Notifier for Pipeline Events.

Discord Webhook을 통한 파이프라인 이벤트 알림.
Telegram보다 설정이 간단하고 Embed로 시각적 알림 가능.
"""

from __future__ import annotations

import json
import os
import urllib.error
import urllib.request
from datetime import datetime
from typing import Optional

# 환경변수
DISCORD_ENABLED = os.getenv("DISCORD_ENABLED", "false").lower() == "true"
DISCORD_WEBHOOK_URL = os.getenv("DISCORD_WEBHOOK_URL", "")
DISCORD_NOTIFY_ERRORS = os.getenv("DISCORD_NOTIFY_ERRORS", "true").lower() == "true"
DISCORD_NOTIFY_MILESTONES = os.getenv("DISCORD_NOTIFY_MILESTONES", "true").lower() == "true"
DISCORD_MILESTONE_INTERVAL = int(os.getenv("DISCORD_MILESTONE_INTERVAL", "1000"))

# 마일스톤 추적
_last_milestone = 0


def _send_webhook(payload: dict) -> bool:
    """Discord Webhook으로 메시지 전송."""
    if not DISCORD_ENABLED or not DISCORD_WEBHOOK_URL:
        return False

    try:
        data = json.dumps(payload).encode("utf-8")
        req = urllib.request.Request(
            DISCORD_WEBHOOK_URL,
            data=data,
            headers={"Content-Type": "application/json"},
            method="POST",
        )
        with urllib.request.urlopen(req, timeout=10) as resp:
            return resp.status == 204  # Discord returns 204 No Content on success
    except (urllib.error.URLError, urllib.error.HTTPError, OSError):
        return False


def _make_embed(
    title: str,
    description: str,
    color: int,
    fields: Optional[list] = None,
) -> dict:
    """Discord Embed 생성."""
    embed = {
        "title": title,
        "description": description,
        "color": color,
        "timestamp": datetime.utcnow().isoformat(),
    }
    if fields:
        embed["fields"] = fields
    return embed


def notify_error(worker_id: str, file_name: str, error: str) -> bool:
    """에러 발생 알림."""
    if not DISCORD_NOTIFY_ERRORS:
        return False

    embed = _make_embed(
        title="Pipeline Error",
        description=f"```{error[:500]}```",
        color=0xFF0000,  # Red
        fields=[
            {"name": "Worker", "value": worker_id, "inline": True},
            {"name": "File", "value": file_name[:100], "inline": True},
        ],
    )
    return _send_webhook({"embeds": [embed]})


def notify_dlq(worker_id: str, file_name: str, reason: str) -> bool:
    """DLQ 이동 알림."""
    if not DISCORD_NOTIFY_ERRORS:
        return False

    embed = _make_embed(
        title="File Moved to DLQ",
        description="파일이 Dead Letter Queue로 이동되었습니다.",
        color=0xFF6600,  # Orange
        fields=[
            {"name": "Worker", "value": worker_id, "inline": True},
            {"name": "File", "value": file_name[:100], "inline": True},
            {"name": "Reason", "value": reason, "inline": True},
        ],
    )
    return _send_webhook({"embeds": [embed]})


def notify_milestone(done_count: int, queue_count: int, failed_count: int) -> bool:
    """마일스톤 달성 알림 (N건 단위)."""
    global _last_milestone

    if not DISCORD_NOTIFY_MILESTONES:
        return False

    current_milestone = (done_count // DISCORD_MILESTONE_INTERVAL) * DISCORD_MILESTONE_INTERVAL
    if current_milestone <= _last_milestone or current_milestone == 0:
        return False

    _last_milestone = current_milestone

    total = done_count + queue_count + failed_count
    progress = (done_count / total * 100) if total > 0 else 0

    embed = _make_embed(
        title=f"Milestone: {current_milestone:,} Files Completed",
        description=f"진행률: **{progress:.1f}%**",
        color=0x00FF00,  # Green
        fields=[
            {"name": "Completed", "value": f"{done_count:,}", "inline": True},
            {"name": "Remaining", "value": f"{queue_count:,}", "inline": True},
            {"name": "Failed", "value": f"{failed_count:,}", "inline": True},
        ],
    )
    return _send_webhook({"embeds": [embed]})


def notify_worker_crash(worker_id: str, error: str) -> bool:
    """워커 크래시 알림."""
    embed = _make_embed(
        title="Worker Crashed",
        description=f"```{error[:500]}```",
        color=0x990000,  # Dark Red
        fields=[
            {"name": "Worker", "value": worker_id, "inline": True},
        ],
    )
    return _send_webhook({"embeds": [embed]})


def notify_pipeline_start(total_files: int) -> bool:
    """파이프라인 시작 알림."""
    embed = _make_embed(
        title="Pipeline Started",
        description="파이프라인이 시작되었습니다.",
        color=0x0099FF,  # Blue
        fields=[
            {"name": "Total Files", "value": f"{total_files:,}", "inline": True},
        ],
    )
    return _send_webhook({"embeds": [embed]})


def notify_pipeline_complete(done_count: int, failed_count: int, elapsed_sec: float) -> bool:
    """파이프라인 완료 알림."""
    hours = int(elapsed_sec // 3600)
    minutes = int((elapsed_sec % 3600) // 60)

    embed = _make_embed(
        title="Pipeline Completed",
        description="파이프라인이 완료되었습니다.",
        color=0x00FF00,  # Green
        fields=[
            {"name": "Completed", "value": f"{done_count:,}", "inline": True},
            {"name": "Failed", "value": f"{failed_count:,}", "inline": True},
            {"name": "Duration", "value": f"{hours}h {minutes}m", "inline": True},
        ],
    )
    return _send_webhook({"embeds": [embed]})
