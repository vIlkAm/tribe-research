#!/usr/bin/env bash
# Read-only export of the video/outcome tables into results/metrics/*.csv.
#
# Owner decision 2026-09-26: tribe-research may READ the video, snapshot,
# account, deal and media-archive tables (public performance data). Every
# statement runs in a read-only transaction; nothing is written to the DB.
# Output stays in results/ (git-ignored). Run on the owner's server only.
set -euo pipefail
cd "$(dirname "$0")/.."
OUT=${1:-results/metrics}
mkdir -p "$OUT"

q() {  # q <name> <select>
  docker exec -i -e PGOPTIONS='-c default_transaction_read_only=on' supabase-db \
    psql -U supabase_admin -d postgres -X -q -v ON_ERROR_STOP=1 \
    -c "COPY ($2) TO STDOUT WITH (FORMAT csv, HEADER)" > "$OUT/$1.csv.tmp"
  mv "$OUT/$1.csv.tmp" "$OUT/$1.csv"
  echo "$1: $(($(wc -l < "$OUT/$1.csv") - 1)) rows" >&2
}

q deals "select id, deal_name as name, deal_type, operating_model, caar_tracking_only, is_active from deals"
q social_accounts "select id, deal_id, platform, handle, user_id, account_type, tracking_mode, status,
  follower_count, follower_count_updated_at, created_at from social_accounts"
q social_account_stat_snapshots "select social_account_id, follower_count, snapshot_at, source
  from social_account_stat_snapshots"
q video_performances "select id, social_account_id, video_link, video_title, views, likes, comments, shares, saves,
  engagement_rate, upload_date, last_updated, provider_observed_at, duration_seconds, is_campaign_video,
  payout_eligible, deal_campaign_id, category_id, history_truncated, caar_video_gone_since from video_performances"
q video_snapshots "select video_performance_id, snapshot_at, snapshot_date, views, likes, comments, shares, saves, source
  from video_snapshots"
q cross_platform_members "select m.link_id, m.video_performance_id, l.method, l.confidence
  from media_cross_platform_link_members m join media_cross_platform_links l on l.id = m.link_id"
q media_backfill_results "select video_performance_id, deal_id, platform, status, local_path, sha256, file_size_bytes,
  updated_at from media_backfill_results"
q media_archive_assets "select video_performance_id, deal_id, platform, archive_status, local_archive_path,
  local_retention_status, local_deleted_at, sha256, file_size_bytes, drive_file_id is not null as on_drive
  from media_archive_assets"
date -u +%FT%TZ > "$OUT/EXPORTED_AT"
