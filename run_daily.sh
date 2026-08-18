#!/bin/bash
# CANSLIM-KR 매일 자동 분석 스크립트
# crontab 예시 (매일 오후 4시): 0 16 * * 1-5 /path/to/run_daily.sh

set -e
DIR="$(cd "$(dirname "$0")" && pwd)"
cd "$DIR"

DATE=$(date +%Y%m%d)
OUT="canslim_result_${DATE}.json"
LATEST="canslim_result.json"

echo "===== CANSLIM-KR Daily $(date) ====="

if [ ! -f watchlist.txt ]; then
  echo "watchlist.txt 없음. 종료."
  exit 1
fi

# DART 키가 환경에 있으면 자동 사용
python3 canslim_analyzer.py --watchlist watchlist.txt --out "$OUT"

cp "$OUT" "$LATEST"
echo "최신 결과 → $LATEST"
echo "완료."
