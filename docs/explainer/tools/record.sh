#!/bin/bash
# Records a floor's narration with Fish Audio's drama-3 voice model, trims its silences and writes audio/<chunk>.mp3.
# Usage, from docs/explainer: FISH_KEY=... tools/record.sh <floor> [chunk...]
# With no chunks named it skips those already recorded. Needs curl, ffmpeg and python3.
set -u
FLOOR=$1; shift
TMP=$(mktemp -d); trap 'rm -rf "$TMP"' EXIT
python3 -I - "narration/floor$FLOOR.json" "$TMP" "$@" <<'PY' | while read -r n; do
import json, sys
path, tmp, only = sys.argv[1], sys.argv[2], sys.argv[3:]
d = json.load(open(path))
for c, ids in d["chunks"].items():
    if only and c not in only: continue
    json.dump({"text": " ".join(d["lines"][i][0] for i in ids), "reference_id": "f8dfe9c83081432386f143e2fe9767ef", "format": "mp3", "mp3_bitrate": 128}, open(f"{tmp}/{c}.json", "w"))
    print(c)
PY
  [ $# -eq 0 ] && [ -s "audio/$n.mp3" ] && continue
  code=$(curl -s -o "$TMP/$n.mp3" -w '%{http_code}' -X POST https://api.fish.audio/v1/tts -H "Authorization: Bearer $FISH_KEY" -H 'Content-Type: application/json' -H 'model: drama-3' --data @"$TMP/$n.json")
  [ "$code" = 200 ] || { echo "$n HTTP $code"; head -c 300 "$TMP/$n.mp3"; echo; continue; }
  ffmpeg -loglevel error -y -i "$TMP/$n.mp3" -af "silenceremove=start_periods=1:start_threshold=-45dB:start_silence=0.05,areverse,silenceremove=start_periods=1:start_threshold=-45dB:start_silence=0.1,areverse,silenceremove=stop_periods=-1:stop_duration=0.8:stop_threshold=-45dB" -ac 1 -b:a 128k "audio/$n.mp3"
  echo "$n"
done
