#!/bin/bash
cd "$(dirname "$0")/.."
M=${MODELS:-$HOME/mosieung}
for p in $(pgrep -f "bin/uvicorn main:app"); do kill "$p"; done
export ONMAEUM_STT_WHISPER=${ONMAEUM_STT_WHISPER:-$M/stt/small-v3}
export ONMAEUM_LLM_PATH=${ONMAEUM_LLM_PATH:-$M/mosieung-escalator_w8a8_opt1_cal500.rkllm}
export ONMAEUM_TTS=1
export ONMAEUM_TTS_DIR=${ONMAEUM_TTS_DIR:-$M/tts/vits-mimic3-ko_KO-kss_low}
setsid nohup ${VENV:-$M/venv}/bin/uvicorn main:app --host 0.0.0.0 --port 8080 > server.log 2>&1 < /dev/null &
for i in $(seq 60); do
  curl -sf http://localhost:8080/api/health && echo && exit 0
  sleep 1
done
echo "서버가 켜지지 않았습니다. server.log 를 확인하세요."
exit 1
