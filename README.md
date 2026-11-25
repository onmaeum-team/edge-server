# 모시응 엣지 서버

에스컬레이터 음성 점검 서버. Orange Pi 5 Plus(RK3588)에서 인터넷 없이 STT, LLM, TTS를 모두 처리한다.

앱이 말소리 구간을 WAV로 보내면 서버가 받아쓰고, 점검표 항목 호출(`<esc_X_Y>(...)<esc_end>`)로 바꿔 기록한 뒤 안내 음성을 돌려준다.

```
앱 (VAD, 16 kHz WAV) -> STT -> 텍스트 정규화 -> LLM -> 검증 -> 기록 -> TTS -> 앱
```

## 구성

| 단계 | 모델 | 실행 |
| --- | --- | --- |
| STT | Whisper small (한국어 파인튜닝), int8 | faster-whisper, CPU 큰 코어 4개 |
| LLM | Llama-3.2-1B LoRA, w8a8 | RKLLM 1.3, NPU 3코어 |
| TTS | VITS (vits-mimic3-ko_KO-kss_low) | sherpa-onnx 1.12.40, CPU 2스레드 |

| 파일 | 내용 |
| --- | --- |
| `main.py` | FastAPI 라우트, 점검 세션, 음성 명령 |
| `pipeline.py` | 모델 로드, STT/LLM/TTS 호출, 출력 검증, 판정 규칙, 점검표 |
| `stt_fast.py` | 말소리 길이만큼만 인코딩하는 Whisper 경로 |
| `rkllm_runner.py` | `esc_demo`(RKLLM C API)를 상주 프로세스로 실행 |
| `tts_runner.py` | VITS 합성, 숫자/단위 읽기 변환 |
| `textnorm.py` | STT 결과를 학습 데이터 표기로 정규화 |
| `storage.py` | 설비, 점검 기록, 보고서 파일 저장 |
| `report.py` | 점검 보고서 HTML |
| `deploy/` | 실행 스크립트, systemd 서비스, 핫스팟 |

## 보드 측정 (평가 음성 37개)

| 항목 | 결과 |
| --- | --- |
| STT 시간 (중앙값) | 3.80 s -> 1.22 s (`stt_fast`), CER 0.025 동일 |
| TTS 시간 | 문장당 0.4 ~ 1.1 s |
| 함수 일치 | 37/37 |
| 전체 응답 (중앙값 / 최대) | 4.3 s / 8.1 s |

Whisper는 입력을 30초로 채워 인코딩하므로 짧은 발화에서도 인코더 비용이 같다. `stt_fast`는 말소리 길이에 2초를 더한 만큼만 인코딩하고, 결과가 반복되거나 너무 길면 30초 경로로 다시 돌린다. 여유를 1초로 줄이면 반복 출력이 생겼다.

## 실행

모델 파일은 저장소에 포함하지 않는다.

```bash
python3 -m venv ~/mosieung/venv
~/mosieung/venv/bin/pip install -r requirements.txt -r requirements-stt.txt -r requirements-llm.txt
bash deploy/start.sh
```

`deploy/start.sh`는 기본으로 `~/mosieung` 아래의 모델을 쓴다. 경로는 환경변수로 바꿀 수 있다.

| 환경변수 | 내용 |
| --- | --- |
| `ONMAEUM_STT_WHISPER` | Whisper CTranslate2 모델 폴더 |
| `ONMAEUM_STT_DIR` | SenseVoice 모델 폴더 (Whisper 대신) |
| `ONMAEUM_STT_FAST` | `0`이면 30초 인코딩 경로 사용 (기본 `1`) |
| `ONMAEUM_STT_THREADS` | STT 스레드 (기본 4) |
| `ONMAEUM_LLM_PATH` | `.rkllm` 파일. 같은 폴더에 `esc_demo`가 있어야 함 |
| `ONMAEUM_ESC_DEMO` | `esc_demo` 경로를 따로 지정 |
| `ONMAEUM_LLM_CT2` | CTranslate2 LLM 폴더 (PC 실행용) |
| `ONMAEUM_TTS` | `1`이면 TTS 사용 |
| `ONMAEUM_TTS_DIR` | VITS 모델 폴더 (`*.onnx`, `tokens.txt`, `espeak-ng-data/`) |
| `ONMAEUM_TTS_SPEED` | 말 빠르기 (기본 1.0) |
| `ONMAEUM_DATA_DIR` | 점검 기록 저장 위치 |

모델이 하나도 없으면 고정 응답으로 동작하므로 PC에서 API만 확인할 수 있다.

```bash
pip install -r requirements.txt
uvicorn main:app --host 0.0.0.0 --port 8080
```

sherpa-onnx 2.x는 espeak 발음 변환이 빠져 이 VITS 모델이 소리를 내지 못한다. 1.12.40을 쓴다.

부팅 시 자동 실행은 `deploy/onmaeum.service`를 `/etc/systemd/system/`에 복사한 뒤 `sudo systemctl enable --now onmaeum`으로 켠다.

## API

| 메서드 | 경로 | 내용 |
| --- | --- | --- |
| GET | `/api/health` | 상태, 로드된 모델 |
| GET | `/api/audio/{audioId}` | 안내 음성 WAV |
| GET, POST | `/api/escalators` | 설비 목록, 등록 |
| GET | `/api/escalators/{id}` | 설비 정보 |
| GET | `/api/escalators/{id}/histories` | 점검 이력 |
| GET | `/api/escalators/{id}/items` | 이번 점검 대상 항목 |
| POST | `/api/inspections` | 점검 시작 |
| GET | `/api/inspections/{iid}` | 점검 상태 |
| POST | `/api/inspections/{iid}/utterances` | 발화 (WAV 또는 텍스트) |
| POST | `/api/inspections/{iid}/undo` | 마지막 기록 취소 |
| PATCH | `/api/inspections/{iid}/items/{code}` | 항목 직접 수정 |
| POST | `/api/inspections/{iid}/finish` | 점검 종료 |
| GET | `/api/inspections/{iid}/result` | 결과 요약 |
| GET | `/api/inspections/{iid}/report` | 보고서 HTML |
| POST | `/api/inspections/{iid}/submit` | 제출 |

앱: [escalator-inspection-assistant](https://github.com/onmaeum-team/escalator-inspection-assistant)

## 라이선스 참고

TTS 모델 vits-mimic3-ko_KO-kss_low는 KSS 데이터셋 기반이라 비상업 용도로만 쓸 수 있다.
