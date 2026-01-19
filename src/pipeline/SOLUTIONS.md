# Dyarchy Pipeline Solutions

> 운영 중 발생한 문제와 해결책 모음
> 마지막 업데이트: 2026-01-18

---

## 1. GPU Segfault (libcuda.so crash)

### 증상
- Core 서버 10-20분마다 프리징/재부팅
- `journalctl`에서 `segfault at ... in libcuda.so` 메시지
- 다수 워커 동시 crash

### 원인
다중 워커의 동시 CUDA 컨텍스트 생성으로 GPU 드라이버 충돌

### 해결책

```yaml
# docker-compose.ralph-workers.yml - x-env-base
environment:
  # 핵심 (필수)
  CUDA_DEVICE_MAX_CONNECTIONS: "1"    # GPU당 동시 연결 1개로 직렬화
  CUDA_LAUNCH_BLOCKING: "1"           # GPU 연산 동기화 강제

  # 보조 (권장)
  PYTORCH_CUDA_ALLOC_CONF: "expandable_segments:True,max_split_size_mb:256"
  CUDA_DEVICE_ORDER: "PCI_BUS_ID"     # GPU ID 일관성
  PYTORCH_NO_CUDA_MEMORY_CACHING: "1" # PyTorch 캐시 비활성화
  CUDA_MODULE_LOADING: "LAZY"         # 지연 로딩
  CUDA_CACHE_DISABLE: "1"             # CUDA 캐시 비활성화
```

```yaml
# 워커 순차 시작 (depends_on chain)
pipeline-gpu0-w1:
  depends_on:
    pipeline-gpu0-w0:
      condition: service_healthy
```

### 검증
```bash
# 20분간 segfault 모니터링
ssh forge "ssh core 'journalctl -f | grep -i segfault'"
```

---

## 2. OOM (Out of Memory)

### 증상
- 워커 갑자기 종료
- `dmesg`에 `oom-killer` 메시지
- 호스트 시스템 불안정

### 해결책

```yaml
# 워커별 메모리 제한
deploy:
  resources:
    limits:
      memory: 12G

# OOM 시 워커 우선 종료 (호스트 보호)
oom_score_adj: 500  # 양수일수록 먼저 kill됨
```

```yaml
# 메모리 조기 반환
environment:
  MALLOC_TRIM_THRESHOLD_: "131072"
  PYTHONMALLOC: "malloc"
```

---

## 3. 좀비 프로세스 (Zombie Process)

### 증상
- `docker ps`에 exited 컨테이너 누적
- 시스템 프로세스 테이블 포화
- `defunct` 프로세스 다수

### 해결책

```yaml
# tini를 PID 1으로 사용
init: true  # Docker의 내장 tini 활성화
```

```bash
# Redis 시작 시 exec로 PID 1 확보
exec redis-server --appendonly yes ...
```

---

## 4. Redis AOF 손상

### 증상
- Redis 시작 실패
- `Bad file format reading the append only file` 에러
- 파이프라인 전체 중단

### 해결책

```bash
# Redis 시작 전 자동 복구 스크립트
AOF_MANIFEST="/data/appendonlydir/appendonly.aof.manifest"
if [ -f "$AOF_MANIFEST" ]; then
  if ! redis-check-aof "$AOF_MANIFEST" >/dev/null 2>&1; then
    echo y | redis-check-aof --fix "$AOF_MANIFEST" || true
  fi
fi
```

```yaml
# 완료 세트 재구축 (수동 복구)
docker compose run --rm rebuild-done
```

---

## 5. Core Dump 디스크 고갈

### 증상
- 디스크 용량 급격히 감소
- `/var/lib/systemd/coredump/` 비대
- 워커 crash 시 대용량 core 파일 생성

### 해결책

```yaml
ulimits:
  core:
    soft: 0
    hard: 0  # core dump 완전 비활성화
```

---

## 6. DNS 해결 실패

### 증상
- `pip install` 타임아웃
- 외부 패키지 다운로드 실패
- 컨테이너 내 네트워크 불안정

### 해결책

```yaml
dns: [8.8.8.8, 1.1.1.1]  # 외부 DNS 직접 지정
```

---

## 7. GPU 메모리 단편화

### 증상
- CUDA OOM 에러 (충분한 VRAM에도)
- `torch.cuda.OutOfMemoryError`
- 워커 재시작 필요

### 해결책

```yaml
environment:
  PYTORCH_CUDA_ALLOC_CONF: "expandable_segments:True,max_split_size_mb:256"
  PYTORCH_NO_CUDA_MEMORY_CACHING: "1"
```

---

## 8. 문제 파일 무한 재시도

### 증상
- 특정 PDF에서 반복 실패
- 큐 진행 정체
- 워커 리소스 낭비

### 해결책

```yaml
environment:
  MAX_RETRIES: "1"        # 3→1로 축소 (빠른 격리)
  TIMEOUT: "600"          # 파일당 최대 10분
  SAFE_MODE: "true"       # 문제 파일 skip 디렉토리로 이동
```

---

## 9. 워커 동시 시작 충돌

### 증상
- 파이프라인 시작 직후 다수 워커 crash
- GPU 초기화 경쟁 상태
- 불안정한 첫 수분

### 해결책

```yaml
# GPU별 워커 체인 (순차 시작)
# GPU0: w0 → w1 → w2 → w3 → w4
# GPU1: w0 → w1 → w2 → w3 → w4
# GPU2: w0 → w1

pipeline-gpu0-w1:
  depends_on:
    redis:
      condition: service_healthy
    pipeline-gpu0-w0:
      condition: service_healthy
```

---

## 10. CPU 핀닝 (NUMA 최적화)

### 설정
```yaml
# CCX0 (코어 0-5) → GPU0
# CCX1 (코어 6-11) → GPU1
# 하이퍼스레딩 페어 함께 할당

pipeline-gpu0-w0:
  cpuset: "0,12"   # 코어 0 + HT 12

pipeline-gpu1-w0:
  cpuset: "6,18"   # 코어 6 + HT 18
```

### 이점
- L3 캐시 지역성 향상
- NUMA 메모리 접근 최적화
- GPU-CPU 친화도 개선

---

## Quick Reference

| 문제 | 핵심 설정 |
|------|-----------|
| GPU Segfault | `CUDA_DEVICE_MAX_CONNECTIONS=1` |
| OOM | `oom_score_adj: 500` + `memory: 12G` |
| 좀비 프로세스 | `init: true` |
| Redis AOF 손상 | `redis-check-aof --fix` |
| Core Dump | `ulimits.core: 0` |
| DNS 실패 | `dns: [8.8.8.8, 1.1.1.1]` |
| GPU 메모리 단편화 | `PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True` |
| 무한 재시도 | `MAX_RETRIES=1` |
| 동시 시작 충돌 | `depends_on` chain |
| NUMA 최적화 | `cpuset` 핀닝 |

---

## 관련 파일

- [docker-compose.ralph-workers.yml](../docker-compose.ralph-workers.yml) - 파이프라인 설정
- [scripts/rebuild_done_set.py](../scripts/rebuild_done_set.py) - Redis 복구
- [src/pipeline/parsers/config_schema.py](../src/pipeline/parsers/config_schema.py) - 타임아웃 설정

---

*Dyarchy v3 Pipeline Operations*
