# pdf-yaml-pipeline 저장소 헌법

CONSTITUTION_VERSION: `1.0.0`

이 파일은 `pdf-yaml-pipeline` 저장소 안에서 가장 높은 규칙입니다. 이 저장소는 공개 Python PDF/HWP/HWPX-to-YAML 통합 제품의 명시적으로 검증된 경계만 소유하며, 다른 제품·개인 문서·운영 환경·저장소 이력을 모으는 루트나 범용 모노레포가 아닙니다.

## 우선순위와 상속

`REPOSITORY_LOCAL_SUPREMACY`

- 이 최상위 `AGENTS.md`, `PROVENANCE.md`, `SECURITY.md`, 검증 문서와 나머지 파일 순으로 적용합니다.
- 문서, 코드, workflow 또는 자동화는 이 헌법의 공개 안전성, provenance, hostile-input, 재현성 및 release 조건을 완화하거나 스스로 승인 상태로 승격할 수 없습니다.

`NESTED_RULES_MAY_ONLY_STRENGTHEN`

- 하위 `AGENTS.md`는 해당 영역의 규칙을 구체화하거나 강화할 수 있지만 이 헌법을 삭제·완화·우회·재정의할 수 없습니다.
- 충돌 시 더 강한 privacy, provenance, fail-closed, resource-limit 및 evidence 조건을 적용합니다.

## 제품 경계와 현재 HOLD

`INTEGRATED_LEGACY_PRODUCT_BOUNDARY`

- 이 저장소는 현재 공개 Python 구현의 direct PDF/HWP/HWPX parsing, OCR, YAML conversion, Redis worker, quality, RAG와 QA 표면을 하나의 통합 레거시 제품으로 보존합니다.
- 이름이나 일부 기능이 비슷하다는 이유로 별도 parser, normalizer, schema, queue runtime 또는 다른 저장소의 source authority가 되지 않습니다.
- 이 저장소를 대신하는 중복 원격을 만들거나 외부 제품을 이 저장소에 강제로 합치지 않습니다.

`CURRENT_PUBLICATION_AND_RELEASE_HOLD`

- 도달 가능한 공개 non-main ref의 운영 메타데이터와 legacy unsafe-deserialization 표면, 그리고 불완전한 인간 권리·출처 확인이 해결되기 전 source import, history merge, mirror, release, package publication과 production cutover를 승인하지 않습니다.
- 현재 `main`에서 파일이 보이지 않는다는 사실은 다른 공개 ref의 reachable history를 안전하게 만들지 않습니다.
- ref 삭제, history rewrite, force-push, archive 또는 rename은 공개 흔적을 회수했다는 증거가 아니며 명시적인 인간 소유자 승인과 incident 판단 없이 수행하지 않습니다.

`HUMAN_RELEASE_AUTHORITY_REQUIRED`

- AI agent, bot, 자동화 계정 또는 자기검증 결과만으로 copyright, publication rights, security acceptance, release 또는 disclosure 결정을 승인할 수 없습니다.
- 실제 권리를 가진 인간 소유자가 provenance와 공개 권한을 검토하고, 보안상 민감한 변경과 release를 명시적으로 승인해야 합니다.

## 공개 데이터와 provenance

`PUBLIC_DATA_FAIL_CLOSED`

- 실제 고객·응시자·분쟁·보험·의료·금융 문서, raw PDF/HWP/HWPX/OCR, 생성 YAML/JSONL/index/log, 원문·파일명·오류 전문, credential, private key, 내부 host·절대경로·운영 snapshot을 commit, fixture, artifact, issue 또는 PR에 포함하지 않습니다.
- fixture는 최소 synthetic generator 출력만 허용합니다. 공개 자료도 원본 URL, immutable revision, copyright, SPDX license, 재배포 근거, 비식별 검토와 hash가 없으면 사용하지 않습니다.
- 생성 output은 source code와 별개의 민감 데이터이며 공개 release 범위에 자동 포함되지 않습니다.

`IMPORT_PROVENANCE_REQUIRED`

- 외부 source import에는 원본 repository, exact commit과 blob, 원래 path, copyright owner, SPDX license, NOTICE 의무, 생성·변환 recipe, 변경 목록과 인간 reviewer를 기록합니다.
- private/internal history의 merge parent, submodule, bundle, mirror, `--all` push 또는 공통 조상을 공개 계보로 가져오지 않습니다.
- 승인된 import는 필요한 최소 파일을 새 public-human-owned commit으로 구성하고, 공개 범위에 필요하지 않은 history와 metadata를 포함하지 않습니다.

`NO_SOURCE_OR_HISTORY_DELETION_WITHOUT_APPROVAL`

- 이 저장소, 기존 branch/ref, 원본 repository, worktree 또는 source history를 새 경계가 생겼다는 이유로 삭제·이동·archive·rename·rewrite하지 않습니다.
- 보안 문제를 발견해도 증거를 임의로 지우지 않고 인간 소유자와 coordinated remediation 절차를 따릅니다.

## hostile document와 runtime 경계

`UNTRUSTED_DOCUMENT_LIMITS_REQUIRED`

- PDF, HWP, HWPX, ZIP, XML, OLE, image, FAISS index와 legacy metadata는 모두 hostile input으로 취급합니다.
- input bytes, page 수와 dimensions, rendered pixels, archive entry 수, entry·총 압축해제 bytes, compression ratio, XML depth/events/attributes/text, OLE stream/record/table/cell 수, output bytes, wall time, memory와 concurrency에 명시적 상한을 둡니다.
- 상한 초과, malformed, encrypted 또는 unsupported 입력은 native parser, model 또는 외부 process 실행 전에 typed error로 fail closed하고 부분 output을 성공으로 승격하지 않습니다.

`DESERIALIZATION_FAIL_CLOSED`

- untrusted pickle, arbitrary Python global, executable model metadata와 암묵적 legacy fallback을 금지합니다.
- legacy metadata가 필요하면 JSON 같은 비실행 형식, explicit opt-in, strict schema와 malicious `__reduce__` negative canary를 요구합니다.

`QUEUE_PATH_CONTAINMENT_REQUIRED`

- Redis나 다른 queue에서 받은 file name은 신뢰하지 않습니다. absolute path, drive/UNC prefix, `..`, NUL, symlink escape와 허용되지 않은 suffix를 거부합니다.
- input과 output은 open/write 직전에 resolved containment를 다시 확인합니다.
- Redis는 기본적으로 host network에 공개하지 않고 실제 server ACL/authentication과 최소 권한 network를 사용합니다.

`EXTERNAL_PROCESS_AND_NETWORK_OPT_IN`

- qpdf, LibreOffice, OCR/model runtime, webhook, sample download와 model registry 접근은 암묵적 fallback이 될 수 없습니다.
- 외부 실행은 고정 executable/version, argument-array 호출, 격리 tempdir, timeout과 process-tree 종료, artifact 정리를 요구합니다.
- network egress는 기본 비활성 또는 명시 allowlist이며, 파일명·경로·문서 내용·오류 전문을 webhook으로 전송하지 않습니다.

`CONTAINER_LEAST_PRIVILEGE_REQUIRED`

- hostile parser container는 non-root, read-only root filesystem, 최소 capability, no-new-privileges, 제한된 writable mount와 명시적 egress policy를 가져야 합니다.
- runtime image에는 불필요한 compiler, downloader와 package manager를 남기지 않습니다.

## 결정성, 공급망과 제품 주장

`DETERMINISTIC_PRIVATE_OUTPUT_REQUIRED`

- schema/version, ordering, newline, error semantics를 고정하고 clock, hostname, user name, absolute source/temp path와 file URI를 output에서 제거하거나 명시적으로 주입된 deterministic 값으로 바꿉니다.
- 동일 input/options를 서로 다른 clean path에서 반복 실행했을 때 canonical semantic result가 같아야 합니다.

`IMMUTABLE_SUPPLY_CHAIN_REQUIRED`

- dependency, container image, GitHub Action, direct VCS dependency와 model artifact는 immutable commit 또는 digest와 hash-checked lock으로 고정합니다.
- SBOM, license inventory, vulnerability scan과 offline/cache-only behavior가 없으면 reproducible 또는 supply-chain verified라고 주장하지 않습니다.

`SINGLE_VERSION_AND_ENTRYPOINT_AUTHORITY`

- package metadata, importable version, CLI, Docker entrypoint, README, artifact와 release tag는 하나의 권위에서 생성되어 일치해야 합니다.
- 존재하지 않는 module, 실행되지 않은 Quick Start, GPU-only image를 CPU 지원이라고 문서화한 상태에서는 runnable 또는 released라고 주장하지 않습니다.

`CLAIMS_REQUIRE_EVIDENCE`

- 지원 format, CPU/GPU mode, 안전성, 성능, 완전 보존과 호환성 주장은 clean-environment positive/negative evidence로 뒷받침해야 합니다.
- 과거 review report, commit subject, local syntax pass, clean mergeability 또는 README 문구는 hosted runtime/release 증거가 아닙니다.

## 하네스 진화

`HARNESS_EVOLUTION_REQUIRED`

- 새 parser failure, path/output leak, queue escape, unsafe deserialization, dependency advisory, provenance gap, entrypoint drift 또는 nondeterminism이 발견되면 같은 변경에서 synthetic fixture, negative canary, verifier, 문서와 CI를 함께 보강합니다.
- 실패한 test를 skip하거나 assertion, bound, privacy scan, provenance field 또는 supported-platform 조건을 삭제·완화해 통과시키지 않습니다.

`AGENTS_HARNESS_COHERENCE`

- 이 헌법의 machine-readable marker와 verifier/workflow는 같은 계약을 검사해야 합니다.
- 현재 공개-history와 rights HOLD가 해제되기 전에는 private/internal harness를 복사하지 않으며, 새 public harness도 별도 `verify/*` branch와 인간 보안 검토를 거칩니다.

`POSITIVE_AND_NEGATIVE_BEHAVIOR_REQUIRED`

- 승인된 format과 entrypoint에는 최소 synthetic 성공 case가 필요합니다.
- malformed/oversized input, ZIP/XML bomb, malicious pickle, queue traversal, symlink escape, dependency/model drift, forbidden corpus, path leak, external-tool/network failure는 명시적으로 거부되어야 합니다.

## branch와 worktree

`ROLE_SEPARATED_WORKTREES`

- `main`: 인간이 승인하고 검증한 공개 제품과 공통 거버넌스만 보유합니다.
- `governance/*`: 이 헌법, security, ownership와 provenance policy만 다룹니다.
- `verify/*`: 공개 verifier, synthetic fixture, workflow와 evidence만 다룹니다.
- `feature/*`: 승인된 한 제품 동작 또는 보안 수정만 다룹니다.
- `migration/*`: 별도 승인된 public-source import와 provenance mapping만 다룹니다.
- 역할 branch는 별도 sibling worktree를 사용하며 governance branch에서 제품 source, test, dependency, Docker 또는 runtime configuration을 변경하지 않습니다.

부모 통합 담당자만 공통 파일 충돌, branch merge 순서, PR 상태와 최종 correctness/security/release 판단을 소유합니다. 인간 소유자 승인이 필요한 결정을 자동화가 대신할 수 없습니다.

## 필수 검증과 완료 주장

변경 범위에 따라 최소한 다음을 수행합니다.

```text
all-public-ref history and internal-metadata scan
credential, raw-document and generated-output path scan
provenance, copyright, SPDX license, NOTICE and SBOM review
python package build and clean-wheel CLI/import smoke
synthetic parser positive and adversarial negative matrix
Redis authentication, queue traversal and symlink-containment tests
docker compose config and CPU/GPU end-to-end smoke
dependency, image, Action and model immutable-pin verification
git diff --check
git fsck --full --strict
```

`EXTERNAL_CI_AND_FRESH_CLONE_REQUIRED`

- merge 전 exact PR head의 GitHub-hosted jobs가 실제 runner와 steps를 만들고 지원 환경에서 통과해야 합니다.
- 별도 fresh clone에서 같은 exact integration commit을 검증하고, merge 후 실제 final `main` commit을 다시 fresh-clone 검증하기 전 완료, release, cutover 또는 history-safe를 선언하지 않습니다.
- account/billing lock, `startup_failure`, jobs 0, local-only PASS, draft review 생략, branch protection 부재와 clean mergeability는 성공이 아닙니다.
