# Security policy

## Current security status

No published version or public branch is currently certified for hostile or sensitive PDF, HWP, HWPX, ZIP, XML, image, FAISS-index, pickle, Redis-queue, or model input. The repository is under a release and history-review hold.

Do not run the current software with privileged credentials, an internet-exposed Redis service, unrestricted network egress, or access to sensitive documents solely because the repository is public or the README describes a supported workflow.

## Supported versions

| Surface | Status |
| --- | --- |
| `main` | Best-effort source baseline; not supported for hostile or sensitive inputs |
| Non-main and legacy refs | Unsupported; do not deploy or import their history |
| Published packages or container tags | No security-supported release is currently declared |

## Reporting a vulnerability

Private vulnerability reporting is enabled for this repository. Use [GitHub's private vulnerability report form](https://github.com/seunghyuoffice-design/pdf-yaml-pipeline/security/advisories/new). Do not open a public issue for a suspected vulnerability or operational-data disclosure.

Do not attach real documents, converted output, customer or exam data, filenames, absolute paths, Redis dumps, model indexes, credentials, internal host details, or full logs. Prefer minimal synthetic bytes, a generator, a redacted stack trace, and the affected commit.

Please include:

- the affected commit, branch, package/container reference, platform, and execution mode;
- whether the issue involves parser resource exhaustion, archive/XML handling, unsafe deserialization, queue/path escape, Redis exposure, external process or network behavior, dependency/model provenance, or output privacy;
- the smallest synthetic reproduction and expected versus observed behavior;
- whether files, queue state, logs, network requests, webhooks, or output artifacts were created after failure.

Pending approval by a verified human security owner, the proposed best-effort targets are acknowledgement within 7 calendar days, initial triage within 14 days, and status updates at least every 30 days while remediation is active. These are proposals, not a guaranteed service-level agreement.

## Deployment safety

- Treat every document, archive, index, queue item, filename, and metadata object as untrusted.
- Do not expose Redis on a public or untrusted network. Use actual server-side ACL/authentication, network isolation, and validated relative queue paths.
- Run parsers as a non-root user with a read-only root filesystem, narrow writable mounts, resource limits, and no network egress unless a reviewed capability explicitly requires it.
- Disable legacy pickle metadata. Prefer JSON with a strict schema; never load attacker-controlled pickle or model artifacts.
- Pin dependencies, container images, Actions, direct VCS dependencies, and model artifacts to immutable commits or digests and verify their hashes.
- Generated YAML/JSONL, indexes, logs, filenames, source paths, parser errors, and webhook payloads may contain sensitive source content and must be handled as private data.

## Disclosure and fixes

Security fixes must preserve evidence, add a synthetic positive or negative regression test as appropriate, and pass exact-head hosted CI plus fresh-clone verification before release. Tests, bounds, privacy scans, provenance fields, or supported-environment checks may not be removed merely to make a change pass.

Deleting a ref or rewriting history does not retract information that was already public. Any history remediation requires explicit human-owner approval, a coordinated disclosure decision, and a complete scan of every remaining public ref and tag.
