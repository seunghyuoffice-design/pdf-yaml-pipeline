# Public-source provenance and history review

This file records only public-repository evidence and a template for future imports. It is not proof that all copyright, publication-rights, dependency-license, or security questions have been resolved.

## Current public baseline

- Repository: `https://github.com/seunghyuoffice-design/pdf-yaml-pipeline`
- Visibility observed on 2026-08-31: `PUBLIC`
- Default branch: `main`
- Baseline commit: `4f09c2b47bb48255dde3ce92085a2466559e3c2b`
- Baseline root tree: `b47ddb1f64374bc8eb47f22b9b8495a50fd85d47`
- Baseline tracked paths: `106`
- Repository account owner: `@seunghyuoffice-design`
- Human publication-rights and release approver: `UNVERIFIED`
- Publication-rights approval: `UNVERIFIED`
- File-level source and dependency-license review: `UNVERIFIED`
- Current lifecycle: `HOLD_HUMAN_RIGHTS_HISTORY_SECURITY_REVIEW`

The current repository is preserved as an integrated public Python document-processing product. It is not an automatic source authority for another parser, normalizer, schema, queue runtime, or repository.

## Reachable-history boundary

The public remote has more than one root lineage. The non-main ref `feature/add-quality-models` at `c68886a904af2b6612759de71bc21abdf44ee04b` has no common ancestor with the current `main` lineage and remains `PRESERVED_REVIEW_REQUIRED`.

The scan observed only these refs on 2026-08-31:

- `refs/heads/main` at `4f09c2b47bb48255dde3ce92085a2466559e3c2b`
- `refs/heads/feature/add-quality-models` at `c68886a904af2b6612759de71bc21abdf44ee04b`
- sorted 13-commit reachable-set SHA-256, with one lowercase commit per LF-terminated line: `43a28c0f6eb1bc3d0f0c9ea414dcc3e667fad42ba7067ff57b4c035dc4e82743`

A targeted scan of those 13 commits found no tracked real document/model files, real `.env`, or the known credential signatures in the review pattern. This is not a comprehensive secret clearance and does not clear the history: the non-main lineage still contains operational metadata and legacy unsafe-deserialization behavior requiring human security and disclosure review. These findings apply only to the observed refs and become stale when any public ref or tag changes.

- Mirror, bundle, `--all` push, history merge, rebase, release, archive, deletion, force-push, and rewrite approval: `false`
- Import from the non-main lineage: `false`
- Claim that ref deletion retracts already-public content: `false`

## Future source-import record

Every imported file must have a reviewed row containing all of the following fields before it enters a migration branch:

| Field | Required value |
| --- | --- |
| Source repository | Public canonical URL |
| Source revision | Immutable commit or signed release |
| Source path | Exact original path |
| Source blob | Exact Git blob or artifact SHA-256 |
| Copyright owner | Human or legal entity with authority |
| License | SPDX identifier and full license/NOTICE obligations |
| Publication permission | Named human approver and dated evidence |
| Transformation | Reproducible recipe and resulting hash |
| Privacy review | Synthetic/public-data justification and reviewer |
| Security review | Threat boundary, tests, and reviewer |

Missing, inferred, AI-only, branch-name-only, or mutable-URL evidence is insufficient. Private/internal history, real documents, generated output, host metadata, credentials, and operational snapshots are never valid import inputs.

## Release transition

The HOLD may be lifted only after a human owner confirms publication rights and dependency provenance; every public ref and tag passes the approved history/privacy scan; unsafe legacy deserialization and operational-data exposure receive coordinated remediation; hostile-input, queue/path, container, supply-chain, package/CLI, and CPU/GPU runtime gates pass; exact-head hosted CI runs real jobs; and the final merged `main` passes a separate fresh-clone verification.
