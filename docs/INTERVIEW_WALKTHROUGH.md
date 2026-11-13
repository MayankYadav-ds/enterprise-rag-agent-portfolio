# Interview walkthrough

This guide is a speaking aid. Describe what the code and tests actually do;
avoid presenting synthetic tests as real-model benchmarks or deployment
certification.

## 30-second introduction

“I built an evidence-first document question-answering system. It accepts PDF
documents, extracts page-aware text, chunks and indexes it, then combines
semantic and keyword retrieval. An optional cross-encoder reranks the evidence.
The API can ask a configured language model to select supporting excerpts, but
it validates the quote and citation against retrieved passages before returning
an answer. The React UI supports owner login, document lifecycle, retrieval,
and source inspection. I focused especially on tenant-scoped access and
fail-closed citation behavior.”

## Two-minute architecture explanation

“The browser calls a FastAPI service. Authentication is session-cookie based;
state-changing requests also need a CSRF token. The backend derives a tenant
scope from the authenticated user, so a request body cannot choose a different
tenant.

“An uploaded PDF is written beneath a server-owned per-tenant directory and
registered in SQLite. The document manager parses and chunks the PDF, uses the
configured embedder, then writes vectors and metadata payloads to Qdrant. SQLite
is authoritative for users, sessions, rate limiting, and API-managed document
lifecycle state. Qdrant is the vector and chunk-payload search store.

“At query time, dense retrieval uses embeddings; BM25 scores lexical overlap;
hybrid search fuses ranked lists with reciprocal rank fusion. A cross-encoder
can rerank candidates. The answer component receives those retrieved chunks and
asks the configured provider for structured evidence selections. It rejects
invalid IDs or quotes and renders citations from server-side source metadata.
When evidence is insufficient, the API returns an explicit insufficient-evidence
state. Citation validation checks provenance and quote matching, but it cannot
guarantee semantic entailment or perfect retrieval.”

## End-to-end demo narration

1. “I’m signing in through the actual session flow. The project has no public
   registration; the first owner is provisioned from the local console.”
2. “I’m uploading the included fictional two-page portfolio brief. The API
   validates the PDF and records a queued/processing state before indexing.”
3. “I’ll ask how many kits had been delivered by the end of July. The expected
   answer is stated on page 1, and I can expand the citation and retrieved
   passage to inspect it.”
4. “Now I’ll ask what the source says about future demand. It explicitly says
   that it provides no such forecast, so I check for an insufficient-evidence
   response rather than treating a plausible guess as a fact.”
5. “I can compare dense, BM25, or hybrid retrieval and optionally request
   reranking. These settings change retrieval behavior; the interface does not
   claim their scores are confidence percentages.”
6. “The synthetic file can be removed from the Documents screen after the
   walkthrough.”

Use [LOCAL_DEMO.md](LOCAL_DEMO.md) for prerequisites and commands. Phase 10.19
also verified the actual React application in Microsoft Edge against the actual
FastAPI routes over a same-origin loopback Vite proxy. Because Uvicorn was
unavailable, that run used a bounded TEST-ONLY ASGI-over-HTTP adapter and
deterministic in-memory vector, embedding, and generation adapters. It was not
a production-server, Qdrant, or real-model run. The five browser screenshots
are linked in the local demo guide.

For an interview, be precise about the distinction: “The UI and FastAPI
application were exercised together with real browser HTTP, but the external
retrieval and generation services were deterministic test doubles. I have not
claimed real-model quality or production deployment.”

## Design decisions

### Why combine dense and BM25 search?

Dense retrieval can match concepts expressed with different words. BM25 can
match exact names, identifiers, and uncommon terms. Hybrid RRF combines ranks
without pretending that the component score scales are comparable. It does not
automatically make every result better; retrieval must be measured on an
appropriate labeled corpus.

### Why use a cross-encoder?

A bi-encoder supports efficient candidate search by embedding a query and
documents separately. A cross-encoder jointly scores query-passage pairs and
can reorder a smaller candidate set at higher inference cost. The reranker is
optional, lazy-loaded, and its output is not a calibrated probability.

### How are citations validated?

The generator is given a bounded evidence set. Each selected evidence ID must
refer to a supplied passage, and its quote must match the source passage under
the implementation's validation rules. The API then builds citation metadata
from that passage. This prevents fabricated source IDs and nonmatching quotes
from being accepted; it does not prove that the quote logically entails the
answer.

### How is tenant isolation enforced?

The API takes tenant identity from the authenticated user's server-side record.
It scopes uploaded storage, SQLite document rows, Qdrant payload filters,
retrieval, answer evidence, status, recovery, and deletion. Unscoped legacy
records are hidden rather than automatically assigned. The current product
model is one user per corpus; shared organizations and cross-user administration
are not implemented.

### Why SQLite and Qdrant?

SQLite provides transactional local state for accounts, sessions, limiter
reservations, and document lifecycle metadata. Qdrant provides vector search and
chunk payload retrieval. There is no distributed transaction spanning them, so
the document manager uses a journal and compensating/recovery operations. That
does not coordinate arbitrary external writers; the documented single-writer
operating constraint matters.

## Difficult interview questions

**What happens if Qdrant succeeds but the SQLite update fails?**\
The document lifecycle records intent and uses compensating cleanup/recovery.
If cleanup cannot be confirmed, affected operations fail closed. This is not a
cross-system atomic transaction, and remote failure modes still need runtime
validation.

**Can an administrator read every user's files?**\
No. Admin status is for the documented recovery operation within the current
user's own tenant. There is no cross-tenant admin read path.

**Can a client submit `tenant_id` to query another corpus?**\
No. Request schemas reject unknown fields and route handlers derive scope from
the authenticated user. The client does not select the tenant.

**Does a valid citation mean the answer is correct?**\
No. It establishes that the citation refers to retrieved evidence and the quote
passes validation. Retrieval completeness, entailment, and model interpretation
remain separate quality questions.

**Why not return the model's free-form answer directly?**\
The current generator uses a structured evidence-selection contract so the
server can validate evidence IDs and quotes before rendering. It is intentionally
more constrained than open-ended synthesis.

**What are the main production gaps?**\
Real-model answer quality and latency, remote Qdrant TLS/auth and recovery,
container runtime and persistence, native ARM64 compatibility, and an operational
backup/restore rehearsal remain unverified. This project must not be called
production-ready on the basis of unit tests.

**What if an uploaded PDF is scanned?**\
There is no OCR pipeline. Image-only pages are reported as extraction limitations
rather than being represented as successfully understood text.

**Does a synthetic test prove tenant isolation in production Qdrant?**\
No. Synthetic tests exercise application filters and API boundaries with fake
or in-memory adapters. Remote Qdrant behavior and production deployment need
separate integration tests.

## Verified achievements and limits

### Verified in recorded local evidence

- Production retrieval, reranking, answer-contract, and citation-validation
  components were exercised by a deterministic synthetic regression suite.
- The recorded Phase 10.13 focused synthetic suite completed with 11 passed and
  no skips. Additional safe test selections passed, but their counts overlap.
- The Phase 10.15 in-process synthetic sample-PDF test exercises authenticated
  upload, production parsing/chunking, dense retrieval, citation validation,
  and deletion using deterministic test adapters at service/model boundaries.
  It does not use a browser, Qdrant, or real model inference.
- Phase 10.17 reran that synthetic PDF test and passed the API/auth/readiness
  selection (92 passed, one Starlette/HTTPX deprecation warning), plus 47
  retrieval/citation/provider tests. The two
  owner-provisioning failures from the prior run were test-environment
  contamination: an inherited empty `QDRANT_URL`; the tests now set an inert
  loopback URL and assert Qdrant is never initialized. An additional
  import-safe regression selection passed 87 tests, and the existing synthetic
  document-lifecycle selection passed 10. These selections overlap and are not
  a full backend suite.
- Phase 10.18 installed the locked React dependencies from the existing npm
  cache in offline mode with lifecycle scripts disabled. TypeScript checking,
  all 53 Vitest tests, and the production build passed. Six desktop/mobile
  Playwright tests passed against the actual React app and loopback Vite server
  with synthetic API route fixtures; this was not a browser-to-live-FastAPI
  run. Phase 10.19 then exercised the actual React UI over loopback HTTP against
  the actual FastAPI app, using a bounded test-only ASGI adapter and
  deterministic service adapters. The browser flow covered login, CSRF,
  synthetic PDF upload/processing, hybrid query/citations, tenant isolation,
  deletion, and post-delete abstention. Five screenshots and a request-level
  diagnostic trace were captured. The frontend Prettier script now flags only
  the unchanged `package.json`; scoped formatting passes. Ruff's repository-wide
  format check still flags the untouched `tests/test_chunking.py` and
  `tests/test_pdf_parser.py`.
- The synthetic cases checked multiple tenants, retrieval strategies, citations,
  abstention, and response-contract behavior. Their metrics describe those
  fixtures only.
- Authentication, CSRF, upload, document lifecycle, and tenant behavior have
  corresponding automated tests in the repository; those tests are not a
  substitute for a live deployment security review.

### Not verified by those results

- General retrieval quality on unseen or real documents.
- Real embedding, reranker, or LLM quality and performance in this environment.
- Remote Qdrant integration, TLS configuration against a server, or production
  backup and restore.
- ARM64 runtime, production images, public HTTPS deployment, or production
  readiness.
- Uvicorn/production ASGI-server behavior, remote Qdrant, real embeddings,
  reranking quality, and real LLM answer quality.
- Production images, native ARM64 runtime, public HTTPS deployment, and a
  production backup/restore rehearsal.

## Suggested closing statement

“The strongest evidence today is the tested application contract and
tenant-scoped synthetic pipeline. I can explain what is implemented and where
the boundaries are, but I do not claim that real-model quality or production
deployment has been validated. The browser demo used the actual React and
FastAPI application with deterministic test adapters. My next step is to
validate the same acceptance flow with an isolated Qdrant and already-cached
models, then record resource use and citation outcomes.”

## Resume-ready project bullets

Use only bullets that match the role and be ready to explain the test boundary:

- Built a FastAPI and React PDF question-answering application with session
  authentication, CSRF protection, per-user corpus isolation, and document
  lifecycle controls.
- Implemented dense, BM25, and hybrid reciprocal-rank retrieval with optional
  cross-encoder reranking and server-side quote/citation validation.
- Added offline synthetic regressions and real-browser integration coverage for
  PDF upload, tenant-scoped retrieval, citations, deletion, and evidence
  removal, using deterministic adapters where external ML services were not
  available.

Do not describe the synthetic browser run as a real-model benchmark, remote
Qdrant integration, or deployed production service.
