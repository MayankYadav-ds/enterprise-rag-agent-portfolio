# Local portfolio demonstration

This guide uses the real FastAPI application and React frontend. It does not
enable a mock backend. Use only a disposable local Qdrant collection and the
fictional sample PDF included at `examples/synthetic-portfolio-brief.pdf`.

## What the live demo requires

- Python 3.11 and the project API/embedding dependencies installed.
- A Qdrant service that is already running and reachable from this machine.
- A pre-created, isolated collection matching the configured embedding model
  and payload contract. The API does not create or recreate the collection.
- The embedding model already cached locally. Dense ingestion requires the
  configured embedder even when later searches use BM25.
- For generated answers, an already-running Ollama service and an already
  installed generation model. Retrieval-only mode does not need the generation
  provider.
- Node.js 22.12 or later and the lockfile-backed frontend dependencies. Phase
  10.18 installed these once from the existing npm cache using offline mode and
  disabled lifecycle scripts. A new checkout may not have the complete cache;
  a normal `npm ci` can access the package registry.

Do not point this demo at a production Qdrant collection or upload private
documents. The included sample is fictional. The app binds to loopback in the
commands below. Do not expose the development servers to a public or untrusted
network.

## 1. Configure the API process

From the repository root in PowerShell, set development-only paths and local
service endpoints. `.env` is not loaded automatically. The example below keeps
the API registry and uploaded files under an ignored local directory; confirm
that this path is disposable before using it.

```powershell
$env:RAG_ENV = 'development'
$env:RAG_DOCUMENT_DIR = Join-Path $PWD '.local-demo-data'
$env:QDRANT_URL = 'http://127.0.0.1:6333'
$env:RAG_API_COLLECTION = 'rag_chunks'
$env:HF_HUB_OFFLINE = '1'
$env:TRANSFORMERS_OFFLINE = '1'
$env:RAG_LLM_PROVIDER = 'ollama'
$env:RAG_LLM_BASE_URL = 'http://127.0.0.1:11434'
$env:RAG_LLM_MODEL = '<name of an already-installed local model>'
```

If Qdrant requires an API key, supply it through the supported local secret
mechanism (`QDRANT_API_KEY` or an owner-only `QDRANT_API_KEY_FILE`). Never put
the value in source control, a command transcript, or frontend configuration.
For a local Qdrant service, the default collection name is `rag_chunks`; it must
already exist and have the embedding dimensions and payload fields expected by
the selected model. Do not use a destructive collection-recreate option on data
you want to preserve.

Create the first owner before launching the API. This prompts for the password
without putting it in shell history. Owner bootstrap is allowed only while the
authentication database has no account:

```powershell
.venv\Scripts\python.exe scripts/create_user.py demo-owner --admin
```

Start one API worker on loopback:

```powershell
.venv\Scripts\python.exe -m uvicorn rag_agent.api.app:create_app --factory --host 127.0.0.1 --port 8000 --no-access-log
```

The live Uvicorn command remains untested through Phase 10.18; validation used
the in-process TestClient flow described below. The API does not start Qdrant or
Ollama, create a collection, or pull model weights. `/health` is liveness only;
`/ready` checks storage and retrieval infrastructure without loading models,
and `/ready/generation` separately checks that Ollama already lists the
configured model.

## 2. Start the browser workspace

In a second terminal:

```powershell
Set-Location .\frontend
# Works offline only when the exact lockfile dependency graph is already cached.
npm ci --offline --ignore-scripts --no-audit --no-fund
npm run dev
```

The Vite server binds to `127.0.0.1:5173` and proxies `/health`, `/auth`, and
`/api/v1` to `http://127.0.0.1:8000` by default. To use a different already
running local API, set `API_PROXY_TARGET` for the frontend process. Do not set
`VITE_API_BASE_URL` to a server-side secret or a public API key; the browser
must never receive service credentials. The cache-backed install and Vite
startup were exercised by the Phase 10.18 browser suite. Its auth, health,
document, and query endpoints were intercepted by test fixtures; this does not
verify a browser connected to the real FastAPI server.

## 3. Walk through the demo

1. Open `http://127.0.0.1:5173/` and sign in with the owner created above.
   There is no public registration form.
2. Open **Documents** and upload `examples/synthetic-portfolio-brief.pdf`.
   The UI reports transfer progress; the document row reports server processing
   state. A `202 Accepted` means queued, not indexed. Wait until the status is
   `ready` before asking a question.
3. Open **AI workspace** and ask “How many kits had been delivered by 31 July
   2026?” Inspect the cited source and page and expand the retrieved evidence.
   The fictional source says 93 kits had been delivered and 27 remained in
   storage. Ask what the brief says about future demand to see how the system
   handles missing evidence; the model may still make a mistake, so inspect the
   cited passage.
4. Select **Retrieve evidence only** to inspect search results without calling
   the generation provider. Compare dense, BM25, and hybrid retrieval only when
   the collection and embedding model are configured for those strategies.
5. Return to **Documents** to inspect lifecycle status. Delete only the
   synthetic sample and confirm the action in the UI.

Each question is independent; conversation history is not sent to the model.
The browser displays answer text and document text as text nodes, not injected
HTML. Citations are expandable source/page/quote records, not links to a PDF
viewer.

## Offline synthetic API demonstration

The targeted test `tests/test_phase_10_15_offline_pdf_demo.py` runs an in-process
FastAPI `TestClient` using the included PDF. It performs the real login/session
and CSRF flow, raw-PDF upload, document processing and parsing, chunking, dense
retrieval, answer schema/citation validation, and managed deletion. Deterministic
test-only embedding, vector-store, and answer-provider adapters replace the
external model/service boundaries. SQLite and uploaded files are confined to
pytest's temporary directory. No parallel fake FastAPI service or live-backend
fallback is created.

With the existing Python environment installed, the test can be run from the
repository root in PowerShell without starting services or downloading models:

```powershell
$env:PYTHONPATH = "$PWD/src"
$env:HF_HUB_OFFLINE = '1'
$env:TRANSFORMERS_OFFLINE = '1'
$env:PYTHONDONTWRITEBYTECODE = '1'
$testBase = Join-Path $env:TEMP ("phase-10.15-" + [guid]::NewGuid().ToString('N'))
.venv\Scripts\python.exe -m pytest -p no:cacheprovider -o addopts= --tb=short --basetemp $testBase tests/test_phase_10_15_offline_pdf_demo.py
```

This is a deterministic software integration regression, not an interactive
browser session or genuine AI inference. Its passing result must not be
described as real Qdrant, real-model, or production-runtime validation.

## Browser UI regression

With the frontend dependencies already installed, run:

```powershell
Set-Location .\frontend
$env:PLAYWRIGHT_OUTPUT_DIR = Join-Path $env:TEMP 'enterprise-rag-playwright-evidence'
npm run test:browser -- --workers=1
```

The Playwright configuration starts only the loopback Vite server on port
5175 and uses an already installed Microsoft Edge. The browser suite provides
synthetic auth/session, health, document, and query responses through Playwright
route fixtures. It validates the actual React UI and request construction, not
the live FastAPI/Qdrant/model stack. The Phase 10.18 run passed six desktop and
mobile tests and confirmed the Vite listener exited afterward.

## Phase 10.19 real-browser integration

Phase 10.19 also exercised the actual React application against the actual
FastAPI application over loopback HTTP. A clearly TEST-ONLY bounded ASGI
adapter was used because Uvicorn is not installed in the approved offline
environment. The adapter binds only to `127.0.0.1`, enforces request limits,
and does not replace authentication, CSRF, routes, document processing,
retrieval, or citation validation. The API process denies outbound network
connections and name resolution; its Qdrant constructor is explicitly made
fail-closed. Deterministic in-memory vector, embedder, and answer-provider
adapters were injected for this run. No Qdrant, Ollama, model weights, or
external network services were used.

The actual browser workflow passed login for two disposable tenant identities,
authenticated upload of the fictional two-page PDF, document-ready polling,
hybrid question answering with a matching filename/page/quote citation,
cross-tenant inventory and query isolation, rejection of a client-supplied
tenant ID, CSRF rejection for a state-changing request without its token,
document deletion, and a post-delete query with no evidence. An unscoped
legacy vector was present in the synthetic store and did not appear in either
tenant's answer. This verifies application wiring and security behavior with
test doubles; it does not measure real model quality or remote Qdrant behavior.

The API used a minimal TEST-ONLY ASGI-over-HTTP adapter, not Uvicorn or the
production container. The adapter buffers a bounded request body. Browser
requests were restricted to the two loopback listeners (`127.0.0.1:5175` for
Vite and `127.0.0.1:8765` for the API); the run recorded 23 API responses and
zero attempted external page requests. Six browser console resource-load
errors and five `net::ERR_ABORTED` request events were recorded alongside the
expected 401/403/422 negative checks and UI refresh/cancellation activity. The
runner did not establish a one-to-one cause mapping for those events, so they
remain warnings rather than being attributed conclusively. There were no page
exceptions.

The repository-local test-only runner can be invoked from PowerShell when the
approved Python environment, frontend dependencies, and Microsoft Edge are
already available. It does not install anything. Each run directory must be
new and outside the repository:

```powershell
$repo = (Resolve-Path .).Path
$evidence = Join-Path (Split-Path $repo -Parent) 'evidence'
$run = Join-Path $evidence ('phase-10.19-validation-' + [guid]::NewGuid().ToString('N'))
New-Item -ItemType Directory -Path $run | Out-Null
node .\frontend\tests\tools\phase_10_19_real_browser.cjs $repo $run
```

The script refuses to reuse its runtime-data directory and writes its logs,
browser response summary, and screenshots only to the supplied run directory.
It generates ephemeral synthetic account credentials in process memory and
does not print them. The test-only scripts are not application entry points
and must not be used as production servers.

Screenshots captured from the actual run are stored in
[`docs/screenshots/phase-10.19`](screenshots/phase-10.19/):

- [Authenticated workspace](screenshots/phase-10.19/01-authenticated-workspace.png)
- [Uploaded document ready](screenshots/phase-10.19/02-uploaded-document-ready.png)
- [Answer with citation](screenshots/phase-10.19/03-answer-with-citation.png)
- [Document deleted](screenshots/phase-10.19/04-document-deleted.png)
- [Mobile workspace](screenshots/phase-10.19/05-mobile-workspace-after-delete.png)

## Genuine inference versus test doubles

The browser path calls the configured API. PDF parsing, chunking, embeddings,
Qdrant search, optional cross-encoder scoring, and Ollama generation are genuine
only when those configured local components are available. The automated
synthetic suite substitutes deterministic fake embedders, vector storage,
cross-encoders, and LLM responses at test boundaries; it validates wiring and
contracts, not model quality. There is no silent fallback from the live API to
that fixture.

## Troubleshooting

- **Login is unavailable:** bootstrap the first owner before the first login;
  inspect only sanitized API status and avoid placing passwords in commands.
- **Upload accepted but not ready:** check the document status in the UI and
  the API's sanitized logs. A missing embedder, incompatible collection,
  malformed PDF, or Qdrant outage can prevent indexing.
- **HTTP 503 from retrieval:** verify the isolated Qdrant endpoint, collection,
  and model availability. `/ready` intentionally does not claim that generation
  is ready.
- **Answer request unavailable:** verify the configured Ollama endpoint and
  already-installed model. Do not run a model pull as an automatic recovery.
- **Need search without generation:** use “Retrieve evidence only”; this still
  requires indexed evidence, and dense/hybrid strategies still require their
  embedding model.
- **Login or mutation rejected:** keep the browser and API on the configured
  trusted origins. State-changing requests require the in-memory CSRF token.
- **Windows model load blocked:** this machine may enforce Application Control
  against PyTorch native libraries. Do not bypass that control; use the
  import-safe synthetic tests and report live inference as blocked.

This guide is not production deployment guidance. Production secrets, HTTPS,
backup/restore, remote Qdrant behavior, and container runtime require separate
validation; see `docs/DEPLOYMENT.md`.
