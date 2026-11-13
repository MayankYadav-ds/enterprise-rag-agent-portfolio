# Production Deployment (Initial Packaging)

This document describes the first production packaging for a **single-owner,
public portfolio demonstration**. It is not a multi-user or tenant-isolated
deployment. Do not upload sensitive or private documents. Phase 10.5C-1 adds
container packaging only; backup and restore procedures are not implemented.

## Architecture

`compose.production.yaml` defines an API, a static frontend served by Caddy,
Qdrant, and Ollama. Only Caddy publishes ports 80 and 443 (including HTTP/3 on
UDP 443). Caddy obtains and renews HTTPS certificates for `RAG_PUBLIC_DOMAIN`;
public DNS must point to the host and inbound TCP ports 80/443 and UDP 443 must
be routed to it. Caddy's `/data` volume holds ACME certificates and private
keys. The API and Qdrant/Ollama communicate on internal-only Docker networks.
Qdrant and Ollama have no published host ports.

The API runs as UID/GID 10001, with one Uvicorn worker, a read-only root
filesystem, dropped Linux capabilities, and dedicated writable document and
Hugging Face cache volumes. PDF processing is scheduled with FastAPI's
in-process background task mechanism, not a durable queue. An abrupt API
restart can interrupt an accepted upload; inspect its document status and use
the documented recovery or re-upload procedure as appropriate.

Qdrant TLS is enabled and its official entrypoint is invoked through
`scripts/qdrant-entrypoint.sh`, which reads the API key from a mounted secret
file and exports it to the Qdrant entrypoint/process. The selected
`qdrant/qdrant:v1.19.2` image defaults to UID 0; the Compose service does not
override that identity. It has a read-only root filesystem, dropped Linux
capabilities, and `no-new-privileges`, but remains a root-running service. The
upstream image source supports a non-root build configuration; this Compose
file does not use it because the current Qdrant secret and volume ownership
model has not been validated for a non-root image.

The selected Ollama image is not hardened equivalently: Compose does not set a
user, read-only root filesystem, or capability restrictions for Ollama. Its
upstream `0.40.14` Dockerfile does not declare a non-root user, so the image's
default is root. It has no published host port and is attached only to the
internal data network, but it can write to its model volume and container root
filesystem. Treat further Ollama container hardening as an unvalidated item.
The Caddy service also leaves the user to the selected image default; it does
use a read-only root filesystem and drops capabilities except
`NET_BIND_SERVICE`. Confirm actual image identities during runtime validation.

The private `app_internal` network assigns Caddy `172.30.105.2` and the API
`172.30.105.3` in `172.30.105.0/29`. Uvicorn trusts forwarded headers only from
the Caddy address. Confirm this subnet does not conflict with host, VPN, or
other Docker networks before deployment. Login and upload limits use the
client address resolved by Uvicorn; direct clients cannot set that identity
through an untrusted `X-Forwarded-For` header.

`GET /ready` is the API container healthcheck and indicates storage, SQLite,
and Qdrant retrieval readiness only. It does not probe Ollama. The default
Qdrant request timeout is 10 seconds; the container healthcheck allows 20
seconds and its local HTTP client allows 15 seconds. If
`RAG_API_QDRANT_TIMEOUT` is increased, increase both healthcheck budgets as
well. A readiness failure is reported generically. `GET
/ready/generation` separately checks for the configured, already installed
Ollama model with a bounded `/api/tags` request. It never pulls or loads the
model. Both endpoints return only generic readiness status and are proxied by
Caddy.

## Prerequisites

- A Linux ARM64 host with Docker Compose v2 and enough memory/disk for the CPU
  embedding model, reranker, and selected Ollama model.
- The API service explicitly selects `linux/arm64`. The frontend, Qdrant, and
  Ollama services use the host's default platform. Run on Linux ARM64 and
  verify that each selected image tag publishes a compatible ARM64 manifest.
  Base and service images are tag-pinned, not digest-pinned; version tags do
  not provide immutable image provenance. Verify and record registry digests
  before production use.
- A DNS name and public reachability on the ports above for Caddy automatic
  HTTPS. Keep the host firewall closed to all other inbound ports.
- A production `.env` file based on `.env.example` containing the domain,
  trusted HTTPS origin, and an **already provisioned** Ollama model tag.
- A private deployment-secrets directory with the Qdrant API key, server
  certificate, server private key, and CA certificate at the paths named in
  the `.env` file. Create two separate Qdrant API-key files with the same key:
  set the API-only file owner to UID 10001/mode `0400`, and the Qdrant-only
  file owner to UID 0/mode `0400`. File-backed Compose secrets are bind mounts;
  Compose does not remap their host ownership or mode. Verify host ownership
  and mode before startup, and verify mounted-file metadata during the approved
  runtime validation without reading file contents. The Qdrant wrapper checks
  readability and secret format, but does not enforce owner or mode. Keep the
  Qdrant TLS private key root-owned with restrictive mode `0400`. The public CA
  and server certificate must be readable by their consumers. Back up CA and
  server credentials securely.
- A populated `hf_cache` named volume containing the exact approved embedding
  and reranker model snapshots. Containers run with Hugging Face offline mode;
  they will not fetch missing weights.
- An initialized Qdrant collection and a preloaded, non-sensitive sample
  corpus. The stack does not ingest a sample corpus automatically.
- The single owner account must be provisioned through the documented local
  console/bootstrap procedure before the service is made public.

Do not store `.env`, certificates, keys, or PDFs in the Docker build context.
The root `.dockerignore` excludes these paths and evaluation data. Never put
secret values in image build arguments or frontend variables.

## Configuration

Copy `.env.example` to `.env` and set the production values locally. Required
Compose inputs are:

| Variable | Purpose |
| --- | --- |
| `RAG_PUBLIC_DOMAIN` | Public DNS name used by Caddy for automatic HTTPS. |
| `RAG_TRUSTED_ORIGINS` | Exact HTTPS origin allowed by backend origin checks, e.g. `https://rag.example.com`. |
| `RAG_LLM_MODEL` | Exact model tag already present in the isolated Ollama volume. |
| `QDRANT_API_KEY_FILE` | Host path to the API-only copy of the Qdrant key (UID 10001, mode `0400`). |
| `QDRANT_SERVER_API_KEY_FILE` | Host path to the Qdrant-only copy of the same key (UID 0, mode `0400`). |
| `QDRANT_SERVER_TLS_CERT_FILE` | Host path to the Qdrant server certificate. |
| `QDRANT_SERVER_TLS_KEY_FILE` | Host path to the Qdrant server private key. |
| `QDRANT_CA_CERT_FILE` | Host path to the CA certificate trusted by the API. |

Compose mounts the API-only key file only into the API at
`/run/secrets/qdrant_api_key`. The API reads it via `QDRANT_API_KEY_FILE` and
rejects symlinks, non-regular files, group/world permission bits on POSIX, and
invalid content; the running UID must also be able to read it. The validator
does not directly assert owner UID 10001 or require the exact mode `0400`, so
those remain operator provisioning requirements. Compose mounts the separate
Qdrant-only copy at the same in-container path only in Qdrant; its wrapper
checks readability and format, then sets the server's supported
`QDRANT__SERVICE__API_KEY` environment setting. Keep the two host files
synchronized when rotating the key. The API trusts the
configured CA through `QDRANT_CA_CERT_FILE` and connects only to
`https://qdrant:6333`. There is no TLS downgrade path. Keep the CA certificate
and server certificate/key matched, including the `qdrant` DNS subject name;
readiness fails if Qdrant TLS or
authentication is unavailable.

## Owner account

The API image includes the existing interactive bootstrap and recovery scripts.
After the API is running, create the first owner from an authorized console;
passwords are entered at the prompt and are not command-line arguments:

```bash
docker compose --env-file .env -f compose.production.yaml exec -it api \
  python /app/scripts/create_user.py owner --admin
```

For local-console password recovery, which revokes the owner's existing
sessions:

```bash
docker compose --env-file .env -f compose.production.yaml exec -it api \
  python /app/scripts/reset_owner_password.py owner
```

Protect access to the Docker daemon and host console as administrator access.

## Review configuration and start

From the repository root, after creating the `.env`, model cache, Qdrant
certificates/key, collection, and sample corpus, first render the configuration
and check for missing variables or host paths:

```powershell
docker compose --env-file .env -f compose.production.yaml config --quiet
```

Then an operator may start the stack:

```powershell
docker compose --env-file .env -f compose.production.yaml up -d --build
docker compose --env-file .env -f compose.production.yaml ps
```

These commands are documentation only. No images or containers are built or
started during configuration/static validation. Do not run them until the
separate runtime-validation gate is approved.

## Persistent data and operations

Compose creates project-scoped named volumes for managed documents and the
SQLite registry, Hugging Face model cache, Qdrant storage and snapshots, Ollama
models, and Caddy ACME/configuration state. Do not rename the Compose project
after deployment: doing so creates a different set of volumes and can make
existing data appear missing. Do not use `docker compose down -v` or prune
commands against this project.

This phase provides no coordinated backup/restore, recovery drill, or
startup-recovery validation. Named volumes are not backups. Before handling
valuable data, implement and test the later backup/restore phase and maintain
encrypted off-host copies of the SQLite registry, uploaded PDFs, Qdrant
snapshots, model inventory, and TLS recovery material.

## Security and known limitations

- Public registration and multi-user account administration are not provided;
  deployment currently provisions the owner through the local console. API
  document access is scoped to the authenticated user's server-side identity,
  with no client-selected tenant or shared tenant membership. Existing unscoped
  SQLite rows and Qdrant points remain quarantined until an explicit, backed-up
  migration is reviewed and validated. This tenant behavior has only local
  in-memory test evidence; production Qdrant migration and runtime validation
  remain outstanding.
- Login rate limiting is persistent, but this deployment still needs an
  external edge-level request limit and monitoring for abusive traffic.
- No automated certificate/key rotation, backup/restore, disaster recovery,
  or deployment health alerting is supplied by this packaging stage.
- Qdrant's port is private to Docker networks. The host firewall should expose
  only Caddy. Verify the actual cloud firewall and routing separately.
- Image tags are not immutable digests. Pin verified digests before a
  production release; registry manifests and runtime image identities have
  not been validated in this packaging phase.
- Ollama starts without downloading model weights. Generation is unavailable
  until the exact configured model is provisioned into the project-scoped
  `ollama_models` volume by a separately approved operation.
- No multi-user, private-document, or public registration use is supported.
- Free-tier VM availability and pricing are provider-controlled; this Compose
  package does not guarantee a ₹0/month deployment.

Do not expose the demo publicly until the later runtime, TLS/authentication,
backup/restore, owner-account, and recovery validations have passed.
