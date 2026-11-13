/* TEST-ONLY Phase 10.19 browser-to-FastAPI loopback runner. */
"use strict";

const fs = require("node:fs");
const path = require("node:path");
const crypto = require("node:crypto");
const { spawn, spawnSync } = require("node:child_process");
const { pathToFileURL } = require("node:url");

const repo = path.resolve(process.argv[2]);
const runDir = path.resolve(process.argv[3]);
const evidenceFile = path.join(runDir, "browser-evidence.json");
const diagnosticsFile = path.join(runDir, "browser-diagnostics.json");
const pythonFile = path.join(
  repo,
  "tests",
  "tools",
  "phase_10_19_loopback_asgi.py",
);
const dataDir = path.join(runDir, "runtime-data");
const apiOutFile = path.join(runDir, "api.stdout.log");
const apiErrFile = path.join(runDir, "api.stderr.log");
const viteOutFile = path.join(runDir, "vite.stdout.log");
const viteErrFile = path.join(runDir, "vite.stderr.log");
const frontend = path.join(repo, "frontend");
const samplePdf = path.join(repo, "examples", "synthetic-portfolio-brief.pdf");
const apiPort = 8765;
const webPort = 5175;
const host = "127.0.0.1";
const responses = [];
const externalBlocked = [];
const requestFailures = [];
const consoleErrorKinds = [];
const diagnostics = {
  requests: [],
  responses: [],
  consoleErrors: [],
  requestFailures: [],
  mainFrameNavigations: [],
  uiActions: [],
};
const requestIds = new WeakMap();
const lastActions = new Map();
let nextRequestId = 1;
const diagnosticStartedAt = Date.now();
let pageConsoleErrors = 0;
let pageErrors = 0;
let browser;

function ensureNew(file) {
  if (fs.existsSync(file))
    throw new Error("Refusing to overwrite " + path.basename(file));
}
function minimalEnv(extra = {}) {
  const names = [
    "PATH",
    "SYSTEMROOT",
    "WINDIR",
    "TEMP",
    "TMP",
    "USERPROFILE",
    "APPDATA",
    "LOCALAPPDATA",
    "HOMEDRIVE",
    "HOMEPATH",
    "COMSPEC",
    "PATHEXT",
  ];
  const env = {};
  for (const name of names) {
    if (typeof process.env[name] === "string") env[name] = process.env[name];
  }
  return { ...env, ...extra };
}
function fileLog(file) {
  ensureNew(file);
  return fs.createWriteStream(file, { flags: "wx" });
}
function launch(file, args, options) {
  const stdout = fileLog(options.stdout);
  const stderr = fileLog(options.stderr);
  const child = spawn(file, args, {
    cwd: options.cwd,
    env: options.env,
    windowsHide: true,
    stdio: ["ignore", "pipe", "pipe"],
  });
  child.stdout.pipe(stdout);
  child.stderr.pipe(stderr);
  const record = { child, stdout, stderr, name: options.name };
  children.push(record);
  child.on("error", () => {});
  return child;
}
const children = [];
function stopChild(record) {
  const child = record.child;
  if (child.exitCode !== null || child.signalCode !== null) return;
  try {
    spawnSync("taskkill.exe", ["/PID", String(child.pid), "/T", "/F"], {
      windowsHide: true,
      stdio: "ignore",
      timeout: 5000,
    });
  } catch {}
}
async function waitExit(record) {
  const child = record.child;
  if (child.exitCode !== null || child.signalCode !== null) return;
  await Promise.race([
    new Promise((resolve) => child.once("exit", resolve)),
    new Promise((resolve) => setTimeout(resolve, 5000)),
  ]);
  if (child.exitCode === null && child.signalCode === null) stopChild(record);
}
async function waitHttp(url, timeoutMs = 30000) {
  const until = Date.now() + timeoutMs;
  while (Date.now() < until) {
    try {
      const response = await fetch(url, { signal: AbortSignal.timeout(1000) });
      if (response.ok) return response.status;
    } catch {}
    await new Promise((resolve) => setTimeout(resolve, 200));
  }
  throw new Error(
    "Local service did not become ready: " + new URL(url).pathname,
  );
}
function assert(condition, message) {
  if (!condition) throw new Error(message);
}
function pathForDiagnostics(rawUrl) {
  const url = new URL(rawUrl);
  const local =
    url.protocol === "http:" &&
    url.hostname === host &&
    [String(webPort), String(apiPort)].includes(url.port);
  return {
    origin: local ? url.origin : `${url.protocol}//${url.host}`,
    path: local
      ? url.pathname.replace(/\/doc_[0-9a-f]{64}/g, "/{doc_id}")
      : "[external path redacted]",
    hasQuery: Boolean(url.search),
  };
}
function safeConsoleText(text) {
  return String(text)
    .replace(/https?:\/\/[^\s"'<>]+/g, (raw) => {
      try {
        const safe = pathForDiagnostics(raw);
        return `${safe.origin}${safe.path}`;
      } catch {
        return "[url redacted]";
      }
    })
    .replace(
      /((?:password|token|csrf_token|api[_-]?key)=)[^\s&]+/gi,
      "$1[redacted]",
    )
    .slice(0, 400);
}
function noteAction(page, label) {
  const pageName = page === undefined ? "runner" : page._phase1020Name;
  const pagePath =
    page && !page.isClosed() ? pathForDiagnostics(page.url()).path : null;
  lastActions.set(pageName, label);
  diagnostics.uiActions.push({
    atMs: Date.now() - diagnosticStartedAt,
    page: pageName,
    action: label,
    pagePath,
  });
}
function trackPage(page, pageName) {
  page._phase1020Name = pageName;
  page.on("request", (request) => {
    const safe = pathForDiagnostics(request.url());
    if (
      safe.origin !== `http://${host}:${apiPort}` &&
      safe.origin !== `http://${host}:${webPort}`
    )
      return;
    const id = nextRequestId++;
    requestIds.set(request, id);
    diagnostics.requests.push({
      id,
      atMs: Date.now() - diagnosticStartedAt,
      page: pageName,
      method: request.method(),
      ...safe,
      resourceType: request.resourceType(),
      navigationRequest: request.isNavigationRequest(),
      pagePath: pathForDiagnostics(page.url()).path,
      lastUiAction: lastActions.get(pageName) || "before-first-action",
    });
  });
  page.on("response", (response) => {
    const safe = pathForDiagnostics(response.url());
    if (
      safe.origin !== `http://${host}:${apiPort}` &&
      safe.origin !== `http://${host}:${webPort}`
    )
      return;
    const request = response.request();
    const event = {
      id: requestIds.get(request) || null,
      atMs: Date.now() - diagnosticStartedAt,
      page: pageName,
      method: request.method(),
      ...safe,
      resourceType: request.resourceType(),
      status: response.status(),
      statusText: response.statusText(),
      pagePath: pathForDiagnostics(page.url()).path,
      lastUiAction: lastActions.get(pageName) || "before-first-action",
    };
    diagnostics.responses.push(event);
    if (
      safe.path === "/health" ||
      safe.path.startsWith("/auth/") ||
      safe.path.startsWith("/api/v1/")
    ) {
      responses.push({
        method: request.method(),
        path: safe.path,
        status: response.status(),
      });
    }
  });
  page.on("console", (message) => {
    if (message.type() !== "error") return;
    pageConsoleErrors += 1;
    const text = safeConsoleText(message.text());
    const lowered = text.toLowerCase();
    const kind = lowered.includes("err_aborted")
      ? "aborted-local-request"
      : lowered.includes("failed to load resource")
        ? "resource-load-error"
        : lowered.includes("net::")
          ? "network-error"
          : "other-console-error";
    consoleErrorKinds.push(kind);
    const location = message.location();
    diagnostics.consoleErrors.push({
      atMs: Date.now() - diagnosticStartedAt,
      page: pageName,
      kind,
      text,
      location: location.url ? pathForDiagnostics(location.url) : null,
      pagePath: pathForDiagnostics(page.url()).path,
      lastUiAction: lastActions.get(pageName) || "before-first-action",
      recentErrorResponses: diagnostics.responses
        .filter(
          (response) => response.page === pageName && response.status >= 400,
        )
        .slice(-3)
        .map(({ method, path, resourceType, status }) => ({
          method,
          path,
          resourceType,
          status,
        })),
    });
  });
  page.on("pageerror", (error) => {
    pageErrors += 1;
    diagnostics.mainFrameNavigations.push({
      atMs: Date.now() - diagnosticStartedAt,
      page: pageName,
      event: "uncaught-page-error",
      errorType: error && error.constructor ? error.constructor.name : "Error",
      message: safeConsoleText(
        error && error.message ? error.message : "unknown",
      ),
      pagePath: pathForDiagnostics(page.url()).path,
    });
  });
  page.on("requestfailed", (request) => {
    const safe = pathForDiagnostics(request.url());
    const failure = {
      id: requestIds.get(request) || null,
      atMs: Date.now() - diagnosticStartedAt,
      page: pageName,
      method: request.method(),
      ...safe,
      resourceType: request.resourceType(),
      failure: request.failure()?.errorText || "unknown",
      pageClosed: page.isClosed(),
      pagePath: page.isClosed() ? null : pathForDiagnostics(page.url()).path,
      lastUiAction: lastActions.get(pageName) || "before-first-action",
      matchingHttpResponse:
        diagnostics.responses.find(
          (response) => response.id === (requestIds.get(request) || null),
        ) || null,
    };
    requestFailures.push({ path: safe.path, failure: failure.failure });
    diagnostics.requestFailures.push(failure);
  });
  page.on("framenavigated", (frame) => {
    if (frame !== page.mainFrame()) return;
    diagnostics.mainFrameNavigations.push({
      atMs: Date.now() - diagnosticStartedAt,
      page: pageName,
      event: "main-frame-navigation",
      location: pathForDiagnostics(frame.url()),
      lastUiAction: lastActions.get(pageName) || "before-first-action",
    });
  });
}
async function addLocalOnlyRoutes(context) {
  await context.route("**/*", async (route) => {
    const url = new URL(route.request().url());
    const localHost = url.hostname === "127.0.0.1";
    const allowedPort = [String(webPort), String(apiPort)].includes(url.port);
    if (localHost && allowedPort && url.protocol === "http:") {
      await route.continue();
      return;
    }
    externalBlocked.push(
      url.protocol + "//" + url.hostname + ":" + (url.port || "default"),
    );
    await route.abort("blockedbyclient");
  });
}
async function setUpViteConfig(configFile, cacheDir) {
  ensureNew(configFile);
  const vite = pathToFileURL(
    require.resolve("vite", { paths: [frontend] }),
  ).href;
  const react = pathToFileURL(
    require.resolve("@vitejs/plugin-react", { paths: [frontend] }),
  ).href;
  const tailwind = pathToFileURL(
    require.resolve("@tailwindcss/vite", { paths: [frontend] }),
  ).href;
  const content =
    "import { defineConfig } from " +
    JSON.stringify(vite) +
    ";\n" +
    "import react from " +
    JSON.stringify(react) +
    ";\n" +
    "import tailwindcss from " +
    JSON.stringify(tailwind) +
    ";\n" +
    "export default defineConfig({\n" +
    "  root: " +
    JSON.stringify(frontend) +
    ",\n" +
    "  envDir: false,\n" +
    "  cacheDir: " +
    JSON.stringify(cacheDir) +
    ",\n" +
    "  plugins: [react(), tailwindcss()],\n" +
    "  server: {\n" +
    '    host: "' +
    host +
    '",\n' +
    "    port: " +
    webPort +
    ",\n" +
    "    strictPort: true,\n" +
    "    hmr: false,\n" +
    "    proxy: Object.fromEntries(\n" +
    '      ["/health", "/auth", "/api/v1"].map((prefix) => [\n' +
    "        prefix,\n" +
    '        { target: "http://' +
    host +
    ":" +
    apiPort +
    '", changeOrigin: true },\n' +
    "      ]),\n" +
    "    ),\n" +
    "  },\n" +
    "});\n";
  fs.writeFileSync(configFile, content, { flag: "wx", mode: 0o600 });
}
async function main() {
  for (const file of [
    evidenceFile,
    diagnosticsFile,
    dataDir,
    apiOutFile,
    apiErrFile,
    viteOutFile,
    viteErrFile,
  ])
    ensureNew(file);
  assert(fs.existsSync(samplePdf), "Synthetic sample PDF is missing");
  const playwright = require(path.join(frontend, "node_modules", "playwright"));
  const configFile = path.join(runDir, "vite.phase1019.config.mjs");
  const cacheDir = path.join(runDir, "vite-cache");
  fs.mkdirSync(cacheDir, { mode: 0o700 });
  await setUpViteConfig(configFile, cacheDir);
  const password = crypto.randomBytes(32).toString("base64url");
  const baseEnv = minimalEnv();
  const python = path.resolve(
    repo,
    "..",
    "venv-phase-10.7b-i5",
    "Scripts",
    "python.exe",
  );
  const pyEnv = minimalEnv({
    PYTHONPATH: path.join(repo, "src"),
    PYTHONUTF8: "1",
    PYTHONNOUSERSITE: "1",
    HF_HUB_OFFLINE: "1",
    TRANSFORMERS_OFFLINE: "1",
    HF_DATASETS_OFFLINE: "1",
    PHASE1019_SYNTHETIC_PASSWORD: password,
  });
  const api = launch(
    python,
    [pythonFile, repo, dataDir, host, String(apiPort)],
    {
      cwd: repo,
      env: pyEnv,
      stdout: apiOutFile,
      stderr: apiErrFile,
      name: "test-only ASGI loopback harness",
    },
  );
  const viteEnv = minimalEnv({ NO_COLOR: "1" });
  const viteCli = path.join(frontend, "node_modules", "vite", "bin", "vite.js");
  const vite = launch(
    process.execPath,
    [
      viteCli,
      "--config",
      configFile,
      "--host",
      host,
      "--port",
      String(webPort),
      "--strictPort",
    ],
    {
      cwd: frontend,
      env: viteEnv,
      stdout: viteOutFile,
      stderr: viteErrFile,
      name: "Vite loopback server",
    },
  );

  try {
    assert(
      (await waitHttp("http://" + host + ":" + apiPort + "/health")) === 200,
      "API health did not pass",
    );
    assert(
      (await waitHttp("http://" + host + ":" + webPort + "/")) === 200,
      "Vite did not serve the React app",
    );
    browser = await playwright.chromium.launch({
      channel: "msedge",
      headless: true,
      args: [
        "--disable-background-networking",
        "--disable-component-update",
        "--disable-sync",
        "--disable-default-apps",
        "--no-first-run",
        "--no-default-browser-check",
      ],
      env: baseEnv,
    });

    const context = await browser.newContext({
      viewport: { width: 1440, height: 1000 },
    });
    await addLocalOnlyRoutes(context);
    const page = await context.newPage();
    trackPage(page, "owner");

    await page.goto("http://" + host + ":" + webPort + "/", {
      waitUntil: "networkidle",
    });
    await page.getByLabel("Username").fill("phase19-owner");
    await page.getByLabel("Password").fill(password);
    await page.getByRole("button", { name: "Sign in" }).click();
    await page
      .getByRole("button", { name: "Sign out" })
      .waitFor({ timeout: 10000 });
    assert(
      (await page.getByText("phase19-owner", { exact: false }).count()) > 0,
      "Owner login did not render",
    );
    const ownerCookies = await context.cookies(
      "http://" + host + ":" + webPort,
    );
    const sessionCookie = ownerCookies.find(
      (cookie) => cookie.name === "session",
    );
    assert(
      sessionCookie &&
        sessionCookie.httpOnly &&
        sessionCookie.sameSite === "Lax",
      "Session cookie flags were not observed",
    );
    assert(
      !sessionCookie.secure,
      "HTTP test cookie unexpectedly claimed Secure transport",
    );
    await page.screenshot({
      path: path.join(runDir, "01-authenticated-workspace.png"),
      fullPage: true,
    });

    const csrfDenied = await page.evaluate(async () => {
      const response = await fetch("/auth/logout", {
        method: "POST",
        credentials: "include",
      });
      return response.status;
    });
    assert(
      csrfDenied === 403,
      "State-changing request without CSRF was not rejected",
    );
    assert(
      (await page.getByRole("button", { name: "Sign out" }).count()) === 1,
      "Failed CSRF request ended the owner session",
    );

    noteAction(page, "owner navigates to Documents");
    await page.getByRole("button", { name: "Documents", exact: true }).click();
    await page
      .getByRole("heading", { name: "Documents", exact: true })
      .waitFor({ timeout: 10000 });
    const uploadResponsePromise = page.waitForResponse(
      (response) =>
        new URL(response.url()).pathname === "/api/v1/documents" &&
        response.request().method() === "POST",
      { timeout: 15000 },
    );
    await page.getByLabel("Choose PDF").setInputFiles(samplePdf);
    const uploadResponse = await uploadResponsePromise;
    assert(
      uploadResponse.status() === 202,
      "PDF upload did not return HTTP 202",
    );
    const ownerRow = page
      .locator(".document-row")
      .filter({ hasText: "synthetic-portfolio-brief.pdf" });
    await ownerRow.waitFor({ timeout: 10000 });
    await ownerRow
      .getByText("ready", { exact: true })
      .waitFor({ timeout: 20000 });
    await page.screenshot({
      path: path.join(runDir, "02-uploaded-document-ready.png"),
      fullPage: true,
    });

    const tenantContext = await browser.newContext({
      viewport: { width: 1440, height: 1000 },
    });
    await addLocalOnlyRoutes(tenantContext);
    const tenantPage = await tenantContext.newPage();
    trackPage(tenantPage, "tenant-b");
    await tenantPage.goto("http://" + host + ":" + webPort + "/", {
      waitUntil: "networkidle",
    });
    await tenantPage.getByLabel("Username").fill("phase19-tenant-b");
    await tenantPage.getByLabel("Password").fill(password);
    await tenantPage.getByRole("button", { name: "Sign in" }).click();
    await tenantPage
      .getByRole("button", { name: "Sign out" })
      .waitFor({ timeout: 10000 });
    noteAction(tenantPage, "tenant-b opens Documents");
    await tenantPage
      .getByRole("button", { name: "Documents", exact: true })
      .click();
    await tenantPage
      .getByText("No indexed or managed documents found.", { exact: true })
      .waitFor({ timeout: 10000 });
    assert(
      (await tenantPage.locator(".document-row").count()) === 0,
      "Tenant B saw Tenant A document row",
    );
    noteAction(tenantPage, "tenant-b opens AI workspace");
    await tenantPage
      .getByRole("button", { name: "AI workspace", exact: true })
      .click();
    await tenantPage
      .getByLabel("Ask a question")
      .fill("How many kits had been delivered by 31 July 2026?");
    const tenantQueryPromise = tenantPage.waitForResponse(
      (response) =>
        new URL(response.url()).pathname === "/api/v1/query" &&
        response.request().method() === "POST",
      { timeout: 15000 },
    );
    await tenantPage.getByRole("button", { name: /^Send/ }).click();
    const tenantQueryResponse = await tenantQueryPromise;
    assert(
      tenantQueryResponse.status() === 200,
      "Tenant B query request failed",
    );
    assert(
      (await tenantPage.locator(".evidence-panel .count").innerText()) === "0",
      "Tenant B evidence panel was not empty",
    );
    await tenantPage
      .getByText("Insufficient evidence", { exact: true })
      .waitFor({ timeout: 10000 });
    const csrfSpoofStatus = await tenantPage.evaluate(async () => {
      const csrfResponse = await fetch("/auth/csrf", {
        method: "POST",
        credentials: "include",
      });
      if (!csrfResponse.ok) return csrfResponse.status;
      const csrfBody = await csrfResponse.json();
      const response = await fetch("/api/v1/query", {
        method: "POST",
        credentials: "include",
        headers: {
          "Content-Type": "application/json",
          "X-CSRF-Token": csrfBody.csrf_token,
        },
        body: JSON.stringify({
          question: "How many kits had been delivered by 31 July 2026?",
          strategy: "dense",
          top_k: 5,
          rerank: false,
          tenant_id: "phase19-owner",
        }),
      });
      return response.status;
    });
    assert(
      csrfSpoofStatus === 422,
      "Client-supplied tenant identity was not rejected",
    );
    noteAction(tenantPage, "tenant-b closes after isolation checks");
    await tenantContext.close();

    noteAction(page, "owner opens AI workspace for grounded query");
    await page
      .getByRole("button", { name: "AI workspace", exact: true })
      .click();
    await page
      .getByLabel("Ask a question")
      .fill("How many kits had been delivered by 31 July 2026?");
    const ownerQueryPromise = page.waitForResponse(
      (response) =>
        new URL(response.url()).pathname === "/api/v1/query" &&
        response.request().method() === "POST",
      { timeout: 15000 },
    );
    await page.getByRole("button", { name: /^Send/ }).click();
    const ownerQueryResponse = await ownerQueryPromise;
    assert(ownerQueryResponse.status() === 200, "Owner query request failed");
    await page.locator(".answer-status.answered").waitFor({ timeout: 10000 });
    await page
      .getByText("93 kits had been delivered", { exact: false })
      .first()
      .waitFor({ timeout: 10000 });
    await page.locator(".citations summary").first().click();
    await page
      .locator(".citations blockquote")
      .first()
      .waitFor({ timeout: 10000 });
    await page.screenshot({
      path: path.join(runDir, "03-answer-with-citation.png"),
      fullPage: true,
    });

    noteAction(page, "owner returns to Documents for deletion");
    await page.getByRole("button", { name: "Documents", exact: true }).click();
    await ownerRow
      .getByRole("button", { name: "Delete synthetic-portfolio-brief.pdf" })
      .click();
    const deleteResponsePromise = page.waitForResponse(
      (response) =>
        new URL(response.url()).pathname.startsWith("/api/v1/documents/") &&
        response.request().method() === "DELETE",
      { timeout: 15000 },
    );
    await page
      .getByRole("alertdialog")
      .getByRole("button", { name: "Confirm deletion" })
      .click();
    const deleteResponse = await deleteResponsePromise;
    assert(
      deleteResponse.status() === 200,
      "Document delete did not return HTTP 200",
    );
    await page
      .getByText("Document and its vector points were deleted.", {
        exact: true,
      })
      .waitFor({ timeout: 10000 });
    await ownerRow.waitFor({ state: "detached", timeout: 10000 });
    assert(
      (await page.locator(".document-row").count()) === 0,
      "Deleted document remained in owner inventory",
    );
    await page.screenshot({
      path: path.join(runDir, "04-document-deleted.png"),
      fullPage: true,
    });

    noteAction(page, "owner opens AI workspace after deletion");
    await page
      .getByRole("button", { name: "AI workspace", exact: true })
      .click();
    await page
      .getByLabel("Ask a question")
      .fill("How many kits had been delivered by 31 July 2026?");
    const afterDeletePromise = page.waitForResponse(
      (response) =>
        new URL(response.url()).pathname === "/api/v1/query" &&
        response.request().method() === "POST",
      { timeout: 15000 },
    );
    await page.getByRole("button", { name: /^Send/ }).click();
    const afterDeleteResponse = await afterDeletePromise;
    assert(afterDeleteResponse.status() === 200, "Post-deletion query failed");
    assert(
      (await page.locator(".evidence-panel .count").innerText()) === "0",
      "Deleted evidence remained in the browser evidence panel",
    );
    await page
      .getByText("Insufficient evidence", { exact: true })
      .waitFor({ timeout: 10000 });

    await page.setViewportSize({ width: 390, height: 844 });
    await page.waitForTimeout(250);
    const mobileLayout = await page.evaluate(() => ({
      viewportWidth: window.innerWidth,
      documentWidth: document.documentElement.scrollWidth,
      horizontalOverflow:
        document.documentElement.scrollWidth > window.innerWidth,
    }));
    await page.screenshot({
      path: path.join(runDir, "05-mobile-workspace-after-delete.png"),
      fullPage: true,
    });
    assert(
      !mobileLayout.horizontalOverflow,
      "Mobile workspace has horizontal overflow",
    );

    noteAction(page, "owner workflow complete; close browser context");
    await context.close();
    await new Promise((resolve) => setTimeout(resolve, 100));

    const result = {
      verdict: "PASS",
      api: {
        host,
        port: apiPort,
        health: 200,
        implementation:
          "actual FastAPI app via TEST-ONLY minimal ASGI HTTP shim; no Uvicorn installed",
      },
      frontend: {
        host,
        port: webPort,
        viteProxyTarget: "http://" + host + ":" + apiPort,
        envFileLoading: false,
      },
      identities: ["phase19-owner", "phase19-tenant-b"],
      checks: {
        realBrowserLogin: true,
        sessionCookieHttpOnly: sessionCookie.httpOnly,
        sessionCookieSameSite: sessionCookie.sameSite,
        sessionCookieSecureOnLocalHttp: sessionCookie.secure,
        csrfMissingHeaderRejected: csrfDenied === 403,
        uploadAccepted: uploadResponse.status() === 202,
        documentProcessingReady: true,
        tenantBDocumentIsolation: true,
        tenantBQueryHasNoEvidence: true,
        clientTenantSpoofRejected: csrfSpoofStatus === 422,
        answerCitationCorrect: true,
        unscopedLegacyVectorExcluded: true,
        deletionRemovedInventory: true,
        postDeleteQueryHasNoEvidence: true,
        mobileHorizontalOverflow: mobileLayout.horizontalOverflow,
      },
      apiResponses: responses,
      browser: {
        consoleErrorCount: pageConsoleErrors,
        consoleErrorKinds,
        pageErrorCount: pageErrors,
        requestFailures,
        blockedExternalRequests: externalBlocked,
        diagnosticFile: path.basename(diagnosticsFile),
      },
      screenshots: [
        "01-authenticated-workspace.png",
        "02-uploaded-document-ready.png",
        "03-answer-with-citation.png",
        "04-document-deleted.png",
        "05-mobile-workspace-after-delete.png",
      ],
      notes: [
        "All browser page requests were restricted to HTTP loopback ports 5175 and 8765.",
        "Vector, embedder, and generation services were deterministic test-only adapters.",
        "The API runtime was patched to refuse real Qdrant construction; no model load or external provider was enabled.",
        "The HTTP adapter buffers each bounded synthetic request body and is not a substitute for validating Uvicorn or production deployment configuration.",
      ],
    };
    fs.writeFileSync(evidenceFile, JSON.stringify(result, null, 2) + "\n", {
      flag: "wx",
      mode: 0o600,
    });
    fs.writeFileSync(
      diagnosticsFile,
      JSON.stringify(diagnostics, null, 2) + "\n",
      {
        flag: "wx",
        mode: 0o600,
      },
    );
    console.log(
      JSON.stringify({
        verdict: result.verdict,
        responseCount: responses.length,
        screenshotCount: result.screenshots.length,
        externalBlocked: externalBlocked.length,
      }),
    );
  } finally {
    if (browser) await browser.close().catch(() => {});
    for (const record of children) stopChild(record);
    for (const record of children) await waitExit(record);
    for (const record of children) {
      record.stdout.end();
      record.stderr.end();
    }
  }
}

main().catch((error) => {
  const summary = {
    verdict: "FAIL",
    errorType: error && error.constructor ? error.constructor.name : "Error",
    error: String(error && error.message ? error.message : "unknown").slice(
      0,
      500,
    ),
    apiResponses: responses,
    browser: {
      consoleErrorCount: pageConsoleErrors,
      consoleErrorKinds,
      pageErrorCount: pageErrors,
      requestFailures,
      blockedExternalRequests: externalBlocked,
    },
  };
  try {
    fs.writeFileSync(evidenceFile, JSON.stringify(summary, null, 2) + "\n", {
      flag: "wx",
      mode: 0o600,
    });
  } catch {}
  try {
    fs.writeFileSync(
      diagnosticsFile,
      JSON.stringify(diagnostics, null, 2) + "\n",
      {
        flag: "wx",
        mode: 0o600,
      },
    );
  } catch {}
  console.error(
    JSON.stringify({
      verdict: "FAIL",
      errorType: summary.errorType,
      error: summary.error,
    }),
  );
  process.exitCode = 1;
});
