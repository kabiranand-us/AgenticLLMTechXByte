# Plan: Model Usage & Error-Rate Dashboard (Prometheus + Grafana)

## Goal

Build a Grafana dashboard showing:
1. **Which model was invoked, how many times** (totals, broken down by provider/model)
2. **Error percentage** (failed requests / total requests)
3. **Requests per model over a time window** (time-series graph — "which model was hit how many times" over time, not just a total)

## Why this isn't a Grafana-only task

Grafana doesn't store data itself — it only **visualizes** data that already exists in a data
source (Prometheus, in your case, since it's already running on the VPS alongside Elasticsearch
and Kibana). Right now, `llm-gateway` has **zero metrics**. It only writes plain stdout logs
(captured by `docker logs`), with no structured counters for "model X was called Y times" or
"request succeeded/failed." Prometheus has nothing to scrape from this app today.

So before any dashboard can exist, the app itself needs to start **emitting metrics** that
Prometheus can pull. The full chain looks like:

```
llm-gateway (FastAPI app)
   |  exposes /metrics endpoint (counter: requests by model, by status)
   v
Prometheus (already running on VPS, port 9090, proxied at /prometheus/)
   |  scrapes llm-gateway:8000/metrics every N seconds, stores time-series data
   v
Grafana (already running on VPS, port 3000, proxied at /grafana/)
   |  queries Prometheus, renders panels/graphs
   v
Dashboard you actually look at
```

## Decisions made

- **Error granularity:** success/error only (not split by failure type). Simpler, answers the
  3 asks directly; finer-grained labels (rate-limit vs invalid-key vs other) can be added later
  without breaking this dashboard.
- **Dashboard delivery:** provisioned as a JSON file (Grafana dashboard-as-code), not built by
  hand in the UI — survives container recreation/redeploys instead of living only in Grafana's
  database/volume.

## Step-by-step plan

### Step 1 — Instrument the FastAPI app with Prometheus metrics

Add the `prometheus-client` Python package and define one metric in `llm_service.py`:

- **`llm_requests_total`** — a `Counter`, labeled by `provider`, `model`, and `status`
  (`success` / `error`). Every time `invoke_with_fallback()` resolves (success or final
  failure), increment this counter with the right labels. This single metric answers all three
  asks: total-by-model (sum the counter), error percentage (ratio of `status="error"` to total),
  and requests-over-time (apply `rate()` in the Grafana query).

Expose a `/metrics` endpoint on the FastAPI app via `prometheus-client`'s ready-made ASGI
handler — a few lines, not a custom implementation.

**Where to instrument:** inside `invoke_with_fallback()` in `llm_service.py`, since that's the
single chokepoint every chat/blog request already passes through (added in commit `7e74dfe`).
Wrapping there means we don't need to duplicate instrumentation in both `main.py` and
`content_engineer.py`.

### Step 2 — Expose the metrics port

The container already runs on port `8000` internally (mapped to `8089` externally via
`docker-compose.yml`). `/metrics` will be served on that same port — no new port mapping needed,
since Prometheus reaches it over the internal `techbyte-net` Docker network as
`llm-gateway:8000`, not through the public `8089` mapping.

### Step 3 — Add a Prometheus scrape target

Edit `/home/kabir/TechByteApp/prometheus/prometheus.yml` on the VPS to add a new scrape job:

```yaml
scrape_configs:
  - job_name: 'llm-gateway'
    static_configs:
      - targets: ['llm-gateway:8000']
```

Then reload Prometheus's config without a full restart:

```bash
curl -X POST http://localhost:9090/-/reload
```

(This works because the container already runs with `--web.enable-lifecycle`, visible in its
existing `command:` block.)

### Step 4 — Verify Prometheus is actually scraping it

Visit `https://monitor.techxbytes.com/prometheus/targets` and confirm the `llm-gateway` job
shows `UP`. If it shows `DOWN`, the likely cause is `llm-gateway` and `prometheus` not sharing
the same Docker network (`techbyte-net`) — worth checking `docker-compose.yml` for both
services' `networks:` blocks before assuming anything more complex is wrong.

### Step 5 — Build the Grafana dashboard (provisioned as JSON)

Three panels, each backed by a PromQL query against `llm_requests_total`:

1. **Total invocations by model** (bar chart or pie chart)
   ```promql
   sum by (provider, model) (llm_requests_total)
   ```

2. **Error percentage** (stat panel, single number with a threshold/color)
   ```promql
   sum(llm_requests_total{status="error"}) / sum(llm_requests_total) * 100
   ```

3. **Requests per model over time** (time-series line graph)
   ```promql
   sum by (model) (rate(llm_requests_total[5m]))
   ```
   (`rate()` over a 5-minute window turns the raw counter into "requests per second," which
   Grafana then renders as a readable trend line — a raw cumulative counter on its own would
   just be a constantly climbing line, not useful for "how many in a time frame.")

The dashboard JSON gets dropped into Grafana's dashboard-provisioning directory on the VPS so it
loads automatically on container start, rather than being created once by hand and lost if the
Grafana volume/container is ever recreated.

### Step 6 — Ship it through the agreed workflow

Per the standing rule for this repo: branch with a meaningful name (`feature/prometheus-metrics`)
→ commit → push → PR → review → merge. CI/CD then builds, pushes, and redeploys automatically.
The Prometheus config + Grafana dashboard provisioning changes happen directly on the VPS
(they're infra, not part of this git repo), so those are separate manual steps documented above,
not part of the PR diff.
