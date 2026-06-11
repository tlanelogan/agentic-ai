# Beaver's Choice / Munder Difflin — Multi-Agent Paper Company

A multi-agent system that automates order handling for a fictional paper company.
For each free-text customer request it parses the order, maps fuzzy item names to a
canonical catalog, checks stock and reorders when needed (within budget and the
customer's deadline), prices the order with bulk discounts grounded in historical
quotes, finalizes the affordable/in-stock lines as sales, and returns one transparent,
customer-safe reply.

Built with **smolagents** (`ToolCallingAgent` + `OpenAIServerModel`) on `gpt-4o-mini`
via an OpenAI-compatible proxy, in a single file. This README doubles as the project's
design write-up and evaluation report.

**In this repo:** `template.py` (the full single-file system) · `agent_workflow_diagram.png`
(architecture) · `test_results.csv` (evaluation output over the 20-request sample) ·
`quote_requests_sample.csv` / `quotes.csv` / `quote_requests.csv` (data) ·
`.env.example` (required environment variables).

---

## 1. System overview

The system automates order handling for a fictional paper company. For each
incoming customer request it: parses the free-text order, maps fuzzy item names
to the canonical catalog, checks stock and reorders when needed (within budget
and the customer's deadline), prices the order with bulk discounts grounded in
historical quotes, finalizes the affordable/in-stock lines as sales, and returns
one transparent, customer-safe reply. It is built on **smolagents**
(`ToolCallingAgent` + `OpenAIServerModel`) against `gpt-4o-mini` via the Vocareum
proxy, in a single file (`template.py`), and runs the full
`quote_requests_sample.csv` set (20 requests), emitting `test_results.csv`.

## 2. Agent workflow & architecture

The architecture (see `agent_workflow_diagram.png`) is one **orchestrator** plus
four **worker agents**, each with non-overlapping responsibilities and its own
tools. Every one of the 7 sanctioned database helper functions is wrapped in at
least one tool:

| Agent | Responsibility | Tools → helper function |
|---|---|---|
| **Orchestrator** | Parse + normalize the request, run the supervised flow, assemble the reply | 4 delegation tools (`delegate_inventory/quoting/sales/advisor`) |
| **Inventory** | Stock checks and restock decisions | `check_full_inventory`→`get_all_inventory`; `check_item_stock`→`get_stock_level`; `estimate_restock_eta`→`get_supplier_delivery_date`; `place_restock_order`→`create_transaction` |
| **Quoting** | Pricing with bulk discounts, grounded in history | `find_similar_quotes`→`search_quote_history`; `price_line_item` (catalog price + discount ladder) |
| **Sales / Ordering** | Finalize transactions, confirm delivery | `verify_stock`→`get_stock_level`; `check_funds`→`get_cash_balance`; `record_sale`→`create_transaction`; `confirm_delivery_date`→`get_supplier_delivery_date` |
| **Business Advisor** | Financial snapshot + internal recommendations | `financial_snapshot`→`generate_financial_report`; `cash_position`→`get_cash_balance` |

### The key decision: hybrid orchestration

The most important design choice was **deterministic orchestration with agentic
workers**. The orchestrator is a genuine `ToolCallingAgent` that exposes four
delegation tools (matching the diagram), but the per-request control flow lives
in a plain-Python `process_request()` method, not in an LLM "figure it out" loop.
Parsing, item-name normalization, unit conversion, and the
reorder/affordability/deadline decisions are deterministic Python; the worker
agents are invoked at the points where their tools must act.

The rationale: across 20 stateful requests, letting `gpt-4o-mini` freelance a
five-agent dance is fragile and token-hungry — small extraction or routing
mistakes compound and silently corrupt the shared database. Pushing the
deterministic work into Python and reserving the agents for tool execution and
judgment made the run reliable and auditable, while keeping the heavy lifting
verifiable.

Two further reliability mechanisms back this up:

- **SQLite as the single source of truth.** All state lives in the
  `transactions` table; the system never keeps a parallel in-memory inventory
  cache. After each request the run loop recomputes finances from the DB.
- **Memory-inspection fallback.** After the sales agent runs, the orchestrator
  inspects the agent's tool-call memory to see which `record_sale` calls actually
  executed, and deterministically fills any gap exactly once — so a flaky model
  step degrades gracefully into a correct sale rather than a lost one, with no
  double-counting.

## 3. Multi-Agent Workflow and Industry best practices applied

- **Failures are expected, explicit, and structured.** Every tool returns a
  `ToolResult` dict (`status` / `error_type` / `data` / `message`) and wraps its
  helper call in `try/except`; raw exceptions and internal text never reach the
  customer.
- **Graceful degradation.** A line that can't be fulfilled (not carried,
  deadline infeasible, unaffordable restock) is declined with a clear reason
  while the rest of the order proceeds; a failed worker call is contained so one
  bad request never aborts the 20-request run.
- **Validation between stages.** Item names are normalized and validated against
  the catalog before any transaction; the orchestrator branches on each tool's
  `status`.
- **Bounded work.** Worker agents have low `max_steps` and serialized tool
  execution (`max_tool_threads=1`); the orchestrator loop is finite and
  non-recursive (one restock attempt and one quote per line).
- **Customer-safe output.** Replies are assembled deterministically and contain
  only what the customer needs — itemized lines, the discount rationale, the
  promised delivery date, and plain-language declines — never margins, cash
  position, raw errors, or PII.

## 4. Evaluation results & strengths

Running the full sample (`test_results.csv`, 20 requests) clears every rubric
threshold with room to spare:

| Metric | Result | Rubric minimum |
|---|---|---|
| Requests that changed the cash balance | **17** | ≥ 3 |
| Requests fulfilled (≥1 line) | **18** | ≥ 3 |
| Requests with declined lines | **13** | ≥ 1 |

Outcome mix: **7 fully fulfilled, 11 partial, 2 fully declined, 0 parse
failures.** Of 60 parsed line items, 36 were fulfilled and 24 declined — most
because the item isn't carried (A3/A5 paper, balloons, cardboard) or because a
required restock couldn't arrive by the customer's stated deadline.

**Specific strengths observed in the results:**

- **Correctness is provable.** The 36 recorded sales total **$13,088.44**,
  exactly equal to the sum of `total_charged` across all requests — the
  agent/fallback reconciliation produced zero double-counts and zero lost sales.
- **Transparent, attractive pricing.** Bulk discounts scale sensibly (2% at
  100+, 5% at 500+, 8% at 1000+) and every quote line states the discount, e.g.
  Request 8 fulfilled 500/1000/3000-unit lines at 5%/8%/8% and declined the lone
  A5 line with a clear reason — exactly the partial-fulfillment behavior the
  rubric asks for.
- **Robust free-text handling.** The deterministic parser + alias/fuzzy
  normalizer resolves messy phrasings ("A4 glossy paper" → *Glossy paper*, "heavy
  cardstock (white)" → *Cardstock*, "kraft paper envelopes" → *Envelopes*) and
  correctly declines items the company doesn't stock.
- **Financial integrity.** Cash never went negative (minimum $44,290.45); the
  restock-affordability and deadline gates held throughout.

## 5. Limitations & suggested improvements

1. **Add a pricing margin over cost.** The simulation uses the catalog
   `unit_price` as *both* the restock cost and the sale price, so any
   restocked-then-discounted item is sold slightly below cost — which is why
   final cash ($44,746.69) dipped modestly from the $45,059.70 start. A quoting
   improvement would price at `cost × (1 + target_margin)` before applying bulk
   discounts, so discounts erode margin rather than principal, turning the
   operation profitable while staying competitive.
2. **Smarter, demand-aware reordering.** Today restocks cover only the immediate
   shortfall. Using each item's `min_stock_level` and recent sales velocity to
   reorder to a target level (economic order quantity) would cut repeated
   small-batch restocks and their delivery delays, improving fulfillment rates on
   tight deadlines.
3. **LLM-assisted parsing fallback.** The regex/alias parser is fast and
   deterministic but can miss unusual phrasings or ambiguous units (e.g. "50
   packets of envelopes" is treated as 50 units, not 50×100). A bounded LLM
   extraction step, invoked only when the deterministic parser returns nothing or
   flags ambiguity, would raise coverage without sacrificing the reliable default
   path.

## 6. Deliberate design choices & boundaries

A few choices were made consciously — calling them out so they read as decisions
rather than omissions:

- **Full centralization (no agent-to-agent communication).** Every step routes
  through the orchestrator's deterministic `process_request`; workers never talk
  to each other. With `gpt-4o-mini` over a small batch this maximizes reliability
  and traceability — decentralized agent chatter would add failure surface for no
  benefit here. At larger scale, selective direct hand-offs would be the first
  thing to introduce.
- **Static pipeline routing.** Requests always flow inventory → quote → sales →
  advisor, because every input in this dataset is an *order*. Content-based
  routing (a classifier choosing a path) would be over-engineering for a
  homogeneous input domain; it's the natural addition if request types diversified
  (quotes vs. orders vs. complaints).
- **Reseed, don't resume.** `init_database` rebuilds the DB on every run with a
  fixed seed, so evaluations are deterministic and reproducible. A production
  deployment would checkpoint and resume; here repeatability is the more valuable
  property.
- **Configuration split.** Deployment/sensitive settings (model id, endpoint, API
  key) live in `.env`; business rules (discount ladder, restock/parsing policy,
  agent step limits) are centralized in an in-code `CONFIG` dataclass. Keeping the
  submission a single self-contained file is why those rules aren't an external
  config file yet — that's the natural next step at scale.

## 7. How to run

```bash
# Use any Python 3.11+ (developed on 3.13). Create a venv and install deps:
python3 -m venv .venv
.venv/bin/python -m pip install -r requirements.txt smolagents rich

# Provide credentials: copy the template and fill in your key.
cp .env.example .env        # then edit .env (OPENAI_API_KEY, OPENAI_BASE_URL, OPENAI_MODEL)

# Run the full evaluation (must be run from this directory):
.venv/bin/python template.py     # writes test_results.csv
```

Credentials and the model id (`OPENAI_API_KEY`, `OPENAI_BASE_URL`, `OPENAI_MODEL`)
are read from a `.env` file in this directory. `.env` is gitignored; see
`.env.example` for the required variables. Business rules (discount ladder,
restock/parsing policy, agent step limits) live in the `CONFIG` dataclass near the
top of `template.py`.
