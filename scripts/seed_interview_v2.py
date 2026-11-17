"""
seed_interview_v2.py
--------------------
Populates the `interview_questions` table with 45 high-quality, difficulty-aware
questions (3 roles × 3 difficulties × 5 questions each).

Idempotent: skips the entire insert if the table already holds >= 45 rows.
Per-question deduplication: skips any individual question whose text is already present.

Usage (from repo root  22CSP69_1_FINAL-YEAR-PROJECT/):
    python scripts/seed_interview_v2.py
"""

from __future__ import annotations

import sys
from pathlib import Path

# ---------------------------------------------------------------------------
# Bootstrap: add backend/ to sys.path and load .env
# ---------------------------------------------------------------------------
_BACKEND_DIR = Path(__file__).resolve().parent.parent / "backend"
sys.path.insert(0, str(_BACKEND_DIR))

_ENV_FILE = _BACKEND_DIR / "app" / ".env"
if _ENV_FILE.exists():
    try:
        from dotenv import load_dotenv
        load_dotenv(dotenv_path=_ENV_FILE)
    except ImportError:
        pass  # python-dotenv not installed; rely on environment variables

from app.db.database import SessionLocal, engine  # noqa: E402
from app.db import models                          # noqa: E402
from app.db.models import InterviewQuestion        # noqa: E402

# ---------------------------------------------------------------------------
# Question bank  –  3 roles × 3 difficulties × 5 questions = 45 total
# ---------------------------------------------------------------------------
_QUESTIONS: list[dict] = [

    # =========================================================================
    # ROLE: backend developer  |  DIFFICULTY: easy
    # =========================================================================
    {
        "role": "backend developer",
        "question_text": "What is a RESTful API and what are its core constraints?",
        "question_type": "technical",
        "difficulty": "easy",
        "rubric": (
            "Full marks: candidate names the 6 REST constraints (stateless, "
            "client-server, cacheable, uniform interface, layered system, "
            "code-on-demand optional) and maps HTTP verbs to CRUD operations. "
            "Partial: says 'API over HTTP' with correct verb usage but misses "
            "statelessness or uniform interface. "
            "Zero: confuses REST with SOAP, GraphQL, or gives purely marketing language."
        ),
        "tags": ["REST", "API", "HTTP", "architecture"],
    },
    {
        "role": "backend developer",
        "question_text": "Explain the difference between SQL and NoSQL databases. When would you choose one over the other?",
        "question_type": "technical",
        "difficulty": "easy",
        "rubric": (
            "Full marks: contrasts schema-on-write (SQL) vs schema-on-read (NoSQL), "
            "ACID vs eventual consistency, mentions use-case trade-offs (relational "
            "integrity vs horizontal scale/unstructured data). "
            "Partial: lists database names without explaining underlying trade-offs. "
            "Zero: cannot distinguish relational from document stores."
        ),
        "tags": ["SQL", "NoSQL", "databases", "architecture"],
    },
    {
        "role": "backend developer",
        "question_text": "What is an HTTP status code? Give examples of 2xx, 4xx, and 5xx codes.",
        "question_type": "technical",
        "difficulty": "easy",
        "rubric": (
            "Full marks: explains status code categories clearly — 2xx (success: 200 OK, "
            "201 Created), 4xx (client error: 400 Bad Request, 404 Not Found, 401/403), "
            "5xx (server error: 500 Internal Server Error, 503 Unavailable). "
            "Partial: knows at least one correct code from each class. "
            "Zero: cannot categorise or gives wrong class assignments."
        ),
        "tags": ["HTTP", "status codes", "API"],
    },
    {
        "role": "backend developer",
        "question_text": "What is the purpose of environment variables and why should secrets not be hard-coded?",
        "question_type": "technical",
        "difficulty": "easy",
        "rubric": (
            "Full marks: explains configuration portability across environments, "
            "security risk of committing secrets to version control, mentions .env "
            "files, secrets managers, and 12-factor app principles. "
            "Partial: knows secrets shouldn't be in code but cannot explain how to "
            "handle them properly. "
            "Zero: unaware of the concept or suggests comments as a solution."
        ),
        "tags": ["security", "environment", "best practices", "12-factor"],
    },
    {
        "role": "backend developer",
        "question_text": "Describe what happens when a client sends a GET request to a web server and receives a JSON response.",
        "question_type": "technical",
        "difficulty": "easy",
        "rubric": (
            "Full marks: covers DNS resolution, TCP/TLS handshake, HTTP request "
            "headers (Accept: application/json), server routing, database query or "
            "cache hit, response serialisation, HTTP 200 with Content-Type header. "
            "Partial: covers request → server → response without network or "
            "serialisation details. "
            "Zero: cannot trace the basic request-response cycle."
        ),
        "tags": ["HTTP", "networking", "web fundamentals"],
    },

    # =========================================================================
    # ROLE: backend developer  |  DIFFICULTY: medium
    # =========================================================================
    {
        "role": "backend developer",
        "question_text": "How does database indexing work and what are the trade-offs of adding too many indexes?",
        "question_type": "technical",
        "difficulty": "medium",
        "rubric": (
            "Full marks: explains B-tree/hash index structures, index scan vs full "
            "table scan, covering indexes; trade-offs: write amplification (every "
            "INSERT/UPDATE must update the index), storage overhead, index bloat. "
            "Partial: says 'indexes speed up reads' but cannot explain write overhead "
            "or index structure. "
            "Zero: confused with database constraints or cannot explain the concept."
        ),
        "tags": ["databases", "indexing", "performance", "SQL"],
    },
    {
        "role": "backend developer",
        "question_text": "Explain the N+1 query problem. How do you detect and fix it in an ORM-based application?",
        "question_type": "technical",
        "difficulty": "medium",
        "rubric": (
            "Full marks: demonstrates N+1 with a concrete code example (e.g., looping "
            "over posts and fetching author per post), explains eager loading / JOIN "
            "strategies, ORM-specific solutions (SQLAlchemy joinedload/selectinload, "
            "Django select_related/prefetch_related), mentions query logging to detect. "
            "Partial: identifies the problem conceptually but cannot explain the ORM fix. "
            "Zero: unaware of the problem or conflates it with connection pooling."
        ),
        "tags": ["ORM", "SQL", "performance", "SQLAlchemy"],
    },
    {
        "role": "backend developer",
        "question_text": "What is JWT (JSON Web Token) authentication? Walk through how a user logs in and subsequent requests are authorised.",
        "question_type": "technical",
        "difficulty": "medium",
        "rubric": (
            "Full marks: covers header.payload.signature structure, HS256/RS256 signing, "
            "stateless validation on each request, refresh token rotation, expiry, "
            "storage best practices (httpOnly cookie vs localStorage XSS risk). "
            "Partial: understands login → token → bearer header flow but misses "
            "signature verification or refresh strategy. "
            "Zero: confuses JWT with session cookies or OAuth scopes."
        ),
        "tags": ["JWT", "authentication", "security", "API"],
    },
    {
        "role": "backend developer",
        "question_text": "How would you design a simple rate-limiting system for a public API endpoint?",
        "question_type": "technical",
        "difficulty": "medium",
        "rubric": (
            "Full marks: discusses fixed-window vs sliding-window vs token-bucket "
            "algorithms, using Redis to store counters per IP/user/API key, returning "
            "429 Too Many Requests with Retry-After header, distributed deployment "
            "concerns (single Redis vs cluster). "
            "Partial: knows 'count requests per second' but cannot implement it in "
            "a distributed environment. "
            "Zero: suggests blocking at the database layer or cannot explain the concept."
        ),
        "tags": ["rate limiting", "Redis", "API", "scalability"],
    },
    {
        "role": "backend developer",
        "question_text": "What is database connection pooling and why is it critical for backend performance?",
        "question_type": "technical",
        "difficulty": "medium",
        "rubric": (
            "Full marks: explains TCP connection setup cost, pooling pre-warming "
            "connections, pool_size vs max_overflow (SQLAlchemy), thread-safety, "
            "pool exhaustion symptoms and how to tune. "
            "Partial: knows pooling reuses connections but cannot explain pool sizing "
            "or exhaustion. "
            "Zero: thinks each request opens a new DB connection is acceptable at scale."
        ),
        "tags": ["databases", "performance", "connection pooling", "SQLAlchemy"],
    },

    # =========================================================================
    # ROLE: backend developer  |  DIFFICULTY: hard
    # =========================================================================
    {
        "role": "backend developer",
        "question_text": "Design a URL-shortening service (like bit.ly). Discuss database schema, hashing strategy, redirect performance, and how you'd handle 100 million URLs.",
        "question_type": "technical",
        "difficulty": "hard",
        "rubric": (
            "Full marks: proposes base62 encoding of an auto-increment ID or "
            "MD5/SHA256 truncation with collision handling, schema (id, long_url, "
            "short_code, created_at, clicks), HTTP 301 vs 302 redirect trade-offs, "
            "Redis caching hot URLs, DB sharding or consistent hashing for 100M+ scale, "
            "analytics pipeline. "
            "Partial: proposes a working schema and hashing but misses caching or "
            "scaling strategy. "
            "Zero: cannot propose a schema or explains redirects incorrectly."
        ),
        "tags": ["system design", "hashing", "caching", "scalability"],
    },
    {
        "role": "backend developer",
        "question_text": "Explain database transactions and ACID properties. How do isolation levels (READ COMMITTED, REPEATABLE READ, SERIALIZABLE) affect concurrency and performance?",
        "question_type": "technical",
        "difficulty": "hard",
        "rubric": (
            "Full marks: defines Atomicity, Consistency, Isolation, Durability with "
            "concrete examples; explains dirty read, non-repeatable read, phantom read "
            "anomalies; maps each isolation level to what anomalies it prevents; "
            "discusses locking overhead vs concurrency throughput trade-offs. "
            "Partial: defines ACID but cannot explain isolation levels or anomalies. "
            "Zero: cannot define at least 3 of the 4 ACID properties correctly."
        ),
        "tags": ["databases", "ACID", "transactions", "concurrency"],
    },
    {
        "role": "backend developer",
        "question_text": "How would you implement an event-driven architecture using a message queue (e.g., RabbitMQ or Kafka) for an order-processing system?",
        "question_type": "technical",
        "difficulty": "hard",
        "rubric": (
            "Full marks: contrasts Kafka (log-based, replay, consumer groups, "
            "partitions) vs RabbitMQ (message broker, queue per consumer, ACK/NACK), "
            "draws event flow (OrderPlaced → inventory, payment, notification services), "
            "discusses idempotency, dead-letter queues, at-least-once vs exactly-once "
            "delivery, schema registry, backpressure. "
            "Partial: knows events are published to a queue and consumed but misses "
            "idempotency and failure handling. "
            "Zero: cannot distinguish synchronous RPC from async messaging."
        ),
        "tags": ["message queue", "event-driven", "Kafka", "RabbitMQ", "architecture"],
    },
    {
        "role": "backend developer",
        "question_text": "Explain how you would debug and resolve a memory leak in a long-running Python FastAPI service.",
        "question_type": "technical",
        "difficulty": "hard",
        "rubric": (
            "Full marks: mentions profiling with tracemalloc or memory_profiler, "
            "objgraph for reference cycle detection, checking for global caches growing "
            "unbounded, background tasks holding references, SQLAlchemy session scope "
            "leaks, asyncio task leaks, Prometheus memory metrics for trend detection. "
            "Partial: identifies profiling tools but cannot trace the root cause in an "
            "async context. "
            "Zero: suggests restarting the server as the only solution."
        ),
        "tags": ["debugging", "memory", "Python", "FastAPI", "profiling"],
    },
    {
        "role": "backend developer",
        "question_text": "Walk me through designing a distributed cache invalidation strategy for a microservices system where multiple services share a Redis cache.",
        "question_type": "technical",
        "difficulty": "hard",
        "rubric": (
            "Full marks: explains write-through vs write-behind vs cache-aside patterns, "
            "TTL-based expiry, event-driven invalidation via pub/sub, cache stampede "
            "prevention (mutex lock, probabilistic early expiry), versioned cache keys, "
            "consistency guarantees across service boundaries. "
            "Partial: knows TTL and cache-aside but cannot handle cache stampede or "
            "cross-service invalidation. "
            "Zero: proposes clearing the entire cache on any update."
        ),
        "tags": ["caching", "Redis", "microservices", "distributed systems"],
    },

    # =========================================================================
    # ROLE: frontend developer  |  DIFFICULTY: easy
    # =========================================================================
    {
        "role": "frontend developer",
        "question_text": "What is the difference between `let`, `const`, and `var` in JavaScript?",
        "question_type": "technical",
        "difficulty": "easy",
        "rubric": (
            "Full marks: explains block scope (let/const) vs function scope (var), "
            "var hoisting with undefined initialisation, const preventing reassignment "
            "but not mutation of objects/arrays, temporal dead zone for let/const. "
            "Partial: knows const can't be reassigned and var is older but misses "
            "hoisting or block scope details. "
            "Zero: cannot distinguish at least two of the three."
        ),
        "tags": ["JavaScript", "ES6", "scope", "variables"],
    },
    {
        "role": "frontend developer",
        "question_text": "Explain the CSS box model. How does `box-sizing: border-box` change element sizing?",
        "question_type": "technical",
        "difficulty": "easy",
        "rubric": (
            "Full marks: describes content → padding → border → margin layers, "
            "default content-box (width = content only), border-box includes padding "
            "and border in the declared width making layout calculations predictable. "
            "Partial: knows padding and margin exist but cannot explain how border-box "
            "affects computed width. "
            "Zero: cannot describe what the box model is."
        ),
        "tags": ["CSS", "box model", "layout"],
    },
    {
        "role": "frontend developer",
        "question_text": "What is the virtual DOM in React and why was it introduced?",
        "question_type": "technical",
        "difficulty": "easy",
        "rubric": (
            "Full marks: explains real DOM manipulation is slow (layout/paint "
            "reflows), React keeps an in-memory VDOM, diffs the previous and next "
            "trees (reconciliation), and batches the minimal set of real DOM updates "
            "(commit phase). "
            "Partial: says 'React uses a copy of the DOM for performance' without "
            "explaining reconciliation or batching. "
            "Zero: confuses VDOM with Shadow DOM or server-side rendering."
        ),
        "tags": ["React", "virtual DOM", "performance", "reconciliation"],
    },
    {
        "role": "frontend developer",
        "question_text": "What is event delegation in JavaScript and why is it useful?",
        "question_type": "technical",
        "difficulty": "easy",
        "rubric": (
            "Full marks: explains event bubbling — events propagate from target up "
            "the DOM tree, so attaching one listener on a parent handles events from "
            "many children; avoids attaching/removing listeners for dynamically added "
            "elements, reduces memory usage. "
            "Partial: knows a listener on the parent fires for child clicks but cannot "
            "explain bubbling or dynamic element benefits. "
            "Zero: cannot explain what event delegation is."
        ),
        "tags": ["JavaScript", "DOM", "events", "performance"],
    },
    {
        "role": "frontend developer",
        "question_text": "What is the difference between `==` and `===` in JavaScript?",
        "question_type": "technical",
        "difficulty": "easy",
        "rubric": (
            "Full marks: == performs type coercion before comparison (e.g. '5' == 5 "
            "is true), === checks both value and type with no coercion, provides "
            "examples of surprising == results (null == undefined, 0 == false). "
            "Partial: knows === is stricter but cannot give a coercion example. "
            "Zero: says they are identical or reverses which is strict."
        ),
        "tags": ["JavaScript", "type coercion", "operators"],
    },

    # =========================================================================
    # ROLE: frontend developer  |  DIFFICULTY: medium
    # =========================================================================
    {
        "role": "frontend developer",
        "question_text": "Explain React's `useEffect` hook. How do the dependency array variations (empty, absent, with values) change its behaviour?",
        "question_type": "technical",
        "difficulty": "medium",
        "rubric": (
            "Full marks: no array → runs after every render; empty [] → runs once "
            "after mount (like componentDidMount); [dep1, dep2] → runs when those "
            "values change; cleanup return function (like componentWillUnmount for "
            "subscriptions/timers); mentions stale closure pitfalls. "
            "Partial: knows the three array forms but cannot explain stale closures "
            "or cleanup. "
            "Zero: cannot distinguish the dependency array variations."
        ),
        "tags": ["React", "hooks", "useEffect", "lifecycle"],
    },
    {
        "role": "frontend developer",
        "question_text": "What is CSS specificity and how is it calculated? How do you resolve specificity conflicts without using `!important`?",
        "question_type": "technical",
        "difficulty": "medium",
        "rubric": (
            "Full marks: specificity = (inline, ID, class/attr/pseudo-class, element) "
            "as 4-tuple; calculates examples; resolves conflicts via BEM, CSS Modules, "
            "or component-scoped styles instead of !important; mentions cascade order "
            "as last resort. "
            "Partial: knows IDs beat classes but cannot calculate specificity or "
            "resolve conflicts systematically. "
            "Zero: says whichever comes last always wins."
        ),
        "tags": ["CSS", "specificity", "cascade", "BEM"],
    },
    {
        "role": "frontend developer",
        "question_text": "What are React keys and why must they be unique and stable? What goes wrong when you use array indexes as keys?",
        "question_type": "technical",
        "difficulty": "medium",
        "rubric": (
            "Full marks: keys help React's reconciler identify which list items "
            "changed/reordered/removed without re-creating the DOM; using array index "
            "causes bugs when list items are reordered or deleted (wrong component "
            "state maps to wrong item), leads to animation glitches and input focus "
            "issues. "
            "Partial: knows keys are required for lists but cannot explain the "
            "index-key bug. "
            "Zero: cannot explain what keys do in React."
        ),
        "tags": ["React", "reconciliation", "lists", "keys"],
    },
    {
        "role": "frontend developer",
        "question_text": "How does the browser's Critical Rendering Path work? What techniques improve First Contentful Paint?",
        "question_type": "technical",
        "difficulty": "medium",
        "rubric": (
            "Full marks: HTML → DOM, CSS → CSSOM, combined → Render Tree → Layout → "
            "Paint → Composite; render-blocking resources (synchronous <script>, "
            "unloaded CSS); techniques: defer/async scripts, preload critical CSS, "
            "inline above-the-fold CSS, reduce unused JS (tree shaking), server-side "
            "rendering. "
            "Partial: mentions defer/async or inline CSS but cannot trace the full "
            "rendering path. "
            "Zero: cannot describe what happens between HTML and pixels."
        ),
        "tags": ["performance", "CRP", "FCP", "browser", "rendering"],
    },
    {
        "role": "frontend developer",
        "question_text": "Explain how Promises and async/await work in JavaScript. What is the difference between `Promise.all` and `Promise.allSettled`?",
        "question_type": "technical",
        "difficulty": "medium",
        "rubric": (
            "Full marks: Promise states (pending/fulfilled/rejected), async/await "
            "as syntactic sugar over Promises, microtask queue execution order; "
            "Promise.all rejects fast on first rejection (fail-fast), "
            "Promise.allSettled always waits for all and returns {status, value/reason} "
            "array — use case for independent parallel fetches where partial results "
            "are acceptable. "
            "Partial: knows async/await syntax but cannot explain Promise.all vs "
            "allSettled failure modes. "
            "Zero: cannot write a correct async/await function."
        ),
        "tags": ["JavaScript", "async", "promises", "concurrency"],
    },

    # =========================================================================
    # ROLE: frontend developer  |  DIFFICULTY: hard
    # =========================================================================
    {
        "role": "frontend developer",
        "question_text": "How would you architect a large-scale React application to minimise unnecessary re-renders? Discuss `useMemo`, `useCallback`, `React.memo`, and when NOT to use them.",
        "question_type": "technical",
        "difficulty": "hard",
        "rubric": (
            "Full marks: explains referential equality problem triggering child "
            "re-renders, React.memo for component memoisation, useMemo for expensive "
            "derived values, useCallback for stable function references; critically "
            "notes premature optimisation cost (closure allocation, comparison overhead) "
            "and when profiling should drive decisions (React DevTools Profiler). "
            "Partial: knows the three APIs exist but cannot explain when they help vs "
            "hurt performance. "
            "Zero: says 'just use useMemo everywhere'."
        ),
        "tags": ["React", "performance", "memoisation", "optimisation"],
    },
    {
        "role": "frontend developer",
        "question_text": "Explain the Webpack/Vite module bundler pipeline. How does code splitting and lazy loading with `React.lazy` improve load performance?",
        "question_type": "technical",
        "difficulty": "hard",
        "rubric": (
            "Full marks: explains entry point → dependency graph → chunk generation → "
            "tree shaking dead code elimination; dynamic import() creates async "
            "chunk boundaries; React.lazy + Suspense defers loading until component "
            "is needed; discusses route-level vs component-level splitting; preloading "
            "hints (rel=prefetch/preload). "
            "Partial: knows lazy loading splits code but cannot explain the bundler "
            "graph or tree shaking. "
            "Zero: cannot explain what a module bundler does."
        ),
        "tags": ["Webpack", "Vite", "code splitting", "performance", "React"],
    },
    {
        "role": "frontend developer",
        "question_text": "How does the browser security model (Same-Origin Policy, CORS, CSP) work? How would you configure CORS on a FastAPI backend for a React frontend?",
        "question_type": "technical",
        "difficulty": "hard",
        "rubric": (
            "Full marks: SOP blocks cross-origin reads; CORS is a server-side "
            "opt-in (Access-Control-Allow-Origin header), simple vs preflight (OPTIONS) "
            "requests, credentials mode; CSP restricts script/resource sources to "
            "prevent XSS; FastAPI config: CORSMiddleware with explicit origins list, "
            "allow_credentials=True, allow_methods, allow_headers. "
            "Partial: knows CORS exists and needs server config but misses preflight "
            "or CSP. "
            "Zero: says CORS is a frontend problem or cannot explain SOP."
        ),
        "tags": ["security", "CORS", "CSP", "SOP", "FastAPI"],
    },
    {
        "role": "frontend developer",
        "question_text": "Design an accessible, keyboard-navigable modal component from scratch in React without a UI library. What ARIA attributes are required?",
        "question_type": "technical",
        "difficulty": "hard",
        "rubric": (
            "Full marks: role='dialog' + aria-modal='true' + aria-labelledby pointing "
            "to heading; focus trap (Tab cycles within modal, Shift+Tab reverse); "
            "Escape closes modal; return focus to trigger element on close; "
            "prevent scroll of background content; portal rendering via createPortal "
            "to avoid z-index issues. "
            "Partial: implements a working modal but misses focus trap or ARIA roles. "
            "Zero: creates a styled div with no accessibility considerations."
        ),
        "tags": ["accessibility", "ARIA", "React", "keyboard", "modal"],
    },
    {
        "role": "frontend developer",
        "question_text": "Explain micro-frontend architecture. When is it appropriate and what are the integration challenges (routing, shared state, CSS isolation)?",
        "question_type": "technical",
        "difficulty": "hard",
        "rubric": (
            "Full marks: motivations (team autonomy, independent deployability, "
            "polyglot tech); implementation approaches (Module Federation, iframes, "
            "web components, single-spa); challenges: shared dependencies version "
            "conflicts, cross-MFE routing, global state sharing (custom events vs "
            "shared store), CSS bleeding (Shadow DOM, CSS Modules, BEM namespacing), "
            "performance impact of duplicate React instances. "
            "Partial: knows micro-frontends split the UI into teams but cannot explain "
            "integration challenges. "
            "Zero: confuses micro-frontends with microservices or component libraries."
        ),
        "tags": ["micro-frontends", "architecture", "Module Federation", "scalability"],
    },

    # =========================================================================
    # ROLE: software engineer  |  DIFFICULTY: easy
    # =========================================================================
    {
        "role": "software engineer",
        "question_text": "What is the difference between a process and a thread?",
        "question_type": "technical",
        "difficulty": "easy",
        "rubric": (
            "Full marks: process has its own memory space, file handles, and OS "
            "resources; threads share memory within a process; context-switch cost "
            "comparison; mentions GIL in CPython limiting true thread parallelism for "
            "CPU-bound work. "
            "Partial: says threads are lighter but cannot explain shared memory or GIL. "
            "Zero: says they are the same thing."
        ),
        "tags": ["OS", "concurrency", "processes", "threads"],
    },
    {
        "role": "software engineer",
        "question_text": "Explain the SOLID principles of object-oriented design. Give a one-line example for each.",
        "question_type": "technical",
        "difficulty": "easy",
        "rubric": (
            "Full marks: S - Single Responsibility (one class, one reason to change), "
            "O - Open/Closed (open for extension, closed for modification via "
            "inheritance/composition), L - Liskov Substitution (subclasses replaceable "
            "for base), I - Interface Segregation (many specific interfaces over one "
            "general), D - Dependency Inversion (depend on abstractions). "
            "Partial: correctly explains 3 of 5. "
            "Zero: cannot name or explain more than one."
        ),
        "tags": ["OOP", "SOLID", "design principles", "architecture"],
    },
    {
        "role": "software engineer",
        "question_text": "What is Git and explain the difference between `git merge` and `git rebase`.",
        "question_type": "technical",
        "difficulty": "easy",
        "rubric": (
            "Full marks: Git is a distributed VCS; merge creates a merge commit "
            "preserving branch history (non-destructive); rebase replays commits on "
            "top of target branch creating linear history (rewrites SHAs — never "
            "rebase shared/public branches). "
            "Partial: knows both integrate branches but cannot explain history "
            "implications or SHA rewriting. "
            "Zero: cannot explain the difference."
        ),
        "tags": ["Git", "version control", "merge", "rebase"],
    },
    {
        "role": "software engineer",
        "question_text": "What is Big O notation? What is the time complexity of searching in a sorted array vs a hash map?",
        "question_type": "technical",
        "difficulty": "easy",
        "rubric": (
            "Full marks: Big O describes worst-case asymptotic growth; binary search "
            "on sorted array is O(log n); hash map lookup is O(1) average (O(n) worst "
            "with collisions); explains why hash is faster for exact lookups but "
            "doesn't support range queries. "
            "Partial: knows O(1) vs O(log n) but cannot explain hash collision or "
            "range query limitation. "
            "Zero: cannot state the time complexity for either structure."
        ),
        "tags": ["algorithms", "Big O", "data structures", "complexity"],
    },
    {
        "role": "software engineer",
        "question_text": "Tell me about a time you found and fixed a difficult bug. What was your debugging process?",
        "question_type": "behavioral",
        "difficulty": "easy",
        "rubric": (
            "Full marks: uses STAR format (Situation, Task, Action, Result); "
            "describes systematic reproduction steps, hypothesis generation, "
            "tool use (logs, debugger, profiler), root cause identification, "
            "and verifying the fix didn't break related functionality. "
            "Partial: describes a real bug fix but process was ad-hoc without "
            "systematic steps or tooling. "
            "Zero: cannot recall a specific example or describes a trivial "
            "typo fix without process detail."
        ),
        "tags": ["behavioral", "debugging", "STAR", "problem solving"],
    },

    # =========================================================================
    # ROLE: software engineer  |  DIFFICULTY: medium
    # =========================================================================
    {
        "role": "software engineer",
        "question_text": "Compare stack and heap memory allocation. How does garbage collection work in Python?",
        "question_type": "technical",
        "difficulty": "medium",
        "rubric": (
            "Full marks: stack = LIFO, fixed-size frames for local variables/function "
            "calls, automatic deallocation; heap = dynamic allocation for objects, "
            "managed by GC; Python uses reference counting as primary GC + cyclic GC "
            "for reference cycles (gc module), generational collection (gen 0/1/2). "
            "Partial: knows stack vs heap broadly but cannot explain Python's reference "
            "counting or cycle collector. "
            "Zero: cannot distinguish stack from heap."
        ),
        "tags": ["memory", "Python", "garbage collection", "OS"],
    },
    {
        "role": "software engineer",
        "question_text": "What is the difference between unit testing, integration testing, and end-to-end testing? Describe the testing pyramid.",
        "question_type": "technical",
        "difficulty": "medium",
        "rubric": (
            "Full marks: testing pyramid — wide base of fast/isolated unit tests, "
            "middle layer of integration tests (real DB/services, slower), narrow tip "
            "of E2E tests (full user flows, slowest/most brittle); explains trade-offs "
            "in cost, speed, and isolation; mentions mocking/stubbing for unit tests. "
            "Partial: distinguishes the three types but cannot explain pyramid "
            "trade-offs or mock strategy. "
            "Zero: cannot distinguish unit from integration tests."
        ),
        "tags": ["testing", "TDD", "testing pyramid", "quality assurance"],
    },
    {
        "role": "software engineer",
        "question_text": "Explain the CAP theorem. How does it apply to choosing between PostgreSQL and Cassandra?",
        "question_type": "technical",
        "difficulty": "medium",
        "rubric": (
            "Full marks: CAP — Consistency (every read gets latest write or error), "
            "Availability (every request gets a response), Partition Tolerance (system "
            "works despite network partition) — can only guarantee 2 of 3; PostgreSQL "
            "is CP (sacrifices availability in partition for consistency); Cassandra "
            "is AP (sacrifices consistency for availability via eventual consistency "
            "with tunable quorum). "
            "Partial: states the theorem but cannot classify the two databases. "
            "Zero: cannot state what the C, A, and P stand for."
        ),
        "tags": ["distributed systems", "CAP theorem", "databases", "architecture"],
    },
    {
        "role": "software engineer",
        "question_text": "Describe the design patterns you use most often. Explain the Observer and Strategy patterns with concrete examples.",
        "question_type": "technical",
        "difficulty": "medium",
        "rubric": (
            "Full marks: Observer — subject maintains subscriber list, notifies on "
            "state change (event emitters, React state, pub/sub); Strategy — "
            "encapsulates interchangeable algorithms behind a common interface "
            "(sorting strategies, payment methods); correctly identifies both as "
            "GoF patterns, explains Open/Closed principle connection. "
            "Partial: describes one pattern correctly with an example but conflates "
            "the other with a different pattern. "
            "Zero: cannot explain either pattern correctly."
        ),
        "tags": ["design patterns", "OOP", "Observer", "Strategy"],
    },
    {
        "role": "software engineer",
        "question_text": "Tell me about a time you had to work in a team with conflicting technical opinions. How did you reach a decision?",
        "question_type": "behavioral",
        "difficulty": "medium",
        "rubric": (
            "Full marks: uses STAR; demonstrates active listening and empathy, "
            "data/prototype-driven decision making (e.g. benchmark or spike), "
            "escalation strategy if consensus fails, outcome and what was learned. "
            "Partial: describes the conflict but resolution was based on authority "
            "or avoidance rather than evidence-based discussion. "
            "Zero: cannot recall a specific conflict or describes it as 'everyone agreed'."
        ),
        "tags": ["behavioral", "teamwork", "conflict resolution", "STAR"],
    },

    # =========================================================================
    # ROLE: software engineer  |  DIFFICULTY: hard
    # =========================================================================
    {
        "role": "software engineer",
        "question_text": "Design a highly available notification service (email, SMS, push) that must deliver at least once to 10 million users within 5 minutes of an event.",
        "question_type": "technical",
        "difficulty": "hard",
        "rubric": (
            "Full marks: event ingestion via Kafka (partitioned by user segment), "
            "fan-out workers per channel (email via SES, SMS via Twilio, push via "
            "FCM/APNs), idempotent delivery with deduplication ID, retry with "
            "exponential backoff to dead-letter queue, rate limiting per third-party "
            "API quota, monitoring (delivery rate, latency P99), horizontal "
            "auto-scaling. "
            "Partial: proposes a queue-based approach but misses idempotency, "
            "rate limiting, or monitoring. "
            "Zero: proposes a synchronous for-loop over all users."
        ),
        "tags": ["system design", "scalability", "notifications", "Kafka"],
    },
    {
        "role": "software engineer",
        "question_text": "Explain lock-free data structures and compare them with mutex-based synchronisation. When would you use `threading.Lock` vs `asyncio.Lock` in Python?",
        "question_type": "technical",
        "difficulty": "hard",
        "rubric": (
            "Full marks: lock-free uses atomic CPU operations (CAS - compare-and-swap) "
            "to avoid OS scheduler involvement, eliminates deadlock and priority "
            "inversion at cost of ABA problem complexity; threading.Lock blocks OS "
            "thread (suitable for CPU-bound workers/multiprocessing), asyncio.Lock "
            "suspends coroutine without blocking event loop thread (I/O-bound async "
            "code); mixing them causes deadlocks. "
            "Partial: distinguishes threading.Lock vs asyncio.Lock but cannot explain "
            "lock-free or CAS. "
            "Zero: cannot explain why asyncio.Lock doesn't block the event loop."
        ),
        "tags": ["concurrency", "lock-free", "Python", "asyncio", "threading"],
    },
    {
        "role": "software engineer",
        "question_text": "Walk me through implementing a distributed rate limiter using Redis that works correctly across 50 horizontally-scaled API server instances.",
        "question_type": "technical",
        "difficulty": "hard",
        "rubric": (
            "Full marks: Redis INCR + EXPIRE for fixed window (race condition "
            "analysis), Lua scripting for atomic sliding-window (ZADD + ZRANGE + "
            "atomic eval), token bucket using Redis HASH + Lua, Redis Cluster "
            "with consistent hashing for HA, handling Redis failover (graceful "
            "degradation), clock skew across instances. "
            "Partial: proposes Redis INCR approach but misses atomicity race "
            "condition or HA failover. "
            "Zero: proposes in-process counters (won't work across instances)."
        ),
        "tags": ["distributed systems", "Redis", "rate limiting", "scalability"],
    },
    {
        "role": "software engineer",
        "question_text": "Describe a production incident you led the resolution of. Walk through how you identified the root cause, communicated with stakeholders, and implemented the fix.",
        "question_type": "behavioral",
        "difficulty": "hard",
        "rubric": (
            "Full marks: demonstrates clear incident command mindset (declare "
            "severity, establish communication channel), structured debugging "
            "(metrics → logs → traces), clear non-technical stakeholder "
            "communication cadence, blameless post-mortem with 5 Whys root cause, "
            "follow-up action items to prevent recurrence. "
            "Partial: describes the fix correctly but lacked systematic communication "
            "or post-mortem process. "
            "Zero: cannot recall a production incident or describes it without "
            "structured debugging or stakeholder communication."
        ),
        "tags": ["behavioral", "incident management", "STAR", "leadership"],
    },
    {
        "role": "software engineer",
        "question_text": "How does a CPU cache (L1/L2/L3) affect algorithm performance? Explain cache-friendly vs cache-unfriendly data access patterns with examples.",
        "question_type": "technical",
        "difficulty": "hard",
        "rubric": (
            "Full marks: explains cache line (64 bytes), spatial locality (sequential "
            "array access is cache-friendly, cache pre-fetcher works), temporal "
            "locality; cache-unfriendly: linked list traversal (pointer chasing, "
            "random memory access), column-major access on row-major matrix; "
            "demonstrates with matrix multiply row vs column traversal benchmark; "
            "mentions struct-of-arrays vs array-of-structs layout for SIMD. "
            "Partial: knows caches exist and sequential is faster but cannot explain "
            "cache lines or give a concrete unfriendly access pattern. "
            "Zero: unaware that memory access patterns affect performance."
        ),
        "tags": ["computer architecture", "cache", "performance", "algorithms"],
    },
]

# ---------------------------------------------------------------------------
# Main seeder
# ---------------------------------------------------------------------------

def main() -> None:
    print("=" * 60)
    print("PlaceBuddy — Interview Question Seeder v2")
    print("=" * 60)

    # Ensure tables exist (safe on existing schema)
    models.Base.metadata.create_all(bind=engine)

    db = SessionLocal()
    try:
        total_expected = 45
        current_count = db.query(InterviewQuestion).count()

        if current_count >= total_expected:
            print(
                f"[SKIP] Table already contains {current_count} questions "
                f"(threshold={total_expected}). Nothing to insert."
            )
            return

        # Fetch existing question texts for per-row deduplication
        existing_texts: set[str] = {
            row.question_text
            for row in db.query(InterviewQuestion.question_text).all()
        }

        inserted = 0
        skipped = 0

        for q in _QUESTIONS:
            if q["question_text"] in existing_texts:
                skipped += 1
                continue

            record = InterviewQuestion(
                role=q["role"],
                question_text=q["question_text"],
                question_type=q["question_type"],
                difficulty=q["difficulty"],
                rubric=q["rubric"],
                tags=q["tags"],
            )
            db.add(record)
            existing_texts.add(q["question_text"])
            inserted += 1

        db.commit()

        final_count = db.query(InterviewQuestion).count()
        print(f"[DONE] Inserted {inserted} new questions, skipped {skipped} duplicates.")
        print(f"[INFO] Total questions in table: {final_count}")

        # Distribution summary
        print("\nDistribution summary:")
        for role in ("backend developer", "frontend developer", "software engineer"):
            for diff in ("easy", "medium", "hard"):
                count = (
                    db.query(InterviewQuestion)
                    .filter(
                        InterviewQuestion.role == role,
                        InterviewQuestion.difficulty == diff,
                    )
                    .count()
                )
                print(f"  {role:<22} | {diff:<8} → {count} questions")

    except Exception as exc:
        db.rollback()
        print(f"[ERROR] Seeding failed: {exc}")
        raise
    finally:
        db.close()

    print("=" * 60)
    print("Seeding complete.")


if __name__ == "__main__":
    main()
