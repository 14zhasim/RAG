# Module 7: LLMs

## Lessons Learned

- **LLM query enhancement:** improves the query before retrieval.
  - Users type vague, misspelled, informal, or incomplete searches.
  - The enhanced query is still passed into the existing search system; the LLM does not replace retrieval.
- **Enhancement methods:** solve different query problems.
  - **Spell correction** fixes clear typos conservatively.
  - **Rewriting** turns vague intent into concise searchable terms.
  - **Expansion** adds related terms to increase recall, with some risk of query drift.
    - Manual (programmatic) expansion is also option for expansion!
    - E.g. filter search for list of film genres, like comedy:
      EXPANSIONS = {"comedy": ["funny", "humorous", "lighthearted"]}
- **Best LLM prompting methods**: 
  - https://platform.claude.com/docs/en/build-with-claude/prompt-engineering/overview
  - Example queries for each enhancement method given below

---

## Setup and Context

The search system is now strong enough to retrieve movies, but the user can still send a weak query:

```text
"Padington"
"that bear movie where leo gets attacked"
"scary movie with bear from few years ago"
```

The retrieval engine cannot always infer what the user meant if the query itself is broken or underspecified. Module 7 adds an optional LLM preprocessing stage:

```mermaid
flowchart LR
    UserQuery[Original query] --> Enhance{Enhance?}
    Enhance -->|none| RRF[RRF search]
    Enhance -->|spell/rewrite/expand| LLM[LLM query enhancement]
    LLM --> EnhancedQuery[Enhanced query]
    EnhancedQuery --> RRF
    RRF --> Results[Search results]
```

The LLM does not replace BM25, semantic search, or RRF. It only changes the query those systems receive.

---

## Core LLM Enhancement Pipeline

### 1. Install and configure the API client

The module adds two dependencies:

```bash
uv add python-dotenv==1.1.0
uv add openai==2.44.0
```

The API key goes in `.env`:

```text
OPENROUTER_API_KEY="your_api_key_here"
```

The `.env` file must be ignored by Git:

```text
.env
```

The code loads the key at runtime:

```python
from dotenv import load_dotenv

load_dotenv()
api_key = os.environ.get("OPENROUTER_API_KEY")
if not api_key:
    raise RuntimeError("OPENROUTER_API_KEY environment variable not set")
```

That keeps secrets out of commits while still making local development convenient.

### 2. Point the OpenAI client at OpenRouter

OpenRouter uses an OpenAI-compatible API, so the project can use the OpenAI SDK with a custom `base_url`.

```python
from openai import OpenAI

return OpenAI(
    base_url="https://openrouter.ai/api/v1",
    api_key=api_key,
)
```

The model ID used in this repo is:

```python
model = "openrouter/free"
```

That routes to an available free model, which is useful for learning but can hit provider limits.

The code creates the API client inside `get_client()`:

```python
def get_client():
    api_key = os.environ.get("OPENROUTER_API_KEY")
    if not api_key:
        raise RuntimeError("OPENROUTER_API_KEY environment variable not set")

    return OpenAI(
        base_url="https://openrouter.ai/api/v1",
        api_key=api_key,
    )
```

Only commands that actually use `--enhance` need the key.

### 3. Use one shared helper for LLM calls

Spell correction, rewriting, and expansion all need the same API boilerplate. The only major difference is the prompt.

```python
def run_query_enhancement(query: str, prompt: str) -> str:
    message = [
        {
            "role": "user",
            "content": prompt,
        }
    ]

    client = get_client()
    response = client.chat.completions.create(
        model=model,
        messages=message,
    )
```

The helper also reads and cleans the response:

```python
enhanced_query = (response.choices[0].message.content or "").strip().strip('"')
return enhanced_query if enhanced_query else query
```

That fallback handles empty or unexpected LLM output.

### 4. Spell correction fixes clear typos

Spell correction is the most conservative enhancement. It should fix obvious mistakes without changing intent.

```python
def spell_correct(query: str) -> str:
    prompt = f"""Fix any spelling errors in the user-provided movie search query below.
    Correct only clear, high-confidence typos. Do not rewrite, add, remove, or reorder words.
    Preserve punctuation and capitalization unless a change is required for a typo fix.
    If there are no spelling errors, or if you're unsure, output the original query unchanged.
    Output only the final query text, nothing else.
    User query: "{query}"
    """
    return run_query_enhancement(query, prompt)
```

Example:

```text
Original: "Padington"
Enhanced: "Paddington"
```

This helps keyword search especially because BM25 cannot match a misspelled token unless that misspelling appears in the index.

### 5. Query rewriting turns messy intent into searchable terms

Rewriting is more aggressive. It transforms informal language into concise search terms.

```python
def rewrite_query(query: str) -> str:
    prompt = f"""Rewrite the user-provided movie search query below to be more specific and searchable.
    ...
    User query: "{query}"
    """
    return run_query_enhancement(query, prompt)
```

Examples from the prompt shape:

```text
"that bear movie where leo gets attacked"
    -> "The Revenant Leonardo DiCaprio bear attack"

"movie about bear in london with marmalade"
    -> "Paddington London marmalade"
```

This can help both BM25 and semantic search because it adds concrete entities and terms that exist in the dataset.

### 6. Query expansion increases recall

Expansion adds related terms:

```python
def expand_query(query: str) -> str:
    prompt = f"""Expand the user-provided movie search query below with related terms.
    ...
    User query: "{query}"
    """
    return run_query_enhancement(query, prompt) + query
```

The idea is to broaden the match surface:

```text
"funny"  -> "comedy humorous amusing"
"family" -> "family kids children animated"
"grizzly" -> "grizzly bear animal"
```

The benefit is recall: more potentially relevant documents may be retrieved. The risk is drift: extra terms can pull in results that are nearby but not actually right.

### 7. Route enhancements through the CLI

The CLI controls enhancement explicitly:

```python
rrf_search_parser.add_argument(
    "--enhance",
    type=str,
    choices=SEARCH_ENHANCEMENT_METHODS,
    help="Query enhancement method",
)
```

The available choices live in shared constants:

```python
SEARCH_ENHANCEMENT_METHODS = ["spell", "rewrite", "expand"]
```

The command prints the before/after query when enhancement is used:

```python
if args.enhance:
    enhanced_query = enhance_query(args.query, args.enhance)
    print(f"Enhanced query ({args.enhance}): '{args.query}' -> '{enhanced_query}'\n")
    results = hybrid_search.rrf_search(enhanced_query, args.k, limit)
```

The user can see exactly what query was searched.

---

## Mental Model

Module 7 adds a pre-retrieval LLM step:

```text
Original query
    -> optional LLM enhancement
    -> improved query
    -> RRF hybrid search
    -> results
```

Choose enhancement by failure mode:

```text
Typo problem:
    use spell correction

Vague-language problem:
    use query rewriting

Too-narrow query problem:
    use query expansion
```

Boundary:

```text
Module 7 LLMs improve the query.
Later RAG modules use LLMs to generate answers.
```

---

## Implementation Notes

Main files involved:

- `cli/test_llm.py`
- `cli/lib/enhance_query.py`
- `cli/lib/hybrid_search.py`
- `cli/hybrid_search_cli.py`
- `cli/lib/search_utils.py`
- `.env`
- `.gitignore`
- `pyproject.toml`
- `uv.lock`

Relevant commits:

- `df97ddb lesson 7.2`: added OpenRouter/OpenAI dependencies, `.env` handling, `.gitignore` updates, and a test LLM script.
- `19d86ef lesson 7.3`: added spell-correction query enhancement and `--enhance` wiring.
- `22d85de lesson 7.5`: added rewrite and expansion enhancement modes and shared query-enhancement routing.

Useful commands:

```bash
uv add python-dotenv==1.1.0
uv add openai==2.44.0
uv run cli/test_llm.py
uv run cli/hybrid_search_cli.py rrf-search "Padington" --enhance spell --limit 5
uv run cli/hybrid_search_cli.py rrf-search "that bear movie where leo gets attacked" --enhance rewrite --limit 5
uv run cli/hybrid_search_cli.py rrf-search "grizzly" --enhance expand --limit 5
```
