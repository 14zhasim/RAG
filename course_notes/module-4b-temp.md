# 4b Temp: Production RAG Storage Gaps To Review

This file is a temporary holding area for the extra production-RAG points to consider before editing `module-04b-production-semantic-search.md`.

The goal is not to add all of this directly. The goal is to review each point, decide what belongs in `4b`, and keep graph-specific material in `4c`.

---

## 1. Metadata In The Vector DB vs Metadata For Managing The System

Vector databases can store metadata. The key distinction is what the metadata is for:

```text
Vector DB metadata:
    helps retrieval

Application/document metadata:
    helps run and manage the application
```

Only the chunk text gets embedded, unless metadata is deliberately included in the text sent to the embedding model. Metadata is stored separately as filterable fields.

Example vector DB record:

```python
{
    "text": "Apple reported net sales of...",
    "embedding": [...],
    "metadata": {
        "ticker": "AAPL",
        "fiscal_year": 2024,
        "fiscal_quarter": "Q4",
        "filing_type": "10-K",
        "section": "income_statement",
    },
}
```

At search time:

```text
Find chunks semantically similar to "revenue"
but only where:
    ticker = AAPL
    fiscal_year = 2024
    fiscal_quarter = Q4
```

This is the metadata needed for filtered retrieval. For example:

```text
chunk_id
document_id
ticker
filing_type
fiscal_year
fiscal_quarter
section
page_number
```

The separate database question is about broader system state. That metadata answers questions like:

```text
Debugging: why did the answer cite the wrong page?
Auditing: what documents were used for this query?
Monitoring: which ingestion jobs failed?
Reprocessing: which filings need to be re-parsed or re-embedded?
Permissions: can this user access this filing?
```

That is why a fuller production application may use a separate application/document metadata store, such as:

- PostgreSQL
- MongoDB
- Elasticsearch / OpenSearch

The rule:

```text
For the current dissertation/prototype, the vector DB can store chunks, embeddings,
and the metadata needed for retrieval filtering.

For a fuller production application, teams may add an authoritative document/application DB,
then copy only retrieval-relevant fields into the vector DB.
```

If that separate production DB is needed, PostgreSQL is often a practical default for structured app data such as users, companies, filings, permissions, conversations, ingestion jobs, citations, and audit records. MongoDB can also work, especially when the parsed document structure is more document-shaped than relational.

This means some metadata is duplicated. That is okay if the stores have different roles:

```text
PostgreSQL / MongoDB:
    source of truth

Vector DB:
    fast retrieval/filtering copy
```

It is not master/slave replication. It is closer to:

```text
source database -> indexing pipeline -> vector database
```

If they disagree:

```text
source database wins
vector database gets rebuilt/refreshed
```

For a dissertation/prototype, manual reindexing is fine:

```bash
python ingest.py --document AAPL_2024_10K.pdf
python rebuild_index.py
```

For production, automated reindexing is normal:

1. Document metadata changes in Postgres.
2. App marks affected chunks as `needs_reindex`.
3. Background worker picks them up.
4. Worker regenerates embeddings if needed.
5. Worker upserts updated metadata/vectors into Qdrant.
6. Chunk is marked `indexed`.

Sometimes teams also run consistency checks:

```text
Find chunks where source.updated_at > vector_indexed_at.
Reindex those chunks.
```

Two extra production caveats follow from this.

First, **embedding model versioning** matters. Embeddings from different models do not live in the same vector space. If chunks are embedded with one model and queries are embedded with another, similarity scores stop being meaningful.

Every embedded chunk should therefore store fields such as:

```text
embedding_model
embedding_dimensions
embedded_at
embedding_version / index_version
```

For a prototype, if the embedding model changes, rebuild the index. For production, teams may use separate collections/index versions or dual-write during migration.

Second, **index staleness** is possible. A query can hit the vector DB while a document is mid-reindex, which can return stale chunks, duplicate chunks, or mixed old/new embeddings.

For a prototype, avoid this by running ingestion/rebuilds offline. For production, common fixes include:

```text
stable chunk IDs
indexing_status = indexed
active index_version filters
blue/green collections
```

Development implication: Chroma can still be a reasonable starting point, but every chunk should carry stable IDs and provenance fields so retrieved answers can be traced back to the source document, page, parser run, and retrieval event.

Question for `4b`:

Should we add a short section explaining that vector DB metadata is enough for retrieval filters, but a separate metadata/application store becomes useful when citations, provenance, permissions, audits, ingestion jobs, and reprocessing matter?

---

## 2. Object / Blob Storage

The other AI's point:

> Where do the actual source PDFs live?

This is correct, but it should be phrased carefully.

The real requirement is:

```text
Every chunk/result must be traceable back to the original source file.
```

For a dissertation prototype, this does not necessarily require cloud blob storage. A local filesystem can play the same role:

```text
data/
  raw_pdfs/
    AAPL_2024_10K.pdf
    MSFT_2024_10K.pdf

  parsed/
    AAPL_2024_10K.json

  chunks/
    AAPL_2024_10K_chunks.json
```

Each chunk should store source metadata:

```python
{
    "document_id": "aapl-2024-10k",
    "source_path": "data/raw_pdfs/AAPL_2024_10K.pdf",
    "page_number": 17,
    "section": "income_statement",
    "chunk_id": "aapl-2024-10k-page17-chunk03",
}
```

That is enough for:

```text
debugging citations
checking bad retrievals
rerunning parsing
showing provenance
reprocessing documents
```

In production, the same role is usually handled by object/blob storage:

```text
S3
Azure Blob Storage
Google Cloud Storage
MinIO
```

Blob storage becomes useful when:

```text
multiple machines need access to the PDFs
the app is deployed beyond one laptop
files are too large or numerous for local storage
upload/download APIs are needed
durability, backups, and access control matter
```

The vector database stores chunks and embeddings. The metadata database stores parsed structure and provenance. The original PDF or CIM should still be retrievable by ID or URI.

Example:

```text
source_uri = "data/raw_pdfs/AAPL_2024_10K.pdf"

or, in production:

source_uri = "s3://bucket/sec/aapl/2024-10k.pdf"

chunk_id = "aapl-2024-10k-risk-0042"
page_number = 17
bbox = [x1, y1, x2, y2]
```

This matters because if an answer cites page 17, or if the document needs to be re-parsed with a better layout parser, the original file must still exist.

For the dissertation, the practical recommendation is:

```text
Use local filesystem storage for raw PDFs.
Keep full PDFs out of git.
Store stable source paths/URIs, document IDs, page numbers, and section metadata with each chunk.
Design the metadata so source_uri could later become an Azure Blob or S3 path.
```

Question for `4b`:

Should `4b` explain that original documents need a stable source location, where local filesystem storage is fine for a dissertation prototype and object/blob storage is the production version?

---

## 3. Three-Tier RAG Storage

The other AI's point:

> Production systems are usually tri-layered.

This is the cleanest way to summarize the missing architecture:

```text
Raw files
    -> object storage

Parsed document structure / metadata / provenance
    -> PostgreSQL or MongoDB

Chunks + embeddings + retrieval metadata
    -> vector DB / vector store
```

For GraphRAG, there may be a fourth layer:

```text
Entities + relationships
    -> graph DB
```

But graph-specific detail belongs in `module-04c-graphrag-and-relationship-retrieval.md`, not `4b`.

Question for `4b`:

Should we add a concise `Production RAG Storage Layers` section with this exact shape?

---

## 4. Weaviate And Milvus

The other AI's point:

> Weaviate and Milvus are named but not explained.

This is fair.

Weaviate and Milvus fit into the same layer as Qdrant and Pinecone:

```text
Chunks + embeddings + retrieval filters
    -> Chroma / Qdrant / Pinecone / Weaviate / Milvus / pgvector
```

They are not replacements for raw PDF storage, and they are not mainly the application metadata database. They are alternative products in the **vector database / vector search layer**.

Useful distinction:

```text
pgvector:
    PostgreSQL extension for vector similarity search.
    Useful when vectors should live inside an existing Postgres-backed application.

sqlite-vec:
    SQLite extension for local vector similarity search.
    Useful for tiny, local, single-file prototypes or edge apps.

LanceDB:
    local-first embedded vector database.
    Useful when you want disk-backed vector search without running a separate server.

Qdrant:
    production vector DB with good filtering, self-hosting, and operational control.

Weaviate:
    production vector DB with built-in hybrid search and richer schema features.
    It can combine vector search, BM25 keyword search, filters, and schema/class-based modelling.

Milvus:
    large-scale distributed vector DB.
    More relevant when the main challenge is huge vector collections and serious infrastructure.

Pinecone:
    managed vector DB service.
    Useful when you want vector search without operating the infrastructure yourself.
```

These tools are all "good enough" for different shapes of vector retrieval. The choice is not a ranking from bad to good; it is about whether the project wants a local embedded store, an extension inside an existing database, a dedicated vector database, or a managed service.

Concise comparison:

```text
Chroma:
    simplest local vector store

Qdrant:
    dedicated vector DB, strong general choice

Weaviate:
    dedicated vector DB with built-in hybrid/schema features

pgvector:
    useful when Postgres is already central and you want vectors in the same DB
```

The tradeoff:

```text
pgvector reduces infrastructure if Postgres is already part of the app.
Qdrant/Weaviate are stronger dedicated vector-search systems.
```

Hybrid dense + BM25 search affects this choice because there are two main implementation paths:

```text
Path A: one retrieval system supports dense + sparse search
    Qdrant dense vectors + sparse vectors in one collection
    Weaviate built-in hybrid search
    OpenSearch/Elasticsearch with lexical and vector search

Path B: separate retrievers fused in application code
    Chroma/Qdrant for semantic vector search
    BM25 in Python, Elasticsearch, or OpenSearch for keyword search
    RRF or weighted fusion combines the results
```

The course prototype naturally follows Path B because it manually builds BM25, semantic search, and RRF. A production version could keep that split or move more of the hybrid retrieval into a database that supports both dense and sparse retrieval.

Weaviate is relevant because this course later implements hybrid search manually: BM25 plus semantic search plus fusion. Weaviate can support that kind of dense + sparse retrieval inside one vector database.

Milvus is relevant as a scale-oriented vector database. It is less likely to be the first choice for a small dissertation build, but useful to know as the large-scale distributed option.

For the notes, this does not need to become a long product comparison. The main clarification is:

```text
Weaviate and Milvus are not new architectural layers.
They are alternative vector database choices.
```

Question for `4b`:

Should the existing tool table add one clearer sentence each for Weaviate and Milvus?

---

## 5. Embedded vs Client-Server Vector Stores

The other AI's point:

> Embedded vs client-server vector DBs are not framed as a real axis.

This is correct, with one nuance.

The axis is:

```text
Embedded / local:
    runs inside or near your app
    minimal infrastructure

Client-server:
    separate database service your app connects to over an API
```

Examples:

```text
Embedded / local-ish:
    Chroma PersistentClient
    LanceDB
    sqlite-vec

Database-extension:
    pgvector inside PostgreSQL
    sqlite-vec inside SQLite

Client-server:
    Qdrant
    Pinecone
    Weaviate
    Milvus
```

Nuance: Chroma can also run in client-server mode, so it is not purely embedded. But it is commonly used as a lightweight local/prototype vector store.

LanceDB is worth mentioning because it is an embedded/local-first vector database: more like a local vector DB than a separate hosted service.

`pgvector` and `sqlite-vec` are slightly different from Chroma/LanceDB because they add vector search to an existing database. `pgvector` fits PostgreSQL-backed apps; `sqlite-vec` fits tiny/local SQLite apps.

Question for `4b`:

Should we add a short subsection called `Embedded vs Client-Server Vector Stores`?

---

## 6. Multi-Model / Hybrid Databases

The other AI's point:

> Some databases can serve as a single-system answer rather than forcing separate relational, vector, and graph systems.

This is correct, but it should not be read as "put everything in PostgreSQL."

The useful idea is:

```text
Do not add separate infrastructure unless the separate tool solves a real problem.
```

For the project, the tradeoff is **simplicity** versus **specialisation**.

An all-in-Postgres design could mean:

```text
PostgreSQL:
    users
    filings
    metadata
    chunks
    embeddings via pgvector
    maybe graph-like edge tables
    maybe full-text search
```

This is attractive because it gives:

```text
one database
simpler deployment
simpler backups
simpler debugging
fewer services to run
fewer moving parts during dissertation development
```

But it is not better at everything.

A specialised multi-system design could mean:

```text
PostgreSQL:
    app data / metadata

Qdrant:
    vector retrieval

Neo4j:
    graph relationships

object storage:
    raw PDFs

queues + workers:
    ingestion and reindexing jobs
```

This is attractive because each tool is specialised:

```text
Qdrant is better as a dedicated vector DB.
Neo4j is better as a graph DB.
Object storage is better for raw files.
Queues/workers are better for async ingestion.
```

But it adds operational overhead:

```text
more services
more configuration
more failure points
more deployment work
more time spent wiring infrastructure
```

For the dissertation, the sane answer is progressive architecture:

```text
Stage 1:
    local files for PDFs
    Chroma/Qdrant for chunks + embeddings + retrieval metadata
    scripts for ingestion/reindexing

Stage 2:
    add PostgreSQL only if persistent app/document state is needed:
        ingestion jobs
        parser runs
        citations
        audits
        users/conversations

Stage 3:
    add Neo4j only if GraphRAG is genuinely part of Experiment 2
    and relationship traversal is central to the method

Stage 4:
    add object storage / queues / workers only if deployment or scale requires them
```

The rule:

```text
Use the simplest architecture that still preserves the research requirements.
```

So the point is not:

```text
Put everything in Postgres.
```

It is:

```text
A single-system design can be good when infrastructure simplicity matters.
A multi-system design is good when specialised capabilities matter enough to justify the complexity.
```

Hybrid search is one example of this tradeoff. A single system like Weaviate or Qdrant with dense+sparse support can reduce moving parts. Separate BM25 and semantic retrievers can be easier to understand, test, and swap independently, but require application-level fusion.

Question for `4b`:

Should we add this as an architectural tradeoff under the existing Qdrant/Pinecone/Postgres decision section, with the progressive architecture framing?

---

## Possible `4b` Addition

If we decide to edit `4b`, the cleanest addition may be one concise section:

```md
## Production RAG Storage Layers
```

Suggested content:

```text
Object storage:
    raw PDFs / source files

Document metadata store:
    parsed structure, provenance, page numbers, sections

Vector store:
    chunks, embeddings, retrieval metadata

Optional graph store:
    entities, relationships, dependencies
```

Then a short subsection:

```md
## Embedded vs Client-Server Vector Stores
```

Possible comparison table:

| Tool | Shape | Best For |
| ---- | ----- | -------- |
| Chroma | Local/persistent or client-server | Quick prototypes |
| LanceDB | Embedded/local-first | Simple local vector search without a separate service |
| sqlite-vec | SQLite extension | Tiny/local/edge apps |
| Qdrant | Client-server vector DB | Production retrieval with control |
| Pinecone | Managed vector DB service | Production without operating infra |
| Weaviate | Client-server vector DB | Built-in hybrid search |
| Milvus | Distributed vector DB | Very large-scale vector search |
| pgvector | Postgres extension | Apps already centered on Postgres |

---

## Current Recommendation

The other AI is broadly right, but it mixes two different missing topics.

Add to `4b`:

- production storage layers
- object storage
- document metadata stores
- embedded vs client-server vector stores
- clearer one-line distinctions for Weaviate, Milvus, LanceDB, and sqlite-vec
- the tradeoff of using Postgres as a simpler multi-purpose system

Keep in `4c`:

- Neo4j
- graph databases
- Microsoft GraphRAG
- GraphRAG retrieval patterns
- relationship retrieval
- graph storage versus `networkx`

---

## Reference Docs To Consult

These are source docs worth reading when refining `module-04`, `module-04b`, and `module-04c`.

### Semantic Search And Embeddings

- [Sentence Transformers Semantic Search docs](https://sbert.net/examples/sentence_transformer/applications/semantic-search/README.html)
  - Best practical explanation of embedding documents, embedding queries, and comparing with cosine similarity.
- [Sentence Transformers main docs](https://sbert.net/)
  - Useful for understanding bi-encoders, cross-encoders, embeddings, and similarity models.

### Vector Search, Filtering, And Indexing

- [Qdrant filtering guide](https://qdrant.tech/articles/vector-search-filtering/)
  - Good for understanding metadata/payload filtering.
- [Qdrant indexing docs](https://qdrant.tech/documentation/manage-data/indexing/)
  - Good for vector indexes and payload indexes.
- [Qdrant search docs](https://qdrant.tech/documentation/search/search/)
  - Useful for filtered vector search in practice.
- [pgvector GitHub](https://github.com/pgvector/pgvector)
  - Best source for PostgreSQL vector search, HNSW, IVFFlat, cosine distance, and related operators.

### Production RAG Storage

- [Amazon S3 user guide](https://docs.aws.amazon.com/AmazonS3/latest/userguide/Welcome.html)
  - Useful for the object/blob storage mental model.
- [MongoDB Vector Search docs](https://www.mongodb.com/docs/vector-search/)
  - Useful mainly to see how document fields and vector search can live together.
- [LanceDB docs](https://docs.lancedb.com/)
  - Good for local-first / embedded vector database thinking.
- [Chroma docs](https://docs.trychroma.com/docs/overview/getting-started)
  - Good for simple local vector store workflows.

### GraphRAG And Graph Databases

- [Microsoft GraphRAG docs](https://microsoft.github.io/graphrag/)
  - Best for understanding GraphRAG as a pipeline.
- [Microsoft Research GraphRAG](https://www.microsoft.com/en-us/research/project/graphrag/)
  - Good conceptual overview.
- [Microsoft GraphRAG GitHub](https://github.com/microsoft/graphrag)
  - Practical implementation reference.
- [Neo4j GraphRAG Python docs](https://neo4j.com/docs/neo4j-graphrag-python/current/user_guide_rag.html)
  - Good for graph database and RAG integration.
- [LlamaIndex Property Graph guide](https://developers.llamaindex.ai/python/framework/module_guides/indexing/lpg_index_guide/)
  - Best framework-style intro to property graph RAG.



Good instinct — for a dissertation you want primary/authoritative sources you can actually cite anyway, not AI summaries. Here's where each piece actually lives.

**Vector DB / ANN fundamentals**
- **Pinecone's learning center** (pinecone.io/learn) — genuinely the best free resource on ANN algorithms (HNSW, IVF, PQ), written by people who build this for a living. Vendor-authored but technically honest, widely cited even by non-Pinecone-users.
- **Qdrant's documentation + blog** (qdrant.tech/documentation, qdrant.tech/articles) — directly relevant since it's your likely production choice; covers hybrid search, sparse vectors, filtering internals with real benchmarks.
- **The original HNSW paper** — Malkov & Yashunin, "Efficient and robust approximate nearest neighbor search using Hierarchical Navigable Small World graphs" (2018, arXiv:1603.09320). Citable, and you'll want at least one primary ANN paper in your lit review if you're justifying an index choice.
- **FAISS wiki on GitHub** (github.com/facebookresearch/faiss/wiki) — if you want ANN internals from the library that started the space.

**Hybrid search / RRF**
- Cormack, Clarke, Büttcher, "Reciprocal Rank Fusion outperforms Condorcet and individual Rank Learning Methods" (SIGIR 2009) — the actual RRF paper, short and citable.
- Elasticsearch/OpenSearch blog posts on hybrid search — practical implementation detail beyond the paper.

**Reranking / cross-encoders**
- Sentence-Transformers documentation (sbert.net) — the standard reference for bi-encoder vs cross-encoder, written by the people who maintain the dominant library for both.
- Nils Reimers' original Sentence-BERT paper (2019, arXiv:1908.10084) if you want the primary source.

**GraphRAG**
- Microsoft Research's GraphRAG paper: Edge et al., "From Local to Global: A Graph RAG Approach to Query-Focused Summarization" (2024, arXiv:2404.16130) — this is the one to actually cite, not a blog summary of it.
- Neo4j's own GraphRAG documentation and blog (neo4j.com/developer-blog) — practical Cypher/LlamaIndex integration examples.
- Microsoft's GraphRAG GitHub repo (microsoft/graphrag) — read the actual pipeline code/docs rather than secondhand explanations.

**Database architecture tradeoffs (relational/vector/graph)**
- Martin Kleppmann, *Designing Data-Intensive Applications* — not RAG-specific but the standard reference for storage-layer tradeoffs (consistency, replication, when to split systems); worth having on your shelf regardless of this project.
- pgvector, LanceDB, and Weaviate's own docs for the specific extension/embedded/client-server distinctions.

**General orientation**
- LlamaIndex and LangChain documentation — both frameworks' docs walk through the same architectural decisions (chunking, hybrid retrieval, graph indices) with working code, and you'll be reading LlamaIndex docs anyway for the Jerry Liu course.

Practical approach given your timeline: don't read all of this cover to cover. Use Pinecone's learn center and Qdrant's blog for the ANN/hybrid mechanics (an afternoon), pull the four papers above (RRF, HNSW, GraphRAG, Sentence-BERT) as citable primary sources for Chapter 2/3 alongside your existing four must-reads, and treat Kleppmann as background you dip into only if a specific architecture question comes up during the build.