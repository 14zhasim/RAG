# Module 4b: Productionising Semantic Search With RAG

## Lessons Learned

- **Core RAG retrieval need:** given a query embedding, find the nearest document or chunk embeddings.
  - Everything else exists to make that operation fast, filterable, persistent, and manageable.
- **Brute force search:** works conceptually but does not scale.
  - It compares the query embedding against every stored embedding, sorts the scores, and returns the top results.
  - This is fine for small datasets, but not for millions of chunks.
- **ANN indexes:** make vector search fast by searching a smaller candidate space.
  - HNSW, IVF, LSH, PQ, and DiskANN are indexing/compression techniques, not full databases by themselves.
- **Vector databases:** combine vector search with storage, metadata, IDs, filtering, APIs, persistence, and operational concerns.
  - FAISS is an ANN library.
  - Chroma is a convenient local vector store.
  - Qdrant is a production vector database server.
  - PostgreSQL + pgvector adds vector search to an existing relational database.
- **Architecture matters more than tiny ANN differences.**
  - If the rest of the app is relational, pgvector can be attractive.
  - If the app is mainly retrieval over chunks and metadata, a dedicated vector DB like Qdrant is often cleaner.
- **Managed services solve operations problems.**
  - Pinecone is not simply a "bigger Qdrant"; it is a managed vector database service.
  - The tradeoff is control versus convenience, not just scale.

---

## The Scale Limit

Module 4 used brute-force semantic search:

```text
query -> query embedding -> compare against every stored embedding -> sort -> top K
```

That is the simplest possible implementation. If there are `50,000` chunks, Python can store each chunk with its text, metadata, and embedding:

```python
chunks = [
    {
        "text": "...",
        "embedding": [...],
        "company": "Apple",
        "year": 2024,
    },
    ...
]
```

When a query comes in:

1. Compute the query embedding.
2. Compare it against all stored embeddings.
3. Sort by similarity.
4. Return the top results.

This works, but it becomes too slow as the corpus grows.

Example:

```text
100M stored vectors
query: "how do I reset my password?"
```

Brute force compares the query against all `100M` vectors. Approximate nearest neighbor indexes avoid scanning everything by searching a smaller candidate space.

---

## ANN Indexes

ANN means approximate nearest neighbor. The goal is:

```text
Find vectors that are probably very close to the query vector, much faster than brute force.
```

The tradeoff is speed versus exactness. ANN search may not always return the absolute nearest vector, but a good index gets high recall much faster than scanning the whole dataset.

| Method | Category | Main Idea | Pros | Cons | Typical Use Cases |
| ------ | -------- | --------- | ---- | ---- | ----------------- |
| **HNSW** | ANN index | Build a graph connecting nearby vectors | Excellent recall; very fast queries; widely adopted | Higher memory usage; index construction can be slower; may not always find the absolute nearest vector | Default choice for many RAG systems, including Qdrant, Weaviate, Pinecone, and Milvus |
| **IVF** | ANN index | Cluster vectors, then search only relevant clusters | Scales well to large datasets; tunable speed vs. accuracy | May miss neighbors near cluster boundaries if too few clusters are searched | Millions to billions of vectors; often paired with PQ |
| **LSH** | ANN index | Hash similar vectors into the same buckets | Simple; fast for some similarity measures; theoretical guarantees | Lower recall for modern dense embeddings; may need many hash tables | Older systems, specific similarity-search problems, and research applications |
| **PQ** | Compression | Compress vectors into compact codes | Greatly reduces memory; speeds up distance computations | Some accuracy loss due to quantization | Massive datasets; often combined with IVF as IVF-PQ |
| **DiskANN** | ANN index | Use a graph optimized for SSD-resident data | Handles datasets larger than RAM while keeping high performance | More complex implementation; depends on fast storage | Billion-scale vector search when RAM is limited |

Important distinction:

```text
ANN index:
    How do I find nearby vectors quickly?

Vector database:
    How do I store, manage, filter, update, and serve vectors in an application?
```

---

## FAISS Is An ANN Library

FAISS is not a database. It is a library of ANN algorithms.

It gives you tools such as:

- HNSW
- IVF
- PQ
- flat/brute-force search

You give FAISS vectors, and it builds an index:

```text
Embeddings
    -> FAISS
    -> HNSW / IVF / PQ / Flat index
    -> fast nearest-neighbor search
```

FAISS mostly solves:

```text
"Find nearest vectors."
```

It does not try to be a full application database. It does not primarily handle users, APIs, persistence, metadata filtering, authentication, backups, or multi-client server deployment.

Use FAISS directly when you are building your own retrieval layer or learning ANN internals. In many production apps, you use a vector database that has its own indexing backend under the hood.

---

## Chroma, Qdrant, And pgvector

### Chroma

Chroma is a convenient vector store for local RAG apps and prototypes.

Instead of manually managing an ANN index:

```python
index.add(vectors)
```

you work with collections:

```python
collection.add(
    documents=...,
    embeddings=...,
    metadatas=...,
)
```

Chroma stores the text, embeddings, IDs, and metadata together. It can persist to disk, so persistence is not the main difference between Chroma and a production vector DB. The bigger difference is operational maturity: concurrency, deployment, APIs, scaling, monitoring, and production tooling.

Think of Chroma as a very easy way to get a local RAG pipeline working.

### Qdrant

Qdrant is a production vector database server.

```text
Your app
    -> REST/gRPC API
    -> Qdrant server
    -> HNSW index
    -> stored vectors + metadata
```

It is designed for:

- persistent vector storage
- metadata filtering
- APIs
- concurrency
- deployment
- production operations

Qdrant is a good fit when the application is mainly retrieval over chunks and metadata.

### PostgreSQL + pgvector

PostgreSQL was not originally designed for vector search. `pgvector` adds:

- a vector data type such as `VECTOR(1536)`
- vector distance operators
- ANN indexes such as HNSW or IVF inside Postgres

Example table shape:

| id | company | year | section | text | embedding |
| -- | ------- | ---- | ------- | ---- | --------- |
| 1 | Apple | 2024 | Risk Factors | `We face...` | `[0.34, ...]` |

The embedding is just another column. You can then run queries like:

```sql
ORDER BY embedding <=> query_embedding
```

This means Postgres can store both structured data and vector embeddings:

```text
Users
Companies
Documents
SEC filings
Chunks
Embeddings
```

The benefit is architectural simplicity when the application already depends heavily on relational data, SQL joins, permissions, and transactions.

---

## Metadata Filtering

Metadata filtering is one of the biggest reasons vector databases matter in RAG.

Suppose a user asks:

```text
"Show me Microsoft's AI risk factors from 2024."
```

Without metadata filtering:

```text
Search all chunks.
Let the ANN index find nearby vectors.
```

With metadata filtering:

```text
company = Microsoft
year = 2024
document_type = 10-K
section = Risk Factors
```

Now maybe only a few hundred chunks remain. Vector search runs inside the right subset, which is usually faster and more accurate.

This is why Qdrant and PostgreSQL + pgvector are attractive for production RAG: they combine structured filtering with vector similarity search.

---

## Operations: Qdrant vs Pinecone

Qdrant and Pinecone are often compared as if Pinecone is just the "bigger" option. A better distinction is:

```text
Qdrant:
    vector database software
    can be self-hosted or used as a managed service

Pinecone:
    managed vector database service
    proprietary infrastructure operated for you
```

Analogy:

```text
PostgreSQL              -> database software
Amazon RDS PostgreSQL   -> managed database service

Qdrant                  -> vector database software
Pinecone                -> managed vector database service
```

As systems grow, the hardest questions become operational:

- What happens if a server dies?
- How do we scale across regions?
- How do we back up the data?
- How do we upgrade without downtime?
- How do we handle thousands of simultaneous queries?

Pinecone exists for teams that want:

```text
index.upsert(...)
index.query(...)
```

without managing Docker, monitoring, replication, autoscaling, backups, security patches, or load balancing.

Qdrant can still serve very large production workloads. Companies may choose to self-host Qdrant when they need more control, have DevOps support, or cannot let data leave their own infrastructure.

| Question | Qdrant | Pinecone |
| -------- | ------ | -------- |
| Can I run it myself? | Yes | No, it is managed |
| Do I manage upgrades? | Yes, if self-hosted | No |
| Do I manage scaling? | Yes, if self-hosted | No |
| Can it scale to large datasets? | Yes | Yes |
| Main tradeoff | More control | More convenience |

So the project-size ladder is not:

```text
small -> Chroma
medium -> Qdrant
large -> Pinecone
```

It is closer to:

```text
prototype -> Chroma
production -> Qdrant or Pinecone
choice -> depends on operations, control, and infrastructure preferences
```

---

## Choosing A Tool

| Need | Good Fit | Why |
| ---- | -------- | --- |
| Learn ANN internals | FAISS | Direct control over indexes like HNSW, IVF, PQ, and flat search |
| Build a local RAG prototype quickly | Chroma | Simple API, stores text/embeddings/metadata together, little setup |
| Build a production semantic-search service with control | Qdrant | Dedicated vector DB with APIs, metadata filtering, persistence, and self-hosting options |
| Avoid operating vector-search infrastructure | Pinecone | Managed vector database service; handles scaling and operations for you |
| Add vector search to an existing relational app | PostgreSQL + pgvector | Keep SQL, joins, permissions, transactions, and vectors in one database |
| Search billions of vectors across machines | Milvus / Pinecone / Weaviate / Qdrant | Designed for large-scale managed or distributed vector search |

For an SEC-filings RAG project, the decision is mostly architectural:

- **Choose Qdrant** if the app is primarily a retrieval system over chunks, embeddings, and metadata.
- **Choose PostgreSQL + pgvector** if the app already has rich relational data: users, companies, filings, permissions, conversations, and transactions.
- **Start with Chroma** if the goal is to learn the full RAG pipeline quickly before adding production infrastructure.
- **Choose Pinecone** if the goal is production vector search without operating the vector database yourself.

For many RAG systems, the vector database choice is not the biggest driver of answer quality. Mature systems using HNSW or similar indexes can retrieve similar results when configured well.

The bigger gains usually come from:

- better chunking that preserves document sections
- better metadata and filtering
- choosing an appropriate embedding model
- hybrid retrieval with keyword and vector search
- reranking results before sending them to the LLM

---

## Mental Model

Think in layers:

```text
Application
    -> Vector database or vector store
    -> ANN index
    -> Embeddings
```

Examples:

```text
Application -> Qdrant -> HNSW -> embeddings
Application -> PostgreSQL + pgvector -> HNSW/IVF -> embeddings
Python app -> FAISS -> HNSW/IVF/PQ -> embeddings
```

The part that often unlocks the confusion:

```text
HNSW, IVF, LSH, PQ, DiskANN:
    indexing/search techniques

FAISS:
    ANN algorithm library

Chroma, Qdrant, Pinecone, Weaviate, pgvector:
    systems that store/manage/query vectors for applications
```

As an application developer, you usually choose the storage/serving system first, then configure or accept its indexing strategy based on scale and performance needs.

For an SEC project, changing from Qdrant to Pinecone alone would probably not transform answer quality. Better chunking and metadata filters by company, filing type, year, and section would likely matter more.
