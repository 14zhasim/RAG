## embedding-driven boundary detection 

- create embedding for each sentence in document
- (i.e., using cosine similarity between sentences to decide where to split) - if similarity between two sentences falls below certain threshold, chunk it
- example is LlamaIndex's SemanticSplitterNodeParser
- Useful when:
  - text is 'unstructured wall-of-text' with high variable topic density
  - have compute/latency budget to spare, given it requires embedding each sentence BEFORE chunking
  - retrieval quality critical enough to justify extra complexity

**production RAG systems use simple strategies: EXPLOIT ANY STRUCTURE IN YOUR DOCUMENT BEFORE USING EXPENSIVE, GENERAL-PURPOSE TECHNIQUES**

- Fixed-size token/word chunking with overlap (what we did first, built above)
- Sentence or paragraph-based chunking (what we just built above)
- Structure-aware chunking (splitting on markdown headers, HTML tags, metadata (titles, section headers) etc.)

**normal RAG production strategy:**

- chunk by document structure (headers/paragraphs)
- tune chunks size and overlap to sue case
- use good reranker downstream to fix mediocre retrieval

**LLM-based chunking (llm decides chunk sizes). it is worth it when:**

- Tables and financial statements need to stay intact. A naive sentence or token splitter will happily slice a balance sheet in half. 
  - An LLM (or a rules engine) can recognize "this is a table, keep it as one unit" or "this is Item 7A (market risk disclosures), keep the whole item together."
- Sections carry semantic weight that maps to real query patterns. Analysts often ask things like "what are the risk factors" or "what's the liquidity discussion" — mapping directly to Item 1A, Item 7, etc. 
  - An LLM (or even simpler regex/structure parsing) that chunks along these known section boundaries will outperform arbitrary splits.
- summarization-aware chunking
  - You need summarization-aware chunking, e.g., an LLM chunker that also tags each chunk with metadata (fiscal year, section, subsidiary mentioned) 
    for later filtering.

**However, LLM based chunking is costly. Can get value from doing structure-aware parsing first (from the actual HTML/XBRL tags SEC filings ship with),**

- table-aware extraction, where tables are treated as atomic units
- Then, as a fallback, use LLM-based or even embedding-driven chunking as a fallback/refinement: reserved for unstructured narrative sections (e.g. MD&A) where 
  paragraph boundaries are not enough

---

## How useful is the RAG course?

You're right that in industry, most RAG systems are built on frameworks like LangChain, LlamaIndex, LangGraph, or vector DB SDKs (Pinecone, Weaviate, etc.) rather than hand-rolled from scratch. Nobody writing a production system is implementing their own BM25 or cosine similarity function from first principles the way this course has you do.

But here's why the "under the hood" approach still pays off, especially for something like SEC filings:

- Frameworks are leaky abstractions. When your retrieval quality is bad on 10-K filings (and it will be, at first — financial documents are notoriously hard for RAG), you need to know why. Is it a chunking problem? An embedding model mismatch? A similarity metric issue? A tokenization edge case with numbers/tables? If you only know "call RecursiveCharacterTextSplitter," you can tune its parameters, but you can't diagnose why it's failing or design a custom splitter for tables and Item boundaries. This course is teaching you what's inside that black box.

- You will need custom logic for domain-specific documents. LangChain/LlamaIndex give you generic chunkers. They do NOT know that "Item 7A" should stay together, or that a financial table shouldn't be split by paragraph. Building that custom parser requires understanding chunking mechanics, not just calling .split_text().

- Debugging retrieval failures requires the theory. If an analyst asks "what's the debt maturity schedule" and your system retrieves garbage, you need to reason about whether it's a keyword-search miss (BM25 term mismatch), a semantic-search miss (embedding didn't capture "debt maturity" contextually), or a chunking failure (the schedule got split across chunks). This course walks through BM25, embeddings, chunking, and hybrid search — the actual failure modes you'll debug in a real system.

- Frameworks change fast; concepts don't. LangChain's API has churned significantly over the past couple years. Cosine similarity, TF-IDF, and chunking tradeoffs haven't changed at all. Learning the framework without the theory means you're one API refactor away from being lost.

---

## Official docs (best starting point for each):

LangChain: python.langchain.com — has a good conceptual walkthrough plus a "RAG tutorial" that mirrors what you just built, but using their abstractions (TextSplitter, VectorStore, Retriever, RunnableSequence for orchestration via LCEL — LangChain Expression Language).

LlamaIndex: docs.llamaindex.ai — particularly strong for document parsing/node-based indexing; their "Understanding" section walks through loading, indexing, storing, querying, which maps closely to what you've been building by hand.

### For orchestration patterns specifically:

LangGraph docs (LangChain's graph-based orchestration layer for more complex/agentic flows) — useful once you're past basic chains and want branching/looping logic.

LlamaIndex's "Workflows" docs cover a similar orchestration concept.

Practical tip given what you just learned: When you go through those docs, keep mapping their abstractions back to concepts from this course — e.g., LangChain's RecursiveCharacterTextSplitter is doing sentence/paragraph-aware chunking similar to what you just implemented, and their VectorStoreRetriever wraps the cosine-similarity search from earlier in this course. That mapping is what makes the framework docs click faster instead of feeling like a black box.

## ME:

can a lot of these issues by avoided using hybrid search, and combine with LLMs to retrieve relevant results - where we follow filing structure first for chunking etc? and using finance-specific embedding models? for tables - use specialised partsing? and use graph-rag for cross-references?

### GraphRAG for cross-references — good fit, with a caveat.

This one's legitimate but heavier machinery. GraphRAG (building an explicit knowledge graph of entities/relationships, e.g., linking "Item 7A" mentions to their actual content, or connecting subsidiary names to parent companies across filings) does solve the cross-referencing problem well, and is used in some financial/legal RAG products for this reason. The caveat: building and maintaining a knowledge graph from filings is a significant engineering investment (extraction pipelines, entity resolution, graph storage/query layer). It's usually justified only when cross-reference resolution is a primary user need (e.g., "trace this obligation across all referenced sections") rather than applied by default.

*Why it's hard — breaking down the actual work:*

- Entity extraction. You need to reliably pull named entities (companies, people, dollar amounts, dates, financial metrics) out of unstructured text. This usually means running an LLM or NER model over every chunk, which is slow and imperfect — it will miss entities, misclassify them, or extract inconsistent forms (e.g., "Apple Inc." vs "Apple" vs "AAPL" as three different nodes when they should be one).

- Entity resolution/deduplication. Once extracted, you need to merge duplicate references to the same entity. This is a notoriously hard problem on its own — is "the Company" in paragraph 5 the same as "Registrant" in paragraph 50? Real GraphRAG systems (like Microsoft's original implementation) spend a lot of their complexity budget here.

- Relationship extraction. Beyond entities, you need to extract how things relate — again typically LLM-driven, again imperfect and requiring validation/cleanup.

- Graph construction and storage. You need a graph database (Neo4j, or a lighter-weight library) and a schema design — what counts as a node type, what counts as an edge type. This is a real data-modeling task, not just "dump extracted triples into a database."

- Community detection / summarization (in the full Microsoft GraphRAG approach). The original technique also clusters the graph into "communities" and generates LLM summaries of each cluster, so you can answer high-level questions ("what are the main themes across all filings") rather than just point lookups. This adds another full LLM-processing pass over your data.

- Query-time graph traversal logic. You need to decide, given a query, which nodes/edges to pull, how many hops to traverse, and how to merge graph context with any vector-retrieved text. This routing logic itself is nontrivial.

### A realistic one-month version:

- Skip full automated entity/relationship extraction across a huge corpus. Instead, pick a narrow, well-defined relationship type relevant to 10-Ks — e.g., just "which sections reference which other sections" (Item cross-references) or "company → subsidiary" mentions. This is a much smaller, tractable extraction task than general-purpose entity/relationship extraction.
- Use an LLM with structured output (e.g., asking it to return JSON of entities/relationships found in a chunk) rather than building a custom NER pipeline.
- Use a lightweight graph representation — even a Python dict of adjacency lists or NetworkX — instead of standing up a full graph database. This keeps you focused on demonstrating the retrieval improvement, not infrastructure.
- Skip community detection/summarization (the most expensive part of full GraphRAG) unless you have time left over.
- Combine it with the vector/hybrid search you already know — GraphRAG doesn't need to replace your pipeline, it can supplement it for cross-reference-heavy queries specifically, while hybrid search handles everything else.

### What Microsoft's graphrag library gives you out of the box:

It's an open-source Python package (github.com/microsoft/graphrag) that automates the pipeline I described:

- Entity/relationship extraction via LLM prompts (already engineered and tested)
- Entity resolution/deduplication logic
- Graph construction (outputs to Parquet files, can load into Neo4j or just query directly)
- Community detection (using the Leiden algorithm) and hierarchical community summarization
- Both "global search" (for broad, thematic questions) and "local search" (for specific entity-focused questions) query modes already implemented
- So the hard engineering — entity resolution, graph algorithms, prompt engineering for extraction — is done for you. You mainly need to: configure it, point it at your document corpus, run the indexing pipeline, and wire up the query interface.

### What still takes real time even with the library:

- Cost and runtime at scale. The indexing pipeline makes many LLM calls per document (extraction, summarization at multiple graph levels). For a large corpus of 10-Ks (each 100+ pages), this adds up fast in both dollars and wall-clock time. You'll likely need to start with a small subset (a handful of filings) rather than a full corpus, purely for cost/time reasons within a month.

- Domain adaptation. The default prompts are tuned for general entity extraction. Financial filings have domain-specific entities (financial line items, regulatory terms, subsidiary structures) that benefit from prompt customization — this is real tuning work, not zero-effort.

- Table handling is still your problem. GraphRAG's extraction pipeline is built for prose text. It doesn't magically solve the "financial tables get mangled" issue from earlier — you still need to preprocess/parse tables separately before feeding text into the GraphRAG pipeline.

- Integration with your existing hybrid pipeline. You'll still need to write the glue code that decides when to query the graph vs. your vector/keyword index, and how to merge results — GraphRAG doesn't automatically know about your BM25/embedding setup from the rest of the course.

- Learning the library's configuration surface. It has a fair number of config options (chunk size, LLM model choice, community detection parameters) that require some experimentation to get sensible outputs, especially on a new domain like financial filings.

### Verdict for your one-month timeline:

Using Microsoft's graphrag library instead of hand-building the pipeline makes this meaningfully more feasible — you're skipping the hardest engineering (entity resolution, graph algorithms) entirely. A realistic scope:

- Use graphrag on a small set of 10-Ks (maybe 3-5 filings) rather than a full corpus
- Handle table extraction/preprocessing yourself before feeding text in
- Do light prompt customization for financial entity types
- Wire up "graph search" as one retrieval path alongside your hybrid keyword+semantic search from this course, rather than replacing it
- Compare retrieval quality on cross-reference-heavy queries (graph) vs. general queries (hybrid) as your evaluation story

---

## ColBERT vs late chunking

### ColBERT

At retrieval time, ColBERT keeps **every token embedding**, contextualized by the chunk it was encoded in (the chunk is typically what's fed through the model at once).

There is no final "squash into one chunk vector" step. The chunk's representation *is* its bag of token embeddings, stored as-is.

At query time, you also embed the query into per-token vectors, then do a fine-grained matching (ColBERT uses something called "MaxSim": for each query token, find the *most similar* document token, then sum those max scores across all query tokens). This lets you compare token-to-token instead of chunk-to-chunk.

So yes: no single "chunk embedding" ever gets created in ColBERT's pipeline. The multiple token vectors are the retrieval unit.

### Late chunking

You run the **entire document** (not just one chunk) through the model, so every token embedding is contextualized by the *whole document*, not just its local chunk.

After getting these full-document, context-rich token embeddings, you then decide chunk boundaries and **pool together** (e.g., average) the token embeddings that fall inside each chunk to produce **one embedding per chunk**.

So the "lateness" refers to chunking happening *after* contextualization, rather than chunking the raw text first and then embedding each piece independently (which is regular/early chunking).

The end result of late chunking is a single vector per chunk, just like regular chunking, but that vector is richer because it was computed with full-document context before being pooled.

### Summary of the key difference

| Method | Context window used | Final representation |
| --- | --- | --- |
| ColBERT | chunk-level context | many vectors (one per token), no pooling |
| Late chunking | whole-document context | one vector per chunk (pooled from token vectors) |

### When to use these approaches

**"Extremely precise search results" — what does this mean concretely?**

Precision here means: when you search, the *specific* relevant passage ranks at the top, not just something topically related. Regular chunk-level embeddings average everything in a chunk into one vector, so subtle distinctions can get blurred out.

Example:

> A chunk contains: "The patient reported no history of penicillin allergy but experienced a severe reaction to amoxicillin during treatment last year."

If someone searches "penicillin allergy," a single chunk-level embedding might score this chunk as a decent match, because "penicillin" and "allergy" are both present, even though the actual meaning is the *opposite* (no allergy to penicillin, but a reaction to a related drug).

With ColBERT's token-level matching, the retrieval score is built from many local token matches rather than one averaged chunk vector. This gives the model more opportunity to preserve exact evidence around terms like "no history" and "penicillin," though it still may not perfectly reason over negation.

**Does this relate to keyword search with contextual meaning?**

Sort of, but it's more general than keyword search. It's not "find the word penicillin," it's "understand what role the word penicillin is playing in this sentence, in context, and match that against what role it's playing in the query."

Regular embeddings do this too, but at the chunk level. ColBERT does it at the token level, so it's finer-grained.

**Examples of complex, nuanced text where context is critical:**

- Legal contracts: "The tenant shall not be liable for damages caused by the landlord's failure to maintain the property, except where such failure results from the tenant's own negligence." A search for "tenant liable for damages" needs to distinguish the exceptions and conditions, not just match on overlapping words.
- Medical records: Negations like "no history of X" versus affirmations like "history of X" completely flip the meaning, but the words overlap heavily.
- Scientific literature with technical qualifiers: "This treatment shows promise in mice models but has not been validated in human clinical trials." A query like "treatment effective in humans" should not strongly match this chunk, even though most of the relevant words appear.
- Code documentation with version-specific caveats: "This function was deprecated in version 3.0; use new_function() instead." A query for "how to use this function" needs to recognize this is telling you *not* to use it.

**Practical decision guide:**

| Technique | Best for | Tradeoff |
| --- | --- | --- |
| Regular chunk embeddings | General FAQ, blogs, product docs, and search where "roughly relevant" is acceptable | Fast and simple, but subtle negations or exceptions can be blurred by pooling |
| Semantic chunking with overlap | Default baseline for most RAG systems | Usually strong enough, but chunk boundaries and overlap still need tuning |
| Late chunking | Cases where broader document context improves each chunk's meaning, while keeping normal single-vector retrieval | Richer chunk vectors, but needs a long-context embedding model and still pools into one vector |
| ColBERT | Exact passage-level precision and fine-grained term interactions | Better token-level evidence matching, but more storage and retrieval complexity |
| Cross-encoder or LLM reranker | Final top results need stronger reasoning over negation, exceptions, or entailment | More accurate reranking, but slower and usually applied only after first-stage retrieval |
| Hybrid search | Exact terms, IDs, names, legal clauses, financial metrics, or rare vocabulary matter | Handles lexical precision better, but requires score fusion or result merging |

For most general-purpose search, semantic chunking with overlap does a good job because subtle negations, exceptions, and high-cost misreadings are less common, and being "roughly right" is acceptable.

The lesson is essentially: use semantic chunking with overlap as the default baseline; add hybrid search when exact terms matter; use late chunking when wider context would make each chunk embedding better; reach for ColBERT when token-level matching is worth the extra cost; and use a reranker when the final ordering needs stronger reasoning over nuance.

## Keyword vs Semantic
Use semantic for searching concepts - and other techniques for accuracy
Usekeyword for exact / string results
