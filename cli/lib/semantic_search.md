#embedding-driven boundary detection 
#create embedding for each sentence in document
#(i.e., using cosine similarity between sentences to decide where to split) - if similarity between two sentences falls below certain threshold, chunk it
#example is LlamaIndex's SemanticSplitterNodeParser
#Useful when:
###text is 'unstructured wall-of-text' with high variable topic density
###have compute/latency budget to spare, given it requires embedding each sentence BEFORE chunking
###retrieval quality critical enough to justify extra complexity

#production RAG systems use simple strategies: EXPLOIT ANY STRUCTURE IN YOUR DOCUMENT BEFORE USING EXPENSIVE, GENERAL-PURPOSE TECHNIQUES
#Fixed-size token/word chunking with overlap (what we did first, built above)
#Sentence or paragraph-based chunking (what we just built above)
#Structure-aware chunking (splitting on markdown headers, HTML tags, metadata (titles, section headers) etc.)

#normal RAG production strategy:
#chunk by document structure (headers/paragraphs)
#tune chunks size and overlap to sue case
#use good reranker downstream to fix mediocre retrieval

#LLM-based chunking (llm decides chunk sizes). it is worth it when:
#Tables and financial statements need to stay intact. A naive sentence or token splitter will happily slice a balance sheet in half. 
###An LLM (or a rules engine) can recognize "this is a table, keep it as one unit" or "this is Item 7A (market risk disclosures), keep the whole item together."
#Sections carry semantic weight that maps to real query patterns. Analysts often ask things like "what are the risk factors" or "what's the liquidity discussion" — mapping directly to Item 1A, Item 7, etc. 
###An LLM (or even simpler regex/structure parsing) that chunks along these known section boundaries will outperform arbitrary splits.
#summarization-aware chunking
###You need summarization-aware chunking, e.g., an LLM chunker that also tags each chunk with metadata (fiscal year, section, subsidiary mentioned) 
###for later filtering.

#However, LLM based chunking is costly. Can get value from doing structure-aware parsing first (from the actual HTML/XBRL tags SEC filings ship with), 
#table-aware extraction, where tables are treated as atomic units
#Then, as a fallback, use LLM-based or even embedding-driven chunking as a fallback/refinement: reserved for unstructured narrative sections (e.g. MD&A) where 
#paragraph boundaries are not enough


---
How useful is the RAG course?

You're right that in industry, most RAG systems are built on frameworks like LangChain, LlamaIndex, LangGraph, or vector DB SDKs (Pinecone, Weaviate, etc.) rather than hand-rolled from scratch. Nobody writing a production system is implementing their own BM25 or cosine similarity function from first principles the way this course has you do.

But here's why the "under the hood" approach still pays off, especially for something like SEC filings:

- Frameworks are leaky abstractions. When your retrieval quality is bad on 10-K filings (and it will be, at first — financial documents are notoriously hard for RAG), you need to know why. Is it a chunking problem? An embedding model mismatch? A similarity metric issue? A tokenization edge case with numbers/tables? If you only know "call RecursiveCharacterTextSplitter," you can tune its parameters, but you can't diagnose why it's failing or design a custom splitter for tables and Item boundaries. This course is teaching you what's inside that black box.

- You will need custom logic for domain-specific documents. LangChain/LlamaIndex give you generic chunkers. They do NOT know that "Item 7A" should stay together, or that a financial table shouldn't be split by paragraph. Building that custom parser requires understanding chunking mechanics, not just calling .split_text().

- Debugging retrieval failures requires the theory. If an analyst asks "what's the debt maturity schedule" and your system retrieves garbage, you need to reason about whether it's a keyword-search miss (BM25 term mismatch), a semantic-search miss (embedding didn't capture "debt maturity" contextually), or a chunking failure (the schedule got split across chunks). This course walks through BM25, embeddings, chunking, and hybrid search — the actual failure modes you'll debug in a real system.

- Frameworks change fast; concepts don't. LangChain's API has churned significantly over the past couple years. Cosine similarity, TF-IDF, and chunking tradeoffs haven't changed at all. Learning the framework without the theory means you're one API refactor away from being lost.

---

Official docs (best starting point for each):

LangChain: python.langchain.com — has a good conceptual walkthrough plus a "RAG tutorial" that mirrors what you just built, but using their abstractions (TextSplitter, VectorStore, Retriever, RunnableSequence for orchestration via LCEL — LangChain Expression Language).
LlamaIndex: docs.llamaindex.ai — particularly strong for document parsing/node-based indexing; their "Understanding" section walks through loading, indexing, storing, querying, which maps closely to what you've been building by hand.
For orchestration patterns specifically:

LangGraph docs (LangChain's graph-based orchestration layer for more complex/agentic flows) — useful once you're past basic chains and want branching/looping logic.
LlamaIndex's "Workflows" docs cover a similar orchestration concept.

Practical tip given what you just learned: When you go through those docs, keep mapping their abstractions back to concepts from this course — e.g., LangChain's RecursiveCharacterTextSplitter is doing sentence/paragraph-aware chunking similar to what you just implemented, and their VectorStoreRetriever wraps the cosine-similarity search from earlier in this course. That mapping is what makes the framework docs click faster instead of feeling like a black box.

ME:
can a lot of these issues by avoided using hybrid search, and combine with LLMs to retrieve relevant results - where we follow filing structure first for chunking etc? and using finance-specific embedding models? for tables - use specialised partsing? and use graph-rag for cross-references?

GraphRAG for cross-references — good fit, with a caveat.
This one's legitimate but heavier machinery. GraphRAG (building an explicit knowledge graph of entities/relationships, e.g., linking "Item 7A" mentions to their actual content, or connecting subsidiary names to parent companies across filings) does solve the cross-referencing problem well, and is used in some financial/legal RAG products for this reason. The caveat: building and maintaining a knowledge graph from filings is a significant engineering investment (extraction pipelines, entity resolution, graph storage/query layer). It's usually justified only when cross-reference resolution is a primary user need (e.g., "trace this obligation across all referenced sections") rather than applied by default.

*Why it's hard — breaking down the actual work:*

- Entity extraction. You need to reliably pull named entities (companies, people, dollar amounts, dates, financial metrics) out of unstructured text. This usually means running an LLM or NER model over every chunk, which is slow and imperfect — it will miss entities, misclassify them, or extract inconsistent forms (e.g., "Apple Inc." vs "Apple" vs "AAPL" as three different nodes when they should be one).

- Entity resolution/deduplication. Once extracted, you need to merge duplicate references to the same entity. This is a notoriously hard problem on its own — is "the Company" in paragraph 5 the same as "Registrant" in paragraph 50? Real GraphRAG systems (like Microsoft's original implementation) spend a lot of their complexity budget here.

- Relationship extraction. Beyond entities, you need to extract how things relate — again typically LLM-driven, again imperfect and requiring validation/cleanup.

- Graph construction and storage. You need a graph database (Neo4j, or a lighter-weight library) and a schema design — what counts as a node type, what counts as an edge type. This is a real data-modeling task, not just "dump extracted triples into a database."

- Community detection / summarization (in the full Microsoft GraphRAG approach). The original technique also clusters the graph into "communities" and generates LLM summaries of each cluster, so you can answer high-level questions ("what are the main themes across all filings") rather than just point lookups. This adds another full LLM-processing pass over your data.

- Query-time graph traversal logic. You need to decide, given a query, which nodes/edges to pull, how many hops to traverse, and how to merge graph context with any vector-retrieved text. This routing logic itself is nontrivial.


A realistic one-month version:

- Skip full automated entity/relationship extraction across a huge corpus. Instead, pick a narrow, well-defined relationship type relevant to 10-Ks — e.g., just "which sections reference which other sections" (Item cross-references) or "company → subsidiary" mentions. This is a much smaller, tractable extraction task than general-purpose entity/relationship extraction.
- Use an LLM with structured output (e.g., asking it to return JSON of entities/relationships found in a chunk) rather than building a custom NER pipeline.
- Use a lightweight graph representation — even a Python dict of adjacency lists or NetworkX — instead of standing up a full graph database. This keeps you focused on demonstrating the retrieval improvement, not infrastructure.
- Skip community detection/summarization (the most expensive part of full GraphRAG) unless you have time left over.
- Combine it with the vector/hybrid search you already know — GraphRAG doesn't need to replace your pipeline, it can supplement it for cross-reference-heavy queries specifically, while hybrid search handles everything else.


*What Microsoft's graphrag library gives you out of the box:*

It's an open-source Python package (github.com/microsoft/graphrag) that automates the pipeline I described:

- Entity/relationship extraction via LLM prompts (already engineered and tested)
- Entity resolution/deduplication logic
- Graph construction (outputs to Parquet files, can load into Neo4j or just query directly)
- Community detection (using the Leiden algorithm) and hierarchical community summarization
- Both "global search" (for broad, thematic questions) and "local search" (for specific entity-focused questions) query modes already implemented
- So the hard engineering — entity resolution, graph algorithms, prompt engineering for extraction — is done for you. You mainly need to: configure it, point it at your document corpus, run the indexing pipeline, and wire up the query interface.

*What still takes real time even with the library:*

- Cost and runtime at scale. The indexing pipeline makes many LLM calls per document (extraction, summarization at multiple graph levels). For a large corpus of 10-Ks (each 100+ pages), this adds up fast in both dollars and wall-clock time. You'll likely need to start with a small subset (a handful of filings) rather than a full corpus, purely for cost/time reasons within a month.

- Domain adaptation. The default prompts are tuned for general entity extraction. Financial filings have domain-specific entities (financial line items, regulatory terms, subsidiary structures) that benefit from prompt customization — this is real tuning work, not zero-effort.

- Table handling is still your problem. GraphRAG's extraction pipeline is built for prose text. It doesn't magically solve the "financial tables get mangled" issue from earlier — you still need to preprocess/parse tables separately before feeding text into the GraphRAG pipeline.

- Integration with your existing hybrid pipeline. You'll still need to write the glue code that decides when to query the graph vs. your vector/keyword index, and how to merge results — GraphRAG doesn't automatically know about your BM25/embedding setup from the rest of the course.

- Learning the library's configuration surface. It has a fair number of config options (chunk size, LLM model choice, community detection parameters) that require some experimentation to get sensible outputs, especially on a new domain like financial filings.

*Verdict for your one-month timeline:*

Using Microsoft's graphrag library instead of hand-building the pipeline makes this meaningfully more feasible — you're skipping the hardest engineering (entity resolution, graph algorithms) entirely. A realistic scope:

- Use graphrag on a small set of 10-Ks (maybe 3-5 filings) rather than a full corpus
- Handle table extraction/preprocessing yourself before feeding text in
- Do light prompt customization for financial entity types
- Wire up "graph search" as one retrieval path alongside your hybrid keyword+semantic search from this course, rather than replacing it
- Compare retrieval quality on cross-reference-heavy queries (graph) vs. general queries (hybrid) as your evaluation story