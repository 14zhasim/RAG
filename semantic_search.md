## Overview

When start new project, make sure uv install packages like openAI, numpy, nltk, python-dotenv, sentence-transformers, and 'masters time spent' doc for initial setup stuff to check

It is worth going through the course to understand main engineering decisions/tradeoffs you have to make.

Current understanding

- what stop works you use for keyword search - measures you take to clean up query (strip, filter punctuation with regex, stemming etc.)
- keyword search: k1 (how much increasing repetition of word in search result adds to its score), b (how much document's length vs average document length affects its score for a given word search)
- embedding model used in semnatic search
- semantic search: chunk and overlap size, how to chunk different docs/file types
- alpha paramaeter - weighting on semantics vs keyword score for hybrid search
- how you normalise score - min/max of absolute scores, using ranking to calculate rrf (where decide k parameter)
- search result limit

To prompt LLMs in the best way, use: https://platform.claude.com/docs/en/build-with-claude/prompt-engineering/overview

- use LLMs to clean search queries before performing them

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

_Why it's hard — breaking down the actual work:_

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

There is no final "squash into one chunk vector" step. The chunk's representation _is_ its bag of token embeddings, stored as-is.

At query time, you also embed the query into per-token vectors, then do a fine-grained matching (ColBERT uses something called "MaxSim": for each query token, find the _most similar_ document token, then sum those max scores across all query tokens). This lets you compare token-to-token instead of chunk-to-chunk.

So yes: no single "chunk embedding" ever gets created in ColBERT's pipeline. The multiple token vectors are the retrieval unit.

### Late chunking

You run the **entire document** (not just one chunk) through the model, so every token embedding is contextualized by the _whole document_, not just its local chunk.

After getting these full-document, context-rich token embeddings, you then decide chunk boundaries and **pool together** (e.g., average) the token embeddings that fall inside each chunk to produce **one embedding per chunk**.

So the "lateness" refers to chunking happening _after_ contextualization, rather than chunking the raw text first and then embedding each piece independently (which is regular/early chunking).

The end result of late chunking is a single vector per chunk, just like regular chunking, but that vector is richer because it was computed with full-document context before being pooled.

### Summary of the key difference

| Method        | Context window used    | Final representation                             |
| ------------- | ---------------------- | ------------------------------------------------ |
| ColBERT       | chunk-level context    | many vectors (one per token), no pooling         |
| Late chunking | whole-document context | one vector per chunk (pooled from token vectors) |

### When to use these approaches

**"Extremely precise search results" — what does this mean concretely?**

Precision here means: when you search, the _specific_ relevant passage ranks at the top, not just something topically related. Regular chunk-level embeddings average everything in a chunk into one vector, so subtle distinctions can get blurred out.

Example:

> A chunk contains: "The patient reported no history of penicillin allergy but experienced a severe reaction to amoxicillin during treatment last year."

If someone searches "penicillin allergy," a single chunk-level embedding might score this chunk as a decent match, because "penicillin" and "allergy" are both present, even though the actual meaning is the _opposite_ (no allergy to penicillin, but a reaction to a related drug).

With ColBERT's token-level matching, the retrieval score is built from many local token matches rather than one averaged chunk vector. This gives the model more opportunity to preserve exact evidence around terms like "no history" and "penicillin," though it still may not perfectly reason over negation.

**Does this relate to keyword search with contextual meaning?**

Sort of, but it's more general than keyword search. It's not "find the word penicillin," it's "understand what role the word penicillin is playing in this sentence, in context, and match that against what role it's playing in the query."

Regular embeddings do this too, but at the chunk level. ColBERT does it at the token level, so it's finer-grained.

**Examples of complex, nuanced text where context is critical:**

- Legal contracts: "The tenant shall not be liable for damages caused by the landlord's failure to maintain the property, except where such failure results from the tenant's own negligence." A search for "tenant liable for damages" needs to distinguish the exceptions and conditions, not just match on overlapping words.
- Medical records: Negations like "no history of X" versus affirmations like "history of X" completely flip the meaning, but the words overlap heavily.
- Scientific literature with technical qualifiers: "This treatment shows promise in mice models but has not been validated in human clinical trials." A query like "treatment effective in humans" should not strongly match this chunk, even though most of the relevant words appear.
- Code documentation with version-specific caveats: "This function was deprecated in version 3.0; use new*function() instead." A query for "how to use this function" needs to recognize this is telling you \_not* to use it.

**Practical decision guide:**

| Technique                      | Best for                                                                                                         | Tradeoff                                                                                       |
| ------------------------------ | ---------------------------------------------------------------------------------------------------------------- | ---------------------------------------------------------------------------------------------- |
| Regular chunk embeddings       | General FAQ, blogs, product docs, and search where "roughly relevant" is acceptable                              | Fast and simple, but subtle negations or exceptions can be blurred by pooling                  |
| Semantic chunking with overlap | Default baseline for most RAG systems                                                                            | Usually strong enough, but chunk boundaries and overlap still need tuning                      |
| Late chunking                  | Cases where broader document context improves each chunk's meaning, while keeping normal single-vector retrieval | Richer chunk vectors, but needs a long-context embedding model and still pools into one vector |
| ColBERT                        | Exact passage-level precision and fine-grained term interactions                                                 | Better token-level evidence matching, but more storage and retrieval complexity                |
| Cross-encoder or LLM reranker  | Final top results need stronger reasoning over negation, exceptions, or entailment                               | More accurate reranking, but slower and usually applied only after first-stage retrieval       |
| Hybrid search                  | Exact terms, IDs, names, legal clauses, financial metrics, or rare vocabulary matter                             | Handles lexical precision better, but requires score fusion or result merging                  |

For most general-purpose search, semantic chunking with overlap does a good job because subtle negations, exceptions, and high-cost misreadings are less common, and being "roughly right" is acceptable.

The lesson is essentially: use semantic chunking with overlap as the default baseline; add hybrid search when exact terms matter; use late chunking when wider context would make each chunk embedding better; reach for ColBERT when token-level matching is worth the extra cost; and use a reranker when the final ordering needs stronger reasoning over nuance.

---

## Cross-encoder re-ranking

### Why we installed `nltk` and `sentence-transformers`

`nltk` is for classic text processing in keyword search. In this project it is used for `PorterStemmer`, which reduces related word forms toward a shared stem. That helps BM25 / keyword search match terms more flexibly after tokenization, punctuation cleanup, and stop-word filtering.

`sentence-transformers` is for neural semantic search. In this project it loads embedding models like `all-MiniLM-L6-v2`, turns movie descriptions or chunks into vectors, and lets us compare query/document vectors with cosine similarity. It also supports cross-encoders, which are useful for re-ranking.

### How BERT models work

At a high level, BERT turns text into context-aware token vectors.

First, the text is tokenized. Sometimes tokens are whole words, sometimes they are word pieces.

```text
"Paddington loves marmalade"
```

might become:

```text
[CLS], Paddington, loves, marmalade, [SEP]
```

`[CLS]` is a special start token. `[SEP]` marks the end or separates two texts.

Each token starts as a vector. At first, this is just a learned lookup:

```text
Paddington -> initial vector
loves      -> initial vector
marmalade  -> initial vector
```

BERT also adds positional information, because otherwise it would not know word order.

The core trick is self-attention. Each token looks at the other tokens and decides which ones matter for its meaning.

In:

```text
bear in the woods
```

the token "bear" attends to words like "woods", so it leans animal/movie-ish.

In:

```text
bear the responsibility
```

the token "bear" attends to "responsibility", so it leans toward "carry" or "endure."

That is why BERT outputs contextual token vectors. Same word, different sentence, different vector.

BERT repeats this attention and transformation process many times across transformer layers. Each layer refines the token vectors. Early layers tend to capture simpler patterns, while later layers capture more abstract relationships.

At the end, BERT outputs one vector per token:

```text
[CLS]      -> vector
Paddington -> vector
loves      -> vector
marmalade  -> vector
[SEP]      -> vector
```

Then what happens depends on the task.

For bi-encoder semantic search:

```text
BERT token vectors -> pooling -> one sentence/chunk vector -> cosine similarity
```

For cross-encoder re-ranking:

```text
query + document -> BERT token vectors -> classifier/regression head -> relevance score
```

The simplest mental model:

```text
BERT = context-aware token vector machine
Pooling = token vectors -> one text vector
Classifier/regression head = BERT output -> task-specific prediction
```

### Bi-encoder

A bi-encoder is what the semantic search currently uses.

It handles the query and document separately:

```text
query -> BERT -> pooling -> query vector
doc   -> BERT -> pooling -> doc vector
query vector + doc vector -> cosine similarity
```

Pooling is needed because BERT does not naturally output one vector for a whole sentence or chunk. It outputs one contextual vector per token.

These are contextual token vectors, not simple dictionary definitions of each word. The vector for a token depends on the surrounding words in that specific input.

For example, the token vector for "bear" will be different in:

```text
bear in the woods
bear the responsibility
Paddington Bear
```

BERT produces a different context-aware vector for "bear" in each case because the surrounding words change what the token means.

Example:

```text
Paddington  -> [0.2, 0.9, -0.1, ...]
loves       -> [0.4, 0.1,  0.7, ...]
marmalade   -> [0.8, 0.3,  0.5, ...]
```

That gives multiple token vectors, but semantic search wants one vector for the whole query/chunk so it can do:

```text
query vector vs document vector
```

Pooling means combining many token vectors into one vector. The simplest version is mean pooling, where each vector dimension is averaged.

Tiny fake example:

```text
Paddington -> [2, 4]
loves      -> [4, 6]
marmalade  -> [6, 8]
```

Mean pooling:

```text
first dimension:  (2 + 4 + 6) / 3 = 4
second dimension: (4 + 6 + 8) / 3 = 6
```

Pooled sentence vector:

```text
[4, 6]
```

So:

```text
BERT: token vectors
pooling: all token vectors -> one chunk vector
```

Useful mental model:

```text
Pooling = summarizing all token vectors into one vector.
```

Common pooling methods:

- Mean pooling: average all token vectors into one vector. This is common for sentence-transformer style embeddings because it uses information from every token.
- CLS pooling: take only the final vector for the special `[CLS]` token at the start of the input. BERT-style models often use `[CLS]` as a whole-input summary for classification tasks.
- Max pooling: take the strongest value per vector dimension across all token vectors. This is less common in this project, but it is another way to collapse many token vectors into one.

Example of CLS pooling:

```text
[CLS]      -> [0.7, 0.2, 0.9, ...]
Paddington -> [0.2, 0.9, -0.1, ...]
loves      -> [0.4, 0.1,  0.7, ...]
marmalade  -> [0.8, 0.3,  0.5, ...]
```

With CLS pooling:

```text
sentence vector = final [CLS] vector
```

With mean pooling:

```text
sentence vector = average([CLS], Paddington, loves, marmalade)
```

For sentence embeddings, mean pooling often works better than raw CLS pooling unless the model was trained to make the CLS vector especially useful.

This is also why details can get blurred. If one token says "no" and another says "allergy," averaging everything into one vector may not preserve the exact relationship as well as a model that looks at token interactions directly.

This is fast because document vectors can be precomputed once and saved. At search time, only the query needs to be embedded, then compared against stored vectors.

The tradeoff is that the query and document do not see each other until the final cosine similarity step. That can miss subtle interactions like negation, exceptions, or the exact role of a word in context.

### Cross-encoder

A cross-encoder takes the query and document together:

```text
query + document -> BERT -> classifier/regression head -> relevance score
```

Instead of creating two separate vectors and comparing them, it asks:

> Given this exact query and this exact document, how relevant is this pair?

Because both texts are processed together, query words can directly interact with document words inside the transformer. This usually gives better relevance judgments than a bi-encoder, especially for subtle matches.

The output is already a relevance score, so there is no cosine similarity step.

The classifier/regression head is a small prediction layer attached to the top of BERT. BERT produces contextual token vectors for the combined query/document input. The head takes the final representation, often the `[CLS]` vector, and maps it to the thing we want to predict.

For a cross-encoder input:

```text
[CLS] query [SEP] document [SEP]
```

BERT processes the whole thing and outputs contextual vectors:

```text
[CLS]      -> vector
query toks -> vectors
doc toks   -> vectors
```

The classifier/regression head is usually a small neural network layer that takes the final `[CLS]` vector as the summary of the whole query-document pair.

So:

```text
[CLS] vector -> small prediction layer -> score
```

A very simplified version:

```text
score = weight_vector · cls_vector + bias
```

That means:

- multiply parts of the `[CLS]` vector by learned weights
- add them up
- add a bias
- optionally pass through an activation like sigmoid

For classification, the head might output categories:

```text
relevant vs not relevant
```

For regression/ranking, the head outputs a number:

```text
relevance score = 0.83
```

So in cross-encoder reranking:

```text
[CLS] query [SEP] document [SEP] -> BERT -> head -> relevance score
```

The head is trained on examples of query/document pairs, so it learns patterns like "this document answers the query well" or "these words overlap but the meaning does not match."

Important part: the head is small. Most of the "understanding" comes from BERT processing the query and document together. The head just converts that rich representation into the final relevance score.

### Why cross-encoders are used for re-ranking

Cross-encoders are accurate but expensive at scale. If there are 5,000 movies and one query, a cross-encoder would need to run once for every query/movie pair:

```text
(query, movie 1)
(query, movie 2)
...
(query, movie 5000)
```

That is too slow compared with embedding search.

The normal pipeline is:

```text
1. Use BM25 / semantic search / RRF to get top candidates.
2. Use a cross-encoder to rescore only those candidates.
3. Sort by the cross-encoder score.
```

This is the same shape as the LLM reranker, except the scorer is a specialized relevance model instead of a general chat model.

### LLM reranker vs cross-encoder

The current LLM reranker does this:

```text
Prompt: query + movie
LLM: "8.5"
```

A cross-encoder does this:

```text
model.predict([(query, movie)])
-> relevance score
```

The cross-encoder is less flexible than an LLM, but faster and cheaper. It only has one job: take a query/document pair and output a relevance score. That is why it can be thought of as a regression model.

### Where this fits

Current stack:

```text
BM25: exact lexical matching
semantic bi-encoder: concept matching with vectors
RRF / weighted hybrid: combine retrieval signals
LLM reranker: expensive final judgment
cross-encoder reranker: cheaper/faster final judgment
```

The clean mental model:

- Use `nltk` / BM25 when exact words matter.
- Use bi-encoder embeddings when concepts matter and speed matters.
- Use RRF / hybrid search when both exact words and semantic meaning matter.
- Use cross-encoder re-ranking when top candidates are decent but ordering needs to improve.
- Use LLM re-ranking when flexible judgment matters enough to accept cost, latency, and API limits.

Cross-encoder re-ranking is the practical middle ground between semantic search and LLM re-ranking.

## Keyword vs Semantic

Use semantic for searching concepts - and other techniques for accuracy
Usekeyword for exact / string results

## Cross-encoder fine-tuning

A fast way to get up and running with a re-ranker is to use the Cohere API or one of its competitors. Typically these APIs use cross-encoders.
https://docs.cohere.com/reference/about

The second major advantage of cross-encoders is that they can be fine-tuned on your specific domain relatively easily. You can train them on your own query-document pairs to learn the exact relevance patterns for your use case, and they are cheaper and faster than LLMs.
https://www.ibm.com/think/topics/fine-tuning

## The data problem

FAB v1.1 has 40 public questions. Even the full private set is 537 questions. That is not enough to fine-tune a cross-encoder from scratch — you'd typically want thousands of labelled query-passage pairs to get meaningful signal. So the benchmark itself cannot be your training data.

## Where the training data would actually come from

**Option 1 — Synthetic generation from the filings themselves.** You already have EDGAR. For any company in the benchmark, you can download their 10-Ks, 10-Qs, and 8-Ks, run your ingestion pipeline to generate chunks, then use an LLM to generate synthetic question-passage pairs: "given this passage, what question would this answer?" You'd generate hundreds of pairs per filing, giving you thousands of training examples at near-zero cost. This is the standard approach when domain data is scarce — it's called **LLM-augmented training data generation** and there is citable literature on it (GPL — Generative Pseudo Labelling, Wang et al. 2022).

**Option 2 — Existing financial QA datasets as a proxy.** FinQA (Chen et al. 2021) has 8,281 expert-annotated financial question-answer pairs over earnings reports, with gold evidence passages. That is ready-made cross-encoder training data for financial document relevance. Not perfectly aligned to FAB's question types but close enough for domain adaptation.

**Option 3 — Hard negative mining from your own pipeline.** Run your MVP1 pipeline on the 40 public FAB questions. For each question, your hybrid search will return a ranked list. The passages ranked highly but not containing the correct answer are your **hard negatives** — exactly what cross-encoder fine-tuning needs. This is the highest-signal training data you can get because it is specific to your pipeline's failure modes. But you need a working pipeline first.

## Honest assessment for your dissertation

Fine-tuning a cross-encoder is a defensible contribution but a significant one that requires:

- A working ingestion pipeline first
- A data generation or curation step
- Training infrastructure
- Ablation showing it improves over the generic model

Given your timeline (code stopping ~22 August), this is only worth pursuing if MVP1 is done early and the generic cross-encoder is clearly the bottleneck. If your accuracy is stuck at 60% and error analysis shows the right passage is being retrieved but ranked poorly, fine-tuning is the fix. If the right passage is not being retrieved at all, better chunking or query decomposition is the fix — and fine-tuning won't help.
wh
