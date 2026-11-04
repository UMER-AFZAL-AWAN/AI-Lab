# =============================================================================
# 20_nlp_and_llms / 03 — RAG AND EMBEDDINGS
# =============================================================================
#
# LIKE YOU ARE 5:
# An LLM knows a lot (from training) but can't remember YOUR documents.
# RAG (Retrieval Augmented Generation):
#   1. Store your documents as "fingerprints" (vectors)
#   2. When you ask a question, find the matching fingerprints
#   3. Give the LLM those documents + your question
#   4. LLM answers using those documents
#
# C# ANALOGY:
# Without RAG: the LLM is like a developer with no access to your codebase.
# With RAG: inject the relevant files into the developer's context.
#   RAG = grep your codebase + paste the relevant files into the prompt.
#
# =============================================================================

import torch
import math

print("=" * 60)
print("PART 1: THE PROBLEM RAG SOLVES")
print("=" * 60)
print("""
LLM limitations:
  1. Knowledge cutoff: doesn't know events after training date
  2. No private data: doesn't know your company's internal docs
  3. Hallucination: invents facts when uncertain
  4. Context limit: can't fit all your docs in the prompt

RAG approach:
  1. At indexing time:
     - Convert all documents to embedding vectors
     - Store in a vector database

  2. At query time:
     - Encode the question as a vector
     - Find the top-K most similar document chunks
     - Add those chunks to the LLM prompt
     - LLM answers from the retrieved context

The LLM then answers like a human who can search their notes.
""")

print("=" * 60)
print("PART 2: EMBEDDING VECTORS")
print("=" * 60)
print("""
Embedding = a vector that captures semantic meaning.
  "cat"  →  [0.12, -0.34, 0.78, ...]  (768 or 1536 dimensions)
  "kitten" → [0.13, -0.33, 0.77, ...]  (similar vector!)

Training embedding models:
  Contrastive learning: similar pairs → nearby vectors, different → far apart
  SBERT (Sentence-BERT): fine-tuned BERT on (sentence1, sentence2, similarity) triplets

Popular embedding models:
  OpenAI text-embedding-3-large: 3072 dims, $0.00013/1K tokens (cloud)
  sentence-transformers/all-MiniLM-L6-v2: 384 dims, runs locally (free)
  intfloat/e5-large-v2: 1024 dims, runs locally (better quality)
  BAAI/bge-m3: multilingual, 1024 dims, runs locally
""")

# Embedding math without an actual model
def cosine_similarity(a, b):
    return (a @ b) / (a.norm() * b.norm())

def l2_distance(a, b):
    return (a - b).norm()

torch.manual_seed(1)
dim = 16
# Fake embeddings where similar texts are close
emb_cat    = torch.randn(dim)
emb_kitten = emb_cat + 0.3 * torch.randn(dim)   # close to cat
emb_dog    = emb_cat + 0.9 * torch.randn(dim)   # somewhat close
emb_car    = torch.randn(dim)                     # completely different

for name, emb in [("kitten", emb_kitten), ("dog", emb_dog), ("car", emb_car)]:
    sim = cosine_similarity(emb_cat, emb)
    l2  = l2_distance(emb_cat, emb)
    print(f"  sim(cat, {name:8s}) = {sim.item():.3f}  (L2={l2.item():.2f})")

print()
print("=" * 60)
print("PART 3: VECTOR DATABASE (the memory)")
print("=" * 60)
print("""
A vector database stores (embedding, metadata) pairs and answers:
  "Find the K nearest embeddings to this query vector."

This is an approximate nearest neighbour (ANN) search.
For millions of vectors, exact search is too slow; ANN is fast enough.

Popular vector databases:
  FAISS:      Facebook's C++ library, fastest, local
              pip install faiss-cpu
  ChromaDB:   simple local vector DB, great for prototyping
              pip install chromadb
  Pinecone:   managed cloud service
  Weaviate:   open source with schema
  pgvector:   PostgreSQL extension (great if you already use Postgres)
""")

# Simple brute-force vector "database" (like FAISS at tiny scale)
class TinyVectorDB:
    """
    Minimal vector database: stores vectors, finds K nearest by cosine similarity.
    Real usage: use FAISS or ChromaDB.
    """
    def __init__(self):
        self.vectors  = []  # list of tensors
        self.metadata = []  # list of dicts

    def add(self, vector, metadata):
        self.vectors.append(vector / vector.norm())   # normalise for cosine sim
        self.metadata.append(metadata)

    def search(self, query_vector, k=3):
        """Return top-K most similar results."""
        if not self.vectors:
            return []
        query_norm = query_vector / query_vector.norm()
        db_matrix  = torch.stack(self.vectors)   # (N, dim)
        scores     = db_matrix @ query_norm       # (N,) cosine similarities
        top_k      = scores.topk(min(k, len(self.vectors)))
        return [
            {"score": top_k.values[i].item(), "metadata": self.metadata[top_k.indices[i]]}
            for i in range(len(top_k.values))
        ]

# Build a tiny knowledge base
db  = TinyVectorDB()
dim = 16

# "Documents" with fake embeddings (in reality: encode with a model)
docs = [
    "PyTorch is a deep learning framework with dynamic computation graphs",
    "LLaMA is a large language model from Meta",
    "CUDA is NVIDIA's parallel computing platform",
    "Backpropagation computes gradients via the chain rule",
    "Transformers use attention to process sequences",
]

torch.manual_seed(42)
doc_embeddings = torch.randn(len(docs), dim)   # fake embeddings

for i, (doc, emb) in enumerate(zip(docs, doc_embeddings)):
    db.add(emb, {"id": i, "text": doc})

# Query
query_emb = doc_embeddings[3] + 0.1 * torch.randn(dim)   # close to "backprop"
results = db.search(query_emb, k=3)

print("Query similar to: 'Backpropagation computes gradients via the chain rule'")
print("Top 3 results:")
for r in results:
    text = r["metadata"]["text"]
    print(f"  score={r['score']:.3f}  |  {text}")

print()
print("=" * 60)
print("PART 4: THE FULL RAG PIPELINE")
print("=" * 60)

rag_pipeline_code = '''
from sentence_transformers import SentenceTransformer
from transformers import pipeline

# ── 1. Embedding model (runs locally, free) ──────────────────────────────────
embed_model = SentenceTransformer("sentence-transformers/all-MiniLM-L6-v2")

# ── 2. Build the index (do this once, then save to disk) ─────────────────────
documents = [
    "The AnimateDiff pipeline requires a base SD model.",
    "Wan 2.1 uses VACE architecture for video generation.",
    "ControlNet guides generation using a conditioning image.",
]

doc_embeddings = embed_model.encode(documents)   # (N, 384) float32

# Using FAISS for scalable search:
import faiss
index = faiss.IndexFlatIP(384)   # Inner Product = cosine sim on normalised vectors
faiss.normalize_L2(doc_embeddings)
index.add(doc_embeddings)

# ── 3. At query time ──────────────────────────────────────────────────────────
user_question = "How do I control AnimateDiff output?"

query_emb = embed_model.encode([user_question])
faiss.normalize_L2(query_emb)
scores, indices = index.search(query_emb, k=2)   # top-2 results

retrieved = [documents[i] for i in indices[0]]

# ── 4. Build the prompt ───────────────────────────────────────────────────────
context = "\\n".join(f"- {doc}" for doc in retrieved)

prompt = f"""Answer the question using ONLY the provided context.

Context:
{context}

Question: {user_question}

Answer:"""

# ── 5. Generate with LLM ─────────────────────────────────────────────────────
# llm = pipeline("text-generation", model="phi-2", ...)
# response = llm(prompt, max_new_tokens=200)[0]["generated_text"]
print("Retrieved context:")
for doc in retrieved:
    print(f"  {doc}")
print(f"\\nPrompt sent to LLM:\\n{prompt}")
'''

print("Full RAG pipeline code (requires pip install sentence-transformers faiss-cpu):")
print(rag_pipeline_code)

print("=" * 60)
print("PART 5: CHUNKING STRATEGY")
print("=" * 60)
print("""
Documents are too long to embed as a whole (embedding ≤512 tokens).
Split documents into CHUNKS before embedding.

Chunking strategies:
  Fixed size:         Split every 512 tokens. Simple but may cut mid-sentence.
  Sentence splitter:  Split at sentence boundaries. Better coherence.
  Semantic chunks:    Use an embedding model to detect topic changes.
  Recursive:          Try paragraph → sentence → word (most common, used by LangChain)

Overlap:
  Chunk 1: tokens 0-512
  Chunk 2: tokens 450-962  (50-token overlap with chunk 1)
  → Prevents missing context that spans a chunk boundary.

Typical parameters:
  chunk_size=512, chunk_overlap=50   ← good starting point
  chunk_size=256, chunk_overlap=32   ← for dense technical docs

Libraries:
  LangChain: RecursiveCharacterTextSplitter (most common)
  LlamaIndex: SentenceSplitter
  NLTK:  sent_tokenize for sentence boundaries
""")

def simple_chunk(text, chunk_size=100, overlap=20):
    """Split text into overlapping chunks by character count."""
    chunks = []
    start  = 0
    while start < len(text):
        end = start + chunk_size
        chunks.append(text[start:end])
        start += chunk_size - overlap
    return chunks

sample = "PyTorch is a deep learning framework. " * 5
chunks = simple_chunk(sample, chunk_size=80, overlap=15)
print(f"Text: {len(sample)} chars → {len(chunks)} chunks of ~80 chars with 15-char overlap")
for i, c in enumerate(chunks[:3]):
    print(f"  chunk {i}: ...{c[:60]}...")

print()
print("=" * 60)
print("QUICK REFERENCE: RAG vs FINE-TUNING")
print("=" * 60)
print("""
                RAG                         Fine-Tuning
──────────────────────────────────────────────────────────────
Good for        Dynamic/private knowledge   Fixed new behaviour
When to use     New docs added regularly    Specific task/style
Cost            Retrieval latency           Training compute
Privacy         Docs never leave your DB    Data in training
Hallucination   Grounded in retrieved docs  Still possible
Knowledge       Can update without retraining  Needs re-training
Example use     Customer support bot        Code assistant
""")

# =============================================================================
# EXPECTED OUTPUT:
# Cosine similarities: kitten closest to cat, car furthest
# RAG search returns "backprop" doc as top result
# Chunking: text split into overlapping chunks
# =============================================================================
