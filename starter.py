"""
L7 — Chunk and Compare  (STARTER)
===================================
Run with:
    python starter.py

Your goal: implement two chunking strategies, embed both sets of chunks, run
the same queries against each, and compare results.

Required features:
    1. A document of at least 500 words (paste your own notes or write a
       multi-paragraph summary of course concepts)
    2. Fixed-size chunking: 300-character chunks, 50-character overlap
    3. Paragraph-based chunking: split on double newlines (\n\n)
    4. Embed all chunks from both strategies
    5. Run 3 different queries against both sets
    6. For each query display the top 2 results from each strategy with scores
    7. A written analysis at the end (as print statements)

Key concepts:
    Fixed-size chunking with overlap:
        chunks = []
        start = 0
        while start < len(text):
            chunks.append(text[start:start + chunk_size])
            start += chunk_size - overlap

    Paragraph-based chunking:
        chunks = [p.strip() for p in text.split("\\n\\n") if p.strip()]

    Searching a list of embedded chunks:
        scores = cosine_similarity(query_emb, chunk_embeddings)[0]
        top_idx = np.argsort(scores)[::-1][:n]
"""

from sentence_transformers import SentenceTransformer
from sklearn.metrics.pairwise import cosine_similarity
import numpy as np

# ── Document ─────────────────────────────────────────────────────────────────
# TODO: Paste or write a 500–800 word document here.
#       It should cover multiple distinct topics so different chunks are
#       retrievable. You can use course notes from Modules 1–6.

DOCUMENT = """
Python programming provides the foundation for building software applications. Variables store information, functions organize reusable instructions, and loops allow programs to repeat operations. Lists and dictionaries organize collections of data. Exception handling helps programs respond to unexpected situations without crashing. Object-oriented programming introduces classes and objects to organize more complex applications. These concepts are important because almost every backend service and AI application depends on reliable Python code.

Pandas is a Python library used to clean, transform, and analyze datasets. DataFrames organize information into rows and columns, making it easier to filter records, calculate statistics, and inspect patterns. Developers often use Pandas to process CSV files, handle missing values, and prepare information for visualization. Data cleaning is essential because inaccurate or inconsistent information can produce unreliable results in both traditional analytics and machine learning applications.

Relational databases store structured information in tables. SQL allows developers to create, read, update, and delete records. Relationships connect related tables using primary and foreign keys. SQLAlchemy is a Python ORM that maps Python classes to database tables. Instead of writing every SQL query manually, developers can interact with database records through Python objects. Database design is important for applications that manage users, tasks, products, or other persistent information.

REST APIs allow different software applications to communicate through HTTP requests. Common HTTP methods include GET for retrieving resources, POST for creating resources, PATCH for partial updates, and DELETE for removing resources. API endpoints provide organized access to application functionality. HTTP status codes communicate whether requests succeed or fail. For example, 200 indicates success, 201 indicates resource creation, 404 indicates a missing resource, and 422 commonly indicates request validation errors in FastAPI.

FastAPI is a Python framework for building web APIs. Developers define endpoints using decorators such as @app.get and @app.post. Pydantic models validate incoming request data and define expected data structures. If a client sends invalid information, FastAPI can automatically return a validation error. FastAPI also generates interactive API documentation through Swagger UI. These features make it easier to develop, test, and maintain backend services.

Authentication protects applications by verifying user identities. JSON Web Tokens, commonly called JWTs, allow users to authenticate and access protected API endpoints. After successful login, a server can issue a signed token that the client includes in future requests. Backend applications verify the token before allowing access to protected resources. Passwords should be securely hashed rather than stored as plain text. Authorization adds another layer of protection by determining which resources an authenticated user is permitted to access.

Python applications can communicate with external services using HTTP client libraries such as requests. Developers send requests to external APIs, process JSON responses, and handle errors when services are unavailable. Environment variables help keep sensitive configuration values, such as API keys, outside source code. These techniques allow applications to integrate weather services, business data, AI models, and other external systems.

Streamlit is a Python framework for creating interactive web applications and dashboards. Developers can display charts, tables, forms, buttons, and other interface elements without writing extensive frontend JavaScript. Session state helps preserve information between interactions. Streamlit applications can connect to APIs, display database records, and visualize analytical results. This makes Streamlit useful for building prototypes, internal business tools, and AI-powered dashboards.

Embedding models convert text into numerical vectors that represent semantic information. Sentences discussing similar concepts tend to produce vectors with similar directions. Cosine similarity compares the direction of vectors to estimate semantic relatedness. A semantic search engine embeds documents, converts a user's question into an embedding, and retrieves the most similar documents. Unlike traditional keyword search, semantic search can identify related information even when the wording differs.

Chunking divides large documents into smaller pieces before embedding. Fixed-size chunking divides text according to character or token limits and can include overlapping content. Paragraph chunking preserves natural document boundaries, keeping related sentences together. Smaller chunks may provide focused retrieval but lose surrounding context, while larger chunks preserve more context but can mix unrelated topics. Vector databases such as ChromaDB store embeddings and support efficient similarity search. Retrieval-augmented generation uses retrieved information to help language models produce grounded answers.
"""

# ── Chunking functions ───────────────────────────────────────────────────────
def fixed_size_chunks(text: str, size: int = 300, overlap: int = 50) -> list[str]:
    """Split text into fixed-size chunks with overlap."""
    # TODO: implement the sliding-window approach described above
    chunks = []
    start = 0 
    while start < len(text):
        end = min(start + size, len(text))
        chunks.append(text[start:end].strip())
        start += size - overlap
    return chunks


def paragraph_chunks(text: str) -> list[str]:
    """Split text on double newlines, discarding empty paragraphs."""
    # TODO: split on "\n\n", strip each piece, filter empty strings
    paragraphs = text.strip().split("\n\n")
    return [p.strip() for p in paragraphs if p.strip()]


# ── Model & embeddings ───────────────────────────────────────────────────────
print("Loading model...")
# TODO: Load SentenceTransformer('all-MiniLM-L6-v2')

model = SentenceTransformer("all-MiniLM-L6-v2")

fixed_chunks = fixed_size_chunks(DOCUMENT)
para_chunks  = paragraph_chunks(DOCUMENT)

print(f"Fixed-size chunks : {len(fixed_chunks)}")
print(f"Paragraph chunks  : {len(para_chunks)}")

# TODO: Encode fixed_chunks  → fixed_embeddings
# TODO: Encode para_chunks   → para_embeddings

fixed_embeddings = model.encode(fixed_chunks)
para_embeddings = model.encode(para_chunks)


# ── Search helper ────────────────────────────────────────────────────────────
def search_chunks(query: str, chunks: list[str], embeddings, n: int = 2):
    """Return top-n (score, chunk) pairs for a query."""
    # TODO: encode query, compute similarities, return top-n sorted by score
    query_embedding = model.encode([query])

    scores = cosine_similarity(query_embedding, embeddings)[0]

    top_indices = np.argsort(scores)[::-1][:n]

    results = [
        (scores[i].item(), chunks[i])
        for i in top_indices
    ]
    return results


# ── Test queries ─────────────────────────────────────────────────────────────
# TODO: Define 3 queries that target different parts of your document

queries = [
    "How does FastAPI validate incoming request data?",
    "How do JWT tokens protect API endpoints?",
    "How does chunking improve semantic search?"
]

# ── Compare results ───────────────────────────────────────────────────────────
for query in queries:
    print(f'\n{"="*60}')
    print(f'Query: "{query}"')

    print("\n  [Fixed-size chunks]")
    # TODO: call search_chunks with fixed_chunks & fixed_embeddings, print results

    fixed_results = search_chunks(
    query, fixed_chunks, fixed_embeddings
)

    for score, chunk in fixed_results:
        print(f"\nScore: {score:.4f}")
        print(f"Chunk: {chunk}")


    print("\n  [Paragraph chunks]")
    # TODO: call search_chunks with para_chunks & para_embeddings, print results

    para_results = search_chunks(
    query, para_chunks, para_embeddings
)

    for score, chunk in para_results:
        print(f"\nScore: {score:.4f}")
        print(f"Chunk: {chunk}")
# ── Analysis ─────────────────────────────────────────────────────────────────
print("\n" + "="*60)
# TODO: Print your observations.
#       Which strategy returned more focused results?
#       Which missed context by cutting mid-sentence?
#       When would you choose each strategy?

print("""
ANALYSIS

1. Which strategy returned more focused results?
Paragraph chunking generally provided more complete and
focused explanations. It achieved higher top similarity
scores for two of the three queries. Fixed-size chunking
had the highest score for FastAPI validation.

2. Which strategy lost context?
Fixed-size chunking sometimes split sentences and
explanations across multiple chunks. For example, the
JWT authentication explanation was divided between
two separate results.

3. When would I choose each strategy?
I would use paragraph chunking for structured documents
with clear topic boundaries, such as course notes and
technical documentation.

I would use fixed-size chunking for unstructured text
or documents without reliable paragraph boundaries.
Overlap helps preserve context between adjacent chunks.

CONCLUSION:
Paragraph chunking worked better overall for this
structured course document, although fixed-size chunking
produced the highest similarity score for one query.
""")
