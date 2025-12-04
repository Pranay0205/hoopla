# Hoopla - Advanced Movie Search System

A sophisticated Retrieval-Augmented Generation (RAG) system for movie search, combining keyword-based, semantic, and hybrid search techniques with LLM-powered query enhancement and result summarization.

## Overview

Hoopla is a movie streaming service search engine that implements multiple search algorithms including BM25 keyword search, semantic embeddings, chunked semantic search, and hybrid approaches. The system features query enhancement, LLM-based re-ranking, and RAG capabilities for generating comprehensive movie recommendations.

## Features

### Search Capabilities

- **Keyword Search**: BM25 algorithm with TF-IDF scoring
- **Semantic Search**: Transformer-based embeddings using sentence-transformers
- **Chunked Semantic Search**: Document chunking for improved semantic matching
- **Hybrid Search**: Combines keyword and semantic search using weighted scoring or Reciprocal Rank Fusion (RRF)

### Query Enhancement

- **Spell Correction**: Fixes common typos in search queries
- **Query Rewriting**: Transforms vague queries into specific, searchable terms
- **Query Expansion**: Adds synonyms and related concepts

### Re-ranking Methods

- **Individual Scoring**: LLM scores each result independently
- **Batch Scoring**: LLM scores multiple results in one prompt
- **Cross-Encoder**: Uses transformer models for relevance scoring

### RAG Features

- Answer generation based on search results
- Citation-supported responses
- Result summarization
- Relevance evaluation

## Requirements

- Python 3.12 or higher
- GEMINI_API_KEY environment variable (for LLM features)

## Installation

1. Clone the repository and navigate to the project directory

2. Install dependencies:

```bash
pip install -r requirements.txt
```

Or using the project configuration:

```bash
pip install nltk==3.9.1 numpy>=2.3.4 sentence-transformers>=5.1.1
```

3. Set up environment variables:
   Create a `.env` file in the project root:

```
GEMINI_API_KEY=your_api_key_here
```

4. Prepare data directory:
   Place `movies.json` in the `data/` directory with movie documents containing:

- `id`: Unique identifier
- `title`: Movie title
- `description`: Movie description

5. Build the search index:

```bash
python cli/keyword_search_cli.py build
```

## Usage

### Keyword Search (BM25)

Basic search:

```bash
python cli/keyword_search_cli.py bm25search "action movies"
```

Search with custom limit:

```bash
python cli/keyword_search_cli.py bm25search "thriller" --limit 10
```

Term frequency analysis:

```bash
python cli/keyword_search_cli.py tf <doc_id> <term>
python cli/keyword_search_cli.py idf <term>
python cli/keyword_search_cli.py tfidf <doc_id> <term>
```

BM25 scoring components:

```bash
python cli/keyword_search_cli.py bm25idf <term>
python cli/keyword_search_cli.py bm25tf <doc_id> <term> [k1] [b]
```

### Semantic Search

Verify model:

```bash
python cli/semantic_search_cli.py verify
```

Generate embeddings:

```bash
python cli/semantic_search_cli.py verify_embeddings
```

Search:

```bash
python cli/semantic_search_cli.py search "romantic comedy" --limit 5
```

Text chunking:

```bash
python cli/semantic_search_cli.py chunk "your text here" --chunk-size 200 --overlap 2
python cli/semantic_search_cli.py semantic_chunk "your text here" --max-chunk-size 4
```

Chunked search:

```bash
python cli/semantic_search_cli.py embed_chunks
python cli/semantic_search_cli.py search_chunked "adventure movies" --limit 5
```

### Hybrid Search

Normalize scores:

```bash
python cli/hybrid_search_cli.py normalize 0.5 0.8 0.3 0.9
```

Weighted search:

```bash
python cli/hybrid_search_cli.py weighted-search "sci-fi movies" --alpha 0.5 --limit 5
```

Alpha controls the weight between keyword (alpha) and semantic (1-alpha) search.

RRF search with query enhancement:

```bash
python cli/hybrid_search_cli.py rrf-search "scary bear movie" \
  --k 60 \
  --limit 5 \
  --enhance rewrite \
  --rerank-method cross_encoder
```

Options:

- `--k`: Controls weight given to higher-ranked results (default: 60)
- `--enhance`: Query enhancement method (spell, rewrite, expand)
- `--rerank-method`: Re-ranking method (individual, batch, cross_encoder)
- `--evaluate`: Evaluate relevance scores using LLM

### RAG (Retrieval-Augmented Generation)

Generate answer with search results:

```bash
python cli/augmented_generation_cli.py rag "What are some good horror movies?" \
  --k 60 \
  --limit 5 \
  --enhance rewrite \
  --rerank-method individual
```

Summarize search results:

```bash
python cli/augmented_generation_cli.py summarize "action movies with cars" --limit 5
```

Generate citations:

```bash
python cli/augmented_generation_cli.py citations "romantic comedies" --limit 5
```

### Evaluation

Evaluate search performance:

```bash
python cli/evaluation_cli.py --limit 5
```

Calculates precision, recall, and F1 scores against golden dataset.

## Configuration

Default constants in `cli/lib/utils/constants.py`:

```python
BM25_K1 = 1.5                    # BM25 term saturation parameter
BM25_B = 0.75                    # BM25 length normalization parameter
DEFAULT_SEARCH_LIMIT = 5         # Default number of results
DEFAULT_WEIGHT = 0.5             # Default alpha for weighted search
DEFAULT_K = 60                   # Default k for RRF
SEARCH_LIMIT_MULTIPLIER = 5      # Multiplier for initial retrieval
GEMINI_MODEL = "gemini-2.0-flash-lite"
RATE_TIME_SECONDS = 1            # Rate limiting for API calls
```

## Cache Management

The system caches embeddings and indexes in the `cache/` directory:

- `movie_embeddings.npy`: Semantic embeddings for movies
- `chunk_embeddings.npy`: Chunked semantic embeddings
- `chunk_metadata.json`: Metadata for chunks
- `index.pkl`: Inverted index
- `docmap.pkl`: Document mapping
- `term_frequencies.pkl`: Term frequency data
- `doc_length.pkl`: Document length data

To rebuild cache, delete files from `cache/` and re-run the appropriate build commands.

## Advanced Features

### Query Enhancement Methods

**Spell Correction**: Fixes typos while preserving correct spellings

```bash
--enhance spell
```

**Rewrite**: Transforms vague queries into specific, searchable terms

```bash
--enhance rewrite
```

Example: "scary bear movie" → "horror bear thriller grizzly attack film"

**Expand**: Adds related terms and synonyms

```bash
--enhance expand
```

### Re-ranking Methods

**Individual**: LLM scores each result independently (most accurate, slower)

```bash
--rerank-method individual
```

**Batch**: LLM scores multiple results in one call (faster, less accurate)

```bash
--rerank-method batch
```

**Cross-Encoder**: Uses transformer model for pairwise relevance scoring (balanced)

```bash
--rerank-method cross_encoder
```

## Performance Tips

1. Use weighted search (`--alpha`) for control over keyword vs semantic balance
2. Use RRF for best overall results combining both search types
3. Enable query enhancement (`--enhance rewrite`) for vague queries
4. Use cross_encoder re-ranking for best speed/accuracy trade-off
5. Adjust `--k` parameter in RRF to control rank fusion behavior
6. Cache is automatically managed but can be cleared for fresh builds

## Troubleshooting

**FileNotFoundError for index files**:
Run `python cli/keyword_search_cli.py build` to create the inverted index.

**Missing embeddings**:
Run semantic search commands to generate embeddings automatically.

**GEMINI_API_KEY error**:
Ensure `.env` file exists with valid API key for LLM features.

**Rate limiting**:
System automatically rate limits API calls to 15 RPM (configurable via RATE_TIME_SECONDS).

## Development

The project uses:

- **sentence-transformers**: all-MiniLM-L6-v2 model for embeddings
- **NLTK**: Text tokenization and stopword removal
- **NumPy**: Efficient array operations for embeddings
- **Google Gemini**: LLM for query enhancement and RAG

## License

This project is part of the Hoopla movie streaming service.

## Notes

- Default search limit is 5 results
- BM25 parameters (k1, b) can be tuned for different datasets
- Semantic search uses cosine similarity for ranking
- Hybrid search combines multiple approaches for improved accuracy
- All LLM operations respect rate limits to comply with API restrictions
