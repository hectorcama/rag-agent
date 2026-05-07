import argparse
import logging
import sys

from backend.app.settings import apply_runtime_env

# Critical: process env before unstructured / HF imports under ingestion path
apply_runtime_env()

from backend.app.services.document_processor import DocumentProcessor
from backend.app.services.embedding_service import EmbeddingService
from backend.app.services.ingestion_service import (
    IngestionService,
    format_chunk_preview,
)

# Preview length for --test-embedding chunk listing (PLR2004)
_TEST_EMBEDDING_PREVIEW_CHARS = 80


def run_ingest(file_path: str, data_dir: str = "data") -> None:
    """Ingest document into ChromaDB vector store."""
    service = IngestionService(data_dir=data_dir)
    doc_id = service.ingest_document(file_path)
    print(f"Ingested: {file_path} -> document_id={doc_id}")


def run_list_chunks(data_dir: str = "data", preview_width: int = 120) -> None:
    """Print every chunk in the store with a one-line preview and document info."""
    service = IngestionService(data_dir=data_dir)
    chunks = service.list_all_chunks()
    if not chunks:
        print("No chunks in the store.")
        return
    docs = {d.id: d for d in service.store.list_documents()}
    print(f"Total chunks: {len(chunks)} (documents in store: {len(docs)})")
    for i, chunk in enumerate(chunks, start=1):
        doc = docs.get(chunk.document_id)
        preview = format_chunk_preview(chunk.content, length=preview_width)
        print(f"\n--- Chunk {i}/{len(chunks)} ---")
        print(f"chunk_id: {chunk.id}")
        print(f"document_id: {chunk.document_id}")
        print(f"chunk_index: {chunk.chunk_index}")
        print(f"preview: {preview}")
        if doc is not None:
            print(
                f"document: {doc.file_name} ({doc.file_type}) | "
                f"path={doc.file_path} | chunk_count={doc.chunk_count}",
            )
        else:
            print("document: (metadata missing)")


def run_query(
    query: str,
    data_dir: str = "data",
    top_k: int = 1,
    rerank_candidates: int = 20,
) -> None:
    """Search the vector store and print top chunk(s) with document metadata."""
    logging.basicConfig(
        level=logging.INFO,
        format="%(levelname)s %(name)s: %(message)s",
    )
    service = IngestionService(data_dir=data_dir)
    results = service.search_similar_chunks(
        query,
        k=top_k,
        rerank_candidates=rerank_candidates,
    )
    if not results:
        print("No matching chunks found.")
        return
    for rank, (chunk, score) in enumerate(results, start=1):
        doc = service.store.get_document(chunk.document_id)
        print(f"\n--- Result {rank} (relevance score={score:.4f}) ---")
        print(f"chunk_id: {chunk.id}")
        print(f"document_id: {chunk.document_id}")
        print(f"chunk_index: {chunk.chunk_index}")
        print("content:")
        print(chunk.content)
        if doc is not None:
            print("\nDocument:")
            print(f"  id: {doc.id}")
            print(f"  file_name: {doc.file_name}")
            print(f"  file_path: {doc.file_path}")
            print(f"  file_type: {doc.file_type}")
            print(f"  chunk_count: {doc.chunk_count}")
        else:
            print("\nDocument: (metadata not found in store)")


def run_test_embedding_pipeline(file_path: str) -> None:
    """Test pipeline: process PDF -> show chunks -> embed -> show embeddings."""
    processor = DocumentProcessor(output_dir="data/output")
    print(f"Processing {file_path}...")

    chunked_elements = processor.process_document(
        file_path=file_path,
        languages=["eng", "fra"],
        chunking_strategy="by_title",
        max_characters=500,
        new_after_n_chars=450,
        combine_text_under_n_chars=500,
        multipage_sections=True,
        pdf_strategy=None,
    )

    print(f"\n--- Chunks ({len(chunked_elements)}) ---")
    chunk_texts = []
    for i, elem in enumerate(chunked_elements):
        text = str(elem).strip()
        chunk_texts.append(text)
        lim = _TEST_EMBEDDING_PREVIEW_CHARS
        preview = (text[:lim] + "…") if len(text) > lim else text
        print(f"[{i}] {preview!r}")

    if not chunk_texts:
        print("No chunks to embed.")
        return

    embedder = EmbeddingService()
    embeddings = embedder.embed_batch(chunk_texts)

    print(
        f"\n--- Embeddings ({len(embeddings)} vectors, dim={embedder.dimension}) ---",
    )
    for i, emb in enumerate(embeddings):
        print(f"[{i}] len={len(emb)}, first 5: {emb[:5]}")


def _build_argument_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=(
            "Convert documents (PDF, DOCX, MD, TXT) to Markdown with context-aware "
            "chunking; ingest into ChromaDB; or query the ingested corpus."
        )
    )
    parser.add_argument(
        "input",
        nargs="?",
        default=None,
        help="Path to the input document (not required with --query / --list-chunks)",
    )
    parser.add_argument(
        "--output-dir",
        default="data/output",
        help="Directory to save the markdown file",
    )
    parser.add_argument(
        "--lang",
        nargs="+",
        default=["eng", "fra"],
        help="Languages for processing (e.g., eng fra)",
    )
    parser.add_argument(
        "--chunking-strategy",
        default="by_title",
        choices=["by_title"],
        help="Chunking strategy to apply",
    )
    parser.add_argument(
        "--max-characters",
        type=int,
        default=500,
        help="Maximum chunk size (hard limit)",
    )
    parser.add_argument(
        "--new-after-n-chars",
        type=int,
        default=450,
        help="Soft maximum chunk size",
    )
    parser.add_argument(
        "--combine-text-under-n-chars",
        type=int,
        default=500,
        help="Combine small sections under this size",
    )
    parser.add_argument(
        "--multipage-sections",
        action="store_true",
        default=True,
        help="Respect page boundaries (default: True)",
    )
    parser.add_argument(
        "--no-multipage-sections",
        action="store_false",
        dest="multipage_sections",
        help="Do not respect page boundaries",
    )
    parser.add_argument(
        "--pdf-strategy",
        choices=["auto", "fast", "hi_res", "ocr_only"],
        default=None,
        help="Partitioning strategy for PDFs (default: auto)",
    )
    parser.add_argument(
        "--test-embedding",
        action="store_true",
        help="Run test: process doc -> show chunks -> embed -> show embeddings",
    )
    parser.add_argument(
        "--ingest",
        action="store_true",
        help="Ingest document into ChromaDB vector store",
    )
    parser.add_argument(
        "--data-dir",
        default="data",
        help="Directory for ChromaDB persistence (default: data)",
    )
    parser.add_argument(
        "--query",
        "-q",
        default=None,
        metavar="TEXT",
        help="Search the ingested corpus; prints top chunk(s) with document info",
    )
    parser.add_argument(
        "--top-k",
        type=int,
        default=1,
        help="How many top chunks to return for --query (default: 1)",
    )
    parser.add_argument(
        "--rerank-candidates",
        type=int,
        default=20,
        help="Candidate pool size for cross-encoder reranking (default: 20)",
    )
    parser.add_argument(
        "--list-chunks",
        action="store_true",
        help="Print every chunk in the store with previews (no document path needed)",
    )
    parser.add_argument(
        "--preview-width",
        type=int,
        default=120,
        help="Max characters per line for --list-chunks preview (default: 120)",
    )
    return parser


def _run_query_cli(
    query: str,
    args: argparse.Namespace,
    parser: argparse.ArgumentParser,
) -> None:
    if not query:
        print("Error: --query must be non-empty", file=sys.stderr)
        sys.exit(1)
    if args.top_k < 1:
        parser.error("--top-k must be >= 1")
    try:
        run_query(
            query,
            data_dir=args.data_dir,
            top_k=args.top_k,
            rerank_candidates=args.rerank_candidates,
        )
    except Exception as e:
        print(f"Error: {e}", file=sys.stderr)
        sys.exit(1)


def _run_list_chunks_cli(
    args: argparse.Namespace,
    parser: argparse.ArgumentParser,
) -> None:
    if args.preview_width < 1:
        parser.error("--preview-width must be >= 1")
    try:
        run_list_chunks(data_dir=args.data_dir, preview_width=args.preview_width)
    except Exception as e:
        print(f"Error: {e}", file=sys.stderr)
        sys.exit(1)


def _run_ingest_cli(args: argparse.Namespace) -> None:
    assert args.input is not None
    try:
        run_ingest(args.input, data_dir=args.data_dir)
    except Exception as e:
        print(f"Error: {e}", file=sys.stderr)
        sys.exit(1)


def _run_test_embedding_cli(args: argparse.Namespace) -> None:
    assert args.input is not None
    try:
        run_test_embedding_pipeline(args.input)
    except Exception as e:
        print(f"Error: {e}", file=sys.stderr)
        sys.exit(1)


def _run_default_process(args: argparse.Namespace) -> None:
    assert args.input is not None
    processor = DocumentProcessor(output_dir=args.output_dir)
    try:
        print(f"Processing {args.input}...")
        chunked_elements = processor.process_document(
            file_path=args.input,
            languages=args.lang,
            chunking_strategy=args.chunking_strategy,
            max_characters=args.max_characters,
            new_after_n_chars=args.new_after_n_chars,
            combine_text_under_n_chars=args.combine_text_under_n_chars,
            multipage_sections=args.multipage_sections,
            pdf_strategy=args.pdf_strategy,
        )
        print(f"Successfully processed document into {len(chunked_elements)} chunks")
        output_path = processor.save_as_markdown(
            args.input,
            languages=args.lang,
            pdf_strategy=args.pdf_strategy,
        )
        print(f"Markdown saved to: {output_path}")
    except Exception as e:
        print(f"Error: {e}", file=sys.stderr)
        sys.exit(1)


def main() -> None:
    """Parse CLI arguments and run ingest, query, chunk test, or document processing."""
    parser = _build_argument_parser()
    args = parser.parse_args()

    if args.query is not None:
        _run_query_cli(args.query.strip(), args, parser)
        return
    if args.list_chunks:
        _run_list_chunks_cli(args, parser)
        return
    if args.input is None:
        parser.error(
            "input document path is required unless using --query or --list-chunks",
        )
    if args.ingest:
        _run_ingest_cli(args)
        return
    if args.test_embedding:
        _run_test_embedding_cli(args)
        return
    _run_default_process(args)


if __name__ == "__main__":
    main()
