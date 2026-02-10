import os
import shutil
import warnings
from pathlib import Path
from typing import Literal

from dotenv import load_dotenv

# Load .env file before accessing environment variables
env_path = Path(__file__).parent.parent.parent.parent.parent / ".env"
load_dotenv(dotenv_path=env_path)

# Set telemetry defaults if not already loaded from .env
# Critical: These must be set before importing unstructured
os.environ.setdefault("SCARF_NO_ANALYTICS", "true")
os.environ.setdefault("DO_NOT_TRACK", "true")

from unstructured.chunking.title import chunk_by_title
from unstructured.documents.elements import Element
from unstructured.partition.docx import partition_docx
from unstructured.partition.md import partition_md
from unstructured.partition.odt import partition_odt
from unstructured.partition.pdf import partition_pdf
from unstructured.partition.text import partition_text


class DocumentProcessor:
    """Handles document ingestion and processing using unstructured.

    Supports PDF, Word, Markdown, and Text documents with French and English language support.
    Implements context-aware partitioning and chunking strategies.
    """

    # Supported file extensions
    SUPPORTED_EXTENSIONS = {
        ".md": "markdown",
        ".markdown": "markdown",
        ".pdf": "pdf",
        ".docx": "docx",
        ".odt": "odt",
        ".txt": "text",
        ".text": "text",
    }

    def __init__(self, output_dir: str = "data/output"):
        """Initialize the document processor.

        Args:
            output_dir: Directory to save processed markdown files.

        """
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)

    def _detect_file_type(
        self, file_path: str
    ) -> Literal["markdown", "pdf", "docx", "odt", "text"]:
        """Detect file type from file extension.

        Args:
            file_path: Path to the file.

        Returns:
            File type string.

        Raises:
            ValueError: If file type is not supported.

        """
        file_path_obj = Path(file_path)
        extension = file_path_obj.suffix.lower()

        if extension not in self.SUPPORTED_EXTENSIONS:
            supported = ", ".join(self.SUPPORTED_EXTENSIONS.keys())
            raise ValueError(
                f"Unsupported file type: {extension}. Supported types: {supported}"
            )

        file_type = self.SUPPORTED_EXTENSIONS[extension]
        return file_type  # type: ignore[return-value]

    def _partition_document(
        self,
        file_path: str,
        file_type: str,
        languages: list[str] = ["eng", "fra"],
        strategy: str | None = None,
    ) -> list[Element]:
        """Partition a document into structured elements.

        Preserves Table and Image elements with full metadata.

        Args:
            file_path: Path to the document.
            file_type: Type of document (markdown, pdf, docx, text).
            languages: List of Tesseract language codes for OCR.
            strategy: Partitioning strategy (for PDFs: "auto", "fast", "hi_res", "ocr_only").

        Returns:
            List of document elements.

        """
        file_path_str = str(file_path)

        if file_type == "markdown":
            return partition_md(filename=file_path_str)
        elif file_type == "pdf":
            kwargs: dict[str, str | list[str]] = {}

            # Check if Tesseract is available in PATH
            tesseract_available = shutil.which("tesseract") is not None

            # Determine if OCR is needed based on strategy
            ocr_strategies = {"ocr_only", "hi_res"}
            needs_ocr = strategy in ocr_strategies

            # Warn if OCR strategy is requested but Tesseract is not available
            if needs_ocr and not tesseract_available:
                warnings.warn(
                    "Tesseract not found in PATH. OCR strategies (hi_res, ocr_only) require Tesseract. "
                    "Falling back to 'fast' strategy. Install Tesseract and add it to PATH: "
                    "https://github.com/UB-Mannheim/tesseract/wiki",
                    UserWarning,
                    stacklevel=2,
                )
                strategy = "fast"

            # Only pass languages if Tesseract is available
            if tesseract_available:
                kwargs["languages"] = languages

            # Set strategy (default to "fast" if not specified and Tesseract unavailable)
            if strategy:
                kwargs["strategy"] = strategy
            elif not tesseract_available:
                kwargs["strategy"] = "fast"

            return partition_pdf(filename=file_path_str, **kwargs)
        elif file_type == "docx":
            return partition_docx(filename=file_path_str)
        elif file_type == "odt":
            return partition_odt(filename=file_path_str)
        elif file_type == "text":
            return partition_text(filename=file_path_str)
        else:
            raise ValueError(f"Unsupported file type for partitioning: {file_type}")

    def _elements_to_markdown(self, elements: list[Element]) -> str:
        """Convert partitioned elements to markdown string.

        Preserves table HTML via element.metadata.text_as_html and handles images.

        Args:
            elements: List of document elements.

        Returns:
            Markdown string representation of the elements.

        """
        markdown_content = ""

        for element in elements:
            element_type = getattr(element, "category", "UncategorizedText")
            text = str(element)

            if element_type == "Title":
                markdown_content += f"# {text}\n\n"
            elif element_type == "Header":
                markdown_content += f"## {text}\n\n"
            elif element_type == "Subheader":
                markdown_content += f"### {text}\n\n"
            elif element_type == "ListItem":
                markdown_content += f"- {text}\n"
            elif element_type == "Table":
                # Preserve table HTML if available
                if hasattr(element, "metadata") and hasattr(
                    element.metadata, "text_as_html"
                ):
                    html_table = element.metadata.text_as_html
                    if html_table:
                        markdown_content += f"\n{html_table}\n\n"
                    else:
                        markdown_content += f"{text}\n\n"
                else:
                    markdown_content += f"{text}\n\n"
            elif element_type == "Image":
                # Preserve image metadata and reference
                image_ref = f"[Image: {text}]" if text else "[Image]"
                markdown_content += f"{image_ref}\n\n"
            else:
                markdown_content += f"{text}\n\n"

        return markdown_content

    def _partition_markdown(self, markdown_content: str) -> list[Element]:
        """Partition markdown string into elements.

        Args:
            markdown_content: Markdown string to partition.

        Returns:
            List of markdown elements.

        """
        # Create a temporary file-like object or use text parameter
        # partition_md can accept text directly
        return partition_md(text=markdown_content)

    def _apply_context_strategy(
        self,
        elements: list[Element],
        chunking_strategy: str = "by_title",
        max_characters: int = 500,
        new_after_n_chars: int = 450,
        combine_text_under_n_chars: int = 500,
        multipage_sections: bool = True,
    ) -> list[Element]:
        """Apply context-aware chunking strategy to elements.

        Args:
            elements: List of elements to chunk.
            chunking_strategy: Chunking strategy ("by_title" supported).
            max_characters: Maximum chunk size (hard limit).
            new_after_n_chars: Soft maximum chunk size.
            combine_text_under_n_chars: Combine small sections under this size.
            multipage_sections: Whether to respect page boundaries.

        Returns:
            List of chunked elements.

        """
        if chunking_strategy == "by_title":
            return chunk_by_title(
                elements=elements,
                max_characters=max_characters,
                new_after_n_chars=new_after_n_chars,
                combine_text_under_n_chars=combine_text_under_n_chars,
                multipage_sections=multipage_sections,
            )
        else:
            raise ValueError(f"Unsupported chunking strategy: {chunking_strategy}")

    def process_document(
        self,
        file_path: str,
        languages: list[str] = ["eng", "fra"],
        chunking_strategy: str = "by_title",
        max_characters: int = 500,
        new_after_n_chars: int = 450,
        combine_text_under_n_chars: int = 500,
        multipage_sections: bool = True,
        pdf_strategy: str | None = None,
    ) -> list[Element]:
        """Process a document through the full pipeline: partition → convert → partition → chunk.

        Args:
            file_path: Path to the document (PDF, DOCX, MD, TXT).
            languages: List of Tesseract language codes for OCR.
            chunking_strategy: Chunking strategy to apply ("by_title").
            max_characters: Maximum chunk size.
            new_after_n_chars: Soft maximum chunk size.
            combine_text_under_n_chars: Combine small sections under this size.
            multipage_sections: Whether to respect page boundaries.
            pdf_strategy: Partitioning strategy for PDFs ("auto", "fast", "hi_res", "ocr_only").

        Returns:
            List of chunked elements ready for embedding/indexing.

        Raises:
            FileNotFoundError: If file does not exist.
            ValueError: If file type is not supported.

        """
        file_path_obj = Path(file_path)
        if not file_path_obj.exists():
            raise FileNotFoundError(f"File not found: {file_path}")

        # Step 1: Detect file type
        file_type = self._detect_file_type(file_path)

        # Step 2: Partition original format (preserves Table/Image elements)
        elements = self._partition_document(
            file_path=str(file_path),
            file_type=file_type,
            languages=languages,
            strategy=pdf_strategy,
        )

        # Step 3: Convert elements to markdown (preserving table HTML)
        markdown_content = self._elements_to_markdown(elements)

        # Step 4: Partition markdown string into elements
        markdown_elements = self._partition_markdown(markdown_content)

        # Step 5: Apply context strategy (chunking)
        chunked_elements = self._apply_context_strategy(
            elements=markdown_elements,
            chunking_strategy=chunking_strategy,
            max_characters=max_characters,
            new_after_n_chars=new_after_n_chars,
            combine_text_under_n_chars=combine_text_under_n_chars,
            multipage_sections=multipage_sections,
        )

        return chunked_elements
