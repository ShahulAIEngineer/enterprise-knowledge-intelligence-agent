from collections.abc import Iterable
from pathlib import Path

from langchain_core.documents import Document
from langchain_text_splitters import RecursiveCharacterTextSplitter
from pypdf import PdfReader

SUPPORTED_SUFFIXES = {".txt", ".md", ".pdf"}


def load_document(path: Path) -> list[Document]:
    suffix = path.suffix.lower()
    if suffix not in SUPPORTED_SUFFIXES:
        raise ValueError(f"Unsupported file type '{suffix}'. Supported: .txt, .md, .pdf")
    if suffix == ".pdf":
        reader = PdfReader(str(path))
        return [
            Document(page_content=page.extract_text() or "", metadata={"source": path.name, "page": i + 1})
            for i, page in enumerate(reader.pages)
            if (page.extract_text() or "").strip()
        ]
    text = path.read_text(encoding="utf-8")
    return [Document(page_content=text, metadata={"source": path.name})] if text.strip() else []


def split_documents(documents: Iterable[Document], chunk_size: int = 800, chunk_overlap: int = 120):
    if chunk_overlap >= chunk_size:
        raise ValueError("chunk_overlap must be smaller than chunk_size")
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=chunk_size, chunk_overlap=chunk_overlap, add_start_index=True
    )
    return splitter.split_documents(list(documents))
