
from pathlib import Path
from threading import RLock

from langchain_community.vectorstores import FAISS
from langchain_core.documents import Document
from langchain_huggingface import HuggingFaceEmbeddings

from knowledge_agent.config import Settings
from knowledge_agent.ingestion import load_document, split_documents


class KnowledgeIndex:
    """Local FAISS index with persistent storage and duplicate-filtered retrieval."""

    def __init__(self, settings: Settings):
        self.settings = settings
        self.settings.data_dir.mkdir(parents=True, exist_ok=True)

        self.index_dir = self.settings.data_dir / "faiss_index"
        self.embeddings = HuggingFaceEmbeddings(
            model_name=self.settings.embedding_model
        )
        self.vectorstore: FAISS | None = None
        self.lock = RLock()

        index_file = self.index_dir / "index.faiss"

        if index_file.exists():
            # Only load indexes from this trusted, locally controlled directory.
            # FAISS metadata loading uses pickle deserialization.
            self.vectorstore = FAISS.load_local(
                str(self.index_dir),
                self.embeddings,
                allow_dangerous_deserialization=True,
            )

    def ingest_path(self, path: Path) -> int:
        """Load, split, and add a document to the persistent FAISS index."""
        docs = split_documents(
            load_document(path),
            self.settings.chunk_size,
            self.settings.chunk_overlap,
        )

        if not docs:
            raise ValueError("The document contains no extractable text.")

        # Preserve the source filename in every chunk for citations.
        for doc in docs:
            doc.metadata["source"] = path.name

        with self.lock:
            if self.vectorstore is None:
                self.vectorstore = FAISS.from_documents(
                    docs,
                    self.embeddings,
                )
            else:
                self.vectorstore.add_documents(docs)

            self.index_dir.mkdir(parents=True, exist_ok=True)
            self.vectorstore.save_local(str(self.index_dir))

        return len(docs)

    def search(self, query: str) -> list[Document]:
        """Retrieve relevant chunks and remove exact duplicate passages."""
        if not query or not query.strip():
            return []

        with self.lock:
            if self.vectorstore is None:
                return []

            candidate_count = max(self.settings.top_k * 3, self.settings.top_k)

            results = self.vectorstore.similarity_search(
                query.strip(),
                k=candidate_count,
            )

            unique_docs: list[Document] = []
            seen: set[tuple[str, str]] = set()

            for doc in results:
                source = str(doc.metadata.get("source", ""))
                content = " ".join(doc.page_content.split())
                key = (source.casefold(), content.casefold())

                if not content or key in seen:
                    continue

                seen.add(key)
                unique_docs.append(doc)

                if len(unique_docs) >= self.settings.top_k:
                    break

            return unique_docs
