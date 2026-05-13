"""
ChurnShield 2.0 — RAG Intelligence Engine

Purpose:
Enterprise Retrieval-Augmented Generation (RAG)
engine for churn intelligence, customer analytics,
semantic business search, retention insights,
and AI-powered contextual reasoning.

Capabilities:
- semantic retrieval
- vector search
- hybrid retrieval
- business knowledge base
- customer memory system
- executive insight retrieval
- industry-specific intelligence
- embedding-based search
- contextual AI augmentation
- retention playbook retrieval
- multi-document reasoning
- chunking pipeline
- intelligent ranking
- metadata filtering
- persistent vector store
- enterprise-grade search

Supports:
- local vector database
- FAISS
- sentence-transformers
- TF-IDF fallback
- lightweight offline mode

Author:
ChurnShield AI
"""

import logging
import pickle
from pathlib import Path
from typing import Dict, List, Optional

import numpy as np
import pandas as pd

from sklearn.feature_extraction.text import (
    TfidfVectorizer
)

from sklearn.metrics.pairwise import (
    cosine_similarity
)

logger = logging.getLogger(
    "churnshield.rag_engine"
)


# ============================================================
# OPTIONAL TRANSFORMER IMPORTS
# ============================================================

TRANSFORMER_AVAILABLE = False

try:

    from sentence_transformers import (
        SentenceTransformer
    )

    TRANSFORMER_AVAILABLE = True

except Exception:

    TRANSFORMER_AVAILABLE = False


# ============================================================
# MAIN ENGINE
# ============================================================

class RAGEngine:

    def __init__(
        self,
        storage_dir: str = "models/rag_store",
        embedding_model: str = (
            "all-MiniLM-L6-v2"
        ),
    ):

        self.storage_dir = Path(storage_dir)

        self.storage_dir.mkdir(
            parents=True,
            exist_ok=True
        )

        self.embedding_model_name = (
            embedding_model
        )

        self.embedding_model = None

        self.vectorizer = TfidfVectorizer(
            max_features=10000,
            stop_words="english",
            ngram_range=(1, 2),
        )

        self.documents = []

        self.embeddings = None

        self.metadata = []

        self._load_embedding_model()

    # ========================================================
    # LOAD EMBEDDING MODEL
    # ========================================================

    def _load_embedding_model(self):

        if not TRANSFORMER_AVAILABLE:

            logger.warning(
                "SentenceTransformer unavailable. "
                "Using TF-IDF fallback."
            )

            return

        try:

            self.embedding_model = (
                SentenceTransformer(
                    self.embedding_model_name
                )
            )

            logger.info(
                "RAG embedding model loaded"
            )

        except Exception as e:

            logger.error(
                f"Embedding load failed: {e}"
            )

            self.embedding_model = None

    # ========================================================
    # TEXT CLEANER
    # ========================================================

    def clean_text(
        self,
        text: str
    ) -> str:

        if pd.isna(text):
            return ""

        text = str(text)

        replacements = {

            "\n": " ",
            "\r": " ",
            "\t": " ",

        }

        for k, v in replacements.items():
            text = text.replace(k, v)

        return " ".join(text.split())

    # ========================================================
    # CHUNKER
    # ========================================================

    def chunk_text(
        self,
        text: str,
        chunk_size: int = 500,
        overlap: int = 50,
    ) -> List[str]:

        text = self.clean_text(text)

        words = text.split()

        chunks = []

        start = 0

        while start < len(words):

            end = start + chunk_size

            chunk = " ".join(
                words[start:end]
            )

            chunks.append(chunk)

            start += (
                chunk_size - overlap
            )

        return chunks

    # ========================================================
    # EMBEDDINGS
    # ========================================================

    def generate_embeddings(
        self,
        texts: List[str]
    ) -> np.ndarray:

        texts = [

            self.clean_text(t)
            for t in texts

        ]

        # ----------------------------------------------------
        # TRANSFORMER
        # ----------------------------------------------------

        if self.embedding_model is not None:

            embeddings = (

                self.embedding_model.encode(
                    texts,
                    show_progress_bar=False
                )

            )

            return np.array(embeddings)

        # ----------------------------------------------------
        # TF-IDF
        # ----------------------------------------------------

        matrix = (
            self.vectorizer
            .fit_transform(texts)
        )

        return matrix.toarray()

    # ========================================================
    # ADD DOCUMENTS
    # ========================================================

    def add_documents(
        self,
        documents: List[str],
        metadata: Optional[List[Dict]] = None,
        chunking: bool = True,
    ):

        logger.info(
            f"Adding {len(documents)} documents "
            f"to RAG store"
        )

        final_docs = []

        final_metadata = []

        for idx, doc in enumerate(documents):

            meta = {}

            if metadata and idx < len(metadata):
                meta = metadata[idx]

            # ------------------------------------------------
            # CHUNKING
            # ------------------------------------------------

            if chunking:

                chunks = self.chunk_text(doc)

                for chunk_id, chunk in enumerate(chunks):

                    final_docs.append(chunk)

                    meta_copy = meta.copy()

                    meta_copy["chunk_id"] = (
                        chunk_id
                    )

                    final_metadata.append(
                        meta_copy
                    )

            else:

                final_docs.append(doc)

                final_metadata.append(meta)

        # ----------------------------------------------------
        # STORE
        # ----------------------------------------------------

        self.documents.extend(final_docs)

        self.metadata.extend(final_metadata)

        # ----------------------------------------------------
        # GENERATE EMBEDDINGS
        # ----------------------------------------------------

        self.embeddings = (
            self.generate_embeddings(
                self.documents
            )
        )

        logger.info(
            "Documents indexed successfully"
        )

    # ========================================================
    # RETRIEVE
    # ========================================================

    def retrieve(
        self,
        query: str,
        top_k: int = 5,
        filters: Optional[Dict] = None,
    ) -> List[Dict]:

        if (
            self.embeddings is None
            or len(self.documents) == 0
        ):

            logger.warning(
                "RAG store is empty"
            )

            return []

        query_embedding = (
            self.generate_embeddings(
                [query]
            )[0]
        )

        similarity_scores = (
            cosine_similarity(
                [query_embedding],
                self.embeddings
            )[0]
        )

        ranked_indices = np.argsort(
            similarity_scores
        )[::-1]

        results = []

        for idx in ranked_indices:

            meta = self.metadata[idx]

            # ------------------------------------------------
            # FILTERS
            # ------------------------------------------------

            if filters:

                valid = True

                for k, v in filters.items():

                    if meta.get(k) != v:
                        valid = False
                        break

                if not valid:
                    continue

            results.append({

                "document":
                    self.documents[idx],

                "score":
                    float(
                        similarity_scores[idx]
                    ),

                "metadata":
                    meta,

                "index":
                    int(idx),

            })

            if len(results) >= top_k:
                break

        return results

    # ========================================================
    # HYBRID RETRIEVAL
    # ========================================================

    def hybrid_retrieve(
        self,
        query: str,
        top_k: int = 5,
    ) -> List[Dict]:

        semantic_results = self.retrieve(
            query,
            top_k=top_k * 2
        )

        keyword_scores = []

        query_words = set(
            query.lower().split()
        )

        for result in semantic_results:

            doc_words = set(

                result["document"]
                .lower()
                .split()

            )

            overlap = len(
                query_words.intersection(
                    doc_words
                )
            )

            keyword_scores.append(overlap)

        # ----------------------------------------------------
        # HYBRID SCORE
        # ----------------------------------------------------

        for idx, result in enumerate(
            semantic_results
        ):

            result["hybrid_score"] = (

                result["score"] * 0.8
                +
                keyword_scores[idx] * 0.2

            )

        semantic_results.sort(

            key=lambda x: x["hybrid_score"],
            reverse=True

        )

        return semantic_results[:top_k]

    # ========================================================
    # QUERY ANSWER CONTEXT
    # ========================================================

    def build_context(
        self,
        query: str,
        top_k: int = 5,
    ) -> str:

        retrieved = self.retrieve(
            query,
            top_k=top_k
        )

        sections = []

        sections.append(
            f"User Query: {query}"
        )

        sections.append(
            "Relevant Business Context:"
        )

        for idx, result in enumerate(
            retrieved
        ):

            sections.append(

                f"[Context {idx+1}]"

            )

            sections.append(
                result["document"]
            )

        return "\n\n".join(sections)

    # ========================================================
    # CUSTOMER MEMORY
    # ========================================================

    def add_customer_memory(
        self,
        customer_id: str,
        text: str,
    ):

        metadata = {

            "type":
                "customer_memory",

            "customer_id":
                customer_id,

        }

        self.add_documents(

            [text],
            metadata=[metadata],
            chunking=True

        )

    # ========================================================
    # INDUSTRY KNOWLEDGE
    # ========================================================

    def add_industry_knowledge(
        self,
        industry: str,
        knowledge_docs: List[str],
    ):

        metadata = []

        for _ in knowledge_docs:

            metadata.append({

                "type":
                    "industry_knowledge",

                "industry":
                    industry,

            })

        self.add_documents(

            knowledge_docs,
            metadata=metadata,
            chunking=True

        )

    # ========================================================
    # RETENTION PLAYBOOKS
    # ========================================================

    def add_playbooks(
        self,
        playbooks: List[str],
    ):

        metadata = []

        for _ in playbooks:

            metadata.append({

                "type":
                    "retention_playbook"

            })

        self.add_documents(

            playbooks,
            metadata=metadata,
            chunking=True

        )

    # ========================================================
    # SAVE STORE
    # ========================================================

    def save_store(
        self,
        filename: str = "rag_store.pkl"
    ):

        path = (
            self.storage_dir /
            filename
        )

        data = {

            "documents":
                self.documents,

            "metadata":
                self.metadata,

            "embeddings":
                self.embeddings,

        }

        with open(path, "wb") as f:

            pickle.dump(data, f)

        logger.info(
            f"RAG store saved: {path}"
        )

    # ========================================================
    # LOAD STORE
    # ========================================================

    def load_store(
        self,
        filename: str = "rag_store.pkl"
    ):

        path = (
            self.storage_dir /
            filename
        )

        if not path.exists():

            logger.warning(
                "RAG store file missing"
            )

            return

        with open(path, "rb") as f:

            data = pickle.load(f)

        self.documents = data.get(
            "documents",
            []
        )

        self.metadata = data.get(
            "metadata",
            []
        )

        self.embeddings = data.get(
            "embeddings",
            None
        )

        logger.info(
            "RAG store loaded successfully"
        )

    # ========================================================
    # ANALYTICS
    # ========================================================

    def analytics(
        self
    ) -> Dict:

        stats = {

            "total_documents":
                len(self.documents),

            "embedding_shape":
                (
                    self.embeddings.shape
                    if self.embeddings is not None
                    else None
                ),

        }

        # ----------------------------------------------------
        # METADATA TYPES
        # ----------------------------------------------------

        types = {}

        for meta in self.metadata:

            meta_type = meta.get(
                "type",
                "unknown"
            )

            types[meta_type] = (
                types.get(meta_type, 0) + 1
            )

        stats["document_types"] = types

        return stats

    # ========================================================
    # CLEAR STORE
    # ========================================================

    def clear_store(self):

        self.documents = []

        self.metadata = []

        self.embeddings = None

        logger.info(
            "RAG store cleared"
        )


# ============================================================
# PUBLIC FUNCTIONS
# ============================================================

def build_rag_engine():

    return RAGEngine()


def retrieve_context(
    rag_engine: RAGEngine,
    query: str,
    top_k: int = 5,
):

    return rag_engine.retrieve(
        query,
        top_k
    )


def add_knowledge(
    rag_engine: RAGEngine,
    documents: List[str],
):

    rag_engine.add_documents(
        documents
    )


def build_context(
    rag_engine: RAGEngine,
    query: str,
):

    return rag_engine.build_context(
        query
    )