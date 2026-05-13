"""
ChurnShield 2.0 — Embedding Intelligence Engine

Purpose:
Enterprise embedding engine for semantic intelligence,
RAG systems, churn understanding, customer clustering,
similarity search, recommendation systems,
and AI-powered analytics.

Capabilities:
- sentence embeddings
- customer embeddings
- semantic similarity
- vector search
- clustering
- churn behavior embeddings
- text feature extraction
- TF-IDF embeddings
- transformer embeddings
- hybrid embeddings
- embedding cache
- cosine similarity engine
- nearest neighbor retrieval
- semantic search
- anomaly semantic detection
- retention intelligence vectors

Supports:
- local embeddings
- HuggingFace transformers
- sentence-transformers
- lightweight TF-IDF fallback
- scalable batch processing

Author:
ChurnShield AI
"""

import logging
import hashlib
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

from sklearn.cluster import (
    KMeans
)

from sklearn.decomposition import (
    TruncatedSVD
)

logger = logging.getLogger(
    "churnshield.embedding_engine"
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

class EmbeddingEngine:

    def __init__(
        self,
        model_name: str = (
            "all-MiniLM-L6-v2"
        ),
        cache_dir: str = "models/embeddings",
    ):

        self.model_name = model_name

        self.cache_dir = Path(cache_dir)

        self.cache_dir.mkdir(
            parents=True,
            exist_ok=True
        )

        self.embedding_model = None

        self.vectorizer = TfidfVectorizer(
            max_features=5000,
            stop_words="english",
            ngram_range=(1, 2),
        )

        self._load_transformer_model()

    # ========================================================
    # LOAD MODEL
    # ========================================================

    def _load_transformer_model(self):

        if not TRANSFORMER_AVAILABLE:

            logger.warning(
                "SentenceTransformer not installed. "
                "Using TF-IDF fallback embeddings."
            )

            return

        try:

            self.embedding_model = (
                SentenceTransformer(
                    self.model_name
                )
            )

            logger.info(
                "Transformer embedding model loaded"
            )

        except Exception as e:

            logger.error(
                f"Embedding model load failed: {e}"
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

        text = str(text).lower()

        replacements = {

            "\n": " ",
            "\r": " ",
            "\t": " ",

        }

        for k, v in replacements.items():
            text = text.replace(k, v)

        return " ".join(text.split())

    # ========================================================
    # TEXT HASH
    # ========================================================

    def text_hash(
        self,
        text: str
    ) -> str:

        return hashlib.md5(
            text.encode()
        ).hexdigest()

    # ========================================================
    # SAVE CACHE
    # ========================================================

    def save_cache(
        self,
        key: str,
        embedding
    ):

        path = (
            self.cache_dir /
            f"{key}.pkl"
        )

        try:

            with open(path, "wb") as f:

                pickle.dump(
                    embedding,
                    f
                )

        except Exception as e:

            logger.error(
                f"Embedding cache save failed: {e}"
            )

    # ========================================================
    # LOAD CACHE
    # ========================================================

    def load_cache(
        self,
        key: str
    ):

        path = (
            self.cache_dir /
            f"{key}.pkl"
        )

        if not path.exists():
            return None

        try:

            with open(path, "rb") as f:

                return pickle.load(f)

        except Exception:

            return None

    # ========================================================
    # SINGLE EMBEDDING
    # ========================================================

    def generate_embedding(
        self,
        text: str,
        use_cache: bool = True,
    ):

        text = self.clean_text(text)

        key = self.text_hash(text)

        # ----------------------------------------------------
        # CACHE
        # ----------------------------------------------------

        if use_cache:

            cached = self.load_cache(key)

            if cached is not None:
                return cached

        # ----------------------------------------------------
        # TRANSFORMER
        # ----------------------------------------------------

        if self.embedding_model is not None:

            embedding = (
                self.embedding_model.encode(
                    text
                )
            )

        # ----------------------------------------------------
        # TF-IDF FALLBACK
        # ----------------------------------------------------

        else:

            embedding = (
                self.vectorizer
                .fit_transform([text])
                .toarray()[0]
            )

        # ----------------------------------------------------
        # SAVE
        # ----------------------------------------------------

        if use_cache:

            self.save_cache(
                key,
                embedding
            )

        return embedding

    # ========================================================
    # BATCH EMBEDDINGS
    # ========================================================

    def generate_embeddings(
        self,
        texts: List[str],
    ) -> np.ndarray:

        cleaned = [

            self.clean_text(t)
            for t in texts

        ]

        # ----------------------------------------------------
        # TRANSFORMER
        # ----------------------------------------------------

        if self.embedding_model is not None:

            embeddings = (
                self.embedding_model.encode(
                    cleaned,
                    show_progress_bar=False,
                )
            )

            return np.array(embeddings)

        # ----------------------------------------------------
        # TF-IDF
        # ----------------------------------------------------

        matrix = (
            self.vectorizer
            .fit_transform(cleaned)
        )

        return matrix.toarray()

    # ========================================================
    # DATAFRAME EMBEDDINGS
    # ========================================================

    def dataframe_embeddings(
        self,
        df: pd.DataFrame,
        text_columns: List[str],
    ) -> np.ndarray:

        logger.info(
            "Generating dataframe embeddings"
        )

        combined = []

        for _, row in df.iterrows():

            text_parts = []

            for col in text_columns:

                if col in row:

                    text_parts.append(
                        str(row[col])
                    )

            combined.append(
                " ".join(text_parts)
            )

        embeddings = (
            self.generate_embeddings(
                combined
            )
        )

        return embeddings

    # ========================================================
    # CUSTOMER EMBEDDINGS
    # ========================================================

    def customer_embeddings(
        self,
        df: pd.DataFrame
    ) -> np.ndarray:

        candidate_columns = [

            "feedback",
            "review",
            "complaint",
            "customer_notes",
            "industry",
            "plan_name",
            "persona",
            "reason",
            "sentiment_label",

        ]

        usable = [

            c for c in candidate_columns
            if c in df.columns

        ]

        if not usable:

            logger.warning(
                "No text columns found for "
                "customer embeddings"
            )

            return np.zeros((len(df), 32))

        return self.dataframe_embeddings(
            df,
            usable
        )

    # ========================================================
    # SIMILARITY MATRIX
    # ========================================================

    def similarity_matrix(
        self,
        embeddings: np.ndarray
    ) -> np.ndarray:

        return cosine_similarity(
            embeddings
        )

    # ========================================================
    # FIND SIMILAR CUSTOMERS
    # ========================================================

    def find_similar_customers(
        self,
        embeddings: np.ndarray,
        customer_index: int,
        top_k: int = 5,
    ) -> List[int]:

        similarity = cosine_similarity(
            [embeddings[customer_index]],
            embeddings
        )[0]

        similar_indices = np.argsort(
            similarity
        )[::-1]

        similar_indices = [

            idx
            for idx in similar_indices
            if idx != customer_index
        ]

        return similar_indices[:top_k]

    # ========================================================
    # SEMANTIC SEARCH
    # ========================================================

    def semantic_search(
        self,
        query: str,
        corpus_embeddings: np.ndarray,
        corpus_texts: List[str],
        top_k: int = 5,
    ) -> List[Dict]:

        query_embedding = (
            self.generate_embedding(query)
        )

        similarity = cosine_similarity(
            [query_embedding],
            corpus_embeddings
        )[0]

        ranked = np.argsort(
            similarity
        )[::-1][:top_k]

        results = []

        for idx in ranked:

            results.append({

                "text":
                    corpus_texts[idx],

                "similarity":
                    float(similarity[idx]),

                "index":
                    int(idx),

            })

        return results

    # ========================================================
    # CLUSTER EMBEDDINGS
    # ========================================================

    def cluster_embeddings(
        self,
        embeddings: np.ndarray,
        n_clusters: int = 5,
    ) -> np.ndarray:

        logger.info(
            "Running embedding clustering"
        )

        model = KMeans(
            n_clusters=n_clusters,
            random_state=42,
            n_init=10,
        )

        clusters = model.fit_predict(
            embeddings
        )

        return clusters

    # ========================================================
    # DIMENSIONALITY REDUCTION
    # ========================================================

    def reduce_dimensions(
        self,
        embeddings: np.ndarray,
        n_components: int = 2,
    ) -> np.ndarray:

        reducer = TruncatedSVD(
            n_components=n_components,
            random_state=42,
        )

        reduced = reducer.fit_transform(
            embeddings
        )

        return reduced

    # ========================================================
    # HYBRID FEATURES
    # ========================================================

    def hybrid_customer_vectors(
        self,
        df: pd.DataFrame,
        text_columns: List[str],
        numeric_columns: List[str],
    ) -> np.ndarray:

        # ----------------------------------------------------
        # TEXT EMBEDDINGS
        # ----------------------------------------------------

        text_embeddings = (
            self.dataframe_embeddings(
                df,
                text_columns
            )
        )

        # ----------------------------------------------------
        # NUMERIC FEATURES
        # ----------------------------------------------------

        numeric = df[
            numeric_columns
        ].fillna(0)

        numeric_array = (
            numeric.to_numpy()
        )

        # ----------------------------------------------------
        # CONCAT
        # ----------------------------------------------------

        hybrid = np.concatenate(

            [
                text_embeddings,
                numeric_array
            ],

            axis=1
        )

        return hybrid

    # ========================================================
    # ANOMALY DETECTION
    # ========================================================

    def semantic_anomaly_scores(
        self,
        embeddings: np.ndarray
    ) -> np.ndarray:

        centroid = np.mean(
            embeddings,
            axis=0
        )

        distances = np.linalg.norm(
            embeddings - centroid,
            axis=1
        )

        normalized = (

            distances /
            max(distances.max(), 1)

        )

        return normalized

    # ========================================================
    # SAVE EMBEDDINGS
    # ========================================================

    def save_embeddings(
        self,
        embeddings,
        path: str
    ):

        with open(path, "wb") as f:

            pickle.dump(
                embeddings,
                f
            )

        logger.info(
            f"Embeddings saved: {path}"
        )

    # ========================================================
    # LOAD EMBEDDINGS
    # ========================================================

    def load_embeddings(
        self,
        path: str
    ):

        with open(path, "rb") as f:

            return pickle.load(f)

    # ========================================================
    # EMBEDDING ANALYTICS
    # ========================================================

    def embedding_statistics(
        self,
        embeddings: np.ndarray
    ) -> Dict:

        stats = {

            "shape":
                embeddings.shape,

            "mean":
                float(np.mean(embeddings)),

            "std":
                float(np.std(embeddings)),

            "min":
                float(np.min(embeddings)),

            "max":
                float(np.max(embeddings)),

        }

        return stats


# ============================================================
# PUBLIC FUNCTIONS
# ============================================================

def generate_text_embeddings(
    texts: List[str]
):

    engine = EmbeddingEngine()

    return engine.generate_embeddings(
        texts
    )


def generate_customer_embeddings(
    df: pd.DataFrame
):

    engine = EmbeddingEngine()

    return engine.customer_embeddings(
        df
    )


def semantic_search(
    query: str,
    corpus_embeddings,
    corpus_texts,
    top_k: int = 5,
):

    engine = EmbeddingEngine()

    return engine.semantic_search(
        query,
        corpus_embeddings,
        corpus_texts,
        top_k
    )


def cluster_customer_embeddings(
    embeddings,
    n_clusters: int = 5,
):

    engine = EmbeddingEngine()

    return engine.cluster_embeddings(
        embeddings,
        n_clusters
    )