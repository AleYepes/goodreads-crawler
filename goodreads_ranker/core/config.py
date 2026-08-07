import os

from dotenv import load_dotenv

load_dotenv()

DEFAULT_EMBEDDING_MODEL = "qwen3-embedding:8b"


def get_goodreads_email() -> str | None:
    return os.environ.get("GOODREADS_EMAIL")


def get_goodreads_password() -> str | None:
    return os.environ.get("GOODREADS_PASSWORD")


def get_embedding_model() -> str:
    return os.environ.get("OLLAMA_EMBEDDING_MODEL", DEFAULT_EMBEDDING_MODEL)


def get_api_url() -> str:
    return os.environ.get(
        "GOODREADS_API_URL", "https://kxbwmqov6jgg3daaamb744ycu4.appsync-api.us-east-1.amazonaws.com/graphql"
    )


def get_api_key() -> str:
    return os.environ.get("GOODREADS_API_KEY", "da2-xpgsdydkbregjhpr6ejzqdhuwy")


def calculate_count_adjusted_rating(rating, ratings_count, global_avg_rating: float = 3.5):
    """Calculate count-adjusted rating:
    rating - ((rating - global_avg_rating) / log10(ratings_count + 10))

    Returns global_avg_rating if ratings_count <= 0 or if rating/ratings_count is missing.
    """
    import math

    import numpy as np
    import pandas as pd

    if isinstance(rating, (pd.Series, np.ndarray)) or isinstance(ratings_count, (pd.Series, np.ndarray)):
        r = np.asarray(rating, dtype=float)
        c = np.asarray(ratings_count, dtype=float)
        valid_mask = np.isfinite(r) & np.isfinite(c) & (c > 0)
        res = np.full_like(r, global_avg_rating, dtype=float)
        if valid_mask.any():
            res[valid_mask] = r[valid_mask] - ((r[valid_mask] - global_avg_rating) / np.log10(c[valid_mask] + 10))
        return res
    else:
        try:
            if rating is None or ratings_count is None:
                return float(global_avg_rating)
            r_val = float(rating)
            c_val = float(ratings_count)
            if math.isnan(r_val) or math.isnan(c_val) or c_val <= 0:
                return float(global_avg_rating)
            return float(r_val - ((r_val - global_avg_rating) / math.log10(c_val + 10)))
        except ValueError, TypeError:
            return float(global_avg_rating)
