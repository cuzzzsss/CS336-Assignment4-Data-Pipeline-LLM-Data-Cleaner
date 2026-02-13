from __future__ import annotations

import os
from typing import Any

from cs336_data.preprocessing import (
    extract_text_from_html_bytes,
    identify_language,
    mask_emails,
    mask_phone_numbers,
    mask_ips,
    classify_nsfw,
    classify_toxic_speech,
    compute_quality_metrics, # 之前加的
    gopher_quality_filter    # <--- 新加这个！
)

from cs336_data.deduplication import (
    get_shingles, 
    compute_minhash_signature, 
    get_lsh_buckets,
    exact_line_deduplication, # 新加
    minhash_deduplication     # 新加
)

def run_extract_text_from_html_bytes(html_bytes: bytes) -> str | None:
    return extract_text_from_html_bytes(html_bytes)


def run_identify_language(text: str) -> tuple[Any, float]:
    return identify_language(text)


def run_mask_emails(text: str) -> tuple[str, int]:
    return mask_emails(text)

def run_mask_phone_numbers(text: str) -> tuple[str, int]:
    return mask_phone_numbers(text)

def run_mask_ips(text: str) -> tuple[str, int]:
    return mask_ips(text)


def run_classify_nsfw(text: str) -> tuple[Any, float]:
    return classify_nsfw(text)

def run_classify_toxic_speech(text: str) -> tuple[Any, float]:
    return classify_toxic_speech(text)


def run_compute_quality_metrics(text: str) -> dict[str, float | int]:
    return compute_quality_metrics(text)


def run_classify_quality(text: str) -> tuple[str, float]:
    """
    Quality Classifier adapter.
    Uses Gopher rules to predict quality.
    Returns: (label, score)
    """
    # 直接复用我们的过滤器逻辑
    is_high_quality = gopher_quality_filter(text)
    
    if is_high_quality:
        return "high", 1.0
    else:
        return "low", 0.0

def run_gopher_quality_filter(text: str) -> bool:
    return gopher_quality_filter(text)

def run_gopher_quality_filter(text: str) -> bool:
    return gopher_quality_filter(text)

def run_get_shingles(text: str, n: int):
    return get_shingles(text, n)

def run_compute_minhash_signature(shingles: Set[str], num_hashes: int):
    return compute_minhash_signature(shingles, num_hashes)

def run_get_lsh_buckets(signature: List[int], b: int, r: int):
    return get_lsh_buckets(signature, b, r)


def run_exact_line_deduplication(input_files, output_directory):
    return exact_line_deduplication(input_files, output_directory)

def run_minhash_deduplication(
    input_files, num_hashes, num_bands, ngrams, jaccard_threshold, output_directory
):
    return minhash_deduplication(
        input_files, num_hashes, num_bands, ngrams, jaccard_threshold, output_directory
    )
