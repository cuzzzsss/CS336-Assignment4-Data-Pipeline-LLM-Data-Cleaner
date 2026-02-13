import hashlib
import re
import os
from pathlib import Path
from typing import Iterable, Set, List

# --- 1. Shingling ---
def get_shingles(text: str, n: int) -> Set[str]:
    text = text.lower()
    text = re.sub(r'[^\w\s]', ' ', text)
    tokens = text.split()
    if len(tokens) < n:
        return set()
    shingles = set()
    for i in range(len(tokens) - n + 1):
        shingles.add(" ".join(tokens[i : i + n]))
    return shingles

# --- 2. MinHash Signature ---
def compute_minhash_signature(shingles: Set[str], num_hashes: int) -> List[int]:
    if not shingles:
        return [0] * num_hashes
    signature = []
    for i in range(num_hashes):
        min_val = float('inf')
        for shingle in shingles:
            h = hashlib.sha256(f"{i}_{shingle}".encode()).hexdigest()
            val = int(h, 16)
            if val < min_val:
                min_val = val
        signature.append(min_val)
    return signature

# --- 3. LSH Bucketing ---
def get_lsh_buckets(signature: List[int], b: int, r: int) -> List[tuple]:
    if len(signature) != b * r:
        raise ValueError(f"Signature length {len(signature)} must equal b * r")
    buckets = []
    for i in range(b):
        band = tuple(signature[i * r : (i + 1) * r])
        band_hash = hashlib.md5(str(band).encode()).hexdigest()
        buckets.append((i, band_hash))
    return buckets

# --- 4. Exact Line Deduplication ---
def exact_line_deduplication(input_files: list[os.PathLike], output_directory: os.PathLike):
    """
    更精确的行去重实现。
    根据报错，测试期望的是：如果一行内容是重复的噪声（比如反爬警告），
    则在结果中应该将其彻底移除。
    """
    from collections import Counter
    output_dir = Path(output_directory)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # 1. 统计所有行出现的次数
    line_counts = Counter()
    all_docs_lines = []
    
    for file_path in input_files:
        with open(file_path, 'r', encoding='utf-8') as f:
            lines = f.readlines()
            all_docs_lines.append((file_path, lines))
            for line in lines:
                line_counts[line] += 1
                
    # 2. 根据频率过滤并写出文件
    for file_path, lines in all_docs_lines:
        deduped_lines = []
        for line in lines:
            # 如果这行只出现过一次，说明是独特内容，保留
            # 如果出现多次，说明是重复噪声，根据测试预期应删掉
            if line_counts[line] == 1:
                deduped_lines.append(line)
        
        output_file = output_dir / Path(file_path).name
        with open(output_file, 'w', encoding='utf-8') as f:
            f.writelines(deduped_lines)

# --- 5. MinHash LSH Deduplication ---
def minhash_deduplication(
    input_files: list[os.PathLike],
    num_hashes: int,
    num_bands: int,
    ngrams: int,
    jaccard_threshold: float,
    output_directory: os.PathLike,
):
    output_dir = Path(output_directory)
    output_dir.mkdir(parents=True, exist_ok=True)
    buckets = {} # (band_idx, band_hash) -> [first_file_found]
    for file_path in input_files:
        with open(file_path, 'r', encoding='utf-8') as f:
            text = f.read()
        shingles = get_shingles(text, ngrams)
        sig = compute_minhash_signature(shingles, num_hashes)
        bands = get_lsh_buckets(sig, num_bands, num_hashes // num_bands)
        is_duplicate = False
        for band in bands:
            if band in buckets:
                is_duplicate = True
                break
        if not is_duplicate:
            for band in bands:
                buckets[band] = file_path
            output_file = output_dir / Path(file_path).name
            with open(output_file, 'w', encoding='utf-8') as f:
                f.write(text)