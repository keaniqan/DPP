import re
import gzip
import zlib
import bz2
from collections import Counter
import numpy as np

def clean_and_split_sentences(text):
    """Splits text into sentences and cleans them."""
    sentences = re.split(r'(?<!\w\.\w.)(?<![A-Z][a-z]\.)(?<=\.|\?|\!)\s+|\n+', text)
    sentences = [s.strip() for s in sentences if len(s.strip()) > 10]
    return sentences

def load_embedding_model():
    """Loads a lightweight Sentence Transformer model."""
    try:
        from sentence_transformers import SentenceTransformer
        return SentenceTransformer('all-MiniLM-L6-v2')
    except ImportError:
        return None

################################################
#          Text Analysis Functions             #
################################################

def get_word_frequency(text):
    """Count word frequency in the text."""
    # Extract words (lowercase, alphanumeric only)
    words = re.findall(r'\b[a-zA-Z]+\b', text.lower())
    return Counter(words)



def get_total_words(text):
    return sum(get_word_frequency(text).values())



def get_unique_words(text):
    return len(get_word_frequency(text))



def calculate_diversity_compression_ratio(sentences, algorithm='gzip'):
    """Calculate compression ratio: ratio of compressed size to original size.
    Lower ratio = more repetitive text. Higher ratio = more diverse text."""
    try:
        text = ' '.join(sentences)
        text_bytes = text.encode('utf-8')
        original_size = len(text_bytes)
        if original_size == 0:
            return None

        compressors = {
            'gzip': lambda b: gzip.compress(b),
            'zlib': lambda b: zlib.compress(b),
            'bz2': lambda b: bz2.compress(b),
        }
        compress = compressors.get(algorithm, compressors['gzip'])
        compressed_size = len(compress(text_bytes))
        return compressed_size / original_size
    except Exception:
        return None



def calculate_ngram_diversity(sentences, num_n=4):
    """Calculate n-gram diversity: ratio of unique n-grams to total n-grams.
    Averaged across n=1..num_n. Score near 1 = highly diverse, near 0 = repetitive."""
    try:
        text = ' '.join(sentences)
        words = text.lower().split()
        if len(words) < num_n:
            return None

        scores = []
        for n in range(1, num_n + 1):
            ngrams = [tuple(words[i:i + n]) for i in range(len(words) - n + 1)]
            if ngrams:
                scores.append(len(set(ngrams)) / len(ngrams))
        return sum(scores) / len(scores) if scores else None
    except Exception:
        return None
    


################################################
#        Embedding Analysis Functions          #
################################################

def get_embeddings(sentences, model):
    """Generates vector embeddings for a list of sentences."""
    if not sentences:
        return np.array([])
        
    if model:
        embeddings = model.encode(sentences)
        norms = np.linalg.norm(embeddings, axis=1, keepdims=True)
        norms[norms == 0] = 1
        return embeddings / norms
    else:
        from sklearn.feature_extraction.text import TfidfVectorizer
        vectorizer = TfidfVectorizer()
        embeddings = vectorizer.fit_transform(sentences).toarray()
        return embeddings



def build_kernel_matrix(embeddings, alpha=1.0):
    """Constructs the DPP Kernel Matrix (L) = V · V^T.
    
    Args:
        embeddings: Matrix V of size (n × D) where n is number of sentences, D is embedding dimension
        alpha: Scaling factor for the similarity
    
    Returns:
        L: Similarity kernel matrix where L[i,j] = cosine similarity between sentence i and j
    """
    if len(embeddings) == 0:
        return np.array([[]])
    # L = V · V^T (dot product gives cosine similarity for normalized vectors)
    similarity_matrix = np.dot(embeddings, embeddings.T)
    L = similarity_matrix * alpha
    return L



def calculate_dpp_diversity_score(kernel_matrix):
    """
    Calculate the DPP diversity score using log-determinant of (L + I).
    
    The log-determinant measures the overall diversity/volume of the document.
    - Higher log-det = More diverse content (vectors span more space)
    - Lower log-det = More redundant content (vectors clustered together)
    
    We use slogdet to prevent overflow/underflow with large matrices.
    
    Args:
        kernel_matrix: The similarity kernel L
    
    Returns:
        log_det: The log-determinant score
        sign: The sign of the determinant (should be positive for valid kernel)
    """
    if kernel_matrix.size == 0:
        return 0.0, 1
    
    # Add identity matrix for numerical stability: det(L + I)
    L_plus_I = kernel_matrix + np.eye(len(kernel_matrix))
    
    # Use slogdet to prevent overflow/underflow
    sign, log_det = np.linalg.slogdet(L_plus_I)
    
    return log_det, sign



def calculate_theoretical_max_log_det(n_sentences):
    """
    Calculate the theoretical maximum log-determinant assuming perfect orthogonality.
    
    If every sentence were completely semantically unique (orthogonal), 
    the similarity matrix L would be an identity matrix I.
    Then L + I = 2I, and det(2I) = 2^n for n sentences.
    Therefore, max log-det = log(2^n) = n * log(2)
    
    Args:
        n_sentences: Number of sentences
    
    Returns:
        theoretical_max: The theoretical maximum log-determinant
    """
    import math
    return n_sentences * math.log(2)



def calculate_diversity_percentage(actual_log_det, n_sentences):
    """
    Calculate what percentage of theoretical maximum diversity is achieved.
    
    Args:
        actual_log_det: The actual log-determinant score
        n_sentences: Number of sentences
    
    Returns:
        percentage: Percentage of maximum theoretical diversity (0-100%)
    """
    theoretical_max = calculate_theoretical_max_log_det(n_sentences)
    if theoretical_max <= 0:
        return 0.0
    return min(100.0, (actual_log_det / theoretical_max) * 100)



def calculate_vendi_score(kernel_matrix):
    """
    Calculate the Vendi Score - measures the effective number of independent factors.
    
    Unlike Log-Det which grows indefinitely, Vendi Score hits a "hard ceiling" 
    determined by the actual number of unique concepts.
    
    Formula:
        K_bar = K / n  (normalize by number of items)
        H(λ) = -Σ λ_i * log(λ_i)
        VS = exp(H(λ))
    
    Interpretation:
        - 100 paraphrases → VS ≈ 1.0 (1 unique idea)
        - 100 unique facts → VS ≈ 100.0 (100 unique ideas)
    
    Args:
        kernel_matrix: The similarity kernel K (with 1s on diagonal)
    
    Returns:
        vendi_score: The effective number of unique concepts
    """
    if kernel_matrix.size == 0 or len(kernel_matrix) < 2:
        return 1.0
    
    n = len(kernel_matrix)
    
    # Normalize matrix by n (number of sentences)
    K_bar = kernel_matrix / n
    
    # Compute eigenvalues of normalized matrix
    eigenvalues = np.linalg.eigvalsh(K_bar)
    
    # Filter numerical noise (eigenvalues must be positive)
    eigenvalues = eigenvalues[eigenvalues > 1e-10]
    
    if len(eigenvalues) == 0:
        return 1.0
    
    # Compute Shannon Entropy
    # Note: Sum of eigenvalues is already approx 1.0 due to normalization
    entropy = -np.sum(eigenvalues * np.log(eigenvalues))
    
    # Vendi Score
    score = np.exp(entropy)
    
    return score



def novascore_calculation(target_embeddings, weights=None, threshold=0.00):
    """Calculates NovAScore using vector similarity with Iterative Self-Comparison."""
    num_target = len(target_embeddings)
    if num_target == 0:
        return np.array([]), 0.0, []

    max_sim_ref = np.zeros(num_target)
    ref_match_indices = np.full(num_target, -1, dtype=int)
    
    max_sim_self = np.zeros(num_target)
    self_match_indices = np.full(num_target, -1, dtype=int)

    sim_vs_self = np.dot(target_embeddings, target_embeddings.T)
    
    for i in range(1, num_target):
        previous_sims = sim_vs_self[i, :i]
        if len(previous_sims) > 0:
            max_sim_self[i] = np.max(previous_sims)
            self_match_indices[i] = np.argmax(previous_sims)
            
    final_max_sim = np.zeros(num_target)
    match_data = []

    for i in range(num_target):
        score_ref = max_sim_ref[i]
        score_self = max_sim_self[i]
        
        if score_ref >= score_self:
            final_max_sim[i] = score_ref
            if ref_match_indices[i] != -1:
                match_data.append(('ref', ref_match_indices[i]))
            else:
                match_data.append(('none', -1))
        else:
            final_max_sim[i] = score_self
            match_data.append(('self', self_match_indices[i]))

    similarity_cutoff = 1.0 - threshold
    final_max_sim[final_max_sim < similarity_cutoff] = 0.0
    
    novelty_scores = 1 - final_max_sim
    novelty_scores = np.clip(novelty_scores, 0, 1)
    
    if weights is None:
        weights = np.ones(num_target)
        
    weighted_score = np.average(novelty_scores, weights=weights)
    
    return novelty_scores, weighted_score, match_data

################################################
#                EVALUATE ALL                  #
################################################

def evaluate_all(text):
    sentences = clean_and_split_sentences(text)

    word_freq = get_word_frequency(text)
    total_words = get_total_words(text)

    unique_words = get_unique_words(text)
    diversity_ratio = calculate_diversity_compression_ratio(sentences)
    ngram_diversity = calculate_ngram_diversity(sentences)

    embedding_model = load_embedding_model()
    embeddings = get_embeddings(sentences, embedding_model)
    kernel_matrix = build_kernel_matrix(embeddings)
    
    log_det, sign = calculate_dpp_diversity_score(kernel_matrix)
    diversity_percentage = calculate_diversity_percentage(log_det, len(sentences))
    vendi_score = calculate_vendi_score(kernel_matrix)
    novascore = novascore_calculation(embeddings) if embeddings.size > 0 else ([], 0.0, [])

    return {
        "word_frequency": word_freq,
        "total_words": total_words,
        "unique_words": unique_words,
        "diversity_compression_ratio": diversity_ratio,
        "ngram_diversity": ngram_diversity,
        "dpp_log_determinant": log_det,
        "dpp_sign": sign,
        "diversity_percentage": diversity_percentage,
        "vendi_score": vendi_score,
        "novascore": novascore
    }

# results = evaluate_all("""Your input text goes here. This function will analyze the text and return various metrics related to word frequency, diversity, and novelty.""")

# print(results)

if __name__ == "__main__":
    sample_text = """Your input text goes here. This function will analyze the text and return various metrics related to word frequency, diversity, and novelty."""
    results = evaluate_all(sample_text)
    print(results)