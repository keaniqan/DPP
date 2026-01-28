import streamlit as st
import numpy as np
import pandas as pd
import plotly.express as px
import re
import gzip
from io import StringIO
from collections import Counter
from sklearn.cluster import KMeans
from sklearn.manifold import TSNE
from sklearn.decomposition import PCA

# --- Configuration & Setup ---
st.set_page_config(
    page_title="DPP Diversity Explorer",
    page_icon="🧬",
    layout="wide"
)

# --- Helper: Text Processing ---
def clean_and_split_sentences(text):
    """Splits text into sentences and cleans them."""
    sentences = re.split(r'(?<!\w\.\w.)(?<![A-Z][a-z]\.)(?<=\.|\?|\!)\s+|\n+', text)
    sentences = [s.strip() for s in sentences if len(s.strip()) > 10]
    return sentences

def get_word_frequency(text):
    """Count word frequency in the text."""
    # Extract words (lowercase, alphanumeric only)
    words = re.findall(r'\b[a-zA-Z]+\b', text.lower())
    return Counter(words)

def get_high_frequency_words(word_counts, max_occurrences):
    """Get words that exceed the max occurrence threshold."""
    return {word for word, count in word_counts.items() if count > max_occurrences}

def filter_sentences_by_word_frequency(sentences, words_to_remove):
    """
    Remove high-frequency words from sentences.
    Returns filtered sentences with the words removed.
    """
    if not words_to_remove:
        return sentences
    
    filtered_sentences = []
    for sentence in sentences:
        # Split sentence into words, filter out high-frequency words, rejoin
        words = sentence.split()
        filtered_words = [
            word for word in words 
            if re.sub(r'[^a-zA-Z]', '', word.lower()) not in words_to_remove
        ]
        filtered_sentence = ' '.join(filtered_words)
        # Only keep if there's meaningful content left
        if len(filtered_sentence.strip()) > 5:
            filtered_sentences.append(filtered_sentence)
    
    return filtered_sentences

def remove_duplicate_sentences(sentences):
    """
    Remove exact duplicate sentences, keeping only the first occurrence.
    Useful for removing repeated headers/footers in documents.
    Returns deduplicated list and count of removed duplicates.
    """
    seen = set()
    unique_sentences = []
    duplicates_removed = 0
    
    for sentence in sentences:
        # Normalize for comparison (lowercase, strip whitespace)
        normalized = sentence.lower().strip()
        if normalized not in seen:
            seen.add(normalized)
            unique_sentences.append(sentence)
        else:
            duplicates_removed += 1
    
    return unique_sentences, duplicates_removed

# --- Helper: Embedding (The Semantic Space) ---
@st.cache_resource
def load_embedding_model():
    """Loads a lightweight Sentence Transformer model."""
    try:
        from sentence_transformers import SentenceTransformer
        return SentenceTransformer('all-MiniLM-L6-v2')
    except ImportError:
        return None

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

# --- Core Math: Determinantal Point Process ---
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

def calculate_normalized_diversity(log_det, n_sentences):
    """
    Normalize the diversity score to be comparable across documents of different sizes.
    
    Args:
        log_det: The raw log-determinant
        n_sentences: Number of sentences in the document
    
    Returns:
        Normalized diversity score (per-sentence average)
    """
    if n_sentences == 0:
        return 0.0
    return log_det / n_sentences

def get_diversity_interpretation(normalized_score):
    """
    Provide human-readable interpretation of the diversity score.
    """
    if normalized_score > 0.8:
        return "High Diversity", "success", "Document contains highly varied content with minimal redundancy."
    elif normalized_score > 0.5:
        return "Moderate Diversity", "warning", "Document has reasonable variety with some overlapping themes."
    elif normalized_score > 0.2:
        return "Low Diversity", "warning", "Document contains repetitive or closely related content."
    else:
        return "Very Low Diversity", "error", "Document is highly redundant with very similar sentences."

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

def cluster_sentences(embeddings, n_clusters):
    """
    Cluster sentences using K-Means clustering.
    
    Args:
        embeddings: Sentence embeddings (n_sentences x embedding_dim)
        n_clusters: Number of clusters to create
    
    Returns:
        cluster_labels: Array of cluster assignments (0 to n_clusters-1)
        kmeans: Fitted KMeans model
    """
    # Use K-Means clustering
    kmeans = KMeans(n_clusters=n_clusters, random_state=42, n_init=10)
    cluster_labels = kmeans.fit_predict(embeddings)
    
    return cluster_labels, kmeans

def get_cluster_keywords(sentences, cluster_labels, n_clusters, top_n=5):
    """
    Extract representative keywords for each cluster.
    
    Args:
        sentences: List of sentences
        cluster_labels: Cluster assignment for each sentence
        n_clusters: Total number of clusters
        top_n: Number of top keywords to extract per cluster
    
    Returns:
        Dictionary mapping cluster_id to list of top keywords
    """
    cluster_keywords = {}
    
    for cluster_id in range(n_clusters):
        # Get all sentences in this cluster
        cluster_sentences = [sentences[i] for i in range(len(sentences)) if cluster_labels[i] == cluster_id]
        
        if len(cluster_sentences) == 0:
            cluster_keywords[cluster_id] = []
            continue
        
        # Combine all sentences in the cluster
        cluster_text = ' '.join(cluster_sentences)
        
        # Count word frequencies (excluding common words)
        words = re.findall(r'\b[a-zA-Z]{4,}\b', cluster_text.lower())  # Words with 4+ letters
        word_counts = Counter(words)
        
        # Filter out very common words
        stopwords = {'that', 'this', 'with', 'from', 'have', 'been', 'were', 'will', 
                     'your', 'their', 'what', 'when', 'where', 'which', 'about', 'would',
                     'there', 'could', 'should', 'these', 'those', 'then', 'than', 'them'}
        
        filtered_counts = {word: count for word, count in word_counts.items() if word not in stopwords}
        
        # Get top N words
        top_words = sorted(filtered_counts.items(), key=lambda x: x[1], reverse=True)[:top_n]
        cluster_keywords[cluster_id] = [word for word, count in top_words]
    
    return cluster_keywords

def split_into_sections(sentences, n_sections=10):
    """
    Split sentences into n equal sections.
    
    Args:
        sentences: List of sentences
        n_sections: Number of sections to split into
    
    Returns:
        List of lists, where each inner list contains sentences for that section
    """
    if len(sentences) == 0:
        return []
    
    section_size = max(1, len(sentences) // n_sections)
    sections = []
    
    for i in range(n_sections):
        start_idx = i * section_size
        # Last section gets all remaining sentences
        end_idx = len(sentences) if i == n_sections - 1 else (i + 1) * section_size
        
        if start_idx < len(sentences):
            sections.append(sentences[start_idx:end_idx])
    
    return sections

def iterative_diversity_analysis(sentences, model, n_sections=10):
    """
    Perform iterative diversity analysis by cumulatively adding sections.
    
    Returns:
        DataFrame with iteration results
    """
    sections = split_into_sections(sentences, n_sections)
    results = []
    
    cumulative_sentences = []
    
    for i, section in enumerate(sections):
        if len(section) == 0:
            continue
            
        cumulative_sentences.extend(section)
        
        if len(cumulative_sentences) < 2:
            continue
        
        # Calculate diversity for cumulative text
        embeddings = get_embeddings(cumulative_sentences, model)
        kernel = build_kernel_matrix(embeddings)
        log_det, sign = calculate_dpp_diversity_score(kernel)
        normalized_score = calculate_normalized_diversity(log_det, len(cumulative_sentences))
        
        # Calculate Vendi Score
        vendi_score = calculate_vendi_score(kernel)
        
        # Calculate average similarity
        n = len(kernel)
        avg_similarity = (np.sum(kernel) - n) / (n * n - n) if n > 1 else 0
        
        results.append({
            'Section': i + 1,
            'Cumulative Sections': f"1-{i + 1}",
            'Sentences': len(cumulative_sentences),
            'Log-Det': log_det,
            'Normalized Score': normalized_score,
            'Vendi Score': vendi_score,
            'Avg Similarity': avg_similarity
        })
    
    return pd.DataFrame(results)

def iterative_compression_analysis(sentences, n_sections=10):
    """
    Perform iterative compression analysis by cumulatively adding sections.
    
    Returns:
        DataFrame with iteration results
    """
    sections = split_into_sections(sentences, n_sections)
    results = []
    
    cumulative_sentences = []
    
    for i, section in enumerate(sections):
        if len(section) == 0:
            continue
            
        cumulative_sentences.extend(section)
        
        if len(cumulative_sentences) < 2:
            continue
        
        # Calculate compression ratio
        cr, uncompressed, compressed = calculate_compression_ratio(cumulative_sentences)
        
        results.append({
            'Section': i + 1,
            'Sentences': len(cumulative_sentences),
            'Compression Ratio': cr,
            'Uncompressed': uncompressed,
            'Compressed': compressed
        })
    
    return pd.DataFrame(results)

def iterative_novascore_analysis(sentences, reference_embeddings, model, n_sections=10, threshold=0.15):
    """
    Perform iterative NovAScore analysis by cumulatively adding sections.
    
    Returns:
        DataFrame with iteration results
    """
    sections = split_into_sections(sentences, n_sections)
    results = []
    
    cumulative_sentences = []
    
    for i, section in enumerate(sections):
        if len(section) == 0:
            continue
            
        cumulative_sentences.extend(section)
        
        if len(cumulative_sentences) < 2:
            continue
        
        # Calculate NovAScore
        embeddings = get_embeddings(cumulative_sentences, model)
        novelty_scores, weighted_score, _ = novascore_calculation(
            embeddings, reference_embeddings, threshold=threshold
        )
        
        results.append({
            'Section': i + 1,
            'Sentences': len(cumulative_sentences),
            'NovAScore': weighted_score,
            'Avg Novelty': np.mean(novelty_scores) if len(novelty_scores) > 0 else 0
        })
    
    return pd.DataFrame(results)

# --- Core Math: Semantic Compression ---
def calculate_compression_ratio(sentences):
    """Calculates the Compression Ratio (CR) using gzip."""
    if not sentences:
        return 0.0, 0, 0
    full_text = " ".join(sentences)
    encoded_text = full_text.encode('utf-8')
    compressed_data = gzip.compress(encoded_text)
    uncompressed_size = len(encoded_text)
    compressed_size = len(compressed_data)
    if compressed_size == 0: return 0.0, 0, 0
    cr = uncompressed_size / compressed_size
    return cr, uncompressed_size, compressed_size

def novascore_calculation(target_embeddings, reference_embeddings, weights=None, threshold=0.15):
    """Calculates NovAScore using vector similarity with Iterative Self-Comparison."""
    num_target = len(target_embeddings)
    if num_target == 0:
        return np.array([]), 0.0, []

    max_sim_ref = np.zeros(num_target)
    ref_match_indices = np.full(num_target, -1, dtype=int)
    
    max_sim_self = np.zeros(num_target)
    self_match_indices = np.full(num_target, -1, dtype=int)

    if reference_embeddings is not None and len(reference_embeddings) > 0:
        sim_vs_ref = np.dot(target_embeddings, reference_embeddings.T)
        max_sim_ref = np.max(sim_vs_ref, axis=1)
        ref_match_indices = np.argmax(sim_vs_ref, axis=1)

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

# --- Embedding Visualization Helpers ---
def reduce_dimensions(embeddings, method='pca', n_components=2):
    """Reduce embedding dimensions for visualization."""
    if len(embeddings) < 2:
        return embeddings[:, :n_components] if embeddings.shape[1] >= n_components else embeddings
    
    from sklearn.decomposition import PCA
    
    if method == 'pca':
        n_comp = min(n_components, len(embeddings), embeddings.shape[1])
        pca = PCA(n_components=n_comp)
        return pca.fit_transform(embeddings)
    elif method == 'tsne':
        from sklearn.manifold import TSNE
        perplexity = min(30, len(embeddings) - 1)
        tsne = TSNE(n_components=n_components, perplexity=max(1, perplexity), random_state=42)
        return tsne.fit_transform(embeddings)
    return embeddings

def get_word_embeddings(sentence, model):
    """Get individual word embeddings and their contributions to the sentence."""
    if model is None:
        return None, None, None
    
    words = re.findall(r'\b\w+\b', sentence.lower())
    if len(words) == 0:
        return None, None, None
    
    sentence_emb = model.encode([sentence])[0]
    sentence_emb = sentence_emb / (np.linalg.norm(sentence_emb) + 1e-9)
    
    word_embs = model.encode(words)
    norms = np.linalg.norm(word_embs, axis=1, keepdims=True)
    norms[norms == 0] = 1
    word_embs = word_embs / norms
    
    contributions = np.dot(word_embs, sentence_emb)
    
    return words, word_embs, contributions

def calculate_word_similarity_matrix(words1, embs1, words2, embs2):
    """Calculate pairwise word similarities between two sentences."""
    if embs1 is None or embs2 is None:
        return None
    similarity = np.dot(embs1, embs2.T)
    return similarity

# --- UI Layout ---
st.title("🧬 Diversity & Novelty Explorer")
st.markdown("""
Evaluate text using three mathematical frameworks:
1. **DPP (Determinantal Point Process):** Geometric diversity within a single document.
2. **Semantic Compression:** Information theoretic redundancy.
3. **NovAScore:** Novelty quantification against historical/reference documents.
""")

# --- Sidebar ---
with st.sidebar:
    st.header("⚙️ Global Settings")
    model = load_embedding_model()
    if model is None:
        st.warning("⚠️ Using TF-IDF Fallback")
    
    st.divider()
    
    # Reference Document
    st.subheader("📚 Reference Document")
    st.caption("For NovAScore comparison")
    ref_file = st.file_uploader("Upload Reference (.txt)", type=['txt'], key="ref_upload")
    default_ref_text = "Mars is the fourth planet. Rovers have explored the surface."
    if ref_file:
        ref_text = StringIO(ref_file.getvalue().decode("utf-8")).read()
    else:
        ref_text = st.text_area("Or paste reference:", value=default_ref_text, height=100, key="ref_text")
    
    st.divider()
    
    # Analysis Settings
    st.subheader("🔧 Analysis Settings")
    
    novelty_threshold = st.slider(
        "Novelty Threshold", 
        min_value=0.0, max_value=1.0, value=0.85, step=0.01,
        help="Lower = More Forgiving. Higher = Stricter."
    )
    
    enable_word_filter = st.checkbox("Enable Word Filtering", value=False)
    
    word_freq_threshold = st.slider(
        "Max Word Repetition",
        min_value=1, max_value=50, value=10, step=1,
        help="Words appearing more than this many times will be filtered."
    )
    
    enable_duplicate_filter = st.checkbox("Remove Duplicate Sentences", value=True)
    
    st.divider()
    
    # Iterative Analysis Settings
    st.subheader("📈 Iterative Analysis")
    enable_iterative = st.checkbox(
        "Enable Iterative Analysis",
        value=False,
        help="Analyze diversity as sections are cumulatively added."
    )
    
    if enable_iterative:
        n_sections = st.slider(
            "Number of Sections",
            min_value=5,
            max_value=25,
            value=10,
            step=1
        )
    else:
        n_sections = 10

# --- Shared Input Configuration ---
st.header("📝 Input Configuration")

default_target_text = """
Artificial Intelligence is transforming the world. 
Machine learning models are becoming increasingly accurate.
AI systems are reshaping industries across the globe.
Deep learning is a subset of machine learning.
Oranges are a rich source of Vitamin C.
Citrus fruits are excellent for health.
The Eiffel Tower is located in Paris.
Paris is the capital of France.
"""

st.subheader("Target Document")
target_file = st.file_uploader("Upload Target Document (.txt)", type=['txt'], key="target_upload")
if target_file:
    target_text = StringIO(target_file.getvalue().decode("utf-8")).read()
else:
    target_text = st.text_area("Or paste target text:", value=default_target_text, height=200, key="target_text")

st.divider()

# --- Run All Button ---
run_col1, run_col2 = st.columns([1, 3])
with run_col1:
    run_all = st.button("🚀 Run All Analyses", type="primary", use_container_width=True)

with run_col2:
    st.caption("Run DPP Diversity, Semantic Compression, NovAScore, and Word Frequency analysis all at once.")

if run_all:
    with st.spinner("Running all analyses..."):
        # Preprocess text
        sentences_original = clean_and_split_sentences(target_text)
        
        if len(sentences_original) < 2:
            st.error("Need at least 2 sentences.")
        else:
            # Track filtering stats
            filter_stats = {'original': len(sentences_original), 'duplicates_removed': 0, 'words_filtered': 0}
            
            # Remove duplicates
            if enable_duplicate_filter:
                sentences_dedup, dups_removed = remove_duplicate_sentences(sentences_original)
                filter_stats['duplicates_removed'] = dups_removed
            else:
                sentences_dedup = sentences_original
            
            # Word filtering
            if enable_word_filter:
                word_counts = get_word_frequency(target_text)
                words_to_remove = get_high_frequency_words(word_counts, word_freq_threshold)
                filter_stats['words_filtered'] = len(words_to_remove)
                sentences = filter_sentences_by_word_frequency(sentences_dedup, words_to_remove)
            else:
                sentences = sentences_dedup
            
            if len(sentences) >= 2:
                # Show preprocessing info
                if filter_stats['duplicates_removed'] > 0 or filter_stats['words_filtered'] > 0:
                    filter_msg = []
                    if filter_stats['duplicates_removed'] > 0:
                        filter_msg.append(f"{filter_stats['duplicates_removed']} duplicate(s)")
                    if filter_stats['words_filtered'] > 0:
                        filter_msg.append(f"{filter_stats['words_filtered']} high-freq words")
                    st.info(f"📝 Preprocessing: Removed {', '.join(filter_msg)}. Analyzing {len(sentences)} sentences.")
                
                # 1. DPP Analysis
                embeddings = get_embeddings(sentences, model)
                kernel = build_kernel_matrix(embeddings)
                log_det, sign = calculate_dpp_diversity_score(kernel)
                normalized_score = calculate_normalized_diversity(log_det, len(sentences))
                vendi_score = calculate_vendi_score(kernel)
                
                temp_kernel = kernel.copy()
                np.fill_diagonal(temp_kernel, -1)
                most_sim_i, most_sim_j = np.unravel_index(np.argmax(temp_kernel), temp_kernel.shape)
                max_similarity = kernel[most_sim_i, most_sim_j]
                
                np.fill_diagonal(temp_kernel, 2)
                most_div_i, most_div_j = np.unravel_index(np.argmin(temp_kernel), temp_kernel.shape)
                min_similarity = kernel[most_div_i, most_div_j]
                
                n = len(kernel)
                avg_similarity = (np.sum(kernel) - n) / (n * n - n) if n > 1 else 0
                
                # Iterative DPP analysis
                iterative_dpp_df = None
                if enable_iterative and len(sentences) >= n_sections:
                    iterative_dpp_df = iterative_diversity_analysis(sentences, model, n_sections=n_sections)
                
                st.session_state['dpp_results'] = {
                    'sentences': sentences,
                    'embeddings': embeddings,
                    'kernel': kernel,
                    'log_det': log_det,
                    'sign': sign,
                    'normalized_score': normalized_score,
                    'vendi_score': vendi_score,
                    'most_sim_pair': (most_sim_i, most_sim_j, max_similarity),
                    'most_div_pair': (most_div_i, most_div_j, min_similarity),
                    'avg_similarity': avg_similarity,
                    'iterative_df': iterative_dpp_df
                }
                
                # 2. Compression Analysis
                full_cr, full_sz, comp_sz = calculate_compression_ratio(sentences)
                iterative_compression_df = None
                if enable_iterative and len(sentences) >= n_sections:
                    iterative_compression_df = iterative_compression_analysis(sentences, n_sections=n_sections)
                
                st.session_state['compression_results'] = {
                    'sentences': sentences,
                    'full_cr': full_cr,
                    'full_sz': full_sz,
                    'comp_sz': comp_sz,
                    'iterative_df': iterative_compression_df
                }
                
                # 3. NovAScore Analysis
                r_units = clean_and_split_sentences(ref_text)
                if enable_duplicate_filter:
                    r_units, _ = remove_duplicate_sentences(r_units)
                r_embs = get_embeddings(r_units, model)
                
                weights = np.ones(len(sentences))
                novelty_scores, final_score, match_data = novascore_calculation(
                    embeddings, r_embs, weights, threshold=novelty_threshold
                )
                
                matched_content = []
                match_locations = []
                for score, (m_type, m_idx) in zip(novelty_scores, match_data):
                    if m_idx == -1:
                        matched_content.append("None")
                        match_locations.append("New Concept")
                    else:
                        if m_type == 'ref':
                            matched_content.append(r_units[m_idx] if m_idx < len(r_units) else "Ref Error")
                            match_locations.append("Reference Doc")
                        elif m_type == 'self':
                            matched_content.append(sentences[m_idx] if m_idx < len(sentences) else "Self Error")
                            match_locations.append(f"Self (Unit {m_idx + 1})")
                
                # Iterative NovAScore
                iterative_nova_df = None
                if enable_iterative and len(sentences) >= n_sections:
                    iterative_nova_df = iterative_novascore_analysis(
                        sentences, r_embs, model, n_sections=n_sections, threshold=novelty_threshold
                    )
                
                st.session_state['nova_results'] = {
                    't_units': sentences,
                    'r_units': r_units,
                    't_embs': embeddings,
                    'r_embs': r_embs,
                    'novelty_scores': novelty_scores,
                    'final_score': final_score,
                    'match_locations': match_locations,
                    'matched_content': matched_content,
                    'iterative_df': iterative_nova_df
                }
                
                # 4. Word Frequency Analysis
                word_counts = get_word_frequency(target_text)
                st.session_state['word_freq_results'] = {
                    'word_counts': word_counts,
                    'total_words': sum(word_counts.values()),
                    'unique_words': len(word_counts)
                }
                
                st.success("✅ All analyses complete! Check the tabs below for results.")

st.divider()

# Tabs for results
tab_main, tab_compression, tab_novascore, tab_word_freq, tab_iterative = st.tabs([
    "📊 Diversity (DPP)", 
    "📦 Compression", 
    "🔬 Novelty (NovAScore)", 
    "📈 Word Frequency",
    "📉 Iterative Analysis"
])

# --- TAB 1: DPP ---
with tab_main:
    st.subheader("Document Diversity Analysis (DPP)")
    st.markdown("""
    Evaluates the **overall diversity** of your document using Determinantal Point Processes.
    
    **How it works:**
    1. Convert sentences to embeddings → Matrix $V$ of size $(n \\times D)$
    2. Build similarity kernel $L = V \\cdot V^T$ (cosine similarity between all sentence pairs)
    3. Calculate $\\log\\det(L + I)$ — the **diversity score**
    """)

    if st.button("Calculate Document Diversity", key="btn_dpp"):
        input_text = target_text
        sentences_original = clean_and_split_sentences(input_text)
        
        if len(sentences_original) < 2:
            st.error("Need at least 2 sentences.")
        else:
            # Track filtering stats
            filter_stats = {'original': len(sentences_original), 'duplicates_removed': 0, 'words_filtered': 0}
            
            # Step 0a: Remove duplicate sentences if enabled
            if enable_duplicate_filter:
                sentences_dedup, dups_removed = remove_duplicate_sentences(sentences_original)
                filter_stats['duplicates_removed'] = dups_removed
            else:
                sentences_dedup = sentences_original
            
            # Step 0b: Apply word frequency filtering if enabled
            if enable_word_filter:
                word_counts = get_word_frequency(input_text)
                words_to_remove = get_high_frequency_words(word_counts, word_freq_threshold)
                filter_stats['words_filtered'] = len(words_to_remove)
                sentences = filter_sentences_by_word_frequency(sentences_dedup, words_to_remove)
            else:
                sentences = sentences_dedup
            
            if len(sentences) < 2:
                st.error("After filtering, less than 2 sentences remain. Try adjusting filter settings.")
            else:
                # Show filtering info
                if filter_stats['duplicates_removed'] > 0 or filter_stats['words_filtered'] > 0:
                    filter_msg = []
                    if filter_stats['duplicates_removed'] > 0:
                        filter_msg.append(f"{filter_stats['duplicates_removed']} duplicate(s) removed")
                    if filter_stats['words_filtered'] > 0:
                        filter_msg.append(f"{filter_stats['words_filtered']} high-freq words filtered")
                    st.info(f"📝 Preprocessing: {', '.join(filter_msg)}. Analyzing {len(sentences)} sentences (from {filter_stats['original']} original).")
                
                # Step 1: Convert sentences to embeddings (V matrix)
                embeddings = get_embeddings(sentences, model)
                
                # Step 2: Build similarity kernel L = V · V^T
                kernel = build_kernel_matrix(embeddings)
                
                # Step 3: Calculate log-det(L + I) for diversity score
                log_det, sign = calculate_dpp_diversity_score(kernel)
                normalized_score = calculate_normalized_diversity(log_det, len(sentences))
                
                # Calculate Vendi Score
                vendi_score = calculate_vendi_score(kernel)
                
                # Find most similar pair for comparison
                temp_kernel = kernel.copy()
                np.fill_diagonal(temp_kernel, -1)
                most_sim_i, most_sim_j = np.unravel_index(np.argmax(temp_kernel), temp_kernel.shape)
                max_similarity = kernel[most_sim_i, most_sim_j]
                
                # Find most diverse pair
                np.fill_diagonal(temp_kernel, 2)
                most_div_i, most_div_j = np.unravel_index(np.argmin(temp_kernel), temp_kernel.shape)
                min_similarity = kernel[most_div_i, most_div_j]
                
                # Average similarity (off-diagonal)
                n = len(kernel)
                avg_similarity = (np.sum(kernel) - n) / (n * n - n) if n > 1 else 0
                
                # Perform iterative analysis if enabled
                iterative_df = None
                if enable_iterative and len(sentences) >= n_sections:
                    with st.spinner(f"Running iterative analysis ({n_sections} sections)..."):
                        iterative_df = iterative_diversity_analysis(sentences, model, n_sections=n_sections)
                
                # Store results in session state
                st.session_state['dpp_results'] = {
                    'sentences': sentences,
                    'embeddings': embeddings,
                    'kernel': kernel,
                    'log_det': log_det,
                    'sign': sign,
                    'normalized_score': normalized_score,
                    'vendi_score': vendi_score,
                    'most_sim_pair': (most_sim_i, most_sim_j, max_similarity),
                    'most_div_pair': (most_div_i, most_div_j, min_similarity),
                    'avg_similarity': avg_similarity,
                    'iterative_df': iterative_df
                }
    
    if 'dpp_results' in st.session_state:
        results = st.session_state['dpp_results']
        sentences = results['sentences']
        embeddings = results['embeddings']
        kernel = results['kernel']
        log_det = results['log_det']
        sign = results['sign']
        normalized_score = results['normalized_score']
        vendi_score = results['vendi_score']
        most_sim_pair = results['most_sim_pair']
        most_div_pair = results['most_div_pair']
        avg_similarity = results['avg_similarity']
        
        # Get interpretation
        interpretation, status, description = get_diversity_interpretation(normalized_score)
        
        st.divider()
        
        # Main metrics row
        st.subheader("📊 Diversity Metrics")
        
        m1, m2, m3, m4 = st.columns(4)
        
        with m1:
            st.metric(
                "Log-Determinant", 
                f"{log_det:.4f}",
                help="log det(L + I) — Higher = More Diverse (grows with noise)"
            )
        
        with m2:
            st.metric(
                "Vendi Score",
                f"{vendi_score:.2f}",
                help="Effective number of unique concepts — Plateaus for redundant content"
            )
        
        with m3:
            st.metric(
                "Normalized Score",
                f"{normalized_score:.4f}",
                help="Log-det divided by number of sentences (per-sentence diversity)"
            )
        

        
        # Interpretation
        if status == "success":
            st.success(f"**{interpretation}**: {description}")
        elif status == "error":
            st.error(f"**{interpretation}**: {description}")
        else:
            st.warning(f"**{interpretation}**: {description}")
        
        st.divider()
        
        # Detailed analysis
        col1, col2 = st.columns(2)
        
        with col1:
            st.markdown("**🔴 Most Similar Pair (Redundant)**")
            i, j, sim = most_sim_pair
            st.metric("Similarity", f"{sim:.4f}")
            st.caption(f"Sentence {i+1}: {sentences[i][:80]}...")
            st.caption(f"Sentence {j+1}: {sentences[j][:80]}...")
            if sim > 0.9:
                st.error("⚠️ Very high similarity — consider removing one")
        
        with col2:
            st.markdown("**🟢 Most Diverse Pair**")
            i, j, sim = most_div_pair
            st.metric("Similarity", f"{sim:.4f}")
            st.caption(f"Sentence {i+1}: {sentences[i][:80]}...")
            st.caption(f"Sentence {j+1}: {sentences[j][:80]}...")
        
        st.divider()
        st.markdown(f"**Average Pairwise Similarity:** {avg_similarity:.4f}")
        st.caption("Lower average similarity = more diverse document")
        
        st.divider()
        st.subheader("🔬 The Mathematics")
        
        vis_tab1, vis_tab2, vis_tab3 = st.tabs(["Similarity Matrix (Kernel L)", "Diversity Visualization", "🎯 Concept Clustering"])
            
        with vis_tab1:
            st.markdown("**Kernel Matrix $L = V \\cdot V^T$**")
            st.write("Each cell shows the cosine similarity between two sentences. Darker red = More Similar.")
            
            labels = [f"S{i+1}: {s[:20]}..." for i, s in enumerate(sentences)]
            
            fig = px.imshow(
                kernel,
                labels=dict(x="Sentence", y="Sentence", color="Similarity"),
                x=labels,
                y=labels,
                color_continuous_scale="RdBu_r",
                zmin=-1, zmax=1
            )
            
            st.plotly_chart(fig, use_container_width=True)
            st.caption("Diagonal = 1.0 (sentence vs itself). Off-diagonal shows pairwise similarity.")
            
            # Show the formula
            st.markdown("""
            **The DPP Diversity Formula:**
            
            $$\\text{Diversity Score} = \\log\\det(L + I)$$
            
            Where:
            - $L$ = Similarity kernel matrix (shown above)
            - $I$ = Identity matrix (for numerical stability)
            - Higher log-det = vectors span more volume = **more diverse**
            """)

        with vis_tab2:
            st.markdown("### 📊 Embedding Space Visualization")
            st.markdown("""
            This shows all sentences projected into 2D space using PCA.
            - Sentences **spread apart** = High diversity
            - Sentences **clustered together** = Low diversity (redundant)
            """)
            
            if len(embeddings) > 0:
                # Use PCA to reduce to 2D for visualization
                from sklearn.decomposition import PCA
                n_components = min(2, len(embeddings), embeddings.shape[1])
                pca = PCA(n_components=n_components)
                coords_2d = pca.fit_transform(embeddings)
                
                # Calculate distance from centroid for coloring
                centroid = np.mean(coords_2d, axis=0)
                distances = np.linalg.norm(coords_2d - centroid, axis=1)
                
                # Create dataframe for plotting
                viz_df = pd.DataFrame({
                    'X': coords_2d[:, 0],
                    'Y': coords_2d[:, 1] if n_components > 1 else np.zeros(len(coords_2d)),
                    'Sentence': [f"S{i+1}: {s[:50]}..." for i, s in enumerate(sentences)],
                    'Distance from Center': distances,
                    'Index': [f"S{i+1}" for i in range(len(sentences))]
                })
                
                fig = px.scatter(
                    viz_df,
                    x='X',
                    y='Y',
                    color='Distance from Center',
                    hover_data=['Sentence'],
                    text='Index',
                    color_continuous_scale='Viridis',
                    title="Sentence Embeddings in 2D Space (PCA Projection)"
                )
                
                fig.update_traces(textposition='top center', marker=dict(size=12))
                
                # Add centroid marker
                fig.add_scatter(
                    x=[centroid[0]], 
                    y=[centroid[1]], 
                    mode='markers',
                    marker=dict(size=20, color='red', symbol='x'),
                    name='Centroid'
                )
                
                fig.update_layout(
                    height=500,
                    xaxis_title=f"PC1 ({pca.explained_variance_ratio_[0]:.1%} variance)",
                    yaxis_title=f"PC2 ({pca.explained_variance_ratio_[1]:.1%} variance)" if n_components > 1 else "PC2"
                )
                
                st.plotly_chart(fig, use_container_width=True)
                st.caption("Wider spread = higher diversity. Red X marks the centroid (average position).")
                
            st.divider()
            
            # Similarity distribution
            st.markdown("### 📈 Similarity Distribution")
            
            # Get upper triangle of similarity matrix (excluding diagonal)
            upper_tri = kernel[np.triu_indices(len(kernel), k=1)]
            
            fig_hist = px.histogram(
                x=upper_tri, 
                nbins=30,
                labels={'x': 'Pairwise Similarity', 'y': 'Count'},
                title="Distribution of Pairwise Similarities"
            )
            fig_hist.add_vline(x=avg_similarity, line_dash="dash", line_color="red",
                              annotation_text=f"Mean: {avg_similarity:.3f}")
            
            st.plotly_chart(fig_hist, use_container_width=True)
            st.caption("Left-skewed distribution = diverse document. Right-skewed = redundant document.")
            
            st.info(f"""
            **Summary Statistics:**
            - Mean Similarity: {avg_similarity:.4f}
            - Min Similarity: {np.min(upper_tri):.4f}
            - Max Similarity: {np.max(upper_tri):.4f}
            - Std Deviation: {np.std(upper_tri):.4f}
            """)
        
        with vis_tab3:
            st.markdown("### 🎯 Concept Clustering")
            st.markdown(f"""
            Based on the **Vendi Score of {vendi_score:.0f}**, your document contains approximately 
            **{vendi_score:.0f} unique concepts**. Let's cluster the sentences to identify what these concepts are.
            """)
            
            # Let user choose number of clusters
            n_clusters = st.slider(
                "Number of clusters to create",
                min_value=2,
                max_value=min(50, len(sentences)),
                value=int(round(vendi_score)),
                help="Default is set to the Vendi Score (number of unique concepts detected)"
            )
            
            # Perform clustering
            with st.spinner(f"Clustering {len(sentences)} sentences into {n_clusters} groups..."):
                cluster_labels, kmeans = cluster_sentences(embeddings, n_clusters)
                cluster_keywords = get_cluster_keywords(sentences, cluster_labels, n_clusters, top_n=5)
            
            # Visualization method
            viz_method = st.radio(
                "Visualization method:",
                ["t-SNE (better separation)", "PCA (faster)"],
                horizontal=True
            )
            
            # Reduce to 2D for visualization
            if viz_method.startswith("t-SNE"):
                with st.spinner("Computing t-SNE projection..."):
                    tsne = TSNE(n_components=2, random_state=42, perplexity=min(30, len(sentences)-1))
                    coords_2d = tsne.fit_transform(embeddings)
                    method_name = "t-SNE"
            else:
                pca = PCA(n_components=2, random_state=42)
                coords_2d = pca.fit_transform(embeddings)
                method_name = "PCA"
            
            # Create visualization dataframe
            cluster_df = pd.DataFrame({
                'X': coords_2d[:, 0],
                'Y': coords_2d[:, 1],
                'Cluster': [f"Cluster {label}" for label in cluster_labels],
                'Cluster ID': cluster_labels,
                'Sentence': sentences,
                'Preview': [s[:80] + '...' if len(s) > 80 else s for s in sentences]
            })
            
            # Plot clusters
            fig_cluster = px.scatter(
                cluster_df,
                x='X',
                y='Y',
                color='Cluster',
                hover_data=['Preview'],
                title=f'Sentence Clusters ({method_name} Projection)',
                color_discrete_sequence=px.colors.qualitative.Set3 + px.colors.qualitative.Pastel
            )
            
            # Add cluster centers
            for cluster_id in range(n_clusters):
                cluster_points = coords_2d[cluster_labels == cluster_id]
                if len(cluster_points) > 0:
                    center = cluster_points.mean(axis=0)
                    fig_cluster.add_scatter(
                        x=[center[0]],
                        y=[center[1]],
                        mode='markers+text',
                        marker=dict(size=15, color='black', symbol='x', line=dict(width=2, color='white')),
                        text=[f"C{cluster_id}"],
                        textposition='middle center',
                        textfont=dict(color='white', size=10),
                        showlegend=False,
                        hoverinfo='skip'
                    )
            
            fig_cluster.update_traces(marker=dict(size=10, opacity=0.7))
            fig_cluster.update_layout(height=600, showlegend=True)
            
            st.plotly_chart(fig_cluster, use_container_width=True)
            
            # Show cluster statistics
            st.divider()
            st.markdown("### 📊 Cluster Analysis")
            
            cluster_stats = []
            for cluster_id in range(n_clusters):
                cluster_size = np.sum(cluster_labels == cluster_id)
                keywords = cluster_keywords.get(cluster_id, [])
                keyword_str = ', '.join(keywords[:5]) if keywords else '(no keywords)'
                
                cluster_stats.append({
                    'Cluster': f"Cluster {cluster_id}",
                    'Sentences': cluster_size,
                    '% of Total': f"{100 * cluster_size / len(sentences):.1f}%",
                    'Top Keywords': keyword_str
                })
            
            stats_df = pd.DataFrame(cluster_stats)
            stats_df = stats_df.sort_values('Sentences', ascending=False)
            
            st.dataframe(stats_df, use_container_width=True, hide_index=True)
            
            # Show sentences by cluster
            st.divider()
            st.markdown("### 📝 Sentences by Cluster")
            
            selected_cluster = st.selectbox(
                "Select a cluster to view its sentences:",
                options=range(n_clusters),
                format_func=lambda x: f"Cluster {x} ({np.sum(cluster_labels == x)} sentences) - {', '.join(cluster_keywords.get(x, [])[:3])}"
            )
            
            # Get sentences in selected cluster
            cluster_sentences = [
                (i+1, sentences[i]) 
                for i in range(len(sentences)) 
                if cluster_labels[i] == selected_cluster
            ]
            
            st.markdown(f"**Cluster {selected_cluster}** contains **{len(cluster_sentences)} sentences**:")
            st.markdown(f"*Keywords:* {', '.join(cluster_keywords.get(selected_cluster, ['N/A']))}")
            
            for idx, sentence in cluster_sentences[:20]:  # Show first 20
                st.markdown(f"**{idx}.** {sentence}")
            
            if len(cluster_sentences) > 20:
                st.caption(f"... and {len(cluster_sentences) - 20} more sentences")

# --- TAB 2: Semantic Compression ---
with tab_compression:
    st.subheader("Semantic Compression Analysis")
    st.markdown("""
    Measures information redundancy using **gzip compression ratio**.
    A higher compression ratio indicates more repetitive/redundant content.
    """)
    st.caption("Analyzing the **Target Document** for information theoretic redundancy.")
    
    if st.button("Calculate Compression Metrics", key="btn_compression"):
        input_text = target_text
        sentences = clean_and_split_sentences(input_text)
        if len(sentences) < 2:
            st.error("Need at least 2 sentences.")
        else:
            full_cr, full_sz, comp_sz = calculate_compression_ratio(sentences)
            
            # Iterative analysis
            iterative_compression_df = None
            if enable_iterative and len(sentences) >= n_sections:
                with st.spinner(f"Running iterative analysis ({n_sections} sections)..."):
                    iterative_compression_df = iterative_compression_analysis(sentences, n_sections=n_sections)
            
            st.session_state['compression_results'] = {
                'sentences': sentences,
                'full_cr': full_cr,
                'full_sz': full_sz,
                'comp_sz': comp_sz,
                'iterative_df': iterative_compression_df
            }
    
    if 'compression_results' in st.session_state:
        results = st.session_state['compression_results']
        sentences = results['sentences']
        full_cr = results['full_cr']
        full_sz = results['full_sz']
        comp_sz = results['comp_sz']
        
        st.divider()
        
        # Metrics display
        m1, m2, m3 = st.columns(3)
        
        with m1:
            st.metric("Compression Ratio", f"{full_cr:.2f}", help="Compression Ratio = Uncompressed / Compressed")
        
        with m2:
            st.metric("Original Size", f"{full_sz:,} bytes")
        
        with m3:
            st.metric("Compressed Size", f"{comp_sz:,} bytes")
        
        # Interpretation
        if full_cr > 3.0:
            st.error("**High Redundancy**: Document compresses very well, indicating repetitive content.")
        elif full_cr > 2.0:
            st.warning("**Moderate Redundancy**: Document has some repetitive patterns.")
        else:
            st.success("**Low Redundancy**: Document has relatively unique content.")
        
        st.divider()
        
        # Explanation
        st.markdown("""
        **How to interpret Compression Ratio:**
        - **CR > 3.0** = Highly redundant (lots of repeated phrases/ideas)
        - **CR 2.0-3.0** = Moderately redundant
        - **CR < 2.0** = Low redundancy (more unique content)
        
        The compression ratio measures how much the text can be compressed using gzip.
        Repetitive text compresses better, resulting in a higher ratio.
        """)

# --- TAB 3: NovAScore ---
with tab_novascore:
    st.subheader("NovAScore: Novelty Evaluation")
    st.markdown("""
    Based on *Ai et al. (2024)*, NovAScore measures how much **new** information a document contributes compared to a reference.
    
    **Workflow:**
    1. **Decompose:** Break text into sentences.
    2. **Embed:** Convert sentences to vectors.
    3. **Score:** Iteratively calculate novelty. A sentence is penalized if it matches Reference History OR any previous sentence in the target text.
    """)
    st.caption("Comparing **Target Document** against **Reference Document**.")
    
    if st.button("Calculate NovAScore", key="btn_nova"):
        t_text = target_text
        r_text = ref_text
        with st.spinner("Analyzing Novelty..."):
            # Get original sentences
            t_units_original = clean_and_split_sentences(t_text)
            r_units_original = clean_and_split_sentences(r_text)
            
            # Remove exact duplicates if enabled
            if enable_duplicate_filter:
                t_units_dedup, t_dups_removed = remove_duplicate_sentences(t_units_original)
                r_units_dedup, r_dups_removed = remove_duplicate_sentences(r_units_original)
                
                if t_dups_removed > 0 or r_dups_removed > 0:
                    st.info(f"📝 Removed {t_dups_removed} duplicate(s) from target, {r_dups_removed} from reference")
            else:
                t_units_dedup = t_units_original
                r_units_dedup = r_units_original
            
            # Apply word frequency filtering if enabled
            if enable_word_filter:
                # Get word frequency from target text
                word_counts = get_word_frequency(t_text)
                words_to_remove = get_high_frequency_words(word_counts, word_freq_threshold)
                
                if words_to_remove:
                    st.info(f"🔄 Filtering {len(words_to_remove)} high-frequency words: {', '.join(sorted(words_to_remove)[:10])}{'...' if len(words_to_remove) > 10 else ''}")
                
                t_units = filter_sentences_by_word_frequency(t_units_dedup, words_to_remove)
                r_units = filter_sentences_by_word_frequency(r_units_dedup, words_to_remove)
            else:
                t_units = t_units_dedup
                r_units = r_units_dedup
            
            t_embs = get_embeddings(t_units, model)
            r_embs = get_embeddings(r_units, model)
            
            weights = np.ones(len(t_units)) 
            novelty_scores, final_score, match_data = novascore_calculation(t_embs, r_embs, weights, threshold=novelty_threshold)
            
            matched_content = []
            match_locations = []

            for score, (m_type, m_idx) in zip(novelty_scores, match_data):
                if m_idx == -1:
                     matched_content.append("None")
                     match_locations.append("New Concept")
                else:
                    if m_type == 'ref':
                        if m_idx < len(r_units):
                            matched_content.append(r_units[m_idx])
                        else:
                            matched_content.append("Ref Error")
                        match_locations.append("Reference Doc")
                    elif m_type == 'self':
                        if m_idx < len(t_units):
                            matched_content.append(t_units[m_idx])
                        else:
                            matched_content.append("Self Error")
                        match_locations.append(f"Self (Unit {m_idx + 1})")

            st.session_state['nova_results'] = {
                't_units': t_units,
                'r_units': r_units,
                't_embs': t_embs,
                'r_embs': r_embs,
                'novelty_scores': novelty_scores,
                'final_score': final_score,
                'match_locations': match_locations,
                'matched_content': matched_content,
                'iterative_df': None  # Will be populated if iterative is enabled
            }
            
            # Iterative NovAScore analysis
            if enable_iterative and len(t_units) >= n_sections:
                with st.spinner(f"Running iterative NovAScore analysis ({n_sections} sections)..."):
                    iterative_nova_df = iterative_novascore_analysis(
                        t_units, r_embs, model, n_sections=n_sections, threshold=novelty_threshold
                    )
                    st.session_state['nova_results']['iterative_df'] = iterative_nova_df
    
    if 'nova_results' in st.session_state:
        results = st.session_state['nova_results']
        t_units = results['t_units']
        r_units = results['r_units']
        t_embs = results['t_embs']
        r_embs = results['r_embs']
        novelty_scores = results['novelty_scores']
        final_score = results['final_score']
        match_locations = results['match_locations']
        matched_content = results['matched_content']
        
        col_u1, col_u2 = st.columns(2)
        with col_u1: st.write(f"**Target Units:** {len(t_units)}")
        with col_u2: st.write(f"**Reference Units:** {len(r_units)}")

        st.divider()
        
        # Scoring section above the table
        score_col1, score_col2, score_col3 = st.columns([1, 1, 2])
        
        with score_col1:
            st.metric("NovAScore (Novelty)", f"{final_score:.2%}")
        
        with score_col2:
            if final_score > 0.7:
                st.success("High Novelty")
            elif final_score < 0.3:
                st.error("Low Novelty (Redundant)")
            else:
                st.warning("Moderate Novelty")
        
        st.divider()
        
        # Full-width table
        st.subheader("Unit Breakdown")
        
        df = pd.DataFrame({
            "Unit #": range(1, len(t_units) + 1),
            "Content Unit (ACU)": t_units,
            "Novelty Score": novelty_scores,
            "Match Source": match_locations,
            "Closest Match Text": matched_content
        })
        
        st.dataframe(
            df,
            column_config={
                "Unit #": st.column_config.NumberColumn(
                    "Unit #",
                    help="Unit number (1-indexed)",
                    width="small"
                ),
                "Novelty Score": st.column_config.ProgressColumn(
                    "Novelty Score",
                    help="0.0 = Redundant (found in history or self-repetitive), 1.0 = Completely New",
                    format="%.2f",
                    min_value=0,
                    max_value=1,
                ),
                "Closest Match Text": st.column_config.TextColumn(
                    "Closest Match Text",
                    help="The text that is most similar to the target unit (causing the redundancy)",
                    width="large"
                )
            },
            use_container_width=True,
            hide_index=True
        )
        
        st.caption("Novelty Score: 0.0 = Information exists in Reference or appeared earlier in document. 1.0 = Completely new information.")

        st.divider()
        st.subheader("🔍 Embedding Visualizations")
        
        vis_tab1, vis_tab2, vis_tab3 = st.tabs([
            "📊 Similarity Heatmap", 
            "🗺️ Embedding Space (2D)", 
            "🔤 Word-Level Analysis"
        ])
        
        with vis_tab1:
            st.markdown("**Cross-Document Similarity Matrix**")
            st.caption("Shows how similar each Target unit is to each Reference unit. Darker = More Similar.")
            
            if len(t_embs) > 0 and len(r_embs) > 0:
                cross_sim = np.dot(t_embs, r_embs.T)
                
                t_labels = [f"T{i+1}: {u[:25]}..." for i, u in enumerate(t_units)]
                r_labels = [f"R{i+1}: {u[:25]}..." for i, u in enumerate(r_units)]
                
                fig_cross = px.imshow(
                    cross_sim,
                    labels=dict(x="Reference Units", y="Target Units", color="Similarity"),
                    x=r_labels,
                    y=t_labels,
                    color_continuous_scale="RdBu_r",
                    zmin=-1, zmax=1,
                    aspect="auto"
                )
                fig_cross.update_layout(height=400)
                st.plotly_chart(fig_cross, use_container_width=True)
                
                st.markdown("**Target Self-Similarity Matrix**")
                st.caption("Shows internal redundancy within the Target document.")
                
                self_sim = np.dot(t_embs, t_embs.T)
                fig_self = px.imshow(
                    self_sim,
                    labels=dict(x="Target Units", y="Target Units", color="Similarity"),
                    x=t_labels,
                    y=t_labels,
                    color_continuous_scale="RdBu_r",
                    zmin=-1, zmax=1,
                    aspect="auto"
                )
                fig_self.update_layout(height=400)
                st.plotly_chart(fig_self, use_container_width=True)
            else:
                st.warning("Not enough embeddings to visualize.")
        
        with vis_tab2:
            st.markdown("**2D Embedding Space Projection**")
            st.caption("Units close together in this space are semantically similar.")
            
            dim_method = st.radio("Reduction Method", ["PCA", "t-SNE"], horizontal=True, key="dim_method")
            
            if len(t_embs) > 0:
                all_embs = np.vstack([t_embs, r_embs]) if len(r_embs) > 0 else t_embs
                all_labels = t_units + r_units if len(r_embs) > 0 else t_units
                all_types = ['Target'] * len(t_units) + ['Reference'] * len(r_units)
                all_ids = [f"T{i+1}" for i in range(len(t_units))] + [f"R{i+1}" for i in range(len(r_units))]
                
                all_novelty = list(novelty_scores) + [None] * len(r_units)
                
                reduced = reduce_dimensions(all_embs, method=dim_method.lower())
                
                if reduced.shape[1] >= 2:
                    plot_df = pd.DataFrame({
                        'x': reduced[:, 0],
                        'y': reduced[:, 1],
                        'text': [t[:50] + "..." if len(t) > 50 else t for t in all_labels],
                        'type': all_types,
                        'id': all_ids,
                        'novelty': all_novelty
                    })
                    
                    fig_2d = px.scatter(
                        plot_df,
                        x='x', y='y',
                        color='type',
                        hover_data=['id', 'text', 'novelty'],
                        text='id',
                        color_discrete_map={'Target': '#1f77b4', 'Reference': '#ff7f0e'},
                        title=f"Embedding Space ({dim_method})"
                    )
                    fig_2d.update_traces(textposition='top center', marker=dict(size=12))
                    fig_2d.update_layout(height=500)
                    st.plotly_chart(fig_2d, use_container_width=True)
                    
                    st.caption("**T** = Target units, **R** = Reference units. Hover for details.")
                else:
                    st.warning("Not enough dimensions for 2D projection.")
            else:
                st.warning("No embeddings to visualize.")
        
        with vis_tab3:
            st.markdown("**Word-Level Similarity Analysis**")
            st.caption("Explore which words in one sentence correlate with words in another.")
            
            if model is None:
                st.warning("Word-level analysis requires the Sentence Transformer model.")
            else:
                all_sentences = t_units + r_units
                all_sentence_labels = [f"T{i+1}: {s[:40]}..." for i, s in enumerate(t_units)] + \
                                      [f"R{i+1}: {s[:40]}..." for i, s in enumerate(r_units)]
                
                if len(all_sentences) >= 2:
                    col_s1, col_s2 = st.columns(2)
                    with col_s1:
                        sent1_idx = st.selectbox("Select First Sentence", range(len(all_sentences)), 
                                                 format_func=lambda x: all_sentence_labels[x], key="word_sent1")
                    with col_s2:
                        sent2_idx = st.selectbox("Select Second Sentence", range(len(all_sentences)), 
                                                 index=min(1, len(all_sentences)-1),
                                                 format_func=lambda x: all_sentence_labels[x], key="word_sent2")
                    
                    sent1 = all_sentences[sent1_idx]
                    sent2 = all_sentences[sent2_idx]
                    
                    words1, embs1, contrib1 = get_word_embeddings(sent1, model)
                    words2, embs2, contrib2 = get_word_embeddings(sent2, model)
                    
                    if words1 is not None and words2 is not None and len(words1) > 0 and len(words2) > 0:
                        word_sim = calculate_word_similarity_matrix(words1, embs1, words2, embs2)
                        
                        st.markdown("**Word-to-Word Similarity Matrix**")
                        st.caption("Shows which words from Sentence 1 are semantically similar to words in Sentence 2.")
                        
                        fig_word = px.imshow(
                            word_sim,
                            labels=dict(x="Sentence 2 Words", y="Sentence 1 Words", color="Similarity"),
                            x=words2,
                            y=words1,
                            color_continuous_scale="Viridis",
                            zmin=0, zmax=1,
                            aspect="auto"
                        )
                        fig_word.update_layout(height=400)
                        st.plotly_chart(fig_word, use_container_width=True)
                        
                        st.markdown("**Word Importance (Contribution to Sentence Meaning)**")
                        
                        word_col1, word_col2 = st.columns(2)
                        with word_col1:
                            st.caption(f"**Sentence 1:** {sent1[:60]}...")
                            contrib_df1 = pd.DataFrame({
                                'Word': words1,
                                'Contribution': contrib1
                            }).sort_values('Contribution', ascending=False)
                            
                            fig_contrib1 = px.bar(contrib_df1, x='Word', y='Contribution', 
                                                  color='Contribution', color_continuous_scale='Blues')
                            fig_contrib1.update_layout(height=250, showlegend=False)
                            st.plotly_chart(fig_contrib1, use_container_width=True)
                        
                        with word_col2:
                            st.caption(f"**Sentence 2:** {sent2[:60]}...")
                            contrib_df2 = pd.DataFrame({
                                'Word': words2,
                                'Contribution': contrib2
                            }).sort_values('Contribution', ascending=False)
                            
                            fig_contrib2 = px.bar(contrib_df2, x='Word', y='Contribution',
                                                  color='Contribution', color_continuous_scale='Oranges')
                            fig_contrib2.update_layout(height=250, showlegend=False)
                            st.plotly_chart(fig_contrib2, use_container_width=True)
                        
                        st.markdown("**Strongest Word Connections**")
                        top_k = min(5, word_sim.size)
                        flat_indices = np.argsort(word_sim.flatten())[-top_k:][::-1]
                        
                        connections = []
                        for idx in flat_indices:
                            i, j = np.unravel_index(idx, word_sim.shape)
                            connections.append({
                                'Word 1': words1[i],
                                'Word 2': words2[j],
                                'Similarity': f"{word_sim[i, j]:.3f}"
                            })
                        
                        st.table(pd.DataFrame(connections))
                    else:
                        st.warning("Could not extract words from the selected sentences.")
                else:
                    st.warning("Need at least 2 sentences for word-level comparison.")

# --- TAB 3: Word Frequency Analysis ---
with tab_word_freq:
    st.subheader("📈 Word Frequency Analysis")
    st.markdown("""
    Analyze word usage patterns in your target document. Use this to identify common/repetitive words 
    that might skew novelty analysis. High-frequency words like articles and common verbs can be filtered out.
    """)
    
    if st.button("Analyze Word Frequency", key="btn_word_freq"):
        with st.spinner("Counting words..."):
            # Get word frequency
            word_counts = get_word_frequency(target_text)
            
            if not word_counts:
                st.error("No words found in the target document.")
            else:
                # Store in session state
                st.session_state['word_freq_results'] = {
                    'word_counts': word_counts,
                    'total_words': sum(word_counts.values()),
                    'unique_words': len(word_counts)
                }
    
    if 'word_freq_results' in st.session_state:
        results = st.session_state['word_freq_results']
        word_counts = results['word_counts']
        total_words = results['total_words']
        unique_words = results['unique_words']
        
        # Summary metrics
        metric_col1, metric_col2, metric_col3, metric_col4 = st.columns(4)
        with metric_col1:
            st.metric("Total Words", f"{total_words:,}")
        with metric_col2:
            st.metric("Unique Words", f"{unique_words:,}")
        with metric_col3:
            vocab_richness = unique_words / total_words if total_words > 0 else 0
            st.metric("Vocabulary Richness", f"{vocab_richness:.2%}")
        with metric_col4:
            words_above_threshold = len([w for w, c in word_counts.items() if c > word_freq_threshold])
            st.metric(f"Words > {word_freq_threshold} occurrences", words_above_threshold)
        
        st.divider()
        
        # Create DataFrame sorted by frequency
        df_words = pd.DataFrame([
            {'Word': word, 'Count': count, 'Frequency %': (count / total_words) * 100}
            for word, count in word_counts.items()
        ]).sort_values('Count', ascending=False).reset_index(drop=True)
        
        # Add rank column
        df_words.insert(0, 'Rank', range(1, len(df_words) + 1))
        
        # Mark words that would be filtered
        df_words['Would Be Filtered'] = df_words['Count'] > word_freq_threshold
        
        # Display options
        view_col1, view_col2 = st.columns([2, 1])
        with view_col1:
            show_top_n = st.slider("Show top N words", 10, min(200, len(df_words)), 50, key="show_top_n")
        with view_col2:
            show_filtered_only = st.checkbox("Show only filterable words", value=False)
        
        # Filter display
        display_df = df_words.copy()
        if show_filtered_only:
            display_df = display_df[display_df['Would Be Filtered']]
        display_df = display_df.head(show_top_n)
        
        # Two columns: chart and table
        chart_col, table_col = st.columns([1, 1])
        
        with chart_col:
            st.markdown("**Word Frequency Distribution**")
            
            # Bar chart of top words
            fig = px.bar(
                display_df.head(30),
                x='Word',
                y='Count',
                color='Would Be Filtered',
                color_discrete_map={True: '#ff6b6b', False: '#4ecdc4'},
                labels={'Would Be Filtered': 'Exceeds Threshold'}
            )
            fig.update_layout(
                xaxis_tickangle=-45,
                height=400,
                showlegend=True
            )
            fig.add_hline(
                y=word_freq_threshold, 
                line_dash="dash", 
                line_color="red",
                annotation_text=f"Threshold: {word_freq_threshold}"
            )
            st.plotly_chart(fig, use_container_width=True)
        
        with table_col:
            st.markdown("**Word Frequency Table**")
            
            # Style the dataframe
            styled_df = display_df[['Rank', 'Word', 'Count', 'Frequency %', 'Would Be Filtered']].copy()
            styled_df['Frequency %'] = styled_df['Frequency %'].apply(lambda x: f"{x:.2f}%")
            
            st.dataframe(
                styled_df,
                column_config={
                    'Rank': st.column_config.NumberColumn('Rank', width='small'),
                    'Word': st.column_config.TextColumn('Word', width='medium'),
                    'Count': st.column_config.NumberColumn('Count', width='small'),
                    'Frequency %': st.column_config.TextColumn('Freq %', width='small'),
                    'Would Be Filtered': st.column_config.CheckboxColumn('Filter?', width='small')
                },
                use_container_width=True,
                height=400
            )
        
        st.divider()
        
        # Show words that would be filtered
        filtered_words = df_words[df_words['Would Be Filtered']]['Word'].tolist()
        if filtered_words:
            st.warning(f"⚠️ **{len(filtered_words)} words** exceed the threshold of {word_freq_threshold} and would be filtered:")
            
            # Display in columns
            n_cols = 5
            word_cols = st.columns(n_cols)
            for i, word in enumerate(filtered_words[:50]):  # Limit to 50
                with word_cols[i % n_cols]:
                    count = word_counts[word]
                    st.code(f"{word}: {count}")
            
            if len(filtered_words) > 50:
                st.caption(f"... and {len(filtered_words) - 50} more words")
        else:
            st.success(f"✅ No words exceed the threshold of {word_freq_threshold}. No filtering will be applied.")
        
        # Common stopwords suggestion
        st.divider()
        st.markdown("**💡 Common Stopwords Reference**")
        st.caption("These are typical high-frequency words you might want to filter:")
        
        common_stopwords = ['the', 'a', 'an', 'and', 'or', 'but', 'in', 'on', 'at', 'to', 'for', 
                           'of', 'with', 'by', 'from', 'as', 'is', 'was', 'are', 'were', 'been',
                           'be', 'have', 'has', 'had', 'do', 'does', 'did', 'will', 'would', 
                           'could', 'should', 'may', 'might', 'must', 'shall', 'this', 'that',
                           'these', 'those', 'it', 'its', 'they', 'them', 'their', 'we', 'us',
                           'our', 'you', 'your', 'he', 'she', 'him', 'her', 'his', 'who', 'which']
        
        # Show which stopwords appear in the document and their counts
        stopwords_in_doc = {word: word_counts.get(word, 0) for word in common_stopwords if word in word_counts}
        if stopwords_in_doc:
            stopwords_df = pd.DataFrame([
                {'Word': w, 'Count': c} for w, c in sorted(stopwords_in_doc.items(), key=lambda x: -x[1])
            ])
            st.dataframe(stopwords_df, use_container_width=True, height=200)

# --- TAB 5: Iterative Analysis ---
with tab_iterative:
    st.subheader("📉 Iterative Analysis (Diminishing Returns)")
    st.markdown("""
    This tab shows how all metrics change as more content is added to the document.
    
    **Enable iterative analysis in the sidebar** to see results here.
    
    **Key Insights:**
    - **Diminishing returns** = Metric plateaus; adding more text doesn't improve diversity/novelty
    - **Linear growth** = Each section adds genuine new value
    """)
    
    if not enable_iterative:
        st.info("👈 Enable **Iterative Analysis** in the sidebar to see diminishing returns analysis.")
    else:
        # Check if we have iterative results
        has_dpp_iter = 'dpp_results' in st.session_state and st.session_state['dpp_results'].get('iterative_df') is not None
        has_compression_iter = 'compression_results' in st.session_state and st.session_state['compression_results'].get('iterative_df') is not None
        has_nova_iter = 'nova_results' in st.session_state and st.session_state['nova_results'].get('iterative_df') is not None
        
        if not any([has_dpp_iter, has_compression_iter, has_nova_iter]):
            st.warning("⚠️ No iterative analysis results yet. Click **Run All Analyses** above to generate results.")
        else:
            # Combined overview chart
            st.markdown("### 📊 Combined Metrics Overview")
            
            combined_data = []
            
            if has_dpp_iter:
                dpp_df = st.session_state['dpp_results']['iterative_df']
                for _, row in dpp_df.iterrows():
                    combined_data.append({
                        'Sentences': row['Sentences'],
                        'Metric': 'Vendi Score',
                        'Value': row['Vendi Score'],
                        'Category': 'Diversity'
                    })
            
            if has_compression_iter:
                comp_df = st.session_state['compression_results']['iterative_df']
                for _, row in comp_df.iterrows():
                    combined_data.append({
                        'Sentences': row['Sentences'],
                        'Metric': 'Compression Ratio',
                        'Value': row['Compression Ratio'],
                        'Category': 'Compression'
                    })
            
            if has_nova_iter:
                nova_df = st.session_state['nova_results']['iterative_df']
                for _, row in nova_df.iterrows():
                    combined_data.append({
                        'Sentences': row['Sentences'],
                        'Metric': 'NovAScore',
                        'Value': row['NovAScore'] * 100,  # Convert to percentage scale
                        'Category': 'Novelty'
                    })
            
            if combined_data:
                combined_df = pd.DataFrame(combined_data)
                
                fig_combined = px.line(
                    combined_df,
                    x='Sentences',
                    y='Value',
                    color='Metric',
                    markers=True,
                    title='All Metrics vs Document Size',
                    labels={'Value': 'Score', 'Sentences': 'Cumulative Sentences'}
                )
                fig_combined.update_layout(height=400, hovermode='x unified')
                st.plotly_chart(fig_combined, use_container_width=True)
            
            st.divider()
            
            # Individual metric tabs
            iter_sub_tabs = st.tabs(["📊 DPP Diversity", "📦 Compression", "🔬 NovAScore"])
            
            with iter_sub_tabs[0]:
                if has_dpp_iter:
                    st.markdown("### DPP Diversity Over Time")
                    dpp_df = st.session_state['dpp_results']['iterative_df']
                    
                    col1, col2 = st.columns(2)
                    
                    with col1:
                        fig_dpp = px.line(
                            dpp_df,
                            x='Sentences',
                            y=['Log-Det', 'Vendi Score'],
                            markers=True,
                            title='Diversity Metrics Growth'
                        )
                        fig_dpp.update_layout(height=350)
                        st.plotly_chart(fig_dpp, use_container_width=True)
                    
                    with col2:
                        # Rate of change
                        change_df = dpp_df.copy()
                        change_df['Vendi Change'] = change_df['Vendi Score'].diff()
                        
                        fig_change = px.bar(
                            change_df[1:],
                            x='Section',
                            y='Vendi Change',
                            title='Vendi Score Increase per Section',
                            color='Vendi Change',
                            color_continuous_scale='RdYlGn'
                        )
                        fig_change.add_hline(y=0, line_dash="dash", line_color="red")
                        fig_change.update_layout(height=350)
                        st.plotly_chart(fig_change, use_container_width=True)
                    
                    with st.expander("📋 View DPP Data"):
                        st.dataframe(dpp_df, use_container_width=True, hide_index=True)
                else:
                    st.info("Run DPP analysis to see iterative results.")
            
            with iter_sub_tabs[1]:
                if has_compression_iter:
                    st.markdown("### Compression Ratio Over Time")
                    comp_df = st.session_state['compression_results']['iterative_df']
                    
                    col1, col2 = st.columns(2)
                    
                    with col1:
                        fig_comp = px.line(
                            comp_df,
                            x='Sentences',
                            y='Compression Ratio',
                            markers=True,
                            title='Compression Ratio Growth'
                        )
                        fig_comp.update_layout(height=350)
                        st.plotly_chart(fig_comp, use_container_width=True)
                    
                    with col2:
                        # Size comparison
                        fig_size = px.area(
                            comp_df,
                            x='Sentences',
                            y=['Uncompressed', 'Compressed'],
                            title='Document Size (bytes)'
                        )
                        fig_size.update_layout(height=350)
                        st.plotly_chart(fig_size, use_container_width=True)
                    
                    with st.expander("📋 View Compression Data"):
                        st.dataframe(comp_df, use_container_width=True, hide_index=True)
                else:
                    st.info("Run Compression analysis to see iterative results.")
            
            with iter_sub_tabs[2]:
                if has_nova_iter:
                    st.markdown("### NovAScore Over Time")
                    nova_df = st.session_state['nova_results']['iterative_df']
                    
                    col1, col2 = st.columns(2)
                    
                    with col1:
                        fig_nova = px.line(
                            nova_df,
                            x='Sentences',
                            y='NovAScore',
                            markers=True,
                            title='Novelty Score Growth'
                        )
                        fig_nova.update_layout(height=350, yaxis_tickformat='.0%')
                        st.plotly_chart(fig_nova, use_container_width=True)
                    
                    with col2:
                        # Rate of change
                        change_df = nova_df.copy()
                        change_df['Novelty Change'] = change_df['NovAScore'].diff() * 100
                        
                        fig_change = px.bar(
                            change_df[1:],
                            x='Section',
                            y='Novelty Change',
                            title='NovAScore Change per Section (%)',
                            color='Novelty Change',
                            color_continuous_scale='RdYlGn'
                        )
                        fig_change.add_hline(y=0, line_dash="dash", line_color="red")
                        fig_change.update_layout(height=350)
                        st.plotly_chart(fig_change, use_container_width=True)
                    
                    st.markdown("""
                    **Interpretation:**
                    - **Declining NovAScore** = Later sections repeat earlier content or reference material
                    - **Stable NovAScore** = Consistent level of novelty throughout
                    - **Increasing NovAScore** = Later sections introduce more unique content
                    """)
                    
                    with st.expander("📋 View NovAScore Data"):
                        st.dataframe(nova_df, use_container_width=True, hide_index=True)
                else:
                    st.info("Run NovAScore analysis to see iterative results.")