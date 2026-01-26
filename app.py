import streamlit as st
import numpy as np
import pandas as pd
import plotly.express as px
import re
import gzip
import json
import requests
from io import StringIO

# --- Configuration & Setup ---
st.set_page_config(
    page_title="DPP Diversity Explorer",
    page_icon="🧬",
    layout="wide"
)

# --- Helper: Text Processing ---
def clean_and_split_sentences(text):
    """Splits text into sentences and cleans them."""
    # Split by standard sentence delimiters (improved regex)
    # Matches . ? ! followed by space, or newlines
    sentences = re.split(r'(?<!\w\.\w.)(?<![A-Z][a-z]\.)(?<=\.|\?|\!)\s+|\n+', text)
    # Remove empty strings and very short sentences
    sentences = [s.strip() for s in sentences if len(s.strip()) > 10]
    return sentences

# --- Helper: Embedding (The Semantic Space) ---
@st.cache_resource
def load_embedding_model():
    """
    Loads a lightweight Sentence Transformer model.
    Using 'all-MiniLM-L6-v2' for speed and efficiency.
    """
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
        # Normalize embeddings to unit length for Cosine Similarity
        norms = np.linalg.norm(embeddings, axis=1, keepdims=True)
        # Avoid division by zero
        norms[norms == 0] = 1
        return embeddings / norms
    else:
        # Fallback: TF-IDF if sentence-transformers is missing
        from sklearn.feature_extraction.text import TfidfVectorizer
        vectorizer = TfidfVectorizer()
        embeddings = vectorizer.fit_transform(sentences).toarray()
        return embeddings

# --- Core Math: Determinantal Point Process ---

def build_kernel_matrix(embeddings, alpha=1.0):
    """
    Constructs the DPP Kernel Matrix (L).
    L_ij = Similarity(i, j) * Quality(i) * Quality(j)
    """
    if len(embeddings) == 0:
        return np.array([[]])
    similarity_matrix = np.dot(embeddings, embeddings.T)
    L = similarity_matrix * alpha
    return L

def dpp_greedy_selection(kernel_matrix, k):
    """Performs Greedy MAP Inference for a k-DPP."""
    num_items = kernel_matrix.shape[0]
    if num_items == 0:
        return []
    if k >= num_items:
        return list(range(num_items))

    selected_indices = []
    available_indices = list(range(num_items))
    
    for _ in range(k):
        best_item = -1
        max_log_det = -np.inf
        
        for item in available_indices:
            current_subset = selected_indices + [item]
            submatrix = kernel_matrix[np.ix_(current_subset, current_subset)]
            try:
                # Regularize diagonal
                d = np.linalg.det(submatrix + np.eye(len(current_subset)) * 1e-6)
                score = np.log(d) if d > 0 else -np.inf
            except np.linalg.LinAlgError:
                score = -np.inf
                
            if score > max_log_det:
                max_log_det = score
                best_item = item
        
        if best_item != -1:
            selected_indices.append(best_item)
            available_indices.remove(best_item)
        else:
            break
            
    return selected_indices

def calculate_log_det_score(matrix):
    if matrix.size == 0: return 0
    sign, logdet = np.linalg.slogdet(matrix + np.eye(matrix.shape[0]))
    return logdet if sign > 0 else 0

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
    """
    Calculates NovAScore using vector similarity with Iterative Self-Comparison.
    
    Args:
        threshold (float): Novelty Threshold (0.0 - 1.0). 
                           Similarities below (1 - threshold) are ignored (treated as 0.0).
                           Lower threshold -> Higher Ignore Cutoff -> More Forgiving.
    
    Returns:
        novelty_scores: Array of novelty scores (0.0 to 1.0)
        weighted_score: Single weighted average score
        match_data: List of tuples (match_type, match_index) indicating where the closest match was found.
                    match_type is 'ref', 'self', or 'none'.
    """
    num_target = len(target_embeddings)
    if num_target == 0:
        return np.array([]), 0.0, []

    # Initialize tracking
    max_sim_ref = np.zeros(num_target)
    ref_match_indices = np.full(num_target, -1, dtype=int)
    
    max_sim_self = np.zeros(num_target)
    self_match_indices = np.full(num_target, -1, dtype=int)

    # 1. Compare against Reference History (if exists)
    if reference_embeddings is not None and len(reference_embeddings) > 0:
        sim_vs_ref = np.dot(target_embeddings, reference_embeddings.T)
        max_sim_ref = np.max(sim_vs_ref, axis=1)
        ref_match_indices = np.argmax(sim_vs_ref, axis=1)

    # 2. Iterative Self-Comparison (Target vs Previous Target)
    sim_vs_self = np.dot(target_embeddings, target_embeddings.T)
    
    for i in range(1, num_target):
        # Compare unit i against all previous units 0...i-1
        previous_sims = sim_vs_self[i, :i]
        if len(previous_sims) > 0:
            max_sim_self[i] = np.max(previous_sims)
            self_match_indices[i] = np.argmax(previous_sims)
            
    # 3. Combine: Find best match source (Ref or Self)
    final_max_sim = np.zeros(num_target)
    match_data = [] # Stores (type, index)

    for i in range(num_target):
        score_ref = max_sim_ref[i]
        score_self = max_sim_self[i]
        
        # Determine which source is more similar (closer match)
        if score_ref >= score_self:
            final_max_sim[i] = score_ref
            if ref_match_indices[i] != -1:
                match_data.append(('ref', ref_match_indices[i]))
            else:
                match_data.append(('none', -1))
        else:
            final_max_sim[i] = score_self
            match_data.append(('self', self_match_indices[i]))

    # --- Forgiveness Logic (Noise Gate) ---
    # User Request: Decrease Threshold -> More Forgiving.
    # We interpret threshold as "Novelty Threshold".
    # If Novelty Threshold is 0.2, we ignore similarity unless it's very high (Sim > 0.8).
    # If Novelty Threshold is 0.8, we ignore similarity only if it's very low (Sim < 0.2).
    
    similarity_cutoff = 1.0 - threshold
    
    # If similarity is below the cutoff, we treat it as "Noise" (0.0 Similarity -> 1.0 Novelty)
    # This makes the metric forgiving for partial matches.
    final_max_sim[final_max_sim < similarity_cutoff] = 0.0
    
    # Novelty is the inverse of similarity
    novelty_scores = 1 - final_max_sim
    novelty_scores = np.clip(novelty_scores, 0, 1)
    
    # Calculate Weighted NovAScore
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
    """
    Get individual word embeddings and their contributions to the sentence.
    Uses a simple approach: embed each word separately and compare to sentence embedding.
    """
    if model is None:
        return None, None, None
    
    # Tokenize into words
    words = re.findall(r'\b\w+\b', sentence.lower())
    if len(words) == 0:
        return None, None, None
    
    # Get sentence embedding
    sentence_emb = model.encode([sentence])[0]
    sentence_emb = sentence_emb / (np.linalg.norm(sentence_emb) + 1e-9)
    
    # Get word embeddings
    word_embs = model.encode(words)
    norms = np.linalg.norm(word_embs, axis=1, keepdims=True)
    norms[norms == 0] = 1
    word_embs = word_embs / norms
    
    # Calculate contribution (similarity to sentence embedding)
    contributions = np.dot(word_embs, sentence_emb)
    
    return words, word_embs, contributions

def calculate_word_similarity_matrix(words1, embs1, words2, embs2):
    """Calculate pairwise word similarities between two sentences."""
    if embs1 is None or embs2 is None:
        return None
    similarity = np.dot(embs1, embs2.T)
    return similarity

# --- Ollama Integration ---
def decompose_with_ollama(text, model="qwen:8b", base_url="http://localhost:11434"):
    """
    Connects to a local Ollama instance to decompose text.
    Uses 'Atomic Content Units' design.
    """
    prompt = f"""
    Task: Decompose the following text into a list of Atomic Content Units (ACUs). 
    An ACU is a short, standalone statement containing a single piece of information.
    Output ONLY a valid JSON list of strings. Do not add any conversational text.
    
    Text: "{text[:3000]}"
    """
    
    payload = {
        "model": model,
        "prompt": prompt,
        "stream": False,
        "format": "json"
    }
    
    try:
        response = requests.post(f"{base_url}/api/generate", json=payload, timeout=30)
        response.raise_for_status()
        result = response.json()
        
        # Parse JSON from response
        try:
            acus = json.loads(result.get("response", "[]"))
            if isinstance(acus, list):
                return [str(a) for a in acus]
            elif isinstance(acus, dict) and "acus" in acus:
                return [str(a) for a in acus["acus"]]
            else:
                return clean_and_split_sentences(text)
        except json.JSONDecodeError:
            return clean_and_split_sentences(text)
            
    except Exception as e:
        st.error(f"Ollama Connection Error: {e}")
        return clean_and_split_sentences(text)

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
    st.header("Global Settings")
    model = load_embedding_model()
    if model is None:
        st.warning("⚠️ Using TF-IDF Fallback")
        
    st.divider()
    st.header("Ollama Settings")
    use_ollama = st.checkbox("Enable Ollama (Local LLM)", value=False, help="Requires Ollama running locally")
    ollama_url = st.text_input("Base URL", "http://localhost:11434")
    ollama_model = st.text_input("Model Name", "qwen:8b")

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

default_ref_text = "Mars is the fourth planet. Rovers have explored the surface."

col_target, col_ref = st.columns(2)

with col_target:
    st.subheader("Target Document")
    target_file = st.file_uploader("Upload Target Document (.txt)", type=['txt'], key="target_upload")
    if target_file:
        target_text = StringIO(target_file.getvalue().decode("utf-8")).read()
    else:
        target_text = st.text_area("Or paste target text:", value=default_target_text, height=200, key="target_text")

with col_ref:
    st.subheader("Reference Document (for NovAScore)")
    ref_file = st.file_uploader("Upload Reference Document (.txt)", type=['txt'], key="ref_upload")
    if ref_file:
        ref_text = StringIO(ref_file.getvalue().decode("utf-8")).read()
    else:
        ref_text = st.text_area("Or paste reference text:", value=default_ref_text, height=200, key="ref_text")

st.divider()

# --- Analysis Settings ---
st.subheader("⚙️ Analysis Settings")
settings_col1, settings_col2 = st.columns(2)

with settings_col1:
    num_sentences = st.slider("DPP: Select k items", 2, 10, 3, help="Number of diverse items to select for DPP")

with settings_col2:
    novelty_threshold = st.slider(
        "NovAScore: Novelty Threshold", 
        min_value=0.0, max_value=1.0, value=0.15, step=0.01,
        help="Controls how distinct text must be to count as Novel. Lower = More Forgiving. Higher = Stricter."
    )

st.divider()

# Tabs for results
tab_main, tab_novascore = st.tabs(["📊 Diversity (DPP) Results", "🔬 Novelty (NovAScore) Results"])

# --- TAB 1: DPP & Compression ---
with tab_main:
    st.subheader("Single Document Diversity Analysis")
    st.caption("Analyzing the **Target Document** for internal diversity and redundancy.")

    if st.button("Calculate Diversity Metrics", key="btn_dpp"):
        input_text = target_text
        sentences = clean_and_split_sentences(input_text)
        if len(sentences) < 2:
            st.error("Need at least 2 sentences.")
        else:
            embeddings = get_embeddings(sentences, model)
            kernel = build_kernel_matrix(embeddings)
            selected = dpp_greedy_selection(kernel, min(num_sentences, len(sentences)))
            
            # Compression
            full_cr, full_sz, comp_sz = calculate_compression_ratio(sentences)
            subset = [sentences[i] for i in selected]
            sub_cr, sub_sz, sub_comp_sz = calculate_compression_ratio(subset)
            
            # Display Results
            c1, c2 = st.columns(2)
            with c1:
                st.success("DPP Selection")
                for s in subset: st.write(f"- {s}")
                
                # Show score
                subset_matrix = kernel[np.ix_(selected, selected)]
                det_score = np.linalg.det(subset_matrix) if len(selected) > 0 else 0
                st.metric("Subset Volume (Determinant)", f"{det_score:.4f}")
                
            with c2:
                st.subheader("Metrics Comparison")
                st.metric("Full Doc Compression Ratio", f"{full_cr:.2f}")
                st.metric("Subset Compression Ratio", f"{sub_cr:.2f}")
                
                if full_cr > 0:
                    improvement = ((full_cr - sub_cr)/full_cr)*100
                    st.caption(f"Redundancy reduced by {improvement:.1f}% in subset")

            # --- Visualizations (Restored) ---
            st.divider()
            st.subheader("The Mathematics Under the Hood")
            
            vis_tab1, vis_tab2 = st.tabs(["Similarity Matrix (The Kernel)", "Geometric Visualization"])
            
            with vis_tab1:
                st.write("This is the Kernel Matrix $L$. Darker red = More Similar (Redundant).")
                st.write("The DPP algorithm tries to pick rows/cols that are **not** highly correlated with each other.")
                
                # Highlight selected indices in the heatmap labels
                labels = [f"S{i}: {s[:20]}..." for i, s in enumerate(sentences)]
                
                fig = px.imshow(
                    kernel,
                    labels=dict(x="Sentence", y="Sentence", color="Similarity"),
                    x=labels,
                    y=labels,
                    color_continuous_scale="RdBu_r",
                    zmin=-1, zmax=1
                )
                
                st.plotly_chart(fig, use_container_width=True)
                st.caption("A value of 1.0 (Dark Red) means sentences are identical. 0.0 means they are orthogonal (unrelated).")

            with vis_tab2:
                st.write("### Why Determinants?")
                st.write("If we selected highly similar sentences, the matrix would look like this (Rank Deficient):")
                
                # Demo of a bad selection
                if len(sentences) >= 2:
                    # Find most similar pair
                    # Use copy to avoid modifying original kernel for other tabs
                    temp_kernel = kernel.copy()
                    np.fill_diagonal(temp_kernel, -1) # Ignore self
                    i, j = np.unravel_index(np.argmax(temp_kernel), temp_kernel.shape)
                    
                    bad_subset = [i, j]
                    bad_matrix = kernel[np.ix_(bad_subset, bad_subset)]
                    bad_det = np.linalg.det(bad_matrix)
                    
                    st.code(f"""
                    Most Similar Pair:
                    1. {sentences[i]}
                    2. {sentences[j]}
                    
                    Matrix:
                    [[{bad_matrix[0,0]:.2f}, {bad_matrix[0,1]:.2f}]
                     [{bad_matrix[1,0]:.2f}, {bad_matrix[1,1]:.2f}]]
                     
                    Determinant (Volume) = {bad_det:.6f} (Approaching Zero!)
                    """)
                    
                    st.write("Because the volume is near zero, the probability of selecting this pair in a DPP is tiny.")

# --- TAB 2: NovAScore ---
with tab_novascore:
    st.subheader("NovAScore: Novelty Evaluation")
    st.markdown("""
    Based on *Ai et al. (2024)*, NovAScore measures how much **new** information a document contributes compared to a reference.
    
    **Workflow:**
    1. **Decompose:** Break text into Atomic Content Units (ACUs). (Uses Ollama or Sentence Splitting fallback)
    2. **Embed:** Convert Units to vectors.
    3. **Score:** Iteratively calculate novelty. A unit is penalized if it matches Reference History OR any previous unit in the target text.
    """)
    st.caption("Comparing **Target Document** against **Reference Document**.")
    
    if st.button("Calculate NovAScore", key="btn_nova"):
        t_text = target_text
        r_text = ref_text
        with st.spinner("Analyzing Novelty..."):
            # 1. Decomposition
            if use_ollama:
                st.info(f"Connecting to Ollama ({ollama_model}) for Atomic Extraction...")
                t_units = decompose_with_ollama(t_text, ollama_model, ollama_url)
                r_units = decompose_with_ollama(r_text, ollama_model, ollama_url)
            else:
                t_units = clean_and_split_sentences(t_text)
                r_units = clean_and_split_sentences(r_text)
            
            # 2. Embeddings
            t_embs = get_embeddings(t_units, model)
            r_embs = get_embeddings(r_units, model)
            
            # 3. Novelty Calculation
            weights = np.ones(len(t_units)) 
            novelty_scores, final_score, match_data = novascore_calculation(t_embs, r_embs, weights, threshold=novelty_threshold)
            
            # Prepare Match Visualization Data
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

            # Store results in session state
            st.session_state['nova_results'] = {
                't_units': t_units,
                'r_units': r_units,
                't_embs': t_embs,
                'r_embs': r_embs,
                'novelty_scores': novelty_scores,
                'final_score': final_score,
                'match_locations': match_locations,
                'matched_content': matched_content
            }
    
    # Display results from session state
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
        
        st.caption("Using Fast Mode (Sentence Splitting). Enable Ollama in sidebar for smarter decomposition.")
        
        col_u1, col_u2 = st.columns(2)
        with col_u1: st.write(f"**Target Units:** {len(t_units)}")
        with col_u2: st.write(f"**Reference Units:** {len(r_units)}")

        # 4. Display Results
        st.divider()
        m1, m2 = st.columns([1,3])
        
        with m1:
            st.metric("NovAScore (Novelty)", f"{final_score:.2%}")
            
            if final_score > 0.7:
                st.success("High Novelty")
            elif final_score < 0.3:
                st.error("Low Novelty (Redundant)")
            else:
                st.warning("Moderate Novelty")
                
        with m2:
            st.subheader("Unit Breakdown")
            
            df = pd.DataFrame({
                "Content Unit (ACU)": t_units,
                "Novelty Score": novelty_scores,
                "Match Source": match_locations,
                "Closest Match Text": matched_content
            })
            
            st.dataframe(
                df,
                column_config={
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
                use_container_width=True
            )
            
            st.caption("Novelty Score: 0.0 = Information exists in Reference or appeared earlier in document. 1.0 = Completely new information.")

        # --- Embedding Visualizations ---
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
                # Target vs Reference similarity
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
                
                # Self-similarity within target
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
                # Combine all embeddings
                all_embs = np.vstack([t_embs, r_embs]) if len(r_embs) > 0 else t_embs
                all_labels = t_units + r_units if len(r_embs) > 0 else t_units
                all_types = ['Target'] * len(t_units) + ['Reference'] * len(r_units)
                all_ids = [f"T{i+1}" for i in range(len(t_units))] + [f"R{i+1}" for i in range(len(r_units))]
                
                # Add novelty scores for coloring
                all_novelty = list(novelty_scores) + [None] * len(r_units)
                
                # Reduce dimensions
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
                # Let user pick two sentences to compare
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
                    
                    # Get word embeddings
                    words1, embs1, contrib1 = get_word_embeddings(sent1, model)
                    words2, embs2, contrib2 = get_word_embeddings(sent2, model)
                    
                    if words1 is not None and words2 is not None and len(words1) > 0 and len(words2) > 0:
                        # Calculate word-word similarity
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
                        
                        # Show word contributions
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
                        
                        # Highlight strongest connections
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