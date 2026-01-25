import streamlit as st
import numpy as np
import pandas as pd
import plotly.express as px
import re
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
    # Split by standard sentence delimiters
    sentences = re.split(r'(?<!\w\.\w.)(?<![A-Z][a-z]\.)(?<=\.|\?)\s', text)
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
    if model:
        embeddings = model.encode(sentences)
        # Normalize embeddings to unit length for Cosine Similarity
        norms = np.linalg.norm(embeddings, axis=1, keepdims=True)
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
    
    Here we assume Similarity is the Dot Product (Cosine Similarity).
    Alpha allows scaling the magnitude (quality) of items.
    """
    # Cosine similarity matrix (since vectors are normalized)
    # S = X @ X.T
    similarity_matrix = np.dot(embeddings, embeddings.T)
    
    # In this simple implementation, Quality is uniform (1.0).
    # We can scale the matrix to control the expected number of items 
    # if we were doing k-DPP sampling, but for greedy, relative values matter.
    L = similarity_matrix * alpha
    return L

def dpp_greedy_selection(kernel_matrix, k):
    """
    Performs Greedy MAP Inference for a k-DPP.
    
    Goal: Select a subset Y of size k that maximizes det(L_Y).
    
    Algorithm:
    1. Start with empty set Y.
    2. Iteratively add item i that maximizes log_det(L_{Y + i}).
    3. Repeat until k items are selected.
    
    Returns: List of selected indices.
    """
    num_items = kernel_matrix.shape[0]
    
    # If requested k is larger than N, return all
    if k >= num_items:
        return list(range(num_items))

    selected_indices = []
    
    # Create a mask for available items
    available_indices = list(range(num_items))
    
    for _ in range(k):
        best_item = -1
        max_log_det = -np.inf
        
        # Try adding each available item to the current set
        for item in available_indices:
            # Construct temporary subset indices
            current_subset = selected_indices + [item]
            
            # Extract submatrix
            # np.ix_ creates a meshgrid to select the submatrix defined by rows/cols in current_subset
            submatrix = kernel_matrix[np.ix_(current_subset, current_subset)]
            
            # Calculate determinant. 
            # We use pseudo-determinant or handle nearly singular matrices for stability
            # But standard det is usually fine for greedy steps unless identical items exist.
            try:
                # Add tiny noise to diagonal for stability
                d = np.linalg.det(submatrix + np.eye(len(current_subset)) * 1e-6)
                if d > 0:
                    score = np.log(d)
                else:
                    score = -np.inf
            except np.linalg.LinAlgError:
                score = -np.inf
                
            if score > max_log_det:
                max_log_det = score
                best_item = item
        
        if best_item != -1:
            selected_indices.append(best_item)
            available_indices.remove(best_item)
        else:
            # If we can't improve (determinant is 0 due to redundancy), stop early
            break
            
    return selected_indices

def calculate_log_det_score(matrix):
    """Calculates the log-determinant of (L + I) for the full matrix."""
    # We use L + I because det(L) -> Probability of a specific subset
    # det(L + I) -> Normalization constant (sum of all probabilities)
    # It serves as a good proxy for the "Total Information Content"
    sign, logdet = np.linalg.slogdet(matrix + np.eye(matrix.shape[0]))
    return logdet if sign > 0 else 0

# --- UI Layout ---

st.title("🧬 DPP: The Mathematics of Diversity")
st.markdown("""
This tool uses a **Determinantal Point Process (DPP)** to extract the most semantically diverse sentences from a document.
Unlike standard ranking (which picks the "best" items), DPPs explicitly model the trade-off between **Quality** and **Diversity** using the volume of the semantic space.
""")

# Sidebar
with st.sidebar:
    st.header("Settings")
    
    num_sentences = st.slider(
        "Sentences to Select (k)", 
        min_value=2, 
        max_value=10, 
        value=3,
        help="How many diverse sentences do you want to extract?"
    )
    
    st.info("""
    **How it works:**
    1. Text is converted to vectors.
    2. A Kernel Matrix $L$ is built ($L_{ij} = Sim(i,j)$).
    3. The algorithm selects a subset $Y$ that maximizes $\det(L_Y)$.
    4. $\det(L_Y)$ corresponds to the volume of the parallelepiped spanned by the vectors.
    """)

    st.markdown("---")
    st.markdown("**References:**")
    st.markdown("- [Kulesza & Taskar (2012)](https://arxiv.org/abs/1207.6081)")
    st.markdown("- [EMNLP 2023 Paper](https://aclanthology.org/2023.emnlp-main.267.pdf)")

# Main Content
input_text = ""
uploaded_file = st.file_uploader("Upload a document (.txt)", type=['txt'])

# Default Example Text
default_text = """
Artificial Intelligence is transforming the world. 
Machine learning models are becoming increasingly accurate.
AI systems are reshaping industries across the globe.
Deep learning is a subset of machine learning.
Oranges are a rich source of Vitamin C.
Citrus fruits are excellent for health.
The Eiffel Tower is located in Paris.
Paris is the capital of France.
"""

if uploaded_file is not None:
    stringio = StringIO(uploaded_file.getvalue().decode("utf-8"))
    input_text = stringio.read()
else:
    input_text = st.text_area("Or paste text here (using default example):", value=default_text, height=150)

if st.button("Calculate Diversity"):
    with st.spinner("Loading AI Models & Computing Math..."):
        
        # 1. Processing
        sentences = clean_and_split_sentences(input_text)
        
        if len(sentences) < 2:
            st.error("Please enter text with at least 2 sentences.")
        else:
            model = load_embedding_model()
            if model is None:
                st.warning("⚠️ `sentence-transformers` not found. Falling back to TF-IDF (lower quality semantic matching).")
            
            # 2. Embeddings
            embeddings = get_embeddings(sentences, model)
            
            # 3. Kernel Construction
            kernel_matrix = build_kernel_matrix(embeddings)
            
            # 4. Greedy Selection
            selected_indices = dpp_greedy_selection(kernel_matrix, min(num_sentences, len(sentences)))
            
            # --- Results Display ---
            
            col1, col2 = st.columns([1, 1])
            
            with col1:
                st.subheader("🎯 Most Diverse Subset (DPP)")
                for idx in selected_indices:
                    st.success(f"**{sentences[idx]}**")
                
                # Show score
                subset_matrix = kernel_matrix[np.ix_(selected_indices, selected_indices)]
                score = np.linalg.det(subset_matrix)
                st.metric("Subset Volume (Determinant)", f"{score:.4f}")

            with col2:
                st.subheader("🔍 Analysis")
                st.write(f"Total Sentences: {len(sentences)}")
                st.write(f"Embedding Dimensions: {embeddings.shape[1]}")
                
                full_volume = calculate_log_det_score(kernel_matrix)
                st.metric("Total Document Semantic Volume (LogDet)", f"{full_volume:.2f}")

            # --- Visualizations ---
            
            st.divider()
            st.subheader("The Mathematics Under the Hood")
            
            tab1, tab2 = st.tabs(["Similarity Matrix (The Kernel)", "Geometric Visualization"])
            
            with tab1:
                st.write("This is the Kernel Matrix $L$. Darker red = More Similar (Redundant).")
                st.write("The DPP algorithm tries to pick rows/cols that are **not** highly correlated with each other.")
                
                # Highlight selected indices in the heatmap labels
                labels = [f"S{i}: {s[:20]}..." for i, s in enumerate(sentences)]
                
                fig = px.imshow(
                    kernel_matrix,
                    labels=dict(x="Sentence", y="Sentence", color="Similarity"),
                    x=labels,
                    y=labels,
                    color_continuous_scale="RdBu_r",
                    zmin=-1, zmax=1
                )
                
                # Add markers for selected items
                # We can't easily overlay on imshow in plotly express without graph_objects, 
                # but we can rely on the user checking the labels.
                st.plotly_chart(fig, use_container_width=True)
                
                st.caption("A value of 1.0 (Dark Red) means sentences are identical. 0.0 means they are orthogonal (unrelated).")

            with tab2:
                st.write("### Why Determinants?")
                st.write("If we selected highly similar sentences, the matrix would look like this (Rank Deficient):")
                
                # Demo of a bad selection
                if len(sentences) >= 2:
                    # Find most similar pair
                    np.fill_diagonal(kernel_matrix, -1) # Ignore self
                    i, j = np.unravel_index(np.argmax(kernel_matrix), kernel_matrix.shape)
                    np.fill_diagonal(kernel_matrix, 1) # Restore
                    
                    bad_subset = [i, j]
                    bad_matrix = kernel_matrix[np.ix_(bad_subset, bad_subset)]
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

# --- Theory Section (Collapsible) ---
with st.expander("📚 Learn more about the Math"):
    st.markdown(r"""
    ### 1. The Kernel $L$
    We treat our items (sentences) as vectors in a high-dimensional space. The kernel $L$ defines the similarity:
    $$L_{ij} = \langle \mathbf{v}_i, \mathbf{v}_j \rangle$$
    
    ### 2. Probability & Volume
    The probability of selecting a subset $Y$ is proportional to the determinant of the submatrix $L_Y$ restricted to those items:
    $$P(Y) \propto \det(L_Y)$$
    
    ### 3. Why it works
    Geometrically, $\det(L_Y)$ is the **squared volume** of the parallelepiped spanned by the feature vectors of the items in $Y$.
    * **Similar vectors** $\rightarrow$ Small angle $\rightarrow$ Flat parallelepiped $\rightarrow$ **Volume $\approx$ 0**.
    * **Orthogonal vectors** $\rightarrow$ 90° angle $\rightarrow$ Box shape $\rightarrow$ **Volume is Max**.
    """)