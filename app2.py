# # --- Core Math: NovAScore Implementation ---
# def novascore_calculation(target_embeddings, reference_embeddings, weights=None, threshold=0.15):
#     """
#     Calculates NovAScore using vector similarity with Iterative Self-Comparison.
    
#     Args:
#         threshold (float): Novelty Threshold (0.0 - 1.0). 
#                            Similarities below (1 - threshold) are ignored (treated as 0.0).
#                            Lower threshold -> Higher Ignore Cutoff -> More Forgiving.
    
#     Returns:
#         novelty_scores: Array of novelty scores (0.0 to 1.0)
#         weighted_score: Single weighted average score
#         match_data: List of tuples (match_type, match_index) indicating where the closest match was found.
#                     match_type is 'ref', 'self', or 'none'.
#     """
#     num_target = len(target_embeddings)
#     if num_target == 0:
#         return np.array([]), 0.0, []

#     # Initialize tracking
#     max_sim_ref = np.zeros(num_target)
#     ref_match_indices = np.full(num_target, -1, dtype=int)
    
#     max_sim_self = np.zeros(num_target)
#     self_match_indices = np.full(num_target, -1, dtype=int)

#     # 1. Compare against Reference History (if exists)
#     if reference_embeddings is not None and len(reference_embeddings) > 0:
#         sim_vs_ref = np.dot(target_embeddings, reference_embeddings.T)
#         max_sim_ref = np.max(sim_vs_ref, axis=1)
#         ref_match_indices = np.argmax(sim_vs_ref, axis=1)

#     # 2. Iterative Self-Comparison (Target vs Previous Target)
#     sim_vs_self = np.dot(target_embeddings, target_embeddings.T)
    
#     for i in range(1, num_target):
#         # Compare unit i against all previous units 0...i-1
#         previous_sims = sim_vs_self[i, :i]
#         if len(previous_sims) > 0:
#             max_sim_self[i] = np.max(previous_sims)
#             self_match_indices[i] = np.argmax(previous_sims)
            
#     # 3. Combine: Find best match source (Ref or Self)
#     final_max_sim = np.zeros(num_target)
#     match_data = [] # Stores (type, index)

#     for i in range(num_target):
#         score_ref = max_sim_ref[i]
#         score_self = max_sim_self[i]
        
#         # Determine which source is more similar (closer match)
#         if score_ref >= score_self:
#             final_max_sim[i] = score_ref
#             if ref_match_indices[i] != -1:
#                 match_data.append(('ref', ref_match_indices[i]))
#             else:
#                 match_data.append(('none', -1))
#         else:
#             final_max_sim[i] = score_self
#             match_data.append(('self', self_match_indices[i]))

#     # --- Forgiveness Logic (Noise Gate) ---
#     # User Request: Decrease Threshold -> More Forgiving.
#     # We interpret threshold as "Novelty Threshold".
#     # If Novelty Threshold is 0.2, we ignore similarity unless it's very high (Sim > 0.8).
#     # If Novelty Threshold is 0.8, we ignore similarity only if it's very low (Sim < 0.2).
    
#     similarity_cutoff = 1.0 - threshold
    
#     # If similarity is below the cutoff, we treat it as "Noise" (0.0 Similarity -> 1.0 Novelty)
#     # This makes the metric forgiving for partial matches.
#     final_max_sim[final_max_sim < similarity_cutoff] = 0.0
    
#     # Novelty is the inverse of similarity
#     novelty_scores = 1 - final_max_sim
#     novelty_scores = np.clip(novelty_scores, 0, 1)
    
#     # Calculate Weighted NovAScore
#     if weights is None:
#         weights = np.ones(num_target)
        
#     weighted_score = np.average(novelty_scores, weights=weights)
    
#     return novelty_scores, weighted_score, match_data

# # --- Ollama Integration ---
# def decompose_with_ollama(text, model="qwen:8b", base_url="http://localhost:11434"):
#     """
#     Connects to a local Ollama instance to decompose text.
#     Uses 'Atomic Content Units' design.
#     """
#     prompt = f"""
#     Task: Decompose the following text into a list of Atomic Content Units (ACUs). 
#     An ACU is a short, standalone statement containing a single piece of information.
#     Output ONLY a valid JSON list of strings. Do not add any conversational text.
    
#     Text: "{text[:3000]}"
#     """
    
#     payload = {
#         "model": model,
#         "prompt": prompt,
#         "stream": False,
#         "format": "json"
#     }
    
#     try:
#         response = requests.post(f"{base_url}/api/generate", json=payload, timeout=30)
#         response.raise_for_status()
#         result = response.json()
        
#         # Parse JSON from response
#         try:
#             acus = json.loads(result.get("response", "[]"))
#             if isinstance(acus, list):
#                 return [str(a) for a in acus]
#             elif isinstance(acus, dict) and "acus" in acus:
#                 return [str(a) for a in acus["acus"]]
#             else:
#                 return clean_and_split_sentences(text)
#         except json.JSONDecodeError:
#             return clean_and_split_sentences(text)
            
#     except Exception as e:
#         st.error(f"Ollama Connection Error: {e}")
#         return clean_and_split_sentences(text)

# # --- UI Layout ---

# st.title("🧬 Diversity & Novelty Explorer")
# st.markdown("""
# Evaluate text using three mathematical frameworks:
# 1. **DPP (Determinantal Point Process):** Geometric diversity within a single document.
# 2. **Semantic Compression:** Information theoretic redundancy.
# 3. **NovAScore:** Novelty quantification against historical/reference documents.
# """)

# # Tabs for different modes
# tab_main, tab_novascore = st.tabs(["Diversity (DPP)", "Novelty (NovAScore)"])

# # --- Sidebar ---
# with st.sidebar:
#     st.header("Global Settings")
#     model = load_embedding_model()
#     if model is None:
#         st.warning("⚠️ Using TF-IDF Fallback")
        
#     st.divider()
#     st.header("Ollama Settings")
#     use_ollama = st.checkbox("Enable Ollama (Local LLM)", value=False, help="Requires Ollama running locally")
#     ollama_url = st.text_input("Base URL", "http://localhost:11434")
#     ollama_model = st.text_input("Model Name", "qwen:8b")

# # --- TAB 1: DPP & Compression ---
# with tab_main:
#     st.subheader("Single Document Diversity")
    
#     uploaded_file = st.file_uploader("Upload Target Document (.txt)", type=['txt'], key="dpp_upload")
    
#     default_text = """
#     Artificial Intelligence is transforming the world. 
#     Machine learning models are becoming increasingly accurate.
#     AI systems are reshaping industries across the globe.
#     Deep learning is a subset of machine learning.
#     Oranges are a rich source of Vitamin C.
#     Citrus fruits are excellent for health.
#     The Eiffel Tower is located in Paris.
#     Paris is the capital of France.
#     """
    
#     if uploaded_file:
#         stringio = StringIO(uploaded_file.getvalue().decode("utf-8"))
#         input_text = stringio.read()
#     else:
#         input_text = st.text_area("Or paste text:", value=default_text, height=150)

#     num_sentences = st.slider("Select k items", 2, 10, 3)

#     if st.button("Calculate Diversity Metrics", key="btn_dpp"):
#         sentences = clean_and_split_sentences(input_text)
#         if len(sentences) < 2:
#             st.error("Need at least 2 sentences.")
#         else:
#             embeddings = get_embeddings(sentences, model)
#             kernel = build_kernel_matrix(embeddings)
#             selected = dpp_greedy_selection(kernel, min(num_sentences, len(sentences)))
            
#             # Compression
#             full_cr, full_sz, comp_sz = calculate_compression_ratio(sentences)
#             subset = [sentences[i] for i in selected]
#             sub_cr, sub_sz, sub_comp_sz = calculate_compression_ratio(subset)
            
#             # Display Results
#             c1, c2 = st.columns(2)
#             with c1:
#                 st.success("DPP Selection")
#                 for s in subset: st.write(f"- {s}")
                
#                 # Show score
#                 subset_matrix = kernel[np.ix_(selected, selected)]
#                 det_score = np.linalg.det(subset_matrix) if len(selected) > 0 else 0
#                 st.metric("Subset Volume (Determinant)", f"{det_score:.4f}")
                
#             with c2:
#                 st.subheader("Metrics Comparison")
#                 st.metric("Full Doc Compression Ratio", f"{full_cr:.2f}")
#                 st.metric("Subset Compression Ratio", f"{sub_cr:.2f}")
                
#                 if full_cr > 0:
#                     improvement = ((full_cr - sub_cr)/full_cr)*100
#                     st.caption(f"Redundancy reduced by {improvement:.1f}% in subset")

#             # --- Visualizations (Restored) ---
#             st.divider()
#             st.subheader("The Mathematics Under the Hood")
            
#             vis_tab1, vis_tab2 = st.tabs(["Similarity Matrix (The Kernel)", "Geometric Visualization"])
            
#             with vis_tab1:
#                 st.write("This is the Kernel Matrix $L$. Darker red = More Similar (Redundant).")
#                 st.write("The DPP algorithm tries to pick rows/cols that are **not** highly correlated with each other.")
                
#                 # Highlight selected indices in the heatmap labels
#                 labels = [f"S{i}: {s[:20]}..." for i, s in enumerate(sentences)]
                
#                 fig = px.imshow(
#                     kernel,
#                     labels=dict(x="Sentence", y="Sentence", color="Similarity"),
#                     x=labels,
#                     y=labels,
#                     color_continuous_scale="RdBu_r",
#                     zmin=-1, zmax=1
#                 )
                
#                 st.plotly_chart(fig, use_container_width=True)
#                 st.caption("A value of 1.0 (Dark Red) means sentences are identical. 0.0 means they are orthogonal (unrelated).")

#             with vis_tab2:
#                 st.write("### Why Determinants?")
#                 st.write("If we selected highly similar sentences, the matrix would look like this (Rank Deficient):")
                
#                 # Demo of a bad selection
#                 if len(sentences) >= 2:
#                     # Find most similar pair
#                     # Use copy to avoid modifying original kernel for other tabs
#                     temp_kernel = kernel.copy()
#                     np.fill_diagonal(temp_kernel, -1) # Ignore self
#                     i, j = np.unravel_index(np.argmax(temp_kernel), temp_kernel.shape)
                    
#                     bad_subset = [i, j]
#                     bad_matrix = kernel[np.ix_(bad_subset, bad_subset)]
#                     bad_det = np.linalg.det(bad_matrix)
                    
#                     st.code(f"""
#                     Most Similar Pair:
#                     1. {sentences[i]}
#                     2. {sentences[j]}
                    
#                     Matrix:
#                     [[{bad_matrix[0,0]:.2f}, {bad_matrix[0,1]:.2f}]
#                      [{bad_matrix[1,0]:.2f}, {bad_matrix[1,1]:.2f}]]
                     
#                     Determinant (Volume) = {bad_det:.6f} (Approaching Zero!)
#                     """)
                    
#                     st.write("Because the volume is near zero, the probability of selecting this pair in a DPP is tiny.")

# # --- TAB 2: NovAScore ---
# with tab_novascore:
#     st.subheader("NovAScore: Novelty Evaluation")
#     st.markdown("""
#     Based on *Ai et al. (2024)*, NovAScore measures how much **new** information a document contributes compared to a reference.
    
#     **Workflow:**
#     1. **Decompose:** Break text into Atomic Content Units (ACUs). (Uses Ollama or Sentence Splitting fallback)
#     2. **Embed:** Convert Units to vectors.
#     3. **Score:** Iteratively calculate novelty. A unit is penalized if it matches Reference History OR any previous unit in the target text.
#     """)
    
#     col_input, col_ref = st.columns(2)
    
#     with col_input:
#         target_file = st.file_uploader("1. Target Document (New)", type=['txt'], key="nova_target")
#         target_txt_val = st.text_area("Target Text", value="New discovery in Mars rover data indicates water flow.", height=150, key="nova_txt_target")
        
#     with col_ref:
#         ref_file = st.file_uploader("2. Reference Document (History)", type=['txt'], key="nova_ref")
#         ref_txt_val = st.text_area("Reference Text", value="Mars is the fourth planet. Rovers have explored the surface.", height=150, key="nova_txt_ref")

#     # Input Logic
#     t_text = target_txt_val
#     if target_file: t_text = StringIO(target_file.getvalue().decode("utf-8")).read()
    
#     r_text = ref_txt_val
#     if ref_file: r_text = StringIO(ref_file.getvalue().decode("utf-8")).read()

#     st.divider()
    
#     # New Control for Threshold (Renamed and Logic Inverted)
#     novelty_threshold = st.slider(
#         "Novelty Threshold (Uniqueness Required)", 
#         min_value=0.0, max_value=1.0, value=0.15, step=0.01,
#         help="Controls how distinct text must be to count as Novel. Lower = More Forgiving (Ignores more similarity). Higher = Stricter (Counts even slight similarity)."
#     )
    
#     if st.button("Calculate NovAScore", key="btn_nova"):
#         with st.spinner("Analyzing Novelty..."):
#             # 1. Decomposition
#             if use_ollama:
#                 st.info(f"Connecting to Ollama ({ollama_model}) for Atomic Extraction...")
#                 t_units = decompose_with_ollama(t_text, ollama_model, ollama_url)
#                 r_units = decompose_with_ollama(r_text, ollama_model, ollama_url)
#             else:
#                 st.caption("Using Fast Mode (Sentence Splitting). Enable Ollama in sidebar for smarter decomposition.")
#                 t_units = clean_and_split_sentences(t_text)
#                 r_units = clean_and_split_sentences(r_text)
                
#             col_u1, col_u2 = st.columns(2)
#             with col_u1: st.write(f"**Target Units:** {len(t_units)}")
#             with col_u2: st.write(f"**Reference Units:** {len(r_units)}")
            
#             # 2. Embeddings
#             t_embs = get_embeddings(t_units, model)
#             r_embs = get_embeddings(r_units, model)
            
#             # 3. Novelty Calculation
#             weights = np.ones(len(t_units)) 
#             # Pass the threshold to the calculation
#             novelty_scores, final_score, match_data = novascore_calculation(t_embs, r_embs, weights, threshold=novelty_threshold)
            
#             # Prepare Match Visualization Data
#             matched_content = []
#             match_locations = []

#             for score, (m_type, m_idx) in zip(novelty_scores, match_data):
#                 if m_idx == -1:
#                      matched_content.append("None")
#                      match_locations.append("New Concept")
#                 else:
#                     if m_type == 'ref':
#                         # Guard against index errors
#                         if m_idx < len(r_units):
#                             matched_content.append(r_units[m_idx])
#                         else:
#                             matched_content.append("Ref Error")
#                         match_locations.append("Reference Doc")
#                     elif m_type == 'self':
#                         if m_idx < len(t_units):
#                             matched_content.append(t_units[m_idx])
#                         else:
#                             matched_content.append("Self Error")
#                         match_locations.append(f"Self (Unit {m_idx + 1})")

#             # 4. Display Results
#             st.divider()
#             m1, m2 = st.columns([1,3])
            
#             with m1:
#                 st.metric("NovAScore (Novelty)", f"{final_score:.2%}")
                
#                 if final_score > 0.7:
#                     st.success("High Novelty")
#                 elif final_score < 0.3:
#                     st.error("Low Novelty (Redundant)")
#                 else:
#                     st.warning("Moderate Novelty")
                    
#             with m2:
#                 st.subheader("Unit Breakdown")
                
#                 df = pd.DataFrame({
#                     "Content Unit (ACU)": t_units,
#                     "Novelty Score": novelty_scores,
#                     "Match Source": match_locations,
#                     "Closest Match Text": matched_content
#                 })
                
#                 st.dataframe(
#                     df,
#                     column_config={
#                         "Novelty Score": st.column_config.ProgressColumn(
#                             "Novelty Score",
#                             help="0.0 = Redundant (found in history or self-repetitive), 1.0 = Completely New",
#                             format="%.2f",
#                             min_value=0,
#                             max_value=1,
#                         ),
#                         "Closest Match Text": st.column_config.TextColumn(
#                             "Closest Match Text",
#                             help="The text that is most similar to the target unit (causing the redundancy)",
#                             width="large"
#                         )
#                     },
#                     use_container_width=True
#                 )
                
#                 st.caption("Novelty Score: 0.0 = Information exists in Reference or appeared earlier in document. 1.0 = Completely new information.")