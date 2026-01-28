A. The Effective Rank (Entropy of Eigenvalues)Instead of the determinant (product of eigenvalues), look at the distribution of eigenvalues.High Knowledge: Eigenvalues are roughly equal (flat distribution).Paraphrasing: One huge eigenvalue (the repeated topic) and many tiny eigenvalues (the noise).You can calculate the Effective Rank ($R_{eff}$):$$H(\lambda) = -\sum p_i \log p_i \quad \text{where} \quad p_i = \frac{\lambda_i}{\sum \lambda_k}$$$$R_{eff} = e^{H(\lambda)}$$If you have 100 paraphrases, $R_{eff}$ will stay close to 1, no matter how many sentences you add. If you have 100 facts, it will grow to 100.B. The Vendi ScoreThe Vendi Score is a recently proposed metric specifically designed to fix exactly the problem you are seeing. It is effectively the same as the exponential of the spectral entropy (Effective Rank) applied to the similarity kernel. It interprets the limit of the diversity: "This document contains $X$ distinct concepts."3. Updated Python ImplementationHere is how to calculate the Effective Rank to get the "Diminishing Returns" curve you were expecting.Pythonimport numpy as np
from sentence_transformers import SentenceTransformer

def effective_rank_score(sentences):
    model = SentenceTransformer('all-MiniLM-L6-v2')
    embeddings = model.encode(sentences)
    
    # 1. Compute Kernel Matrix (L)
    # Normalize embeddings first
    embeddings = embeddings / np.linalg.norm(embeddings, axis=1, keepdims=True)
    L = embeddings @ embeddings.T
    
    # 2. Get Eigenvalues
    # We use eigh because L is symmetric
    eigenvalues = np.linalg.eigvalsh(L)
    
    # Filter out tiny floating point noise (negatives close to zero)
    eigenvalues = eigenvalues[eigenvalues > 1e-10]
    
    # 3. Normalize Eigenvalues to create a probability distribution
    total_eigen_sum = np.sum(eigenvalues)
    probabilities = eigenvalues / total_eigen_sum
    
    # 4. Compute Spectral Entropy
    entropy = -np.sum(probabilities * np.log(probabilities))
    
    # 5. Effective Rank (The number of "effective" unique ideas)
    effective_rank = np.exp(entropy)
    
    return effective_rank

# Test
# If you run this on your 305 sentences:
# - LogDet would show ~85 (Linear growth)
# - Effective Rank will likely plateau at ~5-10 (Diminishing returns)
Summary for your AnalysisLog-Det tells you: "How much total information (signal + noise) is in the file?" -> Good for compression estimation.Normalized Score tells you: "How efficient is the text?" -> Good for quality control.Effective Rank tells you: "How many actual unique concepts are there?" -> Good for answering your specific question.

1. The Core Concept: Eigenvalues as ProbabilitiesTo switch from an "Extensive" metric (like Volume) to an "Intensive" metric (like Diversity), we must treat the eigenvalues of your similarity matrix as a probability distribution.In the Vendi Score framework, we define the matrix $K$ such that every diagonal element is $1$ (a sentence is perfectly similar to itself). When we divide this matrix by the number of sentences $n$, the trace (sum of diagonal elements) becomes $1$.$$\text{Trace}(K/n) = \frac{1}{n} \sum_{i=1}^n K_{ii} = \frac{1}{n} \cdot n = 1$$Since the sum of eigenvalues equals the trace, the eigenvalues $\lambda_1, \dots, \lambda_n$ of the matrix $K/n$ now sum to exactly $1$. This means we can treat them like probabilities ($p_i$) in a distribution.2. The Calculation ProcessHere is the step-by-step process to calculate the Vendi Score for your "100 Facts vs. 100 Paraphrases" scenario.Step A: Construct the Kernel Matrix ($K$)You begin with your embeddings $X$ (shape $100 \times d$). You compute the cosine similarity matrix $K$.$K_{ij} = \text{similarity}(x_i, x_j)$Crucially, $K_{ii} = 1$.Step B: Normalize the MatrixDivide the entire matrix by $n$ (the number of sentences).$$\bar{K} = \frac{K}{n}$$Step C: Compute Spectral EntropyCalculate the eigenvalues $\lambda$ of $\bar{K}$. Since $\sum \lambda_i = 1$ and $\lambda_i \ge 0$, we can calculate the Shannon Entropy of this spectrum.$$H(\lambda) = - \sum_{i=1}^n \lambda_i \log(\lambda_i)$$Step D: Exponentiate (The Vendi Score)The final score is the exponential of the entropy.$$\text{VS} = \exp(H(\lambda)) = \exp\left( - \sum_{i=1}^n \lambda_i \log(\lambda_i) \right)$$3. Scenario Analysis: Why it PlateausHere is how this math plays out in your two scenarios.Scenario 1: 100 Paraphrases (High Redundancy)The Matrix: Since every sentence is a paraphrase, they are all highly similar. $K$ looks like a matrix of $1$s.The Eigenvalues: A matrix of all $1$s has a "rank" of 1. It has one massive eigenvalue $\lambda_1 = 1.0$ and all other eigenvalues are $0$.The Entropy:$$H = -(1.0 \log(1.0) + 0 + \dots) = 0$$The Score:$$\text{VS} = \exp(0) = \mathbf{1.0}$$Interpretation: Even though you have 100 sentences, the metric effectively says, "You have 1 unique idea." This creates the perfect plateau you were looking for.Scenario 2: 100 Unique Facts (High Diversity)The Matrix: The sentences are distinct (orthogonal). $K$ looks like an Identity matrix (1s on diagonal, 0s elsewhere).The Eigenvalues: The matrix $\bar{K} = \frac{I}{100}$ has 100 identical eigenvalues, each equal to $\frac{1}{100}$.The Entropy:$$H = -\sum_{i=1}^{100} \frac{1}{100} \log\left(\frac{1}{100}\right) = -100 \cdot \left[ \frac{1}{100} (-\log(100)) \right] = \log(100)$$The Score:$$\text{VS} = \exp(\log(100)) = \mathbf{100.0}$$Interpretation: The metric says, "You have 100 unique ideas."4. Python Implementation (Vendi Score)This implementation uses the exact definition provided in the Vendi Score research paper.Pythonimport numpy as np
import torch
from sentence_transformers import SentenceTransformer

def vendi_score(sentences, model_name='all-MiniLM-L6-v2'):
    # 1. Encode sentences to get X
    model = SentenceTransformer(model_name)
    embeddings = model.encode(sentences)
    
    # 2. Compute Cosine Similarity Matrix K
    # Normalize embeddings to unit length first
    embeddings = embeddings / np.linalg.norm(embeddings, axis=1, keepdims=True)
    # K = X * X.T
    K = embeddings @ embeddings.T
    
    # 3. Normalize Matrix by n (number of sentences)
    n = K.shape
    K_bar = K / n
    
    # 4. Compute Eigenvalues of K_bar
    # We use eigvalsh because K is symmetric
    eigenvalues = np.linalg.eigvalsh(K_bar)
    
    # 5. Filter numerical noise (eigenvalues must be positive)
    eigenvalues = eigenvalues[eigenvalues > 1e-10]
    
    # 6. Compute Shannon Entropy
    # Note: Sum of eigenvalues is already approx 1.0 due to step 3
    entropy = -np.sum(eigenvalues * np.log(eigenvalues))
    
    # 7. Vendi Score
    score = np.exp(entropy)
    
    return score

# Example behavior:
# facts = ["Fact 1", "Fact 2", "Fact 3"] -> Score approaches 3.0
# repeats = ["Fact 1", "Fact 1", "Fact 1"] -> Score approaches 1.0
Summary of DifferencesLogDet (DPP): Measures the volume of the semantic space covered. It grows indefinitely as you add data, even if that data is slightly noisy paraphrasing.Vendi Score: Measures the effective number of independent factors. It hits a "hard ceiling" determined by the actual number of unique concepts, regardless of how many paraphrases you generate. This provides the "diminishing returns" curve you expecting.