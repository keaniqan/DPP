7. Compression-Based Metrics: The Homogenization Score
A final, highly efficient category of metrics relies on the principles of Algorithmic Information Theory. The core insight is that redundancy is compressible. A text that explores new things is essentially incompressible (high Kolmogorov complexity).39
7.1 Compression Ratio
The simplest metric is the Compression Ratio:

If a text repeats the same fact, a standard compression algorithm (like LZ77 used in gzip) will replace subsequent occurrences with back-references.
Paraphrasing: "The dog jumped. The canine leaped." Standard gzip struggles here because "dog" and "canine" are different strings.
Solution: Semantic Compression.
7.2 Semantic Compression and Homogenization
To make compression work for paraphrasing, we preprocess the text.
Cluster: Map all words/phrases to semantic cluster IDs (e.g., "dog", "canine", "hound"  Cluster_101).
Encode: Rewrite the text as a sequence of Cluster IDs.
Compress: Compress the sequence of IDs.
100 Paraphrases: The sequence of IDs will be C_101, C_205, C_101, C_205... This is highly repetitive and highly compressible.
100 Facts: The sequence of IDs will be random. Incompressible.
This leads to the Homogenization Score, which quantifies how much "easier" the text is to compress than a random baseline. A high homogenization score indicates low knowledge generation (high redundancy).42