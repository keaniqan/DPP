import diversity as div

compression = div.compression_ratio("This is a sample text.")
tokens = div.token_patterns("This is a sample text.", 4) 

print("Compression Ratio:", compression)
print("Token Patterns:", tokens)