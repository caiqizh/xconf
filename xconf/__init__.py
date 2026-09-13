"""XConf: eXperiential Confidence estimation (Recall + Reflect).

Black-box, training-free confidence estimation from a model's own graded
past episodes. The package holds the shared infrastructure -- config, LLM
clients, embedding, episode records, prompts, verifiers -- and the method
itself lives in scripts/ (elicit -> grade -> embed -> lesson -> Recall /
Reflect -> evaluate); scripts/common.py carries the frozen recipe
constants (k=50, sigma=.08, per-fold PCA-30 + logistic reweight, 5-fold
out-of-sample evaluation).
"""

__version__ = "1.0.0"
