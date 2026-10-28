"""RAVI: Representation-Aware Vector Indexing for deep image embeddings.

Pipeline stages (run with ``python -m ravi <stage> -c <config.yaml>``):

  plan      count spaces / index builds / disk before running anything
  extract   images -> native embeddings for every backbone
  reduce    native -> compact spaces (PCA, random projection, truncation, learned heads)
  gt        exact nearest neighbours (ground truth) for every space
  geometry  intrinsic dimension, hubness, anisotropy, density, clusterability ...
  bench     FAISS Flat / HNSW / IVF-Flat / IVF-PQ: recall, QPS, latency, memory, build time
  collect   merge part files into tables and compute operating points
  recommend leave-one-family-out configuration recommender
  stats     correlations, Friedman / Wilcoxon-Holm tests, bootstrap intervals
  figures   publication figures
  report    Markdown summary of all measured tables (results/report.md)

Every stage is resumable: finished units are skipped on restart.
No stage ever writes a number that was not measured.
"""
__version__ = "0.3.0"
