"""
knn_tuning.py — Wrapper for tune_knn.py
"""
from experiments.tune_knn import run_knn_tuning, METRIC if hasattr(locals(), 'METRIC') else 'euclidean'

if __name__ == '__main__':
    run_knn_tuning()
