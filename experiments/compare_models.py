"""
compare_models.py — Wrapper for generate_report.py
"""
from experiments.generate_report import run_all_reporting, generate_comparison_table, plot_comparison_charts, build_final_markdown_report

def run_comparison():
    run_all_reporting()

if __name__ == '__main__':
    run_all_reporting()
