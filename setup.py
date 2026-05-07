from setuptools import setup, find_packages

setup(
    name="bib-checker",
    version="0.2.0",
    packages=find_packages(),
    install_requires=[
        # Core: enough to run on .tex + .bib with TF-IDF alignment.
        "requests>=2.28",
        "scikit-learn>=1.2",
        "rich>=13.0",
        "numpy>=1.24",
    ],
    extras_require={
        # Embedding-based alignment (semantic similarity, sentence-level)
        "embeddings": [
            "sentence-transformers>=2.7",
        ],
        # Local NLI claim-inversion scorer (cross-encoder)
        "nli": [
            "sentence-transformers>=2.7",
        ],
        # PDF input via GROBID (requires Docker for the GROBID service itself)
        "pdf": [
            "lxml>=5.0",
        ],
        # Browser GUI
        "gui": [
            "streamlit>=1.30",
        ],
        # Large bib libraries (>10k entries). Optional even at scale; numpy is
        # plenty fast under that threshold.
        "faiss": [
            "faiss-cpu>=1.7",
        ],
        # Everything except faiss (which most users don't need)
        "all": [
            "sentence-transformers>=2.7",
            "lxml>=5.0",
            "streamlit>=1.30",
        ],
    },
    entry_points={
        "console_scripts": [
            "bib-checker=bib_checker.cli:main",
            "bib-checker-gui=bib_checker.gui:launch",
        ],
    },
    python_requires=">=3.9",
)
