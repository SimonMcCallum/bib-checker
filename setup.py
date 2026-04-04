from setuptools import setup, find_packages

setup(
    name="bib-checker",
    version="0.1.0",
    packages=find_packages(),
    install_requires=[
        "bibtexparser>=2.0.0b",
        "requests>=2.28",
        "scikit-learn>=1.2",
        "rich>=13.0",
    ],
    entry_points={
        "console_scripts": [
            "bib-checker=bib_checker.cli:main",
        ],
    },
    python_requires=">=3.9",
)
