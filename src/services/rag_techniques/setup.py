"""
RAG Techniques - A comprehensive Python package for Retrieval-Augmented Generation
"""

from setuptools import setup, find_packages

with open("README.md", "r", encoding="utf-8") as fh:
    long_description = fh.read()

setup(
    name="rag-techniques",
    version="0.1.0",
    author="RAG Techniques Contributors",
    author_email="your.email@example.com",
    description="A comprehensive toolkit for building advanced RAG systems",
    long_description=long_description,
    long_description_content_type="text/markdown",
    url="https://github.com/NirDiamant/RAG_Techniques",
    packages=find_packages(),
    classifiers=[
        "Development Status :: 3 - Alpha",
        "Intended Audience :: Developers",
        "Topic :: Software Development :: Libraries :: Python Modules",
        "Topic :: Scientific/Engineering :: Artificial Intelligence",
        "License :: OSI Approved :: MIT License",
        "Programming Language :: Python :: 3",
        "Programming Language :: Python :: 3.8",
        "Programming Language :: Python :: 3.9",
        "Programming Language :: Python :: 3.10",
        "Programming Language :: Python :: 3.11",
    ],
    python_requires=">=3.8",
    install_requires=[
        "langchain>=0.3.0",
        "langchain-community>=0.3.0",
        "langchain-openai>=0.3.0",
        "langchain-text-splitters>=0.3.0",
        "openai>=1.0.0",
        "faiss-cpu>=1.7.0",
        "pypdf>=3.0.0",
        "PyMuPDF>=1.23.0",
        "python-dotenv>=1.0.0",
        "rank-bm25>=0.2.2",
        "numpy>=1.24.0",
        "pydantic>=2.0.0",
    ],
    extras_require={
        "dev": [
            "pytest>=7.0.0",
            "pytest-asyncio>=0.21.0",
            "black>=23.0.0",
            "flake8>=6.0.0",
            "mypy>=1.0.0",
        ],
        "evaluation": [
            "deepeval>=3.0.0",
        ],
        "cloud": [
            "langchain-cohere",
            "boto3>=1.28.0",  # For Amazon Bedrock
        ],
    },
    entry_points={
        "console_scripts": [
            "rag-simple=rag_techniques.cli:simple_rag_cli",
        ],
    },
)
