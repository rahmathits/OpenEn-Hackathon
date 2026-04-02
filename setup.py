"""
setup.py

Legacy setup script for compatibility with older pip versions and tools
that do not yet support pyproject.toml (PEP 517/518).

For modern installs, pyproject.toml is used automatically.

Install (development mode):
    pip install -e .

Install with all extras:
    pip install -e ".[all]"

Build distribution:
    python setup.py sdist bdist_wheel
"""

from setuptools import setup, find_packages

setup(
    name="eda-openenv",
    version="1.0.0",
    description=(
        "A real-world Reinforcement Learning environment for EDA pipeline agents, "
        "built on the OpenEnv standard."
    ),
    long_description=open("README.md", encoding="utf-8").read(),
    long_description_content_type="text/markdown",
    author="Rahmath",
    author_email="rahmathits@gmail.com",
    url="https://huggingface.co/spaces/Rahmath1/OpenEnv",
    project_urls={
        "Repository": "https://github.com/rahmathits/OpenEn-Hackathon",
        "Bug Tracker": "https://github.com/rahmathits/OpenEn-Hackathon/issues",
    },
    license="MIT",
    python_requires=">=3.11",
    packages=find_packages(include=["env", "env.*", "tools", "tools.*"]),
    py_modules=[
        "app",
        "server",
        "pipeline",
        "baseline_agent",
        "eda_openenv_client",
    ],
    install_requires=[
        "pandas>=2.0.0",
        "pydantic>=2.0.0",
        "numpy>=1.26.0",
        "scikit-learn>=1.4.0",
        "scipy>=1.12.0",
    ],
    extras_require={
        "app": [
            "streamlit>=1.35.0",
        ],
        "server": [
            "fastapi>=0.111.0",
            "uvicorn>=0.30.0",
        ],
        "baseline": [
            "openai>=1.30.0",
        ],
        "all": [
            "streamlit>=1.35.0",
            "fastapi>=0.111.0",
            "uvicorn>=0.30.0",
            "openai>=1.30.0",
        ],
    },
    entry_points={
        "console_scripts": [
            # Allows: eda-baseline --csv data.csv
            "eda-baseline=baseline_agent:main",
        ],
    },
    classifiers=[
        "Development Status :: 3 - Alpha",
        "Intended Audience :: Developers",
        "Intended Audience :: Science/Research",
        "License :: OSI Approved :: MIT License",
        "Programming Language :: Python :: 3",
        "Programming Language :: Python :: 3.11",
        "Topic :: Scientific/Engineering :: Artificial Intelligence",
    ],
    keywords=[
        "reinforcement-learning",
        "openenv",
        "eda",
        "exploratory-data-analysis",
        "rl-environment",
        "machine-learning",
        "ai-agent",
    ],
    include_package_data=True,
    package_data={
        "*": ["*.md", "*.txt", "*.toml"],
    },
)