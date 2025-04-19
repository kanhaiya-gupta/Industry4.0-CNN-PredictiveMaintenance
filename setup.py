from setuptools import setup, find_packages

setup(
    name="predictive_maintenance",
    version="0.1.0",
    packages=find_packages(),
    install_requires=[
        "fastapi",
        "uvicorn",
        "torch",
        "numpy",
        "pandas",
        "scikit-learn",
        "pydantic",
        "python-dotenv",
        "pytest",
        "pytest-cov",
    ],
) 