from setuptools import setup, find_packages

setup(
    name="industry4_predictive_maintenance",
    version="1.0.0",
    packages=find_packages(),
    install_requires=[
        "numpy",
        "pandas",
        "tensorflow",
        "scikit-learn",
        "matplotlib",
        "seaborn",
        "fastapi",
        "uvicorn",
        "python-multipart",
        "librosa"
    ]
) 