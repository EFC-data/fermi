from setuptools import setup, find_packages

setup(
    name="fermi",
    version="0.1.0",
    description="FitnEss Relatedness and other MetrIcs",
    author="Centro Fermi",
    packages=find_packages(),
    install_requires=[
        "numpy",
        "scipy",
        "pandas",
        "scikit-learn",
        "bokeh"
    ],
    python_requires=">=3.8",
)

