from setuptools import setup, find_packages

setup(
    name="ngrams_generator",
    author="rsh",
    author_email="",
    description="Generates the n-grams from the texts",
    packages=find_packages(where="src"),
    package_dir={"": "src"},
    include_package_data=True,
    # Dependencies
    install_requires=[
        "nltk>=3.8.1",
        "langdetect==1.0.9"
    ],
    entry_points={
        "console_scripts": [
            "download-pkgs = ngrams_generator:download_pkgs",
        ]
    }
    version="0.1",
    license="MIT",
    long_description=open("README.md").read(),
)