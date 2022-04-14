import setuptools

with open("README.md", "r", encoding="utf-8") as fh:
    long_description = fh.read()

setuptools.setup(
    name="causal_dataset",
    version="0.0.1",
    author="Sam Considine",
    author_email="sbc35@cam.ac.uk",
    description="A causal dataset to benchmark k-NN latent graph inference methods",
    long_description=long_description,
    long_description_content_type="text/markdown",
    url="https://github.com/samconsidine/knn_graph_inference",
    project_urls={
    },
    classifiers=[
        "Programming Language :: Python :: 3",
        "License :: OSI Approved :: MIT License",
        "Operating System :: OS Independent",
    ],
    package_dir={"": "src"},
    packages=setuptools.find_packages(where="src"),
    python_requires=">=3.8",
)
