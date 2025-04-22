from setuptools import find_packages, setup

setup(
    name="streamlit_image_comparison",
    version="0.0.5",
    packages=find_packages("src"),
    package_dir={"": "src"},
    install_requires=[
        "streamlit",
        "Pillow",
        "requests",
        "numpy",
    ],
)
