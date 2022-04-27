from setuptools import setup

setup(
    name="EMSRL",
    version="1.0",
    url="https://github.com/kangdj6358/EMSRL",
    author="Dongju Kang, Doeun Kang",
    license="MIT",
    install_requires=[
        "gym == 0.18.3",
        "ray == 1.9.0",
        "ray[rllib] == 1.9.0",
        "pandas == 1.3.3",
        "openpyxl == 3.0.9",
        "torch == 1.9.1",
    ],
    zip_safe=False,
)