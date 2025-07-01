from setuptools import setup, find_packages

setup(
    name='rl-baseline',
    version='0.0.0',
    description='Distributed Post-Training RL Library for LLMs',
    author='Zhicheng Yang',
    packages=find_packages(include=['deepscaler',]),
    install_requires=[
        'torch==2.4.0',
        'deepspeed',
        'pybind11',
        'pylatexenc',
        'pytest',
        'sentence_transformers',
        'tabulate',
        'torchmetrics',
        'vertexai',
        'wandb',
        'math_verify',
        'antlr4-python3-runtime==4.9.3',
    ],
    classifiers=[
        'Programming Language :: Python :: 3',
        'License :: OSI Approved :: MIT License',
        'Operating System :: OS Independent',
    ],
    python_requires='>=3.9',
)