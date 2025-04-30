from setuptools import setup, find_packages

with open("requirements.txt") as f:
    requirements = f.read().splitlines()

setup(
    name="ai_data_science",
    version="0.1.0",
    packages=find_packages(),
    install_requires=requirements,
    author="AI Data Science Team",
    author_email="example@example.com",
    description="A toolkit for data science tasks powered by AI agents",
    keywords="ai, data-science, agents, data-cleaning",
    url="https://github.com/your-username/ai-data-science",
    classifiers=[
        "Development Status :: 3 - Alpha",
        "Intended Audience :: Developers",
        "Intended Audience :: Science/Research",
        "License :: OSI Approved :: MIT License",
        "Programming Language :: Python :: 3",
        "Programming Language :: Python :: 3.9",
        "Programming Language :: Python :: 3.10",
        "Programming Language :: Python :: 3.11",
    ],
    python_requires=">=3.9",
) 