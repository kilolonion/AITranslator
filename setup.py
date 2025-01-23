from setuptools import setup, find_packages

setup(
    name="aitrans",
    version="1.2.4",
    packages=find_packages(),
    install_requires=[
        "openai>=1.0.0",
        "aiohttp>=3.8.0",
        "python-dotenv>=0.19.0",
        "langdetect>=1.0.9",
        "tenacity>=8.0.0",
        "pytest>=7.0.0",
        "pytest-asyncio>=0.18.0",
    ],
    author="Your Name",
    author_email="your.email@example.com",
    description="一个强大的AI翻译库",
    long_description=open("README.md", encoding="utf-8").read(),
    long_description_content_type="text/markdown",
    url="https://github.com/yourusername/aitrans",
    classifiers=[
        "Development Status :: 4 - Beta",
        "Intended Audience :: Developers",
        "License :: OSI Approved :: MIT License",
        "Programming Language :: Python :: 3",
        "Programming Language :: Python :: 3.8",
        "Programming Language :: Python :: 3.9",
        "Programming Language :: Python :: 3.10",
        "Programming Language :: Python :: 3.11",
    ],
    python_requires=">=3.8",
)
