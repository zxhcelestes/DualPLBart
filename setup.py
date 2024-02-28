
from setuptools import setup, find_packages

with open("README.md", "r", encoding="utf-8") as fh:
    long_description = fh.read()

setup(
    name="dualplbart",
    version="0.0.1",
    author="李捷，陈琦，王宠暄",
    author_email="2027407061@stu.suda.edu.cn",
    description="2022年度大学生创新创业训练计划项目（202210285197H）",
    long_description=long_description,
    long_description_content_type="text/markdown",
    url="https://github.com/jaylee1002/dualplbart",
    license="Not open source",
    packages=find_packages(),
    python_requires=">=3.8.0, <3.12",
    install_requires=[
        "torch~=2.0.1",
        "transformers>=4.33.1",
        "yapf>=0.40.2",
        "pandas>=2.1.2"
    ],
    extras_require={
        "dev": [
            "torch~=2.0.1",
            "transformers>=4.33.1",
            "yapf>=0.40.2",
            "pandas>=2.1.2"
        ]
    }
)
