from setuptools import setup, find_packages

setup(
    name="resclu",
    version="0.1.2",
    packages=find_packages(where='src'),
    package_dir={"": "src"},                         # 设置src目录为根目录
    install_requires=[
        'numpy',
        'pandas',
        'scikit-learn',
        'joblib',
        'kneed',
        # 在这里列出你的库所需的其他Python包
    ],

    author="QIJUN GAO,BOYU CHEN and YIZHE XU",
    author_email="10210350465@stu.ecnu.edu.cn",
    description="resclu is used to cluster data without supervision dynamically",
    long_description=open("README.md").read(),
    long_description_content_type="text/markdown",
    license="MIT",
    url="https://github.com/zondaxyz/resclu",
    entry_points={
        "console_scripts": ['mwjApiTest = mwjApiTest.manage:run']
    },  # 安装成功后，在命令行输入mwjApiTest 就相当于执行了mwjApiTest.manage.py中的run了

    classifiers=[
        "Development Status :: 3 - Alpha",
        "Intended Audience :: Developers",
        "License :: OSI Approved :: MIT License",
        "Programming Language :: Python",
        "Programming Language :: Python :: 3",
        "Programming Language :: Python :: 3.6",
        "Programming Language :: Python :: 3.7",
        "Programming Language :: Python :: 3.8",
        "Programming Language :: Python :: 3.9",
    ],
)