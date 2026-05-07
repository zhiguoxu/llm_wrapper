from setuptools import setup, find_packages

with open('requirements.txt') as f:
    required_packages = f.read().splitlines()

setup(
    name='llm_wrapper',
    version='0.1.1',
    packages=find_packages(where='src'),  # 指定包的位置
    package_dir={'': 'src'},
    # package_data={"xxx.config": ["*.json"]},
    install_requires=required_packages,
    author='zhiguo',
    author_email='zhiguoxu2004@163.com',
    description='llm_wrapper',
    long_description="",
    long_description_content_type='text/markdown',
    url='https://github.com/xxx',
    classifiers=[]
)
