from setuptools import setup, find_packages

setup(
    name='pdf_data_extractor',
    version='0.1.0',
    packages=find_packages(),
    install_requires=[
        'jsonpath-ng',
        'llama-index==0.9.48',
        'pillow',
        'PyMuPDF'
    ],
    extras_require={
        'gradio': ['gradio'],
        'jupyter': ['notebook']
    }
)
