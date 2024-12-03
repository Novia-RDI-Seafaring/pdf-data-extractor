from setuptools import setup, find_packages

setup(
    name='pdf_data_extractor',
    version='0.1.0',
    packages=find_packages(),
    install_requires=[
        'jsonpath-ng',
        'llama-index==0.12.2',
        'llama-index-llms-azure-openai',
        'llama-index-embeddings-azure-openai',
        'llama-index-multi-modal-llms-azure-openai',
        'pillow',
        'PyMuPDF'
    ],
    extras_require={
        'gradio': ['gradio'],
        'jupyter': ['notebook']
    }
)
