# Multimodal RAG System

A complete Retrieval-Augmented Generation (RAG) system that processes both text and images without using OCR for image content extraction.

## Overview

This project implements a multimodal RAG system following these key steps:

1. **Content Extraction**: Extract both text and images from documents (PDFs)
2. **Multimodal Embedding**: Create embeddings in a shared vector space
3. **Semantic Retrieval**: Retrieve relevant content based on user queries
4. **Answer Generation**: Generate responses using a multimodal LLM

## Features

- **Direct PDF Processing**: Extract text and images from PDF documents
- **No OCR Required**: Process images as visual content without text recognition
- **Shared Vector Space**: Embed both text and images in a unified vector space
- **API Integration**: Works with Groq, Llama, and other LLM APIs
- **Google Colab Support**: Runs seamlessly in Google Colab notebooks
- **Visual Results**: Display retrieved text and images with relevance scores

## Installation

### Requirements

- Python 3.8+
- PyMuPDF
- Transformers
- Sentence-Transformers
- Groq API key (or alternative LLM API)
- Pillow
- Matplotlib

### Setup

```bash
# Install required packages
pip install groq pymupdf pillow transformers requests sentence-transformers
```

Set your API keys as environment variables or directly in the code:

```python
GROQ_API_KEY = "your_groq_api_key"  
LLAMA_API_KEY = "your_llama_api_key"  
```

## Usage

### Basic Usage

```python
# Initialize the RAG system
rag = MultimodalRAG(llm_api_key=GROQ_API_KEY, llm_provider="groq")

# Process a PDF document
documents = rag.process_document("example.pdf")

# Query the system
result = rag.query("What is machine learning?")

# Visualize the results
rag.visualize_results(result)
```

### Using in Google Colab

1. Copy the entire code into a Colab notebook
2. Set your API keys
3. Upload documents or use the sample dataset
4. Run queries and view results

## Components

### 1. Document Processing

The system extracts both text and images from PDF documents:

```python
documents = extract_content_from_pdf(pdf_path, output_dir)
```

- Text is extracted directly from PDF pages
- Images are extracted and saved as separate files
- No OCR is performed on images - they're processed as visual content

### 2. Embedding

Text and images are embedded into a shared vector space:

- Text embedding uses Sentence Transformers
- Image embedding uses provider APIs or local models
- Embeddings are aligned for unified similarity search

### 3. Retrieval

The system retrieves relevant content based on semantic similarity:

```python
retrieved_docs = rag.vector_db.similarity_search(query_embedding, top_k=3)
```

- Automatically determines if a query is text-focused or image-focused
- Returns both text and images based on relevance
- Supports different modality types (text, image, combined)

### 4. Response Generation

Generates comprehensive answers using multimodal LLMs via API:

```python
response = rag.llm.generate_response(query, retrieved_docs)
```

- Integrates information from both text and images
- Cites sources from the original documents
- Handles both text-only and image-related queries

## Advanced Features

### Image Content Analysis

The current implementation retrieves images based on:
- Text surrounding the image in the document
- Captions and metadata associated with the image

For content-based image retrieval (e.g., searching for specific objects in images), you can extend the system with vision models:

```python
from transformers import CLIPProcessor, CLIPModel

clip_model = CLIPModel.from_pretrained("openai/clip-vit-base-patch32")
clip_processor = CLIPProcessor.from_pretrained("openai/clip-vit-base-patch32")

def analyze_image_content(image_path):
    image = Image.open(image_path)
    # Pre-defined categories for detection
    categories = ["cat", "dog", "bird", "fish", "horse", "elephant"]
    
    inputs = clip_processor(
        text=categories, 
        images=image, 
        return_tensors="pt", 
        padding=True
    )
    
    outputs = clip_model(**inputs)
    probs = outputs.logits_per_image.softmax(dim=1)
    
    top_prob, top_idx = probs.topk(1)
    top_category = categories[top_idx.item()]
    
    return top_category, top_prob.item()
```

## Navigation in colab
- Please navigate to runtime and change the runtime type to gpu
- Open runtime again and select run all
- Navigate to the last kernal
- Choose option 1
- Upload pdf
- Ask a Query

### Custom Document Sources
The default implementation works with PDFs only.

## Acknowledgments
- PDF processing uses the PyMuPDF library

