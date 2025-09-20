import os
import logging
from typing import List, Dict
import pdfplumber
from openai import OpenAI
from dotenv import load_dotenv
import tiktoken
import numpy as np


load_dotenv()


client = OpenAI(api_key=os.getenv('openai_api'))

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class PDFProcessor:
    def __init__(self, chunk_size=1000, chunk_overlap=200):
        self.chunk_size = chunk_size
        self.chunk_overlap = chunk_overlap
        self.chunks = []
        self.embeddings = []
        self.tokenizer = tiktoken.get_encoding("cl100k_base")

    def process_pdf(self, pdf_path: str) -> str:
        """Process PDF using pdfplumber for enhanced text extraction."""
        try:
            full_text = []
            
          
            with pdfplumber.open(pdf_path) as pdf:
               
                if pdf.metadata:
                    full_text.append("Document Metadata:")
                    for key, value in pdf.metadata.items():
                        if value and str(value).strip():
                            full_text.append(f"{key}: {value}")
                    full_text.append("-" * 50)
                
               
                for page_num, page in enumerate(pdf.pages, 1):
                    full_text.append(f"\nPage {page_num}:")
                    
                   
                    text = page.extract_text()
                    if text:
                        full_text.append(text)
                    
                    
                    tables = page.extract_tables()
                    if tables:
                        full_text.append("\nTables found on this page:")
                        for table_num, table in enumerate(tables, 1):
                            full_text.append(f"\nTable {table_num}:")
                           
                            for row in table:
                               
                                formatted_row = " | ".join(
                                    str(cell).strip() if cell else "" for cell in row
                                )
                                full_text.append(formatted_row)
                            full_text.append("-" * 50)
                    
                    # Extract hyperlinks using text extraction with layout
                    words = page.extract_words(x_tolerance=3, y_tolerance=3)
                    links = page.hyperlinks
                    if links:
                        full_text.append("\nLinks found on this page:")
                        for link in links:
                            if 'uri' in link:
                                full_text.append(f"Link: {link['uri']}")
                        full_text.append("-" * 50)
                    
                    # Extract lists (based on indentation and bullets)
                    lines = text.split('\n')
                    list_markers = ['•', '-', '*', '○', '●', '►', '▪', '▫']
                    for line in lines:
                        stripped_line = line.lstrip()
                        if stripped_line and stripped_line[0] in list_markers:
                            full_text.append(f"List item: {stripped_line}")
                    
                    full_text.append("-" * 50)  # Page separator

            return '\n'.join(full_text)
            
            return '\n'.join(full_text)

        except Exception as e:
            logger.error(f"Error processing PDF with Docling: {str(e)}")
            return ""

    def chunk_text(self, text: str) -> List[str]:
        """Split text into overlapping chunks."""
        chunks = []
        start = 0
        text_length = len(text)
        
        while start < text_length:
            end = start + self.chunk_size
            chunk = text[start:end]
            chunks.append(chunk)
            start = end - self.chunk_overlap
            
        return chunks

    def get_embedding(self, text: str) -> List[float]:
        """Get embeddings using OpenAI API."""
        try:
            response = client.embeddings.create(
                model="text-embedding-ada-002",
                input=text
            )
            return response.data[0].embedding
        except Exception as e:
            logger.error(f"Error getting embedding: {str(e)}")
            return []

    def build_retrieval_index(self, chunks: List[str]):
        """Build embeddings index."""
        self.chunks = chunks
        self.embeddings = []
        for chunk in chunks:
            embedding = self.get_embedding(chunk)
            if embedding:
                self.embeddings.append(embedding)

    def cosine_similarity(self, a, b):
        """Calculate cosine similarity between two vectors."""
        return np.dot(a, b) / (np.linalg.norm(a) * np.linalg.norm(b))

    def retrieve(self, query: str, k: int = 3) -> List[Dict]:
        """Retrieve most relevant chunks using OpenAI embeddings."""
        query_embedding = self.get_embedding(query)
        if not query_embedding:
            return []

        # Calculate similarities
        similarities = []
        for i, emb in enumerate(self.embeddings):
            score = self.cosine_similarity(query_embedding, emb)
            similarities.append((score, self.chunks[i]))

        
        similarities.sort(reverse=True)
        
        # Return top k results
        return [{"score": score, "text": text} for score, text in similarities[:k]]

def main():
    pdf_path = "Test.pdf" 
    
    
    pipeline = PDFProcessor()
    
    
    full_text = pipeline.process_pdf(pdf_path)
    if not full_text:
        logger.error("No text extracted from PDF.")
        return
    
   
    chunks = pipeline.chunk_text(full_text)
    logger.info(f"Created {len(chunks)} chunks.")
    
    # Build index
    pipeline.build_retrieval_index(chunks)
    
    
    output_dir = "output"
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
        
    
    output_file = os.path.join(output_dir, "rag_results.txt")
    
    
    question = "Extract the tables and links from the pdf?"

    
    results = pipeline.retrieve(question)
    
    
    with open(output_file, 'w', encoding='utf-8') as f:
        
        f.write(f"Question:\n{question}\n\n")
        
        
        f.write("Retrieved Chunks:\n")
        f.write("="* 50 + "\n\n")
        for i, result in enumerate(results, 1):
            f.write(f"Chunk {i} (Score: {result['score']:.3f}):\n")
            f.write(result['text'])
            f.write("\n" + "-"* 50 + "\n\n")
        
        
        context = "\n".join([r['text'] for r in results])
        try:
            response = client.chat.completions.create(
                model="gpt-3.5-turbo",
                messages=[
                    {"role": "system", "content": "You are a helpful assistant. Answer the question strictly based on the provided context."},
                    {"role": "user", "content": f"Context:\n{context}\n\nQuestion: {question}"}
                ]
            )
            f.write("\nGenerated Answer:\n")
            f.write("="* 50 + "\n\n")
            f.write(response.choices[0].message.content)
            
            
            print(f"Results have been written to: {output_file}")
        except Exception as e:
            logger.error(f"Error generating answer: {str(e)}")
            f.write("\nError: Could not generate an answer.")

if __name__ == "__main__":
    main()
