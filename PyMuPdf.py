import os
import logging
from typing import List, Dict
import fitz  # PyMuPDF
from openai import OpenAI
from dotenv import load_dotenv
import tiktoken
import numpy as np
import json

load_dotenv()


client = OpenAI(api_key=os.getenv('openai_api'))

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class MuPdfRAGPipeline:
    def __init__(self, chunk_size=1000, chunk_overlap=200):
        self.chunk_size = chunk_size
        self.chunk_overlap = chunk_overlap
        self.chunks = []
        self.embeddings = []
        self.tokenizer = tiktoken.get_encoding("cl100k_base")

    def extract_text_from_pdf(self, pdf_path: str) -> str:
        """Extract text and metadata from a PDF file using PyMuPDF with enhanced processing."""
        try:
           
            doc = fitz.open(pdf_path)
            full_text = []

            
            metadata = doc.metadata
            if metadata:
                full_text.append("Document Metadata:")
                for key, value in metadata.items():
                    if value and str(value).strip():
                        full_text.append(f"{key}: {value}")
                full_text.append("-" * 50)

          
            for page_num, page in enumerate(doc, 1):
                full_text.append(f"\nPage {page_num}:")
                
                
                blocks = page.get_text("blocks")
                
               
                tables = page.find_tables()
                if tables and tables.tables:
                    full_text.append("\nTables found on this page:")
                    for table in tables:
                        cells = table.extract()
                        table_text = []
                        for row in cells:
                            # Format each row with pipe separators
                            formatted_row = " | ".join(str(cell) for cell in row if str(cell).strip())
                            if formatted_row:
                                table_text.append(formatted_row)
                        if table_text:
                            full_text.append("\n".join(table_text))
                            full_text.append("-" * 50)

                # Extract links
                links = page.get_links()
                if links:
                    full_text.append("\nLinks found on this page:")
                    for link in links:
                        if "uri" in link:
                            full_text.append(f"Link: {link['uri']}")
                    full_text.append("-" * 50)

                # Extract text with proper formatting
                text_with_format = []
                for block in blocks:
                    if block[6] == 0:  # Regular text block
                        text_with_format.append(block[4])
                    elif block[6] == 1:  # Image block
                        text_with_format.append("[Image]")
                
                full_text.append("\n".join(text_with_format))
                full_text.append("-" * 50)  # Page separator

                # Extract images (optional, commented out by default)
                # images = page.get_images()
                # if images:
                #     full_text.append(f"\nFound {len(images)} images on page {page_num}")

            doc.close()
            return "\n".join(full_text)

        except Exception as e:
            logger.error(f"Error extracting text from PDF: {str(e)}")
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

       
        similarities = []
        for i, emb in enumerate(self.embeddings):
            score = self.cosine_similarity(query_embedding, emb)
            similarities.append((score, self.chunks[i]))

        
        similarities.sort(reverse=True)
        
       
        return [{"score": score, "text": text} for score, text in similarities[:k]]

def main():
    
    pdf_path = "Test.pdf"  
    
    
    pipeline = MuPdfRAGPipeline()
    
    full_text = pipeline.extract_text_from_pdf(pdf_path)
    if not full_text:
        logger.error("No text extracted from PDF.")
        return
    
    chunks = pipeline.chunk_text(full_text)
    logger.info(f"Created {len(chunks)} chunks.")
    
    pipeline.build_retrieval_index(chunks)
    
    output_dir = "output"
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
        
    
    output_file = os.path.join(output_dir, "rag_results.txt")
    
    
    question = "What are the tables and links in the document?"

    
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
