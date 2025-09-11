import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import nltk
from chonkie import SentenceChunker, TokenChunker, SemanticChunker


nltk.download('punkt')
nltk.download('stopwords')
nltk.download('punkt_tab')
nltk.download('averaged_perceptron_tagger')
from sklearn.metrics.pairwise import cosine_similarity
from sentence_transformers import SentenceTransformer, CrossEncoder
from rank_bm25 import BM25Okapi
from dotenv import load_dotenv
from typing import List, Dict, Tuple
import logging
import nltk
from nltk.tokenize import word_tokenize
from nltk.corpus import stopwords
import re


nltk.download('punkt')
nltk.download('stopwords')


logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


load_dotenv()

class ChunkingPipeline:
    def __init__(self):
       
        self.sentence_chunker = SentenceChunker()  
        
        
        self.token_chunker = TokenChunker()
        
        
        self.semantic_chunker = SemanticChunker(model="all-MiniLM-L6-v2")
        
        
        self.embedding_model = SentenceTransformer('all-MiniLM-L6-v2', device='cpu')
        self.cross_encoder = CrossEncoder('cross-encoder/ms-marco-MiniLM-L-6-v2', device='cpu')
        
        
        self.stop_words = set(stopwords.words('english'))
        self.bm25_index = None
        self.chunk_index = []  
        
        # Product categories and synonyms
        self.categories = {
            'mobile': ['phone', 'smartphone', 'mobile', 'cellular', 'handset'],
            'laptop': ['laptop', 'notebook', 'computer', 'pc', 'gaming laptop'],
            'audio': ['headphones', 'earphones', 'headset', 'earbuds', 'wireless', 'bluetooth'],
            'tv': ['television', 'tv', 'smart tv', '4k', 'display', 'screen'],
            'tablet': ['tablet', 'ipad', 'slate'],
            'wearable': ['smartwatch', 'watch', 'fitness tracker', 'band', 'wearable']
        }
        
        # Custom domain stopwords
        self.domain_stop_words = {'official', 'new', 'brand', 'warranty', 'original', 'genuine'}
    
    def chunk_text(self, text: str, chunker_type: str) -> List[str]:
        """Chunk a single text using the specified chunker type."""
        try:
            if chunker_type == "sentence":
                # sentence chunker from chonkie
                initial_chunks = self.sentence_chunker.chunk(text)
                chunks = []
                current_chunk = []
                current_length = 0
                max_length = 30
                
                for chunk in initial_chunks:
                    chunk_text = str(chunk.text).strip()
                    words = chunk_text.split()
                    
                    if current_length + len(words) <= max_length:
                        current_chunk.append(chunk_text)
                        current_length += len(words)
                    else:
                        if current_chunk:
                            chunks.append(" ".join(current_chunk))
                        current_chunk = [chunk_text]
                        current_length = len(words)
                
                if current_chunk:
                    chunks.append(" ".join(current_chunk))
            
            elif chunker_type == "token":
                # Token-based chunking with overlap
                initial_chunks = self.token_chunker.chunk(text)
                chunks = []
                chunk_size = 20
                overlap = 5
                
                # Extracts text from chunk objects
                words = []
                for chunk in initial_chunks:
                    words.extend(str(chunk.text).strip().split())
                
                # Creates overlapping chunks
                for i in range(0, len(words), chunk_size - overlap):
                    chunk = words[i:i + chunk_size]
                    if chunk:
                        chunks.append(" ".join(chunk))
            
            elif chunker_type == "semantic":
                # Uses semantic chunker with post-processing
                initial_chunks = self.semantic_chunker.chunk(text)
                chunks = []
                min_length = 5
                
                for chunk in initial_chunks:
                    chunk_text = str(chunk.text).strip()
                    
                    # Combines very short chunks with the previous chunk
                    if chunks and len(chunk_text.split()) < min_length:
                        chunks[-1] = f"{chunks[-1]} {chunk_text}"
                    else:
                        chunks.append(chunk_text)
            else:
                raise ValueError(f"Unknown chunker type: {chunker_type}")
            
            # Post-processes all chunks
            processed_chunks = []
            seen = set()
            for chunk in chunks:
                chunk_str = str(chunk).strip()
                if chunk_str and chunk_str not in seen:
                    seen.add(chunk_str)
                    processed_chunks.append(chunk_str)
            
            return processed_chunks
        
        except Exception as e:
            logger.error(f"Error chunking with {chunker_type}: {str(e)}")
            return []
    
    def process_texts(self, texts: List[str]) -> Dict[str, List[str]]:
        """Process multiple texts with all chunking strategies."""
        results = {}
        
        # Processes each chunking strategy
        for chunker_type in ["sentence", "token", "semantic"]:
            all_chunks = []
            chunks_per_text = [] 
            
            for text in texts:
                chunks = self.chunk_text(text, chunker_type)
                all_chunks.extend(chunks)
                chunks_per_text.append(len(chunks))
            
            results[chunker_type] = all_chunks
            
            
            logger.info(f"\n{chunker_type.capitalize()} Chunking Results:")
            logger.info(f"Input texts: {len(texts)}")
            logger.info(f"Output chunks: {len(all_chunks)}")
            
            if all_chunks:
                lengths = [len(chunk.split()) for chunk in all_chunks]
                avg_chunks_per_text = sum(chunks_per_text) / len(texts)
                
                logger.info(f"Chunks per text: {avg_chunks_per_text:.1f}")
                logger.info(f"Avg chunk length: {sum(lengths)/len(lengths):.1f} words")
                logger.info(f"Min/Max chunk length: {min(lengths)}/{max(lengths)} words")
                
               
                if len(all_chunks) > 1:
                    overlaps = []
                    for i in range(len(all_chunks)-1):
                        words1 = set(all_chunks[i].lower().split())
                        words2 = set(all_chunks[i+1].lower().split())
                        overlap = len(words1.intersection(words2)) / len(words1.union(words2))
                        overlaps.append(overlap)
                    
                    logger.info(f"Avg word overlap between chunks: {sum(overlaps)/len(overlaps):.2f}")
                    logger.info(f"Max word overlap: {max(overlaps):.2f}")
        
        return results
    
    def expand_query(self, query: str) -> str:
        """Expand query with category-specific synonyms."""
        query_terms = query.lower().split()
        expanded_terms = set(query_terms)
        
        # Adds category-specific terms
        for category, synonyms in self.categories.items():
            if any(term in query_terms for term in synonyms):
                expanded_terms.update(synonyms)
        
        return " ".join(expanded_terms)
    
    def preprocess_text(self, text: str) -> List[str]:
        """Preprocess text for BM25 indexing."""
        
        text = str(text).lower()
        
        # Remove special characters but keep important product identifiers
        text = re.sub(r'[^\w\s+]', ' ', text)
        
        # Tokenize
        words = text.split()
        
        # Remove stopwords and domain-specific stopwords
        return [word for word in words 
                if word not in self.stop_words 
                and word not in self.domain_stop_words
                and word.isalnum()]
    
    def build_retrieval_index(self, chunks: List[str]):
        """Build BM25 and dense retrieval indices."""
        
        tokenized_chunks = [self.preprocess_text(chunk) for chunk in chunks]
        self.bm25_index = BM25Okapi(tokenized_chunks)
        self.chunk_index = chunks
        
        # Pre-compute embeddings for semantic search
        self.chunk_embeddings = self.embedding_model.encode(chunks)
    
    def detect_categories(self, text: str) -> List[str]:
        """Detect product categories in text."""
        text = text.lower()
        detected = []
        for category, terms in self.categories.items():
            if any(term in text for term in terms):
                detected.append(category)
        return detected

    def get_category_score(self, query: str, text: str) -> float:
        """Calculate category-based score."""
        query_categories = self.detect_categories(query)
        text_categories = self.detect_categories(text)
        
        
        if not query_categories:
            return 1.0
        
        
        if query_categories and not text_categories:
            
            return 0.5
        
        # Calculate overlap score
        if query_categories and text_categories:
            overlap = len(set(query_categories).intersection(text_categories))
            if overlap > 0:
                
                return 1.0 + (overlap * 0.5)
        
        # Default case - slight penalty for no category match
        return 0.8
    
    def bm25_search(self, query: str, k: int = 5) -> List[Tuple[str, float]]:
        """Perform BM25 search with strict category matching."""
        query_categories = self.detect_categories(query)
        
        
        
        # Expand query with synonyms
        expanded_query = self.expand_query(query)
        query_tokens = self.preprocess_text(expanded_query)
        
        
        scores = self.bm25_index.get_scores(query_tokens)
        
        
        adjusted_scores = [
            score * self.get_category_score(query, chunk)
            for score, chunk in zip(scores, self.chunk_index)
        ]
        
        # Filter out very low scores
        valid_indices = [i for i, score in enumerate(adjusted_scores) if score > 0.1]
        if not valid_indices:
            return [("No relevant products found", 0.0)]
            
        top_k = sorted(valid_indices, key=lambda i: adjusted_scores[i], reverse=True)[:k]
        return [(self.chunk_index[i], adjusted_scores[i]) for i in top_k]
    
    def semantic_search(self, query: str, k: int = 5) -> List[Tuple[str, float]]:
        """Perform semantic search with strict category matching."""
        query_categories = self.detect_categories(query)
        
        
        
        query_embedding = self.embedding_model.encode(query)
        similarities = cosine_similarity([query_embedding], self.chunk_embeddings)[0]
        
        # Apply category scoring
        adjusted_scores = [
            sim * self.get_category_score(query, chunk)
            for sim, chunk in zip(similarities, self.chunk_index)
        ]
        
        # Filter out very low scores
        valid_indices = [i for i, score in enumerate(adjusted_scores) if score > 0.1]
        if not valid_indices:
            return [("No relevant products found", 0.0)]
            
        top_k = sorted(valid_indices, key=lambda i: adjusted_scores[i], reverse=True)[:k]
        return [(self.chunk_index[i], adjusted_scores[i]) for i in top_k]
    
    def hybrid_search(self, query: str, k: int = 5, alpha: float = 0.5) -> List[Tuple[str, float]]:
        """Combine BM25 and semantic search with enhanced scoring."""
        query_categories = self.detect_categories(query)
        
        
        
        expanded_query = self.expand_query(query)
        
        # Get base scores
        bm25_scores = self.bm25_index.get_scores(self.preprocess_text(expanded_query))
        query_embedding = self.embedding_model.encode(expanded_query)
        semantic_scores = cosine_similarity([query_embedding], self.chunk_embeddings)[0]
        
        # Normalize scores with epsilon to prevent division by zero
        eps = 1e-10
        if np.max(bm25_scores) > np.min(bm25_scores):
            bm25_scores = (bm25_scores - np.min(bm25_scores)) / (np.max(bm25_scores) - np.min(bm25_scores) + eps)
        else:
            bm25_scores = np.full_like(bm25_scores, 0.5)
            
        if np.max(semantic_scores) > np.min(semantic_scores):
            semantic_scores = (semantic_scores - np.min(semantic_scores)) / (np.max(semantic_scores) - np.min(semantic_scores) + eps)
        else:
            semantic_scores = np.full_like(semantic_scores, 0.5)
        
        # Dynamic alpha based on query type
        if any(term in query.lower() for term in ['camera', 'mp', 'photo']):
            alpha = 0.4  # Give more weight to semantic search for camera queries
        elif any(term in query.lower() for term in ['battery', 'mah', 'life']):
            alpha = 0.6  # Give more weight to BM25 for battery queries
        
        # Combine scores with boosting for relevant features
        combined_scores = []
        for i, (bm25_score, sem_score) in enumerate(zip(bm25_scores, semantic_scores)):
            base_score = alpha * bm25_score + (1 - alpha) * sem_score
            
            # Apply feature-specific boosting
            chunk = self.chunk_index[i].lower()
            boost = 1.0
            
            if 'camera' in query.lower() and any(cam in chunk for cam in ['mp', 'camera']):
                boost += 0.2
            if 'battery' in query.lower() and any(bat in chunk for bat in ['mah', 'battery']):
                boost += 0.2
                
            final_score = base_score * boost * self.get_category_score(query, self.chunk_index[i])
            combined_scores.append(final_score)
            
        combined_scores = np.array(combined_scores)
        top_k = np.argsort(combined_scores)[-k:][::-1]
        return [(self.chunk_index[i], combined_scores[i]) for i in top_k]
    
    def rerank_results(self, query: str, initial_results: List[Tuple[str, float]], k: int = 5) -> List[Tuple[str, float]]:
        """Rerank results using cross-encoder with enhanced normalization."""
        if not initial_results:
            return []
            
        # Expand query for better matching
        expanded_query = self.expand_query(query)
        pairs = [[expanded_query, chunk] for chunk, _ in initial_results]
        rerank_scores = self.cross_encoder.predict(pairs)
        
        # Apply min-max normalization first
        if len(rerank_scores) > 1:
            score_min = np.min(rerank_scores)
            score_max = np.max(rerank_scores)
            if score_max > score_min:
                normalized_scores = (rerank_scores - score_min) / (score_max - score_min)
            else:
                normalized_scores = np.full_like(rerank_scores, 0.5)
        else:
            normalized_scores = np.array([0.5])
        
        # Boost scores for better differentiation
        boosted_scores = np.power(normalized_scores, 0.5)  # Square root to boost lower scores
        
        # Apply category scoring with higher weight for relevant features
        final_scores = []
        for score, (chunk, _) in zip(boosted_scores, initial_results):
            category_score = self.get_category_score(query, chunk)
            
            # Check for specific features in chunk
            feature_bonus = 1.0
            if 'camera' in query.lower() and any(cam in chunk.lower() for cam in ['mp', 'camera']):
                feature_bonus += 0.3
            if 'battery' in query.lower() and any(bat in chunk.lower() for bat in ['mah', 'battery']):
                feature_bonus += 0.3
                
            final_score = score * category_score * feature_bonus
            final_scores.append(final_score)
        
        reranked = [(initial_results[i][0], final_scores[i]) 
                    for i in range(len(initial_results))]
                    
        return sorted(reranked, key=lambda x: x[1], reverse=True)[:k]
    
    def save_results_to_file(self, results: List[Tuple[str, float]], method: str, query: str, output_file: str):
        """Save search results to a file in a structured format."""
        with open(output_file, 'a', encoding='utf-8') as f:
            f.write(f"\n{'='*80}\n")
            f.write(f"Query: {query}\n")
            f.write(f"Method: {method}\n")
            f.write('='*80 + '\n\n')
            
            for i, (chunk, score) in enumerate(results, 1):
                f.write(f"Result #{i}:\n")
                f.write(f"Score: {score:.3f}\n")
                f.write(f"Product: {chunk}\n")
                f.write('-'*50 + '\n')
            f.write('\n')
    
    def evaluate_retrieval(self, query: str, k: int = 5):
        """Compare different retrieval strategies and save results to files."""
        
        output_file = f"search_results_{query.replace(' ', '_')}.txt"
        
        
        bm25_results = self.bm25_search(query, k)
        self.save_results_to_file(bm25_results, "BM25 Search", query, output_file)
        
        
        semantic_results = self.semantic_search(query, k)
        self.save_results_to_file(semantic_results, "Semantic Search", query, output_file)
        
        
        hybrid_results = self.hybrid_search(query, k)
        self.save_results_to_file(hybrid_results, "Hybrid Search", query, output_file)
        
        
        reranked_results = self.rerank_results(query, hybrid_results, k)
        self.save_results_to_file(reranked_results, "Reranked Results", query, output_file)
        
        logger.info(f"Search results have been saved to {output_file}")
    
    def analyze_chunks(self, chunks: List[str], name: str):
        """Analyze chunks and create visualizations."""
        if not chunks:
            logger.warning(f"No chunks to analyze for {name}")
            return
        
        
        lengths = [len(chunk.split()) for chunk in chunks]
        
        # Create distribution plot
        plt.figure(figsize=(10, 6))
        plt.hist(lengths, bins=min(20, len(set(lengths))), alpha=0.7)
        if len(lengths) > 1:
            plt.boxplot(lengths, vert=False, positions=[max(plt.ylim())*0.8])
        plt.title(f"Chunk Length Distribution - {name}")
        plt.xlabel("Words per chunk")
        plt.ylabel("Frequency")
        plt.savefig(f"{name}_distribution.png")
        plt.close()
        
        # Calculate semantic similarities
        if len(chunks) > 1:
            try:
                sample_size = min(100, len(chunks))
                sample_chunks = chunks[:sample_size]
                embeddings = self.embedding_model.encode(sample_chunks)
                similarities = cosine_similarity(embeddings)
                np.fill_diagonal(similarities, 0)
                
                plt.figure(figsize=(10, 6))
                plt.hist(similarities.flatten(), bins=50, alpha=0.7)
                plt.title(f"Semantic Similarity Distribution - {name}")
                plt.xlabel("Cosine Similarity")
                plt.ylabel("Frequency")
                plt.savefig(f"{name}_similarity.png")
                plt.close()
                
                logger.info(f"Similarity Analysis for {name}:")
                logger.info(f"Avg similarity: {np.mean(similarities):.3f}")
                logger.info(f"% high overlap (>0.8): {(similarities > 0.8).sum() / similarities.size * 100:.1f}%")
            except Exception as e:
                logger.error(f"Error in similarity analysis for {name}: {str(e)}")

def main():
    
    results_dir = "search_results"
    if not os.path.exists(results_dir):
        os.makedirs(results_dir)
    os.chdir(results_dir)
    
    
    df = pd.read_csv("../products.csv")
    texts = df["product_title"].tolist()

    
    cleaned_texts = []
    for text in texts:
        if isinstance(text, str) and text.strip():
            
            text = text.replace('\n', ' ').strip()
            cleaned_texts.append(text)

    logger.info(f"Loaded {len(cleaned_texts)} texts for processing")

    
    pipeline = ChunkingPipeline()
    chunk_results = pipeline.process_texts(cleaned_texts)

    
    for strategy, chunks in chunk_results.items():
        logger.info(f"\nAnalyzing {strategy} chunks...")
        pipeline.analyze_chunks(chunks, strategy)
        
        
        logger.info(f"\nTesting retrieval with {strategy} chunks...")
        pipeline.build_retrieval_index(chunks)
        
        
        test_queries = [
            "phone with good camera",
            "phones with long battery life"
        ]
        
        for query in test_queries:
            pipeline.evaluate_retrieval(query)

if __name__ == "__main__":
    main()
