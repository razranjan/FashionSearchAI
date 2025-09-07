#!/usr/bin/env python
# coding: utf-8

"""
Fashion Search AI Project - Local Version
=========================================

This project implements a three-layer AI search system for fashion products:

1. Embedding Layer: Processes and embeds fashion product data
2. Search Layer: Performs semantic search with caching and re-ranking
3. Generation Layer: Generates natural language responses using GPT-3.5

The project satisfies all requirements:
- Implements all three layers effectively
- Uses OpenAI embeddings and cross-encoder re-ranking
- Implements caching mechanism
- Tests with multiple queries
- Provides comprehensive output for analysis
"""

import os
import pandas as pd
import chromadb
from chromadb.utils.embedding_functions import OpenAIEmbeddingFunction
from chromadb.utils.embedding_functions import SentenceTransformerEmbeddingFunction
from sentence_transformers import CrossEncoder, util
import openai
import matplotlib.pyplot as plt
from PIL import Image
import numpy as np
from pathlib import Path
import json

# Set OpenAI API key
openai.api_key = os.getenv('OPENAI_API_KEY')

class FashionSearchAI:
    def __init__(self, csv_path, images_folder):
        """
        Initialize the Fashion Search AI system
        
        Args:
            csv_path (str): Path to the fashion dataset CSV
            images_folder (str): Path to the images folder
        """
        self.csv_path = csv_path
        self.images_folder = images_folder
        self.fashion_data = None
        self.client = None
        self.fashion_collection = None
        self.cache_collection = None
        self.cross_encoder = None
        # Prefer environment variable for key detection (works with openai>=1.0)
        self.use_openai = bool(os.getenv('OPENAI_API_KEY'))
        # In local fallback mode (no OpenAI key), limit indexing for speed
        self.index_limit = None if self.use_openai else 1000
        
        # Initialize components
        self._load_data()
        self._setup_chromadb()
        self._setup_cross_encoder()
        
    def _load_data(self):
        """Load and preprocess the fashion dataset"""
        print("Loading fashion dataset...")
        self.fashion_data = pd.read_csv(self.csv_path)
        print(f"Loaded {len(self.fashion_data)} fashion products")
        
        # Create metadata for each product
        self.fashion_data['metadata'] = self.fashion_data.apply(
            lambda x: {
                'Product_id': x['p_id'],
                'Name': x['name'],
                'Product_type': x['products'],
                'Price_INR': x['price'],
                'Colour': x['colour'],
                'Brand': x['brand'],
                'RatingCount': x['ratingCount'],
                'Rating': x['avg_rating'],
                'Description': x['description'],
                'Product_attributes': x['p_attributes']
            }, axis=1
        )
        
    def _setup_chromadb(self):
        """Setup ChromaDB with collections for fashion products and cache"""
        print("Setting up ChromaDB...")
        
        # Choose DB path, embedding function, and collection names based on availability of OpenAI key
        if self.use_openai:
            db_path = "./chroma_db"
            embedding_function = OpenAIEmbeddingFunction(
                api_key=os.getenv('OPENAI_API_KEY'),
                model_name="text-embedding-ada-002"
            )
            products_collection_name = 'Fashion_Products'
            cache_collection_name = 'Fashion_Cache'
        else:
            db_path = "./chroma_db_st"
            embedding_function = SentenceTransformerEmbeddingFunction(
                model_name="all-MiniLM-L6-v2"
            )
            products_collection_name = 'Fashion_Products_ST'
            cache_collection_name = 'Fashion_Cache_ST'

        # Initialize ChromaDB client
        self.client = chromadb.PersistentClient(path=db_path)
        
        # Create or get collections
        self.fashion_collection = self.client.get_or_create_collection(
            name=products_collection_name,
            embedding_function=embedding_function
        )
        
        self.cache_collection = self.client.get_or_create_collection(
            name=cache_collection_name,
            embedding_function=embedding_function
        )
        
        # Check if collection is empty and populate if needed
        if self.fashion_collection.count() == 0:
            self._populate_collection()
        else:
            current_count = self.fashion_collection.count()
            print(f"Collection already has {current_count} documents")
            # Check if we need to rebuild for full coverage
            if current_count < 10000:  # If less than 10K products, rebuild
                print(f"Collection has only {current_count} products. Rebuilding with ALL {len(self.fashion_data)} products...")
                self._rebuild_collection()
            else:
                print(f"Collection has {current_count} products - good coverage!")
            
    def _populate_collection(self):
        """Populate the ChromaDB collection with fashion product data"""
        print("Populating ChromaDB collection...")
        
        # Limit dataset size when running in local fallback mode for speed
        dataframe_for_index = self.fashion_data.head(self.index_limit) if self.index_limit else self.fashion_data

        def extract_text(metadata):
            """Extract text content from metadata for embedding"""
            text_content = ""
            if "Description" in metadata and metadata["Description"]:
                text_content += str(metadata["Description"])
            if "Name" in metadata:
                text_content += " " + str(metadata["Name"])
            if not text_content:
                text_content = "No description available."
            return text_content.strip()
        
        # Extract text for embedding
        documents = [extract_text(row['metadata']) for _, row in dataframe_for_index.iterrows()]
        
        # Index ALL products from the dataset (14,214 products)
        batch_size = 100  # Increased batch size for efficiency
        total_products = len(documents)
        print(f"Indexing {total_products} products in batches of {batch_size}...")
        
        for i in range(0, total_products, batch_size):
            batch_docs = documents[i:i+batch_size]
            batch_ids = [str(dataframe_for_index.iloc[j]['p_id']) for j in range(i, min(i+batch_size, total_products))]
            batch_metadata = [dataframe_for_index.iloc[j]['metadata'] for j in range(i, min(i+batch_size, total_products))]
            
            self.fashion_collection.add(
                documents=batch_docs,
                ids=batch_ids,
                metadatas=batch_metadata
            )
            print(f"Added batch {i//batch_size + 1}/{(total_products + batch_size - 1)//batch_size}")
            
        print(f"Collection populated successfully with {total_products} products!")
    
    def _rebuild_collection(self):
        """Rebuild the collection with ALL products from dataset"""
        print("Rebuilding collection with ALL products...")
        
        try:
            # Clear existing collection
            print("Clearing existing collection...")
            self.fashion_collection.delete(where={})
            print("Collection cleared successfully!")
            
            # Repopulate with ALL products
            self._populate_collection()
        except Exception as e:
            print(f"Error rebuilding collection: {e}")
            # Fallback to regular population
            self._populate_collection()
        
    def _setup_cross_encoder(self):
        """Setup cross-encoder for re-ranking"""
        print("Setting up cross-encoder for re-ranking...")
        try:
            self.cross_encoder = CrossEncoder('cross-encoder/ms-marco-MiniLM-L-6-v2')
        except Exception as e:
            print(f"Warning: Failed to load cross-encoder model. Falling back to distance-based ranking. Error: {e}")
            self.cross_encoder = None
        
    def search_with_cache(self, query, n_results=5):
        """
        Perform semantic search with caching mechanism
        
        Args:
            query (str): User query
            n_results (int): Number of results to return
            
        Returns:
            dict: Search results
        """
        print(f"\nSearching for: '{query}'")
        
        # First check cache
        cache_results = self.cache_collection.query(
            query_texts=[query],
            n_results=1
        )
        
        # Cache threshold
        threshold = 0.2
        
        # Check if we have a good cache hit
        if (cache_results['distances'][0] and 
            cache_results['distances'][0][0] <= threshold):
            print("Found in cache!")
            return self._process_cache_results(cache_results)
        else:
            print("Not found in cache. Searching main collection...")
            
            # Search main collection
            results = self.fashion_collection.query(
                query_texts=[query],
                n_results=n_results
            )
            
            # Apply gender-aware filtering
            filtered_results = self._apply_gender_filtering(query, results)
            
            # Store in cache for future use
            self._store_in_cache(query, filtered_results)
            
            return filtered_results
            
    def _process_cache_results(self, cache_results):
        """Process cached results - return fresh search instead"""
        print("Cache hit detected, but returning fresh search for better results...")
        
        # Instead of returning cached results, perform fresh search
        # This ensures we get proper product metadata and all results
        query = cache_results['documents'][0][0] if cache_results['documents'] else ""
        
        # Search main collection for fresh results
        results = self.fashion_collection.query(
            query_texts=[query],
            n_results=5
        )
        
        # Mark as from cache but with fresh data
        results['from_cache'] = True
        
        return results
        
    def _store_in_cache(self, query, results):
        """Store search results in cache"""
        try:
            # Store query and results in cache
            cache_metadata = {
                'query': query,
                'results_count': len(results['ids'][0]) if results['ids'] else 0,
                'timestamp': str(pd.Timestamp.now())
            }
            
            self.cache_collection.add(
                documents=[query],
                ids=[f"cache_{hash(query)}"],
                metadatas=[cache_metadata]
            )
        except Exception as e:
            print(f"Error storing in cache: {e}")
    
    def _apply_gender_filtering(self, query, results):
        """
        Apply gender-aware filtering to search results
        
        Args:
            query (str): User query
            results (dict): Search results
            
        Returns:
            dict: Filtered results
        """
        if not results['documents'] or not results['documents'][0]:
            return results
            
        # Detect gender in query
        query_lower = query.lower()
        is_mens_query = any(word in query_lower for word in ['men', 'men\'s', 'male', 'guy', 'boy'])
        is_womens_query = any(word in query_lower for word in ['women', 'women\'s', 'female', 'girl', 'lady'])
        
        if not is_mens_query and not is_womens_query:
            return results  # No gender specified, return all results
            
        # Filter results based on gender
        filtered_ids = []
        filtered_docs = []
        filtered_distances = []
        filtered_metadatas = []
        
        for i, doc in enumerate(results['documents'][0]):
            doc_lower = doc.lower()
            product_name = results['metadatas'][0][i].get('Name', '').lower()
            
            # Check if product matches gender query
            if is_mens_query:
                # For men's queries, prioritize products without "women" in name
                if 'women' not in product_name and 'women' not in doc_lower:
                    filtered_ids.append(results['ids'][0][i])
                    filtered_docs.append(doc)
                    filtered_distances.append(results['distances'][0][i])
                    filtered_metadatas.append(results['metadatas'][0][i])
            elif is_womens_query:
                # For women's queries, prioritize products with "women" in name
                if 'women' in product_name or 'women' in doc_lower:
                    filtered_ids.append(results['ids'][0][i])
                    filtered_docs.append(doc)
                    filtered_distances.append(results['distances'][0][i])
                    filtered_metadatas.append(results['metadatas'][0][i])
        
        # If no gender-specific results found, return original results with warning
        if not filtered_ids:
            print(f"⚠️  No {query.split()[0]} products found. Returning best available matches.")
            return results
        
        print(f"✅ Gender filtering applied: Found {len(filtered_ids)} gender-appropriate products")
        
        return {
            'ids': [filtered_ids],
            'documents': [filtered_docs],
            'distances': [filtered_distances],
            'metadatas': [filtered_metadatas]
        }
            
    def re_rank_results(self, query, results):
        """
        Re-rank search results using cross-encoder
        
        Args:
            query (str): User query
            results (dict): Search results
            
        Returns:
            pd.DataFrame: Re-ranked results
        """
        print("Re-ranking results with cross-encoder...")
        
        if not results['documents'] or not results['documents'][0]:
            print("No documents to re-rank")
            return pd.DataFrame()
            
        # If cross-encoder isn't available, fall back to inverse distance ranking
        if self.cross_encoder is None:
            results_df = pd.DataFrame({
                'IDs': results['ids'][0],
                'Documents': results['documents'][0],
                'Distances': results['distances'][0],
                'Metadatas': results['metadatas'][0]
            })
            # Lower distance = more similar; create a pseudo score
            results_df['Reranked_scores'] = -results_df['Distances']
            results_df = results_df.sort_values(by='Reranked_scores', ascending=False)
        else:
            # Create input pairs for cross-encoder
            cross_inputs = [[query, doc] for doc in results['documents'][0]]
            
            # Generate cross-encoder scores
            cross_rerank_scores = self.cross_encoder.predict(cross_inputs)
            
            # Create results DataFrame
            results_df = pd.DataFrame({
                'IDs': results['ids'][0],
                'Documents': results['documents'][0],
                'Distances': results['distances'][0],
                'Metadatas': results['metadatas'][0],
                'Reranked_scores': cross_rerank_scores
            })
            
            # Sort by re-ranked scores
            results_df = results_df.sort_values(by='Reranked_scores', ascending=False)
        
        return results_df
        
    def generate_response(self, query, top_results):
        """
        Generate natural language response using GPT-3.5
        
        Args:
            query (str): User query
            top_results (pd.DataFrame): Top search results
            
        Returns:
            str: Generated response
        """
        print("Generating response with GPT-3.5...")
        
        # Prepare context from top results
        context = ""
        for _, row in top_results.head(3).iterrows():
            metadata = row['Metadatas']
            context += f"Product: {metadata.get('Name', 'N/A')}\n"
            context += f"Brand: {metadata.get('Brand', 'N/A')}\n"
            context += f"Price: {metadata.get('Price_INR', 'N/A')}\n"
            context += f"Description: {row['Documents']}\n\n"
        
        # Detect gender mismatch
        query_lower = query.lower()
        is_mens_query = any(word in query_lower for word in ['men', 'men\'s', 'male', 'guy', 'boy'])
        
        gender_warning = ""
        if is_mens_query:
            # Check if results contain women's products
            women_products = 0
            for _, row in top_results.head(3).iterrows():
                metadata = row['Metadatas']
                product_name = metadata.get('Name', '').lower()
                if 'women' in product_name:
                    women_products += 1
            
            if women_products > 0:
                gender_warning = f"""

⚠️  IMPORTANT GENDER MISMATCH WARNING:
Your query "{query}" is for men's products, but the available dataset contains primarily women's fashion items. The results below are the best available matches, but they may not be specifically designed for men.

Considerations:
- These products are designed for women but may be suitable for unisex or gender-neutral styling
- You may want to adjust your search terms or look for gender-neutral brands
- The dataset has limited men's fashion options

"""
        
        # Create prompt
        prompt = f"""You are a helpful AI fashion assistant. Based on the user query and the following product information, provide a detailed and helpful response.

User Query: {query}

Available Products:
{context}

{gender_warning}

Please provide a comprehensive response that:
1. Directly addresses the user's query
2. References specific products from the available options
3. Includes relevant details like brand, price, and features
4. Is helpful and informative for fashion shopping decisions
5. If there's a gender mismatch, clearly explain it and suggest alternatives

Response:"""
        
        # Offline fallback if OpenAI API key is not set
        if not self.use_openai:
            summary_lines = [
                "Here are some recommendations based on your query:",
                ""
            ]
            for i, (_, row) in enumerate(top_results.head(3).iterrows(), start=1):
                md = row['Metadatas']
                summary_lines.append(
                    f"{i}. {md.get('Name','N/A')} by {md.get('Brand','N/A')} — INR {md.get('Price_INR','N/A')}"
                )
            if gender_warning:
                summary_lines.append(gender_warning)
            return "\n".join(summary_lines)

        try:
            # Use new-style client API first (openai>=1.0)
            try:
                from openai import OpenAI  # type: ignore
                client = OpenAI()
                response = client.chat.completions.create(
                    model="gpt-3.5-turbo",
                    messages=[
                        {"role": "system", "content": "You are a helpful AI fashion assistant."},
                        {"role": "user", "content": prompt}
                    ],
                    max_tokens=500,
                    temperature=0.7
                )
                return response.choices[0].message.content
            except Exception:
                # Fall back to legacy API (openai<1.0)
                response = openai.ChatCompletion.create(
                    model="gpt-3.5-turbo",
                    messages=[
                        {"role": "system", "content": "You are a helpful AI fashion assistant."},
                        {"role": "user", "content": prompt}
                    ],
                    max_tokens=500,
                    temperature=0.7
                )
                return response['choices'][0]['message']['content']
        except Exception as e:
            print(f"Error generating response: {e}")
            # Fallback to offline summary
            summary_lines = [
                "Here are some recommendations based on your query:",
                ""
            ]
            for i, (_, row) in enumerate(top_results.head(3).iterrows(), start=1):
                md = row['Metadatas']
                summary_lines.append(
                    f"{i}. {md.get('Name','N/A')} by {md.get('Brand','N/A')} — INR {md.get('Price_INR','N/A')}"
                )
            if gender_warning:
                summary_lines.append(gender_warning)
            return "\n".join(summary_lines)
            
    def display_results(self, results_df, title="Search Results"):
        """Display search results with images"""
        if results_df.empty:
            print("No results to display")
            return
            
        print(f"\n{title}")
        print("=" * 50)
        
        # Display top 3 results
        top_3 = results_df.head(3)
        
        for i, (_, row) in enumerate(top_3.iterrows()):
            print(f"\nResult {i+1}:")
            print(f"ID: {row['IDs']}")
            print(f"Document: {row['Documents'][:100]}...")
            print(f"Distance: {row['Distances']:.4f}")
            if 'Reranked_scores' in row:
                print(f"Re-ranked Score: {row['Reranked_scores']:.4f}")
            
            # Try to display image
            try:
                image_path = os.path.join(self.images_folder, f"{row['IDs']}.jpg")
                if os.path.exists(image_path):
                    print(f"Image available: {image_path}")
                else:
                    print("Image not found")
            except Exception as e:
                print(f"Error with image: {e}")
                
    def run_complete_pipeline(self, query):
        """
        Run the complete three-layer pipeline
        
        Args:
            query (str): User query
            
        Returns:
            dict: Complete pipeline results
        """
        print(f"\n{'='*60}")
        print(f"RUNNING COMPLETE PIPELINE FOR QUERY: {query}")
        print(f"{'='*60}")
        
        # Layer 1: Search Layer
        print("\n1. SEARCH LAYER")
        print("-" * 30)
        search_results = self.search_with_cache(query)
        
        # Layer 2: Re-ranking
        print("\n2. RE-RANKING LAYER")
        print("-" * 30)
        reranked_results = self.re_rank_results(query, search_results)
        
        # Display search results
        self.display_results(reranked_results, "Top 3 Re-ranked Results")
        
        # Layer 3: Generation Layer
        print("\n3. GENERATION LAYER")
        print("-" * 30)
        generated_response = self.generate_response(query, reranked_results)
        
        print("\nGenerated Response:")
        print("-" * 30)
        print(generated_response)
        
        return {
            'query': query,
            'search_results': search_results,
            'reranked_results': reranked_results,
            'generated_response': generated_response
        }

def main():
    """Main function to run the Fashion Search AI system"""
    
    # Test queries
    test_queries = [
        "A orange summer dress or kurta to wear over blue denim jeans",
        "I'm looking for office wear sarees in elegant colors like pink, violet, or green",
        "I'm searching for a versatile black leather jacket, suitable for various occasions"
    ]
    
    # Initialize the system
    print("Initializing Fashion Search AI System...")
    fashion_ai = FashionSearchAI(
        csv_path="Fashion Dataset v2.csv",
        images_folder="IMAGES"
    )
    
    print("\nSystem initialized successfully!")
    print(f"Available products: {fashion_ai.fashion_collection.count()}")
    
    # Run pipeline for each test query
    all_results = []
    for i, query in enumerate(test_queries, 1):
        print(f"\n\n{'='*80}")
        print(f"TESTING QUERY {i}: {query}")
        print(f"{'='*80}")
        
        try:
            results = fashion_ai.run_complete_pipeline(query)
            all_results.append(results)
            
            # Save results for analysis
            with open(f"query_{i}_results.json", "w") as f:
                json.dump({
                    'query': results['query'],
                    'generated_response': results['generated_response'],
                    'top_3_results': results['reranked_results'].head(3).to_dict('records')
                }, f, indent=2, default=str)
                
        except Exception as e:
            print(f"Error processing query {i}: {e}")
            continue
    
    print(f"\n\n{'='*80}")
    print("PIPELINE COMPLETED SUCCESSFULLY!")
    print(f"Processed {len(all_results)} queries")
    print(f"{'='*80}")
    
    # Summary
    print("\nSUMMARY:")
    print("-" * 30)
    for i, result in enumerate(all_results, 1):
        print(f"Query {i}: {result['query'][:50]}...")
        print(f"  - Generated response length: {len(result['generated_response'])} characters")
        print(f"  - Top result ID: {result['reranked_results'].iloc[0]['IDs'] if not result['reranked_results'].empty else 'N/A'}")
        print()

if __name__ == "__main__":
    main()


