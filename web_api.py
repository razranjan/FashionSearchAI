#!/usr/bin/env python
"""
Fashion Search AI - Web API Interface
=====================================

A simple Flask web API to interact with the Fashion Search AI system.
Provides endpoints for searching and viewing results.
"""

from flask import Flask, request, jsonify, render_template_string
from fashion_search_ai import FashionSearchAI
import json
import os

app = Flask(__name__)

# Initialize the Fashion Search AI system
print("Initializing Fashion Search AI system...")
fashion_ai = FashionSearchAI(
    csv_path="Fashion Dataset v2.csv",
    images_folder="IMAGES"
)
print("System ready!")

@app.route('/')
def home():
    """Home page with search interface"""
    html = """
    <!DOCTYPE html>
    <html>
    <head>
        <title>Fashion Search AI</title>
        <style>
            body { font-family: Arial, sans-serif; max-width: 1200px; margin: 0 auto; padding: 20px; background: #f8f9fa; }
            .search-box { background: white; padding: 25px; border-radius: 12px; margin-bottom: 25px; box-shadow: 0 2px 10px rgba(0,0,0,0.1); }
            .search-input { width: 75%; padding: 15px; font-size: 16px; border: 2px solid #e9ecef; border-radius: 8px; margin-right: 15px; }
            .search-button { padding: 15px 25px; background: linear-gradient(135deg, #667eea 0%, #764ba2 100%); color: white; border: none; border-radius: 8px; cursor: pointer; font-size: 16px; font-weight: bold; }
            .search-button:hover { transform: translateY(-2px); box-shadow: 0 4px 15px rgba(0,0,0,0.2); }
            .results { margin-top: 25px; }
            .product-card { border: 1px solid #dee2e6; padding: 20px; margin: 15px 0; border-radius: 12px; background: white; box-shadow: 0 2px 8px rgba(0,0,0,0.1); }
            .product-header { display: flex; align-items: center; margin-bottom: 15px; }
            .product-image { width: 120px; height: 120px; border-radius: 8px; margin-right: 20px; object-fit: cover; border: 2px solid #e9ecef; }
            .placeholder-image { background: linear-gradient(135deg, #f8f9fa 0%, #e9ecef 100%); display: flex; align-items: center; justify-content: center; color: #6c757d; font-size: 12px; text-align: center; }
            .product-id { font-weight: bold; color: #007bff; font-size: 18px; }
            .score { color: #28a745; font-weight: bold; }
            .metadata { background: #f8f9fa; padding: 15px; border-radius: 8px; margin: 15px 0; border-left: 4px solid #007bff; }
            .generated-response { background: linear-gradient(135deg, #e3f2fd 0%, #bbdefb 100%); padding: 20px; border-radius: 12px; border-left: 4px solid #2196f3; margin-top: 20px; }
            .stats { display: flex; gap: 20px; margin-bottom: 20px; }
            .stat-box { background: white; padding: 15px; border-radius: 8px; text-align: center; flex: 1; box-shadow: 0 2px 5px rgba(0,0,0,0.1); }
            .stat-number { font-size: 24px; font-weight: bold; color: #007bff; }
            .stat-label { color: #6c757d; font-size: 14px; }
            .error-message { background: #fff3cd; border: 1px solid #ffeaa7; color: #856404; padding: 15px; border-radius: 8px; margin: 15px 0; }
            .cache-info { background: #e8f5e8; border: 1px solid #c3e6c3; color: #155724; padding: 10px; border-radius: 8px; margin: 10px 0; font-size: 14px; }
            .fresh-search { background: #fff3cd; border: 1px solid #ffeaa7; color: #856404; padding: 10px; border-radius: 8px; margin: 10px 0; font-size: 14px; }
        </style>
    </head>
    <body>
        <h1 style="text-align: center; color: #2c3e50; margin-bottom: 30px;">🎯 Fashion Search AI System</h1>
        
        <div class="search-box">
            <h3 style="color: #2c3e50; margin-bottom: 20px;">Search Fashion Products</h3>
            <div style="display: flex; align-items: center;">
                <input type="text" id="queryInput" class="search-input" 
                       placeholder="Enter your fashion query (e.g., 'orange summer dress', 'yellow kurta for puja')">
                <button onclick="searchProducts()" class="search-button">🔍 Search</button>
            </div>
            <div style="margin-top: 15px;">
                <label><input type="checkbox" id="forceFreshSearch"> Force fresh search (bypass cache)</label>
            </div>
        </div>
        
        <div id="stats" class="stats">
            <div class="stat-box">
                <div class="stat-number" id="totalProducts">-</div>
                <div class="stat-label">Total Products</div>
            </div>
            <div class="stat-box">
                <div class="stat-number" id="embeddedDocs">-</div>
                <div class="stat-label">Embedded Documents</div>
            </div>
            <div class="stat-box">
                <div class="stat-number" id="cacheSize">-</div>
                <div class="stat-label">Cache Size</div>
            </div>
        </div>
        
        <div id="results" class="results"></div>
        
        <script>
        // Load system stats on page load
        window.onload = function() {
            loadSystemStats();
        };
        
        async function loadSystemStats() {
            try {
                const response = await fetch('/api/status');
                const data = await response.json();
                
                document.getElementById('totalProducts').textContent = data.total_products.toLocaleString();
                document.getElementById('embeddedDocs').textContent = data.embedded_documents;
                document.getElementById('cacheSize').textContent = data.cache_size;
            } catch (error) {
                console.error('Error loading stats:', error);
            }
        }
        
        async function searchProducts() {
            const query = document.getElementById('queryInput').value;
            const forceFresh = document.getElementById('forceFreshSearch').checked;
            
            if (!query) {
                alert('Please enter a search query');
                return;
            }
            
            const resultsDiv = document.getElementById('results');
            resultsDiv.innerHTML = '<p style="text-align: center; color: #6c757d;">🔍 Searching for fashion products...</p>';
            
            try {
                const response = await fetch('/api/search', {
                    method: 'POST',
                    headers: {'Content-Type': 'application/json'},
                    body: JSON.stringify({
                        query: query,
                        force_fresh: forceFresh
                    })
                });
                
                const data = await response.json();
                displayResults(data);
            } catch (error) {
                resultsDiv.innerHTML = '<div class="error-message">❌ Error: ' + error.message + '</div>';
            }
        }
        
        function displayResults(data) {
            const resultsDiv = document.getElementById('results');
            
            let html = '<h2 style="color: #2c3e50; margin-bottom: 20px;">🎯 Search Results</h2>';
            
            // Show search info
            if (data.from_cache) {
                html += '<div class="cache-info">💾 Results loaded from cache for faster response</div>';
            } else {
                html += '<div class="fresh-search">🆕 Fresh search performed</div>';
            }
            
            // Display top 3 results
            if (data.reranked_results && data.reranked_results.length > 0) {
                html += '<h3 style="color: #495057; margin-bottom: 15px;">Top 3 Products Found:</h3>';
                
                data.reranked_results.forEach((result, index) => {
                    const imagePath = `/api/image/${result.IDs}`;
                    
                    html += `
                        <div class="product-card">
                            <div class="product-header">
                                <div class="product-image" id="img-${result.IDs}">
                                    <img src="${imagePath}" alt="Product Image" style="width: 100%; height: 100%; object-fit: cover; border-radius: 8px;" onerror="this.parentElement.innerHTML='🖼️<br>Image<br>Not Available'; this.parentElement.classList.add('placeholder-image');">
                                </div>
                                <div style="flex: 1;">
                                    <h4 style="color: #2c3e50; margin: 0 0 10px 0;">Result ${index + 1}</h4>
                                    <p class="product-id">Product ID: ${result.IDs}</p>
                                    <p><strong>Description:</strong> ${result.Documents.substring(0, 150)}${result.Documents.length > 150 ? '...' : ''}</p>
                                    <p><strong>Semantic Distance:</strong> <span class="score">${result.Distances.toFixed(4)}</span></p>
                                    <p><strong>Re-ranking Score:</strong> <span class="score">${result.Reranked_scores.toFixed(4)}</span></p>
                                </div>
                            </div>
                            
                            <div class="metadata">
                                <h5 style="color: #495057; margin: 0 0 10px 0;">📋 Product Details:</h5>
                                <div style="display: grid; grid-template-columns: repeat(auto-fit, minmax(200px, 1fr)); gap: 10px;">
                                    <p><strong>Name:</strong> ${result.Metadatas.Name || 'N/A'}</p>
                                    <p><strong>Brand:</strong> ${result.Metadatas.Brand || 'N/A'}</p>
                                    <p><strong>Price:</strong> ₹${result.Metadatas.Price_INR || 'N/A'}</p>
                                    <p><strong>Color:</strong> ${result.Metadatas.Colour || 'N/A'}</p>
                                    <p><strong>Rating:</strong> ${result.Metadatas.Rating || 'N/A'}/5</p>
                                    <p><strong>Type:</strong> ${result.Metadatas.Product_type || 'N/A'}</p>
                                </div>
                            </div>
                        </div>
                    `;
                    
                });
            } else {
                html += '<div class="error-message">❌ No products found for your query</div>';
            }
            
            // Display generated response
            if (data.generated_response) {
                html += `
                    <div class="generated-response">
                        <h3 style="color: #1976d2; margin: 0 0 15px 0;">🤖 AI Generated Recommendation:</h3>
                        <p style="line-height: 1.6; margin: 0;">${data.generated_response.replace(/\\n/g, '<br>')}</p>
                    </div>
                `;
            }
            
            resultsDiv.innerHTML = html;
        }
        

        </script>
    </body>
    </html>
    """
    return html

@app.route('/api/image/<product_id>')
def get_product_image(product_id):
    """Serve product images if they exist"""
    try:
        image_path = os.path.join('images', f'{product_id}.jpg')
        if os.path.exists(image_path):
            from flask import send_file
            return send_file(image_path, mimetype='image/jpeg')
        else:
            # Return a placeholder image or 404
            return jsonify({'error': 'Image not found'}), 404
    except Exception as e:
        return jsonify({'error': str(e)}), 500

@app.route('/api/search', methods=['POST'])
def search():
    """API endpoint for searching fashion products"""
    try:
        data = request.get_json()
        query = data.get('query', '')
        force_fresh = data.get('force_fresh', False)
        
        if not query:
            return jsonify({'error': 'Query is required'}), 400
        
        # Clear cache if force fresh search is requested
        if force_fresh:
            print(f"Force fresh search requested for: {query}")
            # Clear the specific query from cache
            try:
                fashion_ai.cache_collection.delete(ids=[f"cache_{hash(query)}"])
                print("Cache cleared for fresh search")
            except:
                pass
        
        # Run the complete pipeline
        results = fashion_ai.run_complete_pipeline(query)
        
        # Convert results to JSON-serializable format
        reranked_results = []
        if not results['reranked_results'].empty:
            reranked_results = results['reranked_results'].head(3).to_dict('records')
        
        # Check if results came from cache
        from_cache = False
        if 'from_cache' in results:
            from_cache = results['from_cache']
        
        return jsonify({
            'query': results['query'],
            'reranked_results': reranked_results,
            'generated_response': results['generated_response'],
            'from_cache': from_cache,
            'total_results': len(reranked_results)
        })
        
    except Exception as e:
        return jsonify({'error': str(e)}), 500

@app.route('/api/results/<int:query_number>')
def get_results(query_number):
    """Get results for a specific query number"""
    try:
        filename = f"query_{query_number}_results.json"
        if os.path.exists(filename):
            with open(filename, 'r') as f:
                results = json.load(f)
            return jsonify(results)
        else:
            return jsonify({'error': f'Results for query {query_number} not found'}), 404
    except Exception as e:
        return jsonify({'error': str(e)}), 500

@app.route('/api/status')
def system_status():
    """Get system status and statistics"""
    try:
        status = {
            'total_products': len(fashion_ai.fashion_data),
            'embedded_documents': fashion_ai.fashion_collection.count(),
            'cache_size': fashion_ai.cache_collection.count(),
            'system_ready': True
        }
        return jsonify(status)
    except Exception as e:
        return jsonify({'error': str(e), 'system_ready': False}), 500

@app.route('/api/clear-cache', methods=['POST'])
def clear_cache():
    """Clear all cache"""
    try:
        fashion_ai.cache_collection.delete(where={})
        return jsonify({'message': 'Cache cleared successfully'})
    except Exception as e:
        return jsonify({'error': str(e)}), 500

if __name__ == '__main__':
    print("Starting Fashion Search AI Web API...")
    print("Open your browser and go to: http://localhost:5000")
    print("API endpoints:")
    print("  - GET  /                    - Web interface")
    print("  - POST /api/search          - Search products")
    print("  - GET  /api/results/<num>   - Get query results")
    print("  - GET  /api/status          - System status")
    print("  - GET  /api/image/<id>      - Get product image")
    print("  - POST /api/clear-cache     - Clear cache")
    
    app.run(debug=True, host='0.0.0.0', port=5000)
