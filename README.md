# Fashion Search AI using Myntra Dataset

**Project Description:**  
This project is based on an upGrad assignment on Semantic search, comprising three layers. It utilizes the Myntra dataset obtained from Kaggle, which was modified for the project. For more detailed information, refer to the Project Report: "Fashion Search AI."

Image folder has randon few images, if one wants to refer the original dataset and Image folder, then please refer to this link: https://www.kaggle.com/datasets/djagatiya/myntra-fashion-product-dataset

## Objectives:
The main objective of this project is to develop a fashion query response system that utilizes AI models to provide detailed and user-friendly responses to fashion-related queries. The system aims to enhance user experience by generating informative and contextually relevant answers, thereby assisting users in finding fashion items based on their preferences.

## Design:
The project involves two main layers: the search layer and the generation layer. The search layer is responsible for retrieving relevant fashion items from the dataset based on keyword matching or predefined criteria. The generation layer utilizes advanced AI models like GPT-3.5 to generate detailed responses to user queries, incorporating context and generating natural language responses.

## Implementation:
The implementation involved several steps, including data preprocessing, model integration, and query response generation.

### Data Preprocessing:
- The CSV dataset was formatted to enhance data quality and readability.
- Blank entries were replaced, decimal points were standardized, and unnecessary columns were removed.
- Text columns were cleaned to remove HTML tags and extra characters.

### Model Integration:
- Advanced AI models like GPT-3.5 were integrated into the system to generate responses to user queries.
- Queries were passed through the model to generate detailed and contextually relevant responses.

### Query Response Generation:
- User queries were processed by both the search layer and the generation layer.
- The search layer retrieved relevant fashion items from the dataset.
- The generation layer utilized AI models to generate detailed responses to user queries, incorporating context and generating natural language responses.

## Testing Queries:
Here are a few queries that were used to test the model:
- Query 1: "A orange summer dress or kurta to wear over blue denim jeans."
- Query 2: "I'm looking for office wear sarees in elegant colors like pink, violet, or green..."
- Query 3: "I'm searching for a versatile black leather jacket, suitable for various occasions..."
- Query 4: "I'm seeking ethnic attire suitable for adults, with a preference for Ishin brand..."

## Challenges:
Several challenges were encountered during the implementation process, including:
- Metadata Processing
- Dataset Chunking

## Lessons Learned:
- Proper data preprocessing is crucial for ensuring data quality and readability.
- Integrating advanced AI models can significantly enhance the system's capabilities.
- Handling large datasets requires careful consideration of memory constraints and implementation of efficient data processing techniques.

## Future Scope:
- Implementing the project as a Flask web application.
- Rephrasing prompts and introducing interactive sessions with criteria-based filters.

## Conclusion:
Upon comparison of the search query outcomes from both the search and generation layers, it becomes apparent that the generation layer produces more detailed and comprehensible results. While the search layer efficiently retrieves relevant information, the generation layer significantly enhances output quality and readability, making it a preferred choice for tasks requiring detailed and user-friendly responses.

## Acknowledgment:
Thank you upGrad for the Modules, which helped in giving a building a started code, and also thankful to GPT.


## How to Run (API + UI)

### Prerequisites
- Python 3.10+
- macOS/Linux recommended
- Optional: set `OPENAI_API_KEY` in your environment to enable LLM-generated responses; without it, the app falls back to offline summarization.

### 1) Create and activate a virtual environment
```bash
python3 -m venv .venv
source .venv/bin/activate
```

### 2) Install dependencies
```bash
pip install --upgrade pip
pip install -r requirements.txt
```

### 3) Prepare data
- Ensure the CSV `Fashion Dataset v2.csv` is present in the project root.
- Ensure the images directory exists as `images/` (filenames are `<product_id>.jpg`). If missing, the UI will still work and show placeholders.

### 4) Run the web API with UI
```bash
python web_api.py
```
The server will start on `http://127.0.0.1:5000`.

Endpoints:
- `GET /` – HTML UI for running searches
- `POST /api/search` – body: `{ "query": "...", "force_fresh": false }`
- `GET /api/status` – system stats
- `GET /api/image/<product_id>` – product image
- `POST /api/clear-cache` – clear all cached queries

### 5) Running locally without OpenAI
If `OPENAI_API_KEY` is not set, the system uses Sentence Transformers for embeddings and falls back to an offline, deterministic response generator. This is suitable for local demos.

### 6) Troubleshooting
- If the UI shows dashes for stats, hard refresh (Cmd+Shift+R).
- First run may populate or rebuild the Chroma collections and can take a while.
- If `torch` wheels fail to install, install a platform-specific build from PyPI or conda-forge.

## How to Know Dependencies
- All Python dependencies are listed in `requirements.txt`. Install with `pip install -r requirements.txt`.
- If you add features, pin new packages in `requirements.txt` and re-install.

