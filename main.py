# --- Imports ---
import sys
import time
import traceback
from fastapi import FastAPI, HTTPException, Depends, Request
from fastapi.middleware.cors import CORSMiddleware
import uvicorn
import os
from dotenv import load_dotenv
import json
import pandas as pd
from pinecone import Pinecone
from sentence_transformers import SentenceTransformer
import torch
from contextlib import asynccontextmanager
from pydantic import BaseModel
import ast

# --- LangChain & Hugging Face Imports ---
from langchain_core.prompts import PromptTemplate
from huggingface_hub import InferenceClient # <-- Import direct client

# --- Static File Imports (FOR FIX) ---
from fastapi.staticfiles import StaticFiles
from fastapi.responses import FileResponse 

# --- Load environment variables ---
load_dotenv()
PINECONE_API_KEY = os.getenv("PINECONE_API_KEY")
HUGGINGFACEHUB_API_TOKEN = os.getenv("HUGGINGFACEHUB_API_TOKEN")

# --- Helper Function ---
def parse_image_urls(image_string):
    valid_urls = []
    if not isinstance(image_string, str): return valid_urls
    try:
        image_list = ast.literal_eval(image_string);
        if isinstance(image_list, list):
            for url in image_list:
                if isinstance(url, str) and url.strip().startswith(('http://', 'https://')): valid_urls.append(url.strip())
    except: pass
    return valid_urls

# --- Lifespan Function ---
@asynccontextmanager
async def lifespan(app: FastAPI):
    # STARTUP
    print("--- Lifespan Event: Startup ---"); print("--- Loading models and data ---")
    app.state.pinecone_index = None; app.state.embedding_model = None
    app.state.product_data = None; app.state.prompt_template = None; app.state.hf_client = None

    # 1. Load CLIP Model
    try:
        model_name = "sentence-transformers/clip-ViT-B-32"; device = 'cuda' if torch.cuda.is_available() else 'cpu'
        print(f"Loading embedding model '{model_name}' onto device: {device}...")
        app.state.embedding_model = SentenceTransformer(model_name, device=device)
        print(f"Embedding model loaded. Type: {type(app.state.embedding_model)}")
    except Exception as e: print(f"!!! ERROR loading embedding model: {e}")

    # 2. Connect to Pinecone
    try:
        print("Connecting to Pinecone..."); pc = Pinecone(api_key=PINECONE_API_KEY)
        index_name = "product-recommender-clip"; index_description = pc.describe_index(index_name)
        index_host = index_description.host; print(f"Connecting to Pinecone index host: {index_host}...")
        app.state.pinecone_index = pc.Index(host=index_host); print("Pinecone connection successful.")
    except Exception as e: print(f"!!! ERROR connecting to Pinecone index '{index_name}': {e}")

    # 3. Load Product Data
    try:
        # 'intern_data_ikarus.csv' is in the root
        csv_file_path = "intern_data_ikarus.csv"; print(f"Loading product data from {csv_file_path}...")
        df = pd.read_csv(csv_file_path); df['image_url_list'] = df['images'].apply(parse_image_urls)
        df.set_index('uniq_id', inplace=True)
        for col in ['title', 'brand', 'description', 'categories', 'material', 'color']:
            if col in df.columns: df[col] = df[col].fillna('')
        app.state.product_data = df; print(f"Product data loaded. Shape: {app.state.product_data.shape}")
    except Exception as e: print(f"!!! ERROR loading product data: {e}")

    # 4. Initialize GenAI Prompt Template and Client
    try:
        print("Initializing GenAI Prompt Template and Client...")
        if not HUGGINGFACEHUB_API_TOKEN: raise ValueError("HUGGINGFACEHUB_API_TOKEN not set.")
        template = """Generate a short, appealing, and creative product description (max 2-3 sentences) for the following furniture item. Focus on its style and potential use.

        Product Name: {title} Brand: {brand} Category: {category} Material: {material} Color: {color}

        Creative Description:"""
        app.state.prompt_template = PromptTemplate(template=template, input_variables=["title", "brand", "category", "material", "color"])
        print("Prompt template created.")
        app.state.hf_client = InferenceClient(token=HUGGINGFACEHUB_API_TOKEN)
        print(f"Hugging Face Inference Client initialized. Type: {type(app.state.hf_client)}")
    except Exception as e: print(f"!!! ERROR initializing GenAI components: {e}")

    print("--- Lifespan startup complete ---"); yield
    # SHUTDOWN
    print("--- Lifespan Event: Shutdown ---")

# --- Initialize FastAPI app ---
app = FastAPI(title="Product Recommendation API", version="0.1.0", lifespan=lifespan)

# --- CORS ---
# This list is correct and allows  local server to work
origins = [
    "http://localhost", 
    "http://localhost:3000", 
    "http://localhost:5173",
    "http://127.0.0.1:5500"  
]
app.add_middleware(CORSMiddleware, allow_origins=origins, allow_credentials=True, allow_methods=["*"], allow_headers=["*"])

# --- Pydantic Models ---
class QueryRequest(BaseModel): query: str; top_k: int = 5

# --- Dependencies ---
def get_pinecone_index(request: Request):
    if not hasattr(request.app.state, 'pinecone_index') or request.app.state.pinecone_index is None: raise HTTPException(503, "Pinecone index not ready")
    return request.app.state.pinecone_index
def get_embedding_model(request: Request):
    if not hasattr(request.app.state, 'embedding_model') or request.app.state.embedding_model is None: raise HTTPException(503, "Embedding model not ready")
    return request.app.state.embedding_model
def get_product_data(request: Request):
    if not hasattr(request.app.state, 'product_data') or request.app.state.product_data is None: raise HTTPException(503, "Product data not ready")
    return request.app.state.product_data
def get_prompt_template(request: Request):
    if not hasattr(request.app.state, 'prompt_template') or request.app.state.prompt_template is None: raise HTTPException(503, "Prompt template not ready")
    return request.app.state.prompt_template
def get_hf_client(request: Request):
    if not hasattr(request.app.state, 'hf_client') or request.app.state.hf_client is None: raise HTTPException(503, "HF client not ready")
    return request.app.state.hf_client

# --- API Endpoints ---
# NOTE: The root path "/" is now handled by the static file serving below
@app.get("/api-check")
async def read_root(): return {"message": "Welcome to the API!"}

@app.get("/analytics")
async def get_analytics_data():
    try:
        # Assumes 'analytics_data.json' is in the root
        with open("analytics_data.json", "r") as f: data = json.load(f); return data
    except Exception as e: raise HTTPException(500, f"Error reading analytics: {e}")

@app.post("/recommend")
def post_recommendations(
    request_body: QueryRequest, request: Request,
    index: Pinecone = Depends(get_pinecone_index),
    model: SentenceTransformer = Depends(get_embedding_model),
    data: pd.DataFrame = Depends(get_product_data),
    prompt_template: PromptTemplate = Depends(get_prompt_template),
    hf_client: InferenceClient = Depends(get_hf_client)
):
    query_text = request_body.query; top_k = request_body.top_k
    print(f"Received query: '{query_text}', top_k: {top_k}")
    if not query_text: raise HTTPException(400, "Query empty.")

    try:
        print("Embedding query..."); query_embedding = model.encode([query_text]).tolist()[0]
        print("Querying Pinecone..."); query_results = index.query(vector=query_embedding, top_k=top_k, include_values=False)
        recommendations = []
        repo_id = "google/flan-t5-small"

        if query_results.matches:
            print("Fetching details and generating descriptions...")
            for match in query_results.matches:
                product_id = match.id; score = match.score
                gen_desc = "Description generation failed due to a runtime error." # Default message
                try:
                    product_info = data.loc[product_id]
                    title = product_info.get('title', ''); brand = product_info.get('brand', '')
                    category = product_info.get('categories', ''); material = product_info.get('material', '')
                    color = product_info.get('color', ''); image_urls = product_info.get('image_url_list', [])
                    first_image = image_urls[0] if isinstance(image_urls, list) and image_urls else None

                    # --- Attempt to generate description ---
                    try:
                        print(f"  Attempting to generate description for: {title[:30]}...")
                        llm_input = {"title": title, "brand": brand, "category": category, "material": material, "color": color}
                        formatted_prompt = prompt_template.format(**llm_input)
                        
                        # Direct call to client
                        generated_output = hf_client.text_generation(
                            prompt=formatted_prompt, model=repo_id, max_new_tokens=100, temperature=0.7
                        )
                        if isinstance(generated_output, str) and generated_output:
                            gen_desc = generated_output
                            print(f"  Generated: {gen_desc[:60]}...")
                        else:
                            print("  GenAI returned empty or invalid response.")
                    except StopIteration as si:
                        print(f"!!! Caught StopIteration for ID {product_id}. This is a known issue. Skipping generation.")
                    except Exception as llm_e:
                        print(f"!!! HF Client Error for ID {product_id}: {llm_e.__class__.__name__}: {llm_e}")

                    score_val = float(score) if score is not None and not pd.isna(score) else None
                    price = product_info.get('price'); price_val = price if pd.notna(price) else "N/A"
                    title_val = title if pd.notna(title) else "N/A"

                    recommendations.append({
                        "id": product_id, "title": title_val, "price": price_val,
                        "imageUrl": first_image, "score": score_val,
                        "generated_description": gen_desc.strip()
                    })
                except KeyError: print(f"Warn: ID '{product_id}' not in DataFrame.")
                except Exception as e:
                    print(f"Warn: Error processing ID '{product_id}' AFTER LLM attempt: {e.__class__.__name__}: {e}")

        print(f"Returning {len(recommendations)} recommendations.")
        return {"recommendations": recommendations}

    except Exception as e:
        print(f"!!! TOP LEVEL ERROR during recommendation: {e}"); traceback.print_Texc()
        raise HTTPException(status_code=500, detail=f"Failed recommendations: {str(e)}")

# --- Deployment/Static File Configuration  ---

# 1. Serve the main index.html for the root route "/"
@app.get("/", include_in_schema=False)
async def serve_frontend_root():
    # Assumes 'index.html' is in the root
    html_file = "index.html"
    if not os.path.exists(html_file):
        print(f"--- WARNING: index.html not found at: {html_file} ---")
        return {"message": "Welcome to the API. Frontend 'index.html' not found."}
    return FileResponse(html_file)

# 2. Serve the main index.html for all other routes
@app.get("/{full_path:path}", include_in_schema=False)
async def serve_frontend_catchall(request: Request, full_path: str):
    """Serve the single-page application (SPA) entry point."""
    html_file = "index.html"
    if not os.path.exists(html_file):
        print(f"--- WARNING: index.html not found at: {html_file} ---")
        return {"message": "Welcome to the API. Frontend 'index.html' not found."}
    return FileResponse(html_file)


if __name__ == "__main__":
    # The 'PORT' environment variable is set by Render.
    # It falls back to 8000 for local development.
    port = int(os.environ.get("PORT", 8000))
    print(f"Preparing to start FastAPI server on port {port}...")
    try:
        uvicorn.run(app, host="0.0.0.0", port=port)
    except Exception as e: 
        print(f"!!! Error starting Uvicorn server: {e}")
