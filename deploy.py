import modal
from pathlib import Path

# ---- 1. Define the environment ----
image = modal.Image.debian_slim(python_version="3.11").uv_pip_install(
    "torch", "transformers", "peft", "sentence-transformers",
    "faiss-cpu", "gradio", "numpy", "accelerate", "huggingface_hub",
    "langchain-core", "fastapi", "uvicorn"
)

app = modal.App("compliance-bot")

# ---- 2. Mount your local files (model + vector store) ----
#   Assumes your repo has these folders at the root
model_mount = modal.Mount.from_local_dir(
    Path(__file__).parent / "model", remote_path="/app/model"
)
vector_mount = modal.Mount.from_local_dir(
    Path(__file__).parent / "vector_store", remote_path="/app/vector_store"
)

# ---- 3. Main inference function ----
@app.function(
    image=image,
    mounts=[model_mount, vector_mount],
    cpu=4.0,
    memory=8192,          # 8GB – enough for TinyLlama
    allow_concurrent_inputs=10
)
def answer_query(query: str) -> str:
    # This runs inside Modal – you can copy your logic from app.py
    # For brevity, we'll put a placeholder; you can replace with your real code.
    return f"Placeholder answer for: {query}"

# ---- 4. Web interface with Gradio ----
web_image = modal.Image.debian_slim(python_version="3.11").uv_pip_install("gradio", "fastapi")

@app.function(
    image=web_image,
    mounts=[model_mount, vector_mount],
    allow_concurrent_inputs=10,
)
@modal.asgi_app()
def ui():
    import gradio as gr
    from fastapi import FastAPI
    from gradio.routes import mount_gradio_app

    def wrapper(q):
        return answer_query.remote(q)

    with gr.Blocks(title="Compliance Audit Bot") as demo:
        gr.Markdown("# 🏢 Enterprise Compliance Bot")
        query = gr.Textbox(label="Your Question", lines=3)
        output = gr.Markdown("Ready...")
        query.submit(wrapper, inputs=query, outputs=output)

    web_app = FastAPI()
    return mount_gradio_app(app=web_app, blocks=demo, path="/")
