import gradio as gr
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
from peft import PeftModel
from sentence_transformers import SentenceTransformer, util
import pickle
import os

print("🚀 Loading Compliance Bot...")

# Load RAG
print("Loading knowledge base...")
embedding_model = SentenceTransformer('sentence-transformers/all-MiniLM-L6-v2')
with open("vector_store/chunks.pkl", "rb") as f:
    chunks = pickle.load(f)
chunk_texts = [ch.page_content for ch in chunks]
chunk_embeddings = embedding_model.encode(chunk_texts, normalize_embeddings=True)

# Load model
print("Loading fine-tuned model...")
base_model = AutoModelForCausalLM.from_pretrained(
    "TinyLlama/TinyLlama-1.1B-Chat-v1.0",
    torch_dtype=torch.float16,
    device_map="auto",
)
model = PeftModel.from_pretrained(base_model, "./model")
tokenizer = AutoTokenizer.from_pretrained("./model")
tokenizer.pad_token = tokenizer.eos_token
model.eval()

def answer_query(query, show_sources=True):
    if not query:
        return "Please enter a question."
    
    # Retrieve
    query_emb = embedding_model.encode(query, normalize_embeddings=True)
    scores = util.cos_sim(query_emb, chunk_embeddings)[0]
    top_indices = scores.topk(min(3, len(chunks))).indices.tolist()
    
    docs = [(chunks[idx], scores[idx].item()) for idx in top_indices if scores[idx].item() > 0.2]
    
    if not docs:
        return "No relevant compliance documents found."
    
    context = "\n\n".join([f"- {doc.page_content}" for doc, _ in docs])
    sources = [doc.metadata.get('source', 'Unknown') for doc, _ in docs]
    confidence = sum(score for _, score in docs) / len(docs)
    
    # Generate
    prompt = f"<|user|>\nQuestion: {query}\nContext: {context}\n<|assistant|>\nAnswer:"
    inputs = tokenizer(prompt, return_tensors="pt", truncation=True, max_length=512).to(model.device)
    
    with torch.no_grad():
        outputs = model.generate(**inputs, max_new_tokens=300, temperature=0.1)
    
    response = tokenizer.decode(outputs[0], skip_special_tokens=True)
    if "<|assistant|>" in response:
        response = response.split("<|assistant|>")[-1].strip()
    
    if show_sources:
        response += f"\n\n---\n📚 Sources: {', '.join(sources[:2])}\n📊 Confidence: {confidence:.1%}"
    
    return response

# Gradio interface
with gr.Blocks(title="Compliance Bot", theme=gr.themes.Soft()) as demo:
    gr.Markdown("# 🏢 Enterprise Compliance Audit Bot")
    gr.Markdown("Ask questions about AWS security, GDPR, SOC2, and employee policies")
    
    with gr.Row():
        query = gr.Textbox(label="Your Question", placeholder="What are S3 encryption requirements?", lines=3)
    
    with gr.Row():
        submit = gr.Button("Ask", variant="primary")
        clear = gr.Button("Clear")
    
    show_src = gr.Checkbox(label="Show sources", value=True)
    output = gr.Markdown("Ready for your question...")
    
    submit.click(answer_query, inputs=[query, show_src], outputs=output)
    clear.click(lambda: ("", "Ready for your question..."), outputs=[query, output])
    query.submit(answer_query, inputs=[query, show_src], outputs=output)

if __name__ == "__main__":
    port = int(os.environ.get("PORT", 7860))
    demo.launch(server_name="0.0.0.0", server_port=port)
