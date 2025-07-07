
import gradio as gr
import os
from llama_index import (
    StorageContext, load_index_from_storage, ServiceContext
)
from llama_index.llms import Vllm

# Load index and model
index = load_index_from_storage("lade_chroma")
retriever = index.as_retriever(search_kwargs={"k":4})

MODEL_NAME = os.getenv("MODEL_NAME", "mistral-7b-instruct-v0.3-gptq")
llm = Vllm(openai_base_url="http://localhost:8000", model=MODEL_NAME, temperature=0.0)
service_context = ServiceContext.from_defaults(llm=llm)

def answer_fn(q):
    ctx_nodes = retriever.retrieve(q)
    context = "\n".join(n.node.text for n in ctx_nodes)
    prompt = f"Answer the question based on context:\n{context}\nQuestion: {q}"
    return llm.complete(prompt, max_tokens=512)["choices"][0]["text"]

with gr.Blocks(title="LogisticGPT Demo") as demo:
    gr.Markdown("# LogisticGPT – RAG demo (Tier model: " + MODEL_NAME + ")")
    with gr.Tab("Ask a Question"):
        inp = gr.Textbox(label="Your logistics question")
        out = gr.Markdown()
        inp.submit(answer_fn, inp, out)
    with gr.Tab("Generate Report"):
        gr.Markdown("_TODO: implement report generation function_")
demo.launch()
