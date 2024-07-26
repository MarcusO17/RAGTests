import pandas
import chromadb
from llama_index.core import Document   
from llama_index.vector_stores.chroma import ChromaVectorStore
from llama_index.core.ingestion import IngestionPipeline
from llama_index.core import StorageContext, SimpleDirectoryReader
from llama_index.llms.llama_cpp import LlamaCPP
from llama_index.core import VectorStoreIndex
from llama_index.core import PromptTemplate
from llama_index.core import Settings
from llama_index.embeddings.huggingface import HuggingFaceEmbedding
from llama_index.postprocessor.flag_embedding_reranker import (
    FlagEmbeddingReranker,
)

# Loading LLM
llm = LlamaCPP(
    model_path="model\stablelm-zephyr-3b.Q5_K_M.gguf",
    temperature=0.5,
    max_new_tokens=1024,
    context_window=4096,
    #generate_kwargs={"stop":['<|endoftext|>']},
    model_kwargs={"n_gpu_layers": -1},  # if compiled to use GPU
    verbose=True,
)

## Loading Embedding Model
embed_model = HuggingFaceEmbedding(
    model_name="Snowflake/snowflake-arctic-embed-s"
)

#Loading Reranker
rerank = FlagEmbeddingReranker(model="mixedbread-ai/mxbai-embed-large-v1", top_n=5)


Settings.embed_model = embed_model
Settings.llm = llm

#Vector Storage init
db = chromadb.EphemeralClient()
chroma_collection = db.get_or_create_collection("CEM5011")
vector_store = ChromaVectorStore(chroma_collection=chroma_collection)
storage_context = StorageContext.from_defaults(vector_store=vector_store)

documents = SimpleDirectoryReader(input_dir='papers').load_data()
pipeline = IngestionPipeline(
    transformations=[
    ],
    vector_store=vector_store,
)

pipeline.run(documents=documents)

index = VectorStoreIndex.from_documents(
    documents, storage_context=storage_context,similarity_top_k=10, node_postprocessors=[rerank]
) 

query_engine = index.as_query_engine(streaming=True,similarity_top_k=7)


qa_prompt_template_str = """<|user|>Context information is below.
---------------------
{context_str}
---------------------
Given the context information and not prior knowledge, answer the query based on the context given ONLY else you will be penalized. Provide your answer in an easy to read and understand format.
Further elaborate your answer by finding examples or information within the context if possible.
Query: {query_str}
Answer:
<|endoftext|>
<|assistant|>
"""

qa_prompt_template = PromptTemplate(qa_prompt_template_str)
query_engine.update_prompts(
    {"response_synthesizer:text_qa_template":qa_prompt_template}
)

response = query_engine.query("What are bonds?")
response.print_response_stream()
