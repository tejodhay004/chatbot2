# main.py

from langchain_ollama.llms import OllamaLLM
from langchain.chains import RetrievalQA
from vector import retriever

# --- Initialize LLaMA model ---
model = OllamaLLM(model="llama3.2")

# --- Build Retrieval QA chain ---
qa_chain = RetrievalQA.from_chain_type(
    llm=model,
    retriever=retriever,
    chain_type="stuff",  # "map_reduce" can also be used for very long documents
    return_source_documents=True
)

# --- Interactive Q&A loop ---
while True:
    print("\n-------------------------------")
    question = input("Ask your question (q to quit): ").strip()
    if question.lower() == "q":
        print("See You Soon!")
        break

    # Run the RAG chain
    result = qa_chain({"query": question})

    # The answer
    print("\n🤖 Answer:\n", result['result'])
