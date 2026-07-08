from pydantic import BaseModel
from typing import Literal
from langchain_openai import ChatOpenAI
from langchain_core.prompts import ChatPromptTemplate
import os
class StructuredResponse(BaseModel):
    answer: str
    sources: list
    confidence: Literal["low", "medium", "high"]

def generate(query:str, retrieved_chunks:list, top_similarity_score:float) -> StructuredResponse:
    chunk_text = []
    source = []
    for i in retrieved_chunks:
        chunk_text.append(i["chunk_text"])
        source.append(i["file_path"])
    chunk_text = "\n\n".join(chunk_text)
    prompt = ChatPromptTemplate.from_template("""
    Act as a support assistant and answer the query given the context. Your answer should be solely based on provided evidence. 
    You cannot use external knowledge and any information other than the context provided to answer the query.
    Query: {query},
    Context:{chunk_text}
    """)
    llm = ChatOpenAI(api_key=os.getenv("OPENAI_API_KEY"))
    chain = prompt | llm
    response = chain.invoke({"query": query, "chunk_text":chunk_text})
    confidence = ""
    if top_similarity_score > 0.85:
        confidence = "high"
    elif top_similarity_score > 0.7:
        confidence = "medium"
    else:
        confidence = "low"
    
    result = StructuredResponse(answer=response.content, sources=source, confidence=confidence)
    return result