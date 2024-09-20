from fastapi import FastAPI
from fastapi.responses import FileResponse

from toolkit.tools import OllamaRouter
from toolkit.tools import LangChainRouter

#from toolkit.tools import ChromaDBRouter
#from toolkit.tools import MongoDBRouter



ROUTER_DIR = {}
ROUTER_DIR["ollama"] = {"prefix":"/ollama", "tags":["ollama"], "endpoints":[]}
ROUTER_DIR["langchain"] = {"prefix":"/langchain", "tags":["langchain"], "endpoints":[]}

#ROUTER_DIR["chromadb"] = {"prefix":"/chromadb", "tags":["chromadb"], "endpoints":[]}
#ROUTER_DIR["mongodb"] = {"prefix":"/mongodb", "tags":["mongodb"], "endpoints":[]}

#ROUTER_DIR["litellm"] = {"prefix":"/litellm", "tags":["litellm"], "endpoints":[]}
#ROUTER_DIR["semantic_kernel"] = {"prefix":"/semantic_kernel", "tags":["semantic_kernel"], "endpoints":[]}
#ROUTER_DIR["magentic"] = {"prefix":"/magentic", "tags":["magentic"], "endpoints":[]}
#ROUTER_DIR["taskweaver"] = {"prefix":"/taskweaver", "tags":["taskweaver"], "endpoints":[]}


###
#
app = FastAPI(
    title="VTP-SYNTHESIS",
    version="0.0.1")

@app.get("/")
async def root():
    return {"message":"HELLO WORLD"}

@app.get("/dir")
async def dir():

    return ROUTER_DIR

@app.get('/favicon.ico', include_in_schema=False)
async def favicon():
    return FileResponse(path='favicon.ico')


#
###


###
#
app.include_router(OllamaRouter, prefix="/ollama", tags=["ollama"])
app.include_router(LangChainRouter, prefix="/langchain", tags=["langchain"])

# app.include_router(ChromaDBRouter, prefix="/chromadb", tags=["chromadb"])
# app.include_router(MongoDBRouter, prefix="/mongodb", tags=["mongodb"])

#
###