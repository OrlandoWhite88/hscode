from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
import json
import os
import logging
from .tree_engine import HTSTree

app = FastAPI()
logger = logging.getLogger("main")
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
cwd = os.getcwd()
hts_data_file = os.path.join(cwd, 'api', 'hts_data.json')
if not os.path.exists(hts_data_file):
    logger.error(f"File not found: {hts_data_file}")
    raise FileNotFoundError(f"File not found: {hts_data_file}")

with open(hts_data_file, "r", encoding="utf-8") as f:
    hts_data = json.load(f)

hts_tree = HTSTree()
hts_tree.build_from_json(hts_data)
logger.info(f"Loaded HTS tree with {len(hts_tree.code_index)} HTS codes and {len(hts_tree.node_index)} total nodes.")

class ClassifyRequest(BaseModel):
    product: str
    interactive: bool = True
    max_questions: int = 3

class ContinueRequest(BaseModel):
    state: dict
    answer: str

@app.post("/classify")
def classify_endpoint(request: ClassifyRequest):
    """
    Start a new classification session for the given product.
    Returns either a final classification or a clarification question with state.
    """
    try:
        result = hts_tree.start_classification(
            product=request.product,
            interactive=request.interactive,
            max_questions=request.max_questions
        )
        return result
    except Exception as e:
        logger.error(f"Error in /classify: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/classify/continue")
def classify_continue_endpoint(request: ContinueRequest):
    """
    Continue a classification session by answering the last question.
    """
    try:
        result = hts_tree.continue_classification(
            state=request.state,
            answer=request.answer,
            interactive=True,
            max_questions=request.state.get("max_questions", 3)
        )
        return result
    except Exception as e:
        logger.error(f"Error in /classify/continue: {e}")
        raise HTTPException(status_code=500, detail=str(e))
    
@app.get("/")
def root():
    """
    Root endpoint to check if the API is running.
    """
    return {"message": "HTS Classification API is running."}

