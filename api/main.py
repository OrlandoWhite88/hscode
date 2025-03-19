from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from typing import Dict, List, Any, Union, Optional
import os
import json
from datetime import datetime

from .tree_engine import HSCodeClassifier, ClarificationQuestion

app = FastAPI()

# Add CORS middleware to allow requests from multiple origins
app.add_middleware(
    CORSMiddleware,
    allow_origins=["http://localhost:8080", "https://preview--ai-hscode-genie.lovable.app", 
                  "https://unihsdashboard.vercel.app", "https://www.uni-customs.com"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

class ClassifyRequest(BaseModel):
    product: str
    interactive: bool = True
    max_questions: int = 3

class FollowUpRequest(BaseModel):
    state: Dict[str, Any]
    answer: str

def start_classification(classifier: HSCodeClassifier, product: str, max_questions: int) -> Dict[str, Any]:
    """
    Run one step of the interactive classification.
    Returns a dictionary that either contains a final classification result or
    a clarification question along with the current state.
    """

    state = {
        "product": product,
        "step": 0,
        "questions_asked": 0,
        "current_code": "",       
        "current_query": product,  
        "conversation": [],
        "selection": {},
        "pending_question": None,
        "options": None,
        "stage": None,
        "steps": []  
    }

    # First, determine the chapter
    chapter_code = classifier.determine_chapter(product)
    if chapter_code:
        state["selection"]["chapter"] = chapter_code

        chapter_option = {
            "code": chapter_code,
            "description": classifier.chapters_map.get(int(chapter_code), "Unknown chapter")
        }
        state["steps"].append({
            "step": 0,
            "current_code": "",
            "selected_code": chapter_code,
            "options": [chapter_option],
            "llm_response": f"Determined chapter {chapter_code}"
        })
        state["current_code"] = chapter_code
        state["step"] = 1
    else:
        raise HTTPException(status_code=500, detail="Unable to determine chapter.")

    # Determine the current stage based on code length
    if len(state["current_code"]) == 2:
        state["stage"] = "heading"
    elif len(state["current_code"]) == 4:
        state["stage"] = "subheading"
    else:
        state["stage"] = "tariff"

    # Get child options for the current code
    options = classifier.get_children(state["current_code"])
    state["options"] = options

    # If we have no further options, finalize the classification
    if not options:
        final_code = state["current_code"]
        full_path = classifier._get_full_context(final_code)
        explanation = classifier.explain_classification(
            state["product"],
            state["current_query"],
            full_path,
            state.get("conversation", [])
        )
        return {
            "original_query": state["product"],
            "enriched_query": state["current_query"],
            "classification": {
                "chapter": state["selection"].get("chapter", ""),
                "heading": state["selection"].get("heading", ""),
                "subheading": state["selection"].get("subheading", ""),
                "tariff": state["selection"].get("tariff", "")
            },
            "final_code": final_code,
            "full_path": full_path,
            "steps": state.get("steps", []),
            "conversation": state.get("conversation", []),
            "explanation": explanation,
            "is_complete": True
        }

    # Create a prompt to classify the product with the available options
    prompt = classifier._create_prompt(state["current_query"], state["current_code"], options)
    response = classifier._call_openai(prompt)
    selected_code, is_final, llm_confidence = classifier._parse_response(response, options)

    # If the LLM is confident or we've reached the question limit, proceed with the selection
    if llm_confidence >= 0.9 or state["questions_asked"] >= classifier.max_questions_per_level:
        state["selection"][state["stage"]] = selected_code
        state["current_code"] = selected_code
        state["steps"].append({
            "step": state["step"],
            "current_code": state["current_code"],
            "selected_code": selected_code,
            "options": options,
            "llm_response": str(response)
        })
        state["step"] += 1

        # Check for further child options at the new level
        options_next = classifier.get_children(state["current_code"])
        if not options_next:
            # If no further options, finalize the classification
            final_code = state["current_code"]
            full_path = classifier._get_full_context(final_code)
            explanation = classifier.explain_classification(
                state["product"],
                state["current_query"],
                full_path,
                state.get("conversation", [])
            )
            return {
                "original_query": state["product"],
                "enriched_query": state["current_query"],
                "classification": {
                    "chapter": state["selection"].get("chapter", ""),
                    "heading": state["selection"].get("heading", ""),
                    "subheading": state["selection"].get("subheading", ""),
                    "tariff": state["selection"].get("tariff", "")
                },
                "final_code": final_code,
                "full_path": full_path,
                "steps": state.get("steps", []),
                "conversation": state.get("conversation", []),
                "explanation": explanation,
                "is_complete": True
            }
        else:
            # Generate a clarification question for the next level
            new_question_obj = classifier.generate_clarification_question(
                state["current_query"], state["current_code"], state["stage"], options_next
            )
            state["pending_question"] = new_question_obj.to_dict()
            state["options"] = options_next
            return {
                "final": False,
                "clarification_question": new_question_obj.to_dict(),
                "state": state
            }
    else:
        # Generate a clarification question for the current level
        question_obj = classifier.generate_clarification_question(
            state["current_query"], state["current_code"], state["stage"], options
        )
        state["pending_question"] = question_obj.to_dict()
        return {
            "final": False,
            "clarification_question": question_obj.to_dict(),
            "state": state
        }

@app.post("/classify/continue")
def classify_continue_endpoint(request: FollowUpRequest):
    """
    POST /classify/continue
    Continues an interactive classification session based on a user's answer.
    Expects:
      - state: Current state of the classification process.
      - answer: User's answer to the clarification question.
    Returns the next question or the final classification.
    """
    try:
        # Initialize the classifier with the tree
        cwd = os.getcwd()
        path = os.path.join(cwd, 'api', 'hts_tree_output.json')
        api_key = os.getenv("OPENAI_API_KEY")
        if not api_key:
            raise ValueError("OPENAI_API_KEY environment variable is not set")
        classifier = HSCodeClassifier(path, api_key)

        # Ensure state is properly parsed
        if isinstance(request.state, str):
            try:
                state = json.loads(request.state)
            except json.JSONDecodeError:
                raise HTTPException(status_code=400, detail="Invalid state format")
        else:
            state = request.state
            
        # Get pending question and options
        pending_question_dict = state.get("pending_question") if isinstance(state, dict) else None
        options = state.get("options") if isinstance(state, dict) else None

        # Generate a new question if needed
        if not pending_question_dict and options:
            if not isinstance(state, dict):
                raise HTTPException(status_code=400, detail="Invalid state format - expected dictionary")
                
            current_query = state.get("current_query")
            current_code = state.get("current_code")
            stage = state.get("stage")
            
            if not all([current_query, current_code, stage]):
                raise HTTPException(status_code=400, detail="Missing required state information")
                
            new_question_obj = classifier.generate_clarification_question(
                current_query, current_code, stage, options
            )
            state["pending_question"] = new_question_obj.to_dict()
            return {
                "final": False,
                "clarification_question": new_question_obj.to_dict(),
                "state": state
            }

        # Reconstruct the clarification question object
        question_obj = ClarificationQuestion()
        question_obj.question_type = pending_question_dict.get("question_type", "text")
        question_obj.question_text = pending_question_dict.get("question_text", "")
        question_obj.options = pending_question_dict.get("options", [])
        question_obj.metadata = pending_question_dict.get("metadata", {})

        # Validate state
        if not isinstance(state, dict):
            raise HTTPException(status_code=400, detail="Invalid state format - expected dictionary")
            
        product = state.get("product")
        if not product:
            raise HTTPException(status_code=400, detail="Missing product information in state")
            
        # Process the user's answer
        updated_query, best_match = classifier.process_answer(
            product, question_obj, request.answer, options
        )
        
        # Update state with the new information
        state["current_query"] = updated_query
        state["questions_asked"] = state.get("questions_asked", 0) + 1
        state.setdefault("conversation", []).append({
            "question": question_obj.question_text,
            "answer": request.answer
        })

        # If we have a confident match, select it and proceed
        if best_match and isinstance(best_match, dict) and best_match.get("confidence", 0) > 0.7:
            state["selection"][state["stage"]] = best_match["code"]
            state["current_code"] = best_match["code"]

            state.setdefault("steps", []).append({
                "step": state.get("step", 0),
                "current_code": state["current_code"],
                "selected_code": best_match["code"],
                "options": options,
                "llm_response": f"Selected based on user answer: {request.answer}"
            })
            state["step"] = state.get("step", 0) + 1

            # Check for further options
            options_next = classifier.get_children(state["current_code"])
            if not options_next:
                # If no further options, finalize the classification
                final_code = state["current_code"]
                full_path = classifier._get_full_context(final_code)
                explanation = classifier.explain_classification(
                    state["product"],
                    state["current_query"],
                    full_path,
                    state.get("conversation", [])
                )
                return {
                    "original_query": state["product"],
                    "enriched_query": state["current_query"],
                    "classification": {
                        "chapter": state["selection"].get("chapter", ""),
                        "heading": state["selection"].get("heading", ""),
                        "subheading": state["selection"].get("subheading", ""),
                        "tariff": state["selection"].get("tariff", "")
                    },
                    "final_code": final_code,
                    "full_path": full_path,
                    "steps": state.get("steps", []),
                    "conversation": state.get("conversation", []),
                    "explanation": explanation,
                    "is_complete": True
                }
            else:
                # Update stage based on the current code length
                if len(state["current_code"]) == 2:
                    state["stage"] = "heading"
                elif len(state["current_code"]) == 4:
                    state["stage"] = "subheading"
                else:
                    state["stage"] = "tariff"
                    
                state["options"] = options_next
                new_question_obj = classifier.generate_clarification_question(
                    state["current_query"], state["current_code"], state["stage"], options_next
                )
                state["pending_question"] = new_question_obj.to_dict()
                return {
                    "final": False,
                    "clarification_question": new_question_obj.to_dict(),
                    "state": state
                }
        else:
            # If not confident enough, use the LLM to select
            prompt = classifier._create_prompt(state["current_query"], state["current_code"], options)
            response = classifier._call_openai(prompt)
            selected_code, is_final, confidence = classifier._parse_response(response, options)
            
            if selected_code:
                state["selection"][state["stage"]] = selected_code
                state["current_code"] = selected_code

                state.setdefault("steps", []).append({
                    "step": state.get("step", 0),
                    "current_code": state["current_code"],
                    "selected_code": selected_code,
                    "options": options,
                    "llm_response": str(response)
                })
                state["step"] = state.get("step", 0) + 1

                # Check for further options
                options_next = classifier.get_children(state["current_code"])
                if not options_next:
                    # If no further options, finalize the classification
                    final_code = state["current_code"]
                    full_path = classifier._get_full_context(final_code)
                    explanation = classifier.explain_classification(
                        state["product"],
                        state["current_query"],
                        full_path,
                        state.get("conversation", [])
                    )
                    return {
                        "original_query": state["product"],
                        "enriched_query": state["current_query"],
                        "classification": {
                            "chapter": state["selection"].get("chapter", ""),
                            "heading": state["selection"].get("heading", ""),
                            "subheading": state["selection"].get("subheading", ""),
                            "tariff": state["selection"].get("tariff", "")
                        },
                        "final_code": final_code,
                        "full_path": full_path,
                        "steps": state.get("steps", []),
                        "conversation": state.get("conversation", []),
                        "explanation": explanation,
                        "is_complete": True
                    }
                else:
                    # Generate a question for the next level
                    new_question_obj = classifier.generate_clarification_question(
                        state["current_query"], state["current_code"], state["stage"], options_next
                    )
                    state["pending_question"] = new_question_obj.to_dict()
                    state["options"] = options_next
                    return {
                        "final": False,
                        "clarification_question": new_question_obj.to_dict(),
                        "state": state
                    }
            else:
                return {
                    "final": False,
                    "error": "Could not determine selection. Please try again.",
                    "state": state
                }

    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/classify")
def classify_endpoint(request: ClassifyRequest):
    """
    POST /classify
    Starts an interactive classification session.
    Expects:
      - product: Product description.
      - interactive: (Optional) Whether to use interactive mode.
      - max_questions: (Optional) Maximum questions allowed.
    Returns either a final classification or a clarification question with state information.
    """
    try:
        cwd = os.getcwd()
        path = os.path.join(cwd, 'api', 'hts_tree_output.json')
        api_key = os.getenv("OPENAI_API_KEY")
        if not api_key:
            raise ValueError("OPENAI_API_KEY environment variable is not set")
            
        classifier = HSCodeClassifier(path, api_key)
        
        if request.interactive:
            result = start_classification(classifier, request.product, request.max_questions)
        else:
            result = classifier.classify_with_questions(request.product, request.max_questions)
            
        return result
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/build")
def build_endpoint():
    """
    GET /build
    Loads the HS code tree information.
    Returns tree statistics (e.g., total nodes, indexed codes, maximum depth, chapters count).
    """
    try:
        # Load the tree directly from JSON
        cwd = os.getcwd()
        path = os.path.join(cwd, 'api', 'hts_tree_output.json')
        
        api_key = os.getenv("OPENAI_API_KEY")
        if not api_key:
            raise ValueError("OPENAI_API_KEY environment variable is not set")
        
        classifier = HSCodeClassifier(path, api_key)
        tree = classifier.tree
        
        # Get statistics
        total_nodes = len(tree.code_index) if hasattr(tree, 'code_index') else 0
        chapters_count = len(tree.root.chapters) if hasattr(tree.root, 'chapters') else 0
        
        stats = {
            "total_nodes": total_nodes,
            "chapters_count": chapters_count,
            "last_updated": datetime.now().isoformat()
        }
        return stats
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/")
def read_root():
    return {"message": "Welcome to the HS Code Classifier API!"}