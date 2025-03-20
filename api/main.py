from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from typing import Dict, Any
import os
import sys
import json

from .tree_engine import (
    HSCodeClassifier,
    build_and_save_tree,
    OPENAI_API_KEY,
    HSCodeTree,
    HSNode,
    ClarificationQuestion
)

sys.modules["__main__"].HSCodeTree = HSCodeTree
sys.modules["__main__"].HSNode = HSNode

app = FastAPI()

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
    try:

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

        chapter_code = classifier.determine_chapter(product)
        print(f"Chapter code: {chapter_code}")

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

        if len(state["current_code"]) == 2:
            state["stage"] = "heading"
        elif len(state["current_code"]) == 4:
            state["stage"] = "subheading"
        else:
            state["stage"] = "tariff"

        options = classifier.get_children(state["current_code"])
        state["options"] = options

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
                "classification": state["selection"],
                "final_code": final_code,
                "full_path": full_path,
                "steps": state.get("steps", []),
                "conversation": state.get("conversation", []),
                "explanation": explanation,
                "is_complete": True
            }

        prompt = classifier._create_prompt(state["current_query"], state["current_code"], options)
        response = classifier._call_openai(prompt)
        selected_code, is_final, llm_confidence = classifier._parse_response(response, options)

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

            options_next = classifier.get_children(state["current_code"])
            if not options_next:
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
                    "classification": state["selection"],
                    "final_code": final_code,
                    "full_path": full_path,
                    "steps": state.get("steps", []),
                    "conversation": state.get("conversation", []),
                    "explanation": explanation,
                    "is_complete": True
                }
            else:
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

            question_obj = classifier.generate_clarification_question(
                state["current_query"], state["current_code"], state["stage"], options
            )
            state["pending_question"] = question_obj.to_dict()
            return {
                "final": False,
                "clarification_question": question_obj.to_dict(),
                "state": state
            }

    except HTTPException as e:
        print(f"HTTP exception occurred: {e}")
        return {"error": f"HTTP error: {e.detail}"}
    except Exception as e:
        print(f"Error during classification process: {e}")
        return {"error": f"An error occurred: {e}"}

@app.post("/classify/continue")
def classify_continue_endpoint(request: FollowUpRequest):
    try:
        cwd = os.getcwd()
        json_file_path = os.path.join(cwd, 'api', 'hts_tree_output.json')
        classifier = HSCodeClassifier(json_file_path, OPENAI_API_KEY)
        classifier = HSCodeClassifier(json_file_path, OPENAI_API_KEY)

        state = request.state
        pending_question_dict = state.get("pending_question")
        options = state.get("options")

        if not pending_question_dict and options:
            new_question_obj = classifier.generate_clarification_question(
                state["current_query"], state["current_code"], state["stage"], options
            )
            state["pending_question"] = new_question_obj.to_dict()
            return {
                "final": False,
                "clarification_question": new_question_obj.to_dict(),
                "state": state
            }

        question_obj = ClarificationQuestion()
        question_obj.question_type = pending_question_dict.get("question_type", "text")
        question_obj.question_text = pending_question_dict.get("question_text", "")
        question_obj.options = pending_question_dict.get("options", [])
        question_obj.metadata = pending_question_dict.get("metadata", {})

        updated_query, best_match = classifier.process_answer(
            state.get("product"), question_obj, request.answer, options
        )
        state["current_query"] = updated_query
        state["questions_asked"] = state.get("questions_asked", 0) + 1
        state.setdefault("conversation", []).append({
            "question": question_obj.question_text,
            "answer": request.answer
        })

        if best_match and best_match.get("confidence", 0) > 0.7:
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

            options_next = classifier.get_children(state["current_code"])
            if not options_next:
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
                    "classification": state["selection"],
                    "final_code": final_code,
                    "full_path": full_path,
                    "steps": state.get("steps", []),
                    "conversation": state.get("conversation", []),
                    "explanation": explanation,
                    "is_complete": True
                }
            else:
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
            return {
                "final": False,
                "error": "Could not determine selection. Please try again.",
                "state": state
            }
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/classify")
def classify_endpoint(request: ClassifyRequest):
    try:
        cwd = os.getcwd()
        json_file_path = os.path.join(cwd, 'api', 'hts_tree_output.json')
        classifier = HSCodeClassifier(json_file_path, OPENAI_API_KEY)
        if request.interactive:
            result = start_classification(classifier, request.product, request.max_questions)
        else:
            result = classifier.classify(request.product)
        return result
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/build")
def build_endpoint():
    try:
        tree_output_path = "/tmp/hs_codes_output.json"
        json_file_path = "api/hs_codes.json"
        tree = build_and_save_tree(json_file_path, tree_output_path)
        if tree is None:
            raise HTTPException(status_code=500, detail="Failed to build tree")
        total_nodes = len(tree.code_index)
        max_depth = tree.get_max_depth()  
        chapters = [child for child in tree.root.children if child.htsno and len(child.htsno.strip()) == 2]
        stats = {
            "total_nodes": total_nodes,
            "indexed_codes": len(tree.code_index),
            "max_depth": max_depth,
            "chapters_count": len(chapters),
            "last_updated": tree.last_updated.isoformat()
        }
        return stats
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/")
def read_root():
    return {"message": "Welcome to the HS Code Classifier API!"}