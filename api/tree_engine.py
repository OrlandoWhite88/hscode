import json
import logging
import re
import time
import openai
import os
from typing import Dict, List, Any, Optional, Tuple, Union

class ClarificationQuestion:
    def __init__(self):
        self.question_type: str = ""
        self.question_text: str = ""
        self.options: List[Dict[str, Any]] = []
        self.metadata: Dict[str, Any] = {}

class ConversationHistory:
    def __init__(self):
        self.entries: List[Dict[str, Any]] = []

    def add(self, question: str, answer: str, metadata: dict):
        self.entries.append({"question": question, "answer": answer, "metadata": metadata})

    def get_by_stage(self, stage: str) -> List[Dict[str, Any]]:
        return [e for e in self.entries if e.get("metadata", {}).get("stage") == stage]

    def format_for_prompt(self) -> str:
        s = ""
        for e in self.entries:
            s += f"Q: {e['question']}\nA: {e['answer']}\n"
        return s

    def to_list(self) -> List[Dict[str, Any]]:
        return self.entries

class HTSNode:
    """
    Represents a node in the HTS hierarchy.
    Each node can be:
      - A heading/subheading/tariff line with an HTS code
      - A group node (no htsno) with children
    """
    def __init__(self, data: Dict[str, Any], node_id: int):
        self.node_id = node_id  
        self.data = data
        self.htsno = data.get('htsno', '')
        self.indent = int(data.get('indent', '0'))
        self.description = data.get('description', '')
        self.is_superior = data.get('superior') == 'true'
        self.units = data.get('units', [])
        self.general = data.get('general', '')
        self.special = data.get('special', '')
        self.other = data.get('other', '')
        self.footnotes = data.get('footnotes', [])
        self.children: List['HTSNode'] = []
        self.parent: Optional['HTSNode'] = None  

    def add_child(self, child: 'HTSNode') -> None:
        child.parent = self
        self.children.append(child)

    def is_group_node(self) -> bool:
        return not self.htsno

    def get_chapter(self) -> Optional[str]:
        if self.htsno and len(self.htsno) >= 2:
            return self.htsno[:2]
        return None

    def get_node_type(self) -> str:
        if not self.htsno:
            return "group"
        clean_code = self.htsno.replace('.', '')
        if len(clean_code) <= 4:
            return "heading"
        elif len(clean_code) <= 6:
            return "subheading"
        elif len(clean_code) >= 10:
            return "tariff_line"
        else:
            return "intermediate"

    def to_dict(self, include_children: bool = True) -> Dict[str, Any]:
        result = {
            'node_id': self.node_id,
            'htsno': self.htsno,
            'description': self.description,
            'indent': self.indent,
            'is_superior': self.is_superior,
            'units': self.units,
            'general': self.general,
            'special': self.special,
            'other': self.other,
            'footnotes': self.footnotes,
            'node_type': self.get_node_type()
        }
        if include_children:
            result['children'] = [child.to_dict() for child in self.children]
        return result

    def __repr__(self) -> str:
        prefix = '  ' * self.indent
        if self.htsno:
            return f"{prefix}({self.node_id}) {self.htsno}: {self.description} [{len(self.children)} children]"
        else:
            return f"{prefix}({self.node_id}) [GROUP] {self.description} [{len(self.children)} children]"

class HTSTree:
    """Represents the full HTS hierarchy and the classification engine."""
    def __init__(self):

        self.root = HTSNode({'htsno': '', 'indent': '-1', 'description': 'ROOT', 'superior': None}, node_id=0)
        self.chapters: Dict[str, List[HTSNode]] = {}
        self.code_index: Dict[str, HTSNode] = {}
        self.node_index: Dict[int, HTSNode] = {}
        self.next_node_id = 1
        self.steps: List[Dict[str, Any]] = []
        self.history: ConversationHistory = ConversationHistory()
        self.max_questions_per_level = 3  

        self.chapters_map = self._init_chapters_map()

        # Get API key from environment variable
        api_key = os.environ.get("OPENAI_API_KEY")
        if not api_key:
            logging.warning("OPENAI_API_KEY environment variable not found. API calls may fail.")
            
        self.client = openai.OpenAI(api_key=api_key)

    def _init_chapters_map(self) -> Dict[int, str]:
        return {
            1: "Live animals",
            2: "Meat and edible meat offal",
            3: "Fish and crustaceans, molluscs and other aquatic invertebrates",
            4: "Dairy produce; birds' eggs; natural honey; edible products of animal origin, not elsewhere specified or included",
            5: "Products of animal origin, not elsewhere specified or included",
            6: "Live trees and other plants; bulbs, roots and the like; cut flowers and ornamental foliage",
            7: "Edible vegetables and certain roots and tubers",
            8: "Edible fruit and nuts; peel of citrus fruit or melons",
            9: "Coffee, tea, mate and spices",
            10: "Cereals",
            11: "Products of the milling industry; malt; starches; inulin; wheat gluten",
            12: "Oil seeds and oleaginous fruits; miscellaneous grains, seeds and fruit; industrial or medicinal plants; straw and fodder",
            13: "Lac; gums, resins and other vegetable saps and extracts",
            14: "Vegetable plaiting materials; vegetable products not elsewhere specified or included",
            15: "Animal or vegetable fats and oils and their cleavage products; prepared edible fats; animal or vegetable waxes",
            16: "Preparations of meat, of fish or of crustaceans, molluscs or other aquatic invertebrates",
            17: "Sugars and sugar confectionery",
            18: "Cocoa and cocoa preparations",
            19: "Preparations of cereals, flour, starch or milk; pastrycooks' products",
            20: "Preparations of vegetables, fruit, nuts or other parts of plants",
            21: "Miscellaneous edible preparations",
            22: "Beverages, spirits and vinegar",
            23: "Residues and waste from the food industries; prepared animal fodder",
            24: "Tobacco and manufactured tobacco substitutes",
            25: "Salt; sulphur; earths and stone; plastering materials, lime and cement",
            26: "Ores, slag and ash",
            27: "Mineral fuels, mineral oils and products of their distillation; bituminous substances; mineral waxes",
            28: "Inorganic chemicals; organic or inorganic compounds of precious metals, of rare-earth metals, of radioactive elements or of isotopes",
            29: "Organic chemicals",
            30: "Pharmaceutical products",
            31: "Fertilizers",
            32: "Tanning or dyeing extracts; tannins and their derivatives; dyes, pigments and other colouring matter; paints and varnishes; putty and other mastics; inks",
            33: "Essential oils and resinoids; perfumery, cosmetic or toilet preparations",
            34: "Soap, organic surface-active agents, washing preparations, lubricating preparations, artificial waxes, prepared waxes, polishing or scouring preparations, candles and similar articles, modelling pastes, 'dental waxes' and dental preparations with a basis of plaster",
            35: "Albuminoidal substances; modified starches; glues; enzymes",
            36: "Explosives; pyrotechnic products; matches; pyrophoric alloys; certain combustible preparations",
            37: "Photographic or cinematographic goods",
            38: "Miscellaneous chemical products",
            39: "Plastics and articles thereof",
            40: "Rubber and articles thereof",
            41: "Raw hides and skins (other than furskins) and leather",
            42: "Articles of leather; saddlery and harness; travel goods, handbags and similar containers; articles of animal gut (other than silkworm gut)",
            43: "Furskins and artificial fur; manufactures thereof",
            44: "Wood and articles of wood; wood charcoal",
            45: "Cork and articles of cork",
            46: "Man-made fibres and articles thereof",
            47: "Pulp of wood or of other fibrous cellulosic material; recovered (waste and scrap) paper or paperboard",
            48: "Paper and paperboard; articles of paper pulp, of paper or of paperboard",
            49: "Printed books, newspapers, pictures and other products of the printing industry; manuscripts, typescripts and plans",
            50: "Silk",
            51: "Wool, fine or coarse animal hair; horsehair yarn and woven fabric",
            52: "Cotton",
            53: "Other vegetable textile fibres; paper yarn and woven fabrics of paper yarn",
            54: "Man-made filament yarn",
            55: "Man-made staple fibres",
            56: "Wadding, felt and nonwovens; special yarns; twine, cordage, ropes and cables and articles thereof",
            57: "Carpets and other textile floor coverings",
            58: "Special woven fabrics; tufted textile fabrics; lace; tapestries; trimmings; embroidery",
            59: "Impregnated, coated, covered or laminated textile fabrics; textile articles of a kind suitable for industrial use",
            60: "Knitted or crocheted fabrics",
            61: "Articles of apparel and clothing accessories, knitted or crocheted",
            62: "Articles of apparel and clothing accessories, not knitted or crocheted",
            63: "Other made up textile articles; sets; worn clothing and worn textile articles; rags",
            64: "Footwear, gaiters and the like; parts of such articles",
            65: "Headgear and parts thereof",
            66: "Umbrellas, sun umbrellas, walking-sticks, seat-sticks, whips, riding-crops and parts thereof",
            67: "Prepared feathers and down and articles made of feathers or of down; artificial flowers; articles of human hair",
            68: "Articles of stone, plaster, cement, asbestos, mica or similar materials",
            69: "Ceramic products",
            70: "Glass and glassware",
            71: "Natural or cultured pearls, precious or semi-precious stones, precious metals, metals clad with precious metal, and articles thereof; imitation jewellery; coin",
            72: "Iron and steel",
            73: "Articles of iron or steel",
            74: "Copper and articles thereof",
            75: "Nickel and articles thereof",
            76: "Aluminium and articles thereof",
            77: "Lead and articles thereof",
            78: "Zinc and articles thereof",
            79: "Tin and articles thereof",
            80: "Alloys of precious metals; clad with precious metal; articles thereof",
            81: "Other base metals; cermets; articles thereof",
            82: "Tools, implements, cutlery, spoons and forks, of base metal; parts thereof of base metal",
            83: "Miscellaneous articles of base metal",
            84: "Nuclear reactors, boilers, machinery and mechanical appliances; parts thereof",
            85: "Electrical machinery and equipment and parts thereof; sound recorders and reproducers, television image and sound recorders and reproducers, and parts and accessories of such articles",
            86: "Railway or tramway locomotives, rolling-stock and parts thereof; railway or tramway track fixtures and fittings and parts thereof; mechanical (including electro-mechanical) traffic signalling equipment of all kinds",
            87: "Vehicles other than railway or tramway rolling-stock, and parts and accessories thereof",
            88: "Aircraft, spacecraft, and parts thereof",
            89: "Ships, boats and floating structures",
            90: "Optical, photographic, cinematographic, measuring, checking, precision, medical or surgical instruments and apparatus; parts and accessories thereof",
            91: "Clocks and watches and parts thereof",
            92: "Musical instruments; parts and accessories of such articles",
            93: "Arms and ammunition; parts and accessories thereof",
            94: "Furniture; bedding, mattresses, couches and similar stuffed furnishings; lamps and lighting fittings, not elsewhere specified or included; illuminated signs, illuminated name-plates and the like; prefabricated buildings",
            95: "Toys, games and sports requisites; parts and accessories thereof",
            96: "Miscellaneous manufactured articles",
            97: "Works of art, collectors' pieces and antiques",
            98: "Special classification provisions",
            99: "Temporary legislation; temporary modifications"
        }

    def build_from_json(self, json_data: Union[str, List[Dict[str, Any]]]) -> None:
        """Build the HTS hierarchy from JSON data. Also build a node index by node_id."""
        if isinstance(json_data, str):
            data = json.loads(json_data)
        else:
            data = json_data
        logging.info(f"Building HTS tree with {len(data)} items")
        parents_by_indent = {-1: self.root}
        for item in data:
            node_id = self.next_node_id
            self.next_node_id += 1
            node = HTSNode(item, node_id=node_id)
            self.node_index[node_id] = node
            if node.htsno:
                self.code_index[node.htsno] = node
            parent_indent = node.indent - 1
            while parent_indent >= -1:
                if parent_indent in parents_by_indent:
                    parent = parents_by_indent[parent_indent]
                    break
                parent_indent -= 1
            else:
                parent = self.root
            parent.add_child(node)
            parents_by_indent[node.indent] = node
            for indent in list(parents_by_indent.keys()):
                if indent > node.indent:
                    del parents_by_indent[indent]
            if node.indent == 0:
                chapter = node.get_chapter()
                if chapter:
                    if chapter not in self.chapters:
                        self.chapters[chapter] = []
                    self.chapters[chapter].append(node)

    def get_node_by_id(self, node_id: int) -> Optional[HTSNode]:
        return self.node_index.get(node_id)

    def get_chapter_nodes(self, chapter: str) -> List[HTSNode]:
        return self.chapters.get(chapter, [])

    def get_children(self, code: str = "") -> List[Dict[str, Any]]:
        """Return the direct children of the node with the given code as a list of dicts."""
        if not code:
            return []
        node = self.code_index.get(code)
        if not node:
            return []
        direct_children = []
        for child in node.children:
            direct_children.append({
                "code": child.htsno,
                "description": child.description,
                "general": child.general,
                "special": child.special,
                "other": child.other,
                "indent": child.indent,
                "superior": child.is_superior
            })
        direct_children.sort(key=lambda x: x["code"])
        return direct_children

    def _format_options(self, options: List[Dict[str, Any]]) -> str:
        """Format options for inclusion in a prompt with hierarchical structure."""
        formatted = []

        if not any(opt.get('description', '').endswith(':') for opt in options):
            for i, opt in enumerate(options, 1):
                code = opt['code']
                description = opt['description']
                context = ""
                if len(code.split('.')) > 2:
                    parts = code.split('.')
                    if len(parts) >= 2:
                        parent_code = '.'.join(parts[:2])
                        parent_node = self.code_index.get(parent_code)
                        if parent_node:
                            context = f" (Under: {parent_node.description})"
                line = f"{i}. {code}: {description}{context}"
                if opt.get('general', '').strip():
                    line += f" (Duty: {opt['general']})"
                formatted.append(line)
            return "\n".join(formatted)

        counter = 1
        formatted.append("Available Classification Options:")
        for header in [opt for opt in options if opt.get('description', '').endswith(':')]:
            header_desc = header.get('description', '')
            header_code = header.get('code', '')
            formatted.append(f"\n{header_desc}")
            prefix = header_code.split('.')[0]
            children = [opt for opt in options if opt != header and opt.get('code', '').startswith(prefix)]
            children.sort(key=lambda x: x.get('code', ''))
            for child in children:
                child_code = child.get('code', '')
                child_desc = child.get('description', '')
                if not child_desc.endswith(':'):
                    line = f"  {counter}. {child_code}: {child_desc}"
                    if child.get('general', '').strip():
                        line += f" (Duty: {child['general']})"
                    formatted.append(line)
                    counter += 1
        return "\n".join(formatted)

    def _get_full_context(self, code: str) -> str:
        """Get the full classification path for a code by traversing parent pointers."""
        node = self.code_index.get(code)
        if not node:
            return f"Code: {code}"
        parts = []
        while node and node.htsno:
            parts.append(f"{node.htsno} - {node.description}")
            node = node.parent
        return " > ".join(reversed(parts))

    def _create_prompt(self, product: str, current_code: str, options: List[Dict[str, Any]]) -> str:
        """Create a prompt for the current classification step with hierarchical context."""
        if not current_code:
            return f"""Classify this product into the most appropriate HS code chapter:

PRODUCT DESCRIPTION: {product}

AVAILABLE OPTIONS:
{self._format_options(options)}

TASK:
Determine which option best classifies the product by analyzing material, function, processing, and form.
"""
        current_path = self._get_full_context(current_code)
        hierarchy_note = ""
        if any(opt.get('description', '').endswith(':') for opt in options):
            hierarchy_note = """
IMPORTANT - HIERARCHICAL CLASSIFICATION:
Some options are headers (ending with ":") providing context. Choose one of the numbered options only.
"""
        return f"""Continue classifying this product at a more detailed level:

PRODUCT DESCRIPTION: {product}

CURRENT CLASSIFICATION PATH: 
{current_code} - {current_path}
{hierarchy_note}
NEXT LEVEL OPTIONS:
{self._format_options(options)}

TASK:
Select the numbered option that best fits the product based on its specific characteristics.
"""

    def _call_openai(self, prompt: str, retries: int = 3) -> Dict[str, Any]:
        """Call OpenAI API with retries and get a structured JSON response."""
        for attempt in range(retries):
            try:
                response = self.client.chat.completions.create(
                    model="o3-mini",
                    messages=[
                        {"role": "system", "content": "You are a customs classification expert helping to assign HS codes."},
                        {"role": "user", "content": prompt}
                    ],
                    response_format={"type": "json_object"}
                )
                content = response.choices[0].message.content
                try:
                    result = json.loads(content)
                    return result
                except json.JSONDecodeError:
                    if "FINAL:" in content:
                        return {"selection": "FINAL", "confidence": 0.9}
                    match = re.search(r'(\d+)', content)
                    if match:
                        return {"selection": int(match.group(1)), "confidence": 0.7}
                    return {"selection": 0, "confidence": 0.1}
            except Exception as e:
                logging.warning(f"OpenAI API call failed (attempt {attempt+1}/{retries}): {e}")
                if attempt < retries - 1:
                    time.sleep(2)
                else:
                    raise e
        return {"selection": 0, "confidence": 0.0}

    def _parse_response(self, response: Dict[str, Any], options: List[Dict[str, Any]]) -> Tuple[str, bool, float]:
        """
        Parse the LLM response to get the selected option and confidence.
        Returns a tuple of (selected_code, is_final, confidence).
        """
        selection = response.get("selection", 0)
        confidence = response.get("confidence", 0.5)
        if selection == "FINAL":
            return options[0]["code"] if options else "", True, confidence
        if isinstance(selection, int) and 1 <= selection <= len(options):
            return options[selection - 1]["code"], False, confidence
        logging.warning(f"Could not parse a valid option from response: {response}")
        return "", False, 0.0

    def _log_step(self, step_num: int, current: str, selected: str, options: List[Dict[str, Any]], response: str) -> None:
        """Log a classification step."""
        logging.info(f"Step {step_num}: {current} → {selected}")
        self.steps.append({
            "step": step_num,
            "current_code": current,
            "selected_code": selected,
            "options": options,
            "llm_response": response
        })

    def generate_clarification_question(
        self, product_description: str, current_code: str, stage: str, options: List[Dict[str, Any]]
    ) -> ClarificationQuestion:
        """Generate a user-friendly clarification question for this classification stage."""
        options_text = self._format_options(options[:5])
        history_text = self.history.format_for_prompt()
        path_context = ""
        if current_code:
            path_context = f"Current classification path: {self._get_full_context(current_code)}"
        stage_prompts = {
            "chapter": "Determine which chapter (broad category) the product belongs to.",
            "heading": "Determine the specific 4-digit heading within the chapter.",
            "subheading": "Determine the 6-digit subheading that best matches the product.",
            "tariff": "Determine the most specific tariff line for the product."
        }
        stage_description = stage_prompts.get(stage, "Classify the product.")
        hierarchy_context = ""
        if any(opt.get('description', '').endswith(':') for opt in options):
            hierarchy_context = """
IMPORTANT - HIERARCHICAL CLASSIFICATION:
Some options are headers (ending with ":") that provide context. Your question should target the numbered options.
"""
        prompt = f"""You are a customs classification expert creating a targeted question.

PRODUCT DESCRIPTION: {product_description}

CLASSIFICATION STAGE: {stage}
{stage_description}

{path_context}

AVAILABLE OPTIONS:
{options_text}
{hierarchy_context}
PREVIOUS CONVERSATION:
{history_text}

TASK:
Formulate ONE precise question that will reveal the missing information needed to choose among these options.
Respond in JSON format as:
{{
  "question_type": "text" or "multiple_choice",
  "question_text": "Your question here",
  "options": [{{"id": "1", "text": "Option 1"}}, {{"id": "2", "text": "Option 2"}}]   // Omit for text questions
}}
"""
        try:
            logging.info(f"Generating clarification question for stage: {stage}")
            response = self.client.chat.completions.create(
                model="o3-mini",
                messages=[
                    {"role": "system", "content": "You are an expert in customs classification."},
                    {"role": "user", "content": prompt}
                ],
                response_format={"type": "json_object"}
            )
            content = response.choices[0].message.content
            question_data = json.loads(content)
            question = ClarificationQuestion()
            question.question_type = question_data.get("question_type", "text")
            question.question_text = question_data.get("question_text", "")
            question.options = question_data.get("options", [])
            question.metadata = {"stage": stage}
            if not question.question_text:
                question.question_text = f"Can you provide more details about the product?"
            return question
        except Exception as e:
            logging.error(f"Error generating clarification question: {e}")
            question = ClarificationQuestion()
            question.question_text = "Could you provide more details about your product?"
            question.metadata = {"stage": stage}
            return question

    def process_answer(
        self, original_query: str, question: ClarificationQuestion, answer: str, options: List[Dict[str, Any]]
    ) -> Tuple[str, Optional[Dict[str, Any]]]:
        """
        Process the user's answer to update the product description and select the best matching option.
        Returns a tuple of (updated_description, best_match).
        """
        history_text = self.history.format_for_prompt()
        answer_text = answer
        if question.question_type == "multiple_choice" and question.options:
            try:
                if answer.strip().isdigit() and 1 <= int(answer.strip()) <= len(question.options):
                    option_index = int(answer.strip()) - 1
                    answer_text = question.options[option_index]["text"]
            except (ValueError, IndexError):
                pass
        options_text = self._format_options(options[:5])
        prompt = f"""You are a customs classification expert.
ORIGINAL PRODUCT DESCRIPTION: "{original_query}"
QUESTION ASKED: "{question.question_text}"
USER'S ANSWER: "{answer_text}"
AVAILABLE OPTIONS:
{options_text}
PREVIOUS CONVERSATION:
{history_text}

TASK:
1. Update the product description with the new information.
2. Determine if one option clearly matches.
Respond in JSON format as:
{{
  "updated_description": "Your updated description here",
  "selected_option": [1-based index of best option, or null],
  "confidence": [decimal between 0.0-1.0],
  "reasoning": "Your reasoning here"
}}
"""
        try:
            logging.info("Processing user's answer")
            response = self.client.chat.completions.create(
                model="o3-mini",
                messages=[
                    {"role": "system", "content": "You are an expert in customs classification."},
                    {"role": "user", "content": prompt}
                ],
                response_format={"type": "json_object"}
            )
            content = response.choices[0].message.content
            result = json.loads(content)
            updated_description = result.get("updated_description", original_query)
            selected_option = result.get("selected_option")
            confidence = result.get("confidence", 0.0)
            best_match = None
            if selected_option is not None and isinstance(selected_option, (int, float)) and 1 <= selected_option <= len(options):
                option_index = int(selected_option) - 1
                best_match = {
                    "code": options[option_index]["code"],
                    "description": options[option_index]["description"],
                    "confidence": confidence
                }
            return updated_description, best_match
        except Exception as e:
            logging.error(f"Error processing answer: {e}")
            return original_query, None


    def determine_chapter(self, product_description: str) -> str:
        """Ask the LLM to pick the best 2-digit chapter."""
        chapter_list = "\n".join([f"{num:02d}: {desc}" for num, desc in sorted(self.chapters_map.items())])
        prompt = f"""Determine the most appropriate HS code chapter for this product:

PRODUCT: {product_description}

CHAPTERS:
{chapter_list}

INSTRUCTIONS:
Return ONLY the 2-digit chapter number (e.g., "03") that best matches this product.
"""
        logging.info("Sending chapter determination prompt to OpenAI")
        try:
            response = self.client.chat.completions.create(
                model="o3-mini",
                messages=[
                    {"role": "system", "content": "You are a customs classification expert."},
                    {"role": "user", "content": prompt}
                ]
            )
            chapter_response = response.choices[0].message.content.strip()
            match = re.search(r'(\d{2})', chapter_response)
            if match:
                chapter = match.group(1)
                logging.info(f"Selected chapter: {chapter}")
                return chapter
            else:
                logging.warning(f"Could not parse chapter from response: {chapter_response}")
                return ""
        except Exception as e:
            logging.error(f"Error determining chapter: {e}")
            return ""

    def explain_classification(self, original_query: str, enriched_query: str, full_path: str, conversation: List[Dict[str, Any]]) -> str:
        """Generate an explanation of the classification."""
        conversation_text = ""
        for i, qa in enumerate(conversation):
            conversation_text += f"Q{i+1}: {qa['question']}\n"
            conversation_text += f"A{i+1}: {qa['answer']}\n\n"
        path_parts = full_path.split(" > ")
        code_hierarchy = []
        for part in path_parts:
            code_match = re.search(r'(\d{2,4}(?:\.\d{2,4})*)', part)
            if code_match:
                code_hierarchy.append(f"{code_match.group(1)} - {part}")
            elif part != "HS Classification Root":
                code_hierarchy.append(part)
        code_hierarchy_text = "\n".join([f"Level {i+1}: {part}" for i, part in enumerate(code_hierarchy) if part])
        prompt = f"""As a customs classification expert, explain how this product was classified.

            PRODUCT INFORMATION:
            - ORIGINAL DESCRIPTION: {original_query}
            - ENRICHED DESCRIPTION: {enriched_query}

            FINAL CLASSIFICATION: 
            {full_path}

            CLASSIFICATION PATH:
            {code_hierarchy_text}

            CONVERSATION THAT LED TO THIS CLASSIFICATION:
            {conversation_text}

            TASK:
            Explain step-by-step why this classification is correct.
            """
        try:
            logging.info("Generating classification explanation")
            response = self.client.chat.completions.create(
                model="o3-mini",
                messages=[
                    {"role": "system", "content": "You are a customs classification expert who explains decisions in simple terms."},
                    {"role": "user", "content": prompt}
                ]
            )
            explanation = response.choices[0].message.content
            logging.info("Explanation generated successfully")
            return explanation
        except Exception as e:
            logging.error(f"Failed to generate explanation: {e}")
            return "Could not generate explanation due to an error."

    def start_classification(self, product: str, interactive: bool = True, max_questions: int = 3) -> Dict:
        """
        Called by /classify to begin classification. Returns either:
          - a "clarification_question" if we need user input
          - a final classification if we are done
        """
        state = {
            "product": product,
            "original_query": product,
            "current_query": product,
            "questions_asked": 0,
            "selection": {},
            "current_node_id": None,   
            "classification_path": [],
            "steps": [],
            "conversation": [],
            "pending_question": None,
            "pending_options": None,
            "pending_stage": None,
            "max_questions": max_questions
        }
        return self.process_classification(state, interactive, max_questions)

    def continue_classification(self, state: Dict, answer: str, interactive: bool = True, max_questions: int = 3) -> Dict:
        """
        Called by /classify/continue to provide an answer to the last question,
        and attempt to continue classification.
        """
        if state.get("pending_question"):
            state["questions_asked"] += 1
            state["conversation"].append({
                "question": state["pending_question"],
                "answer": answer
            })
            state["current_query"] = state["current_query"] + ". " + answer
            state["pending_question"] = None
            state["pending_options"] = None
            state["pending_stage"] = None

        return self.process_classification(state, interactive, max_questions)

    def process_classification(self, state: Dict, interactive: bool, max_questions: int) -> Dict:
        """
        Core classification logic:
          1. Determine chapter (if not done).
          2. Move down the tree recursively until we reach a leaf or no more data.
          3. If confidence < 0.9 and interactive -> ask a question (if <0.5 do "enhanced classification").
          4. Return final classification if done.
        """

        if "chapter" not in state["selection"]:
            chapter = self.determine_chapter(state["current_query"])
            if not chapter:
                return {"error": "Could not determine chapter", "final": False, "state": state}

            state["selection"]["chapter"] = chapter
            chapter_desc = self.chapters_map.get(int(chapter), "")
            state["classification_path"].append({"type": "chapter", "code": chapter, "description": chapter_desc})

            chapter_nodes = self.get_chapter_nodes(chapter)
            headings = [n for n in chapter_nodes if n.get_node_type() == "heading"]
            if not headings:
                return {"error": f"No headings found in chapter {chapter}", "final": False, "state": state}

            heading_node, heading_conf, llm_response = self.classify_next_level(
                state["current_query"], headings, f"Chapter {chapter}"
            )

            if heading_conf < 0.9 and interactive and state["questions_asked"] < max_questions:

                if heading_conf < 0.5:

                    pass

                question = self.generate_classification_question(state["current_query"], headings, "heading")
                if question:
                    state["pending_question"] = question
                    state["pending_options"] = [
                        {
                            "code": n.htsno,
                            "description": n.description,
                            "general": n.general,
                            "special": n.special,
                            "other": n.other,
                            "indent": n.indent,
                            "superior": n.is_superior
                        }
                        for n in headings
                    ]
                    state["pending_stage"] = "heading"
                    return {
                        "final": False,
                        "clarification_question": question,
                        "state": state
                    }

            if not heading_node:
                return {"error": "Could not determine heading", "final": False, "state": state}

            state["selection"]["heading"] = heading_node.htsno
            state["current_node_id"] = heading_node.node_id
            state["steps"].append({
                "step": 1,
                "current_code": "",
                "selected_code": heading_node.htsno,
                "options": [{"code": n.htsno, "description": n.description} for n in headings],
                "llm_response": llm_response
            })
            state["classification_path"].append({
                "type": "heading",
                "code": heading_node.htsno,
                "description": heading_node.description
            })

        while True:
            current_node = self.get_node_by_id(state["current_node_id"])
            if not current_node:
                break

            options = current_node.children
            if not options:

                break

            node_type = current_node.get_node_type()
            if node_type == "heading":
                stage = "subheading"
            elif node_type == "subheading":
                stage = "tariff"
            else:

                stage = "tariff"

            current_path = " > ".join([
                f"{step.get('code','') if step.get('code','') else step.get('description','')} "
                for step in state["classification_path"]
            ])

            next_node, confidence, llm_response = self.classify_next_level(
                state["current_query"], options, current_path
            )

            if next_node and confidence < 0.5 and interactive and state["questions_asked"] < max_questions:

                enhanced_descriptions = []
                for i, opt in enumerate(options):
                    label = opt.htsno if opt.htsno else "[GROUP]"
                    line = f"{i+1}. {label} - {opt.description}"

                    if opt.children:
                        line += "\n   Sub-items:"
                        for child in opt.children:
                            clabel = child.htsno if child.htsno else "[GROUP]"
                            line += f"\n      - {clabel} {('- ' + child.description) if child.description else ''}"
                    enhanced_descriptions.append(line)

                enhanced_prompt = f"""ENHANCED CLASSIFICATION:

We have these main options, each with their children:

{chr(10).join(enhanced_descriptions)}

PRODUCT: {state["current_query"]}
CURRENT PATH: {current_path}

Which MAIN option best fits the product, considering their sub-items?

FORMAT: same as before (option_number|confidence|explanation)
"""
                try:
                    enhanced_response = self.client.chat.completions.create(
                        model="o3-mini",
                        messages=[
                            {"role": "system", "content": "You are a customs classification expert."},
                            {"role": "user", "content": enhanced_prompt}
                        ]
                    )
                    e_result = enhanced_response.choices[0].message.content.strip()
                    e_parts = e_result.split('|')
                    if len(e_parts) >= 2:
                        try:
                            e_option = int(e_parts[0].strip()) - 1
                            e_conf = float(e_parts[1].strip())
                            if 0 <= e_option < len(options):

                                if e_conf > confidence:
                                    next_node = options[e_option]
                                    confidence = e_conf
                                    llm_response += "\n\n[ENHANCED CLASSIFICATION USED]\n" + e_result
                        except Exception as e:
                            print(f"Error parsing enhanced classification: {e}")
                except Exception as e:
                    print(f"Error during enhanced classification request: {e}")

            if next_node and 0.5 <= confidence < 0.9 and interactive and state["questions_asked"] < max_questions:
                question = self.generate_classification_question(state["current_query"], options, stage)
                if question:
                    state["pending_question"] = question
                    state["pending_options"] = [
                        {
                            "code": n.htsno,
                            "description": n.description,
                            "general": n.general,
                            "special": n.special,
                            "other": n.other,
                            "indent": n.indent,
                            "superior": n.is_superior
                        }
                        for n in options
                    ]
                    return {
                        "final": False,
                        "clarification_question": question,
                        "state": state
                    }

            if not next_node:

                break

            if stage == "subheading":
                state["selection"]["subheading"] = next_node.htsno
            elif stage == "tariff":

                if next_node.htsno:
                    state["selection"]["tariff"] = next_node.htsno

            state["current_node_id"] = next_node.node_id

            state["steps"].append({
                "step": len(state["steps"]) + 1,
                "current_code": current_node.htsno,
                "selected_code": next_node.htsno,
                "options": [{"code": n.htsno, "description": n.description} for n in options],
                "llm_response": llm_response
            })
            state["classification_path"].append({
                "type": stage,
                "code": next_node.htsno if next_node.htsno else "",
                "description": next_node.description
            })

        final_node = self.get_node_by_id(state["current_node_id"]) if state["current_node_id"] else None
        final_code = final_node.htsno if (final_node and final_node.htsno) else None

        full_path = " > ".join([
            (step.get("code") if step.get("code") else step.get("description"))
            for step in state["classification_path"]
        ])
        explanation = self.explain_classification(
            state["product"],
            state["current_query"],
            full_path,
            state["conversation"]
        )

        classification_result = {
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
            "steps": state["steps"],
            "conversation": state["conversation"],
            "explanation": explanation,
            "is_complete": True
        }
        return classification_result

    def classify_next_level(self, product_description: str, options: List[HTSNode], previous_path: str) -> tuple[Optional[HTSNode], float, str]:
        """
        Ask the LLM to pick from among 'options' for the next classification step.
        Returns: (selected_node, confidence, raw_llm_response)
        """
        formatted_options = self.format_options_for_prompt(options)

        prompt = f"""Classify this product into one of the following Harmonized Tariff Schedule (HTS) categories:

PRODUCT: {product_description}

CLASSIFICATION PATH SO FAR: {previous_path}

OPTIONS:
{formatted_options}

TASK:
Determine which of the above options is the most appropriate classification for this product.

STEP-BY-STEP ANALYSIS:
1. Identify all key product attributes from the description:
   - Material composition
   - Function/purpose
   - Manufacturing process
   - Form/state (raw material, finished product, etc.)

2. Compare these attributes to each available option:
   - Which options match the product's key attributes?
   - Are there any options that can be immediately eliminated?
   - Which options require more information to decide between?

3. Evaluate your confidence level using these criteria:

   HIGH CONFIDENCE (0.9 or above) - Use ONLY when:
   - The product description contains SPECIFIC terms that clearly match one option
   - Key distinguishing features are explicitly mentioned
   - There is minimal ambiguity between options

   LOW CONFIDENCE (below 0.9) - Use when:
   - Critical information is missing from the product description
   - Multiple options could reasonably apply
   - Technical distinctions between options cannot be determined from the description
   - The product has characteristics that span multiple options

FORMAT YOUR RESPONSE:
1. Selected option number (just the number, e.g., "2")
2. Confidence score (0.0-1.0)
3. Brief explanation (1-2 sentences)

Example: "2|0.5|This product is clearly a bolt action rifle designed for sporting use."
"""
        print(f"Sending classification prompt with {len(options)} options")

        try:
            response = self.client.chat.completions.create(
                model="o3-mini",
                messages=[
                    {"role": "system", "content": "You are a customs classification expert helping to assign HS codes."},
                    {"role": "user", "content": prompt}
                ]
            )
            result = response.choices[0].message.content.strip()
            parts = result.split('|')
            if len(parts) >= 2:
                try:
                    selected_option = int(parts[0].strip()) - 1
                    confidence = float(parts[1].strip())
                    if 0 <= selected_option < len(options):
                        selected_node = options[selected_option]
                        return selected_node, confidence, result
                    else:
                        print(f"Selected option index out of range: {selected_option}")
                except (ValueError, IndexError) as e:
                    print(f"Error parsing classification response: {e}")
            print(f"Could not parse classification from response: {result}")
            return None, 0.0, result
        except Exception as e:
            print(f"Error during classification: {e}")
            return None, 0.0, ""

    def generate_classification_question(self, product_description: str, options: List[HTSNode], context: str) -> Optional[Dict]:
        """
        Optionally ask the LLM for a single clarifying question (if confidence is 0.9> x >0.5, etc.).
        """
        formatted_options = self.format_options_for_prompt(options)
        prompt = f"""You are helping classify a product in the Harmonized Tariff Schedule (HTS).

PRODUCT DESCRIPTION: {product_description}

CLASSIFICATION CONTEXT: {context}

CURRENT OPTIONS BEING CONSIDERED:
{formatted_options}

Your task is to generate ONE clear, specific question that would help distinguish between these options.
The question should help determine which option is the most accurate classification.

GUIDELINES:
* Ask about product characteristics that would distinguish between the options (material, function, process, size, etc.)
* Use "multiple_choice" for questions with a clear, limited set of possible answers
* Use "text" for questions requiring detailed explanations
* Focus on practical distinctions between the options
* Keep questions simple and direct
* Ask about the product itself, not about classification codes or terminology

RESPONSE FORMAT:
Return a JSON object with:
{{
  "question_type": "text" or "multiple_choice",
  "question_text": "Clear, specific question about the product (not about codes)",
  "options": [
      {{"id": "1", "text": "First option"}},
      {{"id": "2", "text": "Second option"}}
  ]
}}

NOTE: Include "options" only if question_type is "multiple_choice".
"""
        try:
            response = self.client.chat.completions.create(
                model="o3-mini",
                messages=[
                    {"role": "system", "content": "You are an expert in customs classification."},
                    {"role": "user", "content": prompt}
                ]
            )
            question_data = json.loads(response.choices[0].message.content)
            return question_data
        except Exception as e:
            print(f"Error generating classification question: {e}")
            return None

    def format_options_for_prompt(self, options: List[HTSNode]) -> str:
        """
        Convert a list of HTSNodes into a numbered list for the LLM.
        """
        formatted_options = []
        for i, option in enumerate(options):
            label = option.htsno if option.htsno else "[GROUP]"
            formatted_options.append(f"{i+1}. {label} {('- ' + option.description) if option.description else ''}")
        return "\n".join(formatted_options)
