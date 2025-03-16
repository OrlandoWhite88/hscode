import os
import sys
import pickle
import json
import time
import logging
import argparse
import re
from typing import Dict, List, Any, Optional, Tuple
from datetime import datetime

try:
    import openai
except ImportError:
    openai = None

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    datefmt='%Y-%m-%d %H:%M:%S'
)
logger = logging.getLogger(__name__)

OPENAI_API_KEY = ""

class HSNode:
    """Node in the HS code hierarchy"""
    def __init__(self, data: Dict[str, Any]):
        self.htsno: str = data.get("htsno", "")
        self.description: str = data.get("description", "")
        self.indent: int = int(data.get("indent", 0))
        self.superior: bool = data.get("superior") == "true"
        self.units: List[str] = data.get("units", [])
        self.general: str = data.get("general", "")
        self.special: str = data.get("special", "")
        self.other: str = data.get("other", "")
        self.footnotes: List[Dict[str, Any]] = data.get("footnotes", [])
        self.children: List['HSNode'] = []
        self.full_context: List[str] = []

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for serialization"""
        return {
            "htsno": self.htsno,
            "description": self.description,
            "indent": self.indent,
            "superior": self.superior,
            "units": self.units,
            "general": self.general,
            "special": self.special,
            "other": self.other,
            "full_path": " > ".join(self.full_context + [self.description]),
            "children": [child.to_dict() for child in self.children]
        }

class HSCodeTree:
    """Manager for the HS code hierarchy"""
    def __init__(self):
        self.root = HSNode({"description": "HS Classification Root", "indent": -1})
        self.last_updated = datetime.now()
        self.code_index = {}  

    def build_from_flat_json(self, data: List[Dict[str, Any]]) -> None:
        """Build tree from flat JSON data"""
        logger.info(f"Building tree from {len(data)} items...")

        sorted_data = sorted(data, key=lambda x: int(x.get("indent", 0)))

        stack = [self.root]

        for i, item in enumerate(sorted_data):
            if i % 1000 == 0 and i > 0:
                logger.info(f"Processed {i} items...")

            node = HSNode(item)
            current_indent = node.indent

            while len(stack) > current_indent + 1:
                stack.pop()

            parent = stack[-1]

            node.full_context = parent.full_context.copy()
            if parent.description:
                node.full_context.append(parent.description)

            parent.children.append(node)

            if node.superior or node.indent < 9:
                stack.append(node)

            if node.htsno and node.htsno.strip():
                self.code_index[node.htsno] = node

        logger.info(f"Tree built successfully with {len(self.code_index)} indexed codes")

    def save(self, filepath: str) -> None:
        """Save the tree to a file"""
        logger.info(f"Saving tree to {filepath}...")
        with open(filepath, 'wb') as f:
            pickle.dump(self, f)
        logger.info("Tree saved successfully")

    @classmethod
    def load(cls, filepath: str) -> 'HSCodeTree':
        """Load a tree from a file"""
        logger.info(f"Loading tree from {filepath}...")
        with open(filepath, 'rb') as f:
            tree = pickle.load(f)
        logger.info(f"Tree loaded successfully with {len(tree.code_index)} indexed codes")
        return tree

    def print_stats(self) -> None:
        """Print statistics about the tree"""
        total_nodes = self._count_nodes(self.root)
        max_depth = self._max_depth(self.root)
        chapters = [child for child in self.root.children if child.htsno and len(child.htsno.strip()) == 2]

        print("\nHS Code Tree Statistics:")
        print(f"Total nodes: {total_nodes}")
        print(f"Indexed codes: {len(self.code_index)}")
        print(f"Maximum depth: {max_depth}")
        print(f"Number of chapters: {len(chapters)}")
        print(f"Last updated: {self.last_updated}")

    def _count_nodes(self, node: HSNode) -> int:
        """Count total nodes in tree"""
        count = 1  
        for child in node.children:
            count += self._count_nodes(child)
        return count

    def _max_depth(self, node: HSNode, current_depth: int = 0) -> int:
        """Find maximum depth of tree"""
        if not node.children:
            return current_depth

        max_child_depth = 0
        for child in node.children:
            child_depth = self._max_depth(child, current_depth + 1)
            max_child_depth = max(max_child_depth, child_depth)

        return max_child_depth

    def find_child_codes(self, parent_code: str) -> List[str]:
        """
        Find all immediate child codes of a parent code using pattern matching

        For example:
        - Children of "" (empty) would include "01", "02", etc.
        - Children of "01" would include "0101", "0102", etc.
        - Children of "0101" would include "0101.21", "0101.29", etc.
        """
        all_codes = list(self.code_index.keys())

        if not parent_code:
            return [code for code in all_codes if re.match(r'^\d{2}$', code)]

        if re.match(r'^\d{2}$', parent_code):
            pattern = f'^{parent_code}\\d{{2}}$'
            return [code for code in all_codes if re.match(pattern, code)]

        if re.match(r'^\d{4}$', parent_code):
            pattern = f'^{parent_code}\\.\\d{{2}}'
            return [code for code in all_codes if re.match(pattern, code)]

        if re.match(r'^\d{4}\.\d{2}$', parent_code):
            pattern = f'^{parent_code}\\.\\d{{2}}'
            return [code for code in all_codes if re.match(pattern, code)]

        if re.match(r'^\d{4}\.\d{2}\.\d{2}$', parent_code):
            pattern = f'^{parent_code}\\.\\d{{2}}'
            return [code for code in all_codes if re.match(pattern, code)]

        return []

class ClarificationQuestion:
    """Represents a clarification question to refine classification"""
    def __init__(self):
        self.question_type: str = "text"  
        self.question_text: str = ""
        self.options: List[Dict[str, str]] = []  
        self.metadata: Dict[str, Any] = {}

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary representation"""
        result = {
            "question_type": self.question_type,
            "question_text": self.question_text,
            "metadata": self.metadata
        }
        if self.options:
            result["options"] = self.options
        return result

    def __str__(self) -> str:
        """String representation for display"""
        result = [self.question_text]
        if self.question_type == "multiple_choice" and self.options:
            for i, option in enumerate(self.options):
                result.append(f"{i+1}. {option['text']}")
        return "\n".join(result)

class ConversationHistory:
    """Tracks the conversation history for classification context"""
    def __init__(self):
        self.qa_pairs: List[Dict[str, Any]] = []

    def add(self, question: str, answer: str, metadata: Dict[str, Any] = None) -> None:
        """Add a Q&A pair to history"""
        self.qa_pairs.append({
            "question": question,
            "answer": answer,
            "metadata": metadata or {}
        })

    def get_by_stage(self, stage: str) -> List[Dict[str, Any]]:
        """Get Q&A pairs for a specific stage"""
        return [qa for qa in self.qa_pairs 
                if qa.get("metadata", {}).get("stage") == stage]

    def format_for_prompt(self) -> str:
        """Format history for inclusion in prompts"""
        if not self.qa_pairs:
            return ""

        result = "Previous questions and answers:\n"
        for i, qa in enumerate(self.qa_pairs):
            result += f"Q{i+1}: {qa['question']}\n"
            result += f"A{i+1}: {qa['answer']}\n"
        return result

    def to_list(self) -> List[Dict[str, str]]:
        """Convert to list representation"""
        return self.qa_pairs.copy()

def build_and_save_tree(json_filepath: str, output_filepath: str = "hs_code_tree.pkl") -> HSCodeTree:
    """Build tree from JSON data and save it"""
    try:

        output_dir = os.path.dirname(output_filepath)
        if output_dir and not os.path.exists(output_dir):
            os.makedirs(output_dir)

        logger.info(f"Loading JSON data from {json_filepath}...")
        with open(json_filepath, 'r') as f:
            data = json.load(f)
        logger.info(f"Loaded {len(data)} items from JSON")

        tree = HSCodeTree()
        tree.build_from_flat_json(data)

        tree.print_stats()

        tree.save(output_filepath)

        return tree

    except json.JSONDecodeError as e:
        logger.error(f"Error parsing JSON file: {e}")
        return None
    except Exception as e:
        logger.error(f"Error building tree: {e}")
        return None

class HSCodeClassifier:
    """Classifier that uses OpenAI to navigate the HS code tree with user questions"""

    def __init__(self, tree_path: str, api_key: str = None):
        """
        Initialize the classifier

        Args:
            tree_path: Path to the pickled HSCodeTree
            api_key: OpenAI API key (if None, looks for OPENAI_API_KEY env var)
        """

        self.tree = self._load_tree(tree_path)

        if openai is None:
            raise ImportError("OpenAI package is required for classification. Install with: pip install openai")

        # Use environment variable if no API key is provided
        if api_key is None:
            api_key = os.getenv("OPENAI_API_KEY")
            if api_key is None:
                raise ValueError("OPENAI_API_KEY environment variable is not set")

        self.client = openai.OpenAI(api_key=api_key)

        self.steps = []

        self.history = ConversationHistory()

        self.max_questions_per_level = 3

        self._init_chapters_map()

    def _init_chapters_map(self):
        """Initialize the chapters map for quick reference"""
        self.chapters_map = {
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

    def _load_tree(self, tree_path: str) -> HSCodeTree:
        """Load the HS code tree from pickle file"""
        try:
            logger.info(f"Loading HS code tree from {tree_path}")
            with open(tree_path, 'rb') as f:
                tree = pickle.load(f)
            logger.info(f"Tree loaded successfully with {len(tree.code_index)} codes")
            return tree
        except Exception as e:
            logger.error(f"Failed to load tree: {e}")
            raise

    def determine_chapter(self, product_description: str) -> str:
        """
        Determine the most appropriate chapter (2-digit code) for a product
        """

        chapter_list = "\n".join([
            f"{num:02d}: {desc}" for num, desc in sorted(self.chapters_map.items())
        ])

        prompt = f"""Determine the most appropriate HS code chapter for this product:

PRODUCT DESCRIPTION: {product_description}

AVAILABLE CHAPTERS:
{chapter_list}

ANALYSIS INSTRUCTIONS:
1. Identify ALL key characteristics of the product: material, function, processing state, origin, etc.
2. Match these characteristics against each chapter description systematically.
3. Apply these classification principles:
   - Specific descriptions take precedence over general ones
   - Composite products are classified by their essential character
   - If multiple chapters could apply, choose the one appearing last in numerical order

CONFIDENCE RATING:
- ONLY return a chapter number if you see EXPLICIT evidence in the product description
- If multiple chapters could be valid based on limited information, respond with "INSUFFICIENT_INFO"

RESPONSE FORMAT:
Return ONLY a 2-digit chapter number (e.g., "01", "27", "84").
Do NOT include any explanations or additional text.
"""

        logger.info(f"Sending chapter determination prompt to OpenAI")
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
                logger.info(f"Selected chapter: {chapter}")
                return chapter
            else:
                logger.warning(f"Could not parse chapter number from response: {chapter_response}")
                return ""
        except Exception as e:
            logger.error(f"Error determining chapter: {e}")
            return ""

    def get_children(self, code: str = "") -> List[Dict[str, Any]]:
        """Get child nodes of the given code using pattern matching"""

        child_codes = self.tree.find_child_codes(code)

        if not child_codes:
            return []

        result = []
        for child_code in child_codes:
            child = self.tree.code_index.get(child_code)
            if child:
                result.append({
                    "code": child.htsno,
                    "description": child.description,
                    "general": child.general,
                    "special": child.special,
                    "other": child.other
                })

        return result

    def _format_options(self, options: List[Dict[str, Any]]) -> str:
        """Format options for inclusion in a prompt"""
        formatted = []
        for i, opt in enumerate(options, 1):
            line = f"{i}. {opt['code']}: {opt['description']}"
            if opt.get('general') and opt['general'].strip():
                line += f" (Duty: {opt['general']})"
            formatted.append(line)
        return "\n".join(formatted)

    def _get_full_context(self, code: str) -> str:
        """Get the full classification path for a code"""
        if not code:
            return ""

        node = self.tree.code_index.get(code)
        if not node:
            return f"Code: {code}"

        path = node.full_context.copy()
        path.append(node.description)
        return " > ".join(path)

    def _create_prompt(self, product: str, current_code: str, options: List[Dict[str, Any]]) -> str:
        """Create a prompt for the current classification step"""

        if not current_code:
            return f"""Classify this product into the most appropriate HS code chapter:

PRODUCT DESCRIPTION: {product}

AVAILABLE OPTIONS:
{self._format_options(options)}

STEP-BY-STEP ANALYSIS REQUIRED:
1. Extract ALL relevant product characteristics from the description: 
   - Physical composition (materials, components)
   - Function/purpose
   - Processing level (raw, semi-processed, finished)
   - Any specific features mentioned

2. Match these characteristics against each option systematically:
   - Identify EXACT matches between product attributes and option descriptions
   - Rule out options that clearly don't match
   - Note which options potentially match

3. CONFIDENCE SCORING - BE EXTREMELY STRICT:
   - Assign confidence ≥ 0.9 ONLY when the product EXPLICITLY matches ONE option
     based on CLEAR information in the description that leaves NO room for doubt
   - Assign confidence < 0.9 whenever:
     * Multiple options could potentially match
     * Key distinguishing information is missing
     * There are ambiguities in the product description
     * Classification requires assumptions beyond what's explicitly stated

CONFIDENCE SCALE DEFINITION:
0.9-1.0: Absolute certainty based on explicit information
0.7-0.8: Strong match but minor confirmatory details missing
0.5-0.6: Plausible match but multiple options could apply
<0.5: Substantial uncertainty, critical information missing

RESPONSE FORMAT:
Return a JSON object with EXACTLY these fields:
{{
  "selection": 3,  // The number of the selected option (1-based index)
  "confidence": 0.7,  // Your precise confidence level (default to lower confidence when uncertain)
  "reasoning": "Detailed explanation citing SPECIFIC words/phrases from the product description that match the selected option and justify your confidence level"
}}
"""

        current_path = self._get_full_context(current_code)

        return f"""Continue classifying this product:

PRODUCT DESCRIPTION: {product}

CURRENT CLASSIFICATION PATH: {current_code} - {current_path}

NEXT LEVEL OPTIONS:
{self._format_options(options)}

SYSTEMATIC ANALYSIS REQUIRED:
1. Examine the product against each option's criteria:
   - Map SPECIFIC product attributes to SPECIFIC classification criteria
   - Apply HS classification principles in this priority:
     a) Specific descriptions take precedence over general ones
     b) Essential character determines classification of mixed/composite goods
     c) The most specific applicable subheading takes priority

2. CONFIDENCE EVALUATION - BE EXTREMELY CONSERVATIVE:
   - Assign confidence ≥ 0.9 ONLY when you can point to SPECIFIC WORDS in the description
     that DEFINITIVELY match ONE option and rule out all others
   - Assign confidence < 0.9 whenever:
     * Key distinguishing information is missing
     * Terms in the description are ambiguous
     * Multiple options could potentially apply
     * Classification requires inference beyond explicit statements

3. If you determine this is the FINAL classification level:
   - Verify the product can't be classified further down the hierarchy
   - Confirm no more specific subheadings would apply

CONFIDENCE SCALE DEFINITION:
0.9-1.0: Absolute certainty based on explicit information
0.7-0.8: Strong match but minor confirmatory details missing
0.5-0.6: Plausible match but multiple options could apply
<0.5: Substantial uncertainty, critical information missing

RESPONSE FORMAT:
Return a JSON object with EXACTLY these fields:
{{
  "selection": 3,  // Option number (1-based index) or "FINAL" if no further classification needed
  "confidence": 0.7,  // Your precise confidence level (default to lower confidence when uncertain)
  "reasoning": "Detailed explanation citing SPECIFIC words/phrases from the product description that match the selected option and justify your confidence level"
}}
"""

    def _call_openai(self, prompt: str, retries: int = 3) -> Dict[str, Any]:
        """Call OpenAI API with retries and get structured JSON response"""
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
                logger.warning(f"OpenAI API call failed (attempt {attempt+1}/{retries}): {e}")
                if attempt < retries - 1:
                    time.sleep(2)  
                else:
                    raise

        return {"selection": 0, "confidence": 0.0}

    def _parse_response(self, response: Dict[str, Any], options: List[Dict[str, Any]]) -> Tuple[str, bool, float]:
        """
        Parse the LLM response to get the selected option and confidence

        Returns:
            Tuple of (selected_code, is_final, confidence)
        """

        selection = response.get("selection", 0)
        confidence = response.get("confidence", 0.5)

        if selection == "FINAL":
            return options[0]["code"] if options else "", True, confidence

        if isinstance(selection, int) and 1 <= selection <= len(options):
            return options[selection - 1]["code"], False, confidence

        logger.warning(f"Could not parse a valid option from response: {response}")
        return "", False, 0.0

    def _log_step(self, step_num: int, current: str, selected: str, options: List[Dict[str, Any]], response: str) -> None:
        """Log a classification step"""
        logger.info(f"Step {step_num}: {current} → {selected}")

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
        """
        Generate a user-friendly clarification question to help with classification

        Args:
            product_description: Description of the product
            current_code: Current code in the classification process
            stage: Current stage ('chapter', 'heading', 'subheading', 'tariff')
            options: Available options at this stage

        Returns:
            ClarificationQuestion object
        """

        options_text = self._format_options(options[:5])  

        history_text = self.history.format_for_prompt()

        path_context = ""
        if current_code:
            path_context = f"Current classification path: {self._get_full_context(current_code)}"

        option_details = []
        for opt in options[:5]:
            code = opt.get('code', '')
            desc = opt.get('description', '')
            keywords = []

            words = re.findall(r'\b\w+\b', desc.lower())
            keywords = [w for w in words if len(w) > 3 and w not in ['with', 'without', 'other', 'than', 'from', 'have', 'been', 'their', 'which', 'that']]

            option_details.append({
                'code': code,
                'description': desc,
                'keywords': keywords
            })

        stage_prompts = {
            "chapter": "We need to determine which chapter (broad category) this product belongs to.",
            "heading": "We need to determine the specific 4-digit heading within the chapter.",
            "subheading": "We need to determine the 6-digit subheading that best matches this product.",
            "tariff": "We need to determine the most specific tariff line for this product."
        }

        stage_description = stage_prompts.get(stage, "We need to classify this product.")

        prompt = f"""You are a customs classification expert creating the optimal question to accurately classify a product.

PRODUCT DESCRIPTION: {product_description}

CLASSIFICATION STAGE: {stage}
{stage_description}

{path_context}

DECISION POINT:
We need to determine which classification is correct:
{options_text}

PREVIOUS Q&A:
{history_text}

INFORMATION GAP ANALYSIS:
1. First, identify what information we ALREADY KNOW from the description and previous answers:
   - Materials/composition
   - Form/structure
   - Purpose/function
   - Processing state
   - Technical specifications
   
2. Identify the SPECIFIC MISSING INFORMATION that would definitively determine classification:
   - What single piece of information would distinguish between the options?
   - Which critical attribute is entirely missing from the description?

QUESTION FORMULATION REQUIREMENTS:
1. Create a question that:
   - Targets the EXACT information gap identified
   - Focuses on PRODUCT CHARACTERISTICS, not classification options
   - Uses simple, non-technical language
   - Will elicit a specific, factual answer that distinguishes between options
   - NEVER asks about information already provided
   - Is phrased to get specific, factual information

2. Question type selection:
   - Use "text" for open-ended information (exact materials, measurements, etc.)
   - Use "multiple_choice" ONLY when all possible answers can be enumerated
     and are mutually exclusive

STRICTLY PROHIBITED:
- NEVER ask about information clearly stated in the description or previous answers
- NEVER ask multiple questions at once
- NEVER use classification terminology or jargon
- NEVER phrase questions in terms of HS codes or classifications
- NEVER ask "Is your product more like X or Y?" - focus on the product itself

IDEAL PRODUCT-FOCUSED QUESTIONS:
- "What is the primary material your product is made from?"
- "What is the main purpose of your product?"
- "How is your product packaged for retail sale?"
- "What is the alcohol content percentage of your beverage?"

RESPONSE FORMAT:
Return a JSON object with these fields:
{{
  "question_type": "text" or "multiple_choice",
  "question_text": "Your precisely formulated question",
  "options": [
    {{"id": "1", "text": "First option"}},
    {{"id": "2", "text": "Second option"}}
  ]
}}

For text questions, omit the "options" field.
"""

        try:
            logger.info(f"Generating clarification question for {stage} stage")
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
                question.question_text = f"Can you tell me more about your {product_description}?"

            return question

        except Exception as e:
            logger.error(f"Error generating question: {e}")

            question = ClarificationQuestion()
            question.question_text = "Could you provide more details about your product?"
            question.metadata = {"stage": stage}
            return question

    def process_answer(
        self, original_query: str, question: ClarificationQuestion, answer: str, options: List[Dict[str, Any]]
    ) -> Tuple[str, Dict[str, Any]]:
        """
        Process the user's answer to help with classification decision

        Args:
            original_query: Original product description
            question: The question that was asked
            answer: User's answer
            options: Available options at this stage

        Returns:
            Tuple of (enriched_query, best_match)
            - enriched_query: Updated product description with new information
            - best_match: Dictionary with the best matching option info or None if inconclusive
        """

        history_text = self.history.format_for_prompt()

        answer_text = answer
        if question.question_type == "multiple_choice" and question.options:
            try:
                if answer.isdigit() and 1 <= int(answer) <= len(question.options):
                    option_index = int(answer) - 1
                    answer_text = question.options[option_index]["text"]
            except (ValueError, IndexError):
                pass

        options_text = self._format_options(options[:5])

        prompt = f"""You are an expert customs classification analyzer determining how a user's answer impacts product classification.

ORIGINAL PRODUCT DESCRIPTION: "{original_query}"

QUESTION ASKED: "{question.question_text}"

USER'S ANSWER: "{answer_text}"

CLASSIFICATION OPTIONS:
{options_text}

PREVIOUS CONVERSATION:
{history_text}

ANALYSIS TASK:
1. Carefully analyze what NEW information the answer provides:
   - Extract SPECIFIC product attributes mentioned
   - Identify EXACT technical details that were unknown before
   
2. Create an updated product description that:
   - Integrates ALL previously known information
   - Incorporates the NEW details from the answer
   - Maintains factual accuracy without losing any details
   
3. Evaluate classification impact:
   - Determine if the answer DEFINITIVELY identifies one option as correct
   - Check if it eliminates any options
   - Assess if ambiguity remains between multiple options

CONFIDENCE EVALUATION - BE CONSERVATIVE:
- Assign confidence 0.9-1.0 ONLY when the answer provides EXPLICIT information that:
  * Perfectly matches ONE option's requirements
  * Rules out all other options definitively
  
- Use lower confidence when:
  * The answer is helpful but doesn't clearly distinguish between options
  * Any ambiguities or assumptions remain

RESPONSE FORMAT:
Return a JSON object with these fields:
{{
  "updated_description": "Complete product description integrating all known information",
  "selected_option": null or option_number,  // null if inconclusive, integer (1-based index) if definitive
  "confidence": 0.7,  // Your precise confidence level (default to lower confidence when uncertain)
  "reasoning": "Detailed explanation of how the answer affects classification, citing SPECIFIC WORDS from the answer that match classification criteria"
}}
"""

        try:
            logger.info("Processing user's answer")
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
            logger.error(f"Error processing answer: {e}")
            return original_query, None

    def classify_with_questions(self, product_description: str, max_questions: int = 9) -> Dict[str, Any]:
        """
        Classify a product with interactive questions

        Args:
            product_description: Description of the product to classify
            max_questions: Maximum number of questions to ask

        Returns:
            Classification result with code, path, confidence, and justification
        """
        logger.info(f"Starting interactive classification for: {product_description}")

        self.steps = []
        self.history = ConversationHistory()
        current_code = ""
        current_query = product_description
        is_final = False
        step = 0
        questions_asked = 0

        selection = {
            "chapter": "",
            "heading": "",
            "subheading": "",
            "tariff": ""
        }

        chapter_code = self.determine_chapter(product_description)
        if not chapter_code:
            logger.warning("Could not determine chapter, using default approach")
        else:

            options = [{
                "code": chapter_code,
                "description": self.chapters_map.get(int(chapter_code), "Unknown chapter")
            }]
            self._log_step(step, "", chapter_code, options, f"Selected chapter {chapter_code}")

            selection["chapter"] = chapter_code
            current_code = chapter_code
            step = 1

        while not is_final and step < 10 and questions_asked < max_questions:

            if len(current_code) == 0:
                stage = "chapter"
            elif len(current_code) == 2:
                stage = "heading"
            elif len(current_code) == 4:
                stage = "subheading"
            else:
                stage = "tariff"

            options = self.get_children(current_code)

            logger.info(f"Found {len(options)} options for code '{current_code}' at stage {stage}")

            if not options:
                logger.info(f"No further options for {current_code}, ending classification")
                break

            prompt = self._create_prompt(current_query, current_code, options)
            response = self._call_openai(prompt)

            selected_code, is_final, llm_confidence = self._parse_response(response, options)

            logger.info(f"Initial LLM confidence: {llm_confidence:.2f}")

            level_questions = len(self.history.get_by_stage(stage))

            if llm_confidence >= 0.9 or level_questions >= self.max_questions_per_level:
                if llm_confidence >= 0.9:
                    logger.info(f"High LLM confidence ({llm_confidence:.2f}), skipping question")
                else:
                    logger.info(f"Reached maximum questions for {stage} level, making best guess")

                if selected_code:
                    selection[stage] = selected_code
                    current_code = selected_code
                    self._log_step(step, current_code, selected_code, options, str(response))
                    step += 1
                    continue
                else:
                    break

            question = self.generate_clarification_question(
                current_query, current_code, stage, options
            )

            print("\n" + question.question_text)

            if question.question_type == "multiple_choice" and question.options:
                for i, option in enumerate(question.options):
                    print(f"{i+1}. {option['text']}")

            user_answer = input("\nYour answer: ")

            updated_query, best_match = self.process_answer(
                current_query, question, user_answer, options
            )

            self.history.add(
                question=question.question_text,
                answer=user_answer,
                metadata=question.metadata
            )

            questions_asked += 1
            current_query = updated_query

            logger.info(f"Updated query: {current_query}")

            if best_match and best_match.get("confidence", 0) > 0.7:
                selected_code = best_match["code"]
                selection[stage] = selected_code
                current_code = selected_code

                self._log_step(step, current_code, selected_code, options, f"Selected based on user answer: {user_answer}")

                step += 1

                print(f"\nBased on your answer, we've selected: {selected_code} - {best_match['description']}")
            else:

                prompt = self._create_prompt(current_query, current_code, options)
                response = self._call_openai(prompt)
                selected_code, is_final, confidence = self._parse_response(response, options)

                if selected_code:
                    selection[stage] = selected_code
                    current_code = selected_code
                    self._log_step(step, current_code, selected_code, options, str(response))

                    selected_desc = next((opt["description"] for opt in options if opt["code"] == selected_code), "")
                    print(f"\nBased on your information, we've selected: {selected_code} - {selected_desc}")

                    step += 1
                else:

                    logger.warning("No valid selection, ending classification")
                    break

        final_code = current_code
        full_path = self._get_full_context(final_code)

        explanation = self.explain_classification(product_description, current_query, full_path, self.history.to_list())

        result = {
            "original_query": product_description,
            "enriched_query": current_query,
            "classification": {
                "chapter": selection.get("chapter", ""),
                "heading": selection.get("heading", ""),
                "subheading": selection.get("subheading", ""),
                "tariff": selection.get("tariff", "")
            },
            "final_code": final_code,
            "full_path": full_path,
            "steps": self.steps,
            "conversation": self.history.to_list(),
            "explanation": explanation,
            "is_complete": is_final
        }

        logger.info(f"Final classification: {final_code} - {full_path}")
        return result

    def explain_classification(self, original_query: str, enriched_query: str, full_path: str, conversation: List[Dict[str, Any]]) -> str:
        """Generate an explanation of the classification"""

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

        prompt = f"""
I need a precise, methodical explanation of how this product was classified through the Harmonized System.

ORIGINAL PRODUCT DESCRIPTION: {original_query}

ENRICHED PRODUCT DESCRIPTION: {enriched_query}

FINAL CLASSIFICATION PATH: {full_path}

HIERARCHICAL CLASSIFICATION BREAKDOWN:
{code_hierarchy_text}

CONVERSATION HISTORY:
{conversation_text}

EXPLANATION REQUIREMENTS:
1. Structure your explanation by classification level:
   - Chapter (first 2 digits)
   - Heading (4 digits)
   - Subheading (6 digits)
   - Tariff line (8-10 digits, if applicable)

2. For EACH level, explain:
   - The EXACT product characteristics that determined this classification
   - Why alternatives were ruled out
   - How the Q&A process clarified critical information

3. Connect every classification decision directly to:
   - Specific words/phrases from the product description
   - Particular answers the user provided

4. Write in clear, concise language that a non-expert can understand, while including sufficient technical detail to explain the reasoning at each level of the classification hierarchy.
"""

        try:
            logger.info("Generating classification explanation")
            response = self.client.chat.completions.create(
                model="o3-mini",
                messages=[
                    {"role": "system", "content": "You are a customs classification expert who explains decisions in simple terms."},
                    {"role": "user", "content": prompt}
                ]
            )
            explanation = response.choices[0].message.content
            logger.info("Explanation generated successfully")
            return explanation
        except Exception as e:
            logger.error(f"Failed to generate explanation: {e}")
            return "Could not generate explanation due to an error."


def main():
    parser = argparse.ArgumentParser(description="HS Code Classification Tool")
    subparsers = parser.add_subparsers(dest="command", help="Command to run")

    # Build tree command
    build_parser = subparsers.add_parser("build", help="Build and save HS code tree")
    build_parser.add_argument("json_file", help="Path to input JSON file with HS code data")
    build_parser.add_argument("--output", "-o", default="hs_code_tree.pkl", help="Output path for the tree file")

    # Classify command
    classify_parser = subparsers.add_parser("classify", help="Classify a product using interactive questions")
    classify_parser.add_argument("tree_file", help="Path to the HS code tree file")
    classify_parser.add_argument("product", help="Description of the product to classify")
    classify_parser.add_argument("--max-questions", "-q", type=int, default=9, help="Maximum number of questions to ask")
    classify_parser.add_argument("--output", "-o", help="Output path for the classification result (JSON)")
    classify_parser.add_argument("--api-key", help="OpenAI API key")

    args = parser.parse_args()

    if args.command == "build":
        tree = build_and_save_tree(args.json_file, args.output)
        if tree:
            print(f"Successfully built and saved HS code tree to {args.output}")
        else:
            print("Failed to build HS code tree")

    elif args.command == "classify":
        try:
            classifier = HSCodeClassifier(args.tree_file, api_key=args.api_key)
            result = classifier.classify_with_questions(args.product, max_questions=args.max_questions)

            print("\n" + "=" * 50)
            print("CLASSIFICATION RESULT")
            print("=" * 50)
            print(f"Product: {args.product}")
            print(f"Final HS Code: {result['final_code']}")
            print(f"Classification Path: {result['full_path']}")
            print("\nExplanation:")
            print(result['explanation'])

            if args.output:
                with open(args.output, 'w') as f:
                    json.dump(result, f, indent=2)
                print(f"\nSaved detailed result to {args.output}")

        except Exception as e:
            print(f"Error during classification: {e}")
            raise

    else:
        parser.print_help()


if __name__ == "__main__":
    main()
