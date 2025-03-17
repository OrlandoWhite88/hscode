import os
import sys
import pickle
import json
import time
import logging
import argparse
import re
import requests
from typing import Dict, List, Any, Optional, Tuple
from datetime import datetime

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    datefmt='%Y-%m-%d %H:%M:%S'
)
logger = logging.getLogger(__name__)

OPENROUTER_API_KEY = ""

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
    """Classifier that uses OpenRouter with Gemini to navigate the HS code tree with user questions"""

    def __init__(self, tree_path: str, api_key: str = None):
        """
        Initialize the classifier

        Args:
            tree_path: Path to the pickled HSCodeTree
            api_key: OpenRouter API key
        """

        self.tree = self._load_tree(tree_path)

        # Use environment variable if no API key is provided
        if api_key is None:
            api_key = os.getenv("OPENROUTER_API_KEY")
            if api_key is None:
                raise ValueError("OPENROUTER_API_KEY environment variable is not set")

        self.api_key = api_key
        self.base_url = "https://openrouter.ai/api/v1/chat/completions"
        self.model = "google/gemini-2.0-flash-thinking-exp:free"
        self.headers = {
            "Authorization": f"Bearer {self.api_key}",
            "Content-Type": "application/json",
            "HTTP-Referer": "https://hs-code-classifier.com",  # Replace with your actual site URL
            "X-Title": "HS Code Classifier"  # Replace with your actual site title
        }

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

        # OPTIMIZED PROMPT
        prompt = f"""Determine the most appropriate HS code chapter for this product:

PRODUCT: {product_description}

CHAPTERS:
{chapter_list}

INSTRUCTIONS:
Your task is to classify the product into the most appropriate 2-digit chapter from the Harmonized System (HS) code.

STEP 1: Carefully analyze ALL key attributes of the product:
- What is it made of? (e.g., metal, textile, wood, plastic)
- What is its function or purpose? (e.g., to cook food, to measure time)
- What is its state or form? (e.g., raw material, semi-finished, finished product)
- Which industry typically produces it? (e.g., agriculture, manufacturing)

STEP 2: Match these attributes against the chapter descriptions.

STEP 3: Select the SINGLE most appropriate chapter.

FORMAT YOUR RESPONSE:
Return ONLY the 2-digit chapter number (01-99) that best matches this product. 
Format your answer as a 2-digit number with leading zero if needed (e.g., "01", "27", "84").

If you're uncertain between two chapters, select the one that appears to be the best match and ONLY return that chapter number.
"""

        logger.info(f"Sending chapter determination prompt to OpenRouter with Gemini")
        try:
            payload = {
                "model": self.model,
                "messages": [
                    {"role": "system", "content": "You are a customs classification expert."},
                    {"role": "user", "content": prompt}
                ],
                "reasoning": {
                    "effort": "high",
                }
            }
            
            response = requests.post(
                self.base_url,
                headers=self.headers,
                json=payload
            )
            
            if response.status_code != 200:
                logger.error(f"API Error: {response.status_code} - {response.text}")
                return ""
                
            response_data = response.json()
            chapter_response = response_data["choices"][0]["message"]["content"].strip()

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
        """Get child nodes of the given code using pattern matching with hierarchy preservation"""

        child_codes = self.tree.find_child_codes(code)

        if not child_codes:
            return []

        # First collect all direct children
        direct_children = []
        for child_code in child_codes:
            child = self.tree.code_index.get(child_code)
            if child:
                direct_children.append({
                    "code": child.htsno,
                    "description": child.description,
                    "general": child.general,
                    "special": child.special,
                    "other": child.other,
                    "indent": child.indent,
                    "superior": child.superior
                })

        # Sort by code to maintain expected order
        direct_children.sort(key=lambda x: x["code"])
        
        return direct_children

    def _format_options(self, options: List[Dict[str, Any]]) -> str:
        """Format options for inclusion in a prompt with hierarchical structure"""
        formatted = []
        
        # Group options by their parent descriptions if possible
        # This is a simplification - ideally we would use the actual hierarchy
        groups = {}
        ungrouped = []
        
        # First pass - identify category headers (items with ":" in description)
        headers = []
        for opt in options:
            desc = opt.get('description', '')
            if desc.endswith(':'):
                headers.append(opt)
        
        # If we have no headers, do standard formatting
        if not headers:
            for i, opt in enumerate(options, 1):
                code = opt['code']
                description = opt['description']
                
                # Get full context for the code if it's a deeper level (tariff line)
                context = ""
                if len(code.split('.')) > 2:  # This is a deeper level code
                    # Extract parent code (subheading)
                    parts = code.split('.')
                    if len(parts) >= 2:
                        parent_code = '.'.join(parts[:2])
                        parent_node = self.tree.code_index.get(parent_code)
                        if parent_node:
                            context = f" (Under subheading: {parent_node.description})"
                
                line = f"{i}. {code}: {description}{context}"
                if opt.get('general') and opt['general'].strip():
                    line += f" (Duty: {opt['general']})"
                formatted.append(line)
            return "\n".join(formatted)
        
        # If we do have headers, use a hierarchical format
        counter = 1
        formatted.append("Available Classification Options:")
        
        # Process each header and its children
        for header in headers:
            header_desc = header.get('description', '')
            header_code = header.get('code', '')
            
            # Add the header
            formatted.append(f"\n{header_desc}")
            
            # Find its children (matching code prefix or items that follow in the list)
            prefix = header_code.split('.')[0]  # Use main part of code as prefix
            
            # Get options that could be children of this header
            children = []
            for opt in options:
                if opt == header:  # Skip the header itself
                    continue
                    
                opt_code = opt.get('code', '')
                opt_desc = opt.get('description', '')
                
                # If it starts with the same prefix, it's potentially a child
                if opt_code.startswith(prefix):
                    children.append(opt)
            
            # Sort children
            children.sort(key=lambda x: x.get('code', ''))
            
            # Add the children with proper indentation
            for child in children:
                child_code = child.get('code', '')
                child_desc = child.get('description', '')
                
                # Check if this is a subheader (ends with colon)
                if child_desc.endswith(':'):
                    formatted.append(f"  {child_desc}")
                    
                    # Find this subheader's children using the same logic
                    subchildren = []
                    subprefix = child_code.split('.')[0]
                    for subopt in options:
                        if subopt == child or subopt == header:
                            continue
                            
                        subopt_code = subopt.get('code', '')
                        if subopt_code.startswith(subprefix):
                            subchildren.append(subopt)
                    
                    # Add subchildren with even more indentation
                    for subchild in subchildren:
                        subchild_code = subchild.get('code', '')
                        subchild_desc = subchild.get('description', '')
                        
                        if not subchild_desc.endswith(':'):  # Only add actual options
                            line = f"    {counter}. {subchild_code}: {subchild_desc}"
                            if subchild.get('general') and subchild['general'].strip():
                                line += f" (Duty: {subchild['general']})"
                            formatted.append(line)
                            counter += 1
                else:
                    # This is a direct child option
                    line = f"  {counter}. {child_code}: {child_desc}"
                    if child.get('general') and child['general'].strip():
                        line += f" (Duty: {child['general']})"
                    formatted.append(line)
                    counter += 1
        
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
        """Create a prompt for the current classification step with hierarchical context"""

        # OPTIMIZED PROMPT FOR INITIAL CLASSIFICATION (CHAPTER LEVEL)
        if not current_code:
            return f"""Classify this product into the most appropriate HS code chapter:

PRODUCT DESCRIPTION: {product}

AVAILABLE OPTIONS:
{self._format_options(options)}

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

RESPONSE FORMAT:
Return a JSON object with:
{{
  "selection": [1-based index of selected option, or "FINAL" if no further classification needed],
  "confidence": [decimal between 0.0-1.0, use 0.9+ ONLY when truly certain],
  "reasoning": "Detailed explanation of why this option was selected and why the confidence is high/low. Include what specific information led to this conclusion or what information is missing."
}}
"""

        # OPTIMIZED PROMPT FOR SUBSEQUENT CLASSIFICATION LEVELS
        current_path = self._get_full_context(current_code)
        
        # Check if we're classifying within a hierarchy with unlabeled headings
        has_hierarchical_structure = any(opt.get('description', '').endswith(':') for opt in options)
        hierarchy_note = ""
        
        if has_hierarchical_structure:
            hierarchy_note = """
