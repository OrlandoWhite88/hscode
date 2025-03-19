import os
import sys
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

# Import the HTSTree components
from hts_parser import HTSNode, HTSTree, parse_hts_json

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    datefmt='%Y-%m-%d %H:%M:%S'
)
logger = logging.getLogger(__name__)

OPENAI_API_KEY = ""

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
    

class HSCodeClassifier:
    """Classifier that uses OpenAI to navigate the HS code tree with user questions"""

    def __init__(self, tree_path: str, api_key: str = None):
        """
        Initialize the classifier
        
        Args:
            tree_path: Path to the HTS tree JSON file
            api_key: OpenAI API key (optional, will use env var if not provided)
        """
        # Load the HTS tree from the JSON file
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

    def _load_tree(self, tree_path: str) -> HTSTree:
        """Load the HTS tree from a JSON file"""
        try:
            logger.info(f"Loading HTS tree from {tree_path}")
            
            # Load the JSON data
            with open(tree_path, 'r', encoding='utf-8') as f:
                json_data = json.load(f)
            
            # Create a new HTSTree
            tree = HTSTree()
            
            # Helper function to recursively build nodes
            def create_node_from_dict(node_dict):
                # Create a new HTSNode with the data
                node = HTSNode({
                    'htsno': node_dict.get('htsno', ''),
                    'description': node_dict.get('description', ''),
                    'indent': node_dict.get('indent', 0),
                    'superior': 'true' if node_dict.get('is_superior', False) else 'false',
                    'units': node_dict.get('units', []),
                    'general': node_dict.get('general', ''),
                    'special': node_dict.get('special', ''),
                    'other': node_dict.get('other', ''),
                    'footnotes': node_dict.get('footnotes', [])
                })
                
                # Add to code index if it has a code
                if node.htsno:
                    tree.code_index[node.htsno] = node
                
                # Process children
                for child_dict in node_dict.get('children', []):
                    child_node = create_node_from_dict(child_dict)
                    node.add_child(child_node)
                
                return node
            
            # Check the format of the JSON
            if isinstance(json_data, dict) and "root" in json_data:
                # This is from tree.to_dict()
                root_dict = json_data["root"]
                
                # Set up dummy root
                tree.root = HTSNode({
                    'htsno': root_dict.get('htsno', ''),
                    'indent': root_dict.get('indent', -1),
                    'description': root_dict.get('description', 'ROOT'),
                    'superior': 'false'
                })
                
                # Process children of root
                for child_dict in root_dict.get('children', []):
                    child_node = create_node_from_dict(child_dict)
                    tree.root.add_child(child_node)
                
                # Add chapter information
                for chapter, codes in json_data.get("chapters", {}).items():
                    if chapter not in tree.chapters:
                        tree.chapters[chapter] = []
                    for code in codes:
                        node = tree.find_by_htsno(code)
                        if node:
                            tree.chapters[chapter].append(node)
            else:
                # If it's in a different format, try parsing it directly
                tree = parse_hts_json(json_data)
            
            logger.info(f"Successfully loaded HTS tree with {len(tree.code_index)} codes")
            return tree
            
        except Exception as e:
            logger.error(f"Error loading HTS tree: {e}")
            raise ValueError(f"Failed to load HTS tree from {tree_path}: {e}")

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

    def find_child_codes(self, parent_code: str) -> List[str]:
        """Find all child codes of a parent code in the HTS tree"""
        if not parent_code:
            # For empty parent code, return the top-level chapter codes
            chapters = self.tree.get_chapters()
            return chapters
        
        # Find the parent node
        parent_node = self.tree.find_by_htsno(parent_code)
        if not parent_node:
            logger.warning(f"Parent code {parent_code} not found in tree")
            return []
        
        # Collect direct children codes
        child_codes = []
        for child in parent_node.children:
            if child.htsno:  # Only include nodes with an actual code
                child_codes.append(child.htsno)
        
        return child_codes

    def get_children(self, code: str = "") -> List[Dict[str, Any]]:
        """Get child nodes of the given code using pattern matching with hierarchy preservation"""
        child_codes = self.find_child_codes(code)

        if not child_codes:
            return []

        # First collect all direct children
        direct_children = []
        for child_code in child_codes:
            child = self.tree.find_by_htsno(child_code)
            if child:
                direct_children.append({
                    "code": child.htsno,
                    "description": child.description,
                    "general": child.general,
                    "special": child.special,
                    "other": child.other,
                    "indent": child.indent,
                    "superior": child.is_superior
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
                        parent_node = self.tree.find_by_htsno(parent_code)
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
        """Get the full context (path) for a specific HTS code"""
        if not code:
            return "HS Classification Root"
            
        # Find the node in the tree
        node = self.tree.find_by_htsno(code)
        if not node:
            return code  # Return just the code if node not found
            
        # Build the path
        path_parts = []
        
        # Add chapter context if available
        chapter_code = node.get_chapter()
        if chapter_code and chapter_code in self.chapters_map:
            chapter_num = int(chapter_code)
            path_parts.append(f"Chapter {chapter_code}: {self.chapters_map.get(chapter_num, '')}")
        
        # Add the full node path
        path_parts.append(node.get_full_path())
        
        return " > ".join(path_parts)

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
IMPORTANT - HIERARCHICAL CLASSIFICATION:
Note that some options are category headers (ending with ":") that provide context for the options beneath them. 
These headers represent product categories or types, while the numbered options are the actual classificable items.

For example, if you see:
Rifles:
  Centerfire:
    1. 9303.30.40.20: Autoloading
    2. 9303.30.40.30: Bolt action

This means options 1 and 2 are both centerfire rifles, with option 1 being autoloading and option 2 being bolt action.
When making your selection, choose one of the NUMBERED options only.
"""

        return f"""Continue classifying this product at a more detailed level:

PRODUCT DESCRIPTION: {product}

CURRENT CLASSIFICATION PATH: 
{current_code} - {current_path}
{hierarchy_note}
NEXT LEVEL OPTIONS:
{self._format_options(options)}

TASK:
Determine which of these more specific options is the most appropriate classification for this product.

STEP-BY-STEP ANALYSIS:
1. Review what we know about the product and what the current path already covers:
   - We've already established that this product is within {current_path}
   - Now we need to determine its more specific classification

2. Analyze what distinguishes these options from each other:
   - Pay attention to the hierarchical structure of options (headers and sub-options)
   - Identify the key differentiating factors between these options (material, manufacturing process, function, etc.)
   - For tariff line options with brief descriptions, interpret them in the context of their parent headings
   - Check if the product description contains information about these differentiating factors

3. Evaluate your confidence level using these criteria:
   
   HIGH CONFIDENCE (0.9 or above) - Use ONLY when:
   - The product description explicitly mentions features that match one specific option
   - The differentiating characteristics between options are clearly addressed in the description
   - There is minimal ambiguity between options

   LOW CONFIDENCE (below 0.9) - Use when:
   - The product description lacks information about key differentiating features
   - Multiple options could potentially apply based on the available information
   - Technical distinctions between options cannot be determined from the description
   - Additional information would significantly improve classification accuracy

RESPONSE FORMAT:
Return a JSON object with:
{{
  "selection": [1-based index of selected option, or "FINAL" if no further classification is possible],
  "confidence": [decimal between 0.0-1.0, use 0.9+ ONLY when truly certain],
  "reasoning": "Detailed explanation of why this option was selected and why the confidence is high/low. Explicitly identify what information led to this conclusion and what information would help if missing."
}}

If none of the options are appropriate or this appears to be the most specific level possible, respond with:
{{
  "selection": "FINAL",
  "confidence": [appropriate confidence level],
  "reasoning": "Detailed explanation of why further classification is not possible or not needed."
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
            "tariff": "We need to determine the most specific tariff line for this product. Note that tariff lines are specialized subcategories within their parent subheadings."
        }

        stage_description = stage_prompts.get(stage, "We need to classify this product.")

        # OPTIMIZED QUESTION GENERATION PROMPT
        has_hierarchical_structure = any(opt.get('description', '').endswith(':') for opt in options)
        hierarchy_context = ""
        
        if has_hierarchical_structure:
            hierarchy_context = """
IMPORTANT - HIERARCHICAL CLASSIFICATION:
Some options are organized hierarchically with category headers (ending with ":") that provide context.
For example, "Rifles:" followed by "Centerfire:" and then specific options like "Autoloading" and "Bolt action".
Your question should help distinguish between the ACTUAL options (not the headers), while using the headers for context.
"""

        prompt = f"""You are a customs classification expert creating targeted questions to accurately classify products.

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
Formulate ONE precise question that will reveal the specific information needed to select between the classification options.

QUESTION CREATION PROCESS:
1. IDENTIFY KEY DIFFERENTIATORS:
   * Analyze what SPECIFICALLY distinguishes these options from each other
   * Focus on: materials, processing methods, functions, dimensions, components, etc.
   * Determine which differentiator is MOST critical for classification

2. CHECK EXISTING INFORMATION:
   * Review what information is already known from the product description and previous answers
   * Never ask for information already provided

3. FORMULATE THE QUESTION:
   * Focus EXCLUSIVELY on product characteristics, NOT on the HS codes themselves
   * Ask about the product directly, not "which option best describes your product"
   * Use simple, non-technical language the user will understand
   * Make it specific and targeted - avoid vague or overly broad questions
   * Frame it to distinguish between the most likely options

QUESTION TYPE SELECTION:
* Use "text" for:
  - Questions requiring detailed descriptions
  - When many possible answers exist
  - When precise values or technical details are needed

* Use "multiple_choice" for:
  - Questions with a clear, limited set of possible answers
  - When the distinguishing factor has distinct options (e.g., material types)
  - When the user might not know technical terminology


NOTE: DO NOT ASK QUESTIONS WHICH ARE OBVIOUS - JUST BECAUSE THEY ARE THE OPTIONS PRESENTED TO YOU AND YOU NEED TO DETERMINE WHICH ONES THEY FIT INTO. YOU SHOULD NOT BE ASKING THE USER WHAT CLASSIFICATION SOMETHING FITS INTO. YOU SHOULD BE ASKING ABOUT THE PRODUCT.
FOR EXAMPLE:
I INPUTTED SEABREAM AND RECIEVED THESE QUESTIONS:
It is a trout species (e.g., Salmo trutta, including farmed rainbow trout variants)
It is a Pacific salmon species (e.g., Oncorhynchus species)
It is not a trout or Pacific salmon species

THESE ARE REALLY STUPID QUESTIONS. DO NOT DO THIS. IT IS CLEARLY OBVIOUSLY NOT TROUT OR SALMON AND ITS SEABREAM. I KNOW THESE WERE THE HEADINGS PRESENTED TO YOU AND YOU WERE TRYING TO UNDERSTAND WHICH IT FITS INTO BUT DONT ASK THIS. AT THE VERY LEAST ASK ABOUT THE PRODUCT TO LEARN MORE. I.E. IS THIS FRESH OR FROZEN OR LIVE OR PROCESSED.

THIS APPLIES AS MUCH TO THE EXAMPLE OF SEABREAM AS TO ANYTHING ELSE. FOR ANY QUERY YOU SHOULD ASK ABOUT THE PRODUCT NOT THE CLASSIFICATION OPTIONS AND DETERMINE WHEN ITS OBVIOUS.

RESPONSE FORMAT:
Return a JSON object with:
{{
  "question_type": "text" or "multiple_choice",
  "question_text": "Clear, specific question about the product (not about codes)",
  "options": [
    {{"id": "1", "text": "First option"}},
    {{"id": "2", "text": "Second option"}},
    etc.
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

        # OPTIMIZED ANSWER PROCESSING PROMPT
        prompt = f"""You are a customs classification expert evaluating how new information affects product classification.

ORIGINAL PRODUCT DESCRIPTION: "{original_query}"

QUESTION ASKED: "{question.question_text}"

USER'S ANSWER: "{answer_text}"

AVAILABLE CLASSIFICATION OPTIONS:
{options_text}

PREVIOUS CONVERSATION:
{history_text}

TASK:
1. Incorporate the new information into a comprehensive product description
2. Determine if this information is sufficient to select a specific classification option

STEP-BY-STEP ANALYSIS:
1. ANALYZE THE ANSWER:
   * What new information does this answer provide?
   * Does it address key differentiating factors between options?
   * Does it confirm or contradict any previous information?

2. UPDATE THE PRODUCT DESCRIPTION:
   * Create a complete, integrated description with ALL information now known
   * Maintain all relevant details from the original description
   * Add the new information in a natural way
   * Resolve any contradictions with prior information

3. EVALUATE CLASSIFICATION IMPLICATIONS:
   * For each option, assess how well it matches the updated description
   * Identify which option(s) are compatible with the new information
   * Determine if one option now clearly stands out

4. ASSESS CONFIDENCE LEVEL:
   HIGH CONFIDENCE (0.8 or above) - Use ONLY when:
   * The answer directly addresses a key differentiating factor
   * The information clearly points to one specific option
   * There is minimal ambiguity remaining

   MEDIUM CONFIDENCE (0.5-0.7) - Use when:
   * The answer provides useful but incomplete information
   * The new information narrows down options but doesn't definitively select one
   * Some ambiguity remains between 2-3 options

   LOW CONFIDENCE (below 0.5) - Use when:
   * The answer provides little relevant information
   * Multiple options remain equally plausible
   * Critical differentiating information is still missing

RESPONSE FORMAT:
Return a JSON object with:
{{
  "updated_description": "Complete updated product description with all information",
  "selected_option": [1-based index of best option, or null if insufficient information],
  "confidence": [decimal between 0.0-1.0],
  "reasoning": "Detailed explanation of how the new information affects classification and why this confidence level is appropriate"
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

        # OPTIMIZED EXPLANATION PROMPT
        prompt = f"""As a customs classification expert, provide a clear explanation of how this product was classified.

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
Explain the classification process in clear, logical terms that anyone can understand. Your explanation should:

1. Walk through the classification journey step-by-step:
   * Begin with identifying the broadest category (chapter)
   * Explain how each subsequent level narrowed down the classification
   * Show how each decision logically followed from product characteristics

2. Highlight the specific product features that determined each classification choice:
   * What characteristics led to the chapter selection?
   * What features determined the heading?
   * What details guided the subheading and tariff selections?

3. Explain how the conversation questions and answers influenced the classification:
   * How did each answer clarify the product's classification?
   * What critical information was revealed through questions?

4. Justify why this is the correct classification:
   * What makes this HS code the most appropriate?
   * How does it align with the product's essential characteristics?
   * Why were alternative classifications rejected?

Use plain, accessible language that a non-expert can understand while maintaining enough technical precision to explain the reasoning accurately. Structure your explanation in a clear, organized manner with logical sections that follow the classification hierarchy.
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
    """Main function"""
    parser = argparse.ArgumentParser(description="HS Code Classifier using LLM")
    parser.add_argument("--tree", "-t", default="hts_tree_output.json", help="Path to the HTS tree JSON file")
    parser.add_argument("--product", "-p", help="Product to classify")
    parser.add_argument("--key", "-k", help="OpenAI API key (optional, will use env var if not set)")
    args = parser.parse_args()
    
    # Check if tree file exists
    if not os.path.exists(args.tree):
        logger.error(f"Tree file not found: {args.tree}")
        sys.exit(1)
    
    # Initialize classifier
    try:
        classifier = HSCodeClassifier(args.tree, api_key=args.key)
    except Exception as e:
        logger.error(f"Failed to initialize classifier: {e}")
        sys.exit(1)
    
    # Get product to classify
    product = args.product
    if not product:
        product = input("Enter product to classify: ")
    
    if not product:
        logger.error("No product specified")
        sys.exit(1)
    
    # Run classification
    try:
        result = classifier.classify_with_questions(product)
        
        # Display results
        print("\n" + "="*50)
        print("CLASSIFICATION RESULT")
        print("="*50)
        print(f"Product: {result['original_query']}")
        print(f"Enriched Description: {result['enriched_query']}")
        print(f"Final HS Code: {result['final_code']}")
        print(f"Classification Path: {result['full_path']}")
        print("\nEXPLANATION:")
        print(result['explanation'])
        print("="*50)
        
        # Option to save result
        save_option = input("\nDo you want to save the result to a file? (y/n): ")
        if save_option.lower() == 'y':
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            filename = f"classification_{timestamp}.json"
            with open(filename, 'w', encoding='utf-8') as f:
                json.dump(result, f, indent=2)
            print(f"Result saved to {filename}")
        
    except Exception as e:
        logger.error(f"Classification failed: {e}")
        sys.exit(1)

if __name__ == "__main__":
    main()