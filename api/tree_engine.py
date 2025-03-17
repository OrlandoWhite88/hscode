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
        # Flag to identify if this is a title node (no code but provides context)
        self.is_title = not self.htsno and self.description and self.description.strip()
        # Store contextual titles that apply to this node
        self.contextual_titles: List[str] = []
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for serialization"""
        # Include full context with titles for complete hierarchical understanding
        full_context = self.full_context.copy()
        full_context.extend(self.contextual_titles)
        
        return {
            "htsno": self.htsno,
            "description": self.description,
            "indent": self.indent,
            "superior": self.superior,
            "units": self.units,
            "general": self.general,
            "special": self.special,
            "other": self.other,
            "is_title": self.is_title,
            "full_path": " > ".join(full_context + [self.description]),
            "contextual_titles": self.contextual_titles,
            "children": [child.to_dict() for child in self.children]
        }


class HSCodeTree:
    """Manager for the HS code hierarchy"""
    def __init__(self):
        self.root = HSNode({"description": "HS Classification Root", "indent": -1})
        self.last_updated = datetime.now()
        self.code_index = {}  # Maps HS codes to nodes
        self.title_nodes = []  # Track all title nodes
    
    def build_from_flat_json(self, data: List[Dict[str, Any]]) -> None:
        """Build tree from flat JSON data"""
        logger.info(f"Building tree from {len(data)} items...")
        
        # First, build the tree based on indent levels
        self._build_tree_by_indent(data)
        
        # Build contextual title information throughout the tree
        self._apply_title_context()
        
        # Then, enhance the tree with HS code hierarchical relationships
        self._build_code_hierarchy()
        
        logger.info(f"Tree built successfully with {len(self.code_index)} indexed codes and {len(self.title_nodes)} title nodes")
    
    def _build_tree_by_indent(self, data: List[Dict[str, Any]]) -> None:
        """Build the initial tree structure based on indent levels"""
        # Sort by indent to ensure parent nodes are processed before children
        sorted_data = sorted(data, key=lambda x: int(x.get("indent", 0)))
        
        # Stack to track the current node at each indent level
        stack = [self.root]
        
        for i, item in enumerate(sorted_data):
            if i % 1000 == 0 and i > 0:
                logger.info(f"Processed {i} items...")
                
            # Create new node
            node = HSNode(item)
            current_indent = node.indent
            
            # Adjust stack to find correct parent
            while len(stack) > current_indent + 1:
                stack.pop()
            
            # Get parent from stack
            parent = stack[-1]
            
            # Update node's context with parent's context
            node.full_context = parent.full_context.copy()
            if parent.description:
                node.full_context.append(parent.description)
            
            # Add node to parent's children
            parent.children.append(node)
            
            # Add to stack if this node could have children
            if node.superior or node.indent < 9:
                stack.append(node)
            
            # Add to index if it has an HS code
            if node.htsno and node.htsno.strip():
                self.code_index[node.htsno] = node
            
            # Track title nodes separately
            if node.is_title:
                self.title_nodes.append(node)
    
    def _apply_title_context(self) -> None:
        """
        Apply contextual information from title nodes to their children
        This ensures all nodes understand their complete hierarchy including titles
        """
        logger.info("Applying title context to nodes...")
        
        def process_node_with_context(node, current_titles=None):
            if current_titles is None:
                current_titles = []
            
            # If this is a title node, add it to the current context
            if node.is_title:
                current_titles = current_titles + [node.description]
            
            # For nodes with HS codes, store the contextual titles
            if node.htsno and node.htsno.strip():
                node.contextual_titles = current_titles.copy()
            
            # Process all children with updated context
            for child in node.children:
                process_node_with_context(child, current_titles)
        
        # Start from the root node
        process_node_with_context(self.root)
        logger.info("Title context applied successfully")
    
    def _build_code_hierarchy(self) -> None:
        """Build additional parent-child relationships based on HS code patterns"""
        logger.info("Building HS code hierarchy relationships...")
        
        # Get all HS codes
        all_codes = list(self.code_index.keys())
        all_codes.sort(key=self._code_sort_key)
        
        # Process each pattern level
        self._process_chapter_codes(all_codes)  # 2-digit codes (01, 02...)
        self._process_heading_codes(all_codes)  # 4-digit codes (0101, 0102...)
        self._process_subheading_codes(all_codes)  # 6-digit codes (0101.21...)
        self._process_tariff_codes(all_codes)  # 8-digit and 10-digit codes
        
        # After rearranging the hierarchy, reapply title context
        self._apply_title_context()
        
        logger.info("HS code hierarchy relationships built successfully")
    
    def _code_sort_key(self, code: str) -> Tuple:
        """Create a sort key for HS codes to ensure proper ordering"""
        # Split the code by dots and then by digits
        parts = []
        for part in code.split('.'):
            # Extract digits and convert to integers for proper numeric sorting
            digits = re.findall(r'\d+', part)
            parts.extend([int(d) for d in digits])
        return tuple(parts)
    
    def _process_chapter_codes(self, all_codes: List[str]) -> None:
        """Process 2-digit chapter codes (01, 02...)"""
        chapter_codes = [code for code in all_codes if re.match(r'^\d{2}$', code)]
        
        # Add chapters as direct children of the root if not already there
        for code in chapter_codes:
            node = self.code_index[code]
            if node not in self.root.children:
                self.root.children.append(node)
                
                # Update context
                node.full_context = ["HS Classification Root"]
    
    def _process_heading_codes(self, all_codes: List[str]) -> None:
        """Process 4-digit heading codes (0101, 0102...)"""
        heading_codes = [code for code in all_codes if re.match(r'^\d{4}$', code)]
        
        for code in heading_codes:
            chapter_code = code[:2]
            if chapter_code in self.code_index:
                parent = self.code_index[chapter_code]
                child = self.code_index[code]
                
                # Add child if not already a child of parent
                if child not in parent.children:
                    parent.children.append(child)
                    
                    # Update context
                    child.full_context = parent.full_context.copy()
                    child.full_context.append(parent.description)
    
    def _process_subheading_codes(self, all_codes: List[str]) -> None:
        """Process 6-digit subheading codes (0101.21...)"""
        subheading_codes = [code for code in all_codes if re.match(r'^\d{4}\.\d{2}', code)]
        
        for code in subheading_codes:
            heading_code = code.split('.')[0]  # Get the 4-digit part (0101)
            
            if heading_code in self.code_index:
                parent = self.code_index[heading_code]
                child = self.code_index[code]
                
                # Add child if not already a child of parent
                if child not in parent.children:
                    parent.children.append(child)
                    
                    # Update context
                    child.full_context = parent.full_context.copy()
                    child.full_context.append(parent.description)
    
    def _process_tariff_codes(self, all_codes: List[str]) -> None:
        """Process 8-digit and 10-digit tariff codes"""
        # Process 8-digit codes (0101.21.00)
        eight_digit_codes = [code for code in all_codes if re.match(r'^\d{4}\.\d{2}\.\d{2}$', code)]
        
        for code in eight_digit_codes:
            parent_code = '.'.join(code.split('.')[:2])  # Get 6-digit part (0101.21)
            
            if parent_code in self.code_index:
                parent = self.code_index[parent_code]
                child = self.code_index[code]
                
                # Add child if not already a child of parent
                if child not in parent.children:
                    parent.children.append(child)
                    
                    # Update context
                    child.full_context = parent.full_context.copy()
                    child.full_context.append(parent.description)
        
        # Process 10-digit codes (0101.21.00.10)
        ten_digit_codes = [code for code in all_codes if re.match(r'^\d{4}\.\d{2}\.\d{2}\.\d{2}$', code)]
        
        for code in ten_digit_codes:
            parent_code = '.'.join(code.split('.')[:3])  # Get 8-digit part (0101.21.00)
            
            if parent_code in self.code_index:
                parent = self.code_index[parent_code]
                child = self.code_index[code]
                
                # Add child if not already a child of parent
                if child not in parent.children:
                    parent.children.append(child)
                    
                    # Update context
                    child.full_context = parent.full_context.copy()
                    child.full_context.append(parent.description)
    
    def find_children(self, code: str) -> List[HSNode]:
        """
        Find all immediate child nodes for a given HS code
        Includes both nodes with HS codes and title nodes
        """
        # If no code provided, return top-level chapters
        if not code:
            return self.root.children
        
        # If code exists in the index, return ALL of its children (including title nodes)
        if code in self.code_index:
            parent_node = self.code_index[code]
            return parent_node.children
            
        # If code not found, fallback to pattern matching
        return self._find_children_by_pattern(code)
    
    def _find_children_by_pattern(self, code: str) -> List[HSNode]:
        """Find children by pattern matching on the HS code structure"""
        results = []
        all_codes = list(self.code_index.keys())
        
        # For chapter codes (2-digit)
        if re.match(r'^\d{2}$', code):
            pattern = f'^{code}\\d{{2}}$'
            child_codes = [c for c in all_codes if re.match(pattern, c)]
            results = [self.code_index[c] for c in child_codes]
            
        # For heading codes (4-digit)
        elif re.match(r'^\d{4}$', code):
            pattern = f'^{code}\\.\\d{{2}}'
            child_codes = [c for c in all_codes if re.match(pattern, c)]
            results = [self.code_index[c] for c in child_codes]
            
        # For subheading codes (6-digit)
        elif re.match(r'^\d{4}\.\d{2}$', code):
            pattern = f'^{code}\\.\\d{{2}}'
            child_codes = [c for c in all_codes if re.match(pattern, c)]
            results = [self.code_index[c] for c in child_codes]
            
        # For 8-digit codes
        elif re.match(r'^\d{4}\.\d{2}\.\d{2}$', code):
            pattern = f'^{code}\\.\\d{{2}}'
            child_codes = [c for c in all_codes if re.match(pattern, c)]
            results = [self.code_index[c] for c in child_codes]
            
        return results
    
    def get_node_with_context(self, code: str) -> Dict[str, Any]:
        """
        Get a node's dictionary representation with complete context
        This includes all title information that would apply to this node
        """
        if code not in self.code_index:
            return None
            
        node = self.code_index[code]
        return node.to_dict()
    
    def find_child_codes(self, parent_code: str) -> List[str]:
        """
        Find all immediate child codes of a parent code
        Only returns codes, not title nodes (for backward compatibility)
        """
        children = self.find_children(parent_code)
        return [child.htsno for child in children if child.htsno and child.htsno.strip()]
    
    def get_formatted_node(self, code: str) -> Dict[str, Any]:
        """
        Get a node with rich context information for display
        Includes title context in a human-readable format
        """
        if code not in self.code_index:
            return None
            
        node = self.code_index[code]
        result = node.to_dict()
        
        # Add formatted contextual information
        if node.contextual_titles:
            titles_str = " > ".join(node.contextual_titles)
            result["contextual_path"] = titles_str
            
            # For display purposes, combine the description with title context
            if titles_str:
                result["display_description"] = f"{titles_str} > {node.description}"
            else:
                result["display_description"] = node.description
        else:
            result["display_description"] = node.description
            
        return result
    
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
        print(f"Title nodes (no HS code): {len(self.title_nodes)}")
        print(f"Maximum depth: {max_depth}")
        print(f"Number of chapters: {len(chapters)}")
        print(f"Last updated: {self.last_updated}")
        
        # Print a sample of the hierarchy
        if chapters:
            print("\nSample of HS Code Hierarchy:")
            sample_chapter = chapters[0]
            print(f"Chapter: {sample_chapter.htsno} - {sample_chapter.description}")
            
            headings = [child for child in sample_chapter.children if child.htsno and len(child.htsno) == 4]
            if headings:
                sample_heading = headings[0]
                print(f"  Heading: {sample_heading.htsno} - {sample_heading.description}")
                
                subheadings = [child for child in sample_heading.children if child.htsno and "." in child.htsno]
                if subheadings:
                    sample_subheading = subheadings[0]
                    print(f"    Subheading: {sample_subheading.htsno} - {sample_subheading.description}")
                    
                    title_nodes = [child for child in sample_subheading.children if child.is_title]
                    if title_nodes:
                        sample_title = title_nodes[0]
                        print(f"      Title Node: {sample_title.description}")
                        
                        title_children = [child for child in sample_title.children if child.htsno]
                        if title_children:
                            sample_child = title_children[0]
                            print(f"        Child with Context: {sample_child.htsno} - {sample_child.description}")
                            print(f"        Contextual Titles: {', '.join(sample_child.contextual_titles)}")
                            print(f"        Full Path: {' > '.join(sample_child.full_context + sample_child.contextual_titles + [sample_child.description])}")
                    
                    tariff_lines = [child for child in sample_subheading.children if child.htsno and child.htsno.count('.') > 1]
                    if tariff_lines:
                        sample_tariff = tariff_lines[0]
                        print(f"      Tariff Line: {sample_tariff.htsno} - {sample_tariff.description}")
                        if sample_tariff.contextual_titles:
                            print(f"      Contextual Titles: {', '.join(sample_tariff.contextual_titles)}")
    
    def _count_nodes(self, node: HSNode) -> int:
        """Count total nodes in tree"""
        count = 1  # Count the current node
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
        # Ensure output directory exists
        output_dir = os.path.dirname(output_filepath)
        if output_dir and not os.path.exists(output_dir):
            os.makedirs(output_dir)
        
        # Load JSON data
        logger.info(f"Loading JSON data from {json_filepath}...")
        with open(json_filepath, 'r', encoding='utf-8') as f:
            data = json.load(f)
        logger.info(f"Loaded {len(data)} items from JSON")
        
        # Build tree
        tree = HSCodeTree()
        tree.build_from_flat_json(data)
        
        # Print statistics
        tree.print_stats()
        
        # Save tree
        tree.save(output_filepath)
        
        return tree
        
    except json.JSONDecodeError as e:
        logger.error(f"Error parsing JSON file: {e}")
        return None
    except Exception as e:
        logger.error(f"Error building tree: {e}")
        logger.exception("Detailed error:")
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

    def get_children(self, code: str = "") -> List[Dict[str, Any]]:
        """Get child nodes of the given code, properly handling title nodes and context"""
        
        # Get all child nodes from the tree, including title nodes
        child_nodes = self.tree.find_children(code)
        
        if not child_nodes:
            return []
            
        # Process and format all children, including title nodes
        processed_children = []
        
        for child in child_nodes:
            # For title nodes (no HS code), mark them as headers in the result
            if child.is_title:
                processed_children.append({
                    "code": "",  # No code for title nodes
                    "description": child.description,
                    "indent": child.indent,
                    "superior": child.superior,
                    "is_title": True,  # Mark as title node for special handling in UI
                    "contextual_titles": child.contextual_titles  # Include any context the title itself has
                })
            # For regular nodes with HS codes
            elif child.htsno:
                # Get the full context for this node, including any title info
                formatted_node = self.tree.get_formatted_node(child.htsno) if child.htsno in self.tree.code_index else None
                
                node_info = {
                    "code": child.htsno,
                    "description": child.description,
                    "general": child.general,
                    "special": child.special,
                    "other": child.other,
                    "indent": child.indent,
                    "superior": child.superior,
                    "is_title": False
                }
                
                # Add contextual information if available
                if formatted_node and "contextual_path" in formatted_node:
                    node_info["contextual_path"] = formatted_node["contextual_path"]
                if formatted_node and "display_description" in formatted_node:
                    node_info["display_description"] = formatted_node["display_description"]
                if hasattr(child, "contextual_titles") and child.contextual_titles:
                    node_info["contextual_titles"] = child.contextual_titles
                    
                processed_children.append(node_info)
                
        # Sort by code to maintain expected order, but place title nodes first within their groups
        processed_children.sort(key=lambda x: (x.get("code", "") if not x.get("is_title") else "", x.get("indent", 0)))
        
        return processed_children

    def _format_options(self, options: List[Dict[str, Any]]) -> str:
        """Format options for inclusion in a prompt with hierarchical structure and title context"""
        formatted = []
        option_counter = 0
        
        # Check if we have title nodes in the options
        has_titles = any(opt.get('is_title', False) for opt in options)
        
        if not has_titles:
            # Simple list format without hierarchical structure
            for opt in options:
                if opt.get('is_title', False):
                    continue  # Skip title nodes in this format
                    
                option_counter += 1
                code = opt.get('code', '')
                description = opt.get('description', '')
                
                # Include contextual title information if available
                context = ""
                if opt.get('contextual_titles'):
                    context = f" (Under: {' > '.join(opt['contextual_titles'])})"
                elif opt.get('contextual_path'):
                    context = f" (Under: {opt['contextual_path']})"
                
                # Use display_description if available
                if opt.get('display_description'):
                    description = opt['display_description']
                
                line = f"{option_counter}. {code}: {description}{context}"
                if opt.get('general') and opt['general'].strip():
                    line += f" (Duty: {opt['general']})"
                formatted.append(line)
            
            return "\n".join(formatted)
        
        # Handle hierarchical format with title nodes
        formatted.append("Available Classification Options:")
        
        # Process directly first-level title nodes
        title_indices = [i for i, opt in enumerate(options) if opt.get('is_title', False)]
        if title_indices:
            # Process each title node and its children
            for title_idx in title_indices:
                title_node = options[title_idx]
                title_desc = title_node.get('description', '')
                
                # Add the title as a header
                formatted.append(f"\n{title_desc}")
                
                # Find all regular nodes that should be grouped under this title
                # We'll use indent levels and contextual info to determine this
                title_indent = title_node.get('indent', 0)
                for i, opt in enumerate(options):
                    # Skip title nodes and nodes that are handled elsewhere
                    if opt.get('is_title', False) or i == title_idx:
                        continue
                        
                    # Check if this node should be under the current title
                    # It might have the title in its contextual titles, or have a higher indent level
                    is_child = False
                    if opt.get('contextual_titles') and title_desc in opt.get('contextual_titles', []):
                        is_child = True
                    elif opt.get('indent', 0) > title_indent:
                        # For simplicity, assume nodes with higher indent directly after the title are children
                        # More complex logic could be added if tree structure is more complex
                        is_child = True
                    
                    if is_child:
                        option_counter += 1
                        code = opt.get('code', '')
                        description = opt.get('description', '')
                        
                        # Use display description if available
                        if opt.get('display_description'):
                            description = opt['display_description']
                            
                        line = f"  {option_counter}. {code}: {description}"
                        if opt.get('general') and opt['general'].strip():
                            line += f" (Duty: {opt['general']})"
                        formatted.append(line)
        
        # Add nodes that aren't under any title
        remaining_nodes = [opt for i, opt in enumerate(options) 
                         if not opt.get('is_title', False) and i not in [title_idx+1 for title_idx in title_indices]]
        
        if remaining_nodes:
            formatted.append("\nOther Options:")
            for opt in remaining_nodes:
                option_counter += 1
                code = opt.get('code', '')
                description = opt.get('description', '')
                
                # Use display description if available
                if opt.get('display_description'):
                    description = opt['display_description']
                    
                line = f"{option_counter}. {code}: {description}"
                if opt.get('general') and opt['general'].strip():
                    line += f" (Duty: {opt['general']})"
                formatted.append(line)
                
        return "\n".join(formatted)

    def _get_full_context(self, code: str) -> str:
        """Get the full classification path for a code including title context"""
        if not code:
            return ""

        # Use the improved get_formatted_node method that includes title context
        node_info = self.tree.get_formatted_node(code)
        if not node_info:
            return f"Code: {code}"
            
        # Return the display description which includes title context if available
        if "display_description" in node_info:
            return f"{code} - {node_info['display_description']}"
        elif "full_path" in node_info:
            return f"{code} - {node_info['full_path']}"
        else:
            return f"{code} - {node_info['description']}"

    def _create_prompt(self, product: str, current_code: str, options: List[Dict[str, Any]]) -> str:
        """Create a prompt for the current classification step with hierarchical context"""

        # Filter out title nodes for the numbered options, but keep their information for context
        numbered_options = [opt for opt in options if not opt.get('is_title', False)]
        
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

        # OPTIMIZED PROMPT FOR SUBSEQUENT CLASSIFICATION LEVELS WITH TITLE CONTEXT
        current_path = self._get_full_context(current_code)
        
        # Check if we're classifying within a hierarchy with title nodes
        has_hierarchical_structure = any(opt.get('is_title', False) for opt in options)
        hierarchy_note = ""
        
        if has_hierarchical_structure:
            hierarchy_note = """
IMPORTANT - HIERARCHICAL CLASSIFICATION:
Note that some descriptions represent category headers/titles that provide context for other options.
These headers represent product categories or types within the classification system.
The options with numbers are the actual classificable items, which may fall under these headers.

For example, if you see:
Rifles:
  1. 9303.30.40.20: Autoloading
  2. 9303.30.40.30: Bolt action

This means options 1 and 2 are both types of rifles, with option 1 being autoloading and option 2 being bolt action.
When making your selection, choose one of the NUMBERED options only.
"""

        return f"""Continue classifying this product at a more detailed level:

PRODUCT DESCRIPTION: {product}

CURRENT CLASSIFICATION PATH: 
{current_path}
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
        # Filter out title nodes - we only want to select from actual code options
        code_options = [opt for opt in options if not opt.get('is_title', False)]
        
        selection = response.get("selection", 0)
        confidence = response.get("confidence", 0.5)

        if selection == "FINAL":
            return code_options[0]["code"] if code_options else "", True, confidence

        if isinstance(selection, int) and 1 <= selection <= len(code_options):
            return code_options[selection - 1]["code"], False, confidence

        logger.warning(f"Could not parse a valid option from response: {response}")
        return "", False, 0.0

    def _log_step(self, step_num: int, current: str, selected: str, options: List[Dict[str, Any]], response: str) -> None:
        """Log a classification step"""
        logger.info(f"Step {step_num}: {current} → {selected}")

        # Filter out title nodes for cleaner logging
        code_options = [opt for opt in options if not opt.get('is_title', False)]
        
        self.steps.append({
            "step": step_num,
            "current_code": current,
            "selected_code": selected,
            "options": code_options,  # Only log code options
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
        # Filter out title nodes for the question options, but keep their information for context
        code_options = [opt for opt in options if not opt.get('is_title', False)]
        
        options_text = self._format_options(options[:10])  # Include titles in the context

        history_text = self.history.format_for_prompt()

        path_context = ""
        if current_code:
            path_context = f"Current classification path: {self._get_full_context(current_code)}"

        # Extract keywords from option descriptions, including title context
        option_details = []
        for opt in code_options[:5]:  # Limit to first 5 code options for conciseness
            code = opt.get('code', '')
            
            # Use the display description if available, which includes title context
            if opt.get('display_description'):
                desc = opt.get('display_description', '')
            else:
                desc = opt.get('description', '')
                # Add contextual titles if available
                if opt.get('contextual_titles'):
                    context = " > ".join(opt.get('contextual_titles', []))
                    desc = f"{context} > {desc}"
            
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

        # OPTIMIZED QUESTION GENERATION PROMPT WITH TITLE CONTEXT
        has_hierarchical_structure = any(opt.get('is_title', False) for opt in options)
        hierarchy_context = ""
        
        if has_hierarchical_structure:
            hierarchy_context = """
IMPORTANT - HIERARCHICAL CLASSIFICATION:
Some options are organized hierarchically with category headers/titles that provide context.
For example, "Rifles:" followed by specific options like "Autoloading" and "Bolt action".
Your question should help distinguish between the actual classification options, while considering their context within the hierarchy.
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
   * Consider the hierarchical context shown in the options (what categories the options belong to)
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
        # Filter out title nodes for matching, but keep them for context in the prompt
        code_options = [opt for opt in options if not opt.get('is_title', False)]

        history_text = self.history.format_for_prompt()

        answer_text = answer
        if question.question_type == "multiple_choice" and question.options:
            try:
                if answer.isdigit() and 1 <= int(answer) <= len(question.options):
                    option_index = int(answer) - 1
                    answer_text = question.options[option_index]["text"]
            except (ValueError, IndexError):
                pass

        options_text = self._format_options(options[:10])  # Include title nodes for context

        # OPTIMIZED ANSWER PROCESSING PROMPT WITH TITLE CONTEXT
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
   * Consider how the answer relates to any category headers or contextual titles in the options

2. UPDATE THE PRODUCT DESCRIPTION:
   * Create a complete, integrated description with ALL information now known
   * Maintain all relevant details from the original description
   * Add the new information in a natural way
   * Resolve any contradictions with prior information

3. EVALUATE CLASSIFICATION IMPLICATIONS:
   * For each option, assess how well it matches the updated description
   * Identify which option(s) are compatible with the new information
   * Determine if one option now clearly stands out
   * Consider the contextual hierarchy when evaluating options

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

Note: When referring to "option index", use only the numbered options. Category headers/titles are not numbered options.
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

            if selected_option is not None and isinstance(selected_option, (int, float)) and 1 <= selected_option <= len(code_options):
                option_index = int(selected_option) - 1
                best_match = {
                    "code": code_options[option_index]["code"],
                    "description": code_options[option_index].get("display_description", code_options[option_index]["description"]),
                    "confidence": confidence
                }

            return updated_description, best_match

        except Exception as e:
            logger.error(f"Error processing answer: {e}")
            return original_query, None

    def classify_with_questions(self, product_description: str, max_questions: int = 9) -> Dict[str, Any]:
        """
        Classify a product with interactive questions, taking title context into account

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
            # Get chapter node with full context
            chapter_node = self.tree.code_index.get(chapter_code)
            chapter_desc = chapter_node.description if chapter_node else self.chapters_map.get(int(chapter_code), "Unknown chapter")
            
            options = [{
                "code": chapter_code,
                "description": chapter_desc
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

            # Get all children including title nodes for context
            options = self.get_children(current_code)

            logger.info(f"Found {len(options)} options (including {sum(1 for o in options if o.get('is_title', False))} title nodes) for code '{current_code}' at stage {stage}")

            if not options:
                logger.info(f"No further options for {current_code}, ending classification")
                break

            prompt = self._create_prompt(current_query, current_code, options)
            response = self._call_openai(prompt)

            # Parse only against code options, not title nodes
            code_options = [opt for opt in options if not opt.get('is_title', False)]
            selected_code, is_final, llm_confidence = self._parse_response(response, code_options)

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
                    self._log_step(step, current_code, selected_code, code_options, str(response))
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

                self._log_step(step, current_code, selected_code, code_options, f"Selected based on user answer: {user_answer}")

                step += 1

                # Get display description with context if available
                node_info = self.tree.get_formatted_node(selected_code)
                selected_desc = node_info.get("display_description", best_match["description"]) if node_info else best_match["description"]
                
                print(f"\nBased on your answer, we've selected: {selected_code} - {selected_desc}")
            else:
                # No clear match from answer, use LLM to select based on updated query
                prompt = self._create_prompt(current_query, current_code, options)
                response = self._call_openai(prompt)
                selected_code, is_final, confidence = self._parse_response(response, code_options)

                if selected_code:
                    selection[stage] = selected_code
                    current_code = selected_code
                    self._log_step(step, current_code, selected_code, code_options, str(response))

                    # Get display description with context if available
                    node_info = self.tree.get_formatted_node(selected_code)
                    selected_desc = ""
                    
                    if node_info and "display_description" in node_info:
                        selected_desc = node_info["display_description"]
                    else:
                        selected_desc = next((opt["description"] for opt in code_options if opt["code"] == selected_code), "")
                    
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
        """Generate an explanation of the classification with title context"""

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

        # OPTIMIZED EXPLANATION PROMPT WITH TITLE CONTEXT AWARENESS
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
   * Highlight how the contextual hierarchy (including any title categories) informed the classification

2. Highlight the specific product features that determined each classification choice:
   * What characteristics led to the chapter selection?
   * What features determined the heading?
   * What details guided the subheading and tariff selections?
   * How did category titles or context contribute to refining the classification?

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
    """Main entry point for the script"""
    parser = argparse.ArgumentParser(description="HS Code Tree Builder & Classifier")
    parser.add_argument("command", choices=["build", "stats", "classify"], help="Command to execute")
    parser.add_argument("--json", help="JSON file with HS code data (for build)")
    parser.add_argument("--tree", help="Path for the output/input tree file")
    parser.add_argument("--product", help="Product description to classify (for classify)")
    parser.add_argument("--api-key", help="OpenAI API key (for classify)")
    
    args = parser.parse_args()
    
    if args.command == "build":
        if not args.json:
            print("Error: --json argument is required for build command")
            sys.exit(1)
            
        tree_path = args.tree or "hs_code_tree.pkl"
        tree = build_and_save_tree(args.json, tree_path)
        
        if tree:
            print(f"\nSuccess! Tree saved to {tree_path}")
            
    elif args.command == "stats":
        if not args.tree:
            print("Error: --tree argument is required for stats command")
            sys.exit(1)
            
        try:
            tree = HSCodeTree.load(args.tree)
            tree.print_stats()
        except Exception as e:
            print(f"Error loading tree: {e}")
            sys.exit(1)
            
    elif args.command == "classify":
        if not args.tree:
            print("Error: --tree argument is required for classify command")
            sys.exit(1)
            
        if not args.product:
            print("Error: --product argument is required for classify command")
            sys.exit(1)
            
        try:
            api_key = args.api_key or os.getenv("OPENAI_API_KEY") or OPENAI_API_KEY
            if not api_key:
                print("Error: OpenAI API key is required. Provide with --api-key or set OPENAI_API_KEY environment variable")
                sys.exit(1)
                
            classifier = HSCodeClassifier(args.tree, api_key)
            result = classifier.classify_with_questions(args.product)
            
            # Print a summary of the result
            print("\n====== Classification Summary ======")
            print(f"Product: {args.product}")
            print(f"Final HS Code: {result['final_code']}")
            print(f"Classification Path: {result['full_path']}")
            print("\nClassification Explanation:")
            print(result["explanation"])
            
        except Exception as e:
            print(f"Error during classification: {e}")
            import traceback
            traceback.print_exc()
            sys.exit(1)

if __name__ == "__main__":
    main()