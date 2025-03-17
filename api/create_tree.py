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

# Set up logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    datefmt='%Y-%m-%d %H:%M:%S'
)
logger = logging.getLogger(__name__)

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


def main():
    """Main entry point for the script"""
    parser = argparse.ArgumentParser(description="HS Code Tree Builder")
    parser.add_argument("command", choices=["build", "stats"], help="Command to execute")
    parser.add_argument("--json", help="JSON file with HS code data")
    parser.add_argument("--tree", help="Path for the output tree file (for build) or input tree file (for stats)")
    
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


if __name__ == "__main__":
    main()