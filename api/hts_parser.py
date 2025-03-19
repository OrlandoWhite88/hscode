import json
from typing import Dict, List, Any, Optional, Union


class HTSNode:
    """Represents a node in the HTS hierarchy."""
    
    def __init__(self, data: Dict[str, Any]):
        # Store all original data
        self.data = data
        
        # Extract common fields for easier access
        self.htsno = data.get('htsno', '')
        self.indent = int(data.get('indent', '0'))
        self.description = data.get('description', '')
        self.is_superior = data.get('superior') == 'true'
        self.units = data.get('units', [])
        self.general = data.get('general', '')
        self.special = data.get('special', '')
        self.other = data.get('other', '')
        self.footnotes = data.get('footnotes', [])
        
        # Initialize children list
        self.children = []
    
    def add_child(self, child: 'HTSNode') -> None:
        """Add a child node to this node."""
        self.children.append(child)
    
    def is_tariff_line(self) -> bool:
        """Check if this node is a complete tariff line."""
        return bool(self.htsno) and not self.is_superior
    
    def is_heading(self) -> bool:
        """Check if this node is a heading/subheading with a code."""
        return bool(self.htsno) and len(self.htsno) <= 6  # Typically 4 or 6 digits
    
    def get_chapter(self) -> Optional[str]:
        """Get the chapter (first 2 digits) of this node's HTS code."""
        if self.htsno and len(self.htsno) >= 2:
            # Extract first 2 digits, handling both "9401" and "94" formats
            return self.htsno[:2]
        return None
    
    def get_node_type(self) -> str:
        """Get the type of node based on the HTS code structure."""
        if not self.htsno:
            return "structural"
        
        # Remove periods if present to count actual digits
        clean_code = self.htsno.replace('.', '')
        
        if len(clean_code) <= 4:
            return "heading"
        elif len(clean_code) <= 6:
            return "subheading"
        elif len(clean_code) >= 10:
            return "tariff_line"
        else:
            return "intermediate"  # 8-digit codes
    
    def get_full_path(self) -> str:
        """Return the full descriptive path of this node."""
        if not self.htsno:
            return self.description
        return f"{self.htsno} - {self.description}"
    
    def to_dict(self, include_children: bool = True) -> Dict:
        """Convert the node to a dictionary representation."""
        result = {
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
        """String representation of the node."""
        prefix = '  ' * self.indent
        if self.htsno:
            return f"{prefix}{self.htsno}: {self.description} ({len(self.children)} children)"
        else:
            return f"{prefix}[GROUP] {self.description} ({len(self.children)} children)"


class HTSTree:
    """Represents the full HTS hierarchy with chapter segmentation."""
    
    def __init__(self):
        # Create a dummy root node to hold all top-level items
        self.root = HTSNode({'htsno': '', 'indent': '-1', 'description': 'ROOT', 'superior': None})
        
        # For chapter segmentation
        self.chapters = {}  # Chapter code -> nodes mapping
        self.code_index = {}  # HTS code -> node mapping for quick lookup
    
    def build_from_json(self, json_data: Union[str, List[Dict]], segment_by_chapter: bool = True) -> None:
        """
        Build the HTS tree from JSON data.
        
        Args:
            json_data: Either a JSON string or a parsed list of dictionaries
            segment_by_chapter: Whether to organize data by chapters
        """
        # Parse JSON if a string is provided
        if isinstance(json_data, str):
            data = json.loads(json_data)
        else:
            data = json_data
        
        # Track the current parent at each indent level
        parents_by_indent = {-1: self.root}
        
        # Process each item in the array
        for item in data:
            # Create a new node
            node = HTSNode(item)
            
            # Add to code index for quick lookup
            if node.htsno:
                self.code_index[node.htsno] = node
            
            # Find the parent based on indent level
            parent_indent = node.indent - 1
            
            # Look for the closest parent with a lower indent
            while parent_indent >= -1:
                if parent_indent in parents_by_indent:
                    parent = parents_by_indent[parent_indent]
                    break
                parent_indent -= 1
            else:
                # Fallback to root if no parent found
                parent = self.root
            
            # Add as child to the parent
            parent.add_child(node)
            
            # Update the parent for this indent level
            parents_by_indent[node.indent] = node
            
            # Clear any higher indent levels, as they are no longer valid parents
            for indent in list(parents_by_indent.keys()):
                if indent > node.indent:
                    del parents_by_indent[indent]
            
            # Organize by chapter if enabled
            if segment_by_chapter and node.indent == 0:  # Top-level item
                chapter = node.get_chapter()
                if chapter:
                    if chapter not in self.chapters:
                        self.chapters[chapter] = []
                    self.chapters[chapter].append(node)
    
    def get_chapters(self) -> List[str]:
        """Get a list of all chapters in the tree."""
        return sorted(self.chapters.keys())
    
    def get_chapter_nodes(self, chapter: str) -> List[HTSNode]:
        """Get all top-level nodes for a specific chapter."""
        return self.chapters.get(chapter, [])
    
    def get_chapter_tree(self, chapter: str) -> 'HTSTree':
        """
        Create a new tree containing only nodes from a specific chapter.
        This is useful for working with just one chapter at a time.
        """
        chapter_tree = HTSTree()
        
        # Add the chapter nodes to the new tree
        for node in self.get_chapter_nodes(chapter):
            chapter_tree.root.add_child(node)
            
            # Add all child nodes to the code index
            def index_node(n):
                if n.htsno:
                    chapter_tree.code_index[n.htsno] = n
                for child in n.children:
                    index_node(child)
            
            index_node(node)
            
            # Add to chapter map
            if chapter not in chapter_tree.chapters:
                chapter_tree.chapters[chapter] = []
            chapter_tree.chapters[chapter].append(node)
        
        return chapter_tree
    
    def continue_from_node(self, node: HTSNode) -> 'HTSTree':
        """
        Create a new tree with the specified node as the root.
        Useful for focusing on a specific branch of the hierarchy.
        """
        subtree = HTSTree()
        
        # Replace the dummy root with our node
        subtree.root = node
        
        # Build the code index for the subtree
        def index_node(n):
            if n.htsno:
                subtree.code_index[n.htsno] = n
            for child in n.children:
                index_node(child)
        
        index_node(node)
        
        # Organize by chapter
        chapter = node.get_chapter()
        if chapter and node.indent == 0:
            if chapter not in subtree.chapters:
                subtree.chapters[chapter] = []
            subtree.chapters[chapter].append(node)
        
        return subtree
    
    def find_by_htsno(self, htsno: str) -> Optional[HTSNode]:
        """Find a node by its HTS number using the code index."""
        return self.code_index.get(htsno)
    
    def find_by_prefix(self, prefix: str) -> List[HTSNode]:
        """Find all nodes with HTS numbers starting with the given prefix."""
        results = []
        
        for code, node in self.code_index.items():
            if code.startswith(prefix):
                results.append(node)
        
        return sorted(results, key=lambda n: n.htsno)
    
    def search_by_description(self, query: str) -> List[HTSNode]:
        """Find all nodes whose description contains the query string."""
        results = []
        query = query.lower()
        
        def search_node(node):
            if node != self.root and query in node.description.lower():
                results.append(node)
            for child in node.children:
                search_node(child)
        
        search_node(self.root)
        return results
    
    def get_flattened_structure(self) -> List[Dict]:
        """
        Get a flattened representation of the tree structure.
        Each item includes its ancestors for easy navigation.
        """
        flat_structure = []
        
        def flatten_node(node, ancestry=None, path=None):
            if ancestry is None:
                ancestry = []
            if path is None:
                path = []
            
            if node != self.root:
                node_info = {
                    'htsno': node.htsno,
                    'description': node.description,
                    'indent': node.indent,
                    'parents': ancestry.copy(),
                    'path': path.copy(),
                    'has_children': len(node.children) > 0,
                    'node_type': node.get_node_type(),
                    'duty_rates': {
                        'general': node.general,
                        'special': node.special,
                        'other': node.other
                    }
                }
                flat_structure.append(node_info)
            
            new_ancestry = ancestry.copy()
            new_path = path.copy()
            if node != self.root:
                if node.htsno:
                    new_ancestry.append(node.htsno)
                new_path.append({
                    'htsno': node.htsno,
                    'description': node.description
                })
            
            for child in node.children:
                flatten_node(child, new_ancestry, new_path)
        
        flatten_node(self.root)
        return flat_structure
    
    def to_dict(self) -> Dict:
        """Convert the tree to a dictionary representation."""
        result = {
            'root': self.root.to_dict(),
            'chapters': {}
        }
        
        # Add chapter information
        for chapter, nodes in self.chapters.items():
            result['chapters'][chapter] = [node.htsno for node in nodes]
        
        return result
    
    def print_tree(self, start_node=None) -> None:
        """
        Print the tree structure starting from a specific node or the root.
        """
        if start_node is None:
            start_node = self.root
            
        self._print_node(start_node)
    
    def _print_node(self, node: HTSNode, level: int = 0) -> None:
        """Recursive helper for print_tree."""
        prefix = '  ' * level
        
        if node != self.root:  # Skip the dummy root node
            node_info = f"{node.htsno}" if node.htsno else "[GROUP]"
            print(f"{prefix}{node_info}: {node.description}")
        
        for child in node.children:
            self._print_node(child, level + 1)


def parse_hts_json(json_data: Union[str, List[Dict]], segment_by_chapter: bool = True) -> HTSTree:
    """
    Parse HTS JSON data and return a structured tree.
    
    Args:
        json_data: Either a JSON string or a parsed list of dictionaries
        segment_by_chapter: Whether to organize data by chapters
        
    Returns:
        An HTSTree object representing the hierarchy
    """
    tree = HTSTree()
    tree.build_from_json(json_data, segment_by_chapter)
    return tree


def main():
    """Example usage."""
    # Load JSON data from file
    with open('hts_data.json', 'r', encoding='utf-8') as f:
        json_data = json.load(f)
    
    # Parse into a tree with chapter segmentation
    tree = parse_hts_json(json_data)
    
    # Print available chapters
    chapters = tree.get_chapters()
    print(f"Available chapters: {', '.join(chapters)}")
    
    # Print structure of a specific chapter
    if chapters:
        chapter = chapters[0]
        print(f"\nStructure of Chapter {chapter}:")
        chapter_tree = tree.get_chapter_tree(chapter)
        chapter_tree.print_tree()
    
    # Example of finding and continuing from a specific node
    example_code = "9403.20"
    node = tree.find_by_htsno(example_code)
    
    if node:
        print(f"\nSubtree for {example_code}:")
        subtree = tree.continue_from_node(node)
        subtree.print_tree()


if __name__ == "__main__":
    main()