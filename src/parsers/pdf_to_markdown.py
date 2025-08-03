"""
PDF to Markdown Converter for IS Codes
Uses GPT-4o vision for complex content interpretation
"""

import re
import logging
from typing import Dict, List, Any, Optional
from pathlib import Path
import yaml
from datetime import datetime
import openai
from config import settings
from .pdf_parser import PDFParser

logger = logging.getLogger(__name__)


class PDFToMarkdownConverter:
    """
    Converts IS Code PDFs to structured Markdown format
    """
    
    def __init__(self, use_gpt_vision: bool = True):
        """
        Initialize converter
        
        Args:
            use_gpt_vision: Whether to use GPT-4o vision for complex content
        """
        self.use_gpt_vision = use_gpt_vision
        if use_gpt_vision:
            openai.api_key = settings.openai_api_key
            
    def convert_pdf_to_markdown(self, pdf_path: str, output_path: Optional[str] = None) -> str:
        """
        Convert a PDF to Markdown format
        
        Args:
            pdf_path: Path to input PDF
            output_path: Optional path to save markdown file
            
        Returns:
            Markdown content as string
        """
        with PDFParser(pdf_path) as parser:
            # Extract metadata
            metadata = parser.metadata
            
            # Extract table of contents
            toc = parser.extract_toc()
            
            # Extract content from all pages
            pages_content = parser.extract_all_pages()
            
            # Identify IS code information
            is_code_info = self._identify_is_code(metadata, pages_content)
            
            # Build markdown
            markdown_content = self._build_markdown(
                is_code_info,
                metadata,
                toc,
                pages_content,
                parser
            )
            
            # Save if output path provided
            if output_path:
                Path(output_path).write_text(markdown_content, encoding='utf-8')
                logger.info(f"Markdown saved to: {output_path}")
                
            return markdown_content
            
    def _identify_is_code(self, metadata: Dict, pages_content: List[Dict]) -> Dict:
        """Identify IS code number, title, and version from content"""
        is_code_info = {
            "code": "",
            "title": "",
            "version": "",
            "year": ""
        }
        
        # Try to extract from first few pages
        first_page_text = self._extract_text_from_blocks(pages_content[0]["blocks"]) if pages_content else ""
        
        # Pattern for IS code (e.g., IS 10262:2019)
        is_pattern = r'IS\s+(\d+)(?::(\d{4}))?'
        match = re.search(is_pattern, first_page_text)
        
        if match:
            is_code_info["code"] = f"IS {match.group(1)}"
            if match.group(2):
                is_code_info["year"] = match.group(2)
                is_code_info["version"] = match.group(2)
                
        # Extract title - usually follows the IS code
        title_pattern = r'IS\s+\d+(?::\d{4})?\s*[-–]\s*(.+?)(?:\n|$)'
        title_match = re.search(title_pattern, first_page_text, re.MULTILINE)
        
        if title_match:
            is_code_info["title"] = title_match.group(1).strip()
            
        return is_code_info
        
    def _extract_text_from_blocks(self, blocks: List[Dict]) -> str:
        """Extract plain text from blocks"""
        text_parts = []
        
        for block in blocks:
            for line in block.get("lines", []):
                line_text = ""
                for span in line.get("spans", []):
                    line_text += span.get("text", "")
                if line_text.strip():
                    text_parts.append(line_text.strip())
                    
        return "\n".join(text_parts)
        
    def _build_markdown(
        self,
        is_code_info: Dict,
        metadata: Dict,
        toc: List[Dict],
        pages_content: List[Dict],
        parser: PDFParser
    ) -> str:
        """Build the complete markdown document"""
        
        # Build YAML frontmatter
        frontmatter = {
            "title": is_code_info.get("title", metadata.get("title", "")),
            "code": is_code_info.get("code", ""),
            "version": is_code_info.get("version", ""),
            "year": is_code_info.get("year", ""),
            "source": metadata.get("file_name", ""),
            "pages": metadata.get("page_count", 0),
            "parsed_date": datetime.now().isoformat(),
            "parser_version": "1.0.0"
        }
        
        markdown_parts = [
            "---",
            yaml.dump(frontmatter, default_flow_style=False).strip(),
            "---",
            "",
            f"# {is_code_info.get('code', '')} - {is_code_info.get('title', '')}",
            ""
        ]
        
        # Add table of contents if available
        if toc:
            markdown_parts.extend([
                "## Table of Contents",
                ""
            ])
            
            for item in toc:
                indent = "  " * (item["level"] - 1)
                markdown_parts.append(f"{indent}- [{item['title']}](#{self._slugify(item['title'])})")
                
            markdown_parts.append("")
            
        # Process each page
        for page_idx, page_content in enumerate(pages_content):
            page_num = page_idx + 1
            
            # Check if page needs GPT-4o vision processing
            if self._needs_vision_processing(page_content):
                logger.info(f"Processing page {page_num} with GPT-4o vision")
                vision_content = self._process_page_with_vision(parser, page_idx)
                if vision_content:
                    markdown_parts.append(vision_content)
                    continue
                    
            # Standard processing
            page_markdown = self._convert_page_to_markdown(page_content)
            if page_markdown.strip():
                markdown_parts.append(page_markdown)
                
        return "\n".join(markdown_parts)
        
    def _needs_vision_processing(self, page_content: Dict) -> bool:
        """Determine if a page needs GPT-4o vision processing"""
        # Complex criteria:
        # - Has multiple tables
        # - Has images with technical diagrams
        # - Has complex multi-column layout
        # - Has equations or formulas
        
        has_multiple_tables = len(page_content.get("tables", [])) > 1
        has_images = len(page_content.get("images", [])) > 0
        
        # Check for complex layout (multiple columns)
        blocks = page_content.get("blocks", [])
        if len(blocks) > 10:
            x_positions = [block["bbox"][0] for block in blocks]
            # If x positions vary significantly, likely multi-column
            if x_positions and (max(x_positions) - min(x_positions)) > 200:
                return True
                
        return has_multiple_tables or has_images
        
    def _process_page_with_vision(self, parser: PDFParser, page_idx: int) -> Optional[str]:
        """Process a page using GPT-4o vision"""
        if not self.use_gpt_vision:
            return None
            
        try:
            # Get page as image
            image_data = parser.get_page_as_image(page_idx)
            if not image_data:
                return None
                
            # Prepare the prompt
            prompt = """
            Please convert this IS code page to clean Markdown format.
            
            Guidelines:
            1. Use proper heading hierarchy (##, ###, ####) for clauses and sub-clauses
            2. Convert tables to markdown table format
            3. Convert equations to LaTeX format using $ for inline and $$ for block equations
            4. For images/figures, use format: ![Figure X: Caption](figure-x.png)
            5. Preserve all text content accurately
            6. Maintain the structure and numbering of clauses
            
            Output only the Markdown content, no explanations.
            """
            
            # Call GPT-4o vision (placeholder - actual implementation would use the API)
            # For now, return None to use standard processing
            logger.warning("GPT-4o vision integration not yet implemented")
            return None
            
        except Exception as e:
            logger.error(f"Error in vision processing: {e}")
            return None
            
    def _convert_page_to_markdown(self, page_content: Dict) -> str:
        """Convert page content to markdown using standard processing"""
        markdown_parts = []
        
        # Process text blocks
        current_section = []
        
        for block in page_content.get("blocks", []):
            block_text = self._extract_text_from_block(block)
            
            if block_text:
                # Detect headings
                if self._is_heading(block):
                    if current_section:
                        markdown_parts.append(" ".join(current_section))
                        current_section = []
                        
                    heading_level = self._get_heading_level(block_text)
                    markdown_parts.append(f"\n{'#' * heading_level} {block_text}\n")
                else:
                    current_section.append(block_text)
                    
        if current_section:
            markdown_parts.append(" ".join(current_section))
            
        # Process tables
        for table in page_content.get("tables", []):
            markdown_parts.append("\n" + self._table_to_markdown(table) + "\n")
            
        return "\n".join(markdown_parts)
        
    def _extract_text_from_block(self, block: Dict) -> str:
        """Extract text from a block"""
        text_parts = []
        
        for line in block.get("lines", []):
            line_text = ""
            for span in line.get("spans", []):
                line_text += span.get("text", "")
            if line_text.strip():
                text_parts.append(line_text.strip())
                
        return " ".join(text_parts)
        
    def _is_heading(self, block: Dict) -> bool:
        """Determine if a block is a heading"""
        # Check font size and style
        for line in block.get("lines", []):
            for span in line.get("spans", []):
                # Larger font size or bold indicates heading
                if span.get("size", 0) > 14 or span.get("flags", 0) & 2**4:
                    return True
                    
        # Check for clause pattern
        text = self._extract_text_from_block(block)
        if re.match(r'^\d+\.?\d*\s+\w+', text):
            return True
            
        return False
        
    def _get_heading_level(self, text: str) -> int:
        """Determine heading level based on text pattern"""
        # Clause patterns
        if re.match(r'^\d+\s+', text):  # Main clause (e.g., "5 Concrete")
            return 2
        elif re.match(r'^\d+\.\d+\s+', text):  # Sub-clause (e.g., "5.1 General")
            return 3
        elif re.match(r'^\d+\.\d+\.\d+\s+', text):  # Sub-sub-clause
            return 4
        else:
            return 2  # Default
            
    def _table_to_markdown(self, table: Dict) -> str:
        """Convert table to markdown format"""
        cells = table.get("cells", [])
        if not cells:
            return ""
            
        markdown_lines = []
        
        # Process header row
        if len(cells) > 0:
            header = "| " + " | ".join(str(cell) if cell else "" for cell in cells[0]) + " |"
            markdown_lines.append(header)
            
            # Separator
            separator = "|" + "|".join([" --- " for _ in cells[0]]) + "|"
            markdown_lines.append(separator)
            
            # Data rows
            for row in cells[1:]:
                row_text = "| " + " | ".join(str(cell) if cell else "" for cell in row) + " |"
                markdown_lines.append(row_text)
                
        return "\n".join(markdown_lines)
        
    def _slugify(self, text: str) -> str:
        """Create a URL-friendly slug from text"""
        # Remove special characters and convert to lowercase
        slug = re.sub(r'[^\w\s-]', '', text.lower())
        # Replace spaces with hyphens
        slug = re.sub(r'[-\s]+', '-', slug)
        return slug