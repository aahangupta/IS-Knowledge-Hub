"""
PDF Parser Module for IS Codes
Uses PyMuPDF for text extraction and GPT-4o for complex content interpretation
"""

import pymupdf
import logging
import json
from pathlib import Path
from typing import Dict, List, Tuple, Optional, Any
import base64
import io
from PIL import Image
import numpy as np

logger = logging.getLogger(__name__)


class PDFParser:
    """
    Parses IS Code PDFs and extracts structured content
    """
    
    def __init__(self, pdf_path: str):
        """
        Initialize PDF parser with a PDF file
        
        Args:
            pdf_path: Path to the PDF file
        """
        self.pdf_path = Path(pdf_path)
        if not self.pdf_path.exists():
            raise FileNotFoundError(f"PDF file not found: {pdf_path}")
        
        self.doc = None
        self.metadata = {}
        
    def __enter__(self):
        """Context manager entry"""
        self.doc = pymupdf.open(self.pdf_path)
        self._extract_metadata()
        return self
        
    def __exit__(self, exc_type, exc_val, exc_tb):
        """Context manager exit"""
        if self.doc:
            self.doc.close()
            
    def _extract_metadata(self) -> None:
        """Extract document metadata"""
        if not self.doc:
            return
            
        self.metadata = {
            "title": self.doc.metadata.get("title", ""),
            "author": self.doc.metadata.get("author", ""),
            "subject": self.doc.metadata.get("subject", ""),
            "keywords": self.doc.metadata.get("keywords", ""),
            "creator": self.doc.metadata.get("creator", ""),
            "producer": self.doc.metadata.get("producer", ""),
            "creation_date": str(self.doc.metadata.get("creationDate", "")),
            "modification_date": str(self.doc.metadata.get("modDate", "")),
            "page_count": self.doc.page_count,
            "file_name": self.pdf_path.name
        }
        
    def extract_text_from_page(self, page_num: int) -> Dict[str, Any]:
        """
        Extract structured text from a single page
        
        Args:
            page_num: Page number (0-indexed)
            
        Returns:
            Dictionary containing page content and structure
        """
        if not self.doc or page_num >= self.doc.page_count:
            return {}
            
        page = self.doc[page_num]
        
        # Extract text with detailed structure
        text_dict = page.get_text("dict", flags=pymupdf.TEXTFLAGS_TEXT)
        
        # Extract tables
        tables = self._extract_tables(page)
        
        # Extract images
        images = self._extract_images(page)
        
        # Build structured content
        page_content = {
            "page_number": page_num + 1,
            "width": page.rect.width,
            "height": page.rect.height,
            "blocks": [],
            "tables": tables,
            "images": images
        }
        
        # Process text blocks
        for block in text_dict.get("blocks", []):
            if block["type"] == 0:  # Text block
                page_content["blocks"].append(self._process_text_block(block))
                
        return page_content
        
    def _process_text_block(self, block: Dict) -> Dict:
        """Process a text block and extract structured information"""
        processed_block = {
            "bbox": block["bbox"],
            "lines": []
        }
        
        for line in block.get("lines", []):
            processed_line = {
                "bbox": line["bbox"],
                "spans": []
            }
            
            for span in line.get("spans", []):
                processed_span = {
                    "text": span["text"],
                    "font": span.get("font", ""),
                    "size": span.get("size", 0),
                    "flags": span.get("flags", 0),
                    "color": self._convert_color(span.get("color", 0)),
                    "bbox": span["bbox"]
                }
                processed_line["spans"].append(processed_span)
                
            processed_block["lines"].append(processed_line)
            
        return processed_block
        
    def _convert_color(self, color_int: int) -> str:
        """Convert integer color to hex string"""
        return f"#{color_int:06x}"
        
    def _extract_tables(self, page: pymupdf.Page) -> List[Dict]:
        """Extract tables from a page"""
        tables = []
        
        try:
            # Find tables on the page
            tabs = page.find_tables()
            
            for i, tab in enumerate(tabs.tables):
                table_data = {
                    "index": i,
                    "bbox": tab.bbox,
                    "rows": tab.row_count,
                    "cols": tab.col_count,
                    "cells": tab.extract()
                }
                tables.append(table_data)
                
        except Exception as e:
            logger.warning(f"Error extracting tables: {e}")
            
        return tables
        
    def _extract_images(self, page: pymupdf.Page) -> List[Dict]:
        """Extract images from a page"""
        images = []
        
        try:
            image_list = page.get_images()
            
            for img_index, img in enumerate(image_list):
                xref = img[0]
                
                # Extract image
                img_dict = self.doc.extract_image(xref)
                
                if img_dict:
                    image_info = {
                        "index": img_index,
                        "xref": xref,
                        "width": img_dict.get("width", 0),
                        "height": img_dict.get("height", 0),
                        "format": img_dict.get("ext", ""),
                        "size": len(img_dict.get("image", b"")),
                        "base64": base64.b64encode(img_dict.get("image", b"")).decode("utf-8")
                    }
                    images.append(image_info)
                    
        except Exception as e:
            logger.warning(f"Error extracting images: {e}")
            
        return images
        
    def extract_all_pages(self) -> List[Dict]:
        """Extract content from all pages"""
        if not self.doc:
            return []
            
        pages_content = []
        
        for page_num in range(self.doc.page_count):
            logger.info(f"Extracting page {page_num + 1}/{self.doc.page_count}")
            page_content = self.extract_text_from_page(page_num)
            pages_content.append(page_content)
            
        return pages_content
        
    def get_page_as_image(self, page_num: int, zoom: float = 2.0) -> Optional[bytes]:
        """
        Render a page as an image for GPT-4o vision processing
        
        Args:
            page_num: Page number (0-indexed)
            zoom: Zoom factor for rendering
            
        Returns:
            PNG image as bytes
        """
        if not self.doc or page_num >= self.doc.page_count:
            return None
            
        page = self.doc[page_num]
        
        # Create transformation matrix for zoom
        mat = pymupdf.Matrix(zoom, zoom)
        
        # Render page to pixmap
        pix = page.get_pixmap(matrix=mat)
        
        # Convert to PNG bytes
        img_data = pix.tobytes("png")
        
        return img_data
        
    def search_text(self, query: str, case_sensitive: bool = False) -> List[Tuple[int, List[pymupdf.Rect]]]:
        """
        Search for text across all pages
        
        Args:
            query: Text to search for
            case_sensitive: Whether search is case-sensitive
            
        Returns:
            List of tuples (page_number, list_of_rectangles)
        """
        if not self.doc:
            return []
            
        results = []
        
        for page_num in range(self.doc.page_count):
            page = self.doc[page_num]
            
            # Search for text on page
            rects = page.search_for(query, quads=False)
            
            if rects:
                results.append((page_num, rects))
                
        return results
        
    def extract_toc(self) -> List[Dict]:
        """Extract table of contents"""
        if not self.doc:
            return []
            
        toc = self.doc.get_toc()
        
        structured_toc = []
        for level, title, page in toc:
            structured_toc.append({
                "level": level,
                "title": title,
                "page": page
            })
            
        return structured_toc