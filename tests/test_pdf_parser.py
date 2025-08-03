"""
Test script for PDF parser module
"""

import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

from src.parsers import PDFParser, PDFToMarkdownConverter
import logging

# Set up logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)

def test_pdf_parser():
    """Test basic PDF parsing functionality"""
    # For testing, we'll create a simple demo
    print("PDF Parser Test")
    print("=" * 50)
    
    # Example usage (would need actual PDF file)
    pdf_path = "example_is_code.pdf"
    
    if not os.path.exists(pdf_path):
        print(f"Note: To test, place an IS code PDF at: {pdf_path}")
        print("\nExample usage:")
        print("```python")
        print("from src.parsers import PDFParser")
        print("")
        print("with PDFParser('IS_456_2000.pdf') as parser:")
        print("    # Extract metadata")
        print("    metadata = parser.metadata")
        print("    print(f'Title: {metadata.get(\"title\")}')")
        print("    print(f'Pages: {metadata.get(\"page_count\")}')")
        print("")
        print("    # Extract table of contents")
        print("    toc = parser.extract_toc()")
        print("    for item in toc:")
        print("        print(f'{\"  \" * (item[\"level\"]-1)}{item[\"title\"]} - Page {item[\"page\"]}')")
        print("")
        print("    # Extract content from first page")
        print("    page_content = parser.extract_text_from_page(0)")
        print("    print(f'Page 1 has {len(page_content.get(\"blocks\", []))} text blocks')")
        print("    print(f'Page 1 has {len(page_content.get(\"tables\", []))} tables')")
        print("    print(f'Page 1 has {len(page_content.get(\"images\", []))} images')")
        print("")
        print("    # Search for text")
        print("    results = parser.search_text('concrete')")
        print("    for page_num, rects in results:")
        print("        print(f'Found \"concrete\" on page {page_num + 1} at {len(rects)} locations')")
        print("```")
        return
        
    # If PDF exists, run actual test
    try:
        with PDFParser(pdf_path) as parser:
            metadata = parser.metadata
            print(f"PDF: {metadata.get('file_name')}")
            print(f"Title: {metadata.get('title')}")
            print(f"Pages: {metadata.get('page_count')}")
            print(f"Author: {metadata.get('author')}")
            print()
            
            # Test page extraction
            page_content = parser.extract_text_from_page(0)
            print(f"Page 1 Analysis:")
            print(f"  - Text blocks: {len(page_content.get('blocks', []))}")
            print(f"  - Tables: {len(page_content.get('tables', []))}")
            print(f"  - Images: {len(page_content.get('images', []))}")
            
    except Exception as e:
        print(f"Error: {e}")

def test_markdown_converter():
    """Test PDF to Markdown conversion"""
    print("\n\nPDF to Markdown Converter Test")
    print("=" * 50)
    
    pdf_path = "example_is_code.pdf"
    
    if not os.path.exists(pdf_path):
        print(f"Note: To test, place an IS code PDF at: {pdf_path}")
        print("\nExample usage:")
        print("```python")
        print("from src.parsers import PDFToMarkdownConverter")
        print("")
        print("converter = PDFToMarkdownConverter(use_gpt_vision=False)")
        print("markdown = converter.convert_pdf_to_markdown(")
        print("    'IS_10262_2019.pdf',")
        print("    'output/IS_10262_2019.md'")
        print(")")
        print("")
        print("# The markdown will have structure like:")
        print("# ---")
        print("# title: Concrete Mix Proportioning - Guidelines")
        print("# code: IS 10262")
        print("# version: 2019")
        print("# year: 2019")
        print("# ---")
        print("# ")
        print("# # IS 10262 - Concrete Mix Proportioning - Guidelines")
        print("# ")
        print("# ## 1 Scope")
        print("# This standard provides guidelines for...")
        print("```")
        return
        
    # If PDF exists, run actual test
    try:
        converter = PDFToMarkdownConverter(use_gpt_vision=False)
        
        output_path = "output/test_output.md"
        os.makedirs("output", exist_ok=True)
        
        print(f"Converting {pdf_path} to Markdown...")
        markdown = converter.convert_pdf_to_markdown(pdf_path, output_path)
        
        print(f"Markdown saved to: {output_path}")
        print(f"Markdown preview (first 500 chars):")
        print("-" * 40)
        print(markdown[:500] + "..." if len(markdown) > 500 else markdown)
        
    except Exception as e:
        print(f"Error: {e}")

if __name__ == "__main__":
    test_pdf_parser()
    test_markdown_converter()