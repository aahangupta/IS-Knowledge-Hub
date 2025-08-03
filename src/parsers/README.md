# PDF Parsing Module

This module provides functionality to parse IS Code PDFs and convert them to structured Markdown format.

## Features

### PDFParser
- Extract text, tables, and images from PDFs
- Search functionality across all pages
- Table of contents extraction
- Metadata extraction
- Page rendering for vision processing

### PDFToMarkdownConverter
- Convert IS Code PDFs to clean Markdown
- YAML frontmatter with metadata
- Proper heading hierarchy for clauses
- Table preservation in Markdown format
- Image extraction and referencing
- Optional GPT-4o vision processing for complex pages

## Usage

### Basic PDF Parsing

```python
from src.parsers import PDFParser

# Parse a PDF file
with PDFParser('IS_456_2000.pdf') as parser:
    # Get metadata
    metadata = parser.metadata
    print(f"Title: {metadata['title']}")
    print(f"Pages: {metadata['page_count']}")
    
    # Extract table of contents
    toc = parser.extract_toc()
    
    # Extract content from a specific page
    page_content = parser.extract_text_from_page(0)
    
    # Search for text
    results = parser.search_text('concrete')
    for page_num, rects in results:
        print(f"Found on page {page_num + 1}")
```

### PDF to Markdown Conversion

```python
from src.parsers import PDFToMarkdownConverter

# Create converter
converter = PDFToMarkdownConverter(use_gpt_vision=True)

# Convert PDF to Markdown
markdown = converter.convert_pdf_to_markdown(
    'IS_10262_2019.pdf',
    'output/IS_10262_2019.md'
)
```

## Output Format

The generated Markdown follows this structure:

```markdown
---
title: Concrete Mix Proportioning - Guidelines
code: IS 10262
version: 2019
year: 2019
source: IS_10262_2019.pdf
pages: 40
parsed_date: 2024-01-20T10:30:00
parser_version: 1.0.0
---

# IS 10262 - Concrete Mix Proportioning - Guidelines

## Table of Contents

- [1 Scope](#1-scope)
- [2 References](#2-references)
- [3 Terminology](#3-terminology)
  - [3.1 Definitions](#31-definitions)

## 1 Scope

This standard provides guidelines for proportioning concrete mixes...

### 1.1 General

The proportioning of concrete mix...

## 2 References

The following standards contain provisions...

| IS No. | Title |
| --- | --- |
| IS 456:2000 | Plain and Reinforced Concrete - Code of Practice |
| IS 383:2016 | Coarse and Fine Aggregate for Concrete |
```

## GPT-4o Vision Integration

For complex pages with:
- Multiple tables
- Technical diagrams
- Multi-column layouts
- Mathematical equations

The converter can use GPT-4o vision to interpret the content more accurately. This requires:
1. Valid OpenAI API key in environment
2. `use_gpt_vision=True` parameter

## Error Handling

The parser handles common issues:
- Missing or corrupted PDFs
- Pages without text content
- Failed table extraction
- Image extraction errors

Errors are logged but don't stop the overall parsing process.

## Performance Considerations

- Large PDFs (>100 pages) may take several minutes to process
- GPT-4o vision processing adds ~2-5 seconds per complex page
- Memory usage scales with PDF size and image content

## Future Enhancements

- [ ] OCR support for scanned PDFs
- [ ] Batch processing for multiple PDFs
- [ ] Parallel page processing
- [ ] Custom extraction rules for specific IS codes
- [ ] Equation extraction and LaTeX conversion