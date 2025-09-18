# PDF to Markdown to Chunks Pipeline

A comprehensive pipeline for converting PDF documents to Markdown format and then intelligently chunking them into structured JSON with rich metadata. This system uses Docling for PDF-to-Markdown conversion and implements an advanced chunking strategy that preserves document structure, handles tables, and maintains contextual information.

## Overview

This project provides a complete solution for processing academic papers and documents:

1. **PDF to Markdown Conversion**: Uses Docling to convert PDF files to structured Markdown with accurate page mapping
2. **Intelligent Chunking**: Breaks down the Markdown into manageable chunks while preserving sentence boundaries, document structure, and contextual metadata
3. **Rich Metadata**: Each chunk includes comprehensive metadata including authors, page numbers, section hierarchy, and table information

## Pipeline Architecture

The main pipeline is implemented in `full_chunking_pipeline.py`, which combines functionality from:
- **`single_processor.py`**: Handles PDF-to-Markdown conversion with page grouping and formatting
- **`Chunker.py`**: Implements the intelligent chunking algorithm with metadata extraction

### Key Features

- **Sentence-level splitting**: Uses NLTK-inspired regex patterns for accurate sentence boundary detection
- **Cross-page sentence completion**: Handles sentences that span across page boundaries
- **Table preservation**: Special handling to keep tables intact and properly formatted
- **Author extraction**: Uses LLM (Gemini API) to extract authors from document content
- **Header/footer detection**: Automatically identifies and filters repetitive page elements
- **Context preservation**: Maintains section hierarchy and heading context for each chunk

## Chunking Strategy Algorithm

### Brief Overview

The chunking algorithm works in several phases:

1. **Structure Parsing**: Analyzes the Markdown to identify pages, headings, tables, and content
2. **Context Tracking**: Maintains current section hierarchy, page numbers, and table states
3. **Sentence-Based Chunking**: Splits content at sentence boundaries while respecting size limits
4. **Cross-Page Completion**: Ensures sentences spanning pages are kept together
5. **Optimization**: Merges small adjacent chunks with similar context
6. **Cleanup**: Removes page headers/footers and normalizes content

### Detailed Function Documentation

#### Core Classes and Data Structures

**`ChunkMetadata` (dataclass)**
- Structured metadata container for each chunk
- Fields: `pdf_name`, `title`, `authors`, `page_number`, `current_heading`, `current_subheading`, `is_table`, `table_title`, `section_hierarchy`, `char_count`, `sentence_count`, `mod_date`

**`MarkdownChunker` (main class)**
- Main chunker class with configurable max chunk size (default: 1024 characters)
- Handles the complete chunking pipeline from markdown input to JSON output

#### Key Functions in Chunker.py

**`extract_pdf_metadata(self) -> Dict[str, str]`**
- Extracts metadata from PDF using PyMuPDF (fitz)
- Parses title, authors, subject, and modification date
- Handles various author name formats and separations
- Provides fallback to filename if metadata is missing

**`extract_title_and_authors_from_markdown(self, markdown_content: str) -> Tuple[str, List[str]]`**
- Two-stage author extraction: LLM first, then PDF metadata fallback
- Searches for document title in first 20 lines of markdown
- Prioritizes LLM extraction for better accuracy with academic papers

**`extract_authors_with_llm(self, markdown_content: str) -> List[str]`**
- Uses Gemini API to extract author names from document content
- Analyzes first 5 chunks of content for author information
- Returns structured JSON array of author names
- Includes error handling and API timeout management

**`split_into_sentences(self, text: str) -> List[str]`**
- Advanced sentence splitting using regex patterns
- Handles academic citations (e.g., `[1,2-5]`) by protecting them during splitting
- Manages abbreviations (e.g., `Dr.`, `et al.`) to prevent false sentence breaks
- Uses pattern: `(?<=[.!?])\s+(?=[A-Z])` for sentence boundary detection

**`parse_markdown_structure(self, content: str) -> List[Dict]`**
- Parses markdown into structured elements: pages, headings, tables, content
- Tracks heading hierarchy and nesting levels
- Identifies table headings using pattern matching
- Filters out page headers and navigation elements
- Returns list of structured items with type, text, page, and hierarchy info

**`_is_table_heading(self, heading_text: str) -> bool`**
- Determines if a heading represents a table title
- Matches patterns: `Table 1`, `Table A`, `Table I` (Roman numerals)
- Handles variations: `Table 1:`, `Table 1.`, `Table 1-`
- Case-insensitive matching with flexible spacing

**`create_chunks_with_context(self, content: str) -> List[Dict]`**
- **Core chunking algorithm** - the heart of the system
- Maintains context tracking for headings, subheadings, tables, and pages
- Uses `flush_buffer_with_cross_page_completion()` nested function for chunk creation
- Special table handling: keeps tables together up to 2x max chunk size
- Implements sentence-by-sentence processing with size limit checking
- Preserves cross-page sentence continuity

**`flush_buffer_with_cross_page_completion()` (nested function)**
- Processes accumulated text buffer into properly sized chunks
- Splits text into sentences and tracks page information for each
- Handles table content specially to maintain formatting
- Creates chunks that respect sentence boundaries
- Updates metadata for each generated chunk

**`detect_headers_and_footers(self, content: str) -> None`**
- Analyzes document to find repetitive headers and footers
- Groups content by pages and examines first/last 3 lines of each page
- Identifies patterns that appear on multiple pages (minimum 50% of pages)
- Creates regex patterns for header/footer matching
- Stores detected patterns for later filtering

**`optimize_chunks(self, chunks: List[Dict]) -> List[Dict]`**
- **Post-processing optimization** to merge small adjacent chunks
- Special table merging: combines table chunks with same normalized title
- Merges chunks under 200 characters with same context (heading, page, etc.)
- Allows much larger size limits for tables (10x max chunk size)
- Preserves line breaks for tables, uses spaces for regular text
- Renumbers chunk IDs after optimization

**`fix_cross_page_breaks(self, chunks: List[Dict]) -> List[Dict]`**
- **Cross-page sentence completion** - fixes sentences split across pages
- Identifies incomplete sentences at chunk boundaries
- Uses heuristics: chunk doesn't end with sentence punctuation, next chunk starts with lowercase
- Checks context similarity between adjacent chunks
- Merges chunks when sentence continuation is detected

**`clean_page_headers_from_chunks(self, chunks: List[Dict]) -> List[Dict]`**
- Removes detected header/footer patterns from chunk content
- Uses regex patterns created during header detection phase
- Filters out generic page number patterns
- Maintains minimum content threshold (50 characters)
- Cleans up extra whitespace after header removal

**`_normalize_table_title(self, table_title: str) -> str`**
- Normalizes table titles for comparison purposes
- Removes continuation indicators: `. cont.`, ` continued`
- Extracts base table identifier (e.g., `Table 1` from `Table 1. Cont.`)
- Enables proper table chunk merging across pages

**`process_markdown(self) -> List[Dict]`**
- **Main processing pipeline** that orchestrates the chunking process
- Executes in sequence: detect headers → create chunks → optimize → fix cross-page → clean headers
- Provides progress feedback at each stage
- Returns final processed chunks ready for JSON output

**`save_chunks_to_json(self, output_path: str) -> None`**
- Saves processed chunks to JSON file with comprehensive metadata
- Creates document-level metadata including processing timestamp
- Includes summary statistics: total chunks, character counts, page distribution
- Identifies pages containing tables for reference

## Usage

### Basic Usage

```python
from Chunker import MarkdownChunker

# Create chunker instance
chunker = MarkdownChunker(
    pdf_path="document.pdf",
    markdown_path="document.md", 
    max_chunk_size=1024
)

# Process and save chunks
chunker.save_chunks_to_json("chunks_document.json")
```

### Full Pipeline Usage

```python
from full_chunking_pipeline import FullPipeline

# Create and run complete pipeline
pipeline = FullPipeline("document.pdf", max_chunk_size=1024)
pipeline.run_full_pipeline()
```

### Configuration

Set the source PDF in `full_chunking_pipeline.py`:
```python
source_pdf = "your_document.pdf"
```

For LLM author extraction, set your Gemini API key in `.env`:
```
GEMINI_API_KEY=your_api_key_here
```

## Output Format

The system generates JSON files with the following structure:

```json
{
  "document_info": {
    "pdf_name": "document.pdf",
    "title": "Document Title",
    "authors": ["Author 1", "Author 2"],
    "processed_at": "2024-01-01T12:00:00",
    "total_chunks": 45,
    "max_chunk_size": 1024,
    "pages_with_table": [3, 7, 12]
  },
  "chunks": [
    {
      "chunk_id": "chunk_001",
      "text": "Chunk content here...",
      "metadata": {
        "pdf_name": "document.pdf",
        "title": "Document Title", 
        "authors": ["Author 1", "Author 2"],
        "page_number": 1,
        "current_heading": "Introduction",
        "current_subheading": null,
        "is_table": false,
        "table_title": null,
        "section_hierarchy": ["Introduction"],
        "char_count": 856,
        "sentence_count": 12,
        "mod_date": "2024-01-01T12:00:00"
      }
    }
  ]
}
```

## Dependencies

- `docling`: PDF to Markdown conversion
- `PyMuPDF (fitz)`: PDF metadata extraction  
- `requests`: LLM API calls
- `python-dotenv`: Environment variable management
- `re`: Regular expression processing
- `json`: JSON file handling
- `dataclasses`: Structured metadata
- `datetime`: Timestamp generation

## Advanced Features

### Table Handling
- Detects table headings with patterns like "Table 1", "Table A", "Table I"
- Preserves table formatting with line breaks
- Allows larger chunk sizes for tables to keep them intact
- Merges table chunks across pages when they share the same title

### Cross-Page Processing
- Maintains sentence integrity across page boundaries
- Tracks page numbers for each sentence
- Completes partial sentences that span multiple pages
- Preserves document flow and readability

### Header/Footer Filtering
- Automatically detects repetitive page elements
- Creates flexible regex patterns for filtering
- Removes page numbers, DOI links, and journal headers
- Maintains content quality by filtering noise

### Context Preservation
- Tracks complete section hierarchy for each chunk
- Maintains heading and subheading context
- Preserves table titles and states
- Enables accurate content categorization and retrieval

This chunking strategy ensures that the resulting chunks are not only appropriately sized but also semantically meaningful and contextually rich, making them ideal for downstream applications like RAG (Retrieval-Augmented Generation) systems, document search, and content analysis.