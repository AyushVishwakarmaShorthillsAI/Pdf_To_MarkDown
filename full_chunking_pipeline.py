#!/usr/bin/env python3
"""
Full PDF to Markdown and Chunking Pipeline

This script combines the PDF-to-markdown conversion and intelligent chunking processes
into a single pipeline. It takes a PDF file as input and produces:
1. A fully processed markdown file with page numbers
2. A chunks JSON file with rich metadata and LLM-extracted authors

Usage:
    python full_chunking_pipeline.py

Configuration:
    Set the source_pdf variable below to your PDF filename
"""

import os
import re
import sys
import json
import fitz  # PyMuPDF
import requests
from collections import defaultdict
from typing import List, Dict, Optional, Tuple
from dataclasses import dataclass
from datetime import datetime
from docling.document_converter import DocumentConverter
from dotenv import load_dotenv

# Load environment variables
load_dotenv()

# --- CONFIGURATION ---
# Set your input PDF filename here.
source_pdf = "Dentici.pdf"
# --- END CONFIGURATION ---

@dataclass
class ChunkMetadata:
    """Structured metadata for each chunk"""
    pdf_name: str
    title: str
    authors: List[str]
    page_number: int
    current_heading: Optional[str]
    current_subheading: Optional[str]
    is_table: bool
    table_title: Optional[str]
    section_hierarchy: List[str]
    char_count: int
    sentence_count: int
    mod_date: str

class FullPipeline:
    """
    Complete pipeline for PDF to markdown to chunks conversion with rich metadata.
    """
    
    def __init__(self, pdf_path: str, max_chunk_size: int = 1024):
        self.pdf_path = pdf_path
        self.max_chunk_size = max_chunk_size
        self.pdf_filename_base = os.path.splitext(os.path.basename(pdf_path))[0]
        self.markdown_path = f"markdown_{self.pdf_filename_base}.md"
        self.chunks_path = f"chunks_{self.pdf_filename_base}.json"
        
        # Extract PDF metadata
        self.pdf_metadata = self.extract_pdf_metadata()
        print(f"Extracted PDF metadata: {self.pdf_metadata}")
    
    def extract_pdf_metadata(self) -> Dict[str, str]:
        """Extract metadata from PDF file using PyMuPDF"""
        try:
            doc = fitz.open(self.pdf_path)
            metadata = doc.metadata
            doc.close()
            
            # Clean and extract key information
            title = metadata.get('title', '').strip()
            author = metadata.get('author', '').strip()
            subject = metadata.get('subject', '').strip()
            mod_date = metadata.get('modDate', '').strip()
            
            # Parse modDate if it's in PDF format (D:YYYYMMDDHHMMSS)
            if mod_date.startswith('D:'):
                try:
                    # Remove 'D:' prefix and any timezone info
                    mod_date_clean = mod_date[2:16]  # Get YYYYMMDDHHMMSS
                    mod_date = datetime.strptime(mod_date_clean, '%Y%m%d%H%M%S').isoformat()
                except ValueError:
                    mod_date = ''  # Fallback to empty string if parsing fails
            
            # If no title in metadata, try to extract from filename
            if not title:
                title = os.path.splitext(os.path.basename(self.pdf_path))[0]
            
            # Parse authors (handle various formats)
            authors = []
            if author:
                # Handle common author separations
                author_patterns = [';', ',', ' and ', ' & ']
                author_list = [author]
                for pattern in author_patterns:
                    temp_list = []
                    for auth in author_list:
                        temp_list.extend([a.strip() for a in auth.split(pattern)])
                    author_list = temp_list
                authors = [a for a in author_list if a and len(a) > 1]
            
            return {
                'title': title,
                'authors': authors,
                'subject': subject,
                'pdf_name': os.path.basename(self.pdf_path),
                'modDate': mod_date
            }
            
        except Exception as e:
            print(f"Warning: Could not extract PDF metadata: {e}")
            return {
                'title': os.path.splitext(os.path.basename(self.pdf_path))[0],
                'authors': [],
                'subject': '',
                'pdf_name': os.path.basename(self.pdf_path),
                'modDate': ''
            }
    
    # === PDF TO MARKDOWN CONVERSION ===
    
    def _manual_resolve_cref(self, doc, cref_string):
        """
        Resolve a Docling content reference string (e.g., '#/texts/1')
        to the actual content item within the document model.
        Returns None if resolution fails.
        """
        try:
            if not isinstance(cref_string, str) or not cref_string.startswith('#/'):
                return None
            parts = cref_string.strip('#/').split('/')
            if len(parts) != 2:
                return None
            list_name, item_index = parts[0], int(parts[1])
            content_list = getattr(doc, list_name)
            return content_list[item_index]
        except (AttributeError, IndexError, ValueError, TypeError):
            return None

    def build_markdown_grouped_by_page(self, doc):
        """
        Construct Markdown grouped by original PDF page numbers using Docling's 
        provenance information, processing ALL content types correctly.
        """
        pages_content = defaultdict(list)
        
        print("Processing all content with accurate provenance-based page mapping...")
        
        # Process texts (most content)
        for text_item in getattr(doc, 'texts', []):
            try:
                # Get page number from provenance
                if hasattr(text_item, 'prov') and text_item.prov:
                    page_num = text_item.prov[0].page_no
                    # Get text content
                    content = getattr(text_item, 'text', '')
                    if content and content.strip():
                        # Format based on label (heading, paragraph, etc.)
                        label = getattr(text_item, 'label', '').lower()
                        if 'heading' in label or 'title' in label or 'section' in label:
                            formatted_content = f"## {content.strip()}"
                        else:
                            formatted_content = content.strip()
                        pages_content[page_num].append(formatted_content)
            except Exception:
                continue
        
        # Process tables
        for table_item in getattr(doc, 'tables', []):
            try:
                if hasattr(table_item, 'prov') and table_item.prov:
                    page_num = table_item.prov[0].page_no
                    # Tables should use export_to_markdown if available, or get text content
                    if hasattr(table_item, 'export_to_markdown'):
                        try:
                            table_md = table_item.export_to_markdown(doc)
                            if table_md:
                                pages_content[page_num].append(str(table_md))
                        except:
                            pass
                    # Fallback to text content if available
                    elif hasattr(table_item, 'text'):
                        content = table_item.text
                        if content and content.strip():
                            pages_content[page_num].append(content.strip())
            except Exception:
                continue
        
        # Process pictures (add placeholder)
        for pic_item in getattr(doc, 'pictures', []):
            try:
                if hasattr(pic_item, 'prov') and pic_item.prov:
                    page_num = pic_item.prov[0].page_no
                    pages_content[page_num].append("<!-- image -->")
            except Exception:
                continue
        
        # Process groups (if they have text content)
        for group_item in getattr(doc, 'groups', []):
            try:
                if hasattr(group_item, 'prov') and group_item.prov:
                    page_num = group_item.prov[0].page_no
                    if hasattr(group_item, 'text') and group_item.text:
                        content = group_item.text.strip()
                        if content:
                            pages_content[page_num].append(content)
            except Exception:
                continue
        
        # Count processed content
        total_pages = len(pages_content)
        total_items = sum(len(items) for items in pages_content.values())
        print(f"Successfully processed content across {total_pages} pages with {total_items} total items")
        print(f"Pages found: {sorted(pages_content.keys())}")
        
        # If we have very little content, fall back
        if total_pages < 2:
            print("Warning: Very few pages detected, falling back to intelligent detection...")
            return self._build_intelligent_page_markdown(doc)
        
        # Assemble full markdown with accurate page headers
        full_markdown_parts = []
        for page_num in sorted(pages_content.keys()):
            full_markdown_parts.append(f"\n## [Page {page_num}]\n\n")
            full_markdown_parts.append("\n\n".join(pages_content[page_num]))

        return "".join(full_markdown_parts)

    def _build_intelligent_page_markdown(self, doc):
        """
        Intelligent fallback: Use the simple export and detect page boundaries 
        using common academic paper patterns and page markers.
        """
        simple_markdown = doc.export_to_markdown()
        
        # Look for common page markers in academic papers
        lines = simple_markdown.split('\n')
        page_boundaries = [0]  # Start of document
        
        for i, line in enumerate(lines):
            line_clean = line.strip()
            # Common patterns that indicate page breaks
            if (re.match(r'.*\d+\s+of\s+\d+', line_clean) or  # "2 of 13"
                re.match(r'Page\s+\d+', line_clean) or         # "Page 2"  
                re.match(r'^\d+$', line_clean) or              # Just a number
                'doi.org' in line_clean):                      # DOI usually at page bottom
                page_boundaries.append(i)
        
        # If no clear markers found, estimate based on content length
        if len(page_boundaries) < 3:  # Need at least start + 1 boundary
            estimated_pages = max(1, len(simple_markdown) // 3500)
            lines_per_page = max(1, len(lines) // estimated_pages)
            page_boundaries = [i * lines_per_page for i in range(estimated_pages + 1)]
        
        # Build markdown with page headers
        full_markdown_parts = []
        for i in range(len(page_boundaries) - 1):
            start_line = page_boundaries[i]
            end_line = page_boundaries[i + 1]
            
            full_markdown_parts.append(f"\n## [Page {i + 1}]\n\n")
            page_content = "\n".join(lines[start_line:end_line])
            full_markdown_parts.append(page_content)
        
        return "".join(full_markdown_parts)

    def process_markdown_formatting(self, markdown_text):
        """
        Applies multiple formatting rules to the raw Markdown text:
        1. Converts plain text table titles into Level 2 (##) headings.
        2. Demotes existing decimal-numbered Level 2 (##) subheadings to Level 3 (###).
        """
        lines = markdown_text.split('\n')
        processed_lines = []
        
        # Rule 1: Regex for finding table titles (case-insensitive, optional period)
        table_title_pattern = re.compile(r"^\s*table\s+\d+\.?.*", re.IGNORECASE)
        
        # Rule 2: Regex for finding existing H2 subheadings with decimal numbers (e.g., "## 2.1 Title")
        subheading_pattern = re.compile(r"^##\s+(\d+\.\d+(\.\d+)*.*)")

        for line in lines:
            # Check for Rule 1: Is it a table title that needs a heading?
            if table_title_pattern.match(line) and not line.strip().startswith('#'):
                # Add '## ' prefix to make it a Level 2 heading
                processed_lines.append(f"## {line.strip()}")
            
            # Check for Rule 2: Is it a subheading that needs to be demoted?
            elif subheading_pattern.match(line):
                # Add an extra '#' to demote it from H2 to H3
                processed_lines.append(f"###{line[2:]}")
            
            # If neither rule applies, keep the line as is
            else:
                processed_lines.append(line)
                
        return "\n".join(processed_lines)
    
    def convert_pdf_to_markdown(self) -> str:
        """Convert PDF to markdown with page grouping and formatting"""
        print(f"\n=== PDF TO MARKDOWN CONVERSION ===")
        print(f"Converting '{self.pdf_path}' to Docling document model...")
        
        converter = DocumentConverter()
        result = converter.convert(self.pdf_path)
        print("Conversion complete.")

        # Step 2: Build Markdown grouped by page numbers
        print("Building Markdown grouped by page numbers...")
        grouped_markdown = self.build_markdown_grouped_by_page(result.document)

        # Step 3: Apply custom formatting rules (tables and subheadings)
        print("Applying custom formatting for tables and subheadings...")
        processed_markdown = self.process_markdown_formatting(grouped_markdown)
        print("Formatting complete.")

        # Step 4: Save the fully processed Markdown content to a file
        with open(self.markdown_path, 'w', encoding='utf-8') as f:
            f.write(processed_markdown)

        print(f"✅ Markdown saved to '{self.markdown_path}'")
        return processed_markdown
    
    # === CHUNKING WITH METADATA ===
    
    def extract_title_and_authors_from_markdown(self, markdown_content: str) -> Tuple[str, List[str]]:
        """Extract title and authors from markdown content using LLM first, then PDF metadata fallback"""
        lines = markdown_content.split('\n')
        
        title = self.pdf_metadata['title']
        authors = []
        
        # Look for title in first few lines (usually after "Article" or as first heading)
        for i, line in enumerate(lines[:20]):
            line = line.strip()
            # Look for main title (usually a ## heading early in document)
            if line.startswith('## ') and not line.startswith('## [Page'):
                potential_title = line[3:].strip()
                if len(potential_title) > 10 and not title:  # Only if we don't have a good title
                    title = potential_title
                break
        
        # Step 1: Try LLM extraction first (primary method)
        print("🤖 Attempting to extract authors using LLM from first 5 chunks...")
        llm_authors = self.extract_authors_with_llm(markdown_content)
        if llm_authors:
            authors = llm_authors
            print(f"✅ Successfully extracted {len(authors)} authors using LLM")
        else:
            # Step 2: Fallback to PDF metadata if LLM fails
            print("📄 LLM extraction failed. Falling back to PDF metadata...")
            authors = self.pdf_metadata['authors'].copy()
            if authors:
                print(f"✅ Using {len(authors)} authors from PDF metadata: {authors}")
            else:
                print("⚠️  No authors found in PDF metadata either. Proceeding with empty authors list.")
        
        return title, authors
    
    def extract_authors_with_llm(self, markdown_content: str) -> List[str]:
        """Extract authors using Gemini API from the first 5 chunks of text"""
        try:
            # Get Gemini API key from environment
            api_key = os.getenv('GEMINI_API_KEY')
            if not api_key:
                print("⚠️  No Gemini API key found. Cannot use LLM extraction.")
                return []
            
            # Create temporary chunks to get the first 5 chunks of content
            structure = self.parse_markdown_structure(markdown_content)
            temp_chunks = []
            text_buffer = ""
            chunk_count = 0
            
            for item in structure:
                if item['type'] == 'content':
                    text_buffer += (" " if text_buffer else "") + item['text']
                    
                    # Create chunk when we have enough content (roughly 800-1000 chars)
                    if len(text_buffer) >= 800:
                        temp_chunks.append(text_buffer)
                        text_buffer = ""
                        chunk_count += 1
                        
                        # Stop after 5 chunks
                        if chunk_count >= 5:
                            break
            
            # Add remaining content as last chunk if any
            if text_buffer and chunk_count < 5:
                temp_chunks.append(text_buffer)
            
            if not temp_chunks:
                print("⚠️  No content chunks found for author extraction.")
                return []
            
            # Combine first 5 chunks for analysis
            analysis_text = ' '.join(temp_chunks[:5])
            
            # Limit text length to avoid API limits
            if len(analysis_text) > 3000:
                analysis_text = analysis_text[:3000] + "..."
            
            # Prepare the prompt for Gemini
            prompt = f"""
Please analyze the following academic paper text and extract the author names. 
Return ONLY a JSON array of author names in the format: ["Author Name 1", "Author Name 2", ...]

If no clear author names are found, return an empty array: []

Text to analyze:
{analysis_text}

Response (JSON array only):"""

            # Make API call to Gemini
            url = f"https://generativelanguage.googleapis.com/v1beta/models/gemini-2.0-flash-exp:generateContent?key={api_key}"
            
            headers = {
                "Content-Type": "application/json"
            }
            
            data = {
                "contents": [
                    {
                        "parts": [
                            {
                                "text": prompt
                            }
                        ]
                    }
                ],
                "generationConfig": {
                    "temperature": 0.1,
                    "maxOutputTokens": 200
                }
            }
            
            print("🤖 Extracting authors using Gemini API...")
            response = requests.post(url, headers=headers, json=data, timeout=30)
            
            if response.status_code == 200:
                result = response.json()
                
                if 'candidates' in result and len(result['candidates']) > 0:
                    generated_text = result['candidates'][0]['content']['parts'][0]['text'].strip()
                    
                    # Try to parse the JSON response
                    try:
                        # Clean the response - remove any markdown formatting
                        cleaned_text = generated_text.replace('```json', '').replace('```', '').strip()
                        authors = json.loads(cleaned_text)
                        
                        if isinstance(authors, list) and all(isinstance(author, str) for author in authors):
                            print(f"✅ Extracted {len(authors)} authors using LLM: {authors}")
                            return authors
                        else:
                            print("⚠️  Invalid author format from LLM")
                            return []
                    except json.JSONDecodeError as e:
                        print(f"⚠️  Could not parse LLM response as JSON: {e}")
                        print(f"Raw response: {generated_text}")
                        return []
                else:
                    print("⚠️  No response from Gemini API")
                    return []
            else:
                print(f"❌ Gemini API error: {response.status_code} - {response.text}")
                return []
                
        except Exception as e:
            print(f"❌ Error in LLM author extraction: {e}")
            return []
    
    def split_into_sentences(self, text: str) -> List[str]:
        """Split text into sentences using improved regex patterns"""
        # Handle academic citations and abbreviations
        text = re.sub(r'\b([A-Z][a-z]*\.)\s+([A-Z])', r'\1<ABBREV>\2', text)  # Handle abbreviations
        text = re.sub(r'\[(\d+[,\-\d]*)\]', r'<CITATION\1>', text)  # Protect citations
        
        # Split on sentence boundaries
        sentences = re.split(r'(?<=[.!?])\s+(?=[A-Z])', text)
        
        # Restore protected patterns
        sentences = [s.replace('<ABBREV>', ' ').replace('<CITATION', '[').replace('>', ']') for s in sentences]
        
        return [s.strip() for s in sentences if s.strip()]
    
    def parse_markdown_structure(self, content: str) -> List[Dict]:
        """Parse markdown structure to identify pages, headings, and tables"""
        lines = content.split('\n')
        structure = []
        current_page = 1
        heading_stack = []  # Track heading hierarchy
        
        for i, line in enumerate(lines):
            line_stripped = line.strip()
            
            # Page boundaries
            page_match = re.match(r'## \[Page (\d+)\]', line_stripped)
            if page_match:
                current_page = int(page_match.group(1))
                structure.append({
                    'type': 'page_break',
                    'page': current_page,
                    'line_num': i
                })
                continue
            
            # Headings
            heading_match = re.match(r'^(#{1,6})\s+(.+)', line_stripped)
            if heading_match and not line_stripped.startswith('## [Page'):
                level = len(heading_match.group(1))
                heading_text = heading_match.group(2).strip()
                
                # Update heading stack
                heading_stack = heading_stack[:level-1] + [heading_text]
                
                # Check if it's a table heading (more precise detection)
                is_table = self._is_table_heading(heading_text)
                
                structure.append({
                    'type': 'heading',
                    'level': level,
                    'text': heading_text,
                    'page': current_page,
                    'line_num': i,
                    'is_table': is_table,
                    'hierarchy': heading_stack.copy()
                })
                continue
            
            # Table rows (markdown tables)
            if line_stripped.startswith('|') and '|' in line_stripped[1:]:
                structure.append({
                    'type': 'table_row',
                    'text': line_stripped,
                    'page': current_page,
                    'line_num': i
                })
                continue
            
            # Regular content - filter out page headers
            if line_stripped and not self._is_page_header(line_stripped):
                structure.append({
                    'type': 'content',
                    'text': line_stripped,
                    'page': current_page,
                    'line_num': i
                })
        
        return structure
    
    def _is_table_heading(self, heading_text: str) -> bool:
        """Check if a heading text represents an actual table heading"""
        text_lower = heading_text.lower().strip()
        
        # Must start with "table" (not just contain it)
        if not text_lower.startswith('table'):
            return False
        
        # Common table heading patterns
        table_patterns = [
            r'^table\s+\d+',           # "Table 1", "Table 2", etc.
            r'^table\s+[a-z]\d*',      # "Table A", "Table A1", etc.
            r'^table\s+[ivx]+',        # "Table I", "Table II", etc. (Roman numerals)
            r'^table\s*:',             # "Table:", "Table :"
            r'^table\s+\d+\.',         # "Table 1.", "Table 2.", etc.
            r'^table\s+\d+\s*:',       # "Table 1:", "Table 2 :", etc.
            r'^table\s+\d+\s*-',       # "Table 1-", "Table 2 -", etc.
        ]
        
        for pattern in table_patterns:
            if re.match(pattern, text_lower):
                return True
        
        # If it's just "table" by itself, it might be a table
        if text_lower == 'table':
            return True
            
        return False
    
    def create_chunks_with_context(self, content: str) -> List[Dict]:
        """Create chunks with rich contextual metadata"""
        # Extract enhanced title and authors from content if needed
        title, authors = self.extract_title_and_authors_from_markdown(content)
        
        structure = self.parse_markdown_structure(content)
        chunks = []
        
        # Track current context
        current_page = 1
        current_heading = None
        current_subheading = None
        current_table_title = None
        section_hierarchy = []
        is_in_table = False
        pages_with_tables = set()  # Track pages that actually contain table content
        
        # Collect text for chunking with page tracking
        text_buffer = ""
        page_buffer = []  # Track which page each sentence comes from
        context_buffer = {
            'page': current_page,
            'heading': current_heading,
            'subheading': current_subheading,
            'table_title': current_table_title,
            'hierarchy': section_hierarchy.copy(),
            'is_table': is_in_table
        }
        
        def flush_buffer_with_cross_page_completion():
            """Process accumulated text buffer into chunks, completing sentences across page breaks"""
            nonlocal text_buffer, page_buffer, context_buffer, chunks
            
            if not text_buffer.strip():
                return
            
            sentences = self.split_into_sentences(text_buffer)
            if not sentences:
                return
            
            # Ensure we have page info for each sentence (approximate)
            if len(page_buffer) == 0:
                page_buffer = [context_buffer['page']] * len(sentences)
            elif len(page_buffer) < len(sentences):
                # Extend page_buffer to match sentences
                last_page = page_buffer[-1] if page_buffer else context_buffer['page']
                page_buffer.extend([last_page] * (len(sentences) - len(page_buffer)))
            
            current_chunk = ""
            sentence_count = 0
            chunk_start_page = page_buffer[0] if page_buffer else context_buffer['page']
            
            # Special handling for tables - try to keep entire table in one chunk
            if context_buffer['is_table'] and len(text_buffer) <= self.max_chunk_size * 2:
                current_chunk = text_buffer
                sentence_count = len(sentences)
            else:
                # Process sentences one by one
                for i, sentence in enumerate(sentences):
                    sentence_page = page_buffer[i] if i < len(page_buffer) else context_buffer['page']
                    
                    # Check if adding this sentence would exceed limit
                    potential_chunk = current_chunk + (" " if current_chunk else "") + sentence
                    
                    if len(potential_chunk) <= self.max_chunk_size:
                        # Add sentence to current chunk
                        current_chunk = potential_chunk
                        sentence_count += 1
                    else:
                        # Current sentence would make chunk too big
                        # Save current chunk if it has content
                        if current_chunk:
                            chunk_metadata = ChunkMetadata(
                                pdf_name=self.pdf_metadata['pdf_name'],
                                title=title,
                                authors=authors,
                                page_number=chunk_start_page,
                                current_heading=context_buffer['heading'],
                                current_subheading=context_buffer['subheading'],
                                is_table=context_buffer['is_table'],
                                table_title=context_buffer['table_title'],
                                section_hierarchy=context_buffer['hierarchy'].copy(),
                                char_count=len(current_chunk),
                                sentence_count=sentence_count,
                                mod_date=self.pdf_metadata['modDate']
                            )
                            
                            chunks.append({
                                'chunk_id': f"chunk_{len(chunks) + 1:03d}",
                                'text': current_chunk,
                                'metadata': chunk_metadata.__dict__
                            })
                        
                        # Start new chunk with current sentence
                        current_chunk = sentence
                        sentence_count = 1
                        chunk_start_page = sentence_page
            
            # Don't forget the last chunk
            if current_chunk:
                chunk_metadata = ChunkMetadata(
                    pdf_name=self.pdf_metadata['pdf_name'],
                    title=title,
                    authors=authors,
                    page_number=chunk_start_page,
                    current_heading=context_buffer['heading'],
                    current_subheading=context_buffer['subheading'],
                    is_table=context_buffer['is_table'],
                    table_title=context_buffer['table_title'],
                    section_hierarchy=context_buffer['hierarchy'].copy(),
                    char_count=len(current_chunk),
                    sentence_count=sentence_count,
                    mod_date=self.pdf_metadata['modDate']
                )
                
                chunks.append({
                    'chunk_id': f"chunk_{len(chunks) + 1:03d}",
                    'text': current_chunk,
                    'metadata': chunk_metadata.__dict__
                })
            
            text_buffer = ""
            page_buffer = []
        
        # Process structure with cross-page sentence completion
        for item in structure:
            if item['type'] == 'page_break':
                # DON'T flush buffer immediately on page break
                # Just update current page tracking
                current_page = item['page']
                context_buffer['page'] = current_page
                
            elif item['type'] == 'heading':
                # Flush buffer before heading change (headings are natural break points)
                flush_buffer_with_cross_page_completion()
                
                level = item['level']
                heading_text = item['text']
                
                # Update heading context based on level
                if level == 1:
                    # Level 1 is usually the document title, keep previous heading as current
                    if heading_text.lower().startswith(('long-term', 'pathogenic', 'clinical')):
                        # This looks like a document title, don't update current_heading
                        pass
                    else:
                        current_heading = heading_text
                        current_subheading = None
                elif level == 2:
                    # Level 2 headings are main sections (like "1. Introduction", "2. Materials")
                    current_heading = heading_text
                    current_subheading = None
                elif level == 3:
                    # Level 3 headings are subsections (like "2.1. Study Objective")
                    current_subheading = heading_text
                else:
                    # Level 4+ are treated as subheadings
                    current_subheading = heading_text
                
                # Handle table titles - be more precise about table state
                if item['is_table']:
                    current_table_title = heading_text
                    is_in_table = True  # Set to true only when we encounter a table heading
                else:
                    # Reset table state when we encounter any non-table heading
                    current_table_title = None
                    is_in_table = False
                
                section_hierarchy = item['hierarchy']
                
                # Update context
                context_buffer.update({
                    'page': item['page'],
                    'heading': current_heading,
                    'subheading': current_subheading,
                    'table_title': current_table_title,
                    'hierarchy': section_hierarchy.copy(),
                    'is_table': is_in_table
                })
                
            elif item['type'] == 'table_row':
                # Only set table state if we have actual table row content (with |)
                if '|' in item['text'] and item['text'].count('|') >= 2:
                    is_in_table = True
                    context_buffer['is_table'] = True
                    # Track this page as having table content
                    pages_with_tables.add(item['page'])
                    # Add table row content to text buffer
                    text_buffer += ("\n" if text_buffer else "") + item['text']
                    # Track page for this content
                    page_buffer.append(item['page'])
                else:
                    # Not actually a table row, treat as regular content
                    text_buffer += (" " if text_buffer else "") + item['text']
                    page_buffer.append(item['page'])
                
            elif item['type'] == 'content':
                # Add content to buffer and track its page
                text_buffer += (" " if text_buffer else "") + item['text']
                page_buffer.append(item['page'])
        
        # Flush any remaining content
        flush_buffer_with_cross_page_completion()
        
        # Store pages_with_tables for later use
        self.pages_with_tables = sorted(list(pages_with_tables))
        
        return chunks
    
    def _is_page_header(self, text: str) -> bool:
        """Check if a line is a page header/footer that should be filtered out"""
        text_clean = text.strip().lower()
        
        # Skip empty or very short lines
        if len(text_clean) < 3:
            return True
        
        # Generic page number patterns
        page_patterns = [
            r'^\d+\s+of\s+\d+$',  # "3 of 13"
            r'^page\s+\d+$',  # "Page 3"
            r'^\d+$',  # Just a number
            r'^\d+\s*$',  # Number with whitespace
        ]
        
        for pattern in page_patterns:
            if re.match(pattern, text_clean):
                return True
        
        return False
    
    def detect_headers_and_footers(self, content: str) -> None:
        """Analyze document to detect repetitive headers and footers"""
        print("🔍 Analyzing document for repetitive headers and footers...")
        
        lines = content.split('\n')
        page_content = {}  # page_num -> list of lines
        current_page = 1
        
        # Group lines by page
        for line in lines:
            line_stripped = line.strip()
            if line_stripped.startswith('## [Page ') and line_stripped.endswith(']'):
                # Extract page number
                page_match = re.search(r'\[Page (\d+)\]', line_stripped)
                if page_match:
                    current_page = int(page_match.group(1))
                    continue
            
            if line_stripped:  # Skip empty lines
                if current_page not in page_content:
                    page_content[current_page] = []
                page_content[current_page].append(line_stripped)
        
        if len(page_content) < 2:
            print("⚠️  Not enough pages to detect patterns")
            self._detected_headers = []
            return
        
        # Find patterns that repeat across multiple pages
        self._detected_headers = []
        
        # Check first few lines of each page (potential headers)
        header_candidates = {}  # text -> list of pages where it appears
        footer_candidates = {}  # text -> list of pages where it appears
        
        for page_num, page_lines in page_content.items():
            if not page_lines:
                continue
                
            # Check first 3 lines for headers
            for i in range(min(3, len(page_lines))):
                line = page_lines[i].strip().lower()
                if len(line) > 5 and len(line) < 200:  # Reasonable header length
                    if line not in header_candidates:
                        header_candidates[line] = []
                    header_candidates[line].append(page_num)
            
            # Check last 3 lines for footers
            for i in range(max(0, len(page_lines) - 3), len(page_lines)):
                line = page_lines[i].strip().lower()
                if len(line) > 5 and len(line) < 200:  # Reasonable footer length
                    if line not in footer_candidates:
                        footer_candidates[line] = []
                    footer_candidates[line].append(page_num)
        
        total_pages = len(page_content)
        min_occurrences = max(2, total_pages // 2)  # Must appear on at least half the pages
        
        # Find headers that appear on multiple pages
        for text, pages in header_candidates.items():
            if len(pages) >= min_occurrences:
                print(f"📄 Detected header: '{text[:50]}...' (appears on {len(pages)} pages)")
        
        # Find footers that appear on multiple pages
        for text, pages in footer_candidates.items():
            if len(pages) >= min_occurrences:
                print(f"📄 Detected footer: '{text[:50]}...' (appears on {len(pages)} pages)")
        
        print(f"✅ Detected header/footer analysis complete")
    
    def process_chunks(self, markdown_content: str) -> List[Dict]:
        """Main chunking function"""
        print(f"\n=== INTELLIGENT CHUNKING ===")
        print(f"Processing markdown content ({len(markdown_content)} characters)")
        
        # First, detect repetitive headers and footers in the document
        self.detect_headers_and_footers(markdown_content)
        
        raw_chunks = self.create_chunks_with_context(markdown_content)
        print(f"Created {len(raw_chunks)} chunks")
        
        return raw_chunks
    
    def save_chunks_to_json(self, chunks: List[Dict]) -> None:
        """Save chunks to JSON file with metadata"""
        output_data = {
            'document_info': {
                'pdf_name': self.pdf_metadata['pdf_name'],
                'title': self.pdf_metadata['title'],
                'authors': self.pdf_metadata['authors'],
                'processed_at': datetime.now().isoformat(),
                'total_chunks': len(chunks),
                'max_chunk_size': self.max_chunk_size,
                'pages_with_table': getattr(self, 'pages_with_tables', [])
            },
            'chunks': chunks
        }
        
        with open(self.chunks_path, 'w', encoding='utf-8') as f:
            json.dump(output_data, f, indent=2, ensure_ascii=False)
        
        print(f"✅ Saved {len(chunks)} chunks to {self.chunks_path}")
        
        # Print summary statistics
        total_chars = sum(chunk['metadata']['char_count'] for chunk in chunks)
        avg_chunk_size = total_chars / len(chunks) if chunks else 0
        
        print(f"\n📊 Chunking Summary:")
        print(f"   • Total chunks: {len(chunks)}")
        print(f"   • Total characters: {total_chars:,}")
        print(f"   • Average chunk size: {avg_chunk_size:.1f} chars")
        print(f"   • Max chunk size limit: {self.max_chunk_size} chars")
        
        # Show page distribution
        page_counts = {}
        for chunk in chunks:
            page = chunk['metadata']['page_number']
            page_counts[page] = page_counts.get(page, 0) + 1
        
        print(f"   • Page distribution: {dict(sorted(page_counts.items()))}")
        
        # Show pages with tables
        if hasattr(self, 'pages_with_tables') and self.pages_with_tables:
            print(f"   • Pages with tables: {self.pages_with_tables}")
        else:
            print(f"   • Pages with tables: None detected")
    
    def run_full_pipeline(self):
        """Execute the complete pipeline from PDF to chunks"""
        print(f"🚀 Starting Full PDF to Chunks Pipeline")
        print(f"📄 Input PDF: {self.pdf_path}")
        print(f"📝 Output Markdown: {self.markdown_path}")
        print(f"📊 Output Chunks: {self.chunks_path}")
        
        try:
            # Step 1: Convert PDF to Markdown
            markdown_content = self.convert_pdf_to_markdown()
            
            # Step 2: Process into intelligent chunks
            chunks = self.process_chunks(markdown_content)
            
            # Step 3: Save chunks to JSON
            self.save_chunks_to_json(chunks)
            
            print(f"\n🎉 PIPELINE COMPLETE!")
            print(f"✅ Markdown file: {self.markdown_path}")
            print(f"✅ Chunks file: {self.chunks_path}")
            
        except Exception as e:
            print(f"\n❌ Pipeline failed: {e}")
            import traceback
            traceback.print_exc()
            sys.exit(1)

def main():
    """Main execution function"""
    # Check if the source file exists before starting
    if not os.path.exists(source_pdf):
        print(f"Error: The source file was not found at '{source_pdf}'", file=sys.stderr)
        print("Please make sure the `source_pdf` variable is set correctly.", file=sys.stderr)
        sys.exit(1)
    
    # Create and run the pipeline
    pipeline = FullPipeline(source_pdf, max_chunk_size=1024)
    pipeline.run_full_pipeline()

if __name__ == "__main__":
    main()
