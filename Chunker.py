import os
import re
import json
import fitz  # PyMuPDF
from typing import List, Dict, Optional, Tuple
from dataclasses import dataclass
from datetime import datetime
import requests
from dotenv import load_dotenv

# Load environment variables
load_dotenv()

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

class MarkdownChunker:
    """
    Advanced markdown chunker that creates up to 1024-character chunks with rich metadata.
    Preserves sentence boundaries, tracks contextual information, and optimizes chunk sizes
    by merging small adjacent chunks with the same context.
    """
    
    def __init__(self, pdf_path: str, markdown_path: str, max_chunk_size: int = 1024):
        self.pdf_path = pdf_path
        self.markdown_path = markdown_path
        self.max_chunk_size = max_chunk_size
        self.chunks = []
        
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
            url = f"https://generativelanguage.googleapis.com/v1beta/models/gemini-2.5-flash:generateContent?key={api_key}"
            
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
        import re
        
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
        import re
        page_patterns = [
            r'^\d+\s+of\s+\d+$',  # "3 of 13"
            r'^page\s+\d+$',  # "Page 3"
            r'^\d+$',  # Just a number
            r'^\d+\s*$',  # Number with whitespace
        ]
        
        for pattern in page_patterns:
            if re.match(pattern, text_clean):
                return True
        
        # Check against detected headers/footers if available
        if hasattr(self, '_detected_headers') and self._detected_headers:
            for header_pattern in self._detected_headers:
                if self._text_matches_pattern(text_clean, header_pattern):
                    return True
        
        return False
    
    def _text_matches_pattern(self, text: str, pattern: dict) -> bool:
        """Check if text matches a detected header/footer pattern"""
        import re
        
        # Exact match
        if text == pattern['text'].lower():
            return True
        
        # Regex pattern match (for patterns with variable parts like dates/numbers)
        if 'regex' in pattern:
            if re.match(pattern['regex'], text):
                return True
        
        # Similarity match for slight variations
        if 'keywords' in pattern:
            text_words = set(text.split())
            pattern_words = set(pattern['keywords'])
            # If most key words match, consider it a header
            overlap = len(text_words & pattern_words)
            if overlap >= len(pattern_words) * 0.7:  # 70% word overlap
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
                import re
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
                # Create pattern for this header
                pattern = self._create_pattern_from_text(text, pages, 'header')
                self._detected_headers.append(pattern)
                print(f"📄 Detected header: '{text[:50]}...' (appears on {len(pages)} pages)")
        
        # Find footers that appear on multiple pages
        for text, pages in footer_candidates.items():
            if len(pages) >= min_occurrences:
                # Create pattern for this footer
                pattern = self._create_pattern_from_text(text, pages, 'footer')
                self._detected_headers.append(pattern)  # Add to same list
                print(f"📄 Detected footer: '{text[:50]}...' (appears on {len(pages)} pages)")
        
        print(f"✅ Detected {len(self._detected_headers)} header/footer patterns")
    
    def _create_pattern_from_text(self, text: str, pages: list, pattern_type: str) -> dict:
        """Create a pattern object from detected repetitive text"""
        import re
        
        pattern = {
            'text': text,
            'pages': pages,
            'type': pattern_type,
            'keywords': text.split()
        }
        
        # Try to create a regex pattern for variations
        # Replace numbers with \d+ to catch page numbers, dates, etc.
        regex_text = re.escape(text)
        regex_text = re.sub(r'\\d+', r'\\d+', regex_text)  # Allow number variations
        regex_text = re.sub(r'\\\s+', r'\\s+', regex_text)  # Allow whitespace variations
        
        pattern['regex'] = f"^{regex_text}$"
        
        return pattern
    
    def _normalize_table_title(self, table_title: str) -> str:
        """Normalize table title for comparison (e.g., 'Table 1. Cont.' -> 'Table 1')"""
        if not table_title:
            return ""
        
        # Remove common continuation indicators
        normalized = table_title.lower()
        normalized = normalized.replace('. cont.', '').replace(' cont.', '')
        normalized = normalized.replace('. continued', '').replace(' continued', '')
        
        # Extract base table identifier (e.g., "table 1", "table 2")
        import re
        match = re.match(r'(table\s+\d+)', normalized)
        if match:
            return match.group(1)
        
        return normalized.strip()
    
    def optimize_chunks(self, chunks: List[Dict]) -> List[Dict]:
        """Merge adjacent small chunks with same context to avoid tiny fragments"""
        if not chunks:
            return chunks
        
        optimized_chunks = []
        i = 0
        
        while i < len(chunks):
            current_chunk = chunks[i]
            
            # Special handling for table merging - look for table chunks to merge
            if current_chunk['metadata']['is_table'] and '|' in current_chunk['text']:
                # This is a real table chunk with data, try to merge with other table chunks
                j = i + 1
                while j < len(chunks):
                    candidate_chunk = chunks[j]
                    candidate_meta = candidate_chunk['metadata']
                    
                    # Skip small page header chunks between table data
                    if (candidate_meta['is_table'] and 
                        candidate_meta['char_count'] < 100 and
                        '|' not in candidate_chunk['text']):
                        j += 1
                        continue
                    
                    # Check if this is another table chunk with same normalized title
                    if (candidate_meta['is_table'] and '|' in candidate_chunk['text']):
                        current_table_base = self._normalize_table_title(current_chunk['metadata']['table_title'])
                        candidate_table_base = self._normalize_table_title(candidate_meta['table_title'])
                        
                        if current_table_base == candidate_table_base:
                            # Try to merge
                            combined_text = current_chunk['text'] + "\n" + candidate_chunk['text']
                            if len(combined_text) <= self.max_chunk_size * 10:
                                # Merge the chunks
                                current_chunk['text'] = combined_text
                                current_chunk['metadata']['char_count'] = len(combined_text)
                                current_chunk['metadata']['sentence_count'] += candidate_meta['sentence_count']
                                
                                # Remove the merged chunk from list
                                chunks.pop(j)
                                continue
                    
                    # If we can't merge with this chunk, stop looking
                    break
                
                optimized_chunks.append(current_chunk)
                i += 1
                
            # If current chunk is small (<200 chars), try to merge with next chunks
            elif (current_chunk['metadata']['char_count'] < 200 and 
                i + 1 < len(chunks)):
                
                next_chunk = chunks[i + 1]
                
                # Check if chunks have same context (heading, subheading, page)
                current_meta = current_chunk['metadata']
                next_meta = next_chunk['metadata']
                
                # Special logic for tables - merge if same table title even across pages
                if current_meta['is_table'] and next_meta['is_table']:
                    # Normalize table titles for comparison (handle "Table 1" vs "Table 1. Cont.")
                    current_table_base = self._normalize_table_title(current_meta['table_title'])
                    next_table_base = self._normalize_table_title(next_meta['table_title'])
                    same_context = (
                        current_table_base == next_table_base and
                        current_meta['is_table'] == next_meta['is_table']
                    )
                else:
                    # Regular context matching for non-table content
                    same_context = (
                        current_meta['page_number'] == next_meta['page_number'] and
                        current_meta['current_heading'] == next_meta['current_heading'] and
                        current_meta['current_subheading'] == next_meta['current_subheading'] and
                        current_meta['is_table'] == next_meta['is_table']
                    )
                
                # If same context and combined size is within limit, merge
                # For tables, preserve line breaks; for regular text, use spaces
                if current_meta['is_table']:
                    combined_text = current_chunk['text'] + "\n" + next_chunk['text']
                else:
                    combined_text = current_chunk['text'] + " " + next_chunk['text']
                # Allow much larger size limit for tables to keep them together
                size_limit = self.max_chunk_size * 10 if current_meta['is_table'] else self.max_chunk_size
                if same_context and len(combined_text) <= size_limit:
                    # Create merged chunk
                    merged_chunk = {
                        'chunk_id': current_chunk['chunk_id'],
                        'text': combined_text,
                        'metadata': current_meta.copy()
                    }
                    # Update metadata for merged chunk
                    merged_chunk['metadata']['char_count'] = len(combined_text)
                    merged_chunk['metadata']['sentence_count'] = (
                        current_meta['sentence_count'] + next_meta['sentence_count']
                    )
                    
                    # Try to merge with more chunks if possible
                    j = i + 2
                    while j < len(chunks):
                        candidate_chunk = chunks[j]
                        candidate_meta = candidate_chunk['metadata']
                        
                        # Check same context - special logic for tables
                        if current_meta['is_table'] and candidate_meta['is_table']:
                            # Normalize table titles for comparison
                            current_table_base = self._normalize_table_title(current_meta['table_title'])
                            candidate_table_base = self._normalize_table_title(candidate_meta['table_title'])
                            same_context_candidate = (
                                current_table_base == candidate_table_base and
                                current_meta['is_table'] == candidate_meta['is_table']
                            )
                        else:
                            same_context_candidate = (
                                current_meta['page_number'] == candidate_meta['page_number'] and
                                current_meta['current_heading'] == candidate_meta['current_heading'] and
                                current_meta['current_subheading'] == candidate_meta['current_subheading'] and
                                current_meta['is_table'] == candidate_meta['is_table']
                            )
                        
                        # Preserve line breaks for tables, use spaces for regular text
                        if current_meta['is_table']:
                            potential_text = merged_chunk['text'] + "\n" + candidate_chunk['text']
                        else:
                            potential_text = merged_chunk['text'] + " " + candidate_chunk['text']
                        
                        # Allow much larger size limit for tables to keep them together
                        size_limit = self.max_chunk_size * 10 if current_meta['is_table'] else self.max_chunk_size
                        if (same_context_candidate and 
                            len(potential_text) <= size_limit):
                            # Merge this chunk too
                            merged_chunk['text'] = potential_text
                            merged_chunk['metadata']['char_count'] = len(potential_text)
                            merged_chunk['metadata']['sentence_count'] += candidate_meta['sentence_count']
                            j += 1
                        else:
                            break
                    
                    optimized_chunks.append(merged_chunk)
                    i = j  # Skip all merged chunks
                else:
                    # Can't merge, add as is
                    optimized_chunks.append(current_chunk)
                    i += 1
            else:
                # Chunk is large enough or can't merge, add as is
                optimized_chunks.append(current_chunk)
                i += 1
        
        # Renumber chunk IDs
        for idx, chunk in enumerate(optimized_chunks):
            chunk['chunk_id'] = f"chunk_{idx + 1:03d}"
        
        return optimized_chunks
    
    def fix_cross_page_breaks(self, chunks: List[Dict]) -> List[Dict]:
        """Fix sentence breaks that occur at page boundaries"""
        if not chunks:
            return chunks
        
        fixed_chunks = []
        i = 0
        
        while i < len(chunks):
            current_chunk = chunks[i]
            
            # Check if this chunk might have a sentence break at the end
            if (i + 1 < len(chunks) and 
                not current_chunk['metadata']['is_table'] and  # Don't merge tables
                current_chunk['metadata']['page_number'] + 1 == chunks[i + 1]['metadata']['page_number']):  # Adjacent pages
                
                next_chunk = chunks[i + 1]
                
                # Check if current chunk ends mid-sentence and next chunk continues it
                current_text = current_chunk['text'].strip()
                next_text = next_chunk['text'].strip()
                
                # Heuristics for sentence continuation:
                # 1. Current chunk doesn't end with sentence-ending punctuation
                # 2. Next chunk starts with lowercase or continues the thought
                # 3. Same or similar context (heading/subheading)
                
                current_ends_incomplete = not current_text.endswith(('.', '!', '?', ':', ';'))
                next_starts_continuation = (
                    next_text and 
                    (next_text[0].islower() or 
                     next_text.startswith(('and', 'or', 'but', 'however', 'therefore', 'thus', 'in', 'of', 'to', 'the', 'a')))
                )
                
                # Check context similarity (but be more lenient for cross-page content)
                same_or_similar_context = (
                    current_chunk['metadata']['current_heading'] == next_chunk['metadata']['current_heading'] or
                    (not current_chunk['metadata']['current_heading'] and not next_chunk['metadata']['current_heading'])
                )
                
                # If it looks like a sentence break, try to merge
                if (current_ends_incomplete and next_starts_continuation and same_or_similar_context):
                    combined_text = current_text + " " + next_text
                    
                    # Only merge if the combined size is reasonable
                    if len(combined_text) <= self.max_chunk_size * 2:
                        # Create merged chunk
                        merged_chunk = {
                            'chunk_id': current_chunk['chunk_id'],
                            'text': combined_text,
                            'metadata': current_chunk['metadata'].copy()
                        }
                        # Update metadata
                        merged_chunk['metadata']['char_count'] = len(combined_text)
                        merged_chunk['metadata']['sentence_count'] = (
                            current_chunk['metadata']['sentence_count'] + 
                            next_chunk['metadata']['sentence_count']
                        )
                        
                        fixed_chunks.append(merged_chunk)
                        i += 2  # Skip both chunks
                        continue
            
            # If no merging, add chunk as-is
            fixed_chunks.append(current_chunk)
            i += 1
        
        # Renumber chunk IDs
        for idx, chunk in enumerate(fixed_chunks):
            chunk['chunk_id'] = f"chunk_{idx + 1:03d}"
        
        return fixed_chunks
    
    def clean_page_headers_from_chunks(self, chunks: List[Dict]) -> List[Dict]:
        """Remove page headers that got mixed into chunk content"""
        cleaned_chunks = []
        
        for chunk in chunks:
            cleaned_text = self._clean_text_from_headers(chunk['text'])
            
            # Only keep the chunk if it has substantial content after cleaning
            if len(cleaned_text.strip()) > 50:  # Minimum content threshold
                cleaned_chunk = chunk.copy()
                cleaned_chunk['text'] = cleaned_text
                cleaned_chunk['metadata'] = chunk['metadata'].copy()
                cleaned_chunk['metadata']['char_count'] = len(cleaned_text)
                cleaned_chunks.append(cleaned_chunk)
        
        # Renumber chunk IDs
        for idx, chunk in enumerate(cleaned_chunks):
            chunk['chunk_id'] = f"chunk_{idx + 1:03d}"
        
        return cleaned_chunks
    
    def _clean_text_from_headers(self, text: str) -> str:
        """Remove detected header/footer patterns from within text content"""
        import re
        
        cleaned_text = text
        
        # Remove detected headers/footers if available
        if hasattr(self, '_detected_headers') and self._detected_headers:
            for pattern in self._detected_headers:
                # Try regex pattern first
                if 'regex' in pattern:
                    cleaned_text = re.sub(pattern['regex'], '', cleaned_text, flags=re.IGNORECASE | re.MULTILINE)
                
                # Try exact text match
                pattern_text = pattern['text']
                cleaned_text = re.sub(re.escape(pattern_text), '', cleaned_text, flags=re.IGNORECASE)
                
                # Try variations with different spacing/punctuation
                flexible_pattern = re.escape(pattern_text)
                flexible_pattern = re.sub(r'\\\s+', r'\\s*', flexible_pattern)  # Allow flexible spacing
                cleaned_text = re.sub(flexible_pattern, '', cleaned_text, flags=re.IGNORECASE | re.MULTILINE)
        
        # Remove common generic page number patterns
        cleaned_text = re.sub(r'^\s*\d+\s+of\s+\d+\s*', '', cleaned_text, flags=re.MULTILINE)
        cleaned_text = re.sub(r'^\s*page\s+\d+\s*$', '', cleaned_text, flags=re.IGNORECASE | re.MULTILINE)
        
        # Clean up extra whitespace
        cleaned_text = re.sub(r'\s+', ' ', cleaned_text)
        cleaned_text = cleaned_text.strip()
        
        return cleaned_text
    
    def process_markdown(self) -> List[Dict]:
        """Main processing function"""
        print(f"Processing markdown file: {self.markdown_path}")
        
        with open(self.markdown_path, 'r', encoding='utf-8') as f:
            content = f.read()
        
        print(f"Loaded {len(content)} characters from markdown file")
        
        # First, detect repetitive headers and footers in the document
        self.detect_headers_and_footers(content)
        
        raw_chunks = self.create_chunks_with_context(content)
        print(f"Created {len(raw_chunks)} initial chunks")
        
        # First optimize chunks by merging small adjacent ones with same context
        optimized_chunks = self.optimize_chunks(raw_chunks)
        print(f"Optimized to {len(optimized_chunks)} chunks after merging small fragments")
        
        # Then fix cross-page sentence breaks
        cross_page_fixed = self.fix_cross_page_breaks(optimized_chunks)
        print(f"After cross-page fixes: {len(cross_page_fixed)} chunks")
        
        # Finally clean any remaining page headers from chunk content
        final_chunks = self.clean_page_headers_from_chunks(cross_page_fixed)
        print(f"Final count: {len(final_chunks)} chunks after cleaning page headers")
        
        return final_chunks
    
    def save_chunks_to_json(self, output_path: str) -> None:
        """Save chunks to JSON file with metadata"""
        chunks = self.process_markdown()
        
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
        
        with open(output_path, 'w', encoding='utf-8') as f:
            json.dump(output_data, f, indent=2, ensure_ascii=False)
        
        print(f"Saved {len(chunks)} chunks to {output_path}")
        
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

def main():
    """Main execution function"""
    # Configuration
    source_file="Dentici"
    pdf_file = f"{source_file}.pdf"  # Change this to your PDF file
    markdown_file = f"markdown_{source_file}.md"  # Change this to your markdown file
    output_file = f"chunks_{source_file}.json"

    # Check if files exist
    if not os.path.exists(pdf_file):
        print(f"Error: PDF file '{pdf_file}' not found!")
        return
    
    if not os.path.exists(markdown_file):
        print(f"Error: Markdown file '{markdown_file}' not found!")
        return
    
    try:
        # Create chunker and process
        chunker = MarkdownChunker(pdf_file, markdown_file, max_chunk_size=1024)
        chunker.save_chunks_to_json(output_file)
        
        print(f"\n✅ Successfully created chunks.json with rich metadata!")
        print(f"   📄 Source PDF: {pdf_file}")
        print(f"   📝 Source Markdown: {markdown_file}")
        print(f"   💾 Output: {output_file}")
        
    except Exception as e:
        print(f"❌ Error during processing: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    main()