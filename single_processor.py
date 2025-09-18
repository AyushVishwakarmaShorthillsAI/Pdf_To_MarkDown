import os
import re
import sys
from collections import defaultdict
from docling.document_converter import DocumentConverter
from docling.document_converter import DocumentConverter

def _manual_resolve_cref(doc, cref_string):
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

def build_markdown_grouped_by_page(doc):
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
        return _build_intelligent_page_markdown(doc)
    
    # Assemble full markdown with accurate page headers
    full_markdown_parts = []
    for page_num in sorted(pages_content.keys()):
        full_markdown_parts.append(f"\n## [Page {page_num}]\n\n")
        full_markdown_parts.append("\n\n".join(pages_content[page_num]))

    return "".join(full_markdown_parts)

def _build_intelligent_page_markdown(doc):
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

def _estimate_page_count(doc, markdown_content):
    """
    Estimate the number of pages in the document using various heuristics.
    """
    # Try to get from document metadata first
    if hasattr(doc, 'pages') and doc.pages:
        return len(doc.pages)
    
    # Estimate based on content length (very rough)
    # Average academic paper page ~ 3000-4000 characters
    estimated_pages = max(1, len(markdown_content) // 3500)
    
    # Cap at reasonable bounds
    return min(max(estimated_pages, 1), 50)

def process_markdown_formatting(markdown_text):
    """
    Applies multiple formatting rules to the raw Markdown text:
    1. Converts plain text table titles into Level 2 (##) headings.
    2. Demotes existing decimal-numbered Level 2 (##) subheadings to Level 3 (###).

    Args:
        markdown_text (str): The original Markdown content.

    Returns:
        str: The fully processed Markdown content.
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

# --- Your Script Logic ---

# --- CONFIGURATION ---
# Set your input PDF filename here.
source_pdf = sys.argv[1] if len(sys.argv) > 1 else "Zhang.pdf" 
# --- END CONFIGURATION ---

# Check if the source file exists before starting
if not os.path.exists(source_pdf):
    print(f"Error: The source file was not found at '{source_pdf}'", file=sys.stderr)
    print("Please make sure the `source_pdf` variable is set correctly.", file=sys.stderr)
    sys.exit(1)

# Automatically create a descriptive output filename
input_filename_base = os.path.splitext(os.path.basename(source_pdf))[0]
output_filename = f"markdown_{input_filename_base}.md"

try:
    # Step 1: Convert the PDF to a Docling document model
    print(f"Converting '{source_pdf}' to Docling document model...")
    converter = DocumentConverter()
    result = converter.convert(source_pdf)
    print("Conversion complete.")

    # Step 2: Build Markdown grouped by page numbers
    print("Building Markdown grouped by page numbers...")
    grouped_markdown = build_markdown_grouped_by_page(result.document)

    # Step 3: Apply custom formatting rules (tables and subheadings)
    print("Applying custom formatting for tables and subheadings...")
    processed_markdown = process_markdown_formatting(grouped_markdown)
    print("Formatting complete.")

    # Step 4: Save the fully processed Markdown content to a file
    with open(output_filename, 'w', encoding='utf-8') as f:
        f.write(processed_markdown)

    print(f"\nSuccess! Fully processed Markdown saved to '{output_filename}'")

except Exception as e:
    print(f"\nAn error occurred during the process: {e}", file=sys.stderr)
    sys.exit(1)