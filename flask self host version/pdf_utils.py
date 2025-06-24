import fitz  # PyMuPDF
from reportlab.lib.styles import ParagraphStyle, getSampleStyleSheet
from reportlab.platypus import SimpleDocTemplate, Paragraph, Spacer
from reportlab.lib.pagesizes import letter
from reportlab.lib import colors
from io import BytesIO
import re
import html
import uuid
import os
from datetime import datetime
from bs4 import BeautifulSoup
from xhtml2pdf import pisa
import concurrent.futures

# Extract text from PDF and convert to HTML to preserve formatting
def extract_text_as_html(pdf_path):
    try:
        doc = fitz.open(pdf_path)
        html_output = ['<!DOCTYPE html><html><head><style>',
                      'body { font-family: Times New Roman, serif; font-size: 11pt; line-height: 1.5; }',
                      '.section { margin-top: 10px; margin-bottom: 10px; }',
                      '.signature { margin-top: 20px; margin-bottom: 20px; }',
                      '.heading { font-weight: bold; }',
                      '.indent { margin-left: 20px; }',
                      '.clause { margin-top: 10px; margin-bottom: 10px; }',
                      '.paragraph { margin-top: 6px; margin-bottom: 6px; }',
                      '</style></head><body>']
        
        # Process pages in parallel for faster extraction
        def process_page(page_info):
            page_num, page = page_info
            page_width = page.rect.width
            page_html = []
            
            # Extract text blocks with position information
            blocks = page.get_text("dict")["blocks"]
            
            for block in blocks:
                if "lines" not in block:
                    continue
                    
                # Process each line of text with its position and style information
                for line in block["lines"]:
                    line_text = ""
                    is_centered = False
                    is_bold = False
                    is_heading = False
                    
                    # Calculate line position (centered, left, etc.)
                    line_x0 = min(span["bbox"][0] for span in line["spans"]) if line["spans"] else 0
                    line_x1 = max(span["bbox"][2] for span in line["spans"]) if line["spans"] else 0
                    line_width = line_x1 - line_x0
                    line_center = line_x0 + (line_width / 2)
                    
                    # Check if line appears centered - simplified calculation
                    if abs(line_center - (page_width / 2)) < 50 and line_width < (page_width * 0.7):
                        is_centered = True
                    
                    # Extract text content and detect formatting
                    for span in line["spans"]:
                        span_text = span["text"]
                        line_text += span_text
                        
                        # Check for bold text - simplified check
                        if "bold" in span.get("font", "").lower() or span.get("flags", 0) & 16:
                            is_bold = True
                    
                    # Skip empty lines but add a spacer
                    if not line_text.strip():
                        page_html.append('<div style="height: 12px;"></div>')
                        continue
                    
                    # Fast pattern matching for content types
                    # Check for headings (most common patterns)
                    if (is_bold or line_text.isupper() or 
                        line_text.strip().endswith(':') or 
                        re.match(r'^\d+\.(\d+\.?)?\s+[A-Z]', line_text.strip())):
                        is_heading = True
                    
                    # Check for signature blocks (common patterns)
                    is_signature = False
                    if any(pattern in line_text.lower() for pattern in ['signature', 'signed by', 'dated', 'provider:', 'client:']):
                        is_signature = True
                    
                    # Check for clauses (simple pattern)
                    is_clause = bool(re.match(r'^\d+\.', line_text.strip()) or re.match(r'^[a-z]\)', line_text.strip()))
                    
                    # Add appropriate HTML tags with unique IDs for better targeting
                    element_id = f"elem_{page_num}_{block['number']}_{line['spans'][0]['origin'][1]}"
                    
                    # Use faster string concatenation for HTML
                    if is_signature:
                        page_html.append(f'<div id="{element_id}" class="signature">{html.escape(line_text)}</div>')
                    elif is_heading:
                        if is_centered:
                            page_html.append(f'<h1 id="{element_id}" class="heading" align="center">{html.escape(line_text)}</h1>')
                        else:
                            page_html.append(f'<h2 id="{element_id}" class="heading">{html.escape(line_text)}</h2>')
                    elif is_clause:
                        page_html.append(f'<div id="{element_id}" class="clause">{html.escape(line_text)}</div>')
                    elif line_x0 > 100:  # Indented text
                        page_html.append(f'<div id="{element_id}" class="indent">{html.escape(line_text)}</div>')
                    else:
                        page_html.append(f'<div id="{element_id}" class="paragraph">{html.escape(line_text)}</div>')
            
            return page_html
        
        # Parallel processing of pages
        page_infos = [(i, doc[i]) for i in range(len(doc))]
        
        # Use ThreadPoolExecutor for parallel processing
        max_workers = min(os.cpu_count() or 4, len(page_infos))
        with concurrent.futures.ThreadPoolExecutor(max_workers=max_workers) as executor:
            results = list(executor.map(process_page, page_infos))
        
        # Combine results from all pages
        for page_html in results:
            html_output.extend(page_html)
        
        html_output.append('</body></html>')
        doc.close()
        
        return '\n'.join(html_output)
    except Exception as e:
        print(f"Error extracting HTML from PDF: {str(e)}")
        # Fallback to simple text extraction
        try:
            doc = fitz.open(pdf_path)
            text_content = ""
            for page_num in range(len(doc)):
                text_content += doc[page_num].get_text()
            doc.close()
            
            # Convert plain text to simple HTML
            html_content = f'<!DOCTYPE html><html><body><pre>{html.escape(text_content)}</pre></body></html>'
            return html_content
        except Exception as e2:
            print(f"Fallback extraction also failed: {str(e2)}")
            return ""

# Process HTML with model changes - optimized version
def process_html_with_model(html_content, model_response):
    try:
        # Parse the HTML - faster parsing with lxml
        soup = BeautifulSoup(html_content, 'lxml')
        
        # Extract all text content from the HTML to create a single document
        if soup.body:
            original_document = ""
            text_nodes = []
            
            # Collect all text nodes with their parent elements - faster with direct string search
            for element in soup.body.find_all(string=True):
                if element.strip():  # Only non-empty text nodes
                    original_document += element + "\n"
                    text_nodes.append((element, element.parent))
        else:
            # If no body found, create simple HTML
            return f'<!DOCTYPE html><html><body><pre>{html.escape(model_response)}</pre></body></html>'
        
        # Split the model response and original document into lines for faster processing
        model_lines = model_response.strip().split('\n')
        original_lines = original_document.strip().split('\n')
        
        # Faster mapping approach using hash-based lookup
        replacements = {}
        original_map = {line.strip(): i for i, line in enumerate(original_lines) if line.strip()}
        
        # For each line in the model response, try to find a match in the original
        for model_line in model_lines:
            model_line = model_line.strip()
            if not model_line:
                continue
                
            # Check for exact match first (fastest)
            if model_line in original_map:
                orig_line = original_lines[original_map[model_line]].strip()
                if orig_line != model_line:  # Only add if different
                    replacements[orig_line] = model_line
                continue
            
            # For non-exact matches, use a faster similarity approach
            for orig_line in original_lines:
                orig_line = orig_line.strip()
                if not orig_line:
                    continue
                
                # Quick length check first
                if abs(len(orig_line) - len(model_line)) > min(len(orig_line), len(model_line)) * 0.3:
                    continue
                
                # Check beginning similarity (faster than full comparison)
                prefix_len = min(len(orig_line), len(model_line), 10)
                if orig_line[:prefix_len].lower() == model_line[:prefix_len].lower():
                    # If beginnings match, this is likely a match with changes
                    if orig_line != model_line:  # Only add if different
                        replacements[orig_line] = model_line
                    break
        
        # Apply replacements to the soup - faster with direct replacements
        for text_node, parent in text_nodes:
            text_stripped = text_node.strip()
            if text_stripped in replacements:
                # Replace the text node with the model's version
                text_node.replace_with(replacements[text_stripped])
        
        # Return the modified HTML
        return str(soup)
    except Exception as e:
        print(f"Error processing HTML with model: {str(e)}")
        # Return a simple HTML with the model response
        return f'<!DOCTYPE html><html><body><pre>{html.escape(model_response)}</pre></body></html>'

# Save PDF to a uniquely named file
def save_pdf(pdf_buffer, original_filename):
    """Save PDF buffer to a file with a unique name"""
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    unique_id = str(uuid.uuid4())[:8]
    pdf_filename = f"{os.path.splitext(original_filename)[0]}_{timestamp}_{unique_id}.pdf"
    pdf_path = os.path.join(os.path.dirname(os.path.dirname(__file__)), "generated_pdfs", pdf_filename)
    
    with open(pdf_path, 'wb') as f:
        f.write(pdf_buffer.getvalue())
    print(f"[PDF Generated] Saved to: {pdf_path}")
    return pdf_path

# Generate a fallback PDF when HTML conversion fails
def generate_fallback_pdf(pdf_buffer, text_content, job_id):
    """Generate a simple PDF from plain text when HTML conversion fails"""
    doc = SimpleDocTemplate(
        pdf_buffer,
        pagesize=letter,
        rightMargin=72,
        leftMargin=72,
        topMargin=36,
        bottomMargin=36
    )

    styles = getSampleStyleSheet()
    
    # Create basic styles - simplified for speed
    normal_style = ParagraphStyle(
        'NormalStyle',
        parent=styles['Normal'],
        fontName='Times-Roman',
        fontSize=11,
        leading=16,
        spaceAfter=10
    )
    
    heading_style = ParagraphStyle(
        'HeadingStyle',
        parent=styles['Heading2'],
        fontName='Times-Bold',
        fontSize=12,
        leading=16,
        spaceBefore=12,
        spaceAfter=8
    )
    
    signature_style = ParagraphStyle(
        'SignatureStyle',
        parent=styles['Normal'],
        fontName='Times-Roman',
        fontSize=11,
        leading=14,
        spaceBefore=20,
        spaceAfter=30
    )
    
    # Faster content building
    content = []
    
    # Split into paragraphs and process in batches for speed
    paragraphs = text_content.split('\n\n')
    batch_size = 20
    
    for batch_start in range(0, len(paragraphs), batch_size):
        batch_end = min(batch_start + batch_size, len(paragraphs))
        batch = paragraphs[batch_start:batch_end]
        
        for para in batch:
            if not para.strip():
                # Add space for empty paragraphs
                content.append(Spacer(1, 10))
                continue
                
            para_text = para.strip()
            
            # Fast pattern matching for content types
            is_heading = bool(para_text.isupper() or para_text.endswith(':') or 
                             (len(para_text.split()) <= 8 and para_text.split()[0][0].isupper()))
                
            is_signature = any(pattern in para_text.lower() 
                              for pattern in ['signature', 'signed by', 'dated', 'provider:', 'client:'])
            
            # Apply appropriate style
            if is_signature:
                content.append(Paragraph(para_text, signature_style))
            elif is_heading:
                content.append(Paragraph(para_text, heading_style))
            else:
                # Replace line breaks with <br/> for proper rendering
                formatted_text = para_text.replace('\n', '<br/>')
                content.append(Paragraph(formatted_text, normal_style))
    
    # Build the document
    doc.build(content)
    
    return pdf_buffer 