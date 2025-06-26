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

# Process HTML with model changes - preserve original structure
def process_html_with_model(html_content, model_response):
    """Process HTML with model changes while preserving original structure"""
    try:
        print(f"Processing HTML with model response (length: {len(model_response)})")
        
        # Clean the model response - remove quotes if wrapped
        cleaned_response = model_response.strip()
        if cleaned_response.startswith('"') and cleaned_response.endswith('"'):
            cleaned_response = cleaned_response[1:-1]
        
        # Remove markdown code fences if present
        if cleaned_response.startswith('```html'):
            cleaned_response = cleaned_response[7:]  # Remove ```html
        if cleaned_response.endswith('```'):
            cleaned_response = cleaned_response[:-3]  # Remove ```
        cleaned_response = cleaned_response.strip()
        
        # Check if the model response is already properly formatted HTML
        if ('<!DOCTYPE' in cleaned_response or '<html' in cleaned_response or 
            ('<div' in cleaned_response and '</div>' in cleaned_response)):
            print("Model returned HTML formatted content - using directly")
            
            # Ensure it has proper document structure
            if not cleaned_response.startswith('<!DOCTYPE'):
                # Wrap in proper HTML structure if it's just body content
                if not cleaned_response.startswith('<html'):
                    cleaned_response = f"""<!DOCTYPE html>
<html>
<head>
    <style>
        body {{ font-family: Times New Roman, serif; font-size: 11pt; line-height: 1.5; }}
        .section {{ margin-top: 10px; margin-bottom: 10px; }}
        .signature {{ margin-top: 20px; margin-bottom: 20px; }}
        .heading {{ font-weight: bold; }}
        .indent {{ margin-left: 20px; }}
        .clause {{ margin-top: 10px; margin-bottom: 10px; }}
        .paragraph {{ margin-top: 6px; margin-bottom: 6px; }}
    </style>
</head>
<body>
{cleaned_response}
</body>
</html>"""
            
            print(f"Using model HTML response directly (length: {len(cleaned_response)})")
            return cleaned_response
        
        # If model didn't return HTML, fall back to the original approach
        print("Model returned plain text - converting back to HTML format")
        
        # Parse the original HTML
        soup = BeautifulSoup(html_content, 'html.parser')
        original_text = soup.get_text()
        
        print(f"Original text length: {len(original_text)}")
        print(f"Model response length: {len(cleaned_response)}")
        
        # If the model response is much shorter, something went wrong
        if len(cleaned_response) < len(original_text) * 0.3:
            print("Model response too short, keeping original HTML")
            return html_content
        
        # Simple approach: replace the body content while keeping the structure
        if soup.body:
            # Clear the body and add the model response as formatted content
            soup.body.clear()
            
            # Parse the model response into paragraphs and format appropriately
            paragraphs = cleaned_response.split('\n\n')
            for para in paragraphs:
                para = para.strip()
                if not para:
                    continue
                
                # Create appropriate HTML elements based on content
                if para.upper() == para and len(para.split()) <= 5:
                    # All caps short text = title
                    new_element = soup.new_tag('h1', **{'class': 'heading', 'style': 'text-align: center;'})
                elif para.endswith(':') or (para.isupper() and len(para.split()) <= 8):
                    # Headings
                    new_element = soup.new_tag('h2', **{'class': 'heading'})
                elif any(sig in para.lower() for sig in ['signature', 'signed by', 'dated', 'by:', 'name:', 'title:']):
                    # Signature blocks
                    new_element = soup.new_tag('div', **{'class': 'signature'})
                elif para.startswith(tuple('123456789')) and '.' in para[:10]:
                    # Numbered clauses
                    new_element = soup.new_tag('div', **{'class': 'clause'})
                elif para.startswith('    ') or para.startswith('\t'):
                    # Indented content
                    new_element = soup.new_tag('div', **{'class': 'indent'})
                else:
                    # Regular paragraph
                    new_element = soup.new_tag('div', **{'class': 'paragraph'})
                
                new_element.string = para
                soup.body.append(new_element)
        
        result_html = str(soup)
        print(f"Generated HTML length: {len(result_html)}")
        return result_html
        
    except Exception as e:
        print(f"Error in HTML processing: {str(e)}")
        import traceback
        traceback.print_exc()
        
        # Fallback: return original HTML
        print("Falling back to original HTML due to error")
        return html_content

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