from langchain.vectorstores import FAISS
from langchain.schema import Document
import re
import numpy as np
from modules.bedrock_integration import BedrockClient

# Initialize Bedrock client
bedrock_client = BedrockClient()

# Process a single chunk sequentially with a simple direct approach
def process_chunk(chunk, instruction, chunk_id, text_generation_pipeline):
    try:
        total_chunks = int(chunk_id.split('/')[1]) if '/' in str(chunk_id) else 1
        current_chunk = int(chunk_id.split('/')[0]) if '/' in str(chunk_id) else chunk_id
        
        print(f"Processing chunk {current_chunk}/{total_chunks}")
        
        # Create Mistral-specific prompt format with [INST] and [/INST] tags
        mistral_prompt = f"<s>[INST] "
        # Add system prompt within the instruction
        mistral_prompt += """You are a precise contract editor that strictly modifies legal documents according to user instructions.

Core requirements:
1. You MUST make ALL changes requested in the user's instructions - this is CRITICAL
2. If there are MULTIPLE instructions (separated by line breaks or numbers), you MUST implement EACH ONE separately
3. Make ONLY the changes specified in the instructions - do not add, remove, or modify anything else
4. NEVER generate or invent new legal language not present in the original document
5. ONLY modify text explicitly mentioned in the user's instructions
6. Maintain exact terminology from the original document - do not substitute or paraphrase legal terms
7. Process each instruction step by step - do not skip any requested change
8. Return the FULL modified text, with ALL the requested changes implemented

IMPORTANT NOTES:
- Pay special attention to company names, addresses, dates, and monetary values
- Look carefully for the specific text mentioned in the instruction and replace it EXACTLY as requested
- If an instruction says to change text from 'X' to 'Y', you MUST find and replace every instance of 'X' with 'Y'
- Instructions often specify entity names with quotes (e.g., from 'ABC Inc.' to 'XYZ Corp.') - these are critical to replace correctly
- If you can't find the exact text mentioned, look for similar text that matches the context
"""
        
        mistral_prompt += f"""You are processing chunk {current_chunk} of {total_chunks} of a document.

Original text for chunk {current_chunk}/{total_chunks}: 
"{chunk}"

---
Instruction to apply to this chunk: {instruction}

The instruction may contain MULTIPLE changes to make. Implement ALL of them that apply to this chunk.
Some instructions may not apply to this specific chunk but to other parts of the document.
Return the FULL modified text for this chunk with ALL applicable changes implemented."""
        mistral_prompt += " [/INST]"
        
        # Use pipeline with optimized generation parameters
        chunk_len = len(chunk)
        max_tokens = min(4000, chunk_len + 500)  # Increased from 2000 to 4000 and added more buffer
        
        chunk_response = text_generation_pipeline(
            mistral_prompt,
            max_new_tokens=max_tokens,
            do_sample=False  # Disable sampling for deterministic generation
        )[0]["generated_text"]
        
        # Debug response
        response_length = len(chunk_response) if chunk_response else 0
        print(f"Completed chunk {current_chunk}/{total_chunks} - Response length: {response_length} chars")
        
        # If response is empty or too short, return the original chunk
        if not chunk_response or len(chunk_response) < 10:
            print(f"WARNING: Empty or very short response for chunk {current_chunk}/{total_chunks}. Using original text.")
            return chunk, False
            
        return chunk_response, True
    except Exception as e:
        print(f"Error processing chunk {chunk_id}: {str(e)}")
        # Return original chunk on error
        return chunk, False

# Process a chunk with change detection
def process_chunk_with_change_detection(chunk, instruction, chunk_id, text_generation_pipeline=None):
    """Process chunk and detect if changes were made using Bedrock"""
    try:
        return process_chunk(chunk, instruction, chunk_id, text_generation_pipeline)
    except Exception as e:
        print(f"Error in change detection for chunk {chunk_id}: {str(e)}")
        # Return original chunk on error, mark as unchanged
        return chunk, False

# Enhanced function to find instruction targets
def find_instruction_targets(instruction, document):
    """Enhanced target identification for better chunk prioritization"""
    targets = []
    
    # Extract quoted text (likely direct references)
    quoted_text = re.findall(r'"([^"]*)"', instruction)
    quoted_text.extend(re.findall(r"'([^']*)'", instruction))
    
    # Process each quoted text segment
    for text in quoted_text:
        if len(text) > 3:
            # Look for case-insensitive matches
            text_lower = text.lower()
            doc_lower = document.lower()
            
            # Try to find all occurrences 
            start_pos = 0
            while start_pos < len(doc_lower):
                found_pos = doc_lower.find(text_lower, start_pos)
                if found_pos == -1:
                    break
                
                # Extract a context window around the match
                context_start = max(0, found_pos - 100)
                context_end = min(len(document), found_pos + len(text) + 200)
                context = document[context_start:context_end]
                targets.append(context)
                
                # Move to next position
                start_pos = found_pos + len(text)
    
    # Split instruction into individual requests (common for multi-part instructions)
    instruction_parts = re.split(r'\n|\d+\.|\s*-\s*', instruction)
    
    # Process each instruction part
    for part in instruction_parts:
        part = part.strip()
        if len(part) < 5:  # Skip very short parts
            continue
            
        # Extract key verbs and subjects that indicate actions
        action_verbs = ["change", "modify", "replace", "update", "set", "remove", "delete", "add", "insert"]
        
        # Look for action verbs followed by potential targets
        for verb in action_verbs:
            if verb in part.lower():
                verb_pos = part.lower().find(verb)
                # Extract the target phrase (text after the verb)
                if verb_pos >= 0 and verb_pos + len(verb) + 1 < len(part):
                    target_phrase = part[verb_pos + len(verb):].strip()
                    if len(target_phrase) > 3:
                        # Find this phrase in the document
                        # Use first few words for searching
                        search_terms = " ".join(target_phrase.split()[:3])
                        if len(search_terms) > 3:
                            # Look for the search terms in document
                            doc_lower = document.lower()
                            search_lower = search_terms.lower()
                            found_pos = doc_lower.find(search_lower)
                            if found_pos >= 0:
                                # Extract context around match
                                context_start = max(0, found_pos - 100)
                                context_end = min(len(document), found_pos + len(search_terms) + 200)
                                context = document[context_start:context_end]
                                targets.append(context)
    
    # Look for common fields in contracts
    common_fields = [
        "effective date", "term", "termination", "governing law", "state", 
        "jurisdiction", "payment terms", "client", "provider", "customer",
        "representative", "fee", "price", "pricing", "deliverable", "party", "parties",
        "section", "article", "clause", "paragraph", "agreement", "contract"
    ]
    
    # Add any common fields mentioned in the instruction
    for field in common_fields:
        if field in instruction.lower():
            # Find this field in the document (could be multiple occurrences)
            field_lower = field.lower()
            doc_lower = document.lower()
            
            # Find all occurrences
            start_pos = 0
            while start_pos < len(doc_lower):
                found_pos = doc_lower.find(field_lower, start_pos)
                if found_pos == -1:
                    break
                
                # Extract a context window around the match
                context_start = max(0, found_pos - 50)
                context_end = min(len(document), found_pos + len(field) + 200)
                context = document[context_start:context_end]
                targets.append(context)
                
                # Move to next position
                start_pos = found_pos + len(field)
    
    # Remove duplicates and near-duplicates
    unique_targets = []
    for target in targets:
        # Check if this target is too similar to existing ones
        is_duplicate = False
        for existing in unique_targets:
            # Simple overlap check
            if target in existing or existing in target:
                is_duplicate = True
                break
        
        if not is_duplicate:
            unique_targets.append(target)
    
    print(f"Found {len(unique_targets)} unique target sections in document")
    return unique_targets

# Enhanced chunk prioritization
def prioritize_chunks(chunks, targets):
    """Enhanced prioritization based on targets and chunk importance"""
    # Score each chunk based on target presence and relevance
    chunk_scores = []
    for i, chunk in enumerate(chunks):
        # Initialize score for this chunk
        score = 0
        chunk_lower = chunk.lower()
        
        # Score based on target presence
        for target in targets:
            target_lower = target.lower()
            
            # If target is fully contained in chunk, add higher score
            if target_lower in chunk_lower:
                score += 5
                
                # Extra points if it's at the beginning or middle of the chunk (more likely to be complete)
                target_pos = chunk_lower.find(target_lower)
                if target_pos < len(chunk) // 3:  # In the first third
                    score += 2
                elif target_pos < len(chunk) * 2 // 3:  # In the middle third
                    score += 1
            
            # If at least half of the target is in the chunk
            elif len(target_lower) > 10:
                # Check for partial matches (first half or second half)
                first_half = target_lower[:len(target_lower)//2]
                second_half = target_lower[len(target_lower)//2:]
                
                if first_half in chunk_lower:
                    score += 2
                if second_half in chunk_lower:
                    score += 2
        
        # Look for section headings/contract structure markers that might indicate important parts
        structure_markers = [
            "section", "article", "clause", "paragraph", 
            "parties", "agreement", "witnesseth", "whereas",
            "term", "termination", "payment", "services", "obligations",
            "governing law", "liability", "indemnification"
        ]
        
        # Add points for structure markers
        for marker in structure_markers:
            if marker in chunk_lower:
                score += 1
        
        # Store score with chunk index
        chunk_scores.append((i, score))
    
    # Sort chunks by score (descending) and original order for equally scored chunks
    chunk_scores.sort(key=lambda x: (-x[1], x[0]))
    
    # Extract the reordering
    prioritized_indices = [i for i, _ in chunk_scores]
    
    # Create debug output to show the prioritization
    print("Chunk prioritization:")
    for idx, (orig_idx, score) in enumerate(chunk_scores):
        print(f"  Position {idx}: Chunk {orig_idx} (score: {score})")
    
    # Reorder chunks based on scores
    prioritized_chunks = [chunks[i] for i in prioritized_indices]
    
    return prioritized_chunks 