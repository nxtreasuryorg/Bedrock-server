import torch
from transformers import AutoTokenizer, AutoModelForCausalLM, pipeline
from langchain.embeddings import HuggingFaceEmbeddings
from accelerate import Accelerator
import os

# Initialize ML components
def initialize_models(base_model_id, embed_id):
    """Initialize the LLM and embedding models"""
    print(f"CUDA available: {torch.cuda.is_available()}")
    print(f"GPU count: {torch.cuda.device_count()}")
    for i in range(torch.cuda.device_count()):
        print(f"GPU {i}: {torch.cuda.get_device_name(i)}")

    # Force GPU usage
    device = "cuda"
    torch.cuda.empty_cache()
    accelerator = Accelerator()

    print("Loading models and initializing components...")
    
    # Initialize embeddings - HuggingFaceEmbeddings handles device internally
    embedding = HuggingFaceEmbeddings(
        model_name=embed_id,
        encode_kwargs={'device': 'cuda:0'}
    )

    # Initialize tokenizer with optimizations
    tokenizer = AutoTokenizer.from_pretrained(
        base_model_id, 
        use_fast=True,  # Using fast tokenizer for Mistral
        trust_remote_code=True,
        model_max_length=32000  # Increased from 4096 to 32000 for larger context
    )

    # Load model with optimized configuration
    base_model = AutoModelForCausalLM.from_pretrained(
        base_model_id,  
        device_map="auto",  # Automatically distributes across available GPUs
        trust_remote_code=True,
        torch_dtype=torch.float16,  # Use FP16 for faster inference
    )

    # Move model to the accelerator
    base_model = accelerator.prepare(base_model)

    # Initialize pipeline with optimized GPU acceleration
    text_generation_pipeline = pipeline(
        "text-generation",
        model=base_model,
        tokenizer=tokenizer,
        max_new_tokens=4000,  # Increased from 2000 to 4000 for longer outputs
        return_full_text=False,
        # Optimized generation parameters
        do_sample=False,  # Disable sampling for deterministic, faster generation
        num_beams=1,      # Disable beam search for faster generation
        pad_token_id=tokenizer.eos_token_id,
        config={
            "temperature": 0.4,  # Lower temperature for faster, more deterministic results
            'top_k': 1,          # Faster with top_k=1
            'top_p': 0.7,        # Lower but still reasonable for legal text
            'repetition_penalty': 2.0  # Lower repetition penalty
        },
        device_map="auto"  # Distribute across available GPUs
    )

    print("All components initialized successfully!")
    
    return embedding, text_generation_pipeline 