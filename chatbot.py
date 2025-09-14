from transformers import AutoTokenizer, AutoModelForCausalLM
import torch
import os
import time

# Add debug statements
print("Chatbot module imported!")

def initialize_model():
    print("Model initialization requested...")
    model_id = "TinyLlama/TinyLlama-1.1B-Chat-v1.0"
    
    # Force CPU usage for integrated graphics
    os.environ["CUDA_VISIBLE_DEVICES"] = "-1"
    device = "cpu"
    print(f"Using device: {device}")
    
    # Set low memory usage parameters
    try:
        print("Loading tokenizer...")
        tokenizer = AutoTokenizer.from_pretrained(
            model_id,
            trust_remote_code=True
        )
        print("Tokenizer loaded successfully!")
        
        print("Loading model (this may take several minutes on first run)...")
        start_time = time.time()
        model = AutoModelForCausalLM.from_pretrained(
            model_id,
            trust_remote_code=True,
            low_cpu_mem_usage=True,
            torch_dtype=torch.float32
        )
        end_time = time.time()
        print(f"Model loaded successfully in {end_time - start_time:.2f} seconds!")
        
        # Force padding side to avoid warning
        tokenizer.padding_side = "left"
        
        return model, tokenizer, device
    except Exception as e:
        print(f"Error initializing model: {e}")
        raise

# Lazy loading - only initialize when first needed
model = None
tokenizer = None
device = None

def get_model():
    global model, tokenizer, device
    if model is None:
        print("First model request, initializing...")
        model, tokenizer, device = initialize_model()
    else:
        print("Model already loaded, reusing...")
    return model, tokenizer, device

# Get response function
def get_response(user_input):
    print(f"Request received: {user_input}")
    try:
        # Get or initialize model
        model, tokenizer, device = get_model()
        
        # Format the prompt according to TinyLlama's expected format
        prompt = f"[INST] {user_input} [/INST]"
        
        # Tokenize input
        print("Tokenizing input...")
        inputs = tokenizer(prompt, return_tensors="pt")
        
        # Generate the response with conservative parameters
        print("Generating response...")
        start_time = time.time()
        with torch.no_grad():
            output = model.generate(
                **inputs,
                max_new_tokens=100,  # Reduced from 150
                do_sample=True,
                top_k=50,
                top_p=0.95,
                temperature=0.7,
                pad_token_id=tokenizer.eos_token_id
            )
        end_time = time.time()
        print(f"Response generated in {end_time - start_time:.2f} seconds")
        
        # Decode the response
        response = tokenizer.decode(output[0], skip_special_tokens=True)
        
        # Remove prompt from the response
        cleaned_response = response.replace(prompt, "").strip()
        
        # Return empty string if no response generated
        if not cleaned_response:
            print("No response generated")
            return "I'm sorry, I couldn't generate a response. Please try again."
            
        print("Response ready to return")
        return cleaned_response
    except Exception as e:
        print(f"Error generating response: {e}")
        return f"Sorry, I encountered an error while processing your request: {str(e)}"

# Add a quick test function to check if the module works standalone
if __name__ == "_main_":
    print("Testing chatbot module standalone...")
    response = get_response("Hello, how are you?")
    print(f"Test response: {response}")
    print("Test complete!")