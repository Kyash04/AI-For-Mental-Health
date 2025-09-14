import tkinter as tk
from tkinter import scrolledtext, messagebox
import threading
import sys
import os
import torch
from textblob import TextBlob

print("1. App Start")
def analyze_emotion(text):
    analysis = TextBlob(text)
    polarity = analysis.sentiment.polarity

    if polarity > 0.2:
        return "positive"
    elif polarity < -0.2:
        return "negative"
    else:
        return "neutral"

class SimpleChatGUI:
    def __init__(self,root):
        print("2. Init GUI")
        self.root = root
        self.root.title("HealthMate - Your Personal Friend")
        self.root.geometry("700x600")

        #CHAT DISPLAY
        self.chat_area = scrolledtext.ScrolledText(
            root, wrap=tk.WORD, state='disabled',
            width=60, height=20, font=("Helvetica", 12)
        )
        self.chat_area.pack(padx=10,
                            pady=10, fill=tk.BOTH, expand=True)
        print("3. Chat area created")

        #Input Area
        self.input_frame = tk.Frame(root)
        self.input_frame.pack(fill=tk.X, padx=10, pady=5)

        self.user_input = tk.Entry(self.input_frame, font=("Helvetica", 12))
        self.user_input.pack(side=tk.LEFT, fill=tk.X, expand=True)
        self.user_input.bind("<Return>", lambda event: self.send_message())

        #Buttons
        self.send_button = tk.Button(self.input_frame, text="Send", command=self.send_message)

        self.send_button.pack(side=tk.RIGHT, padx=10)

        self.load_model_button = tk.Button(root, text="Load Model", command=self.load_model)
        self.load_model_button.pack(pady=10)

        #Emotion indicator
        self.emotion_var = tk.StringVar()
        self.emotion_var.set("Emotion: Not detected")
        self.emotion_indicator = tk.Label(root, textvariable=self.emotion_var, font=("Helvetica", 10))
        self.emotion_indicator.pack(pady=5)

        #Status Bar
        self.status_var = tk.StringVar()
        self.status_var.set("Ready - Model not loaded")
        self.status_bar = tk.Label(root, textvariable=self.status_var, bd=1, relief=tk.SUNKEN, anchor=tk.W)
        self.status_bar.pack(side=tk.BOTTOM, fill=tk.X)

        #Model loading flag
        self.model_loaded = False
        self.model = None
        self.tokenizer = None

        #Welcome message
        self.add_message("System", "Welcome! Click 'Load Model' to initialize the chatbot.")
        print("4. GUI initialization complete")

    def add_message(self, sender, message, emotion=None):
        self.chat_area.config(state='normal')
        if emotion and sender == "You":
            self.chat_area.insert(tk.END, f"{sender} ({emotion}): {message}\n\n")
        else:
            self.chat_area.insert(tk.END, f"{sender}: {message}\n\n")

        self.chat_area.config(state='disabled')
        self.chat_area.yview(tk.END)

    def load_model(self):
        if self.model_loaded:
            self.add_message("System","Model is already loaded.")
            return
        
        self.status_var.set("Loading model... (this may take several minutes)")
        self.load_model_button.config(state='disabled')

        def loading_thread():
            try:
                self.add_message("System", "Initializing TinyLlama model...")
                print("5. Starting model load")

                from transformers import AutoTokenizer, AutoModelForCausalLM
                

                model_id = "TinyLlama/TinyLlama-1.1B-Chat-v1.0"

                print("6. Loading tokenizer")
                self.tokenizer = AutoTokenizer.from_pretrained(model_id, trust_remote_code=True)
                print("7. Model Loading")
                self.model = AutoModelForCausalLM.from_pretrained(model_id, trust_remote_code=True, low_cpu_mem_usage=True,
                                                                  torch_dtype=torch.float32)
                print("8. Model loaded successfully")

                self.model_loaded = True
                self.root.after(0, lambda: self.status_var.set("Ready - Model loaded"))
                self.root.after(0, lambda: self.add_message("System", "Model loaded successfully!"))

            except Exception as e:
                print(f"Error loading model: {e}")
                self.root.after(0, lambda: self.status_var.set(f"Error: {str(e)}"))
                self.root.after(0, lambda: self.add_message("Error", f"Failed to load model: {str(e)}"))

            finally:
                self.root.after(0,lambda: self.load_model_button.config(state='normal'))

        thread = threading.Thread(target=loading_thread, daemon=True)
        thread.start()

    def send_message(self):
        user_msg = self.user_input.get().strip()
        if not user_msg:
            return

        #Analyze emotions in the message
        emotion = analyze_emotion(user_msg)
        #Update emotion indicator
        self.emotion_var.set(f"Emotion: {emotion.capitalize()}")

        # Add message
        self.add_message("You", user_msg, emotion)
        self.user_input.delete(0, tk.END)

        if not self.model_loaded:
            self.add_message("System", "Please load the model first.")
            return
        
        #Process in thread
        def response_thread():
            try:
                self.status_var.set("Generating response...")
                #Format prompt for TinyLlama
                prompt = (
                f"<|system|>\n"
                f"You are HealthMate, a friendly and supportive mental health assistant. "
                f"Your goal is to provide empathetic responses that help users feel heard and understood. "
                f"The user's message has been analyzed as having a {emotion} tone. "
                f"Respond in a way that acknowledges their emotional state appropriately. "
                f"Keep responses helpful and supportive, but relatively brief (2-3 paragraphs maximum).\n"
                f"<|user|>\n{user_msg}\n<|assistant|>\n"
            )

                #Generate response
                inputs = self.tokenizer(prompt, return_tensors = "pt")

                with torch.no_grad():
                    output = self.model.generate( **inputs, max_new_tokens = 200,
                                                 do_sample = True,
                                                 top_k = 50,
                                                 top_p=0.95,
                                                 temperature = 0.7,
                                                 pad_token_id=self.tokenizer.eos_token_id)
                
                response = self.tokenizer.decode(output[0], skip_special_tokens = True)
                cleaned_response = response.split("<|assistant|>\n")[-1].strip()

                self.root.after(0, lambda: self.add_message("Bot", cleaned_response))
                self.root.after(0, lambda: self.status_var.set("Ready"))


            except Exception as e:
                print(f"Error: {e}")
                self.root.after(0, lambda: self.add_message("Error", f"Failed to generate response: {str(e)}"))
                self.root.after(0, lambda: self.status_var.set("Error"))

        thread = threading.Thread(target=response_thread, daemon=True)
        thread.start()

#Main program
def main():
    print("9. Starting main application")
    try:
        root = tk.Tk()
        app = SimpleChatGUI(root)
        print("10. Starting mainloop")
        root.mainloop()
        print("11. Mainloop ended")

    except Exception as e:
        print(f"Critical error: {e}")
        messagebox.showerror("Critical Error", f"Application crashed: {str(e)}")

if __name__ == "__main__":
    main()