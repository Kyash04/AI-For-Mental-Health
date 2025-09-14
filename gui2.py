import tkinter as tk
from tkinter import scrolledtext, messagebox
import threading
import torch
from textblob import TextBlob
from transformers import AutoTokenizer, AutoModelForCausalLM
import time

class EmotionAnalyzer:
    @staticmethod
    def detect(text):
        analysis = TextBlob(text)
        polarity = analysis.sentiment.polarity
        if polarity > 0.2:
            return "positive"
        elif polarity < -0.2:
            return "negative"
        else:
            return "neutral"

class HealthMateChatApp:
    def __init__(self, root):
        self.root = root
        self.root.title("🧠 HealthMate - Your Personal Friend")
        self.root.geometry("800x600")
        self.root.config(bg="#f5f6fa")

        self.model_loaded = False
        self.model = None
        self.tokenizer = None

        self._setup_widgets()

    def _setup_widgets(self):
        # Chat Display
        self.chat_area = scrolledtext.ScrolledText(self.root, wrap=tk.WORD, state='disabled',
                                                   font=("Consolas", 12), bg="#ffffff", fg="#2f3640")
        self.chat_area.pack(padx=10, pady=10, fill=tk.BOTH, expand=True)

        # Input Field
        self.input_frame = tk.Frame(self.root, bg="#dcdde1")
        self.input_frame.pack(fill=tk.X, padx=10, pady=5)

        self.user_input = tk.Entry(self.input_frame, font=("Segoe UI", 12))
        self.user_input.pack(side=tk.LEFT, fill=tk.X, expand=True, padx=(5, 10))
        self.user_input.bind("<Return>", lambda event: self.send_message())

        self.send_button = tk.Button(self.input_frame, text="Send", command=self.send_message,
                                     bg="#44bd32", fg="white", font=("Segoe UI", 10, "bold"))
        self.send_button.pack(side=tk.RIGHT)

        # Load Model Button
        self.load_model_button = tk.Button(self.root, text="Load AI Model", command=self.load_model,
                                           bg="#273c75", fg="white", font=("Segoe UI", 10, "bold"))
        self.load_model_button.pack(pady=(0, 5))

        # Emotion and Status
        self.emotion_var = tk.StringVar(value="Emotion: Not Detected")
        self.status_var = tk.StringVar(value="Status: Ready")

        self.info_bar = tk.Label(self.root, textvariable=self.emotion_var, anchor=tk.W, bg="#f5f6fa",
                                 font=("Segoe UI", 10))
        self.info_bar.pack(fill=tk.X, padx=10)

        self.status_bar = tk.Label(self.root, textvariable=self.status_var, bd=1, relief=tk.SUNKEN,
                                   anchor=tk.W, font=("Segoe UI", 10), bg="#dcdde1")
        self.status_bar.pack(fill=tk.X, side=tk.BOTTOM)

        self.add_message("System", "👋 Welcome to HealthMate! Click 'Load AI Model' to begin.")

    def add_message(self, sender, message, emotion=None):
        self.chat_area.config(state='normal')
        tag = "bot" if sender == "Bot" else "user"
        formatted = f"{sender} ({emotion})" if emotion and sender == "You" else sender
        self.chat_area.insert(tk.END, f"{formatted}: {message}\n\n", tag)
        self.chat_area.config(state='disabled')
        self.chat_area.yview(tk.END)

    def load_model(self):
        if self.model_loaded:
            self.add_message("System", "✅ Model is already loaded.")
            return

        self.status_var.set("⏳ Loading model...")
        self.load_model_button.config(state='disabled')

        def load():
            try:
                self.add_message("System", "⏱️ Loading TinyLlama model. Please wait...")
                model_id = "TinyLlama/TinyLlama-1.1B-Chat-v1.0"

                self.tokenizer = AutoTokenizer.from_pretrained(model_id, trust_remote_code=True)
                self.model = AutoModelForCausalLM.from_pretrained(
                    model_id,
                    trust_remote_code=True,
                    low_cpu_mem_usage=True,
                    torch_dtype=torch.float32
                )
                self.model_loaded = True
                self.status_var.set("✅ Model Loaded")
                self.add_message("System", "🎉 Model loaded successfully!")
            except Exception as e:
                self.status_var.set(f"❌ Error loading model")
                self.add_message("System", f"Error: {e}")
            finally:
                self.load_model_button.config(state='normal')

        threading.Thread(target=load, daemon=True).start()

    def send_message(self):
        user_msg = self.user_input.get().strip()
        if not user_msg:
            return

        emotion = EmotionAnalyzer.detect(user_msg)
        self.emotion_var.set(f"Emotion: {emotion.capitalize()}")
        self.add_message("You", user_msg, emotion)
        self.user_input.delete(0, tk.END)

        if not self.model_loaded:
            self.add_message("System", "⚠️ Please load the model first.")
            return

        def generate_response():
            try:
                self.status_var.set("💬 Generating response...")
                prompt = (
                    f"<|system|>\nYou are a friendly, supportive mental health assistant. "
                    f"The user's tone is {emotion}. Reply empathetically.\n"
                    f"<|user|>\n{user_msg}\n<|assistant|>\n"
                )

                inputs = self.tokenizer(prompt, return_tensors="pt")
                with torch.no_grad():
                    output = self.model.generate(
                        **inputs,
                        max_new_tokens=150,
                        do_sample=True,
                        top_k=40,
                        top_p=0.95,
                        temperature=0.8,
                        pad_token_id=self.tokenizer.eos_token_id
                    )
                response = self.tokenizer.decode(output[0], skip_special_tokens=True)
# Find the assistant's response after the last occurrence of <|assistant|>
                if "<|assistant|>" in response:
                    bot_reply = response.split("<|assistant|>")[-1].strip()
                else:
                    bot_reply = response.strip()

                self._simulate_typing(bot_reply)
            except Exception as e:
                self.add_message("System", f"Generation Error: {str(e)}")
            finally:
                self.status_var.set("✅ Ready")

        threading.Thread(target=generate_response, daemon=True).start()

    def _simulate_typing(self, text, speed=20):
        def type_out():
            typed = ""
            for char in text:
                typed += char
                self.chat_area.config(state='normal')
                self.chat_area.insert(tk.END, char)
                self.chat_area.config(state='disabled')
                self.chat_area.yview(tk.END)
                time.sleep(1 / speed)
            self.chat_area.insert(tk.END, "\n\n")

        threading.Thread(target=type_out, daemon=True).start()

def main():
    try:
        root = tk.Tk()
        app = HealthMateChatApp(root)
        root.mainloop()
    except Exception as e:
        messagebox.showerror("Fatal Error", f"Something went wrong:\n{e}")

if __name__ == "__main__":
    main()
