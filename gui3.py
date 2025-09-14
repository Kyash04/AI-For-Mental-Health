import tkinter as tk
from tkinter import scrolledtext, messagebox, ttk, filedialog
import threading
import torch
import json
import os
from datetime import datetime
import numpy as np
from textblob import TextBlob
from transformers import AutoTokenizer, AutoModelForCausalLM, pipeline
import time

class EmotionAnalyzer:
    """Advanced emotion analysis with multiple detection methods"""
    
    def __init__(self):
        self.sentiment_analyzer = None
        
    def initialize(self):
        """Initialize the sentiment analyzer if not already loaded"""
        if self.sentiment_analyzer is None:
            try:
                self.sentiment_analyzer = pipeline("sentiment-analysis")
                return True
            except Exception as e:
                print(f"Error initializing sentiment analyzer: {e}")
                return False
        return True
                
    def detect(self, text):
        """Detect emotion in text using multiple methods"""
        # Basic TextBlob sentiment analysis as fallback
        blob_analysis = TextBlob(text)
        polarity = blob_analysis.sentiment.polarity
        
        # Try to use the transformers pipeline if available
        emotion = None
        confidence = 0
        
        if self.sentiment_analyzer is not None:
            try:
                result = self.sentiment_analyzer(text)
                if result and len(result) > 0:
                    emotion = result[0]['label'].lower()
                    confidence = result[0]['score']
            except Exception:
                # Fall back to TextBlob if transformer fails
                pass
                
        # Use TextBlob if transformer didn't work
        if emotion is None:
            if polarity > 0.3:
                emotion = "positive"
            elif polarity < -0.3:
                emotion = "negative"
            elif polarity > 0.1:
                emotion = "slightly positive"
            elif polarity < -0.1:
                emotion = "slightly negative"
            else:
                emotion = "neutral"
            confidence = abs(polarity)
            
        return {
            "emotion": emotion,
            "confidence": confidence,
            "polarity": polarity
        }

class ThemeManager:
    """Manages application themes"""
    
    THEMES = {
        "Light": {
            "bg": "#f5f6fa",
            "chat_bg": "#ffffff",
            "chat_fg": "#2f3640",
            "input_bg": "#dcdde1",
            "button_bg": "#44bd32",
            "button_fg": "white",
            "accent": "#273c75",
            "status_bg": "#dcdde1",
            "user_msg": "#e1f5fe",
            "bot_msg": "#f1f8e9",
            "system_msg": "#fff3e0"
        },
        "Dark": {
            "bg": "#2f3640",
            "chat_bg": "#353b48",
            "chat_fg": "#f5f6fa",
            "input_bg": "#1e272e",
            "button_bg": "#44bd32",
            "button_fg": "white",
            "accent": "#0097e6",
            "status_bg": "#1e272e",
            "user_msg": "#01579b",
            "bot_msg": "#1b5e20",
            "system_msg": "#4e342e"
        },
        "Calm": {
            "bg": "#e0f7fa",
            "chat_bg": "#f5f5f5",
            "chat_fg": "#37474f",
            "input_bg": "#b2ebf2",
            "button_bg": "#26a69a",
            "button_fg": "white",
            "accent": "#00897b",
            "status_bg": "#b2ebf2",
            "user_msg": "#bbdefb",
            "bot_msg": "#c8e6c9",
            "system_msg": "#ffecb3"
        }
    }
    
    @staticmethod
    def get_theme(name):
        """Get theme colors by name"""
        return ThemeManager.THEMES.get(name, ThemeManager.THEMES["Light"])
        
    @staticmethod
    def get_theme_names():
        """Get available theme names"""
        return list(ThemeManager.THEMES.keys())


class ChatHistory:
    """Manages chat history and persistence"""
    
    def __init__(self, app_name="HealthMate"):
        self.app_name = app_name
        self.history = []
        self.history_dir = os.path.join(os.path.expanduser("~"), f".{app_name.lower()}")
        
        # Create history directory if it doesn't exist
        if not os.path.exists(self.history_dir):
            try:
                os.makedirs(self.history_dir)
            except Exception as e:
                print(f"Failed to create history directory: {e}")
        
    def add_message(self, sender, message, emotion=None):
        """Add a message to history"""
        self.history.append({
            "sender": sender,
            "message": message,
            "emotion": emotion,
            "timestamp": datetime.now().isoformat()
        })
        
    def save_to_file(self, filename=None):
        """Save history to a JSON file"""
        if not filename:
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            filename = os.path.join(self.history_dir, f"chat_{timestamp}.json")
        
        try:
            with open(filename, 'w', encoding='utf-8') as f:
                json.dump({
                    "app": self.app_name,
                    "created": datetime.now().isoformat(),
                    "messages": self.history
                }, f, indent=2)
            return filename
        except Exception as e:
            print(f"Failed to save history: {e}")
            return None
            
    def load_from_file(self, filename):
        """Load history from a JSON file"""
        try:
            with open(filename, 'r', encoding='utf-8') as f:
                data = json.load(f)
                if "messages" in data:
                    self.history = data["messages"]
                    return True
        except Exception as e:
            print(f"Failed to load history: {e}")
        return False
        
    def clear(self):
        """Clear the history"""
        self.history = []


class HealthMateChatApp:
    def __init__(self, root):
        self.root = root
        self.root.title("🧠 HealthMate - Your Mental Health Assistant")
        self.root.geometry("900x700")
        
        # Initialize components
        self.emotion_analyzer = EmotionAnalyzer()
        self.chat_history = ChatHistory()
        self.current_theme = "Light"
        
        # Model variables
        self.model_loaded = False
        self.model = None
        self.tokenizer = None
        self.model_loading = False
        
        # UI setup
        self._setup_menu()
        self._setup_widgets()
        self._apply_theme(self.current_theme)
        
        # Start with a welcome message
        self._show_welcome_message()

    def _setup_menu(self):
        """Set up the application menu"""
        menubar = tk.Menu(self.root)
        
        # File menu
        file_menu = tk.Menu(menubar, tearoff=0)
        file_menu.add_command(label="Save Conversation", command=self._save_conversation)
        file_menu.add_command(label="Load Conversation", command=self._load_conversation)
        file_menu.add_separator()
        file_menu.add_command(label="Clear Chat", command=self._clear_chat)
        file_menu.add_separator()
        file_menu.add_command(label="Exit", command=self.root.quit)
        menubar.add_cascade(label="File", menu=file_menu)
        
        # Model menu
        model_menu = tk.Menu(menubar, tearoff=0)
        model_menu.add_command(label="Load AI Model", command=self.load_model)
        model_menu.add_command(label="Init Emotion Analyzer", 
                              command=lambda: threading.Thread(target=self._init_emotion_analyzer).start())
        menubar.add_cascade(label="Model", menu=model_menu)
        
        # Theme menu
        theme_menu = tk.Menu(menubar, tearoff=0)
        self.theme_var = tk.StringVar(value=self.current_theme)
        for theme in ThemeManager.get_theme_names():
            theme_menu.add_radiobutton(label=theme, variable=self.theme_var, 
                                      value=theme, command=lambda t=theme: self._change_theme(t))
        menubar.add_cascade(label="Theme", menu=theme_menu)
        
        # Help menu
        help_menu = tk.Menu(menubar, tearoff=0)
        help_menu.add_command(label="About", command=self._show_about)
        menubar.add_cascade(label="Help", menu=help_menu)
        
        self.root.config(menu=menubar)
    
    def _setup_widgets(self):
        """Set up the main UI widgets"""
        # Main frame
        self.main_frame = tk.Frame(self.root)
        self.main_frame.pack(fill=tk.BOTH, expand=True, padx=10, pady=10)
        
        # Upper frame for chat
        self.chat_frame = tk.Frame(self.main_frame)
        self.chat_frame.pack(fill=tk.BOTH, expand=True)
        
        # Chat Display with custom tag configuration
        self.chat_area = scrolledtext.ScrolledText(
            self.chat_frame, wrap=tk.WORD, state='disabled',
            font=("Segoe UI", 11), cursor="arrow"
        )
        self.chat_area.pack(fill=tk.BOTH, expand=True, padx=5, pady=5)
        
        # Configure tags for different message types
        self.chat_area.tag_configure("user", lmargin1=10, lmargin2=20, rmargin=10)
        self.chat_area.tag_configure("bot", lmargin1=10, lmargin2=20, rmargin=10)
        self.chat_area.tag_configure("system", lmargin1=10, lmargin2=20, rmargin=10, font=("Segoe UI", 10, "italic"))
        
        # Lower frame for input
        self.input_frame = tk.Frame(self.main_frame)
        self.input_frame.pack(fill=tk.X, pady=10)
        
        # Progress bar for model loading
        self.progress_var = tk.DoubleVar()
        self.progress_bar = ttk.Progressbar(
            self.input_frame, orient="horizontal", 
            length=200, mode="indeterminate", variable=self.progress_var
        )
        
        # Input field with placeholder
        self.input_frame_inner = tk.Frame(self.input_frame)
        self.input_frame_inner.pack(fill=tk.X)
        
        self.user_input = tk.Text(self.input_frame_inner, font=("Segoe UI", 11), height=3)
        self.user_input.pack(side=tk.LEFT, fill=tk.X, expand=True, padx=(5, 10))
        self.user_input.bind("<Return>", self._handle_enter)
        self.user_input.bind("<Shift-Return>", lambda e: None)  # Allow multi-line with Shift+Enter
        
        # Right button frame
        self.button_frame = tk.Frame(self.input_frame_inner)
        self.button_frame.pack(side=tk.RIGHT, fill=tk.Y)
        
        # Send button
        self.send_button = tk.Button(
            self.button_frame, text="Send", command=self.send_message,
            font=("Segoe UI", 10, "bold"), width=10, height=2
        )
        self.send_button.pack(side=tk.TOP, padx=5, pady=2)
        
        # Load model button
        self.load_model_button = tk.Button(
            self.button_frame, text="Load AI", command=self.load_model,
            font=("Segoe UI", 10), width=10
        )
        self.load_model_button.pack(side=tk.BOTTOM, padx=5, pady=2)
        
        # Status frame
        self.status_frame = tk.Frame(self.root)
        self.status_frame.pack(fill=tk.X, side=tk.BOTTOM)
        
        # Emotion and model status
        self.status_frame_upper = tk.Frame(self.status_frame)
        self.status_frame_upper.pack(fill=tk.X)
        
        self.emotion_var = tk.StringVar(value="Emotion: Not detected")
        self.emotion_label = tk.Label(
            self.status_frame_upper, textvariable=self.emotion_var,
            anchor=tk.W, font=("Segoe UI", 10)
        )
        self.emotion_label.pack(side=tk.LEFT, padx=10)
        
        self.model_status_var = tk.StringVar(value="Model: Not loaded")
        self.model_status_label = tk.Label(
            self.status_frame_upper, textvariable=self.model_status_var,
            anchor=tk.E, font=("Segoe UI", 10)
        )
        self.model_status_label.pack(side=tk.RIGHT, padx=10)
        
        # Main status bar
        self.status_var = tk.StringVar(value="Status: Ready")
        self.status_bar = tk.Label(
            self.status_frame, textvariable=self.status_var,
            bd=1, relief=tk.SUNKEN, anchor=tk.W, font=("Segoe UI", 10)
        )
        self.status_bar.pack(fill=tk.X)
    
    def _apply_theme(self, theme_name):
        """Apply the selected theme to all widgets"""
        theme = ThemeManager.get_theme(theme_name)
        self.current_theme = theme_name
        
        # Main window and frames
        self.root.config(bg=theme["bg"])
        self.main_frame.config(bg=theme["bg"])
        self.chat_frame.config(bg=theme["bg"])
        self.input_frame.config(bg=theme["bg"])
        self.input_frame_inner.config(bg=theme["bg"])
        self.button_frame.config(bg=theme["bg"])
        self.status_frame.config(bg=theme["bg"])
        self.status_frame_upper.config(bg=theme["bg"])
        
        # Chat area
        self.chat_area.config(bg=theme["chat_bg"], fg=theme["chat_fg"])
        self.chat_area.tag_configure("user", background=theme["user_msg"])
        self.chat_area.tag_configure("bot", background=theme["bot_msg"])
        self.chat_area.tag_configure("system", background=theme["system_msg"])
        
        # Input and buttons
        self.user_input.config(bg=theme["input_bg"], fg=theme["chat_fg"])
        self.send_button.config(bg=theme["button_bg"], fg=theme["button_fg"])
        self.load_model_button.config(bg=theme["accent"], fg=theme["button_fg"])
        
        # Status elements
        self.emotion_label.config(bg=theme["bg"], fg=theme["chat_fg"])
        self.model_status_label.config(bg=theme["bg"], fg=theme["chat_fg"])
        self.status_bar.config(bg=theme["status_bg"], fg=theme["chat_fg"])
    
    def _change_theme(self, theme_name):
        """Change the current theme"""
        self._apply_theme(theme_name)
    
    def _show_welcome_message(self):
        """Show welcome message at startup"""
        welcome_msg = (
            "👋 Welcome to HealthMate!\n\n"
            "I'm designed to provide mental health support and friendly conversation. "
            "To get started:\n"
            "1. Click 'Load AI' to initialize the AI assistant model\n"
            "2. Optionally initialize the emotion analyzer for better responses\n"
            "3. Type your message and press Enter or click Send\n\n"
            "You can change themes in the Theme menu and save your conversations from the File menu."
        )
        self.add_message("HealthMate", welcome_msg)
    
    def _show_about(self):
        """Show about dialog"""
        messagebox.showinfo(
            "About HealthMate",
            "HealthMate - Your Mental Health Assistant\n\n"
            "A supportive AI-powered chat application for mental health support.\n\n"
            "Features:\n"
            "- AI-powered conversations\n"
            "- Emotion detection\n"
            "- Multiple themes\n"
            "- Conversation history\n\n"
            "This application uses TinyLlama for AI responses."
        )
    
    def _save_conversation(self):
        """Save the current conversation to a file"""
        filename = filedialog.asksaveasfilename(
            defaultextension=".json",
            filetypes=[("JSON files", "*.json"), ("All files", "*.*")],
            title="Save Conversation"
        )
        
        if filename:
            saved_file = self.chat_history.save_to_file(filename)
            if saved_file:
                self.add_message("System", f"Conversation saved to {os.path.basename(saved_file)}")
            else:
                self.add_message("System", "Failed to save conversation")
    
    def _load_conversation(self):
        """Load a conversation from a file"""
        filename = filedialog.askopenfilename(
            filetypes=[("JSON files", "*.json"), ("All files", "*.*")],
            title="Load Conversation"
        )
        
        if filename:
            if self.chat_history.load_from_file(filename):
                # Clear current chat
                self.chat_area.config(state='normal')
                self.chat_area.delete(1.0, tk.END)
                self.chat_area.config(state='disabled')
                
                # Reload messages
                for msg in self.chat_history.history:
                    self.add_message(
                        msg["sender"], 
                        msg["message"], 
                        msg.get("emotion")
                    )
                
                self.add_message("System", f"Loaded conversation from {os.path.basename(filename)}")
            else:
                self.add_message("System", "Failed to load conversation")
    
    def _clear_chat(self):
        """Clear the chat area and history"""
        if messagebox.askyesno("Clear Chat", "Are you sure you want to clear the chat history?"):
            self.chat_area.config(state='normal')
            self.chat_area.delete(1.0, tk.END)
            self.chat_area.config(state='disabled')
            self.chat_history.clear()
            self._show_welcome_message()
    
    def _init_emotion_analyzer(self):
        """Initialize the emotion analyzer"""
        self.status_var.set("⏳ Initializing emotion analyzer...")
        if self.emotion_analyzer.initialize():
            self.add_message("System", "✅ Emotion analyzer initialized successfully")
            self.status_var.set("✅ Emotion analyzer ready")
        else:
            self.add_message("System", "⚠️ Failed to initialize emotion analyzer. Using basic analysis.")
            self.status_var.set("⚠️ Using basic emotion analysis")
    
    def _handle_enter(self, event):
        """Handle Enter key press in the input field"""
        if not event.state & 0x1:  # Check if Shift is not pressed
            self.send_message()
            return "break"  # Prevent default behavior
    
    def add_message(self, sender, message, emotion=None):
        """Add a message to the chat area"""
        self.chat_area.config(state='normal')
        
        # Determine the tag to use
        if sender == "You":
            tag = "user"
        elif sender in ["HealthMate", "Bot"]:
            tag = "bot"
        else:
            tag = "system"
        
        # Format the message with sender and emotion if available
        if emotion and sender == "You":
            header = f"{sender} ({emotion}): "
        else:
            header = f"{sender}: "
        
        # Insert the message
        self.chat_area.insert(tk.END, header, tag)
        self.chat_area.insert(tk.END, f"{message}\n\n", tag)
        
        self.chat_area.config(state='disabled')
        self.chat_area.see(tk.END)
        
        # Add to history if not a system message
        if sender not in ["System"]:
            self.chat_history.add_message(sender, message, emotion)
    
    def send_message(self):
        """Process and send the user's message"""
        # Get message from input field
        user_msg = self.user_input.get("1.0", "end-1c").strip()
        if not user_msg:
            return
        
        # Clear input field
        self.user_input.delete("1.0", tk.END)
        
        # Analyze emotion
        emotion_result = self.emotion_analyzer.detect(user_msg)
        emotion = emotion_result["emotion"]
        confidence = emotion_result["confidence"]
        
        # Update emotion display
        self.emotion_var.set(f"Emotion: {emotion.capitalize()} ({confidence:.2f})")
        
        # Add message to chat
        self.add_message("You", user_msg, emotion)
        
        # Check if model is loaded
        if not self.model_loaded:
            self.add_message("System", "⚠️ Please load the AI model first to get responses.")
            return
        
        # Generate response in a separate thread
        thread = threading.Thread(target=self._generate_response, args=(user_msg, emotion), daemon=True)
        thread.start()
    
    def load_model(self):
        """Load the AI model"""
        if self.model_loaded:
            self.add_message("System", "✅ Model is already loaded.")
            return
        
        if self.model_loading:
            self.add_message("System", "⏳ Model is already being loaded. Please wait...")
            return
        
        # Set loading state
        self.model_loading = True
        self.status_var.set("⏳ Loading model...")
        self.load_model_button.config(state='disabled')
        self.model_status_var.set("Model: Loading...")
        
        # Show progress bar
        self.progress_bar.pack(fill=tk.X, padx=5, pady=5)
        self.progress_bar.start(10)
        
        # Create a separate thread for loading
        thread = threading.Thread(target=self._load_model_thread, daemon=True)
        thread.start()
    
    def _load_model_thread(self):
        """Thread function for loading the model"""
        try:
            self.add_message("System", "⏱️ Loading TinyLlama model. This may take a few moments...")
            
            model_id = "TinyLlama/TinyLlama-1.1B-Chat-v1.0"
            
            self.tokenizer = AutoTokenizer.from_pretrained(model_id, trust_remote_code=True)
            self.model = AutoModelForCausalLM.from_pretrained(
                model_id,
                trust_remote_code=True,
                low_cpu_mem_usage=True,
                torch_dtype=torch.float32
            )
            
            self.model_loaded = True
            self.status_var.set("✅ Model loaded successfully")
            self.model_status_var.set("Model: Loaded")
            self.add_message("System", "🎉 Model loaded successfully! You can now chat with me.")
            
        except Exception as e:
            self.status_var.set(f"❌ Error loading model")
            self.model_status_var.set("Model: Error")
            error_msg = str(e)
            # Create a more user-friendly error message
            if "CUDA" in error_msg:
                error_msg = "GPU memory error. Try running on CPU instead."
            elif "disk space" in error_msg.lower():
                error_msg = "Not enough disk space to download the model."
            elif "connection" in error_msg.lower():
                error_msg = "Network connection error. Check your internet connection."
            
            self.add_message("System", f"Error loading model: {error_msg}")
        
        finally:
            # Reset loading state
            self.model_loading = False
            self.load_model_button.config(state='normal')
            self.progress_bar.stop()
            self.progress_bar.pack_forget()
    
    def _generate_response(self, user_msg, emotion):
        """Generate a response to the user's message"""
        try:
            self.status_var.set("💬 Generating response...")
            
            # Create a prompt with the emotion context
            prompt = (
                f"<|system|>\n"
                f"You are HealthMate, a friendly and supportive mental health assistant. "
                f"Your goal is to provide empathetic responses that help users feel heard and understood. "
                f"The user's message has been analyzed as having a {emotion} tone. "
                f"Respond in a way that acknowledges their emotional state appropriately. "
                f"Keep responses helpful and supportive, but relatively brief (2-3 paragraphs maximum).\n"
                f"<|user|>\n{user_msg}\n<|assistant|>\n"
            )
            
            # Generate response
            inputs = self.tokenizer(prompt, return_tensors="pt")
            with torch.no_grad():
                output = self.model.generate(
                    **inputs,
                    max_new_tokens=200,
                    do_sample=True,
                    top_k=50,
                    top_p=0.9,
                    temperature=0.7,
                    pad_token_id=self.tokenizer.eos_token_id,
                    repetition_penalty=1.2
                )
            
            # Extract generated text
            response = self.tokenizer.decode(output[0], skip_special_tokens=True)
            
            # Find the assistant's response after the last occurrence of <|assistant|>
            if "<|assistant|>" in response:
                bot_reply = response.split("<|assistant|>")[-1].strip()
            else:
                bot_reply = response.strip()
            
            # Simulate typing for a more natural feel
            self.status_var.set("✏️ HealthMate is typing...")
            self._simulate_typing("HealthMate", bot_reply)
            
        except Exception as e:
            self.add_message("System", f"Error generating response: {str(e)}")
            self.status_var.set("❌ Error generating response")
    
    def _simulate_typing(self, sender, text, delay=0.03):
        """Simulate typing with a realistic delay"""
        self.chat_area.config(state='normal')
        
        # Add the sender part
        self.chat_area.insert(tk.END, f"{sender}: ", "bot")
        self.chat_area.see(tk.END)
        self.chat_area.update()
        
        # Calculate a variable typing speed based on message length
        base_delay = delay
        if len(text) > 200:
            # For longer messages, speed up
            delay = base_delay * 0.7
        
        # Add characters with delay
        for i, char in enumerate(text):
            # Vary typing speed slightly for more realism
            char_delay = delay * np.random.uniform(0.5, 1.5)
            
            # Add pauses at punctuation
            if char in ['.', ',', '!', '?', ';', ':']:
                char_delay *= 3
            
            # Occasional longer pause for realism
            if i > 0 and i % 50 == 0:
                time.sleep(delay * 5)
            
            self.chat_area.insert(tk.END, char, "bot")
            self.chat_area.see(tk.END)
            self.chat_area.update()
            time.sleep(char_delay)
        
        # Add final newlines
        self.chat_area.insert(tk.END, "\n\n", "bot")
        self.chat_area.config(state='disabled')
        self.chat_area.see(tk.END)
        
        # Update status
        self.status_var.set("✅ Ready")
        
        # Add to history
        self.chat_history.add_message(sender, text)


def main():
    """Main entry point for the application"""
    try:
        # Set up better exception handling
        def show_error(exc_type, exc_value, exc_traceback):
            error_msg = f"{exc_type.__name__}: {exc_value}"
            messagebox.showerror(
                "Application Error",
                f"An unexpected error occurred:\n\n{error_msg}\n\n"
                "The application will attempt to continue."
            )
            # Log to console as well
            import traceback
            traceback.print_exception(exc_type, exc_value, exc_traceback)
        
        # Set up the Tkinter root with better styling
        root = tk.Tk()
        root.title("HealthMate")
        
        # Set a minimum size
        root.minsize(800, 600)
        
        # Set app icon if available
        try:
            root.iconbitmap("healthmate.ico")
        except:
            pass  # Icon not critical
        
        # Create and run the app
        app = HealthMateChatApp(root)
        
        # Set up custom exception handler
        tk.Tk.report_callback_exception = show_error
        
        # Start the main loop
        root.mainloop()
        
    except Exception as e:
        messagebox.showerror(
            "Fatal Error",
            f"A critical error occurred while starting the application:\n\n{str(e)}\n\n"
            "The application will now close."
        )


if __name__ == "__main__":
    main()