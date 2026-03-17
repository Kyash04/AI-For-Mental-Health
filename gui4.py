import tkinter as tk
from tkinter import scrolledtext, messagebox, ttk, filedialog
import threading
import torch
import json
import os
import re
from datetime import datetime
import numpy as np
from textblob import TextBlob
from transformers import AutoTokenizer, AutoModelForCausalLM, pipeline, StoppingCriteria, StoppingCriteriaList
import time
from peft import PeftModel

#Configuration

CONFIG = {
    "base_model_id": "TinyLlama/TinyLlama-1.1B-Chat-v1.0",
    "adapter_model_id": "./my_finetuned_healthmate_model",
    "use_finetuned": True,
    "max_new_tokens": 100,          
    "temperature": 0.5,             
    "top_p": 0.85,
    "top_k": 30,                    
    "repetition_penalty": 1.4,      
    "max_history_turns": 4,         
    "typing_delay": 0.02,
    "app_name": "HealthMate",
}

SYSTEM_PROMPT = (
    "You are HealthMate, a compassionate mental health support assistant. "
    "Read the user's message carefully and respond ONLY to what they actually said. "
    "Do NOT contradict what the user said. If they say they feel bad, acknowledge that they feel bad. "
    "You do NOT give medical diagnoses. You do NOT roleplay as anyone else. "
    "Keep responses short, warm, and focused — maximum 3 sentences. "
    "If a user seems in crisis, gently encourage them to contact a professional or helpline."
)

# Hallucination guard: responses containing these patterns get flagged / regenerated
HALLUCINATION_PATTERNS = [
    r"my name is (?!healthmate)\w+",
    r"i am a (?:certified|licensed|registered)\s+\w+",
    r"i work(?: at| in)\s+\w+",
    r"i have \d+ years",
    r"as a (nurse|doctor|therapist|midwife|physician)",
    r"(abortion|politics|religion|drugs how to|illegal)",  # off-topic guard
]

CRISIS_KEYWORDS = [
    "suicide", "kill myself", "end my life", "self harm", "self-harm",
    "hurt myself", "don't want to live", "want to die"
]

CRISIS_RESPONSE = (
    "I hear you, and I'm really glad you reached out. What you're feeling matters deeply. "
    "Please know you're not alone. I strongly encourage you to contact a crisis helpline right away — "
    "the 14416 Suicide & Crisis Lifeline (call or text 14416 in the US) has trained counselors available 24/7. "
    "Would you like to talk about what's been going on?"
)

# Stopping Criteria

class StopOnTokens(StoppingCriteria):
    def __init__(self, stop_ids):
        self.stop_ids = stop_ids

    def __call__(self, input_ids, scores, **kwargs):
        for stop_id in self.stop_ids:
            if input_ids[0][-1] == stop_id:
                return True
        return False

# Emotion Analyzer

class EmotionAnalyzer:
    def __init__(self):
        self.sentiment_analyzer = None

    def initialize(self):
        if self.sentiment_analyzer is None:
            try:
                self.sentiment_analyzer = pipeline(
                    "sentiment-analysis",
                    model="distilbert-base-uncased-finetuned-sst-2-english"
                )
                return True
            except Exception as e:
                print(f"Emotion analyzer init error: {e}")
                return False
        return True

    def detect(self, text):
        blob = TextBlob(text)
        polarity = blob.sentiment.polarity
        emotion, confidence = None, 0

        if self.sentiment_analyzer:
            try:
                result = self.sentiment_analyzer(text[:512])[0] 
                emotion = result['label'].lower()
                confidence = result['score']
            except Exception:
                pass

        if emotion is None:
            if polarity > 0.3:   emotion, confidence = "positive", abs(polarity)
            elif polarity < -0.3: emotion, confidence = "negative", abs(polarity)
            elif polarity > 0.1:  emotion, confidence = "slightly positive", abs(polarity)
            elif polarity < -0.1: emotion, confidence = "slightly negative", abs(polarity)
            else:                  emotion, confidence = "neutral", 0.0

        return {"emotion": emotion, "confidence": confidence, "polarity": polarity}


# Themes

class ThemeManager:
    THEMES = {
        "Light": {
            "bg": "#f5f6fa", "chat_bg": "#ffffff", "chat_fg": "#2f3640",
            "input_bg": "#dcdde1", "button_bg": "#44bd32", "button_fg": "white",
            "accent": "#273c75", "status_bg": "#dcdde1",
            "user_msg": "#e1f5fe", "bot_msg": "#f1f8e9", "system_msg": "#fff3e0",
            "crisis_msg": "#ffebee"
        },
        "Dark": {
            "bg": "#2f3640", "chat_bg": "#353b48", "chat_fg": "#f5f6fa",
            "input_bg": "#1e272e", "button_bg": "#44bd32", "button_fg": "white",
            "accent": "#0097e6", "status_bg": "#1e272e",
            "user_msg": "#01579b", "bot_msg": "#1b5e20", "system_msg": "#4e342e",
            "crisis_msg": "#b71c1c"
        },
        "Calm": {
            "bg": "#e0f7fa", "chat_bg": "#f5f5f5", "chat_fg": "#37474f",
            "input_bg": "#b2ebf2", "button_bg": "#26a69a", "button_fg": "white",
            "accent": "#00897b", "status_bg": "#b2ebf2",
            "user_msg": "#bbdefb", "bot_msg": "#c8e6c9", "system_msg": "#ffecb3",
            "crisis_msg": "#ffcdd2"
        }
    }

    @staticmethod
    def get_theme(name):
        return ThemeManager.THEMES.get(name, ThemeManager.THEMES["Light"])

    @staticmethod
    def get_theme_names():
        return list(ThemeManager.THEMES.keys())


# Chat History

class ChatHistory:
    def __init__(self, app_name="HealthMate"):
        self.app_name = app_name
        self.history = []
        self.history_dir = os.path.join(os.path.expanduser("~"), f".{app_name.lower()}")
        os.makedirs(self.history_dir, exist_ok=True)

    def add_message(self, sender, message, emotion=None):
        self.history.append({
            "sender": sender, "message": message,
            "emotion": emotion, "timestamp": datetime.now().isoformat()
        })

    def get_recent_turns(self, n=6):
        """Return last n conversation turns (user + bot pairs)"""
        turns = [m for m in self.history if m["sender"] in ("You", "HealthMate")]
        return turns[-(n * 2):]  # Each turn = 1 user + 1 bot message

    def save_to_file(self, filename=None):
        if not filename:
            ts = datetime.now().strftime("%Y%m%d_%H%M%S")
            filename = os.path.join(self.history_dir, f"chat_{ts}.json")
        try:
            with open(filename, 'w', encoding='utf-8') as f:
                json.dump({"app": self.app_name, "created": datetime.now().isoformat(),
                           "messages": self.history}, f, indent=2)
            return filename
        except Exception as e:
            print(f"Save error: {e}")
            return None

    def load_from_file(self, filename):
        try:
            with open(filename, 'r', encoding='utf-8') as f:
                data = json.load(f)
                if "messages" in data:
                    self.history = data["messages"]
                    return True
        except Exception as e:
            print(f"Load error: {e}")
        return False

    def clear(self):
        self.history = []


# Anti Hallucination

class ResponseValidator:
    @staticmethod
    def is_hallucinated(text: str) -> bool:
        text_lower = text.lower()
        for pattern in HALLUCINATION_PATTERNS:
            if re.search(pattern, text_lower):
                return True
        return False

    @staticmethod
    def is_crisis(text: str) -> bool:
        text_lower = text.lower()
        return any(kw in text_lower for kw in CRISIS_KEYWORDS)

    @staticmethod
    def clean(text: str) -> str:
        # Remove leftover special tokens
        for tok in ["<|system|>", "<|user|>", "<|assistant|>", "<|endoftext|>", "</s>", "<s>"]:
            text = text.replace(tok, "")
        # Remove leading/trailing whitespace and repeated newlines
        text = re.sub(r'\n{3,}', '\n\n', text).strip()
        return text


# Main App

class HealthMateChatApp:
    def __init__(self, root):
        self.root = root
        self.root.title(f"🧠 {CONFIG['app_name']} - Your Mental Health Assistant")
        self.root.geometry("950x720")
        self.root.minsize(800, 600)

        self.emotion_analyzer = EmotionAnalyzer()
        self.chat_history = ChatHistory(CONFIG["app_name"])
        self.validator = ResponseValidator()
        self.current_theme = "Light"

        self.model = None
        self.tokenizer = None
        self.model_loaded = False
        self.model_loading = False
        self.is_generating = False

        self._setup_menu()
        self._setup_widgets()
        self._apply_theme(self.current_theme)
        self._show_welcome_message()

    # Menu
    def _setup_menu(self):
        menubar = tk.Menu(self.root)

        file_menu = tk.Menu(menubar, tearoff=0)
        file_menu.add_command(label="Save Conversation", command=self._save_conversation)
        file_menu.add_command(label="Load Conversation", command=self._load_conversation)
        file_menu.add_separator()
        file_menu.add_command(label="Clear Chat", command=self._clear_chat)
        file_menu.add_separator()
        file_menu.add_command(label="Exit", command=self.root.quit)
        menubar.add_cascade(label="File", menu=file_menu)

        model_menu = tk.Menu(menubar, tearoff=0)
        model_menu.add_command(label="Load AI Model", command=self.load_model)
        model_menu.add_command(label="Toggle Fine-tuned Adapter",
                               command=self._toggle_adapter)
        model_menu.add_command(label="Init Emotion Analyzer",
                               command=lambda: threading.Thread(
                                   target=self._init_emotion_analyzer, daemon=True).start())
        menubar.add_cascade(label="Model", menu=model_menu)

        theme_menu = tk.Menu(menubar, tearoff=0)
        self.theme_var = tk.StringVar(value=self.current_theme)
        for t in ThemeManager.get_theme_names():
            theme_menu.add_radiobutton(label=t, variable=self.theme_var, value=t,
                                       command=lambda th=t: self._apply_theme(th))
        menubar.add_cascade(label="Theme", menu=theme_menu)

        help_menu = tk.Menu(menubar, tearoff=0)
        help_menu.add_command(label="About", command=self._show_about)
        menubar.add_cascade(label="Help", menu=help_menu)

        self.root.config(menu=menubar)

    # Widgets
    def _setup_widgets(self):
        self.main_frame = tk.Frame(self.root)
        self.main_frame.pack(fill=tk.BOTH, expand=True, padx=10, pady=10)

        # Chat area
        self.chat_frame = tk.Frame(self.main_frame)
        self.chat_frame.pack(fill=tk.BOTH, expand=True)

        self.chat_area = scrolledtext.ScrolledText(
            self.chat_frame, wrap=tk.WORD, state='disabled',
            font=("Segoe UI", 11), cursor="arrow"
        )
        self.chat_area.pack(fill=tk.BOTH, expand=True, padx=5, pady=5)

        for tag in ("user", "bot", "system", "crisis"):
            self.chat_area.tag_configure(tag, lmargin1=10, lmargin2=20, rmargin=10)
        self.chat_area.tag_configure("system", font=("Segoe UI", 10, "italic"))

        # Input area
        self.input_frame = tk.Frame(self.main_frame)
        self.input_frame.pack(fill=tk.X, pady=10)

        self.progress_bar = ttk.Progressbar(
            self.input_frame, orient="horizontal", length=200, mode="indeterminate"
        )

        self.input_frame_inner = tk.Frame(self.input_frame)
        self.input_frame_inner.pack(fill=tk.X)

        self.user_input = tk.Text(self.input_frame_inner, font=("Segoe UI", 11), height=3)
        self.user_input.pack(side=tk.LEFT, fill=tk.X, expand=True, padx=(5, 10))
        self.user_input.bind("<Return>", self._handle_enter)

        self.button_frame = tk.Frame(self.input_frame_inner)
        self.button_frame.pack(side=tk.RIGHT, fill=tk.Y)

        self.send_button = tk.Button(
            self.button_frame, text="Send ▶", command=self.send_message,
            font=("Segoe UI", 10, "bold"), width=10, height=2
        )
        self.send_button.pack(side=tk.TOP, padx=5, pady=2)

        self.stop_button = tk.Button(
            self.button_frame, text="⛔ Stop", command=self._stop_generation,
            font=("Segoe UI", 10), width=10, state='disabled'
        )
        self.stop_button.pack(side=tk.TOP, padx=5, pady=2)

        self.load_model_button = tk.Button(
            self.button_frame, text="Load AI", command=self.load_model,
            font=("Segoe UI", 10), width=10
        )
        self.load_model_button.pack(side=tk.BOTTOM, padx=5, pady=2)

        # Status bar
        self.status_frame = tk.Frame(self.root)
        self.status_frame.pack(fill=tk.X, side=tk.BOTTOM)

        self.status_frame_upper = tk.Frame(self.status_frame)
        self.status_frame_upper.pack(fill=tk.X)

        self.emotion_var = tk.StringVar(value="Emotion: —")
        tk.Label(self.status_frame_upper, textvariable=self.emotion_var,
                 anchor=tk.W, font=("Segoe UI", 10)).pack(side=tk.LEFT, padx=10)

        self.model_status_var = tk.StringVar(value="Model: Not loaded")
        tk.Label(self.status_frame_upper, textvariable=self.model_status_var,
                 anchor=tk.E, font=("Segoe UI", 10)).pack(side=tk.RIGHT, padx=10)

        self.status_var = tk.StringVar(value="✅ Ready")
        self.status_bar = tk.Label(
            self.status_frame, textvariable=self.status_var,
            bd=1, relief=tk.SUNKEN, anchor=tk.W, font=("Segoe UI", 10)
        )
        self.status_bar.pack(fill=tk.X)

        self._stop_flag = False  # For stopping generation mid-stream

    # Theme
    def _apply_theme(self, theme_name):
        theme = ThemeManager.get_theme(theme_name)
        self.current_theme = theme_name

        widgets_bg = [
            self.main_frame, self.chat_frame, self.input_frame,
            self.input_frame_inner, self.button_frame,
            self.status_frame, self.status_frame_upper
        ]
        for w in widgets_bg:
            w.config(bg=theme["bg"])

        self.root.config(bg=theme["bg"])
        self.chat_area.config(bg=theme["chat_bg"], fg=theme["chat_fg"])
        self.chat_area.tag_configure("user", background=theme["user_msg"])
        self.chat_area.tag_configure("bot", background=theme["bot_msg"])
        self.chat_area.tag_configure("system", background=theme["system_msg"])
        self.chat_area.tag_configure("crisis", background=theme["crisis_msg"])

        self.user_input.config(bg=theme["input_bg"], fg=theme["chat_fg"])
        self.send_button.config(bg=theme["button_bg"], fg=theme["button_fg"])
        self.stop_button.config(bg="#e74c3c", fg="white")
        self.load_model_button.config(bg=theme["accent"], fg=theme["button_fg"])

        for lbl in self.status_frame_upper.winfo_children():
            lbl.config(bg=theme["bg"], fg=theme["chat_fg"])
        self.status_bar.config(bg=theme["status_bg"], fg=theme["chat_fg"])

    # Messages
    def _show_welcome_message(self):
        msg = (
            "👋 Welcome to HealthMate!\n\n"
            "I'm your compassionate mental health support assistant.\n"
            "• Click 'Load AI' to activate the assistant\n"
            "• Press Enter to send, Shift+Enter for a new line\n"
            "• Use File > Save to export your conversation\n\n"
            "⚠️ I am not a replacement for professional mental health care. "
            "If you are in crisis, please contact 14416 (Suicide & Crisis Lifeline)."
        )
        self.add_message("HealthMate", msg)

    def _show_about(self):
        messagebox.showinfo("About HealthMate",
            "HealthMate – Mental Health Support Assistant\n\n"
            "Powered by TinyLlama-1.1B (fine-tuned)\n"
            "Crisis resource: 14416 Suicide & Crisis Lifeline\n\n"
            "⚠️ Not a substitute for professional care.")

    def add_message(self, sender, message, emotion=None, tag_override=None):
        """Thread-safe message insertion"""
        def _insert():
            self.chat_area.config(state='normal')
            tag = tag_override or (
                "user" if sender == "You" else
                "crisis" if sender == "HealthMate" and ResponseValidator.is_crisis(message) else
                "bot" if sender == "HealthMate" else
                "system"
            )
            header = f"{sender} ({emotion}): " if emotion and sender == "You" else f"{sender}: "
            self.chat_area.insert(tk.END, header + f"{message}\n\n", tag)
            self.chat_area.config(state='disabled')
            self.chat_area.see(tk.END)

            if sender not in ("System",):
                self.chat_history.add_message(sender, message, emotion)

        self.root.after(0, _insert)

    # Input Handling
    def _handle_enter(self, event):
        if not (event.state & 0x1):  # Shift not held
            self.send_message()
            return "break"

    def _stop_generation(self):
        self._stop_flag = True
        self.status_var.set("⛔ Stopping...")

    def send_message(self):
        if self.is_generating:
            return
        user_msg = self.user_input.get("1.0", "end-1c").strip()
        if not user_msg:
            return
        self.user_input.delete("1.0", tk.END)

        # Crisis check BEFORE model
        if ResponseValidator.is_crisis(user_msg):
            emotion_result = {"emotion": "distressed", "confidence": 1.0}
            self.emotion_var.set("Emotion: Distressed ⚠️")
            self.add_message("You", user_msg, "distressed")
            self.add_message("HealthMate", CRISIS_RESPONSE, tag_override="crisis")
            return

        emotion_result = self.emotion_analyzer.detect(user_msg)
        emotion = emotion_result["emotion"]
        self.emotion_var.set(f"Emotion: {emotion.capitalize()} ({emotion_result['confidence']:.2f})")
        self.add_message("You", user_msg, emotion)

        if not self.model_loaded:
            self.add_message("System", "⚠️ Please load the AI model first (click 'Load AI').")
            return

        self.is_generating = True
        self._stop_flag = False
        self.send_button.config(state='disabled')
        self.stop_button.config(state='normal')
        threading.Thread(target=self._generate_response, args=(user_msg, emotion), daemon=True).start()

    # Model Loading
    def _toggle_adapter(self):
        CONFIG["use_finetuned"] = not CONFIG["use_finetuned"]
        state = "ON (fine-tuned)" if CONFIG["use_finetuned"] else "OFF (base model only)"
        self.add_message("System", f"🔄 Fine-tuned adapter toggled: {state}. Reload model to apply.")

    def load_model(self):
        if self.model_loaded:
            self.add_message("System", "✅ Model already loaded.")
            return
        if self.model_loading:
            self.add_message("System", "⏳ Still loading, please wait...")
            return
        self.model_loading = True
        self.status_var.set("⏳ Loading model...")
        self.load_model_button.config(state='disabled')
        self.model_status_var.set("Model: Loading...")
        self.progress_bar.pack(fill=tk.X, padx=5, pady=5)
        self.progress_bar.start(10)
        threading.Thread(target=self._load_model_thread, daemon=True).start()

    def _load_model_thread(self):
        try:
            self.add_message("System", "⏱️ Loading base model (TinyLlama)...")
            device = "cuda" if torch.cuda.is_available() else "cpu"
            dtype = torch.float16 if device == "cuda" else torch.float32

            base_model = AutoModelForCausalLM.from_pretrained(
                CONFIG["base_model_id"],
                torch_dtype=dtype,
                trust_remote_code=True,
                low_cpu_mem_usage=True,
            )
            self.tokenizer = AutoTokenizer.from_pretrained(
                CONFIG["base_model_id"], trust_remote_code=True
            )
            # Ensure pad token is set
            if self.tokenizer.pad_token is None:
                self.tokenizer.pad_token = self.tokenizer.eos_token

            if CONFIG["use_finetuned"] and os.path.exists(CONFIG["adapter_model_id"]):
                self.add_message("System", "🔗 Merging fine-tuned adapter...")
                peft_model = PeftModel.from_pretrained(base_model, CONFIG["adapter_model_id"])
                self.model = peft_model.merge_and_unload()
                status_label = "Loaded (Fine-tuned)"
            else:
                self.model = base_model
                status_label = "Loaded (Base)" if not CONFIG["use_finetuned"] else "Loaded (Base — adapter not found)"

            self.model = self.model.to(device)
            self.model.eval()  # IMPORTANT: put in eval mode

            self.model_loaded = True
            self.model_status_var.set(f"Model: {status_label} | Device: {device.upper()}")
            self.status_var.set("✅ Model ready")
            self.add_message("System", f"🎉 Model ready! Running on {device.upper()}.")

        except Exception as e:
            err = str(e)
            self.model_status_var.set("Model: ❌ Error")
            self.status_var.set("❌ Model load failed")
            self.add_message("System", f"❌ Load error: {err}\n\nTip: Check that your adapter folder exists and peft/accelerate are installed.")
        finally:
            self.model_loading = False
            self.root.after(0, lambda: self.load_model_button.config(state='normal'))
            self.root.after(0, lambda: self.progress_bar.stop())
            self.root.after(0, lambda: self.progress_bar.pack_forget())

    # Response Generation
    def _build_prompt(self, user_msg: str, emotion: str) -> str:
        """Build a well-structured TinyLlama chat prompt with capped history"""
        recent = self.chat_history.get_recent_turns(CONFIG["max_history_turns"])

        prompt = f"<|system|>\n{SYSTEM_PROMPT}\nUser emotional tone: {emotion}.\n</s>\n"

        for msg in recent:
            if msg["sender"] == "You":
                prompt += f"<|user|>\n{msg['message']}</s>\n"
            elif msg["sender"] == "HealthMate":
                prompt += f"<|assistant|>\n{msg['message']}</s>\n"

        prompt += f"<|user|>\n{user_msg}</s>\n<|assistant|>\n"
        return prompt

    def _generate_response(self, user_msg: str, emotion: str):
        max_retries = 2
        attempt = 0
        bot_reply = None

        while attempt <= max_retries and not self._stop_flag:
            try:
                self.root.after(0, lambda: self.status_var.set("💬 Generating response..."))
                prompt = self._build_prompt(user_msg, emotion)

                inputs = self.tokenizer(
                    prompt, return_tensors="pt", truncation=True, max_length=1024
                ).to(self.model.device)

                # Dynamic stop tokens
                stop_ids = [
                    self.tokenizer.eos_token_id,
                    self.tokenizer.convert_tokens_to_ids("</s>"),
                    self.tokenizer.convert_tokens_to_ids("<|user|>"),
                    self.tokenizer.convert_tokens_to_ids("<|assistant|>"),
                ]
                stop_ids = [i for i in stop_ids if i is not None]
                stopping = StoppingCriteriaList([StopOnTokens(stop_ids)])

                with torch.no_grad():
                    output = self.model.generate(
                        input_ids=inputs.input_ids,
                        attention_mask=inputs.attention_mask,
                        max_new_tokens=CONFIG["max_new_tokens"],
                        do_sample=True,
                        temperature=CONFIG["temperature"],
                        top_p=CONFIG["top_p"],
                        top_k=CONFIG["top_k"],
                        repetition_penalty=CONFIG["repetition_penalty"],
                        pad_token_id=self.tokenizer.eos_token_id,
                        eos_token_id=stop_ids,
                        stopping_criteria=stopping,
                    )

                prompt_len = inputs.input_ids.shape[-1]
                raw = self.tokenizer.decode(output[0][prompt_len:], skip_special_tokens=False)
                cleaned = ResponseValidator.clean(raw)

                # Validate — retry if hallucinated
                if ResponseValidator.is_hallucinated(cleaned):
                    attempt += 1
                    CONFIG["temperature"] = min(CONFIG["temperature"] + 0.05, 1.0)  # Nudge temp
                    self.root.after(0, lambda: self.status_var.set(
                        f"⚠️ Detected hallucination, retrying ({attempt}/{max_retries})..."
                    ))
                    continue

                bot_reply = cleaned if cleaned else "I'm here to listen. Could you tell me more about what you're experiencing?"
                break

            except Exception as e:
                self.root.after(0, lambda err=e: self.add_message("System", f"❌ Generation error: {err}"))
                break

        if self._stop_flag:
            bot_reply = None
        elif bot_reply is None and attempt > max_retries:
            bot_reply = (
                "I want to make sure I give you a thoughtful response. "
                "Could you share a bit more about how you're feeling?"
            )

        if bot_reply:
            self.root.after(0, lambda: self.status_var.set("✏️ HealthMate is typing..."))
            self._simulate_typing("HealthMate", bot_reply)
        else:
            self.root.after(0, lambda: self.status_var.set("✅ Ready"))

        self.root.after(0, self._reset_ui_after_generation)

    def _reset_ui_after_generation(self):
        self.is_generating = False
        self.send_button.config(state='normal')
        self.stop_button.config(state='disabled')
        self.status_var.set("✅ Ready")

    def _simulate_typing(self, sender, text):
        """Realistic typing simulation with stop support"""
        self.chat_area.config(state='normal')
        self.chat_area.insert(tk.END, f"{sender}: ", "bot")
        self.chat_area.see(tk.END)
        self.chat_area.update()

        base_delay = CONFIG["typing_delay"] * (0.6 if len(text) > 200 else 1.0)

        for i, char in enumerate(text):
            if self._stop_flag:
                break
            delay = base_delay * np.random.uniform(0.5, 1.5)
            if char in ".!?,;:": delay *= 2.5
            if i > 0 and i % 60 == 0: time.sleep(base_delay * 4)

            self.chat_area.insert(tk.END, char, "bot")
            self.chat_area.see(tk.END)
            self.chat_area.update()
            time.sleep(delay)

        self.chat_area.insert(tk.END, "\n\n", "bot")
        self.chat_area.config(state='disabled')
        self.chat_area.see(tk.END)
        self.chat_history.add_message(sender, text)

    # File Operations
    def _save_conversation(self):
        fn = filedialog.asksaveasfilename(
            defaultextension=".json",
            filetypes=[("JSON", "*.json"), ("All", "*.*")],
            title="Save Conversation"
        )
        if fn:
            saved = self.chat_history.save_to_file(fn)
            self.add_message("System", f"{'Saved to ' + os.path.basename(saved) if saved else '❌ Save failed'}")

    def _load_conversation(self):
        fn = filedialog.askopenfilename(
            filetypes=[("JSON", "*.json"), ("All", "*.*")],
            title="Load Conversation"
        )
        if fn and self.chat_history.load_from_file(fn):
            self.chat_area.config(state='normal')
            self.chat_area.delete(1.0, tk.END)
            self.chat_area.config(state='disabled')
            for msg in self.chat_history.history:
                self.add_message(msg["sender"], msg["message"], msg.get("emotion"))
            self.add_message("System", f"Loaded: {os.path.basename(fn)}")
        else:
            self.add_message("System", "❌ Failed to load conversation.")

    def _clear_chat(self):
        if messagebox.askyesno("Clear Chat", "Clear all chat history?"):
            self.chat_area.config(state='normal')
            self.chat_area.delete(1.0, tk.END)
            self.chat_area.config(state='disabled')
            self.chat_history.clear()
            self._show_welcome_message()

    def _init_emotion_analyzer(self):
        self.root.after(0, lambda: self.status_var.set("⏳ Initializing emotion analyzer..."))
        ok = self.emotion_analyzer.initialize()
        msg = "✅ Emotion analyzer ready" if ok else "⚠️ Using basic emotion analysis (TextBlob)"
        self.add_message("System", msg)
        self.root.after(0, lambda: self.status_var.set(msg))


# Entry Points

def main():
    root = tk.Tk()
    root.title(CONFIG["app_name"])
    root.minsize(800, 600)

    try:
        root.iconbitmap("healthmate.ico")
    except Exception:
        pass

    app = HealthMateChatApp(root)

    def _handle_exception(exc_type, exc_value, exc_tb):
        import traceback
        traceback.print_exception(exc_type, exc_value, exc_tb)
        messagebox.showerror("Error", f"{exc_type.__name__}: {exc_value}\n\nThe app will continue.")

    root.report_callback_exception = _handle_exception
    root.mainloop()


if __name__ == "__main__":
    main()