import tkinter as tk
from tkinter import filedialog, messagebox
from PIL import Image, ImageTk
import threading
import pyaudio
import numpy as np
from transformers import pipeline
import ChatAI as ca
import MoviesRecomment as mr
import AgePredict as ap
import sys
sys.path.append('Fine_turning_LLM')
import Fine_turning_LLM as llm
import psutil
import subprocess
import RemoveBackground as rb


class AIAssistantApp:
    def __init__(self):
        self.FORMAT = pyaudio.paInt16  
        self.CHANNELS = 1           
        self.RATE = 16000           
        self.CHUNK = 1024
        self.Amode = False
        self.Qtag = ""   
        audio = pyaudio.PyAudio()
        self.root = tk.Tk()
        self.root.title("AI chat bot")
        self.recording = False
        
        self.canvas = tk.Canvas(self.root, width=220, height=220, bg="black")
        self.canvas.pack()

        self.image_path = "data/ChatAI.png"
        self.image = Image.open(self.image_path)
        self.image = self.image.resize((200, 200),Image.LANCZOS)
        self.photo = ImageTk.PhotoImage(self.image)
        self.image_on_canvas = self.canvas.create_image(111, 111, anchor=tk.CENTER, image=self.photo)

        self.label = tk.Label(self.root, text="💤💤", font=("Arial", 9))
        self.label.pack()

        self.tb_on_path = "data/tb_on.png"
        self.tb_off_path = "data/tb_off.png"

        self.on = Image.open(self.tb_on_path)
        self.on = ImageTk.PhotoImage(self.on.resize((65, 30),Image.LANCZOS))
        self.off = Image.open(self.tb_off_path)
        self.off = ImageTk.PhotoImage(self.off.resize((65, 30),Image.LANCZOS))

        self.toggle_state = tk.BooleanVar()
        self.toggle_state.set(False)

        self.toggle_button = tk.Button(self.root, image = self.off, command=self.handle_toggle)
        self.toggle_button.place(relx=1.0, x=-10, y=10, anchor='ne')

        self.toggle_state.trace_add("write", self.handle_toggle)

        self.upload_button = tk.Button(self.root, text="Remove Background image", command=self.upload_image, bg="green")
        self.upload_button.pack(pady=(5, 5))

        self.clear_button = tk.Button(self.root, text="Clear chat box", command=self.clear_chat_history, bg="green")
        self.clear_button.pack(pady=(5, 5))

        self.chat_frame = tk.Frame(self.root)
        self.chat_frame.pack(padx=10, pady=10, fill=tk.BOTH, expand=True)

        self.chat_log = tk.Text(self.chat_frame, wrap=tk.WORD, height=20, width=50)
        self.chat_log.pack(side=tk.LEFT, fill=tk.BOTH, expand=True)
        
        self.scrollbar = tk.Scrollbar(self.chat_frame, orient=tk.VERTICAL, command=self.chat_log.yview)
        self.scrollbar.pack(side=tk.RIGHT, fill=tk.Y)
        self.chat_log.config(yscrollcommand=self.scrollbar.set)
        self.chat_log.tag_configure('bot', justify='left', foreground='green')
        self.chat_log.tag_configure('you', justify='right', foreground='blue')

        self.entry_frame = tk.Frame(self.root)
        self.entry_frame.pack(padx=10, pady=(0, 10), fill=tk.X)
        
        self.entry = tk.Entry(self.root, width=61)
        self.entry.pack(in_=self.entry_frame, side=tk.LEFT, fill=tk.X, expand=True)
        self.entry.bind("<Return>", self.handle_input)

        self.send_button = tk.Button(self.root, text="Send", anchor=tk.CENTER, command=self.handle_input, state=tk.DISABLED, bg="blue")
        self.send_button.pack(in_=self.entry_frame, side=tk.LEFT, padx=(5, 10), pady=(5, 5))

        self.entry.bind("<KeyRelease>", lambda event: self.check_send_button_state())

        self.audio = pyaudio.PyAudio()
        
        self.stream = audio.open(format=self.FORMAT,
                            channels=self.CHANNELS,
                            rate=self.RATE,
                            input=True,
                            frames_per_buffer=self.CHUNK)
        
        
        # self.recognizer = pipeline("automatic-speech-recognition", model="vinai/PhoWhisper-small")
        # self.recognizer = pipeline("automatic-speech-recognition", model="facebook/wav2vec2-base-100h")
        self.recognizer = pipeline("automatic-speech-recognition", model="openai/whisper-base.en")

        threading.Thread(target=self.listen_for_speech, daemon=True).start()

        self.root.mainloop()

    def clear_chat_history(self):
        if messagebox.askyesno("Confirm", "Are you sure you want to clear the chat history?"):
            self.chat_log.delete('1.0', tk.END)

    def upload_image(self):
        file_path = filedialog.askopenfilename(filetypes=[("Image files", "*.png;*.jpg;*.jpeg")])
        if file_path:
            try:
                self.upload_image = Image.open(file_path)
                nobg_image = rb.generate_image(self.upload_image)
                self.show_image(nobg_image)
            except Exception as e:
                messagebox.showerror("Error", f"Unable to open image file: {e}")

    def show_image(self, image):
        frame = tk.Frame(self.chat_log)
        frame.pack(pady=5, fill=tk.X)

        image = image.resize((200, 200), Image.LANCZOS)
        upload_img = ImageTk.PhotoImage(image)
        
        label = tk.Label(frame, image=upload_img)
        label.image = upload_img
        label.pack(side=tk.LEFT, padx=5)

        save_button = tk.Button(frame, text="Save Image", command=lambda: self.save_image(image))
        save_button.pack(side=tk.RIGHT, padx=10)

        self.chat_log.window_create(tk.END, window=frame)
        self.chat_log.insert(tk.END, "\n")


    def save_image(self, image):
        file_path = filedialog.asksaveasfilename(defaultextension=".png", filetypes=[("PNG files", "*.png")])
        if file_path:
            try:
                image.save(file_path)
                messagebox.showinfo("Success", "Image saved successfully!")
            except Exception as e:
                messagebox.showerror("Error", f"Unable to save image file: {e}")

    def handle_toggle(self):
        if self.Amode:
            self.toggle_button.config(image = self.off)
            self.Amode = False
            self.Qtag = ""
            print("Qtag = None")
            print("mode = ANN")
        else:
            self.toggle_button.config(image = self.on)
            self.Amode = True
            self.Qtag = ""
            print("Qtag = None")
            print("mode = LLM")
    
    def record_audio(self):
        Speech_frames = []
        silent_frames = 0
        silence_threshold = 0.2
        while self.recording:
            data = self.stream.read(self.CHUNK)
            Speech_frames.append(data)
            audio_data = np.frombuffer(data, dtype=np.int16)
            if np.max(audio_data) < silence_threshold * np.iinfo(np.int16).max:
                silent_frames += 1
            else:
                silent_frames = 0
            if silent_frames > 60:
                break

        audio_data = np.frombuffer(b''.join(Speech_frames), dtype=np.int16)
        return audio_data

    def transcribe_audio(self, audio_data):
        audio_input = audio_data.astype(np.float32) / 32768.0
        result = self.recognizer({"sampling_rate": self.RATE, "raw": audio_input})
        return result['text']
    
    def add_text(self, text, tag):
        self.chat_log.insert(tk.END, text + "\n", tag)
        self.chat_log.yview(tk.END)

    def handle_input(self, event=None):
        input_text = self.entry.get()
        self.entry.delete(0, tk.END)
        self.show_text_and_respond(input_text, self.Qtag)
        self.send_button.config(state=tk.DISABLED)

    def check_send_button_state(self):
        message = self.entry.get()
        if message.strip():
            self.send_button.config(state=tk.NORMAL)
        else:
            self.send_button.config(state=tk.DISABLED)

    def show_text_and_respond(self, input_text, Qtag, stage = 1):
        self.add_text(input_text, "you")
        input_text = input_text.lower()
        key , command = self.check_starting_keyword(input_text)
        if Qtag == "Movie":
            rec = mr.get_recommendations(input_text)
            if rec.empty:
                self.add_text("Bot: Sorry, the movie name (year of production) or genres you provided are not in my data", "bot")
            else:
                movies_list = rec.apply(lambda row: f"{row['title']} - {row['genres']} - {round(row['rating'], 1)}", axis=1).tolist()
                for movie in movies_list:
                    self.add_text(movie, "bot")
            self.Qtag=""
            print(self.Qtag)
            return
        else:
            if key == None:
                self.add_text("Bot: I'm thinking, please bear with me and wait a moment...", "bot")
                self.entry.config(state=tk.DISABLED)
                self.send_button.config(state=tk.DISABLED)
                threading.Thread(target=self.get_chatbot_response, args=(input_text, stage,)).start()

        if key == "google search":
            search_results = ca.get_google_search(command)
            link, res = ca.try_get_content(search_results)
            self.add_text(f"Bot: {link}", "bot")
            self.add_text(f"Bot: {res}", "bot")
            if stage == 0:
                ca.speech(res)
            else:
                pass

        if key == "open":
            self.open_app(command)

        if key == "close":
            self.close_app(command)

    def check_starting_keyword(self, input_text):
        keywords = ["google search", "open", "close"]
        
        for command in keywords:
            if input_text.startswith(command):
                keyword = command
                rest = input_text[len(command):].strip()
                return keyword, rest
    
        return None, None
    
    def get_chatbot_response(self, input_text, stage):
        if self.Amode:
            res = llm.get_response(input_text)
            self.chat_log.delete("end-2l", "end-1l")
            self.add_text(f"Bot: {res}", "bot")
            self.entry.config(state=tk.NORMAL)
            self.send_button.config(state=tk.NORMAL)
            if stage == 0:
                ca.speech(res)

        else:
            res, self.Qtag = ca.chatbot_response(input_text)
            print(self.Qtag)
            if res == "none":
                search_results = ca.get_google_search(input_text)
                link, res = ca.try_get_content(search_results)
                self.chat_log.delete("end-2l", "end-1l")
                self.add_text("Bot: sorry, i didn't know this but i found some links to help you", "bot")
                self.add_text(f"Bot: {link}", "bot")
                self.add_text(f"Bot: {res}", "bot")
                self.entry.config(state=tk.NORMAL)
                self.send_button.config(state=tk.NORMAL)
            else:
                self.chat_log.delete("end-2l", "end-1l")
                self.add_text(f"Bot: {res}", "bot")
                self.entry.config(state=tk.NORMAL)
                self.send_button.config(state=tk.NORMAL)
            if stage == 0:
                ca.speech(res)
    
    def open_app(self, app_name):
        try:
            subprocess.Popen(["start", app_name], shell=True)
            self.add_text(f"{app_name} opened successfully.", "bot")
        except Exception as e:
            print(f"Failed to open {app_name}: {str(e)}")
            self.add_text(f"Failed to open {app_name}", "bot")

    def close_app(self, app_name):
        closed = False
        for proc in psutil.process_iter(['pid', 'name']):
            if app_name in proc.info['name'].lower():
                pid = proc.info['pid']
                try:
                    process = psutil.Process(pid)
                    process.terminate()
                    closed = True
                    self.add_text(f"{app_name} closed successfully.", "bot")
                except Exception as e:
                    print(f"Failed to close {app_name}: {str(e)}")
                    self.add_text(f"Failed to close {app_name}", "bot")
        if not closed:
            self.add_text(f"No process found with the name '{app_name}' running.", "bot")
        
    def listen_for_speech(self):
        while True:
            if self.Qtag == "my age":
                self.recording = False
                self.canvas.config(bg="green")
                ca.speech("I'm listenning...")
                self.label.config(text="I'm listenning... (Say stop to turn off the chat bot 😢)")
                self.recording = True
                audio_data = self.record_audio()
                age, accuracy = ap.get_age(audio_data)
                self.add_text(f"I guess your age is: {age} years old - accuracy: {accuracy}%", "bot")
                ca.speech(f"I guess your age is: {age} years old - accuracy: {accuracy}%")
                self.Qtag=""
                print("Qtag = None")
            else:
                try:
                    self.label.config(text="--You can say hello to wake up the Bot and it will listen from your microphone--")
                    self.canvas.config(bg="black")
                    self.recording = True
                    audio_data = self.record_audio()
                    text = self.transcribe_audio(audio_data)
                    # print("Văn bản được nhận diện:", text)
                    # print(self.Qtag)
                    # self.add_text("You: " + text, "you")
                    if "hello" in text.lower():
                        self.recording = False
                        self.canvas.config(bg="green")
                        # self.add_text("Bot: " + "I'm listenning...", "bot")
                        ca.speech("I'm listenning...")
                        self.label.config(text="I'm listenning... (Say stop to turn off the chat bot 😢)")
                        self.recording = True
                        audio_data_1 = self.record_audio()
                        text_1 = self.transcribe_audio(audio_data_1)
                        print("Văn bản được nhận diện:", text)
                        self.show_text_and_respond(text_1, self.Qtag, stage=0)
                        if "stop" in text_1.lower():
                            self.root.destroy()
                except SyntaxError as e:
                    print(e)
                    continue

AIAssistantApp()
