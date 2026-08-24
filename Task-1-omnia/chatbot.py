import json
import random
import tkinter as tk


# Load chatbot responses
with open("responses.json", "r", encoding="utf-8") as file:
    responses = json.load(file)


def get_response(user_input):
    """Return a rule-based response for the user's message."""

    if user_input in ["exit", "quit", "bye"]:
        return "Goodbye! 👋 Have a great day!", True

    if user_input == "help":
        return (
            "I can respond to greetings, simple questions, and thanks.\n\n"
            "Try:\n"
            "• hello\n"
            "• how are you\n"
            "• who are you\n"
            "• thanks\n"
            "• bye"
        ), False

    if user_input in responses:
        return random.choice(responses[user_input]), False

    return "I'm sorry, I don't understand that yet. 🤔", False


# ---------------- Main Window ----------------

root = tk.Tk()
root.title("DecodeBot - AI Assistant")
root.geometry("520x680")
root.minsize(450, 550)
root.configure(bg="#F4F6FB")


# ---------------- Header ----------------

header = tk.Frame(
    root,
    bg="#6C63FF",
    height=100
)

header.pack(fill="x")
header.pack_propagate(False)


# Bot avatar
avatar = tk.Canvas(
    header,
    width=62,
    height=62,
    bg="#6C63FF",
    highlightthickness=0
)

avatar.place(x=18, y=19)

# Circular avatar background
avatar.create_oval(
    3, 3, 59, 59,
    fill="white",
    outline=""
)

# Simple robot face
avatar.create_oval(
    16, 20, 24, 28,
    fill="#6C63FF",
    outline=""
)

avatar.create_oval(
    38, 20, 46, 28,
    fill="#6C63FF",
    outline=""
)

avatar.create_arc(
    18, 25, 44, 45,
    start=200,
    extent=140,
    style=tk.ARC,
    outline="#6C63FF",
    width=2
)


# Title
title_label = tk.Label(
    header,
    text="DecodeBot",
    font=("Segoe UI", 18, "bold"),
    bg="#6C63FF",
    fg="white"
)

title_label.place(x=92, y=22)


# Status
status_label = tk.Label(
    header,
    text="● Online  •  Rule-Based AI",
    font=("Segoe UI", 9),
    bg="#6C63FF",
    fg="#E8E6FF"
)

status_label.place(x=94, y=55)


# ---------------- Chat Area ----------------

chat_container = tk.Frame(
    root,
    bg="#F4F6FB"
)

chat_container.pack(
    fill="both",
    expand=True,
    padx=12,
    pady=12
)


# Canvas for scrolling
chat_canvas = tk.Canvas(
    chat_container,
    bg="#F4F6FB",
    highlightthickness=0
)

scrollbar = tk.Scrollbar(
    chat_container,
    orient="vertical",
    command=chat_canvas.yview
)

chat_frame = tk.Frame(
    chat_canvas,
    bg="#F4F6FB"
)


chat_frame.bind(
    "<Configure>",
    lambda event: chat_canvas.configure(
        scrollregion=chat_canvas.bbox("all")
    )
)


chat_window = chat_canvas.create_window(
    (0, 0),
    window=chat_frame,
    anchor="nw"
)


chat_canvas.configure(
    yscrollcommand=scrollbar.set
)


chat_canvas.pack(
    side="left",
    fill="both",
    expand=True
)

scrollbar.pack(
    side="right",
    fill="y"
)


def resize_chat_frame(event):
    chat_canvas.itemconfig(
        chat_window,
        width=event.width
    )


chat_canvas.bind(
    "<Configure>",
    resize_chat_frame
)


# ---------------- Chat Messages ----------------

def add_message(message, sender):
    """Display a chat bubble."""

    if sender == "bot":
        bubble_bg = "#FFFFFF"
        text_color = "#252525"
        anchor = "w"

        message_frame = tk.Frame(
            chat_frame,
            bg="#F4F6FB"
        )

        message_frame.pack(
            fill="x",
            padx=8,
            pady=5
        )

        avatar_label = tk.Label(
            message_frame,
            text="🤖",
            font=("Segoe UI Emoji", 16),
            bg="#F4F6FB"
        )

        avatar_label.pack(
            side="left",
            padx=(0, 7)
        )

        bubble = tk.Label(
            message_frame,
            text=message,
            bg=bubble_bg,
            fg=text_color,
            font=("Segoe UI", 10),
            justify="left",
            wraplength=330,
            padx=14,
            pady=10
        )

        bubble.pack(
            side="left",
            anchor=anchor
        )

    else:
        bubble_bg = "#6C63FF"
        text_color = "white"
        anchor = "e"

        message_frame = tk.Frame(
            chat_frame,
            bg="#F4F6FB"
        )

        message_frame.pack(
            fill="x",
            padx=8,
            pady=5
        )

        bubble = tk.Label(
            message_frame,
            text=message,
            bg=bubble_bg,
            fg=text_color,
            font=("Segoe UI", 10),
            justify="left",
            wraplength=330,
            padx=14,
            pady=10
        )

        bubble.pack(
            side="right",
            anchor=anchor
        )

    root.update_idletasks()
    chat_canvas.yview_moveto(1.0)


# ---------------- Send Message ----------------

def send_message(event=None):
    user_input = message_entry.get().strip().lower()

    if not user_input:
        return

    add_message(user_input, "user")

    response, should_exit = get_response(user_input)

    add_message(response, "bot")

    message_entry.delete(0, tk.END)

    if should_exit:
        root.after(1200, root.destroy)


# ---------------- Welcome Message ----------------

add_message(
    "Hello! 👋\n"
    "I'm DecodeBot. How can I help you today?\n\n"
    "Type 'help' to see what I can do.",
    "bot"
)


# ---------------- Input Area ----------------

input_container = tk.Frame(
    root,
    bg="white",
    height=65
)

input_container.pack(
    fill="x",
    padx=12,
    pady=(0, 12)
)

input_container.pack_propagate(False)


message_entry = tk.Entry(
    input_container,
    font=("Segoe UI", 11),
    bg="#F1F2F6",
    fg="#333333",
    relief="flat",
    bd=0
)

message_entry.pack(
    side="left",
    fill="both",
    expand=True,
    padx=(10, 6),
    pady=10,
    ipady=8
)


send_button = tk.Button(
    input_container,
    text="➤",
    command=send_message,
    font=("Segoe UI", 15, "bold"),
    bg="#6C63FF",
    fg="white",
    activebackground="#574FCF",
    activeforeground="white",
    relief="flat",
    bd=0,
    cursor="hand2",
    width=4
)

send_button.pack(
    side="right",
    padx=(0, 8),
    pady=10
)


# Press Enter to send
message_entry.bind(
    "<Return>",
    send_message
)


# Put cursor in input box
message_entry.focus()


# ---------------- Start Application ----------------

root.mainloop()