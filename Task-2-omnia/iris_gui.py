# ==========================================
# 🌸 Iris AI Classifier - GUI Demo
# ==========================================

import tkinter as tk
from tkinter import messagebox

from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import accuracy_score


# ==========================================
# 1. Load Iris Dataset
# ==========================================

iris = load_iris()

X = iris.data
y = iris.target


# ==========================================
# 2. Train/Test Split
# ==========================================

X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.2,
    random_state=42,
    stratify=y
)


# ==========================================
# 3. Feature Scaling
# ==========================================

scaler = StandardScaler()

X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)


# ==========================================
# 4. Train Final KNN Model
# ==========================================

best_k = 1

knn = KNeighborsClassifier(
    n_neighbors=best_k
)

knn.fit(
    X_train_scaled,
    y_train
)


# ==========================================
# 5. Calculate Model Accuracy
# ==========================================

y_pred = knn.predict(X_test_scaled)

accuracy = accuracy_score(
    y_test,
    y_pred
)

accuracy_percentage = accuracy * 100


# ==========================================
# 6. Colors
# ==========================================

BG_COLOR = "#0F172A"
CARD_COLOR = "#1E293B"
CARD_COLOR_2 = "#172033"

TEXT_COLOR = "#F8FAFC"
SECONDARY_TEXT = "#94A3B8"

ACCENT_COLOR = "#A855F7"
ACCENT_DARK = "#7E22CE"

SUCCESS_COLOR = "#22C55E"
INPUT_COLOR = "#0B1220"

BUTTON_COLOR = "#A855F7"
BUTTON_HOVER = "#9333EA"


# ==========================================
# 7. Main Window
# ==========================================

window = tk.Tk()

window.title("Iris AI Classifier")

window.geometry("900x650")

window.configure(
    bg=BG_COLOR
)

window.resizable(False, False)


# ==========================================
# 8. Helper Functions
# ==========================================

def clear_inputs():

    sepal_length_entry.delete(0, tk.END)
    sepal_width_entry.delete(0, tk.END)
    petal_length_entry.delete(0, tk.END)
    petal_width_entry.delete(0, tk.END)

    result_title.config(
        text="READY FOR PREDICTION",
        fg=SECONDARY_TEXT
    )

    result_value.config(
        text="—",
        fg=TEXT_COLOR
    )


def use_example():

    clear_inputs()

    sepal_length_entry.insert(0, "5.1")
    sepal_width_entry.insert(0, "3.5")
    petal_length_entry.insert(0, "1.4")
    petal_width_entry.insert(0, "0.2")


def predict_flower():

    try:

        sepal_length = float(
            sepal_length_entry.get()
        )

        sepal_width = float(
            sepal_width_entry.get()
        )

        petal_length = float(
            petal_length_entry.get()
        )

        petal_width = float(
            petal_width_entry.get()
        )

        # Create new flower sample
        new_flower = [[
            sepal_length,
            sepal_width,
            petal_length,
            petal_width
        ]]

        # Scale using the same scaler
        new_flower_scaled = scaler.transform(
            new_flower
        )

        # Make prediction
        prediction = knn.predict(
            new_flower_scaled
        )

        flower_name = iris.target_names[
            prediction[0]
        ]

        # Display result
        result_title.config(
            text="PREDICTION",
            fg=SUCCESS_COLOR
        )

        result_value.config(
            text=flower_name.upper(),
            fg=SUCCESS_COLOR
        )

    except ValueError:

        messagebox.showerror(
            "Invalid Input",
            "Please enter valid numerical values."
        )


# ==========================================
# 9. Header
# ==========================================

header_frame = tk.Frame(
    window,
    bg=BG_COLOR
)

header_frame.pack(
    fill="x",
    padx=50,
    pady=(35, 10)
)


title_label = tk.Label(
    header_frame,
    text="🌸  IRIS AI CLASSIFIER",
    font=("Arial", 26, "bold"),
    bg=BG_COLOR,
    fg=TEXT_COLOR
)

title_label.pack(
    anchor="w"
)


subtitle_label = tk.Label(
    header_frame,
    text="Machine Learning • K-Nearest Neighbors",
    font=("Arial", 12),
    bg=BG_COLOR,
    fg=SECONDARY_TEXT
)

subtitle_label.pack(
    anchor="w",
    pady=(5, 0)
)


# ==========================================
# 10. Statistics Cards
# ==========================================

stats_frame = tk.Frame(
    window,
    bg=BG_COLOR
)

stats_frame.pack(
    fill="x",
    padx=50,
    pady=20
)


# Best K Card
k_card = tk.Frame(
    stats_frame,
    bg=CARD_COLOR,
    width=250,
    height=90
)

k_card.pack(
    side="left",
    padx=(0, 15)
)

k_card.pack_propagate(False)


tk.Label(
    k_card,
    text="BEST K",
    font=("Arial", 10, "bold"),
    bg=CARD_COLOR,
    fg=SECONDARY_TEXT
).pack(
    pady=(15, 2)
)


tk.Label(
    k_card,
    text=str(best_k),
    font=("Arial", 22, "bold"),
    bg=CARD_COLOR,
    fg=ACCENT_COLOR
).pack()


# Accuracy Card
accuracy_card = tk.Frame(
    stats_frame,
    bg=CARD_COLOR,
    width=250,
    height=90
)

accuracy_card.pack(
    side="left"
)

accuracy_card.pack_propagate(False)


tk.Label(
    accuracy_card,
    text="TEST ACCURACY",
    font=("Arial", 10, "bold"),
    bg=CARD_COLOR,
    fg=SECONDARY_TEXT
).pack(
    pady=(15, 2)
)


tk.Label(
    accuracy_card,
    text=f"{accuracy_percentage:.2f}%",
    font=("Arial", 22, "bold"),
    bg=CARD_COLOR,
    fg=SUCCESS_COLOR
).pack()


# ==========================================
# 11. Main Content
# ==========================================

content_frame = tk.Frame(
    window,
    bg=BG_COLOR
)

content_frame.pack(
    fill="both",
    expand=True,
    padx=50,
    pady=10
)


# ==========================================
# 12. Input Card
# ==========================================

input_card = tk.Frame(
    content_frame,
    bg=CARD_COLOR,
    width=400,
    height=320
)

input_card.pack(
    side="left",
    fill="y",
    padx=(0, 20)
)

input_card.pack_propagate(False)


tk.Label(
    input_card,
    text="FLOWER MEASUREMENTS",
    font=("Arial", 13, "bold"),
    bg=CARD_COLOR,
    fg=TEXT_COLOR
).pack(
    pady=(20, 15)
)


# Input helper
def create_input(
    parent,
    label_text,
    row
):

    label = tk.Label(
        parent,
        text=label_text,
        font=("Arial", 10),
        bg=CARD_COLOR,
        fg=SECONDARY_TEXT
    )

    label.place(
        x=30,
        y=65 + row * 50
    )

    entry = tk.Entry(
        parent,
        font=("Arial", 11),
        bg=INPUT_COLOR,
        fg=TEXT_COLOR,
        insertbackground=TEXT_COLOR,
        relief="flat",
        width=18
    )

    entry.place(
        x=190,
        y=62 + row * 50,
        height=30
    )

    return entry


sepal_length_entry = create_input(
    input_card,
    "Sepal Length (cm)",
    0
)

sepal_width_entry = create_input(
    input_card,
    "Sepal Width (cm)",
    1
)

petal_length_entry = create_input(
    input_card,
    "Petal Length (cm)",
    2
)

petal_width_entry = create_input(
    input_card,
    "Petal Width (cm)",
    3
)


# ==========================================
# Example Button
# ==========================================

example_button = tk.Button(
    input_card,
    text="Use Example",
    font=("Arial", 9),
    bg=CARD_COLOR_2,
    fg=SECONDARY_TEXT,
    activebackground=CARD_COLOR_2,
    activeforeground=TEXT_COLOR,
    relief="flat",
    command=use_example,
    cursor="hand2"
)

example_button.place(
    x=30,
    y=275,
    width=110,
    height=28
)


# ==========================================
# Predict Button
# ==========================================

predict_button = tk.Button(
    input_card,
    text="✨  PREDICT FLOWER",
    font=("Arial", 11, "bold"),
    bg=BUTTON_COLOR,
    fg="white",
    activebackground=BUTTON_HOVER,
    activeforeground="white",
    relief="flat",
    command=predict_flower,
    cursor="hand2"
)

predict_button.place(
    x=155,
    y=270,
    width=210,
    height=38
)


# ==========================================
# 13. Result Card
# ==========================================

result_card = tk.Frame(
    content_frame,
    bg=CARD_COLOR,
    width=350,
    height=320
)

result_card.pack(
    side="left",
    fill="both",
    expand=True
)

result_card.pack_propagate(False)


tk.Label(
    result_card,
    text="MODEL RESULT",
    font=("Arial", 13, "bold"),
    bg=CARD_COLOR,
    fg=TEXT_COLOR
).pack(
    pady=(45, 20)
)


result_title = tk.Label(
    result_card,
    text="READY FOR PREDICTION",
    font=("Arial", 10, "bold"),
    bg=CARD_COLOR,
    fg=SECONDARY_TEXT
)

result_title.pack(
    pady=10
)


result_value = tk.Label(
    result_card,
    text="—",
    font=("Arial", 34, "bold"),
    bg=CARD_COLOR,
    fg=TEXT_COLOR
)

result_value.pack(
    pady=20
)


tk.Label(
    result_card,
    text="Powered by KNN",
    font=("Arial", 10),
    bg=CARD_COLOR,
    fg=SECONDARY_TEXT
).pack(
    pady=5
)


# ==========================================
# 14. Footer
# ==========================================

footer = tk.Label(
    window,
    text="Iris Classification • Scikit-learn • Python",
    font=("Arial", 9),
    bg=BG_COLOR,
    fg="#64748B"
)

footer.pack(
    pady=(5, 15)
)


# ==========================================
# 15. Start Application
# ==========================================

window.mainloop()