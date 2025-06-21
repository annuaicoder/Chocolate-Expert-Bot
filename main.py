import tkinter as tk
from tkinter import scrolledtext
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.naive_bayes import MultinomialNB
from sklearn.pipeline import make_pipeline

# Training data
data = [
    ("hi", "Hello! Ready to talk chocolate? 🍫"),
    ("hello", "Hey there! Let’s talk about chocolate."),
    ("hey", "Hey! Chocolate time?"),
    ("what is dark chocolate", "Dark chocolate has more cocoa and less sugar. Rich and intense!"),
    ("tell me about milk chocolate", "Milk chocolate is sweet and creamy with milk solids added."),
    ("what's white chocolate", "White chocolate has cocoa butter, sugar, and milk—but no cocoa solids."),
    ("recommend a chocolate", "Try Belgian dark chocolate or Swiss milk chocolate!"),
    ("which chocolate is best", "That depends! For health, go dark. For sweetness, go milk."),
    ("suggest me a chocolate brand", "Lindt, Ghirardelli, and Godiva are all top-tier."),
    ("fun fact about chocolate", "Chocolate was once used as currency by the Aztecs!"),
    ("did you know about chocolate", "Some chocolates have over 70% cocoa—super bitter but healthy!"),
    ("i hate chocolate", "😲 That’s rare! But I respect your taste."),
    ("bye", "Bye! Stay sweet 🍫"),
    ("see you", "See you later, choco-fan!"),
]

# Train model
X = [x[0] for x in data]
y = [x[1] for x in data]
model = make_pipeline(TfidfVectorizer(), MultinomialNB())
model.fit(X, y)

# GUI setup
def send_message():
    user_input = entry.get()
    if user_input.strip() == "":
        return
    chat_log.insert(tk.END, f"You: {user_input}\n")
    if user_input.lower() == "exit":
        response = "See you later, chocostar! 🍫"
    else:
        response = model.predict([user_input])[0]
    chat_log.insert(tk.END, f"Bot: {response}\n\n")
    entry.delete(0, tk.END)

root = tk.Tk()
root.title("🍫 ChocoBot")

chat_log = scrolledtext.ScrolledText(root, width=60, height=20, wrap=tk.WORD)
chat_log.pack(padx=10, pady=10)

entry = tk.Entry(root, width=50)
entry.pack(padx=10, pady=(0, 10), side=tk.LEFT)

send_button = tk.Button(root, text="Send", command=send_message)
send_button.pack(padx=(0, 10), pady=(0, 10), side=tk.LEFT)

root.mainloop()
