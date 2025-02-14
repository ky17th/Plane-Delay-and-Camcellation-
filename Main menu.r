import tkinter as tk
from tkinter import messagebox

def show_choice(choice):
    messagebox.showinfo("Choice", f"You selected: {choice}")

def on_delays_and_cancellations():
    show_choice("Delays and Cancellations")

def on_delays():
    show_choice("Delays")

def on_cancellations():
    show_choice("Cancellations")

# Create the main window
root = tk.Tk()
root.title("Main Menu Example")

# Create the main menu
menu = tk.Menu(root)
root.config(menu=menu)

# Create the 'Options' menu
options_menu = tk.Menu(menu, tearoff=0)
menu.add_cascade(label="Options", menu=options_menu)
options_menu.add_command(label="Delays and Cancellations", command=on_delays_and_cancellations)
options_menu.add_command(label="Delays", command=on_delays)
options_menu.add_command(label="Cancellations", command=on_cancellations)

# Create a simple label
label = tk.Label(root, text="Select an option from the menu")
label.pack(pady=50)

# Run the application
root.mainloop()
