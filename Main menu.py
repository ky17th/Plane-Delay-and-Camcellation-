import tkinter as tk
from tkinter import messagebox

class Application(tk.Tk):
    def __init__(self):
        super().__init__()
        self.title("Main Menu Example")
        self.create_widgets()
        self.create_menu()

    def create_widgets(self):
        label = tk.Label(self, text="Select an option below")
        label.pack(pady=20)

        button_delays_and_cancellations = tk.Button(self, text="Delays and Cancellations", command=self.on_delays_and_cancellations)
        button_delays_and_cancellations.pack(pady=10)

        button_delays = tk.Button(self, text="Delays", command=self.on_delays)
        button_delays.pack(pady=10)

        button_cancellations = tk.Button(self, text="Cancellations", command=self.on_cancellations)
        button_cancellations.pack(pady=10)

    def create_menu(self):
        menu = tk.Menu(self)
        self.config(menu=menu)

        account_menu = tk.Menu(menu, tearoff=0)
        menu.add_cascade(label="Account", menu=account_menu)
        account_menu.add_command(label="Edit Profile", command=self.on_edit_profile)
        account_menu.add_command(label="Settings", command=self.on_settings)
        account_menu.add_separator()
        account_menu.add_command(label="Logout", command=self.on_logout)

    def show_choice(self, choice):
        messagebox.showinfo("Choice", f"You selected: {choice}")

    def on_delays_and_cancellations(self):
        self.show_choice("Delays and Cancellations")

    def on_delays(self):
        self.show_choice("Delays")

    def on_cancellations(self):
        self.show_choice("Cancellations")

    def on_edit_profile(self):
        messagebox.showinfo("Profile", "Edit Profile selected")

    def on_settings(self):
        messagebox.showinfo("Settings", "Settings selected")

    def on_logout(self):
        messagebox.showinfo("Logout", "Logout selected")

if __name__ == "__main__":
    app = Application()
    app.mainloop()
