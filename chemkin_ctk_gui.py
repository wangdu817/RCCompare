import customtkinter as ctk

class App(ctk.CTk):
    def __init__(self):
        super().__init__()

        self.title("CHEMKIN Rate Viewer (CustomTkinter)")
        self.geometry("1000x700")

        ctk.set_appearance_mode("System")  # Modes: "System" (default), "Dark", "Light"
        ctk.set_default_color_theme("blue")  # Themes: "blue" (default), "green", "dark-blue"

        self.label = ctk.CTkLabel(self, text="CustomTkinter Environment Setup Test")
        self.label.pack(pady=20, padx=20)

if __name__ == "__main__":
    app = App()
    app.mainloop()
