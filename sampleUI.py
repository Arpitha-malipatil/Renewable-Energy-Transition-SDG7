import pandas as pd
import numpy as np
import customtkinter as ctk
import matplotlib.pyplot as plt
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
from sklearn.linear_model import LinearRegression
from tkinter import filedialog
import threading

# --- GUI PROTOCOL ---
ctk.set_appearance_mode("Dark")
ctk.set_default_color_theme("blue")

class PantherAnalytics(ctk.CTk):
    def __init__(self):
        super().__init__()

        self.title("PANTHER ANALYTICS v9.0 - SDG 7 MISSION")
        self.geometry("1100x800")
        self.active_df = None

        # Sidebar: System Status & Identity
        self.sidebar = ctk.CTkFrame(self, width=220, corner_radius=0)
        self.sidebar.pack(side="left", fill="y")
        
        self.avatar = ctk.CTkTextbox(self.sidebar, width=190, height=160, font=("Courier", 11), fg_color="transparent")
        self.avatar.insert("0.0", "      _________\n     ________|_        |\n    /         /        |\n   /  _______/    ____/\n  /  /      /    /\n /  /      /____/\n/___________/\n\n [ ANALYTICS CORE ]\n [ STATUS: ACTIVE ]")
        self.avatar.configure(state="disabled")
        self.avatar.pack(pady=40)

        # Main Workspace
        self.main_frame = ctk.CTkFrame(self, fg_color="transparent")
        self.main_frame.pack(padx=20, pady=20, fill="both", expand=True)

        self.setup_ui()

    def setup_ui(self):
        # Data Operations
        self.btn_attach = ctk.CTkButton(self.main_frame, text="ATTACH & HEAL CSV", command=self.data_healer)
        self.btn_attach.pack(pady=10)

        self.btn_run = ctk.CTkButton(self.main_frame, text="EXECUTE TRANSITION MODEL", command=self.launch_model, state="disabled")
        self.btn_run.pack(pady=5)

        # System Log
        self.console = ctk.CTkTextbox(self.main_frame, height=200, width=800, font=("Courier", 13), fg_color="#0a0a0a")
        self.console.pack(pady=10)
        self.log("System initialized. Neural Cognition offline. Analytics Dashboard ready.")

        # Visualization Engine
        self.plot_container = ctk.CTkFrame(self.main_frame, fg_color="#0a0a0a")
        self.plot_container.pack(pady=10, fill="both", expand=True)

    def log(self, text):
        self.console.insert("end", f"[SYSTEM]: {text}\n")
        self.console.see("end")

    def data_healer(self):
        path = filedialog.askopenfilename(filetypes=[("CSV files", "*.csv")])
        if path:
            try:
                df = pd.read_csv(path)
                # Resolve '..' artifacts and enforce Float64
                df = df.replace('..', np.nan)
                for col in df.columns:
                    df[col] = pd.to_numeric(df[col], errors='coerce')
                
                self.active_df = df
                self.log("Dataset sanitized. String artifacts removed. Schema enforced.")
                self.btn_run.configure(state="normal")
            except Exception as e:
                self.log(f"Data Link Failure: {str(e)}")

    def launch_model(self):
        threading.Thread(target=self.run_regression).start()

    def run_regression(self):
        try:
            df = self.active_df.copy()
            target = 'Renewable electricity share of total electricity output (%) [4.1_SHARE.RE.IN.ELECTRICITY]'
            
            # Feature Engineering: 1-Year Lag Implementation
            if 'Subsidies' in df.columns:
                df['Subsidies_Lag1'] = df['Subsidies'].shift(1)
            else:
                self.log("Missing 'Subsidies' column for Lag Analysis.")
                return

            # Selection of Valid Vectors
            df_final = df.dropna(subset=['Subsidies_Lag1', 'Solar_Cost', target])

            if df_final.empty:
                self.log("Insufficient data points for multivariate regression.")
                return

            X = df_final[['Solar_Cost', 'Subsidies_Lag1']]
            y = df_final[target]
            model = LinearRegression().fit(X, y)
            
            self.log(f"Regression Successful. R2 Score: {model.score(X, y):.4f}")
            self.render_graph(df_final, model, target)
        except Exception as e:
            self.log(f"Model Execution Error: {str(e)}")

    def render_graph(self, df, model, target):
        for widget in self.plot_container.winfo_children(): widget.destroy()
        fig, ax = plt.subplots(figsize=(6, 4))
        plt.style.use('dark_background')
        
        ax.plot(df['Time'], df[target], label='Actual Share', marker='o', color='teal')
        ax.plot(df['Time'], model.predict(df[['Solar_Cost', 'Subsidies_Lag1']]), label='Predicted Path', color='magenta', ls='--')
        
        ax.set_title("Renewable Transition Speed (SDG 7)")
        ax.set_xlabel("Year")
        ax.set_ylabel("Share %")
        ax.legend()
        
        canvas = FigureCanvasTkAgg(fig, master=self.plot_container)
        canvas.draw()
        canvas.get_tk_widget().pack(fill="both", expand=True)

if __name__ == "__main__":
    app = PantherAnalytics()
    app.mainloop()