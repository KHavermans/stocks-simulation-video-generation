import matplotlib.pyplot as plt
import matplotlib.animation as animation
import pandas as pd
import numpy as np
import yfinance as yf
from datetime import datetime, timedelta
import matplotlib.ticker as mticker
import matplotlib.dates as mdates
import sys
from matplotlib.offsetbox import OffsetImage, AnnotationBbox 
import os 

import tkinter as tk
from tkinter import filedialog, messagebox
import threading
import webbrowser

# Configure Matplotlib
plt.rcParams['text.usetex'] = False 

# --- Functies uit finance.py ---
def get_company_info(ticker):
    original_stdout = sys.stdout
    sys.stdout = open(os.devnull, 'w')
    try:
        stock = yf.Ticker(ticker)
        info = stock.info
        
        company_name = info.get('longName') or info.get('shortName') or ticker

        hist = stock.history(period="max", interval="1d")
        earliest_date = None
        if not hist.empty:
            earliest_date = hist.index.min().strftime('%Y-%m-%d')
            
        return company_name, earliest_date
    except Exception as e:
        sys.stdout.close()
        sys.stdout = original_stdout
        raise ValueError(f"Could not fetch info for {ticker}: {e}")
    finally:
        sys.stdout.close() 
        sys.stdout = original_stdout


def get_investment_data(ticker, start_date_str, daily_investment_amount=1):
    original_stdout = sys.stdout
    sys.stdout = open(os.devnull, 'w')
    try:
        ticker_data = yf.download(ticker, start=start_date_str, end=datetime.now())
    finally:
        sys.stdout.close() 
        sys.stdout = original_stdout

    if ticker_data.empty:
        raise ValueError(f"Could not download data for {ticker} from {start_date_str} to today.")

    if 'Adj Close' in ticker_data.columns:
        close_prices = ticker_data['Adj Close']
    elif 'Close' in ticker_data.columns:
        close_prices = ticker_data['Close']
    else:
        raise KeyError("Neither 'Adj Close' nor 'Close' column found in downloaded data.")

    if 'Dividends' in ticker_data.columns:
        dividends = ticker_data['Dividends'].fillna(0)
    else:
        dividends = pd.Series(0, index=ticker_data.index)

    df = pd.DataFrame(index=pd.to_datetime(ticker_data.index))
    df['Close'] = close_prices
    df['Dividends'] = dividends
    df['DailyInvestment'] = daily_investment_amount
    df['TotalInvested'] = df['DailyInvestment'].cumsum()
    df['SharesBoughtToday'] = df['DailyInvestment'] / df['Close']
    df['TotalShares'] = df['SharesBoughtToday'].cumsum()
    df['PortfolioValue'] = df['TotalShares'] * df['Close']
    return df

def create_animated_chart_video(df, company_name, start_year_str, daily_investment_amount, 
                                output_filename="investment_simulation_final.mp4", 
                                logo_path=None, gui_progress_callback=None,
                                target_duration_seconds=61, target_fps=24, crf_value=18):
    
    BACKGROUND_COLOR = '#000000'
    LINE_INVESTED_COLOR = '#FFFFFF'
    LINE_PORTFOLIO_COLOR = '#00FF00'
    TEXT_COLOR_WHITE = '#FFFFFF'
    PROFIT_COLOR = '#00FF00'
    LOSS_COLOR = '#CC0000'

    fig, ax = plt.subplots(figsize=(9, 16), dpi=150)
    fig.patch.set_facecolor(BACKGROUND_COLOR)
    ax.set_facecolor(BACKGROUND_COLOR)

    ax.tick_params(axis='x', colors=TEXT_COLOR_WHITE, labelsize=14, labelcolor=TEXT_COLOR_WHITE, direction='out', length=6, width=1.5, pad=10)           
    ax.tick_params(axis='y', colors=TEXT_COLOR_WHITE, labelsize=14, labelcolor=TEXT_COLOR_WHITE, direction='out', length=6, width=1.5, pad=10)           
    
    ax.spines['bottom'].set_color(TEXT_COLOR_WHITE)
    ax.spines['bottom'].set_linewidth(2)
    ax.spines['top'].set_color(TEXT_COLOR_WHITE)
    ax.spines['top'].set_linewidth(2)
    ax.spines['left'].set_color(TEXT_COLOR_WHITE)
    ax.spines['left'].set_linewidth(2)
    ax.spines['right'].set_color(TEXT_COLOR_WHITE)
    ax.spines['right'].set_linewidth(2)

    line_invested, = ax.plot([], [], color=LINE_INVESTED_COLOR, linewidth=4.0, label='Invested') 
    line_portfolio, = ax.plot([], [], color=LINE_PORTFOLIO_COLOR, linewidth=3.5, label='Portfolio Value')

    pov_title_text_str = (
        f'POV: You invested just ${daily_investment_amount} a day\n'
        f'in {company_name.upper()} since {start_year_str}\n'
        f'and never sold.'
    )
    
    pov_title = fig.text(0.5, 0.94, 
                         pov_title_text_str,
                         color=TEXT_COLOR_WHITE, fontsize=24, ha='center', va='top', wrap=True,
                         fontweight='bold') 
    
    text_date = ax.text(0.98, 0.98, '', transform=ax.transAxes, color=TEXT_COLOR_WHITE, fontsize=16, ha='right', va='top', fontweight='bold')
    text_invested = ax.text(0.98, 0.94, '', transform=ax.transAxes, color=TEXT_COLOR_WHITE, fontsize=16, ha='right', va='top', fontweight='bold')
    text_company_value = ax.text(0.98, 0.90, '', transform=ax.transAxes, color=TEXT_COLOR_WHITE, fontsize=16, ha='right', va='top', fontweight='bold')

    ax.set_xlabel('Date', color=TEXT_COLOR_WHITE, fontsize=14, fontweight='bold')
    ax.set_ylabel('Price In $', color=TEXT_COLOR_WHITE, fontsize=16, fontweight='bold') 

    ax.legend(facecolor=BACKGROUND_COLOR, edgecolor=TEXT_COLOR_WHITE, labelcolor=TEXT_COLOR_WHITE, loc='upper left', fontsize=12,
              prop={'weight': 'bold'})

    plt.tight_layout(rect=[0.08, 0.05, 0.95, 0.86], pad=2.0) 

    ax.yaxis.set_major_locator(mticker.MaxNLocator(nbins=10))
    ax.yaxis.set_major_formatter(mticker.FormatStrFormatter('$%.0f'))

    ax.xaxis.set_major_formatter(mdates.DateFormatter('%Y'))
    ax.xaxis.set_major_locator(mdates.YearLocator(base=1)) 
    ax.xaxis.set_minor_locator(mdates.YearLocator(base=1))
    
    if logo_path and os.path.exists(logo_path):
        try:
            logo_image = plt.imread(logo_path)
            imagebox = OffsetImage(logo_image, zoom=0.25)
            ab = AnnotationBbox(imagebox, (0.5, 0.98), xycoords='figure fraction', boxcoords='figure fraction',
                                box_alignment=(0.5, 1.0), 
                                frameon=False, pad=0.0) 
            ax.add_artist(ab)
        except Exception as e:
            if gui_progress_callback:
                gui_progress_callback(0, 1, status=f"Error adding logo: {e}. Please ensure '{logo_path}' is a valid PNG file.")
            print(f"Error adding logo: {e}. Please ensure '{logo_path}' is a valid PNG file.")
    elif gui_progress_callback:
        gui_progress_callback(0, 1, status=f"Warning: Logo file '{logo_path}' not found or not specified. Skipping logo addition.")
    else:
        print(f"Warning: Logo file '{logo_path}' not found or not specified. Skipping logo addition.")

    def init():
        line_invested.set_data([], [])
        line_portfolio.set_data([], [])
        text_date.set_text('')
        text_invested.set_text('')
        text_company_value.set_text('')
        return line_invested, line_portfolio, text_date, text_invested, text_company_value

    def animate(i):
        data_index = int(np.interp(i, [0, desired_frames_count - 1], [0, total_data_points - 1]))
        current_df = df.iloc[:data_index+1]

        x_data = current_df.index
        y_invested = current_df['TotalInvested']
        y_portfolio = current_df['PortfolioValue']

        line_invested.set_data(x_data, y_invested)
        line_portfolio.set_data(x_data, y_portfolio)

        ax.set_xlim(df.index[0], current_df.index[-1] + timedelta(days=365*0.1)) 
        max_y_value = max(y_invested.max(), y_portfolio.max())
        ax.set_ylim(0, max_y_value * 1.30)

        if not current_df.empty:
            last_date = current_df.index[-1].strftime('%Y-%m-%d')
            current_invested = current_df['TotalInvested'].iloc[-1]
            current_portfolio_value = current_df['PortfolioValue'].iloc[-1]
            profit_loss = current_portfolio_value - current_invested

            value_amount_color = PROFIT_COLOR if profit_loss >= 0 else LOSS_COLOR
            
            text_date.set_text(f'Date: {last_date}')
            text_invested.set_text(f'Invested: ${current_invested:,.2f}')
            text_company_value.set_text(f'{company_name.upper()}: ${current_portfolio_value:,.2f}')
            text_company_value.set_color(value_amount_color)

        fig.canvas.draw()
        return line_invested, line_portfolio, text_date, text_invested, text_company_value

    total_data_points = df.shape[0]
    desired_frames_count = target_duration_seconds * target_fps

    def ffmpeg_progress_callback(current_frame, total_frames_ffmpeg):
        if gui_progress_callback:
            gui_progress_callback(current_frame, desired_frames_count)
    
    ani = animation.FuncAnimation(
        fig, animate, frames=desired_frames_count,
        init_func=init, blit=True, interval=1000/target_fps
    )

    try:
        ani.save(output_filename, writer='ffmpeg', fps=target_fps, 
                 extra_args=['-vcodec', 'libx264', '-crf', str(crf_value), '-threads', '0'],
                 progress_callback=ffmpeg_progress_callback)
        if gui_progress_callback:
            gui_progress_callback(desired_frames_count, desired_frames_count, status="Video generation complete!")
    except Exception as e:
        if gui_progress_callback:
            gui_progress_callback(0, 1, status=f"Error during video rendering: {e}")
        raise e 

# --- Klasse uit financegui.py ---
class FinanceApp:
    def __init__(self, master):
        self.master = master
        master.title("AI Finance Video Generator")
        master.geometry("600x700")
        master.resizable(False, False)

        master.configure(bg='#282c34')
        font_label = ('Arial', 12, 'bold')
        font_entry = ('Arial', 12)
        font_button = ('Arial', 12, 'bold')
        text_color = '#FFFFFF'

        input_frame = tk.LabelFrame(master, text="Video Parameters", bg='#3a404c', fg=text_color, font=font_label, padx=20, pady=20)
        input_frame.pack(pady=20, padx=20, fill="x")

        self.ticker_label = tk.Label(input_frame, text="Company Ticker (e.g., RHM.DE, MSFT):", bg='#3a404c', fg=text_color, font=font_label)
        self.ticker_label.grid(row=0, column=0, sticky="w", pady=5)
        self.ticker_label.bind("<Button-1>", self.open_ticker_lookup)
        self.ticker_label.config(cursor="hand2")

        self.ticker_entry = tk.Entry(input_frame, width=30, font=font_entry, bg='#4c515d', fg=text_color, insertbackground=text_color)
        self.ticker_entry.grid(row=0, column=1, sticky="ew", pady=5, padx=5)
        self.ticker_entry.insert(0, "RHM.DE")
        self.ticker_entry.bind("<FocusOut>", self.update_company_info_from_ticker)
        self.ticker_entry.bind("<Return>", self.update_company_info_from_ticker)

        tk.Label(input_frame, text="Company Name:", bg='#3a404c', fg=text_color, font=font_label).grid(row=1, column=0, sticky="w", pady=5)
        self.company_name_entry = tk.Entry(input_frame, width=30, font=font_entry, bg='#4c515d', fg=text_color, insertbackground=text_color, state='readonly')
        self.company_name_entry.grid(row=1, column=1, sticky="ew", pady=5, padx=5)
        self.master.after(100, self.update_company_info_from_ticker, None)

        tk.Label(input_frame, text="Daily Investment Amount ($):", bg='#3a404c', fg=text_color, font=font_label).grid(row=2, column=0, sticky="w", pady=5)
        self.investment_entry = tk.Entry(input_frame, width=30, font=font_entry, bg='#4c515d', fg=text_color, insertbackground=text_color)
        self.investment_entry.grid(row=2, column=1, sticky="ew", pady=5, padx=5)
        self.investment_entry.insert(0, "1")

        tk.Label(input_frame, text="Data Starts From:", bg='#3a404c', fg=text_color, font=font_label).grid(row=3, column=0, sticky="w", pady=5)
        self.display_start_year_label = tk.Label(input_frame, text="N/A", bg='#3a404c', fg=text_color, font=font_entry, anchor="w")
        self.display_start_year_label.grid(row=3, column=1, sticky="ew", pady=5, padx=5)

        tk.Label(input_frame, text="Video Duration (seconds):", bg='#3a404c', fg=text_color, font=font_label).grid(row=4, column=0, sticky="w", pady=5)
        self.duration_entry = tk.Entry(input_frame, width=30, font=font_entry, bg='#4c515d', fg=text_color, insertbackground=text_color)
        self.duration_entry.grid(row=4, column=1, sticky="ew", pady=5, padx=5)
        self.duration_entry.insert(0, "61")

        tk.Label(input_frame, text="Frames Per Second (FPS):", bg='#3a404c', fg=text_color, font=font_label).grid(row=5, column=0, sticky="w", pady=5)
        self.fps_entry = tk.Entry(input_frame, width=30, font=font_entry, bg='#4c515d', fg=text_color, insertbackground=text_color)
        self.fps_entry.grid(row=5, column=1, sticky="ew", pady=5, padx=5)
        self.fps_entry.insert(0, "24")

        tk.Label(input_frame, text="Video Quality (CRF, 0-51, lower=better):", bg='#3a404c', fg=text_color, font=font_label).grid(row=6, column=0, sticky="w", pady=5)
        self.crf_entry = tk.Entry(input_frame, width=30, font=font_entry, bg='#4c515d', fg=text_color, insertbackground=text_color)
        self.crf_entry.grid(row=6, column=1, sticky="ew", pady=5, padx=5)
        self.crf_entry.insert(0, "18")

        tk.Label(input_frame, text="Company Logo (PNG):", bg='#3a404c', fg=text_color, font=font_label).grid(row=7, column=0, sticky="w", pady=5)
        self.logo_path_entry = tk.Entry(input_frame, width=30, font=font_entry, bg='#4c515d', fg=text_color, insertbackground=text_color, state='readonly')
        self.logo_path_entry.grid(row=7, column=1, sticky="ew", pady=5, padx=5)
        self.logo_path_entry.bind("<Button-1>", lambda e: self.browse_logo())
        self.browse_logo_button = tk.Button(input_frame, text="Browse", command=self.browse_logo, font=font_button, bg='#6a737d', fg=text_color)
        self.browse_logo_button.grid(row=7, column=2, sticky="w", pady=5, padx=5)
        self.logo_path = None

        self.generate_button = tk.Button(master, text="Generate Video", command=self.start_video_generation, font=font_button, bg='#4CAF50', fg=text_color, activebackground='#45a049')
        self.generate_button.pack(pady=20, ipadx=10, ipady=5)

        status_frame = tk.LabelFrame(master, text="Status", bg='#3a404c', fg=text_color, font=font_label, padx=20, pady=10)
        status_frame.pack(pady=10, padx=20, fill="x")

        self.status_label = tk.Label(status_frame, text="Ready.", bg='#3a404c', fg=text_color, font=font_entry, wraplength=500, justify="left")
        self.status_label.pack(fill="x", pady=5)
        
        self.progress_label = tk.Label(status_frame, text="0%", bg='#3a404c', fg=text_color, font=font_entry)
        self.progress_label.pack(pady=5)
        
        self.earliest_data_date = None

    def open_ticker_lookup(self, event=None):
        webbrowser.open_new("https://finance.yahoo.com/")

    def update_company_info_from_ticker(self, event=None):
        ticker = self.ticker_entry.get().strip()
        if not ticker:
            self.company_name_entry.config(state='normal')
            self.company_name_entry.delete(0, tk.END)
            self.company_name_entry.config(state='readonly')
            self.display_start_year_label.config(text="N/A")
            self.earliest_data_date = None
            return

        def fetch_info():
            try:
                company_name, earliest_date = get_company_info(ticker)
                self.master.after(0, self._update_gui_company_info, company_name, earliest_date)
            except ValueError as e:
                self.master.after(0, self._update_gui_company_info_error, str(e))
            except Exception as e:
                self.master.after(0, self._update_gui_company_info_error, f"An unexpected error occurred: {e}")

        threading.Thread(target=fetch_info).start()

    def _update_gui_company_info(self, company_name, earliest_date):
        self.company_name_entry.config(state='normal')
        self.company_name_entry.delete(0, tk.END)
        self.company_name_entry.insert(0, company_name)
        self.company_name_entry.config(state='readonly')
        
        self.earliest_data_date = earliest_date
        self.display_start_year_label.config(text=earliest_date if earliest_date else "N/A")
        self.update_status(0, 0, "Ready.")

    def _update_gui_company_info_error(self, error_message):
        self.company_name_entry.config(state='normal')
        self.company_name_entry.delete(0, tk.END)
        self.company_name_entry.insert(0, "Error fetching name")
        self.company_name_entry.config(state='readonly')
        self.display_start_year_label.config(text="N/A")
        self.earliest_data_date = None
        self.update_status(0, 0, f"Error: {error_message}")


    def browse_logo(self):
        file_path = filedialog.askopenfilename(
            title="Select Company Logo",
            filetypes=[("PNG files", "*.png"), ("All files", "*.*")]
        )
        if file_path:
            self.logo_path = file_path
            self.logo_path_entry.config(state='normal')
            self.logo_path_entry.delete(0, tk.END)
            self.logo_path_entry.insert(0, os.path.basename(file_path))
            self.logo_path_entry.config(state='readonly')

    def update_status(self, current_frame, total_frames, status=None):
        if status:
            self.status_label.config(text=status)
        
        if total_frames > 0:
            percent = (current_frame / total_frames) * 100
            self.progress_label.config(text=f"{percent:.2f}% ({current_frame}/{total_frames})")
        else:
            self.progress_label.config(text="0%")

    def start_video_generation(self):
        self.generate_button.config(state=tk.DISABLED)
        self.update_status(0, 0, "Initializing...")
        self.progress_label.config(text="0%")

        ticker = self.ticker_entry.get().strip()
        company_name = self.company_name_entry.get().strip()
        daily_investment_str = self.investment_entry.get().strip()
        
        duration_str = self.duration_entry.get().strip()
        fps_str = self.fps_entry.get().strip()
        crf_str = self.crf_entry.get().strip()

        if not ticker or not company_name or not daily_investment_str or \
           not duration_str or not fps_str or not crf_str:
            self.update_status(0, 0, "Error: All main fields must be filled.")
            self.generate_button.config(state=tk.NORMAL)
            return
        
        if not self.earliest_data_date:
            self.update_status(0, 0, "Error: Could not determine data start year for ticker. Please try again.")
            self.generate_button.config(state=tk.NORMAL)
            return

        try:
            daily_investment = float(daily_investment_str)
            if daily_investment <= 0: raise ValueError("Investment amount must be positive.")
            duration = int(duration_str)
            if duration <= 0: raise ValueError("Duration must be positive.")
            fps = int(fps_str)
            if fps <= 0: raise ValueError("FPS must be positive.")
            crf = int(crf_str)
            if not (0 <= crf <= 51): raise ValueError("CRF must be between 0 and 51.")
        except ValueError as e:
            self.update_status(0, 0, f"Input Error: {e}")
            self.generate_button.config(state=tk.NORMAL)
            return

        start_year_from_data = self.earliest_data_date.split('-')[0]

        self.video_thread = threading.Thread(target=self.run_video_generation, 
                                             args=(ticker, company_name, daily_investment, 
                                                   start_year_from_data, duration, fps, crf, self.logo_path))
        self.video_thread.start()

    def run_video_generation(self, ticker, company_name, daily_investment, start_year, duration, fps, crf, logo_path):
        try:
            self.master.after(0, self.update_status, 0, 0, "Fetching data...")
            investment_df = get_investment_data(ticker, f"{start_year}-01-01", daily_investment)

            if len(investment_df) < 2:
                self.master.after(0, self.update_status, 0, 0, "Error: Not enough data for the selected period.")
                return

            output_filename = f"{company_name.replace(' ', '_').lower()}_investment_simulation.mp4"

            self.master.after(0, self.update_status, 0, 0, f"Generating video: {output_filename}...")
            
            create_animated_chart_video(
                investment_df, 
                company_name, 
                str(start_year), 
                daily_investment, 
                output_filename, 
                logo_path=logo_path,
                gui_progress_callback=self.update_status,
                target_duration_seconds=duration,
                target_fps=fps,
                crf_value=crf
            )

            self.master.after(0, self.update_status, 1, 1, f"Video '{output_filename}' generated successfully!")
            messagebox.showinfo("Success", f"Video '{output_filename}' generated successfully!")

        except ValueError as e:
            self.master.after(0, self.update_status, 0, 0, f"Error: {e}")
            messagebox.showerror("Error", f"An error occurred: {e}")
        except KeyError as e:
            self.master.after(0, self.update_status, 0, 0, f"Data Error: {e}. Check ticker or data source.")
            messagebox.showerror("Error", f"An error occurred: {e}. Check ticker or data source.")
        except Exception as e:
            self.master.after(0, self.update_status, 0, 0, f"An unexpected error occurred: {e}")
            messagebox.showerror("Unexpected Error", f"An unexpected error occurred: {e}")
        finally:
            # Gebruik een lambda om de functie met het keyword argument correct aan te roepen
            self.master.after(0, lambda: self.generate_button.config(state=tk.NORMAL))

if __name__ == "__main__":
    root = tk.Tk()
    app = FinanceApp(root)
    root.mainloop()