import tkinter as tk
from tkinter import ttk
from matplotlib.figure import Figure
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
import numpy as np
import Prediction

class Visualization:
    def __init__(self, root):
        self.root = root
    
    def create_toplevel(self, title):
        visualization_window = tk.Toplevel(self.root)
        visualization_window.title(title)
        return visualization_window

    def visualize_bar_chart(self):
        visualization_window = self.create_toplevel("Bar Chart Visualization")
        categories = ['Delay', 'No Delay']
        values = [30, 70]  # Example data
        fig = Figure(figsize=(6, 4), dpi=100)
        ax = fig.add_subplot(111)
        ax.bar(categories, values, color=['blue', 'green'])
        ax.set_ylabel('Count')
        ax.set_title('Bar Chart of Delays')
        canvas = FigureCanvasTkAgg(fig, master=visualization_window)
        canvas.draw()
        canvas.get_tk_widget().pack()

    def visualize_pie_chart(self):
        visualization_window = self.create_toplevel("Pie Chart Visualization")
        labels = ['Delayed', 'On Time']
        sizes = [15, 85]  # Example data
        fig = Figure(figsize=(6, 4), dpi=100)
        ax = fig.add_subplot(111)
        ax.pie(sizes, labels=labels, autopct='%1.1f%%', startangle=90)
        ax.axis('equal')
        ax.set_title('Pie Chart of Delays')
        canvas = FigureCanvasTkAgg(fig, master=visualization_window)
        canvas.draw()
        canvas.get_tk_widget().pack()

    def visualize_line_chart(self):
        visualization_window = self.create_toplevel("Line Chart Visualization")
        times = list(range(24))  # 0 to 23 hours
        delays = [5, 3, 4, 7, 8, 2, 3, 4, 6, 7, 5, 4, 3, 2, 4, 5, 7, 6, 4, 3, 5, 6, 4, 3]  # Example data
        fig = Figure(figsize=(6, 4), dpi=100)
        ax = fig.add_subplot(111)
        ax.plot(times, delays, marker='o', linestyle='-', color='blue')
        ax.set_xlabel('Hour of Day')
        ax.set_ylabel('Number of Delays')
        ax.set_title('Line Chart of Delays Throughout the Day')
        canvas = FigureCanvasTkAgg(fig, master=visualization_window)
        canvas.draw()
        canvas.get_tk_widget().pack()

    def visualize_histogram(self):
        visualization_window = self.create_toplevel("Histogram Visualization")
        delays = [5, 3, 4, 7, 8, 2, 3, 4, 6, 7, 5, 4, 3, 2, 4, 5, 7, 6, 4, 3, 5, 6, 4, 3]  # Example data
        fig = Figure(figsize=(6, 4), dpi=100)
        ax = fig.add_subplot(111)
        ax.hist(delays, bins=10, color='blue', edgecolor='black')
        ax.set_xlabel('Delay Time (minutes)')
        ax.set_ylabel('Frequency')
        ax.set_title('Histogram of Delay Times')
        canvas = FigureCanvasTkAgg(fig, master=visualization_window)
        canvas.draw()
        canvas.get_tk_widget().pack()

    def visualize_box_plot(self):
        visualization_window = self.create_toplevel("Box Plot Visualization")
        delays = [5, 3, 4, 7, 8, 2, 3, 4, 6, 7, 5, 4, 3, 2, 4, 5, 7, 6, 4, 3, 5, 6, 4, 3]  # Example data
        fig = Figure(figsize=(6, 4), dpi=100)
        ax = fig.add_subplot(111)
        ax.boxplot(delays)
        ax.set_ylabel('Delay Time (minutes)')
        ax.set_title('Box Plot of Delay Times')
        canvas = FigureCanvasTkAgg(fig, master=visualization_window)
        canvas.draw()
        canvas.get_tk_widget().pack()

    def visualize_scatter_plot(self):
        visualization_window = self.create_toplevel("Scatter Plot Visualization")
        distances = [100, 200, 300, 400, 500, 600, 700, 800, 900, 1000]
        delays = [5, 10, 3, 8, 6, 4, 7, 9, 5, 8]  # Example data
        fig = Figure(figsize=(6, 4), dpi=100)
        ax = fig.add_subplot(111)
        ax.scatter(distances, delays, color='blue')
        ax.set_xlabel('Distance (miles)')
        ax.set_ylabel('Delay Time (minutes)')
        ax.set_title('Scatter Plot of Distance vs. Delay Time')
        canvas = FigureCanvasTkAgg(fig, master=visualization_window)
        canvas.draw()
        canvas.get_tk_widget().pack()

    def visualize_heatmap(self):
        visualization_window = self.create_toplevel("Heatmap Visualization")
        data = np.random.rand(10, 10)  # Random data for example
        fig = Figure(figsize=(6, 4), dpi=100)
        ax = fig.add_subplot(111)
        cax = ax.imshow(data, interpolation='nearest', cmap='hot')
        fig.colorbar(cax)
        ax.set_title('Heatmap of Random Data')
        canvas = FigureCanvasTkAgg(fig, master=visualization_window)
        canvas.draw()
        canvas.get_tk_widget().pack()

    def visualize_violin_plot(self):
        visualization_window = self.create_toplevel("Violin Plot Visualization")
        data = [np.random.normal(0, std, 100) for std in range(1, 4)]  # Random data for example
        fig = Figure(figsize=(6, 4), dpi=100)
        ax = fig.add_subplot(111)
        ax.violinplot(data)
        ax.set_title('Violin Plot of Random Data')
        canvas = FigureCanvasTkAgg(fig, master=visualization_window)
        canvas.draw()
        canvas.get_tk_widget().pack()

    def visualize_stacked_bar_chart(self):
        visualization_window = self.create_toplevel("Stacked Bar Chart Visualization")
        categories = ['Category 1', 'Category 2', 'Category 3']
        sub_category1 = [5, 3, 4]
        sub_category2 = [2, 5, 6]
        fig = Figure(figsize=(6, 4), dpi=100)
        ax = fig.add_subplot(111)
        ax.bar(categories, sub_category1, label='Sub-category 1')
        ax.bar(categories, sub_category2, bottom=sub_category1, label='Sub-category 2')
        ax.set_ylabel('Values')
        ax.set_title('Stacked Bar Chart')
        ax.legend()
        canvas = FigureCanvasTkAgg(fig, master=visualization_window)
        canvas.draw()
        canvas.get_tk_widget().pack()

def open_visualization_selector():
    selector_window = tk.Toplevel(root)
    selector_window.title("Select Visualization")

    ttk.Button(selector_window, text="Bar Chart", command=visualization.visualize_bar_chart).pack(fill=tk.X, padx=10, pady=5)
    ttk.Button(selector_window, text="Pie Chart", command=visualization.visualize_pie_chart).pack(fill=tk.X, padx=10, pady=5)
    ttk.Button(selector_window, text="Line Chart", command=visualization.visualize_line_chart).pack(fill=tk.X, padx=10, pady=5)
    ttk.Button(selector_window, text="Histogram", command=visualization.visualize_histogram).pack(fill=tk.X, padx=10, pady=5)
    ttk.Button(selector_window, text="Box Plot", command=visualization.visualize_box_plot).pack(fill=tk.X, padx=10, pady=5)
    ttk.Button(selector_window, text="Scatter Plot", command=visualization.visualize_scatter_plot).pack(fill=tk.X, padx=10, pady=5)
    ttk.Button(selector_window, text="Heatmap", command=visualization.visualize_heatmap).pack(fill=tk.X, padx=10, pady=5)
    ttk.Button(selector_window, text="Violin Plot", command=visualization.visualize_violin_plot).pack(fill=tk.X, padx=10, pady=5)
    ttk.Button(selector_window, text="Stacked Bar Chart", command=visualization.visualize_stacked_bar_chart).pack(fill=tk.X, padx=10, pady=5)

# Create the main window
root = tk.Tk()
root.title("Flight Delay and Cancellation Prediction")

visualization = Visualization(root)

# Create and place the widgets
ttk.Label(root, text="Airline").grid(row=0, column=0)
airline_entry = ttk.Entry(root)
airline_entry.grid(row=0, column=1)

ttk.Label(root, text="Origin").grid(row=1, column=0)
origin_entry = ttk.Entry(root)
origin_entry.grid(row=1, column=1)

ttk.Label(root, text="Destination").grid(row=2, column=0)
dest_entry = ttk.Entry(root)
dest_entry.grid(row=2, column=1)

ttk.Label(root, text="CRS Departure Time").grid(row=3, column=0)
crs_dep_time_entry = ttk.Entry(root)
crs_dep_time_entry.grid(row=3, column=1)

ttk.Label(root, text="Distance").grid(row=4, column=0)
distance_entry = ttk.Entry(root)
distance_entry.grid(row=4, column=1)

delay_result = tk.StringVar()
ttk.Label(root, textvariable=delay_result).grid(row=7, column=0, columnspan=2)

cancel_result = tk.StringVar()
ttk.Label(root, textvariable=cancel_result).grid(row=8, column=0, columnspan=2)

# Run the application
root.mainloop()
