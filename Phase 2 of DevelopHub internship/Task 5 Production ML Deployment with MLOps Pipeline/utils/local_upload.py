import tkinter as tk
from tkinter import filedialog
import pandas as pd
import os

def upload_csv():
    """Upload and read a CSV file from local PC."""
    root = tk.Tk()
    root.withdraw()  # Hide the main window
    file_path = filedialog.askopenfilename(
        title="Select CSV File",
        filetypes=[("CSV files", "*.csv")]
    )
    if file_path:
        print(f"Selected: {file_path}")
        df = pd.read_csv(file_path)
        print("CSV loaded successfully!")
        print(df.head())
        return df
    return None

def upload_txt():
    """Upload and read a TXT file from local PC."""
    root = tk.Tk()
    root.withdraw()
    file_path = filedialog.askopenfilename(
        title="Select TXT File",
        filetypes=[("Text files", "*.txt")]
    )
    if file_path:
        print(f"Selected: {file_path}")
        with open(file_path, 'r') as f:
            content = f.read()
        print("TXT loaded successfully!")
        print(content[:100] + "...") # Print first 100 chars
        return content
    return None

def upload_image():
    """Upload an image file from local PC."""
    root = tk.Tk()
    root.withdraw()
    file_path = filedialog.askopenfilename(
        title="Select Image File",
        filetypes=[("Image files", "*.jpg *.jpeg *.png *.bmp")]
    )
    if file_path:
        print(f"Selected: {file_path}")
        # In a real app, you might use PIL to open it
        # from PIL import Image
        # img = Image.open(file_path)
        return file_path
    return None

def upload_ipynb():
    """Upload an IPYNB file from local PC."""
    root = tk.Tk()
    root.withdraw()
    file_path = filedialog.askopenfilename(
        title="Select Jupyter Notebook",
        filetypes=[("IPYNB files", "*.ipynb")]
    )
    if file_path:
        print(f"Selected: {file_path}")
        import json
        with open(file_path, 'r', encoding='utf-8') as f:
            nb_content = json.load(f)
        print("Notebook loaded successfully!")
        return nb_content
    return None

if __name__ == "__main__":
    print("Choose an option to upload:")
    print("1. CSV")
    print("2. TXT")
    print("3. Image")
    print("4. IPYNB")
    
    choice = input("Enter choice (1-4): ")
    if choice == '1':
        upload_csv()
    elif choice == '2':
        upload_txt()
    elif choice == '3':
        upload_image()
    elif choice == '4':
        upload_ipynb()
    else:
        print("Invalid choice")
