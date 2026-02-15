import ipywidgets as widgets
from IPython.display import display
import io
import pandas as pd
from PIL import Image
import json

def handle_csv_upload(change):
    uploaded_file = list(uploader_csv.value.values())[0]
    content = uploaded_file['content']
    df = pd.read_csv(io.BytesIO(content))
    print("CSV Uploaded Successfully!")
    display(df.head())

def handle_txt_upload(change):
    uploaded_file = list(uploader_txt.value.values())[0]
    content = uploaded_file['content'].decode('utf-8')
    print("TXT Uploaded Successfully!")
    print(content[:200])

def handle_img_upload(change):
    uploaded_file = list(uploader_img.value.values())[0]
    content = uploaded_file['content']
    img = Image.open(io.BytesIO(content))
    print("Image Uploaded Successfully!")
    display(img.resize((200, 200)))

def handle_ipynb_upload(change):
    uploaded_file = list(uploader_ipynb.value.values())[0]
    content = uploaded_file['content'].decode('utf-8')
    nb_json = json.loads(content)
    print("IPYNB Uploaded Successfully!")
    print(f"Number of cells: {len(nb_json['cells'])}")

# Create Uploaders
uploader_csv = widgets.FileUpload(accept='.csv', multiple=False)
uploader_txt = widgets.FileUpload(accept='.txt', multiple=False)
uploader_img = widgets.FileUpload(accept='.jpg,.png,.jpeg', multiple=False)
uploader_ipynb = widgets.FileUpload(accept='.ipynb', multiple=False)

# Observe changes
uploader_csv.observe(handle_csv_upload, names='value')
uploader_txt.observe(handle_txt_upload, names='value')
uploader_img.observe(handle_img_upload, names='value')
uploader_ipynb.observe(handle_ipynb_upload, names='value')

print("Upload CSV:")
display(uploader_csv)

print("\nUpload TXT:")
display(uploader_txt)

print("\nUpload Image:")
display(uploader_img)

print("\nUpload IPYNB:")
display(uploader_ipynb)
