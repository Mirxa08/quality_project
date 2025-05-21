import os
import fitz  # PyMuPDF
import json

# --- Configuration ---
PDF_DIR = "pdfs"
OUTPUT_DIR = "Text"
META_FILE = os.path.join(OUTPUT_DIR, "metadata.json")

# Create output directory if it doesn't exist
os.makedirs(OUTPUT_DIR, exist_ok=True)

def extract_text_and_metadata(pdf_path):
    doc = fitz.open(pdf_path)
    full_text = ""
    for page in doc:
        full_text += page.get_text()

    title_guess = doc[0].get_text("text").split("\n")[0].strip()  # First line of first page
    metadata = {
        "file_name": os.path.basename(pdf_path),
        "title": title_guess,
        "num_pages": len(doc),
        "text_length": len(full_text),
    }
    return full_text, metadata

def process_all_pdfs(pdf_dir, output_dir):
    all_metadata = []

    for filename in os.listdir(pdf_dir):
        if filename.lower().endswith(".pdf"):
            pdf_path = os.path.join(pdf_dir, filename)
            text, metadata = extract_text_and_metadata(pdf_path)

            txt_filename = os.path.splitext(filename)[0] + ".txt"
            txt_path = os.path.join(output_dir, txt_filename)

            with open(txt_path, "w", encoding="utf-8") as f:
                f.write(text)

            metadata["output_txt_file"] = txt_filename
            all_metadata.append(metadata)

            print(f"[✔] Extracted: {filename} → {txt_filename}")

    # Save metadata
    with open(META_FILE, "w", encoding="utf-8") as f:
        json.dump(all_metadata, f, indent=2)

    print(f"✅ Metadata written to: {META_FILE}")

if __name__ == "__main__":
    process_all_pdfs(PDF_DIR, OUTPUT_DIR)
