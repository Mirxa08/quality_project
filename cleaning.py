import os
import re

# --- Configuration ---
INPUT_DIR = "./Text"
OUTPUT_DIR = "./Clean"
os.makedirs(OUTPUT_DIR, exist_ok=True)

def clean_text(text):
    # Remove page numbers like "Page 1 of 18"
    text = re.sub(r'Page\s+\d+\s+of\s+\d+', '', text)

    # Remove lines with repeated SOP titles/headers
    text = re.sub(r'Title of Policy/ SOP:.*?\n', '', text)
    text = re.sub(r'Policy/ SOP Code:.*?\n', '', text)
    text = re.sub(r'Department:.*?\n', '', text)
    text = re.sub(r'Version #.*?\n', '', text)
    text = re.sub(r'Effective date:.*?\n', '', text)
    text = re.sub(r'Section:.*?\n', '', text)
    text = re.sub(r'Revision date:.*?\n', '', text)

    # Collapse multiple newlines
    text = re.sub(r'\n\s*\n+', '\n\n', text)

    # Strip leading/trailing spaces
    return text.strip()

def process_files(input_dir, output_dir):
    for fname in os.listdir(input_dir):
        if fname.lower().endswith(".txt"):
            with open(os.path.join(input_dir, fname), "r", encoding="utf-8") as f:
                raw_text = f.read()

            cleaned_text = clean_text(raw_text)

            output_path = os.path.join(output_dir, fname)
            with open(output_path, "w", encoding="utf-8") as f:
                f.write(cleaned_text)

            print(f"[✔] Cleaned: {fname}")

if __name__ == "__main__":
    process_files(INPUT_DIR, OUTPUT_DIR)
    print("✅ Cleaning complete.")
