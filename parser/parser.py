import fitz
from collections import defaultdict
import json
import re

class PDFTextExtractor:
    def __init__(self, pdf_path):
        self.pdf_path = pdf_path
        self.doc = fitz.open(pdf_path)
        self.font_counts = defaultdict(int)
        self.paragraph_size = None
        self.tagged_text = {}

    def analyze_fonts(self):
        """Analyzes the font sizes in the document to determine the most common paragraph font size."""
        for page in self.doc:
            blocks = page.get_text("dict")["blocks"]
            for block in blocks:
                if "lines" in block:
                    for line in block["lines"]:
                        for span in line["spans"]:
                            font_size = span["size"]
                            self.font_counts[font_size] += 1
        self.paragraph_size = max(self.font_counts, key=self.font_counts.get)

    def tag_text(self, subs= False):
        """Extracts and tags text as header, paragraph, or subscript based on font size."""
        if self.paragraph_size is None:
            raise ValueError("Font analysis must be performed before tagging text.")
        text_list = []
        current_header = None
        for page in self.doc:
            blocks = page.get_text("dict")["blocks"]
            # text_list = []
            for block in blocks:
                if "lines" in block:
                    for line in block["lines"]:
                        # text_list = []
                        for span in line["spans"]:
                            font_size = span["size"]
                            text = span["text"].strip()
                            if not text:
                                continue
                            if font_size > self.paragraph_size:
                                if re.match(r'^\d+$', text): 
                                    continue
                                if text.lower() not in self.tagged_text:  # Add new header
                                    sentence = ' '.join(text_list)
                                    print(text_list)
                                    try:
                                        self.tagged_text[current_header].append(sentence)
                                    except KeyError:
                                        print("KeyError as first header")
                                    text_list = []
                                    current_header = text
                                    self.tagged_text[current_header] = []

                            elif font_size < self.paragraph_size:
                                tag = "subscript"
                                if current_header:
                                    if subs:
                                        self.tagged_text[current_header].append({"subscript": text})
                                    else:
                                        text_list.append(text)
                                        # print(text_list)
                                        # self.tagged_text[current_header].append(text)
                                        # self.tagged_text[current_header].append(" ")
                            else:
                                tag = "paragraph"
                                if current_header:
                                    if subs:
                                        self.tagged_text[current_header].append({"paragraph": text})
                                    else:
                                        # self.tagged_text[current_header].append(text)
                                        # self.tagged_text[current_header].append(" ")
                                        text_list.append(text)
                    
                            # self.tagged_text[current_header] = " ".join(self.tagged_text[current_header])


    def save_as_json(self, output_path):
        """Saves the tagged text to a JSON file."""
        with open(output_path, "w", encoding="utf-8") as json_file:
            json.dump(self.tagged_text, json_file, ensure_ascii=False, indent=4)

# Usage
if __name__ == "__main__":
    extractor = PDFTextExtractor('P002.pdf')
    extractor.analyze_fonts()
    extractor.tag_text(False)
    extractor.save_as_json('seydd.json')
