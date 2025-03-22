# pylint: disable=all

""" 
pip install PyMuPDF Pillow numpy transformers torch pix2tex pandas pytesseract 
PyMuPDF-1.25.3 entmax-1.3 munch-4.0.0 pix2tex-0.1.4 timm-0.5.4 x-transformers-0.15.0
"""

import fitz  # PyMuPDF
from PIL import Image
import numpy as np
from transformers import TableTransformerForObjectDetection, DetrImageProcessor
import torch
from pix2tex.cli import LatexOCR
import os
import pandas as pd
from typing import List, Dict, Tuple
import logging
from pathlib import Path
import zipfile
import pytesseract

# Set up logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

class ResearchPaperParser:
    """A class to parse research papers from PDF, handling multi-column layouts."""

    def __init__(self, df, output_dir: str = "/kaggle/working/output"):
        """Initialize the parser with PDF path from df (string or DataFrame) and output directory."""
        if isinstance(df, str):
            self.pdf_path = df
        elif isinstance(df, pd.DataFrame):
            if 'pdf_path' not in df.columns:
                raise ValueError("DataFrame must contain a 'pdf_path' column")
            self.pdf_path = df['pdf_path'].iloc[0]
        else:
            raise ValueError("df must be a string path or a DataFrame with 'pdf_path' column")
        
        self.output_dir = output_dir
        self.document = None
        
        # Initialize Table Transformer
        self.table_processor = DetrImageProcessor()
        self.table_model = TableTransformerForObjectDetection.from_pretrained(
            "microsoft/table-transformer-detection"
        )
        
        # Initialize Pix2Tex for LaTeX OCR
        self.latex_ocr = LatexOCR()
        
        # Create output directory
        Path(self.output_dir).mkdir(parents=True, exist_ok=True)

    def load_pdf(self) -> None:
        """Load the PDF document."""
        try:
            self.document = fitz.open(self.pdf_path)
            logger.info(f"Loaded PDF with {len(self.document)} pages from {self.pdf_path}")
        except Exception as e:
            logger.error(f"Error loading PDF: {str(e)}")
            raise

    def detect_columns(self, page_num: int) -> List[Tuple[float, float]]:
        """Detect column boundaries on a page using text block positions."""
        if not self.document:
            self.load_pdf()
        
        try:
            page = self.document[page_num]
            blocks = page.get_text("blocks")
            
            text_blocks = [block for block in blocks if block[4].strip()]
            if not text_blocks:
                return [(0, page.rect.width)]
            
            x_coords = sorted([block[0] for block in text_blocks])
            x_max_coords = sorted([block[2] for block in text_blocks])
            
            gaps = [x_coords[i+1] - x_max_coords[i] for i in range(len(x_coords)-1)]
            significant_gaps = [x for x in gaps if x > 50]
            
            if len(significant_gaps) == 0:
                columns = [(0, page.rect.width)]
            elif len(significant_gaps) == 1:
                columns = [
                    (0, x_max_coords[x_coords.index(x_coords[0] + significant_gaps[0])]),
                    (x_coords[x_coords.index(x_coords[0] + significant_gaps[0]) + 1], page.rect.width)
                ]
            else:
                columns = [
                    (0, x_max_coords[x_coords.index(x_coords[0] + significant_gaps[0])]),
                    (x_coords[x_coords.index(x_coords[0] + significant_gaps[0]) + 1], 
                     x_max_coords[x_coords.index(x_coords[0] + significant_gaps[1])]),
                    (x_coords[x_coords.index(x_coords[0] + significant_gaps[1]) + 1], page.rect.width)
                ]
            
            logger.info(f"Detected {len(columns)} columns on page {page_num}")
            return columns
        except Exception as e:
            logger.error(f"Error detecting columns on page {page_num}: {str(e)}")
            return [(0, page.rect.width)]

    def extract_text(self, page_num: int) -> str:
        """Extract plain text from a specific page in column order."""
        if not self.document:
            self.load_pdf()
        
        try:
            page = self.document[page_num]
            columns = self.detect_columns(page_num)
            full_text = ""
            
            for col_left, col_right in columns:
                clip_rect = fitz.Rect(col_left, 0, col_right, page.rect.height)
                blocks = page.get_text("blocks", clip=clip_rect)
                blocks.sort(key=lambda b: (b[1], b[0]))
                column_text = "\n".join(block[4].strip() for block in blocks if block[4].strip())
                full_text += column_text + "\n\n"
            
            return full_text.strip()
        except Exception as e:
            logger.error(f"Error extracting text from page {page_num}: {str(e)}")
            return ""

    def convert_page_to_image(self, page_num: int, zoom: float = 2.0) -> Image.Image:
        """Convert a PDF page to an image with specified zoom level."""
        if not self.document:
            self.load_pdf()
        
        try:
            page = self.document[page_num]
            pix = page.get_pixmap(matrix=fitz.Matrix(zoom, zoom))
            img = Image.frombytes("RGB", [pix.width, pix.height], pix.samples)
            return img
        except Exception as e:
            logger.error(f"Error converting page {page_num} to image: {str(e)}")
            raise

    def detect_tables(self, image: Image.Image) -> List[Dict]:
        """Detect tables in an image using Table Transformer."""
        try:
            inputs = self.table_processor(images=image, return_tensors="pt")
            with torch.no_grad():
                outputs = self.table_model(**inputs)
            
            target_sizes = torch.tensor([image.size[::-1]])
            results = self.table_processor.post_process_object_detection(
                outputs, threshold=0.9, target_sizes=target_sizes
            )[0]
            
            tables = []
            for score, label, box in zip(results["scores"], results["labels"], results["boxes"]):
                if score > 0.9:
                    box = [int(i) for i in box.tolist()]
                    tables.append({
                        "confidence": float(score),
                        "box": box
                    })
            return tables
        except Exception as e:
            logger.error(f"Error detecting tables: {str(e)}")
            return []

    def extract_table_content(self, image: Image.Image, box: List[int], table_index: int, page_num: int) -> Tuple[str, str]:
        """Extract content from a detected table region as text and save as image."""
        try:
            table_img = image.crop(box)
            table_path = os.path.join(self.output_dir, f"table_page_{page_num}_{table_index}.png")
            table_img.save(table_path)
            
            # Use pytesseract to extract text from the table image
            table_text = pytesseract.image_to_string(table_img)
            return table_text, table_path
        except Exception as e:
            logger.error(f"Error extracting table content: {str(e)}")
            return "", ""

    def detect_and_convert_math(self, image: Image.Image) -> List[Tuple[str, List[int]]]:
        """Detect and convert mathematical equations to LaTeX using Pix2Tex."""
        try:
            latex_code = self.latex_ocr(image)
            equations = [(latex_code, [0, 0, image.width, image.height])]
            return equations
        except Exception as e:
            logger.error(f"Error in math detection: {str(e)}")
            return []

    def extract_images(self, page_num: int) -> List[str]:
        """Extract embedded images from a PDF page."""
        if not self.document:
            self.load_pdf()
        
        try:
            page = self.document[page_num]
            image_list = page.get_images(full=True)
            extracted_images = []
            
            for img_index, img in enumerate(image_list):
                xref = img[0]
                base_image = self.document.extract_image(xref)
                image_bytes = base_image["image"]
                image_ext = base_image["ext"]
                image_path = os.path.join(self.output_dir, f"page_{page_num}_img_{img_index}.{image_ext}")
                
                with open(image_path, "wb") as img_file:
                    img_file.write(image_bytes)
                extracted_images.append(image_path)
            
            return extracted_images
        except Exception as e:
            logger.error(f"Error extracting images from page {page_num}: {str(e)}")
            return []

    def process_document(self) -> Dict:
        """Process the entire document and return structured results."""
        if not self.document:
            self.load_pdf()
        
        results = {
            "text": {},
            "tables": {},
            "equations": {},
            "images": {},
            "embedded_images": {}
        }
        
        for page_num in range(len(self.document)):
            logger.info(f"Processing page {page_num + 1}")
            
            results["text"][page_num] = self.extract_text(page_num)
            page_image = self.convert_page_to_image(page_num)
            
            tables = self.detect_tables(page_image)
            results["tables"][page_num] = [
                {
                    "text": self.extract_table_content(page_image, table["box"], idx, page_num)[0],
                    "box": table["box"],
                    "confidence": table["confidence"],
                    "image_path": self.extract_table_content(page_image, table["box"], idx, page_num)[1]
                }
                for idx, table in enumerate(tables)
            ]
            
            results["equations"][page_num] = self.detect_and_convert_math(page_image)
            
            image_path = os.path.join(self.output_dir, f"page_{page_num}.png")
            page_image.save(image_path)
            results["images"][page_num] = image_path
            
            results["embedded_images"][page_num] = self.extract_images(page_num)
        
        return results

    def save_results(self, results: Dict) -> None:
        """Save the processed results to files."""
        try:
            # Define file paths
            text_file = os.path.join(self.output_dir, "text_content.txt")
            tables_file = os.path.join(self.output_dir, "tables_content.txt")
            equations_file = os.path.join(self.output_dir, "equations.tex")
            summary_file = os.path.join(self.output_dir, "summary.txt")
            
            # Write text content
            with open(text_file, "w", encoding="utf-8") as f:
                for page, text in results["text"].items():
                    f.write(f"Page {page}:\n{text}\n{'='*50}\n")
            
            # Write table content as text
            with open(tables_file, "w", encoding="utf-8") as f:
                for page, tables in results["tables"].items():
                    for idx, table in enumerate(tables):
                        f.write(f"Page {page} - Table {idx} (Confidence: {table['confidence']}):\n")
                        f.write(f"Box: {table['box']}\n")
                        f.write(f"Text:\n{table['text']}\n{'-'*50}\n")
            
            # Write LaTeX equations
            with open(equations_file, "w", encoding="utf-8") as f:
                f.write("\\documentclass{article}\n\\usepackage{amsmath}\n\\begin{document}\n")
                for page, eqs in results["equations"].items():
                    f.write(f"\\section{{Page {page}}}\n")
                    for eq, box in eqs:
                        f.write(f"% Position: {box}\n")
                        f.write(f"\\begin{{equation}}\n{eq}\n\\end{{equation}}\n\n")
                f.write("\\end{document}")
            
            # Write summary with equations
            with open(summary_file, "w", encoding="utf-8") as f:
                f.write(f"Pages processed: {len(results['text'])}\n")
                f.write(f"Tables found: {sum(len(t) for t in results['tables'].values())}\n")
                f.write(f"Equations found: {sum(len(e) for e in results['equations'].values())}\n")
                f.write(f"Embedded images extracted: {sum(len(i) for i in results['embedded_images'].values())}\n")
                f.write("\nDetected Equations:\n")
                for page, eqs in results["equations"].items():
                    f.write(f"Page {page}:\n")
                    for eq, box in eqs:
                        f.write(f"  - {eq} (Position: {box})\n")
            
            logger.info(f"Results saved to {self.output_dir}")
        except Exception as e:
            logger.error(f"Error saving results: {str(e)}")

    def zip_output_directory(self, zip_name: str = "output.zip") -> str:
        """Zip the entire output directory."""
        try:
            zip_path = os.path.join("/kaggle/working", zip_name)
            with zipfile.ZipFile(zip_path, 'w', zipfile.ZIP_DEFLATED) as zipf:
                for root, _, files in os.walk(self.output_dir):
                    for file in files:
                        file_path = os.path.join(root, file)
                        arcname = os.path.relpath(file_path, self.output_dir)
                        zipf.write(file_path, os.path.join("output", arcname))
            logger.info(f"Output directory zipped as {zip_path}")
            return zip_path
        except Exception as e:
            logger.error(f"Error zipping output directory: {str(e)}")
            raise

if __name__ == "__main__":
    # Example usage
    # pdf_path = "/kaggle/input/research-paper-sets/A Comprehensive Survey of Retrieval-Augmented Generation (RAG) Evolution Current Landscape and Future Directions.pdf"
    pdf_path = "/kaggle/input/research-paper-sets/Chain-of-Retrieval Augmented Generation.pdf"
    parser = ResearchPaperParser(pdf_path, output_dir="/kaggle/working/output")
    try:
        results = parser.process_document()
        parser.save_results(results)
        zip_path = parser.zip_output_directory(zip_name="output.zip")
        
        print(f"Processed {len(results['text'])} pages")
        print(f"Found {sum(len(t) for t in results['tables'].values())} tables")
        print(f"Found {sum(len(e) for e in results['equations'].values())} equations")
        print(f"Extracted {sum(len(i) for i in results['embedded_images'].values())} embedded images")
        print(f"Output directory saved as '{zip_path}'")
    except Exception as e:
        logger.error(f"Error in main execution: {str(e)}")
