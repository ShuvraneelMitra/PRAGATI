import io
import fitz  # PyMuPDF
import pytesseract
from PIL import Image
import warnings
import logging
import os
import time

warnings.filterwarnings("ignore")
logging.basicConfig(
    filename='PRAGATI.log',
    level=logging.INFO, 
    format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

class PDFParser:
    """
    A class to parse PDF files and extract text, with selective OCR application.
    OCR is only used when necessary and no images are saved to disk.
    """
    
    def __init__(self, pdf_path, ocr_engine=pytesseract):
        """
        Initialize the PDF parser.
        
        Args:
            pdf_path (str): Path to the PDF file
            ocr_engine: The OCR engine to use (default: pytesseract)
        """
        self.pdf_path = pdf_path
        self.ocr_engine = ocr_engine
        self.extracted_text = ""
        self.has_parsed = False
        logger.info(f"Initialized PDFParser for file: {pdf_path}")
        
        # Validate PDF file existence
        if not os.path.exists(pdf_path):
            logger.error(f"PDF file not found: {pdf_path}")
            raise FileNotFoundError(f"PDF file not found: {pdf_path}")
    
    def parse(self):
        """
        Parse the PDF file and extract text, applying OCR only when necessary.
        No images are saved locally during processing.
        
        Returns:
            str: The extracted text
        """
        logger.info(f"Starting to parse PDF: {self.pdf_path}")
        start_time = time.time()
        
        try:
            doc = fitz.open(self.pdf_path)
            total_pages = len(doc)
            logger.info(f"Successfully opened PDF with {total_pages} pages")
            
            full_text = []
            ocr_applied_count = 0
            
            for page_num in range(total_pages):
                logger.debug(f"Processing page {page_num+1}/{total_pages}")
                page = doc.load_page(page_num)
                
                # Try to extract text directly
                text = page.get_text()
                text_length = len(text.strip())
                logger.debug(f"Page {page_num+1}: Extracted {text_length} characters")
                
                # Apply OCR if needed
                if text_length < 50:
                    logger.info(f"Applying OCR to page {page_num+1} (only {text_length} characters found)")
                    ocr_start_time = time.time()
                    
                    try:
                        pix = page.get_pixmap()
                        img_bytes = pix.tobytes("png")
                        img = Image.open(io.BytesIO(img_bytes))
                        ocr_text = self.ocr_engine.image_to_string(img)
                        ocr_length = len(ocr_text.strip())
                        
                        logger.info(f"OCR completed for page {page_num+1}: extracted {ocr_length} characters in {time.time() - ocr_start_time:.2f} seconds")
                        full_text.append(ocr_text)
                        ocr_applied_count += 1
                    except Exception as e:
                        logger.error(f"OCR failed for page {page_num+1}: {str(e)}")
                        full_text.append(f"[OCR ERROR ON PAGE {page_num+1}]")
                else:
                    logger.debug(f"Using direct text extraction for page {page_num+1}")
                    full_text.append(text)
            
            # Store total pages and OCR count before closing the document
            logger.info(f"PDF processing completed. Applied OCR to {ocr_applied_count} of {total_pages} pages")
            
            # Now close the document
            doc.close()
            
            self.extracted_text = "\n\n".join(full_text)
            self.has_parsed = True
            
            total_time = time.time() - start_time
            logger.info(f"PDF parsing completed in {total_time:.2f} seconds, extracted {len(self.extracted_text)} characters")
            
            return self.extracted_text
        
        except Exception as e:
            logger.error(f"Error parsing PDF: {str(e)}", exc_info=True)
            raise
    
    def save_text_to_file(self, output_path=None):
        """
        Save the extracted text to a file.
        
        Args:
            output_path (str): Path where to save the text file.
                               If None, uses the PDF filename with .txt extension.
        
        Returns:
            str: Path to the saved text file
        """
        if not self.has_parsed:
            logger.info("No parsed text found, parsing PDF before saving")
            self.parse()
        
        if output_path is None:
            output_path = self.pdf_path.rsplit('.', 1)[0] + '.txt'
            logger.info(f"No output path specified, using default: {output_path}")
        
        try:
            with open(output_path, 'w', encoding='utf-8') as f:
                f.write(self.extracted_text)
            
            logger.info(f"Text successfully saved to {output_path} ({len(self.extracted_text)} characters)")
            return output_path
        
        except Exception as e:
            logger.error(f"Failed to save text to {output_path}: {str(e)}")
            raise
    
    def get_text(self):
        """
        Get the extracted text.
        
        Returns:
            str: The extracted text
        """
        if not self.has_parsed:
            logger.info("No parsed text found, parsing PDF before returning text")
            self.parse()
        
        logger.debug(f"Returning extracted text ({len(self.extracted_text)} characters)")
        return self.extracted_text
    
if __name__ == "__main__":
    try:
        pdf_path = "/home/naba/Desktop/PRAGATI/satya.pdf"
        logger.info(f"Starting PDF parsing process for: {pdf_path}")
        filename = os.path.splitext(os.path.basename(pdf_path))[0]
        parser = PDFParser(pdf_path)
        extracted_text = parser.parse()
        preview_length = min(1000, len(extracted_text))
        logger.info(f"Text preview (first {preview_length} characters):")
        print(extracted_text[:preview_length])
    
    except Exception as e:
        logger.error(f"PDF processing failed: {str(e)}", exc_info=True)