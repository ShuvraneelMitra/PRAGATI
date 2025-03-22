import warnings
import logging
from pdfparse.parse import ResearchPaperParser

warnings.filterwarnings("ignore")
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

########################################################################################################################

pdf_path = "C:/Users/MITRA/Desktop/Books/Tiny Machine Learning.pdf"
parser = ResearchPaperParser(pdf_path, output_dir="output", save=True)
try:
    results = parser.process_document()
    print(f"results={results}")
    parser.save_results(results)
    zip_path = parser.zip_output_directory(zip_name="output.zip")

    print(f"Processed {len(results['text'])} pages")
    print(f"Found {sum(len(t) for t in results['tables'].values())} tables")
    print(f"Found {sum(len(e) for e in results['equations'].values())} equations")
    print(f"Extracted {sum(len(i) for i in results['embedded_images'].values())} embedded images")
    print(f"Output directory saved as")
except Exception as e:
    logger.error(f"Error in main execution: {str(e)}")