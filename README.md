
This is the version 2.0 of PRAGATI. To find the earlier work, please visit branch "PRAGATI-legacy".

## Installation and Running instructions

1. Clone the repository on to your local machine: `git clone https://github.com/ShuvraneelMitra/PRAGATI.git`
2. Navigate into the `PRAGATI` folder: `cd PRAGATI`
3. Create a virtual environment in your directory: `python -m venv venv`
4. Now activate the virtual environment using `source venv/bin/activate` for Linux or (either `.\venv\Scripts\activate.
   ps1` or `.\venv\Scripts\activate.bat` for Windows
5. Sync with the `requirements.txt` file inside your `venv`: `pip install -r requirements.txt`. This will install all 
   the 
   relevant packages inside your virtual environment.
6. Install locally, an older version of timm: `pip install timm==0.5.4 -t old_pkgs/timm0.5.4` (for the latest version of PRAGATI, we provide this package as part of our repo so this step might not be needed)
7. Run the app with `uvicorn ui:app --reload --port 8080`. You can choose any port on the localhost of your liking 
   and change it if that port turns out to be blocked.
8. Open `http://127.0.0.1:<port-number>` to get the development server running.

# Abstract

The academic process of peer-review can be taxing at times. The feedback turn-around time can take multiple weeks, and in case the reviewers' response is not affirmative, the loop of improving and re-submitting the research paper to different conferences continues, piling on misery for the poor academic. Adding to that pain is the fact that the huge number of submissions to premier conferences often have researchers rushing through papers and at times, not getting time to read past the abstract. At other times, inexperienced or junior researchers who might not have the necessary level of experience to sieve the radical ideas from the merely flagrant, are recruited for review. To partly automate this process and solve these problems, we use an agentic AI workflow, PRAGATI, to give researchers an approximate idea beforehand about potential improvements and overlooked pitfalls.

# Introduction

PRAGATI is written using the Langgraph, chosen over Smolagents due to its fine-grained control and visually representable graph structure. We use a process which slightly resembles and mimics the human way of reviewing a paper: it asks questions of the paper, dividing it into several subqueries which cover broadly all the aspects of the question. Then the score is used to determine the suitability of publishing the paper at elite conferences.

# Methodology

The overall schematic is shown below:

## (a) The parser
PRAGATI utilizes the `fitz` library (PyMuPDF) to navigate PDF documents, extracting textual content while preserving the structural integrity of the original document. This ensures that sections, paragraphs, and other formatting elements are accurately represented.​

Academic papers often employ multi-column layouts, which can complicate text extraction. To address this, PRAGATI analyzes the spatial distribution of text blocks to detect column boundaries. By identifying significant gaps between text blocks, the system determines the number and positions of columns, facilitating accurate text extraction from complex layouts.​

As an additional feature, PRAGATI employs the `TableTransformerForObjectDetection` model from the `Transformers` library to detect and extract tabular data. Detected tables are processed using Optical Character Recognition (OCR) techniques. To preserve the semantic meaning of complex formulas, PRAGATI integrates the LatexOCR tool to detect and convert mathematical expressions within the document images into LaTeX code.​

## (b)  The Fact Checker

Facts are checked using a combination of a web-search tool, specifically `Tavily`, and the chunks of the PDF document to treat each major part as a fact and then use all available resources, such as `Arxiv` and `GScholar`, to check the "fact" as much as possible.

Finally, a score is assigned to the claims which is based on `Likert-Scale` which is a 5 point scale with 1 being the completely false and 5 being the completely true statement. If the average score of a text chunk is more than 3 then it's considered as factually correct.

## (c) The Critic

The critic works in the same way paper reviewers review papers : it continuously ask questions about different sections of the paper, and it has some personas which essentially refers to the specialization of each critic. The Questions are asked iteratively and answered from the paper in $O(numReviewers\cdot numSections \cdot numSubqueriesPerQuestion)$ time.

Finally, Based on the interview it generates some suggestions/ Action Items for the author.

## (d) Scorer

The scorer part works simply on the basis of fact-checker score and the critic's publishability assessment, which finally give us this idea whether the paper is publishable or not.

## (e) Conference Recommendation:

The conference recommendation is done on the basis of the answers given to the `critic` and finally 
it provides a few recommendations.

# Results

### Dataset Description and performance
We created a dataset containing **150 reseaarch papers** containing both publishable and non-publishable ones also for the publishable ones those were published in CVPR, ICLR, KDD, TMLR  or NIPS and after evaluating the workflow on that we got an accuracy of **89%** for the conference recommendation.

The resulting solution was deployed using the Render deployment service for AI models and agentic workflows, and has been hosted on the server.

### Limitations

 AI may struggle with nuanced aspects of research quality, such as novelty and theoretical impact, which often require domain expertise and subjective judgment. Our system is still not capable of evaluating a novelty or a radical new idea on the same level as a human reviewer.

### Industry Insights

The use of agentic AI workflows in assessing research paper publishability is revolutionizing academic peer review by automating quality checks, fact verification, and alignment with top conferences. This enhances efficiency, reduces biases, and helps detect fraudulent or low-quality submissions, improving overall research integrity.

### Future Directions
Future directions include enhancing multimodal analysis by integrating AI-driven reasoning on figures, tables, and citations, and expanding PRAGATI’s capabilities to provide real-time feedback to authors for improving their papers before submission. Also we can provide a feature to have real-time editing for papers like the `canvas` on ChatGPT.

# Conclusion
Finally the overall workflow is robust in nature containing multiple as it mimics the actual process of reviewing with the help of agents reducing significant amount of human bias and error finally resulting in an omptimum recommendation of conferences along with effective Action Items for the authors.
