<div align="center">
  <img src="https://github.com/ShuvraneelMitra/PRAGATI/blob/main/assets/PRAGATI.png" width="400">
  <h1>PRAGATI</h1>
  <p><strong>Paper Review and Guidance for Academic Target Identification</strong></p>
</div>

<div align="center">

[![GitHub stars](https://img.shields.io/github/stars/ShuvraneelMitra/PRAGATI.svg?style=social&label=Star)](https://github.com/ShuvraneelMitra/PRAGATI)
[![GitHub forks](https://img.shields.io/github/forks/ShuvraneelMitra/PRAGATI.svg?style=social&label=Fork)](https://github.com/ShuvraneelMitra/PRAGATI/fork)
[![GitHub watchers](https://img.shields.io/github/watchers/ShuvraneelMitra/PRAGATI.svg?style=social&label=Watch)](https://github.com/ShuvraneelMitra/PRAGATI)
[![License](https://img.shields.io/badge/License-MIT-blue.svg)](LICENSE)
[![Contributor Covenant](https://img.shields.io/badge/Contributor%20Covenant-v2.0%20adopted-ff69b4.svg)](CODE_OF_CONDUCT.md)
[![Python](https://img.shields.io/badge/Python-3.8+-blue.svg)](https://www.python.org/downloads/)
[![LangGraph](https://img.shields.io/badge/LangGraph-Powered-orange.svg)](https://github.com/langchain-ai/langgraph)

</div>

## Awards

Winner of **Outstanding Solution Implementation** 🎉 at the [**Ready Tensor Agentic AI Innovation Challenge 2025**](https://app.readytensor.ai/publications/pragati-paper-review-and-guidance-for-academic-target-identification-Nkv6cLXGp3Hp)

## About

PRAGATI automates the academic paper review process using an agentic AI workflow. It analyzes research papers, checks facts, provides critical feedback, and recommends suitable conferences for submission. This tool helps researchers improve their papers before submission, saving time and increasing chances of acceptance.

> **Note**: This is version 2.0 of PRAGATI. To find the earlier work, please visit the ["PRAGATI-legacy" branch](https://github.com/ShuvraneelMitra/PRAGATI/tree/PRAGATI-legacy).

## Installation and Running Instructions

1. Clone the repository:
   ```bash
   git clone https://github.com/ShuvraneelMitra/PRAGATI.git
   ```

2. Navigate into the project folder:
   ```bash
   cd PRAGATI
   ```

3. Create a virtual environment:
   ```bash
   python -m venv venv
   ```

4. Activate the virtual environment:
   - **Linux/macOS**:
     ```bash
     source venv/bin/activate
     ```
   - **Windows** (PowerShell):
     ```bash
     .\venv\Scripts\activate.ps1
     ```
   - **Windows** (Command Prompt):
     ```bash
     .\venv\Scripts\activate.bat
     ```

5. Install dependencies:
   ```bash
   pip install -r requirements.txt
   ```

6. Install older version of timm (may not be needed for latest version):
   ```bash
   pip install timm==0.5.4 -t old_pkgs/timm0.5.4
   ```

7. Run the application:
   ```bash
   uvicorn ui:app --reload --port 8080
   ```

8. Open your browser and navigate to:
   ```
   http://127.0.0.1:8080
   ```

## Abstract

The academic peer-review process can be time-consuming and inconsistent. Feedback turnaround times often take weeks, and reviewers may not give papers thorough attention due to high submission volumes. Sometimes, inexperienced reviewers may struggle to evaluate research quality effectively. PRAGATI addresses these issues by providing researchers with automated feedback about potential improvements and overlooked pitfalls before formal submission.

## Methodology

PRAGATI consists of five key components:

### 1. The Parser

- Utilizes the `fitz` library (PyMuPDF) to extract text while preserving document structure
- Analyzes spatial distribution of text blocks to handle multi-column layouts
- Employs `TableTransformerForObjectDetection` to extract tabular data
- Integrates LatexOCR to convert mathematical expressions into LaTeX code

### 2. The Fact Checker

- Checks facts using `Tavily` web search and PDF document analysis
- Utilizes resources like `Arxiv` and `GScholar` to verify claims
- Scores factual accuracy using a 5-point Likert Scale (1=false, 5=true)
- Considers text chunks with average scores above 3 as factually correct

### 3. The Critic

- Mimics human paper reviewers by asking questions about different sections
- Uses specialized personas for different areas of evaluation
- Iteratively processes questions and answers from the paper
- Generates actionable suggestions for authors

### 4. The Scorer

- Evaluates publishability based on fact-checker scores and critic assessments
- Determines whether a paper is ready for submission

### 5. Conference Recommender

- Analyzes responses from the critic to match papers with appropriate conferences
- Provides targeted venue recommendations

## Results

- **Dataset**: 150 research papers (both publishable and non-publishable)
- **Publication Venues**: CVPR, ICLR, KDD, TMLR, and NeurIPS
- **Accuracy**: 89% for conference recommendations

## Limitations

- AI may struggle with nuanced aspects of research quality, such as novelty and theoretical impact
- The system cannot evaluate radical new ideas at the same level as human reviewers

## Industry Insights

PRAGATI is revolutionizing academic peer review by:
- Automating quality checks and fact verification
- Reducing biases in the review process
- Detecting fraudulent or low-quality submissions
- Improving overall research integrity

## Future Directions

- Enhancing multimodal analysis of figures, tables, and citations
- Expanding capabilities to provide real-time feedback to authors
- Developing real-time editing features similar to ChatGPT's canvas

## Contributing

Contributions are welcome! Please feel free to submit a Pull Request.

## License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

## Contact

For questions or feedback, please reach out to [your-email@example.com](mailto:your-email@example.com).
