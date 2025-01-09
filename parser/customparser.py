from __future__ import annotations

import io
import re
from typing import Any, Callable, List
from collections import defaultdict
from unstructured.partition.auto import partition
import pathway as pw
from pathway.xpacks.llm._parser_utils import parse
from sentence_transformers import SentenceTransformer
from sklearn.cluster import KMeans

class SectionParser(pw.UDF):
    """
    A custom section parser for extracting and organizing section-wise text from PDF documents.

    Attributes:
        mode (str): Parsing mode, default is "paged".
        embedding_model (str): The sentence transformer model for embedding paragraphs.
        num_sections (int): The number of sections to cluster the content into.
        post_processors (list[Callable] | None): Optional list of post-processing functions.
        unstructured_kwargs (dict): Additional arguments for the Unstructured library.
    """

    def __init__(
        self,
        mode: str = "paged",
        embedding_model: str = "sentence-transformers/all-MiniLM-L6-v2",
        num_sections: int = 6,
        post_processors: list[Callable] | None = None,
        **unstructured_kwargs: Any,
    ):
        """
        Initializes the SectionParser with specified settings.

        Args:
            mode (str): Parsing mode to use (paged, elements, or single).
            embedding_model (str): Name of the sentence transformer model to use for embeddings.
            num_sections (int): Number of sections to divide content into.
            post_processors (list[Callable], optional): Functions to apply to the extracted text.
            **unstructured_kwargs: Additional arguments for partitioning.
        """
        super().__init__()
        self.mode = mode
        self.post_processors = post_processors or []
        self.unstructured_kwargs = unstructured_kwargs
        self.num_sections = num_sections
        self.model = SentenceTransformer(embedding_model)

    def parse_pdf(self, contents: bytes) -> list[str]:
        """
        Parses the PDF content and returns a list of text paragraphs.

        Args:
            contents (bytes): PDF file contents in binary format.

        Returns:
            list[str]: A list of text paragraphs extracted from the PDF.
        """
        elements = partition(file=io.BytesIO(contents), **self.unstructured_kwargs)
        return [element.text for element in elements if element.text]

    def embed_paragraphs(self, paragraphs: List[str]) -> List[Any]:
        """
        Generates embeddings for a list of paragraphs using the specified model.

        Args:
            paragraphs (List[str]): List of paragraphs to embed.

        Returns:
            List[Any]: A list of embeddings for each paragraph.
        """
        return self.model.encode(paragraphs)

    def cluster_paragraphs(self, embeddings: List[Any], num_sections: int) -> List[int]:
        """
        Clusters paragraph embeddings into the specified number of sections.

        Args:
            embeddings (List[Any]): Embeddings of paragraphs.
            num_sections (int): Number of clusters (sections) to form.

        Returns:
            List[int]: A list of cluster labels corresponding to each paragraph.
        """
        kmeans = KMeans(n_clusters=num_sections, random_state=42)
        return kmeans.fit_predict(embeddings)

    def extract_sections(self, paragraphs: List[str], labels: List[int]) -> dict[int, List[str]]:
        """
        Groups paragraphs into sections based on their cluster labels.

        Args:
            paragraphs (List[str]): List of paragraphs.
            labels (List[int]): Cluster labels corresponding to each paragraph.

        Returns:
            dict[int, List[str]]: A dictionary mapping section IDs to paragraphs.
        """
        sections = defaultdict(list)
        for label, paragraph in zip(labels, paragraphs):
            sections[label].append(paragraph)
        return sections

    def identify_titles(self, section_content: List[str]) -> str:
        """
        Identifies a potential title for a section based on heuristic rules.

        Args:
            section_content (List[str]): List of paragraphs in the section.

        Returns:
            str: The identified title or "Untitled Section" if no title is found.
        """
        title_candidates = [p for p in section_content if re.match(r'^[A-Z\s]{4,}', p)]
        return title_candidates[0] if title_candidates else "Untitled Section"

    def __wrapped__(self, contents: bytes, **kwargs) -> list[tuple[str, dict]]:
        """
        Parses the PDF contents and returns section-wise text with metadata.

        Args:
            contents (bytes): The binary contents of the PDF.

        Returns:
            list[tuple[str, dict]]: A list of tuples containing section text and metadata.
        """
        paragraphs = self.parse_pdf(contents)
        embeddings = self.embed_paragraphs(paragraphs)
        labels = self.cluster_paragraphs(embeddings, self.num_sections)

        sections = self.extract_sections(paragraphs, labels)
        section_data = []

        for label, content in sections.items():
            title = self.identify_titles(content)
            section_text = "\n".join(content)
            metadata = {"section_title": title, "section_id": label}
            section_data.append((section_text, metadata))

        return section_data

    def __call__(self, contents: pw.ColumnExpression, **kwargs) -> pw.ColumnExpression:
        """
        Applies the section parser to a Pathway ColumnExpression.

        Args:
            contents (pw.ColumnExpression): Input column expression containing binary content.

        Returns:
            pw.ColumnExpression: Column expression with parsed section-wise data.
        """
        return super().__call__(contents, **kwargs)

# Example usage:
if __name__ == "__main__":
    parser = SectionParser(mode="paged", num_sections=6)
    with open("example.pdf", "rb") as f:
        pdf_contents = f.read()
    result = parser.__wrapped__(pdf_contents)
    for text, metadata in result:
        print(f"Section Title: {metadata['section_title']}")
        print(f"Content:\n{text}\n")
