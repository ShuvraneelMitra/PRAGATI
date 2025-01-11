from __future__ import annotations

import asyncio
# from docling.document_converter import DocumentConverter
from pydantic import BaseModel, ValidationError
from typing import List
import re
import tempfile
import inspect
import io
import logging
import os
import subprocess
import tempfile
from collections.abc import Callable
from functools import partial
from io import BytesIO
from typing import TYPE_CHECKING, Any, Literal

from PIL import Image
from pydantic import BaseModel

import pathway as pw
from pathway.internals import udfs
from pathway.internals.config import _check_entitlements
from pathway.optional_import import optional_imports
from pathway.xpacks.llm import llms, prompts
# from pathway.xpacks.llm._parser_utils import (
#     img_to_b64,
#     maybe_downscale,
#     parse,
#     parse_image_details,
# )
# from pathway.xpacks.llm.constants import DEFAULT_VISION_MODEL

# if TYPE_CHECKING:
#     from openparse.processing import IngestionPipeline

class CustomParse(pw.UDF):
    
    """
    All arguments can be overridden during UDF application.

    Args:
        - mode: single, elements or paged.
          When single, each document is parsed as one long text string.
          When elements, each document is split into unstructured's elements.
          When paged, each pages's text is separately extracted.
        - post_processors: list of callables that will be applied to all extracted texts.
        - **unstructured_kwargs: extra kwargs to be passed to unstructured.io's `partition` function
    """
    
    def __init__(
        self,
        mode: str = "single",
        post_processors: list[Callable] | None = None,
        **unstructured_kwargs: Any,
    ):
        with optional_imports("xpack-llm-docs"):
            import unstructured.partition.auto

        super().__init__()
        _valid_modes = {"single", "elements", "paged"}
        if mode not in _valid_modes:
            raise ValueError(
                f"Got {mode} for `mode`, but should be one of `{_valid_modes}`"
            )

        self.kwargs = dict(
            mode=mode,
            post_processors=post_processors or [],
            unstructured_kwargs=unstructured_kwargs,
        )

    def _combine_metadata(self, left: dict, right: dict) -> dict:
        result = {}
        links = left.pop("links", []) + right.pop("links", [])
        languages = list(set(left.pop("languages", []) + right.pop("languages", [])))
        result.update(left)
        result.update(right)
        result["links"] = links
        result["languages"] = languages
        result.pop("coordinates", None)
        result.pop("parent_id", None)
        result.pop("category_depth", None)
        return result

    def __wrapped__(self, contents: bytes, **kwargs) -> list[tuple[str, dict]]:
        """
        Parse the given document:

        Args:
            - contents: document contents
            - **kwargs: override for defaults set in the constructor

        Returns:
            a list of pairs: text chunk and metadata
            The metadata is obtained from Unstructured, you can check possible values
            in the `Unstructed documentation <https://unstructured-io.github.io/unstructured/metadata.html>`
            Note that when `mode` is set to `single` or `paged` some of these fields are
            removed if they are specific to a single element, e.g. `category_depth`.
        """
        import unstructured.partition.auto

        kwargs = {**self.kwargs, **kwargs}
        elements = unstructured.partition.auto.partition(
            file=BytesIO(contents), **kwargs.pop("unstructured_kwargs")
        )
        post_processors = kwargs.pop("post_processors")
        for element in elements:
            for post_processor in post_processors:
                element.apply(post_processor)

        mode = kwargs.pop("mode")
        if kwargs:
            raise ValueError(f"Unknown arguments: {', '.join(kwargs.keys())}")

        docs = []

        if mode == "elements":
            paragraph = 0
            for element in elements:
                metadata = getattr(element, "metadata", {}).to_dict() if hasattr(element, "metadata") else {}
                if hasattr(element, "category"):
                    metadata["category"] = element.category
                    
                if len(element.text) >= 60:
                    metadata["paragraph"] = paragraph
                    paragraph += 1

                docs.append((str(element), metadata))
        elif mode == "paged" or mode == "single":
            text_dict = {}
            meta_dict = {}
            paragraph = 0
            for element in elements:
                metadata = getattr(element, "metadata", {}).to_dict() if hasattr(element, "metadata") else {}
                if hasattr(element, "category"):
                    metadata["category"] = element.category
                page_number = metadata.get("page_number", 1)
                
                if len(element.text) >= 60:
                    metadata["paragraph"] = paragraph
                    paragraph += 1
                    
                if page_number not in text_dict:
                    text_dict[page_number] = str(element) + "\n\n"
                    meta_dict[page_number] = metadata
                else:
                    text_dict[page_number] += str(element) + "\n\n"
                    meta_dict[page_number] = self._combine_metadata(
                        meta_dict[page_number], metadata
                    )
            docs = [(text_dict[key], meta_dict[key]) for key in text_dict.keys()]



        # print('-----------------------------------------------------------------------------------------')

        # print(vars(docs))

        # print('----------------------------------------------------------------------------------------------')

        
        return docs

    def __call__(self, contents: pw.ColumnExpression, **kwargs) -> pw.ColumnExpression:
        print("Incoming contents:", contents)
        result = super().__call__(contents, **kwargs)
        print("Processed result:", result)
        return result