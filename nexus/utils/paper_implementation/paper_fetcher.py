import arxiv
import PyPDF2
import requests
import tempfile
import json
from typing import Dict, Any, Optional
from pathlib import Path

class PaperFetcher:
    def __init__(self, cache_dir: str = ".paper_cache"):
        self.cache_dir = Path(cache_dir)
        self.cache_dir.mkdir(exist_ok=True)
        
    def fetch_paper(self, arxiv_id: str) -> Dict[str, Any]:
        """Fetch paper from arxiv and extract content"""
        # Check cache first
        cache_file = self.cache_dir / f"{arxiv_id}.json"
        if cache_file.exists():
            return json.load(open(cache_file))
            
        # Fetch from arxiv
        search = arxiv.Search(id_list=[arxiv_id])
        paper = next(search.results())
        
        # Download PDF
        with tempfile.NamedTemporaryFile() as tmp:
            paper.download_pdf(tmp.name)
            content = self._extract_pdf_content(tmp.name)
            
        paper_data = {
            "title": paper.title,
            "authors": [str(author) for author in paper.authors],
            "abstract": paper.summary,
            "content": content,
            "categories": paper.categories
        }
        
        # Cache results
        json.dump(paper_data, open(cache_file, "w"))
        return paper_data
        
    def _extract_pdf_content(self, pdf_path: str) -> str:
        """Extract text content from PDF"""
        reader = PyPDF2.PdfReader(pdf_path)
        content = []
        for page in reader.pages:
            content.append(page.extract_text())
        return "\n".join(content) 