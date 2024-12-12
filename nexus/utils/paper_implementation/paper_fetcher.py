import arxiv
import PyPDF2
import requests
import tempfile
import json
import logging
from typing import Dict, Any, Optional, List, Union
from pathlib import Path
from tenacity import retry, stop_after_attempt, wait_exponential
from ...core.base import NexusModule
from ..experiment import ExperimentTracker

class PaperFetcher(NexusModule):
    def __init__(
        self,
        cache_dir: str = ".paper_cache",
        max_retries: int = 3,
        backup_enabled: bool = True,
        max_backups: int = 5
    ):
        """Initialize paper fetcher with caching and backup support
        
        Args:
            cache_dir: Directory for caching paper data
            max_retries: Maximum number of retry attempts for failed requests
            backup_enabled: Whether to backup cache files
            max_backups: Maximum number of backups to keep
        """
        self.cache_dir = Path(cache_dir)
        self.cache_dir.mkdir(parents=True, exist_ok=True)
        self.backup_dir = self.cache_dir / "backups"
        self.backup_dir.mkdir(exist_ok=True)
        self.backup_enabled = backup_enabled
        self.max_backups = max_backups
        self.logger = logging.getLogger(__name__)
        self.experiment_tracker = ExperimentTracker()
        
    @retry(stop=stop_after_attempt(3), wait=wait_exponential(min=1, max=10))
    def fetch_paper(self, arxiv_id: str, force_refresh: bool = False) -> Dict[str, Any]:
        """Fetch paper from arxiv and extract content with retry logic
        
        Args:
            arxiv_id: ArXiv paper ID
            force_refresh: If True, bypass cache and fetch fresh data
            
        Returns:
            Dictionary containing paper metadata and content
        """
        cache_file = self.cache_dir / f"{arxiv_id}.json"
        
        # Check cache unless force refresh
        if not force_refresh and cache_file.exists():
            try:
                with open(cache_file) as f:
                    return json.load(f)
            except json.JSONDecodeError:
                self.logger.warning(f"Corrupted cache file for {arxiv_id}, refetching")
                
        try:
            # Fetch from arxiv
            search = arxiv.Search(id_list=[arxiv_id])
            paper = next(search.results())
            
            # Download PDF with progress tracking
            with tempfile.NamedTemporaryFile() as tmp:
                paper.download_pdf(tmp.name)
                self.logger.info(f"Downloaded PDF for {arxiv_id}")
                content = self._extract_pdf_content(tmp.name)
                
            paper_data = {
                "arxiv_id": arxiv_id,
                "title": paper.title,
                "authors": [str(author) for author in paper.authors],
                "abstract": paper.summary,
                "content": content,
                "categories": paper.categories,
                "links": paper.links,
                "published": str(paper.published),
                "updated": str(paper.updated)
            }
            
            # Cache results with backup
            self._save_cache(paper_data, cache_file)
            
            # Track as artifact
            self.experiment_tracker.save_artifact(
                paper_data,
                f"paper_{arxiv_id}.json",
                use_torch=False
            )
            
            return paper_data
            
        except Exception as e:
            self.logger.error(f"Failed to fetch paper {arxiv_id}: {e}")
            raise
            
    def _extract_pdf_content(self, pdf_path: str) -> str:
        """Extract text content from PDF with enhanced parsing
        
        Args:
            pdf_path: Path to PDF file
            
        Returns:
            Extracted text content
        """
        try:
            reader = PyPDF2.PdfReader(pdf_path)
            content = []
            
            # Skip cover page
            for page in reader.pages[1:]:
                text = page.extract_text()
                # Basic cleaning
                text = text.replace('\n\n', '\n').strip()
                if text:
                    content.append(text)
                    
            return "\n".join(content)
            
        except Exception as e:
            self.logger.error(f"Failed to extract PDF content: {e}")
            raise
            
    def _save_cache(self, data: Dict[str, Any], cache_file: Path) -> None:
        """Save cache with backup management"""
        try:
            with open(cache_file, "w") as f:
                json.dump(data, f, indent=2)
                
            if self.backup_enabled:
                self._manage_backups(cache_file)
                
        except Exception as e:
            self.logger.error(f"Failed to save cache: {e}")
            
    def _manage_backups(self, source: Path) -> None:
        """Manage rotating backups of cache files"""
        backup_path = self.backup_dir / f"{source.name}.bak"
        if source.exists():
            try:
                import shutil
                shutil.copy(source, backup_path)
                
                # Remove old backups
                existing_backups = sorted(
                    self.backup_dir.glob(f"{source.stem}*.bak")
                )
                while len(existing_backups) > self.max_backups:
                    existing_backups.pop(0).unlink()
                    
            except Exception as e:
                self.logger.error(f"Backup failed: {e}")