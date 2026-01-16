from typing import Dict, Any, Optional, List, Union
from pathlib import Path
import yaml
import json
import shutil
from datetime import datetime
from ..logging import Logger
from ...core.base import NexusModule
from ..metrics import MetricsTracker
from ..profiler import ModelProfiler

class ExperimentGenerator(NexusModule):
    def __init__(
        self,
        base_dir: str = "experiments",
        backup_dir: Optional[str] = None,
        max_backups: int = 5,
        metrics_enabled: bool = True,
        profiling_enabled: bool = True
    ):
        """Initialize experiment generator with enhanced features
        
        Args:
            base_dir: Base directory for experiments
            backup_dir: Directory for experiment backups 
            max_backups: Maximum number of backups to keep
            metrics_enabled: Enable metrics tracking
            profiling_enabled: Enable model profiling
        """
        self.base_dir = Path(base_dir)
        self.backup_dir = Path(backup_dir) if backup_dir else self.base_dir / "backups"
        self.max_backups = max_backups
        self.logger = Logger(__name__)
        
        # Initialize components
        self.metrics = MetricsTracker() if metrics_enabled else None
        self.profiler = ModelProfiler() if profiling_enabled else None
        
        # Create directories with error handling
        try:
            self.base_dir.mkdir(parents=True, exist_ok=True)
            self.backup_dir.mkdir(parents=True, exist_ok=True)
            
            # Additional directories for artifacts
            self.cache_dir = self.base_dir / "cache"
            self.cache_dir.mkdir(exist_ok=True)
            self.archive_dir = self.base_dir / "archive" 
            self.archive_dir.mkdir(exist_ok=True)
            
        except PermissionError as e:
            self.logger.error(f"Permission denied creating directories: {e}")
            raise
        except Exception as e:
            self.logger.error(f"Failed to create directories: {e}")
            raise
            
    def generate_experiment(
        self,
        paper_data: Dict[str, Any],
        implementation: Dict[str, Any],
        dry_run: bool = False,
        backup: bool = True,
        validate_deps: bool = True,
        track_metrics: bool = True
    ) -> Union[Path, Dict[str, Any]]:
        """Generate experiment files and configuration with enhanced validation
        
        Args:
            paper_data: Paper metadata and configuration
            implementation: Implementation details and parameters
            dry_run: If True, only validate inputs without creating files
            backup: If True, backup existing experiment with same name
            validate_deps: Validate dependencies before generation
            track_metrics: Track metrics during generation
            
        Returns:
            Path to generated experiment directory or validation results dict
        """
        # Enhanced input validation
        validation_results = self._validate_inputs(paper_data, implementation)
        if not validation_results["valid"]:
            self.logger.error(f"Input validation failed: {validation_results['errors']}")
            if dry_run:
                return validation_results
            raise ValueError(validation_results["errors"])
            
        # Dependency validation
        if validate_deps:
            self._validate_dependencies(implementation)
            
        if dry_run:
            self.logger.info("Dry run - validation successful")
            return validation_results
            
        # Create experiment directory with enhanced error handling
        try:
            exp_dir = self._create_experiment_dir(paper_data, backup=backup)
            
            # Start metrics tracking
            if track_metrics and self.metrics:
                self.metrics.start_tracking()
            
            # Generate artifacts with comprehensive error handling
            try:
                # Generate and save config
                config = self._generate_config(paper_data, implementation)
                self._save_config(exp_dir, config)
                
                # Generate training infrastructure
                train_script = self._generate_training_script(config)
                eval_script = self._generate_eval_script(config)
                self._save_training_scripts(exp_dir, train_script, eval_script)
                
                # Generate documentation and metadata
                self._generate_readme(exp_dir, paper_data)
                self._generate_metadata(exp_dir, paper_data, implementation)
                
                # Generate additional experiment artifacts
                self._generate_data_loaders(exp_dir, config)
                self._generate_model_checkpointing(exp_dir)
                self._generate_visualization_tools(exp_dir)
                
                # Profile if enabled
                if self.profiler:
                    profile_results = self.profiler.profile_implementation(implementation)
                    self._save_profile_results(exp_dir, profile_results)
                
                self.logger.info(f"Successfully generated experiment at {exp_dir}")
                
                # Stop metrics tracking
                if track_metrics and self.metrics:
                    metrics_data = self.metrics.stop_tracking()
                    self._save_metrics(exp_dir, metrics_data)
                    
                return exp_dir
                
            except Exception as e:
                self.logger.error(f"Failed during experiment generation: {str(e)}")
                if exp_dir.exists():
                    # Backup failed experiment for debugging
                    failed_backup = self.archive_dir / f"failed_{exp_dir.name}"
                    shutil.move(exp_dir, failed_backup)
                    self.logger.info(f"Failed experiment backed up to {failed_backup}")
                raise
                
        except Exception as e:
            self.logger.error(f"Critical error in experiment generation: {str(e)}")
            raise
        
    def _create_experiment_dir(
        self,
        paper_data: Dict[str, Any],
        backup: bool = True
    ) -> Path:
        """Create experiment directory structure with enhanced organization"""
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        exp_name = f"{paper_data['name']}_{timestamp}"
        exp_dir = self.base_dir / exp_name
        
        if exp_dir.exists():
            if backup:
                backup_path = self.backup_dir / f"{exp_name}_backup"
                shutil.copytree(exp_dir, backup_path)
                self._cleanup_old_backups()
            shutil.rmtree(exp_dir)
            
        # Create comprehensive directory structure
        try:
            exp_dir.mkdir(parents=True)
            (exp_dir / "config").mkdir()
            (exp_dir / "scripts").mkdir()
            (exp_dir / "data").mkdir()
            (exp_dir / "models").mkdir()
            (exp_dir / "results").mkdir()
            (exp_dir / "logs").mkdir()
            (exp_dir / "checkpoints").mkdir()
            (exp_dir / "visualizations").mkdir()
            (exp_dir / "metrics").mkdir()
            (exp_dir / "profiling").mkdir()
            
            return exp_dir
            
        except Exception as e:
            self.logger.error(f"Failed to create experiment directory structure: {e}")
            if exp_dir.exists():
                shutil.rmtree(exp_dir)
            raise