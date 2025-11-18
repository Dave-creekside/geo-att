"""
Main experiment logger for comprehensive training tracking.
Manages directory structure, file I/O, and coordinates all logging activities.
"""

import json
import os
from pathlib import Path
from datetime import datetime
from typing import Dict, Any, Optional, List
import torch

from .geometry_tracker import (
    GeometryTracker, 
    format_curvatures_for_logging,
    extract_geometry_statistics
)


class ExperimentLogger:
    """
    Comprehensive experiment logger for geometric attention research.
    
    Handles three levels of logging:
    1. High-frequency: Training metrics every step
    2. Medium-frequency: Geometry data every N steps
    3. Low-frequency: Attention analysis every epoch
    """
    
    def __init__(self, 
                 experiment_name: str,
                 config: Dict[str, Any],
                 base_dir: str = 'experiments',
                 geometry_log_freq: int = 10,
                 enable_attention_logging: bool = False,
                 resume_from_dir: Optional[str] = None,
                 start_epoch: int = 0,
                 start_step: int = 0):
        """
        Initialize experiment logger.
        
        Args:
            experiment_name: Name/description of experiment
            config: Configuration dictionary (model, training params, etc.)
            base_dir: Base directory for all experiments
            geometry_log_freq: Log geometry every N steps
            enable_attention_logging: Whether to log detailed attention patterns
            resume_from_dir: Directory to resume from (for continue training)
            start_epoch: Starting epoch number (for continue training)
            start_step: Starting global step (for continue training)
        """
        # Handle resumption
        if resume_from_dir:
            # Resume from existing experiment
            self.experiment_dir = Path(resume_from_dir)
            self.run_id = self.experiment_dir.name
            
            # Load existing config
            config_path = self.experiment_dir / 'config.json'
            if config_path.exists():
                with open(config_path) as f:
                    existing_config = json.load(f)
                self.training_session = existing_config.get('training_session', 1) + 1
            else:
                self.training_session = 2  # Default if config missing
            
            print(f"📊 Resuming experiment: {self.run_id}")
            print(f"   Training session: {self.training_session}")
            print(f"   Starting from epoch {start_epoch}")
        else:
            # Create new experiment
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            self.run_id = f"{experiment_name}_{timestamp}"
            self.experiment_dir = Path(base_dir) / self.run_id
            self.training_session = 1
            
            print(f"📊 Experiment logger initialized: {self.run_id}")
            print(f"   Directory: {self.experiment_dir}")
        
        # Configuration
        self.config = config
        self.geometry_log_freq = geometry_log_freq
        self.enable_attention_logging = enable_attention_logging
        
        # Setup directory structure (safe for existing dirs)
        self._setup_directories()
        
        # Save/update configuration
        self._save_config()
        
        # Open log files (append mode if resuming)
        file_mode = 'a' if resume_from_dir else 'w'
        self.training_log_file = open(self.experiment_dir / 'logs' / 'training.jsonl', file_mode)
        self.geometry_log_file = open(self.experiment_dir / 'logs' / 'geometry.jsonl', file_mode)
        
        if self.enable_attention_logging:
            self.attention_log_file = open(self.experiment_dir / 'logs' / 'attention.jsonl', file_mode)
        else:
            self.attention_log_file = None
        
        # Initialize geometry tracker
        self.geometry_tracker = GeometryTracker(threshold=0.1)
        
        # Track global step across all epochs
        self.global_step = start_step
        
        if resume_from_dir:
            print(f"   Resuming from global step {start_step}")
    
    def _setup_directories(self):
        """Create experiment directory structure."""
        dirs_to_create = [
            self.experiment_dir,
            self.experiment_dir / 'logs',
            self.experiment_dir / 'checkpoints',
            self.experiment_dir / 'visualizations',
            self.experiment_dir / 'samples'
        ]
        
        for dir_path in dirs_to_create:
            dir_path.mkdir(parents=True, exist_ok=True)
    
    def _save_config(self):
        """Save experiment configuration."""
        config_path = self.experiment_dir / 'config.json'
        
        # Add metadata
        config_with_meta = {
            'experiment': {
                'run_id': self.run_id,
                'timestamp': datetime.now().isoformat(),
                'base_dir': str(self.experiment_dir)
            },
            'training_session': self.training_session,
            **self.config
        }
        
        with open(config_path, 'w') as f:
            json.dump(config_with_meta, f, indent=2)
    
    def log_training_step(self, 
                         epoch: int,
                         step: int,
                         loss: float,
                         perplexity: Optional[float] = None,
                         learning_rate: Optional[float] = None,
                         grad_norm: Optional[float] = None,
                         extra_metrics: Optional[Dict] = None):
        """
        Log training metrics for a single step.
        
        Args:
            epoch: Current epoch
            step: Current step within epoch
            loss: Training loss
            perplexity: Optional perplexity (exp(loss))
            learning_rate: Current learning rate
            grad_norm: Gradient norm after clipping
            extra_metrics: Additional metrics to log
        """
        self.global_step += 1
        
        entry = {
            'timestamp': datetime.now().isoformat(),
            'epoch': epoch,
            'step': step,
            'global_step': self.global_step,
            'loss': float(loss),
        }
        
        if perplexity is not None:
            entry['perplexity'] = float(perplexity)
        
        if learning_rate is not None:
            entry['learning_rate'] = float(learning_rate)
        
        if grad_norm is not None:
            entry['grad_norm'] = float(grad_norm)
        
        if extra_metrics:
            entry.update(extra_metrics)
        
        # Write to file
        self.training_log_file.write(json.dumps(entry) + '\n')
        self.training_log_file.flush()
    
    def log_geometry(self,
                    epoch: int,
                    step: int,
                    curvatures: List[List[float]],
                    force: bool = False):
        """
        Log geometry data (curvatures, distribution, statistics).
        
        Args:
            epoch: Current epoch
            step: Current step
            curvatures: List of lists (per layer, per head) of curvatures
            force: Force logging even if not at log frequency
        """
        # Check if we should log (every N steps or forced)
        if not force and step % self.geometry_log_freq != 0:
            return
        
        # Get statistics
        stats = self.geometry_tracker.add_snapshot(epoch, step, curvatures)
        
        # Format for logging
        entry = {
            'timestamp': datetime.now().isoformat(),
            'epoch': epoch,
            'step': step,
            'global_step': self.global_step,
            
            # Geometry distribution
            'geometry_counts': {
                'hyperbolic': stats['n_hyperbolic'],
                'euclidean': stats['n_euclidean'],
                'spherical': stats['n_spherical']
            },
            
            'geometry_percentages': {
                'hyperbolic': stats['pct_hyperbolic'],
                'euclidean': stats['pct_euclidean'],
                'spherical': stats['pct_spherical']
            },
            
            # Curvature statistics
            'curvature_stats': {
                'mean': stats['curvature_mean'],
                'std': stats['curvature_std'],
                'min': stats['curvature_min'],
                'max': stats['curvature_max'],
                'median': stats['curvature_median']
            },
            
            # Per-layer curvatures (detailed)
            'curvatures': format_curvatures_for_logging(curvatures)
        }
        
        # Write to file
        self.geometry_log_file.write(json.dumps(entry) + '\n')
        self.geometry_log_file.flush()
    
    def log_epoch_summary(self,
                         epoch: int,
                         train_loss: float,
                         train_ppl: float,
                         val_loss: float,
                         val_ppl: float,
                         samples: Optional[List[Dict]] = None):
        """
        Log end-of-epoch summary with validation metrics.
        
        Args:
            epoch: Epoch number
            train_loss: Training loss
            train_ppl: Training perplexity
            val_loss: Validation loss
            val_ppl: Validation perplexity
            samples: Optional list of generated samples
        """
        entry = {
            'timestamp': datetime.now().isoformat(),
            'epoch': epoch,
            'train_loss': float(train_loss),
            'train_perplexity': float(train_ppl),
            'val_loss': float(val_loss),
            'val_perplexity': float(val_ppl)
        }
        
        if samples:
            entry['samples'] = samples
        
        # Get current geometry distribution
        current_geom = self.geometry_tracker.get_current_distribution()
        if current_geom:
            entry['geometry'] = current_geom
        
        # Write to separate epoch summary file
        epoch_log_path = self.experiment_dir / 'logs' / 'epoch_summary.jsonl'
        with open(epoch_log_path, 'a') as f:
            f.write(json.dumps(entry) + '\n')
    
    def save_checkpoint(self, epoch: int, model_state: Dict, metrics: Dict,
                       optimizer_state: Optional[Dict] = None,
                       scheduler_state: Optional[Dict] = None) -> str:
        """
        Save model checkpoint.
        
        Args:
            epoch: Epoch number
            model_state: Model state dict
            metrics: Dictionary with current metrics
            optimizer_state: Optional optimizer state dict
            scheduler_state: Optional scheduler state dict
            
        Returns:
            Path to saved checkpoint
        """
        checkpoint_name = f"epoch_{epoch:04d}.pt"
        checkpoint_path = self.experiment_dir / 'checkpoints' / checkpoint_name
        
        checkpoint = {
            'epoch': epoch,
            'model_state_dict': model_state,
            'metrics': metrics,
            'config': self.config,
            'run_id': self.run_id
        }
        
        if optimizer_state is not None:
            checkpoint['optimizer_state_dict'] = optimizer_state
        
        if scheduler_state is not None:
            checkpoint['scheduler_state_dict'] = scheduler_state
        
        torch.save(checkpoint, checkpoint_path)
        return str(checkpoint_path)
    
    def save_samples(self, epoch: int, samples: List[str], prompts: List[str]):
        """
        Save generated text samples.
        
        Args:
            epoch: Epoch number
            samples: List of generated texts
            prompts: List of prompts used
        """
        samples_file = self.experiment_dir / 'samples' / f'epoch_{epoch:04d}_samples.txt'
        
        with open(samples_file, 'w') as f:
            f.write(f"Epoch {epoch} - Generated Samples\n")
            f.write("=" * 70 + "\n\n")
            
            for i, (prompt, sample) in enumerate(zip(prompts, samples), 1):
                f.write(f"Sample {i}:\n")
                f.write(f"  Prompt: {prompt}\n")
                f.write(f"  Generated: {sample}\n\n")
    
    def get_summary(self) -> Dict[str, Any]:
        """
        Get overall experiment summary.
        
        Returns:
            Dictionary with experiment summary
        """
        evolution_summary = self.geometry_tracker.get_evolution_summary()
        
        summary = {
            'run_id': self.run_id,
            'experiment_dir': str(self.experiment_dir),
            'total_steps': self.global_step,
            'geometry_evolution': evolution_summary
        }
        
        return summary
    
    def finalize(self):
        """
        Finalize logging and close files.
        Call at end of training.
        """
        # Close log files
        if self.training_log_file:
            self.training_log_file.close()
        if self.geometry_log_file:
            self.geometry_log_file.close()
        if self.attention_log_file:
            self.attention_log_file.close()
        
        # Save final summary
        summary = self.get_summary()
        summary_path = self.experiment_dir / 'summary.json'
        with open(summary_path, 'w') as f:
            json.dump(summary, f, indent=2)
        
        print(f"\n📊 Experiment logging finalized")
        print(f"   Summary saved to: {summary_path}")
        
        # Print geometry evolution summary
        if summary['geometry_evolution']:
            print(f"\n📐 Geometry Evolution Summary:")
            
            initial = summary['geometry_evolution']['initial_geometry']
            final = summary['geometry_evolution']['final_geometry']
            
            print(f"   Initial: {initial['hyperbolic']:.1f}% H, "
                  f"{initial['euclidean']:.1f}% E, "
                  f"{initial['spherical']:.1f}% S")
            
            print(f"   Final:   {final['hyperbolic']:.1f}% H, "
                  f"{final['euclidean']:.1f}% E, "
                  f"{final['spherical']:.1f}% S")
            
            transition = summary['geometry_evolution'].get('phase_transition_epoch')
            if transition:
                print(f"   ⚠️ Phase transition detected at epoch {transition}")
    
    def __enter__(self):
        """Context manager support."""
        return self
    
    def __exit__(self, exc_type, exc_val, exc_tb):
        """Context manager cleanup."""
        self.finalize()
        return False
