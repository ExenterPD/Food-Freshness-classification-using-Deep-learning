#!/usr/bin/env python3
"""
Enhanced MobileViT Optimization with Optuna and BOHB

This script provides comprehensive hyperparameter optimization for MobileViT
using both Optuna (with TPE sampler) and BOHB (Bayesian Optimization and 
Hyperband) for food freshness classification.

Features:
- Optuna optimization with TPE sampler and Median pruner
- BOHB optimization with multi-fidelity learning
- Comprehensive hyperparameter search spaces
- Training, validation, and testing results
- Detailed logging and visualization
- Model checkpointing and result saving
"""

import os
import sys
import time
import json
import logging
from pathlib import Path
from datetime import datetime
from typing import Dict, Any, List, Optional, Tuple
import argparse

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from tqdm import tqdm

# Optuna imports
import optuna
from optuna.samplers import TPESampler
from optuna.pruners import MedianPruner, SuccessiveHalvingPruner
from optuna.visualization import plot_optimization_history, plot_param_importances

# BOHB imports
import hpbandster.core.nameserver as hpns
import hpbandster.core.result as hpres
from hpbandster.optimizers import BOHB as BOHBOptimizer
from hpbandster.core.worker import Worker
from ConfigSpace import ConfigurationSpace, UniformFloatHyperparameter, UniformIntegerHyperparameter, CategoricalHyperparameter

# Add project root to path
sys.path.append(str(Path(__file__).parent))

from kfu.utils.io import load_yaml, save_json, ensure_dir
from kfu.utils.seed import set_seed
from kfu.data.dataset_reader import DatasetManager
from kfu.models.mobilevit_finetune import create_mobilevit_model
from kfu.eval.metrics import calculate_metrics


def sanitize(obj):
    """
    Recursively convert numpy scalars to native Python types for serialization.
    
    Args:
        obj: Object to sanitize (dict, list, tuple, or any other type)
        
    Returns:
        Sanitized object with numpy scalars converted to Python types
    """
    if isinstance(obj, dict):
        return {key: sanitize(value) for key, value in obj.items()}
    elif isinstance(obj, (list, tuple)):
        return type(obj)(sanitize(item) for item in obj)
    elif isinstance(obj, np.integer):
        return int(obj)
    elif isinstance(obj, np.floating):
        return float(obj)
    elif isinstance(obj, np.str_):
        return str(obj)
    elif isinstance(obj, np.bool_):
        return bool(obj)
    elif isinstance(obj, np.ndarray):
        return obj.tolist()
    else:
        return obj


class MobileViTBOHBWorker(Worker):
    """BOHB Worker for MobileViT optimization."""
    
    def __init__(self, config: Dict[str, Any], train_loader: DataLoader, 
                 val_loader: DataLoader, test_loader: DataLoader, 
                 device: str = 'auto', **kwargs):
        super().__init__(**kwargs)
        self.config = config
        self.train_loader = train_loader
        self.val_loader = val_loader
        self.test_loader = test_loader
        self.device = self._get_device(device)
        
    def _get_device(self, device: str) -> str:
        """Get device for training."""
        if device == 'auto':
            device = 'cuda' if torch.cuda.is_available() else 'cpu'
        
        # Force GPU usage if available
        if torch.cuda.is_available():
            device = 'cuda'
            print(f"🚀 Using GPU: {torch.cuda.get_device_name(0)}")
        else:
            device = 'cpu'
            print("⚠️  GPU not available, using CPU")
        
        return device
    
    def compute(self, config, budget, **kwargs):
        """Compute function for BOHB optimization."""
        try:
            # Create model with hyperparameters
            model = create_mobilevit_model(
                backbone=config.get('backbone', 'mobilevit_s'),
                num_classes=2,
                freeze_backbone=config.get('freeze_backbone', False),
                dropout=config.get('dropout', 0.1)
            )
            model.to(self.device)
            
            # Setup optimizer
            if config.get('optimizer', 'adamw').lower() == 'adamw':
                optimizer = optim.AdamW(
                    model.parameters(),
                    lr=config.get('lr', 0.0003),
                    weight_decay=config.get('weight_decay', 0.0005)
                )
            else:  # SGD
                optimizer = optim.SGD(
                    model.parameters(),
                    lr=config.get('lr', 0.0003),
                    momentum=config.get('momentum', 0.9),
                    weight_decay=config.get('weight_decay', 0.0005)
                )
            
            # Setup scheduler
            scheduler = optim.lr_scheduler.ReduceLROnPlateau(
                optimizer, mode='max', factor=0.5, patience=2
            )
            
            # Loss function
            criterion = nn.CrossEntropyLoss()
            
            # Training loop with budget
            best_val_f1 = 0.0
            epochs = int(budget)
            
            # Print device being used
            print(f"🔧 BOHB Training on device: {self.device}")
            
            for epoch in range(epochs):
                # Training
                model.train()
                train_loss = 0.0
                train_predictions = []
                train_labels = []
                batch_count = 0
                
                for batch in self.train_loader:
                    images = batch['image'].to(self.device)
                    labels = batch['label'].to(self.device)
                    
                    optimizer.zero_grad()
                    outputs = model(images)
                    loss = criterion(outputs, labels)
                    loss.backward()
                    optimizer.step()
                    
                    train_loss += loss.item()
                    predictions = torch.argmax(outputs, dim=1)
                    train_predictions.extend(predictions.cpu().numpy())
                    train_labels.extend(labels.cpu().numpy())
                    
                    batch_count += 1
                    
                    # Live progress logging every 50 batches
                    if batch_count % 50 == 0:
                        current_loss = loss.item()
                        print(f"📊 BOHB Trial | Epoch {epoch+1}/{epochs} | Batch {batch_count} | Loss: {current_loss:.4f}")
                
                # Validation
                model.eval()
                val_loss = 0.0
                val_predictions = []
                val_labels = []
                
                with torch.no_grad():
                    for batch in self.val_loader:
                        images = batch['image'].to(self.device)
                        labels = batch['label'].to(self.device)
                        
                        outputs = model(images)
                        loss = criterion(outputs, labels)
                        
                        val_loss += loss.item()
                        predictions = torch.argmax(outputs, dim=1)
                        val_predictions.extend(predictions.cpu().numpy())
                        val_labels.extend(labels.cpu().numpy())
                
                # Calculate metrics - convert lists to numpy arrays
                train_metrics = calculate_metrics(np.array(train_labels), np.array(train_predictions))
                val_metrics = calculate_metrics(np.array(val_labels), np.array(val_predictions))
                
                # Update scheduler
                scheduler.step(val_metrics['f1'])
                
                # Print epoch summary
                avg_train_loss = train_loss / len(self.train_loader)
                print(f"📈 BOHB Trial | Epoch {epoch+1}/{epochs} | Avg Train Loss: {avg_train_loss:.4f} | Val Accuracy: {val_metrics['accuracy']:.4f}")
                
                # Track best validation F1
                if val_metrics['f1'] > best_val_f1:
                    best_val_f1 = val_metrics['f1']
            
            # Test evaluation
            model.eval()
            test_predictions = []
            test_labels = []
            
            with torch.no_grad():
                for batch in self.test_loader:
                    images = batch['image'].to(self.device)
                    labels = batch['label'].to(self.device)
                    
                    outputs = model(images)
                    predictions = torch.argmax(outputs, dim=1)
                    test_predictions.extend(predictions.cpu().numpy())
                    test_labels.extend(labels.cpu().numpy())
            
            test_metrics = calculate_metrics(np.array(test_labels), np.array(test_predictions))
            
            result = {
                'loss': 1.0 - best_val_f1,  # BOHB minimizes loss
                'info': {
                    'val_f1': best_val_f1,
                    'val_accuracy': val_metrics['accuracy'],
                    'val_precision': val_metrics['precision'],
                    'val_recall': val_metrics['recall'],
                    'test_f1': test_metrics['f1'],
                    'test_accuracy': test_metrics['accuracy'],
                    'test_precision': test_metrics['precision'],
                    'test_recall': test_metrics['recall'],
                    'config': config,
                    'budget': budget
                }
            }
            
            # Sanitize the result to ensure all numpy types are converted to Python types
            return sanitize(result)
            
        except Exception as e:
            print(f"Error in BOHB worker: {e}")
            error_result = {
                'loss': 1.0,  # Worst possible loss
                'info': {
                    'error': str(e),
                    'config': config,
                    'budget': budget
                }
            }
            return sanitize(error_result)


class EnhancedMobileViTOptimizer:
    """Enhanced MobileViT optimizer with Optuna and BOHB."""
    
    def __init__(self, config: Dict[str, Any], device: str = 'auto', 
                 log_level: str = 'INFO'):
        self.config = config
        self.device = self._get_device(device)
        self.results = {}
        
        # Setup logging
        logging.basicConfig(
            level=getattr(logging, log_level.upper()),
            format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
        )
        self.logger = logging.getLogger(__name__)
        
    def _get_device(self, device: str) -> str:
        """Get device for training."""
        if device == 'auto':
            device = 'cuda' if torch.cuda.is_available() else 'cpu'
        
        # Force GPU usage if available
        if torch.cuda.is_available():
            device = 'cuda'
            print(f"🚀 Using GPU: {torch.cuda.get_device_name(0)}")
        else:
            device = 'cpu'
            print("⚠️  GPU not available, using CPU")
        
        return device
    
    def get_optuna_search_space(self) -> Dict[str, Any]:
        """Define comprehensive Optuna search space."""
        return {
            'backbone': ['mobilevit_s', 'mobilevit_xs', 'mobilevit_xxs'],
            'optimizer': ['adamw', 'sgd'],
            'lr': (1e-5, 1e-2),
            'weight_decay': (1e-6, 1e-2),
            'dropout': (0.0, 0.5),
            'freeze_backbone': [True, False],
            'momentum': (0.8, 0.95),  # For SGD
            'batch_size': [16, 32, 64],
            'scheduler_factor': (0.1, 0.8),
            'scheduler_patience': [2, 3, 5]
        }
    
    def get_bohb_search_space(self) -> ConfigurationSpace:
        """Define comprehensive BOHB search space."""
        cs = ConfigurationSpace()
        
        # Model architecture
        cs.add_hyperparameter(CategoricalHyperparameter('backbone', 
            ['mobilevit_s', 'mobilevit_xs', 'mobilevit_xxs']))
        cs.add_hyperparameter(CategoricalHyperparameter('freeze_backbone', [True, False]))
        cs.add_hyperparameter(UniformFloatHyperparameter('dropout', lower=0.0, upper=0.5))
        
        # Optimizer
        cs.add_hyperparameter(CategoricalHyperparameter('optimizer', ['adamw', 'sgd']))
        cs.add_hyperparameter(UniformFloatHyperparameter('lr', lower=1e-5, upper=1e-2, log=True))
        cs.add_hyperparameter(UniformFloatHyperparameter('weight_decay', lower=1e-6, upper=1e-2, log=True))
        cs.add_hyperparameter(UniformFloatHyperparameter('momentum', lower=0.8, upper=0.95))
        
        # Training
        cs.add_hyperparameter(CategoricalHyperparameter('batch_size', [16, 32, 64]))
        cs.add_hyperparameter(UniformFloatHyperparameter('scheduler_factor', lower=0.1, upper=0.8))
        cs.add_hyperparameter(UniformIntegerHyperparameter('scheduler_patience', lower=2, upper=5))
        
        return cs
    
    def optimize_with_optuna(self, train_loader: DataLoader, val_loader: DataLoader, 
                            test_loader: DataLoader, n_trials: int = 100, 
                            timeout: Optional[int] = None, 
                            pruner_type: str = 'median') -> Dict[str, Any]:
        """Optimize MobileViT using Optuna with advanced features."""
        self.logger.info(f"Starting Optuna optimization with {n_trials} trials...")
        
        def objective(trial):
            # Sample hyperparameters
            config = {
                'backbone': trial.suggest_categorical('backbone', 
                    ['mobilevit_s', 'mobilevit_xs', 'mobilevit_xxs']),
                'optimizer': trial.suggest_categorical('optimizer', ['adamw', 'sgd']),
                'lr': trial.suggest_float('lr', 1e-5, 1e-2, log=True),
                'weight_decay': trial.suggest_float('weight_decay', 1e-6, 1e-2, log=True),
                'dropout': trial.suggest_float('dropout', 0.0, 0.5),
                'freeze_backbone': trial.suggest_categorical('freeze_backbone', [True, False]),
                'batch_size': trial.suggest_categorical('batch_size', [16, 32, 64]),
                'scheduler_factor': trial.suggest_float('scheduler_factor', 0.1, 0.8),
                'scheduler_patience': trial.suggest_int('scheduler_patience', 2, 5)
            }
            
            # Add momentum for SGD
            if config['optimizer'] == 'sgd':
                config['momentum'] = trial.suggest_float('momentum', 0.8, 0.95)
            
            # Create and train model
            try:
                result = self._train_single_config(config, train_loader, val_loader, test_loader, trial.number)
                
                # Report intermediate results for pruning
                trial.report(result['val_f1'], step=0)
                
                # Check if trial should be pruned
                if trial.should_prune():
                    raise optuna.TrialPruned()
                
                return result['val_f1']
            except Exception as e:
                self.logger.error(f"Trial failed: {e}")
                raise optuna.TrialPruned()
        
        # Setup pruner
        if pruner_type == 'median':
            pruner = MedianPruner(n_startup_trials=10, n_warmup_steps=5)
        elif pruner_type == 'successive_halving':
            pruner = SuccessiveHalvingPruner()
        else:
            pruner = None
        
        # Create study
        study = optuna.create_study(
            direction='maximize',
            sampler=TPESampler(seed=42, n_startup_trials=20),
            pruner=pruner
        )
        
        # Optimize
        study.optimize(objective, n_trials=n_trials, timeout=timeout)
        
        # Get best result
        best_trial = study.best_trial
        best_config = best_trial.params
        
        self.logger.info(f"Best Optuna result: F1={best_trial.value:.4f}")
        self.logger.info(f"Best config: {best_config}")
        
        return {
            'method': 'optuna',
            'best_value': best_trial.value,
            'best_config': best_config,
            'study': study,
            'n_trials': len(study.trials),
            'pruned_trials': len([t for t in study.trials if t.state == optuna.trial.TrialState.PRUNED])
        }
    
    def optimize_with_bohb(self, train_loader: DataLoader, val_loader: DataLoader, 
                          test_loader: DataLoader, min_budget: int = 1, 
                          max_budget: int = 10, n_iterations: int = 20) -> Dict[str, Any]:
        """Optimize MobileViT using BOHB with multi-fidelity learning."""
        self.logger.info(f"Starting BOHB optimization with {n_iterations} iterations...")
        
        # Create nameserver
        ns = hpns.NameServer(run_id='mobilevit_bohb', host='127.0.0.1', port=0)
        ns_host, ns_port = ns.start()
        
        # Create worker
        worker = MobileViTBOHBWorker(
            config=self.config,
            train_loader=train_loader,
            val_loader=val_loader,
            test_loader=test_loader,
            device=self.device,
            run_id='mobilevit_bohb',
            nameserver=ns_host,
            nameserver_port=ns_port
        )
        worker.run(background=True)
        
        # Create optimizer
        bohb = BOHBOptimizer(
            configspace=self.get_bohb_search_space(),
            run_id='mobilevit_bohb',
            nameserver=ns_host,
            nameserver_port=ns_port,
            min_budget=min_budget,
            max_budget=max_budget
        )
        
        # Run optimization
        res = bohb.run(n_iterations=n_iterations, min_n_workers=1)
        
        # Shutdown
        bohb.shutdown(shutdown_workers=True)
        ns.shutdown()
        
        # Get best result
        id2config = res.get_id2config_mapping()
        incumbent = res.get_incumbent_id()
        best_config = id2config[incumbent]['config']
        best_loss = res.get_runs_by_id(incumbent)[-1]['loss']
        best_f1 = 1.0 - best_loss
        
        self.logger.info(f"Best BOHB result: F1={best_f1:.4f}")
        self.logger.info(f"Best config: {best_config}")
        
        return {
            'method': 'bohb',
            'best_value': best_f1,
            'best_config': best_config,
            'result': res,
            'n_iterations': n_iterations
        }
    
    def _train_single_config(self, config: Dict[str, Any], train_loader: DataLoader, 
                           val_loader: DataLoader, test_loader: DataLoader, trial_number: int = 0) -> Dict[str, Any]:
        """Train a single configuration and return comprehensive results."""
        # Create model
        model = create_mobilevit_model(
            backbone=config['backbone'],
            num_classes=2,
            freeze_backbone=config['freeze_backbone'],
            dropout=config['dropout']
        )
        model.to(self.device)
        
        # Setup optimizer
        if config['optimizer'].lower() == 'adamw':
            optimizer = optim.AdamW(
                model.parameters(),
                lr=config['lr'],
                weight_decay=config['weight_decay']
            )
        else:  # SGD
            optimizer = optim.SGD(
                model.parameters(),
                lr=config['lr'],
                momentum=config.get('momentum', 0.9),
                weight_decay=config['weight_decay']
            )
        
        # Setup scheduler
        scheduler = optim.lr_scheduler.ReduceLROnPlateau(
            optimizer, 
            mode='max', 
            factor=config.get('scheduler_factor', 0.5),
            patience=config.get('scheduler_patience', 3)
        )
        
        # Loss function
        criterion = nn.CrossEntropyLoss()
        
        # Training loop
        best_val_f1 = 0.0
        epochs = self.config['train']['epochs']
        
        train_history = {'loss': [], 'accuracy': [], 'f1': []}
        val_history = {'loss': [], 'accuracy': [], 'f1': []}
        
        # Print device being used
        print(f"🔧 Training on device: {self.device}")
        
        # Initialize detailed logging
        detailed_log = {
            'trial_number': trial_number,
            'config': config,
            'device': str(self.device),
            'epochs': epochs,
            'batch_logs': [],
            'epoch_logs': [],
            'final_metrics': {}
        }
        
        for epoch in range(epochs):
            # Training
            model.train()
            train_loss = 0.0
            train_predictions = []
            train_labels = []
            batch_count = 0
            
            for batch_idx, batch in enumerate(train_loader):
                images = batch['image'].to(self.device)
                labels = batch['label'].to(self.device)
                
                optimizer.zero_grad()
                outputs = model(images)
                loss = criterion(outputs, labels)
                loss.backward()
                optimizer.step()
                
                train_loss += loss.item()
                predictions = torch.argmax(outputs, dim=1)
                train_predictions.extend(predictions.cpu().numpy())
                train_labels.extend(labels.cpu().numpy())
                
                batch_count += 1
                
                # Live progress logging every 50 batches
                if batch_count % 50 == 0:
                    current_loss = loss.item()
                    print(f"📊 Trial {trial_number} | Epoch {epoch+1}/{epochs} | Batch {batch_count} | Loss: {current_loss:.4f}")
                    
                    # Log detailed batch information
                    detailed_log['batch_logs'].append({
                        'trial': trial_number,
                        'epoch': epoch + 1,
                        'batch': batch_count,
                        'loss': current_loss,
                        'timestamp': time.time()
                    })
            
            # Validation
            model.eval()
            val_loss = 0.0
            val_predictions = []
            val_labels = []
            
            with torch.no_grad():
                for batch in val_loader:
                    images = batch['image'].to(self.device)
                    labels = batch['label'].to(self.device)
                    
                    outputs = model(images)
                    loss = criterion(outputs, labels)
                    
                    val_loss += loss.item()
                    predictions = torch.argmax(outputs, dim=1)
                    val_predictions.extend(predictions.cpu().numpy())
                    val_labels.extend(labels.cpu().numpy())
            
            # Calculate metrics - convert lists to numpy arrays
            train_metrics = calculate_metrics(np.array(train_labels), np.array(train_predictions))
            val_metrics = calculate_metrics(np.array(val_labels), np.array(val_predictions))
            
            # Update scheduler
            scheduler.step(val_metrics['f1'])
            
            # Track history
            avg_train_loss = train_loss / len(train_loader)
            train_history['loss'].append(avg_train_loss)
            train_history['accuracy'].append(train_metrics['accuracy'])
            train_history['f1'].append(train_metrics['f1'])
            
            val_history['loss'].append(val_loss / len(val_loader))
            val_history['accuracy'].append(val_metrics['accuracy'])
            val_history['f1'].append(val_metrics['f1'])
            
            # Print epoch summary
            print(f"📈 Trial {trial_number} | Epoch {epoch+1}/{epochs} | Avg Train Loss: {avg_train_loss:.4f} | Val Accuracy: {val_metrics['accuracy']:.4f}")
            
            # Log detailed epoch information
            epoch_log = {
                'trial': trial_number,
                'epoch': epoch + 1,
                'avg_train_loss': avg_train_loss,
                'train_accuracy': train_metrics['accuracy'],
                'train_f1': train_metrics['f1'],
                'val_loss': val_loss / len(val_loader),
                'val_accuracy': val_metrics['accuracy'],
                'val_f1': val_metrics['f1'],
                'timestamp': time.time()
            }
            detailed_log['epoch_logs'].append(epoch_log)
            
            # Track best validation F1
            if val_metrics['f1'] > best_val_f1:
                best_val_f1 = val_metrics['f1']
        
        # Test evaluation
        model.eval()
        test_predictions = []
        test_labels = []
        test_probabilities = []
        
        with torch.no_grad():
            for batch in test_loader:
                images = batch['image'].to(self.device)
                labels = batch['label'].to(self.device)
                
                outputs = model(images)
                probabilities = torch.softmax(outputs, dim=1)
                predictions = torch.argmax(outputs, dim=1)
                
                test_predictions.extend(predictions.cpu().numpy())
                test_labels.extend(labels.cpu().numpy())
                test_probabilities.extend(probabilities.cpu().numpy())
        
        test_metrics = calculate_metrics(np.array(test_labels), np.array(test_predictions), np.array(test_probabilities))
        
        # Log final metrics
        detailed_log['final_metrics'] = {
            'val_f1': best_val_f1,
            'val_accuracy': val_metrics['accuracy'],
            'val_precision': val_metrics['precision'],
            'val_recall': val_metrics['recall'],
            'test_f1': test_metrics['f1'],
            'test_accuracy': test_metrics['accuracy'],
            'test_precision': test_metrics['precision'],
            'test_recall': test_metrics['recall'],
            'test_auc': test_metrics.get('auc', 0.0),
            'train_history': train_history,
            'val_history': val_history
        }
        
        # Save detailed log to file
        timestamp = int(time.time())
        log_filename = f"detailed_trial_{trial_number}_{timestamp}.json"
        log_path = Path("runs") / log_filename
        log_path.parent.mkdir(exist_ok=True)
        
        with open(log_path, 'w') as f:
            json.dump(sanitize(detailed_log), f, indent=2)
        
        print(f"💾 Detailed results saved to: {log_path}")
        
        return {
            'val_f1': best_val_f1,
            'val_accuracy': val_metrics['accuracy'],
            'val_precision': val_metrics['precision'],
            'val_recall': val_metrics['recall'],
            'test_f1': test_metrics['f1'],
            'test_accuracy': test_metrics['accuracy'],
            'test_precision': test_metrics['precision'],
            'test_recall': test_metrics['recall'],
            'test_auc': test_metrics.get('auc', 0.0),
            'config': config,
            'train_history': train_history,
            'val_history': val_history,
            'detailed_log_path': str(log_path)
        }
    
    def run_comprehensive_optimization(self, train_loader: DataLoader, val_loader: DataLoader, 
                                     test_loader: DataLoader, save_dir: str = "runs",
                                     optuna_trials: int = 100, bohb_iterations: int = 20) -> Dict[str, Any]:
        """Run comprehensive optimization with both Optuna and BOHB."""
        self.logger.info("🚀 Starting Comprehensive MobileViT Optimization")
        self.logger.info("=" * 60)
        
        # Create save directory
        save_path = Path(save_dir)
        ensure_dir(save_path)
        
        results = {}
        
        # Run Optuna optimization
        self.logger.info("\n🔍 Running Optuna Optimization...")
        start_time = time.time()
        optuna_results = self.optimize_with_optuna(
            train_loader, val_loader, test_loader, 
            n_trials=optuna_trials, timeout=None
        )
        optuna_time = time.time() - start_time
        optuna_results['optimization_time'] = optuna_time
        results['optuna'] = optuna_results
        
        # Run BOHB optimization
        self.logger.info("\n🔍 Running BOHB Optimization...")
        start_time = time.time()
        bohb_results = self.optimize_with_bohb(
            train_loader, val_loader, test_loader, 
            min_budget=1, max_budget=5, n_iterations=bohb_iterations
        )
        bohb_time = time.time() - start_time
        bohb_results['optimization_time'] = bohb_time
        results['bohb'] = bohb_results
        
        # Compare results
        self.logger.info("\n📊 Optimization Results Comparison")
        self.logger.info("=" * 60)
        self.logger.info(f"Optuna Best F1: {optuna_results['best_value']:.4f} (Time: {optuna_time:.2f}s)")
        self.logger.info(f"BOHB Best F1: {bohb_results['best_value']:.4f} (Time: {bohb_time:.2f}s)")
        
        if optuna_results['best_value'] > bohb_results['best_value']:
            self.logger.info("🏆 Optuna achieved better results!")
            best_method = 'optuna'
            best_config = optuna_results['best_config']
        else:
            self.logger.info("🏆 BOHB achieved better results!")
            best_method = 'bohb'
            best_config = bohb_results['best_config']
        
        # Train final model with best configuration
        self.logger.info(f"\n🎯 Training Final Model with Best Configuration ({best_method})")
        self.logger.info("=" * 60)
        final_results = self._train_single_config(best_config, train_loader, val_loader, test_loader, 0)
        
        # Create comprehensive results
        optimization_results = {
            'optuna_results': optuna_results,
            'bohb_results': bohb_results,
            'best_method': best_method,
            'best_config': best_config,
            'final_results': final_results,
            'timestamp': datetime.now().strftime('%Y%m%d_%H%M%S'),
            'summary': {
                'optuna_f1': optuna_results['best_value'],
                'bohb_f1': bohb_results['best_value'],
                'final_f1': final_results['val_f1'],
                'final_test_f1': final_results['test_f1'],
                'final_test_accuracy': final_results['test_accuracy'],
                'best_method': best_method
            }
        }
        
        # Save results
        save_json(optimization_results, save_path / "enhanced_mobilevit_optimization_results.json")
        
        # Create visualizations
        self._create_optimization_visualizations(optimization_results, save_path)
        
        self.logger.info(f"\n✅ Optimization Complete!")
        self.logger.info(f"Best Method: {best_method}")
        self.logger.info(f"Best Validation F1-Score: {final_results['val_f1']:.4f}")
        self.logger.info(f"Test F1-Score: {final_results['test_f1']:.4f}")
        self.logger.info(f"Test Accuracy: {final_results['test_accuracy']:.4f}")
        self.logger.info(f"Results saved to: {save_path}")
        
        return optimization_results
    
    def _create_optimization_visualizations(self, results: Dict[str, Any], save_path: Path):
        """Create comprehensive visualizations of optimization results."""
        self.logger.info("Creating optimization visualizations...")
        
        # Set style
        plt.style.use('seaborn-v0_8')
        sns.set_palette("husl")
        
        # 1. Optimization History Comparison
        fig, axes = plt.subplots(2, 2, figsize=(15, 12))
        fig.suptitle('MobileViT Optimization Results', fontsize=16, fontweight='bold')
        
        # Optuna optimization history
        if 'study' in results['optuna_results']:
            study = results['optuna_results']['study']
            trial_values = [t.value for t in study.trials if t.value is not None]
            trial_numbers = [i for i, t in enumerate(study.trials) if t.value is not None]
            
            axes[0, 0].plot(trial_numbers, trial_values, 'b-', alpha=0.7, label='F1 Score')
            axes[0, 0].set_title('Optuna Optimization History')
            axes[0, 0].set_xlabel('Trial Number')
            axes[0, 0].set_ylabel('F1 Score')
            axes[0, 0].grid(True, alpha=0.3)
            axes[0, 0].legend()
        
        # Method comparison
        methods = ['Optuna', 'BOHB']
        f1_scores = [results['optuna_results']['best_value'], results['bohb_results']['best_value']]
        colors = ['skyblue', 'lightcoral']
        
        bars = axes[0, 1].bar(methods, f1_scores, color=colors, alpha=0.7)
        axes[0, 1].set_title('Best F1 Score Comparison')
        axes[0, 1].set_ylabel('F1 Score')
        axes[0, 1].set_ylim(0, 1)
        
        # Add value labels on bars
        for bar, score in zip(bars, f1_scores):
            axes[0, 1].text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.01,
                           f'{score:.4f}', ha='center', va='bottom', fontweight='bold')
        
        # Final model performance
        metrics = ['Accuracy', 'Precision', 'Recall', 'F1-Score']
        test_values = [
            results['final_results']['test_accuracy'],
            results['final_results']['test_precision'],
            results['final_results']['test_recall'],
            results['final_results']['test_f1']
        ]
        
        bars = axes[1, 0].bar(metrics, test_values, color='lightgreen', alpha=0.7)
        axes[1, 0].set_title('Final Model Test Performance')
        axes[1, 0].set_ylabel('Score')
        axes[1, 0].set_ylim(0, 1)
        axes[1, 0].tick_params(axis='x', rotation=45)
        
        # Add value labels
        for bar, value in zip(bars, test_values):
            axes[1, 0].text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.01,
                           f'{value:.4f}', ha='center', va='bottom', fontweight='bold')
        
        # Training curves for final model
        if 'train_history' in results['final_results']:
            train_history = results['final_results']['train_history']
            val_history = results['final_results']['val_history']
            epochs = range(1, len(train_history['f1']) + 1)
            
            axes[1, 1].plot(epochs, train_history['f1'], 'b-', label='Train F1', alpha=0.7)
            axes[1, 1].plot(epochs, val_history['f1'], 'r-', label='Validation F1', alpha=0.7)
            axes[1, 1].set_title('Final Model Training Curves')
            axes[1, 1].set_xlabel('Epoch')
            axes[1, 1].set_ylabel('F1 Score')
            axes[1, 1].legend()
            axes[1, 1].grid(True, alpha=0.3)
        
        plt.tight_layout()
        plt.savefig(save_path / 'enhanced_mobilevit_optimization_results.png', 
                   dpi=300, bbox_inches='tight')
        plt.close()
        
        # 2. Hyperparameter importance (if Optuna study available)
        if 'study' in results['optuna_results']:
            try:
                fig, ax = plt.subplots(figsize=(10, 6))
                plot_param_importances(results['optuna_results']['study'])
                plt.title('Optuna Hyperparameter Importance', fontsize=14, fontweight='bold')
                plt.tight_layout()
                plt.savefig(save_path / 'optuna_hyperparameter_importance.png', 
                           dpi=300, bbox_inches='tight')
                plt.close()
            except Exception as e:
                self.logger.warning(f"Could not create hyperparameter importance plot: {e}")
        
        self.logger.info("Visualizations saved successfully!")
    
    def run_bohb_only(self, train_loader: DataLoader, val_loader: DataLoader, 
                     test_loader: DataLoader, save_dir: str = "runs",
                     bohb_iterations: int = 2) -> Dict[str, Any]:
        """Run BOHB optimization only, skipping Optuna."""
        self.logger.info("🔍 Running BOHB Optimization Only...")
        self.logger.info("=" * 60)
        
        # Create mock Optuna results from existing trial
        mock_optuna_results = {
            'method': 'optuna',
            'best_value': 0.0,  # Will be updated from existing results
            'best_config': {},
            'optimization_time': 0.0,
            'study': None
        }
        
        # Run BOHB optimization
        bohb_results = self.optimize_with_bohb(
            train_loader, val_loader, test_loader,
            n_iterations=bohb_iterations
        )
        
        # Use BOHB results as the best
        best_method = 'bohb'
        best_config = bohb_results['best_config']
        
        # Train final model with best configuration
        self.logger.info(f"\n🎯 Training Final Model with Best Configuration ({best_method})")
        self.logger.info("=" * 60)
        final_results = self._train_single_config(best_config, train_loader, val_loader, test_loader, 0)
        
        # Create comprehensive results
        optimization_results = {
            'optuna_results': mock_optuna_results,
            'bohb_results': bohb_results,
            'final_results': final_results,
            'summary': {
                'best_method': best_method,
                'final_f1': final_results['val_f1'],
                'final_test_f1': final_results['test_f1'],
                'final_test_accuracy': final_results['test_accuracy']
            }
        }
        
        # Save results
        save_path = Path(save_dir)
        save_path.mkdir(exist_ok=True)
        
        results_file = save_path / 'bohb_only_results.json'
        with open(results_file, 'w') as f:
            json.dump(sanitize(optimization_results), f, indent=2)
        
        self.logger.info(f"💾 Results saved to: {results_file}")
        
        return optimization_results
    
    def create_detailed_visualizations(self, detailed_log_path: str, save_dir: str = "runs"):
        """Create detailed visualizations from the detailed log."""
        try:
            with open(detailed_log_path, 'r') as f:
                detailed_log = json.load(f)
            
            save_path = Path(save_dir)
            save_path.mkdir(exist_ok=True)
            
            # Create training progress visualization
            fig, axes = plt.subplots(2, 2, figsize=(15, 10))
            fig.suptitle(f'Detailed Training Results - Trial {detailed_log["trial_number"]}', fontsize=16)
            
            # 1. Training Loss over batches
            batch_logs = detailed_log['batch_logs']
            if batch_logs:
                batch_numbers = [log['batch'] for log in batch_logs]
                losses = [log['loss'] for log in batch_logs]
                axes[0, 0].plot(batch_numbers, losses, 'b-', alpha=0.7)
                axes[0, 0].set_title('Training Loss per Batch (Every 50 batches)')
                axes[0, 0].set_xlabel('Batch Number')
                axes[0, 0].set_ylabel('Loss')
                axes[0, 0].grid(True, alpha=0.3)
            
            # 2. Epoch-wise metrics
            epoch_logs = detailed_log['epoch_logs']
            if epoch_logs:
                epochs = [log['epoch'] for log in epoch_logs]
                train_losses = [log['avg_train_loss'] for log in epoch_logs]
                val_accuracies = [log['val_accuracy'] for log in epoch_logs]
                val_f1s = [log['val_f1'] for log in epoch_logs]
                
                ax2 = axes[0, 1]
                ax2.plot(epochs, train_losses, 'r-', label='Train Loss', marker='o')
                ax2.set_xlabel('Epoch')
                ax2.set_ylabel('Loss', color='r')
                ax2.tick_params(axis='y', labelcolor='r')
                ax2.grid(True, alpha=0.3)
                
                ax2_twin = ax2.twinx()
                ax2_twin.plot(epochs, val_accuracies, 'b-', label='Val Accuracy', marker='s')
                ax2_twin.plot(epochs, val_f1s, 'g-', label='Val F1', marker='^')
                ax2_twin.set_ylabel('Accuracy/F1', color='b')
                ax2_twin.tick_params(axis='y', labelcolor='b')
                
                ax2.set_title('Training Progress per Epoch')
                ax2.legend(loc='upper left')
                ax2_twin.legend(loc='upper right')
            
            # 3. Final metrics comparison
            final_metrics = detailed_log['final_metrics']
            metrics_names = ['Val Accuracy', 'Val F1', 'Test Accuracy', 'Test F1']
            metrics_values = [
                final_metrics['val_accuracy'],
                final_metrics['val_f1'],
                final_metrics['test_accuracy'],
                final_metrics['test_f1']
            ]
            
            bars = axes[1, 0].bar(metrics_names, metrics_values, color=['skyblue', 'lightgreen', 'lightcoral', 'gold'])
            axes[1, 0].set_title('Final Performance Metrics')
            axes[1, 0].set_ylabel('Score')
            axes[1, 0].set_ylim(0, 1)
            
            # Add value labels on bars
            for bar, value in zip(bars, metrics_values):
                axes[1, 0].text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.01,
                               f'{value:.3f}', ha='center', va='bottom')
            
            # 4. Configuration details
            config = detailed_log['config']
            config_text = f"""Configuration Details:
Backbone: {config.get('backbone', 'N/A')}
Optimizer: {config.get('optimizer', 'N/A')}
Learning Rate: {config.get('lr', 'N/A'):.6f}
Weight Decay: {config.get('weight_decay', 'N/A'):.6f}
Dropout: {config.get('dropout', 'N/A'):.3f}
Batch Size: {config.get('batch_size', 'N/A')}
Freeze Backbone: {config.get('freeze_backbone', 'N/A')}

Device: {detailed_log['device']}
Total Epochs: {detailed_log['epochs']}
Trial Number: {detailed_log['trial_number']}"""
            
            axes[1, 1].text(0.05, 0.95, config_text, transform=axes[1, 1].transAxes,
                           fontsize=10, verticalalignment='top', fontfamily='monospace')
            axes[1, 1].set_xlim(0, 1)
            axes[1, 1].set_ylim(0, 1)
            axes[1, 1].axis('off')
            axes[1, 1].set_title('Configuration & Settings')
            
            plt.tight_layout()
            
            # Save the detailed visualization
            viz_filename = f"detailed_visualization_trial_{detailed_log['trial_number']}.png"
            viz_path = save_path / viz_filename
            plt.savefig(viz_path, dpi=300, bbox_inches='tight')
            plt.close()
            
            print(f"📊 Detailed visualization saved to: {viz_path}")
            
        except Exception as e:
            print(f"❌ Error creating detailed visualizations: {e}")
            self.logger.error(f"Error creating detailed visualizations: {e}")


def main():
    """Main function for enhanced MobileViT optimization."""
    parser = argparse.ArgumentParser(description='Enhanced MobileViT Optimization with Optuna and BOHB')
    parser.add_argument('--config', type=str, default='kfu/config/default.yaml',
                       help='Path to configuration file')
    parser.add_argument('--device', type=str, default='auto',
                       help='Device to use (auto, cuda, cpu)')
    parser.add_argument('--optuna-trials', type=int, default=1,
                       help='Number of Optuna trials')
    parser.add_argument('--bohb-iterations', type=int, default=2,
                       help='Number of BOHB iterations')
    parser.add_argument('--skip-optuna', action='store_true',
                       help='Skip Optuna optimization and only run BOHB')
    parser.add_argument('--save-dir', type=str, default='runs',
                       help='Directory to save results')
    parser.add_argument('--log-level', type=str, default='INFO',
                       help='Logging level')
    
    args = parser.parse_args()
    
    # Load configuration
    config = load_yaml(args.config)
    set_seed(config['data']['seed'])
    
    # Create data loaders
    print("📊 Creating data loaders...")
    dataset_manager = DatasetManager(
        data_root=config['data']['root'],
        img_size=config['data']['img_size'],
        seed=config['data']['seed']
    )
    data_loaders = dataset_manager.get_data_loaders(
        batch_size=config['train']['batch_size'],
        num_workers=config['data']['workers']
    )
    
    print(f"   Train batches: {len(data_loaders['train'])}")
    print(f"   Validation batches: {len(data_loaders['val'])}")
    print(f"   Test batches: {len(data_loaders['test'])}")
    
    # Create optimizer
    optimizer = EnhancedMobileViTOptimizer(config, device=args.device, log_level=args.log_level)
    
    # Run comprehensive optimization
    if args.skip_optuna:
        print("🚀 Skipping Optuna - Running BOHB only...")
        results = optimizer.run_bohb_only(
            train_loader=data_loaders['train'],
            val_loader=data_loaders['val'],
            test_loader=data_loaders['test'],
            save_dir=args.save_dir,
            bohb_iterations=args.bohb_iterations
        )
    else:
        results = optimizer.run_comprehensive_optimization(
            train_loader=data_loaders['train'],
            val_loader=data_loaders['val'],
            test_loader=data_loaders['test'],
            save_dir=args.save_dir,
            optuna_trials=args.optuna_trials,
            bohb_iterations=args.bohb_iterations
        )
    
    # Create detailed visualizations if available
    if 'detailed_log_path' in results['final_results']:
        optimizer.create_detailed_visualizations(
            results['final_results']['detailed_log_path'], 
            args.save_dir
        )
    
    # Print final summary
    print("\n" + "="*80)
    print("🎯 ENHANCED MOBILEVIT OPTIMIZATION SUMMARY")
    print("="*80)
    print(f"🏆 Best Method: {results['summary']['best_method'].upper()}")
    print(f"🎯 Best Validation F1: {results['summary']['final_f1']:.4f}")
    print(f"📊 Test F1-Score: {results['summary']['final_test_f1']:.4f}")
    print(f"📈 Test Accuracy: {results['summary']['final_test_accuracy']:.4f}")
    print(f"⏱️  Optuna Time: {results['optuna_results']['optimization_time']:.2f}s")
    print(f"⏱️  BOHB Time: {results['bohb_results']['optimization_time']:.2f}s")
    print("="*80)


if __name__ == "__main__":
    main()