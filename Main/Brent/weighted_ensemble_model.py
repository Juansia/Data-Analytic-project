import numpy as np
import torch
import pandas as pd
import os
import matplotlib.pyplot as plt
from sklearn.metrics import mean_squared_error, mean_absolute_error
from sklearn.model_selection import train_test_split
from utils.evaluation import model_evaluation, plot_trues_preds
from utils.helper import print_params
from fitness.fitness import save_results
from data.data import denormolize_data

# Try to import mealpy components with compatibility handling
try:
    from mealpy.swarm_based import GWO
    from mealpy import Problem
except ImportError:
    from mealpy.swarm_based import GWO
    Problem = None

class WeightedEnsembleModel:
    def __init__(self, model_names, model_predictions, true_values, result_path):
        """
        Initialize the Weighted Ensemble Model
        
        Args:
            model_names: List of model names in the ensemble
            model_predictions: List of numpy arrays containing predictions from each model
            true_values: Numpy array of true values
            result_path: Path to save results
        """
        self.model_names = model_names
        self.model_predictions = model_predictions
        self.true_values = true_values
        self.result_path = result_path
        self.weights = None
        self.ensemble_predictions = None
        self.metrics = None
        
    def optimize_weights(self, pop_size=10, iterations=20, seed=42, validation_split_ratio=0.2):
        """
        Optimize the weights using Grey Wolf Optimizer
        
        Args:
            pop_size: Population size for GWO
            iterations: Number of iterations for GWO
            seed: Random seed for reproducibility
            validation_split_ratio: Fraction of data to use for validation during weight optimization.
                                   If 0 or None, uses all data (original behavior).
        """
        print(f"Optimizing ensemble weights using GWO with population={pop_size}, iterations={iterations}")

        if validation_split_ratio and 0 < validation_split_ratio < 1:
            print(f"Using validation split ratio: {validation_split_ratio}")
            # First check if all predictions have the same length
            pred_lengths = [len(pred) for pred in self.model_predictions]
            true_length = len(self.true_values)
            
            if len(set(pred_lengths)) > 1 or any(pl != true_length for pl in pred_lengths):
                print(f"Warning: Inconsistent lengths detected - true_values: {true_length}, predictions: {pred_lengths}")
                # Use minimum length to ensure consistency
                min_length = min(min(pred_lengths), true_length)
                self.model_predictions = [pred[:min_length] for pred in self.model_predictions]
                self.true_values = self.true_values[:min_length]
                num_samples = min_length
            else:
                num_samples = true_length
            
            # For time series, use a simple split instead of train_test_split
            # to maintain temporal order
            split_point = int(num_samples * (1 - validation_split_ratio))
            
            # Use the last portion for validation (more recent data)
            train_indices = np.arange(split_point)
            val_indices = np.arange(split_point, num_samples)
            
            opt_true_values = self.true_values[val_indices]
            opt_model_predictions = [pred[val_indices] for pred in self.model_predictions]
            
            print(f"Total samples: {num_samples}, Train: {len(train_indices)}, Validation: {len(val_indices)}")
            print(f"Optimizing weights on {len(opt_true_values)} validation samples.")
        else:
            print("No validation split. Optimizing weights on all provided data.")
            opt_true_values = self.true_values
            opt_model_predictions = self.model_predictions

        # Define the objective function
        def ensemble_objective(weights):
            # Normalize weights to sum to 1
            current_sum = np.sum(weights)
            if current_sum == 0:
                normalized_weights = np.ones(len(weights)) / len(weights)
            else:
                normalized_weights = weights / current_sum
            
            # Calculate ensemble prediction
            ensemble_pred = np.zeros_like(opt_true_values)
            for i in range(len(normalized_weights)):
                ensemble_pred += normalized_weights[i] * opt_model_predictions[i]
            
            # Calculate MSE
            mse = mean_squared_error(opt_true_values, ensemble_pred)
            return mse
        
        # Define bounds
        n_dims = len(self.model_names)
        lb = [0.0] * n_dims
        ub = [1.0] * n_dims
        bounds = list(zip(lb, ub))
        
        # Try different approaches based on mealpy version
        try:
            # Approach 1: Using Problem class with FloatVar if available
            if Problem is not None:
                try:
                    from mealpy.utils.space import FloatVar
                    # Create bounds using FloatVar
                    float_bounds = [FloatVar(lb=0.0, ub=1.0, name=f"w{i}") for i in range(n_dims)]
                    problem = Problem(bounds=float_bounds, minmax="min", obj_func=ensemble_objective)
                    optimizer = GWO.OriginalGWO(epoch=iterations, pop_size=pop_size)
                    best_position, best_fitness = optimizer.solve(problem)
                except ImportError:
                    # Fallback to regular bounds
                    problem = Problem(bounds=bounds, minmax="min", obj_func=ensemble_objective)
                    optimizer = GWO.OriginalGWO(epoch=iterations, pop_size=pop_size)
                    best_position, best_fitness = optimizer.solve(problem)
            else:
                raise ImportError("Problem class not available")
        except Exception as e:
            print(f"Mealpy optimization failed: {e}")
            try:
                # Direct scipy optimization as it seems to work well
                from scipy.optimize import differential_evolution
                
                def scipy_objective(weights):
                    # Same normalization and calculation as before
                    current_sum = np.sum(weights)
                    if current_sum == 0:
                        normalized_weights = np.ones(len(weights)) / len(weights)
                    else:
                        normalized_weights = weights / current_sum
                    
                    ensemble_pred = np.zeros_like(opt_true_values)
                    for i in range(len(normalized_weights)):
                        ensemble_pred += normalized_weights[i] * opt_model_predictions[i]
                    
                    return mean_squared_error(opt_true_values, ensemble_pred)
                
                # Use differential evolution as a robust global optimizer
                result = differential_evolution(
                    scipy_objective, 
                    bounds=bounds,
                    maxiter=iterations,
                    popsize=int(pop_size/2),  # scipy uses smaller population
                    seed=seed
                )
                best_position = result.x
                print("Successfully optimized using scipy.optimize")
            except Exception as e_scipy:
                print(f"Scipy optimization also failed: {e_scipy}")
                # Final fallback: Simple equal weights
                print("All optimization approaches failed. Using equal weights.")
                best_position = np.ones(n_dims) / n_dims
        
        # Normalize the final best_position (weights) to sum to 1
        final_sum = np.sum(best_position)
        if final_sum == 0:
            self.weights = np.ones(len(self.model_names)) / len(self.model_names)
            print("Warning: All optimized weights were zero. Assigning equal weights.")
        else:
            self.weights = best_position / final_sum
        
        print("Optimized weights:")
        for i, (model, weight) in enumerate(zip(self.model_names, self.weights)):
            print(f"{model}: {weight:.4f}")
        
        # Calculate ensemble predictions using the full original dataset
        # First check for shape consistency in the full dataset
        full_pred_shapes = [pred.shape for pred in self.model_predictions]
        if len(set(full_pred_shapes)) > 1:
            print(f"Warning: Full model predictions have different shapes: {full_pred_shapes}")
            # Find the minimum common length
            min_full_length = min(pred.shape[0] for pred in self.model_predictions)
            print(f"Truncating all full predictions to minimum length: {min_full_length}")
            self.model_predictions = [pred[:min_full_length] for pred in self.model_predictions]
            self.true_values = self.true_values[:min_full_length]
        
        self.ensemble_predictions = np.zeros_like(self.true_values)
        for i in range(len(self.model_names)):
            self.ensemble_predictions += self.weights[i] * self.model_predictions[i]
        
        # Calculate metrics on the full original dataset
        mae, mse, rmse = model_evaluation(self.true_values, self.ensemble_predictions)
        self.metrics = {
            'MAE': mae,
            'MSE': mse,
            'RMSE': rmse
        }
        
        print(f"Ensemble Metrics (on full data) - MAE: {mae:.4f}, MSE: {mse:.4f}, RMSE: {rmse:.4f}")
        
        # Save results
        self.save_results()
        
        return self.weights, self.ensemble_predictions, self.metrics
    
    def save_results(self):
        """Save ensemble results and visualizations"""
        # Save weights
        weights_df = pd.DataFrame({
            'Model': self.model_names,
            'Weight': self.weights
        })
        weights_df.to_csv(os.path.join(self.result_path, 'ensemble_weights.csv'), index=False)
        
        # Save metrics
        metrics_df = pd.DataFrame({
            'Metric': list(self.metrics.keys()),
            'Value': list(self.metrics.values())
        })
        metrics_df.to_csv(os.path.join(self.result_path, 'ensemble_metrics.csv'), index=False)
        
        # Plot true vs predicted
        plot_trues_preds(self.true_values, self.ensemble_predictions, 
                         os.path.join(self.result_path, 'ensemble_predictions.jpg'))
        
        # Save predictions
        save_results(self.true_values, self.ensemble_predictions, self.result_path)
        
        # Plot weights as bar chart
        plt.figure(figsize=(10, 6))
        plt.bar(self.model_names, self.weights)
        plt.title('Ensemble Model Weights')
        plt.xlabel('Model')
        plt.ylabel('Weight')
        plt.xticks(rotation=45)
        plt.tight_layout()
        plt.savefig(os.path.join(self.result_path, 'ensemble_weights.jpg'))
        plt.close()
        
        return True

def run_ensemble(models_data, args):
    """
    Run the ensemble model with the given models and data
    
    Args:
        models_data: Dictionary with model names as keys and (trues, preds) as values
        args: Arguments for the ensemble model
    
    Returns:
        Ensemble predictions and metrics
    """
    # Create result directory
    result_path = os.path.join("./results", "ensemble_" + "_".join(models_data.keys()))
    os.makedirs(result_path, exist_ok=True)
    
    # Extract model names, predictions and true values
    model_names = list(models_data.keys())
    model_predictions = [models_data[model][1] for model in model_names]
    true_values = models_data[model_names[0]][0]  # Assuming all models have same true values
    
    # Initialize and run ensemble model
    ensemble = WeightedEnsembleModel(model_names, model_predictions, true_values, result_path)
    
    # You can get validation_split_ratio from args if you add it there
    # e.g., validation_split_ratio = args.ensemble_validation_split if hasattr(args, 'ensemble_validation_split') else 0.2
    validation_split_ratio = getattr(args, 'ensemble_validation_split', 0.2) # Default to 0.2 if not in args

    weights, ensemble_preds, metrics = ensemble.optimize_weights(
        pop_size=args.ensemble_pop_size, 
        iterations=args.ensemble_iterations,
        seed=args.seed, # Assuming GWO constructor or mealpy global seed handles this
        validation_split_ratio=validation_split_ratio
    )
    
    return ensemble_preds, metrics