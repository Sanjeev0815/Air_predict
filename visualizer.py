import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.gridspec import GridSpec # Used for subplot customization

class Visualizer:
    """Handles all visualization tasks for the air pollutant forecasting dashboard, 
    using Matplotlib for plotting."""
    
    def __init__(self):
        # Define colors for Matplotlib plots
        self.colors = {
            'o3': '#FF6B6B',  # Red-Orange
            'no2': '#4ECDC4', # Teal
            'actual': '#2E86AB', # Dark Blue
            'predicted': '#A23B72', # Magenta/Purple
            'baseline': '#F18F01', # Yellow-Orange
            'threshold': '#C73E1D', # Dark Red
            'train': '#4ECDC4', # Teal for train data
            'test': '#FF6B6B' # Red-Orange for test data
        }
    
    def plot_pollutant_overview(self, data):
        """Create overview plots of pollutant concentrations using Matplotlib subplots."""
        
        # Create a figure and axes using GridSpec for precise layout
        fig = plt.figure(figsize=(12, 8))
        gs = GridSpec(2, 1, figure=fig, hspace=0.4) # 2 rows, 1 col
        ax1 = fig.add_subplot(gs[0, 0])
        ax2 = fig.add_subplot(gs[1, 0], sharex=ax1)
        
        # Prepare datetime column
        if 'datetime' in data.columns:
            x_axis = pd.to_datetime(data['datetime'])
        else:
            x_axis = data.index
        
        # O3 plot (Top Subplot)
        if 'o3' in data.columns:
            ax1.plot(x_axis, data['o3'], 
                     label='O₃', 
                     color=self.colors['o3'], 
                     linewidth=2)
            ax1.set_title('Ozone (O₃) Concentrations', fontsize=14)
            ax1.set_ylabel('Concentration (µg/m³)')
            ax1.tick_params(axis='x', labelbottom=False) # Hide x-labels on top plot
            ax1.grid(True, linestyle='--', alpha=0.7)
            ax1.legend()
        
        # NO2 plot (Bottom Subplot)
        if 'no2' in data.columns:
            ax2.plot(x_axis, data['no2'], 
                     label='NO₂', 
                     color=self.colors['no2'], 
                     linewidth=2)
            ax2.set_title('Nitrogen Dioxide (NO₂) Concentrations', fontsize=14)
            ax2.set_xlabel('Date')
            ax2.set_ylabel('Concentration (µg/m³)')
            ax2.tick_params(axis='x', rotation=45)
            ax2.grid(True, linestyle='--', alpha=0.7)
            ax2.legend()
        
        fig.suptitle('Air Pollutant Concentration Overview', fontsize=16, y=0.95)
        plt.tight_layout(rect=[0, 0.03, 1, 0.9]) # Adjust layout for suptitle
        return fig
    
    def plot_processed_data(self, processed_data):
        """Visualize processed data with train/test split using Matplotlib subplots."""
        try:
            if 'y_train' not in processed_data or processed_data['y_train'].empty:
                 raise ValueError("Training data is missing or empty.")
            
            y_train = processed_data['y_train']
            y_test = processed_data.get('y_test', pd.DataFrame()) # Handle case where y_test might be missing

            fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(12, 8), sharex=False, gridspec_kw={'hspace': 0.4})
            
            # --- Training Data Plot ---
            ax1.set_title('Training Data - Pollutant Time Series', fontsize=14)
            train_index = y_train.index
            
            # Plot all target columns (e.g., 'o3' and 'no2') in the training set
            for i, col in enumerate(y_train.columns):
                color = self.colors.get(col, self.colors['train'])
                ax1.plot(train_index, y_train[col], 
                         label=f'{col.upper()}', 
                         color=color, 
                         linewidth=1.5,
                         alpha=0.8)

            ax1.set_ylabel('Concentration (µg/m³)')
            ax1.tick_params(axis='x', labelbottom=False)
            ax1.legend(loc='upper right')
            ax1.grid(True, linestyle=':', alpha=0.6)
            
            # --- Test Data Plot ---
            ax2.set_title('Test Data - Pollutant Time Series', fontsize=14)
            if not y_test.empty:
                test_index = y_test.index
                
                # Plot all target columns in the test set
                for i, col in enumerate(y_test.columns):
                    color = self.colors.get(col, self.colors['test'])
                    ax2.plot(test_index, y_test[col], 
                             label=f'{col.upper()}', 
                             color=color, 
                             linewidth=1.5, 
                             linestyle='--')
                
                ax2.set_xlabel('Date')
                ax2.set_ylabel('Concentration (µg/m³)')
                ax2.tick_params(axis='x', rotation=45)
                ax2.legend(loc='upper right')
                ax2.grid(True, linestyle=':', alpha=0.6)
            else:
                ax2.text(0.5, 0.5, 'Test Data Not Available/Empty', transform=ax2.transAxes, 
                         ha='center', va='center', color='red', fontsize=12)
                ax2.set_xlabel('Date')
            
            fig.suptitle('Processed Data: Train/Test Split Overview', fontsize=16, y=0.98)
            plt.tight_layout(rect=[0, 0.03, 1, 0.95])
            return fig
            
        except Exception as e:
            # Return a simple error plot
            fig, ax = plt.subplots(figsize=(8, 4))
            ax.text(0.5, 0.5, f"Error creating visualization: {str(e)}", 
                    ha='center', va='center', color='red', fontsize=16, transform=ax.transAxes)
            ax.set_title('Processed Data Visualization Error')
            ax.axis('off')
            return fig
    
    def plot_training_history(self, history, target):
        """Plot training history (Loss and MAE) for neural network models using Matplotlib subplots."""
        
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 4), sharex=True, gridspec_kw={'wspace': 0.3})
        
        epochs = range(1, len(history.history['loss']) + 1)
        
        # --- Loss Plot (Left Subplot) ---
        ax1.plot(epochs, history.history['loss'], 
                 label='Training Loss', 
                 color=self.colors['o3'], 
                 linewidth=2)
        
        if 'val_loss' in history.history:
            ax1.plot(epochs, history.history['val_loss'], 
                     label='Validation Loss', 
                     color=self.colors['no2'], 
                     linewidth=2)
        
        ax1.set_title('Model Loss', fontsize=14)
        ax1.set_xlabel('Epoch')
        ax1.set_ylabel('Loss (MSE)')
        ax1.legend()
        ax1.grid(True, linestyle=':', alpha=0.6)
        
        # --- MAE Plot (Right Subplot) ---
        if 'mae' in history.history:
            ax2.plot(epochs, history.history['mae'], 
                     label='Training MAE', 
                     color=self.colors['o3'], 
                     linewidth=2, 
                     linestyle='--')
            
            if 'val_mae' in history.history:
                ax2.plot(epochs, history.history['val_mae'], 
                         label='Validation MAE', 
                         color=self.colors['no2'], 
                         linewidth=2, 
                         linestyle='--')
                
            ax2.set_title('Model MAE', fontsize=14)
            ax2.set_xlabel('Epoch')
            ax2.set_ylabel('Mean Absolute Error (MAE)')
            ax2.legend()
            ax2.grid(True, linestyle=':', alpha=0.6)
        else:
            ax2.set_title('MAE History Not Available')
            ax2.text(0.5, 0.5, 'MAE metric not recorded.', transform=ax2.transAxes, ha='center', va='center')
        
        fig.suptitle(f'Training History - {target.upper()}', fontsize=16)
        plt.tight_layout(rect=[0, 0.03, 1, 0.95])
        return fig
    
    def plot_forecast_results(self, predictions, target, baseline_predictions=None, safety_threshold=None):
        """Plot forecast results (Actual, Predicted, Future, Baseline, Threshold) using Matplotlib."""
        
        fig, ax = plt.subplots(figsize=(14, 6))
        
        # Create time index for plotting
        actual_data = predictions.get('actual', [])
        predicted_data = predictions.get('predicted', [])
        future_data = predictions.get('future_predictions', [])

        n_historical = len(actual_data)
        n_future = len(future_data)
        
        historical_times = np.arange(n_historical)
        future_times = np.arange(n_historical, n_historical + n_future)
        
        # 1. Actual values (Historical)
        if actual_data:
            ax.plot(historical_times, actual_data, 
                    label='Actual', 
                    color=self.colors['actual'], 
                    linewidth=3, 
                    alpha=0.8)
        
        # 2. Predicted values (Historical)
        if predicted_data:
            ax.plot(historical_times, predicted_data, 
                    label='Predicted (Historical)', 
                    color=self.colors['predicted'], 
                    linewidth=2, 
                    linestyle='--')
        
        # 3. Future Forecast
        if future_data:
            ax.plot(future_times, future_data, 
                    label='Future Forecast', 
                    color=self.colors['predicted'], 
                    linewidth=3, 
                    marker='o', 
                    markersize=6, 
                    linestyle='-')
            
            # Add vertical line to separate historical and future
            ax.axvline(x=n_historical - 0.5, 
                       color='gray', 
                       linestyle=':', 
                       linewidth=1.5, 
                       label='Forecast Start')
            ax.text(n_historical - 0.5, ax.get_ylim()[1] * 0.95, 'Forecast Start', 
                    rotation=90, va='top', ha='right', color='gray')
        
        # 4. Baseline predictions
        if baseline_predictions and 'predicted' in baseline_predictions:
            baseline_data = baseline_predictions['predicted']
            ax.plot(np.arange(len(baseline_data)), baseline_data, 
                    label='Baseline (Persistence)', 
                    color=self.colors['baseline'], 
                    linewidth=1.5, 
                    linestyle='-.', 
                    alpha=0.7)

        # 5. Safety Threshold
        if safety_threshold:
            ax.axhline(y=safety_threshold, 
                       color=self.colors['threshold'], 
                       linestyle='--', 
                       linewidth=2, 
                       label=f'WHO Threshold ({safety_threshold} µg/m³)')
            ax.text(ax.get_xlim()[1], safety_threshold, f'Threshold', 
                    ha='right', va='bottom', color=self.colors['threshold'])
        
        ax.set_title(f'{target.upper()} Concentration Forecast', fontsize=16)
        ax.set_xlabel('Time Steps')
        ax.set_ylabel('Concentration (µg/m³)')
        ax.legend(loc='best')
        ax.grid(True, linestyle=':', alpha=0.6)
        plt.tight_layout()
        return fig
    
    def plot_model_comparison(self, model_results):
        """Create bar chart comparison plot of different models' performance for RMSE, MAE, R²."""
        
        models = list(model_results.keys())
        metrics = ['RMSE', 'MAE', 'R²']
        
        fig, axes = plt.subplots(1, 3, figsize=(15, 5), sharex=True, gridspec_kw={'wspace': 0.4})
        colors = [self.colors['o3'], self.colors['no2'], self.colors['predicted']] # Use primary colors
        
        for i, metric in enumerate(metrics):
            values = [model_results[model][metric] for model in models]
            ax = axes[i]
            
            # Create Bar Chart
            bars = ax.bar(models, values, color=colors[i % len(colors)], alpha=0.8)
            
            # Add value labels on top of the bars
            for bar in bars:
                yval = bar.get_height()
                ax.text(bar.get_x() + bar.get_width()/2, yval + (yval * 0.05), 
                        f'{yval:.3f}', ha='center', va='bottom')
                
            ax.set_title(metric, fontsize=14)
            ax.set_xlabel('Model')
            
            # Specific label for R²
            if metric == 'R²':
                ax.set_ylabel(metric)
                ax.set_ylim([0, 1.1])
            elif metric in ['RMSE', 'MAE']:
                 ax.set_ylabel('Error Value')
                 # Adjust y-limit to ensure labels are visible
                 max_val = max(values) * 1.2 if values else 1
                 ax.set_ylim([0, max_val])

            ax.grid(axis='y', linestyle=':', alpha=0.6)
            ax.tick_params(axis='x', rotation=30)

        fig.suptitle('Model Performance Comparison', fontsize=16)
        plt.tight_layout(rect=[0, 0.03, 1, 0.95])
        return fig
    
    def plot_feature_importance(self, model, feature_names, top_n=15):
        """Plot feature importance for tree-based models using a Matplotlib horizontal bar chart."""
        if not hasattr(model, 'feature_importances_'):
            return None
        
        importance = model.feature_importances_
        
        # Create DataFrame for easier handling
        importance_df = pd.DataFrame({
            'feature': feature_names,
            'importance': importance
        }).sort_values('importance', ascending=True).tail(top_n)
        
        fig, ax = plt.subplots(figsize=(10, 8))
        
        # Create horizontal bar chart
        bars = ax.barh(importance_df['feature'], 
                       importance_df['importance'], 
                       color=self.colors['no2'], 
                       alpha=0.8)
        
        # Add labels to bars
        for bar in bars:
            ax.text(bar.get_width(), bar.get_y() + bar.get_height()/2, 
                    f'{bar.get_width():.3f}', 
                    va='center')
        
        ax.set_title(f'Top {top_n} Feature Importances', fontsize=16)
        ax.set_xlabel('Importance Score')
        ax.set_ylabel('Features')
        ax.grid(axis='x', linestyle=':', alpha=0.6)
        plt.tight_layout()
        
        return fig
    
    def create_correlation_heatmap(self, data):
        """Create correlation heatmap of variables using Matplotlib."""
        
        numeric_cols = data.select_dtypes(include=[np.number]).columns
        correlation_matrix = data[numeric_cols].corr()
        
        fig, ax = plt.subplots(figsize=(10, 8))
        
        # Use Matplotlib's imshow for the heatmap
        cax = ax.imshow(correlation_matrix, 
                        cmap='RdBu', 
                        aspect='auto', 
                        vmin=-1, 
                        vmax=1)
        
        # Add color bar
        fig.colorbar(cax, ax=ax)
        
        # Set ticks and labels
        ax.set_xticks(np.arange(len(correlation_matrix.columns)))
        ax.set_yticks(np.arange(len(correlation_matrix.columns)))
        ax.set_xticklabels(correlation_matrix.columns, rotation=90)
        ax.set_yticklabels(correlation_matrix.columns)
        
        # Add correlation values (annotations)
        for i in range(len(correlation_matrix.columns)):
            for j in range(len(correlation_matrix.columns)):
                text = ax.text(j, i, f'{correlation_matrix.iloc[i, j]:.2f}',
                               ha="center", va="center", color="black", fontsize=8)

        ax.set_title('Variable Correlation Matrix', fontsize=16)
        plt.tight_layout()
        
        return fig
