import os
import sys
import io
import base64
from typing import List, Optional
import numpy as np
import pandas as pd
from scipy import stats
from scipy.spatial.distance import cdist
import torch
from torch.utils.data import DataLoader
from torch.nn import functional as F
from torchvision.transforms import v2
from torchvision.utils import save_image
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline
from sklearn.model_selection import cross_val_score
from sklearn.metrics import (
    accuracy_score, 
    log_loss, 
    confusion_matrix, 
    classification_report, 
    roc_curve, 
    auc, 
    roc_auc_score
)
import matplotlib.pyplot as plt
from matplotlib.colors import LinearSegmentedColormap, ListedColormap
import seaborn as sns
import plotly.express as px
from PIL import Image
from dash import Dash, html, dcc, Output, Input, no_update

# Optional imports with error handling
try:
    import umap
except ImportError:
    print("UMAP not installed. Some visualization features will be disabled.")

try:
    import piq
except ImportError:
    print("PIQ not installed. Some quality metrics will be disabled.")

# Local imports
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..')))
from src.data.dataset import ZarrCellDataset
from src.data.dataloader import collate_wrapper
from src.data.utils import read_config
from src.models.VAE_resnet18 import VAEResNet18

# Increase recursion limit for deep operations
sys.setrecursionlimit(100000)

class VAE_Evaluator:
    def __init__(self, config):
        self.config = config
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.base_output_dir = config['output_dir']
        self.eval_output_dir = os.path.join(self.base_output_dir, config['checkpoint_epoch'])
        self.output_dir = self._create_output_dirs()
        self._plot_epoch_log_metrics()
        self.model = self._init_model()
    
    def _create_output_dirs(self):
        os.makedirs(self.base_output_dir, exist_ok=True)
        os.makedirs(self.eval_output_dir, exist_ok=True)
    
    def _init_model(self):
        if self.config['model_name'] == 'VAE_ResNet18':
            model = VAEResNet18(nc=self.config['nc'], z_dim=self.config['z_dim'])
        else:
            raise ValueError(f"Model {self.config['model_name']} not supported.")
        
        if self.config['checkpoint_epoch'] is None:       
            checkpoint_path = os.path.join(self.config['checkpoint_dir'], 'best_model.pt')
            model, _ = self._load_checkpoint(checkpoint_path, model)
            return model.to(self.device)
        else:
            checkpoint_path = os.path.join(self.config['checkpoint_dir'], f'{self.config["checkpoint_epoch"]}.pt')
            model, _ = self._load_checkpoint(checkpoint_path, model)
            return model.to(self.device)
        
    def _load_checkpoint(self, checkpoint_path, model):
        print(f"Loading checkpoint from {checkpoint_path}...")
        checkpoint = torch.load(checkpoint_path, map_location=self.device)
        print(f"Loading checkpoint from epoch {checkpoint['epoch']}...")
        model.load_state_dict(checkpoint['model_state_dict'])
        print(f"Epoch: {checkpoint['epoch']}")
        print(f"Train Loss: {checkpoint['train_loss']:.4f}")
        # print(f"Train Reconstruction Loss: {checkpoint['train_recon']:.4f}")
        # print(f"Train KL Divergence: {checkpoint['train_kld']:.4f}")
        print(f"Val Loss: {checkpoint['val_loss']:.4f}")
        # print(f"Val Reconstruction Loss: {checkpoint['val_recon']:.4f}")
        # print(f"Val KL Divergence: {checkpoint['val_kld']:.4f}")
        return model, checkpoint['epoch']

    def _plot_epoch_log_metrics(self):
        # Read the epoch log CSV file
        epoch_log_path = os.path.join(self.config['log_dir'], 'epoch_log.csv')
        if not os.path.exists(epoch_log_path):
            print(f"Epoch log file not found at {epoch_log_path}. Skipping epoch log plot.")
            return

        df = pd.read_csv(epoch_log_path)

        metrics = ['loss', 'recon', 'kld']
        fig, axes = plt.subplots(3, 1, figsize=(10, 15))
        fig.suptitle('Training and Validation Metrics Over Epochs', fontsize=16)

        def smooth(y, box_pts):
            box = np.ones(box_pts)/box_pts
            y_smooth = np.convolve(y, box, mode='same')
            return y_smooth

        for i, metric in enumerate(metrics):
            ax = axes[i]
            
            # Smooth the lines
            train_smooth = smooth(df[f'train_{metric}'], 5)
            val_smooth = smooth(df[f'val_{metric}'], 5)
            
            ax.plot(df['epoch'], train_smooth, label=f'Train {metric.replace("_", " ").capitalize()}')
            ax.plot(df['epoch'], val_smooth, label=f'Validation {metric.replace("_", " ").capitalize()}')
            
            ax.set_xlabel('Epoch')
            ax.set_ylabel(metric.replace('_', ' ').capitalize())
            ax.set_title(f'{metric.replace("_", " ").capitalize()} vs. Epoch')
            ax.legend()
            
            # Remove grid
            ax.grid(False)
            
            # Remove top and right spines
            ax.spines['top'].set_visible(False)
            ax.spines['right'].set_visible(False)

        plt.tight_layout()
        plt.savefig(os.path.join(self.base_output_dir, 'epoch_log_metrics.png'), dpi=300, bbox_inches='tight')
        plt.close()

    def _create_dataloader(self, split, drop_last=True):
        dataset_mean, dataset_std = read_config(self.config['yaml_file'])
        normalizations = v2.Compose([v2.CenterCrop(self.config['crop_size'])])
                                     
        dataset = ZarrCellDataset(
            parent_dir=self.config['parent_dir'],
            csv_file=self.config['csv_file'],
            split=split,
            channels=self.config['channels'],
            mask=self.config['transform'],
            normalizations=normalizations,
            interpolations=None,
            mean=dataset_mean,
            std=dataset_std,
        )
        dataloader = DataLoader(
            dataset, 
            batch_size=self.config['batch_size'],
            shuffle=False,
            num_workers=self.config['num_workers'],
            collate_fn=collate_wrapper(
                metadata_keys=self.config['metadata_keys'],
                images_keys=self.config['images_keys']),
            drop_last=drop_last
        )

        return dataloader

    def _evaluate_model(self, dataloader, split):
        self.model.eval()
        total_loss = 0
        total_recon = 0
        total_kld = 0

        all_latent_vectors = []
        all_metadata = []
        
        with torch.no_grad():
            for batch_idx, batch in enumerate(dataloader):
                data = batch['cell_image'].to(self.device)
                metadata = [batch[key] for key in self.config['metadata_keys']]

                recon_batch, mu, logvar = self.model(data)

                if self.config['loss_type'] == "MSE":
                    RECON = F.mse_loss(recon_batch, data, reduction='none')
                elif self.config['loss_type'] == "L1":
                    RECON = F.l1_loss(recon_batch, data, reduction='none')
                elif self.config['loss_type'] == "SSIM":
                    x_norm = (data - data.min()) / (data.max() - data.min())
                    recon_x_norm = (recon_batch - recon_batch.min()) / (recon_batch.max() - recon_batch.min())
                    ssim = self.loss_ssim(recon_x_norm, x_norm)
                    RECON = F.l1_loss(recon_batch, data, reduction='none') + ssim * self.config['alpha']
                
                KLD = -0.5 * torch.sum(1 + logvar - mu.pow(2) - logvar.exp(), dim = (1, 2, 3))
                
                denominator = torch.prod(torch.tensor(data.shape)) if torch.prod(torch.tensor(data.shape)) > torch.prod(torch.tensor(mu.shape)) else torch.prod(torch.tensor(mu.shape))
                RECON = RECON.sum() / denominator
                KLD = KLD.sum() / denominator

                loss = RECON + KLD * self.config['beta']
                
                total_loss += loss.item()
                total_recon += RECON.item()
                total_kld += KLD.item()

                if batch_idx == 0:
                    self._save_image(data, recon_batch, metadata, split)

                if self.config['sampling_number'] > 1:
                    # print('Sampling {} times...'.format(self.config['sampling_number']))
                    for i in range(self.config['sampling_number']):
                        z = self.model.reparameterize(mu, logvar)
                        all_latent_vectors.append(z.cpu())
                        all_metadata.extend(zip(*metadata))
                else:
                    all_latent_vectors.append(mu.cpu())
                    all_metadata.extend(zip(*metadata))
        
        total_loss /= len(dataloader)
        total_recon /= len(dataloader)
        total_kld /= len(dataloader)
        latent_vectors = torch.cat(all_latent_vectors, dim=0)
        
        return total_loss, total_recon, total_kld, latent_vectors, all_metadata

    def evaluate(self):
        results = {}
        for split in ['train', 'val', 'test']:
            dataloader = self._create_dataloader(split, drop_last=False)
            print(f"Evaluating on {split} data...")

            loss, recon, kld, latents, metadata = self._evaluate_model(dataloader, split)
            
            print(f"{split.capitalize()} - Loss: {loss:.4f}, RECON: {recon:.4f}, KLD: {kld:.4f}")

            if self.config['model_name'] == 'VAE_ResNet18':
                latents = latents.view(latents.shape[0], -1)
                print(f"Latent shape: {latents.shape}")
            else:
                raise ValueError(f"Model {self.config['model_name']} not supported.")
            
            metadata_df = pd.DataFrame(metadata, columns=self.config['metadata_keys'])
            latent_df = pd.DataFrame(latents.numpy(), columns=[f'latent_{i}' for i in range(latents.shape[1])])
            
            results[split] = {
                'loss': loss, 
                'recon': recon, 
                'kld': kld,
                'metadata': metadata_df,
                'latent_df': latent_df
            }        

        # Train classifier on train set, evaluate on val and test sets
        self._train_classifier(results['train']['latent_df'], results['train']['metadata']['gene'])
        
        for split in ['train', 'val', 'test']:
            self._plot_clustermap(results[split]['latent_df'], results[split]['metadata'], split)
            clf_results = self._evaluate_classifier(results[split]['latent_df'], results[split]['metadata']['gene'])
            results[split]['clf_results'] = clf_results

            # Generate and save split-specific results
            self._save_split_results(split, results[split])

        # Summarize results across all splits
        self._summarize_results(results)

        # Generate additional visualizations
        self._generate_latent_space_visualizations(results)

        return results

    def _train_classifier(self, train_latents, train_labels):
        # Get unique class names
        self.class_names = np.unique(train_labels)

        # Create LogisticRegression classifier
        self.classifier = LogisticRegression(
            C=1.0,
            class_weight='balanced',
            random_state=42,
            max_iter=1000  # Increased max iterations to help with convergence
        )
        
        # Train on the full training set
        self.classifier.fit(train_latents, train_labels)

        # Print convergence status
        print(f"Convergence status: {'Converged' if self.classifier.n_iter_ < self.classifier.max_iter else 'Did not converge'}")
        print(f"Number of iterations: {self.classifier.n_iter_}")

    def _evaluate_classifier(self, latents, labels):
        # Make predictions
        y_pred = self.classifier.predict(latents)
        y_prob = self.classifier.predict_proba(latents)
        
        # Calculate metrics
        accuracy = accuracy_score(labels, y_pred)
        loss = log_loss(labels, y_prob)
        
        # Generate classification report and confusion matrix
        class_report = classification_report(labels, y_pred, target_names=self.class_names)
        conf_matrix = confusion_matrix(labels, y_pred)
        
        return {
            'loss': loss,
            'accuracy': accuracy,
            'predictions': y_pred,
            'true_labels': labels,
            'probabilities': y_prob,
            'classification_report': class_report,
            'confusion_matrix': conf_matrix
        }
    
    def _plot_clustermap(self, latents, metadata, split):
            # Check for infinite or NaN values
        if np.any(np.isinf(latents)) or np.any(np.isnan(latents)):
            raise ValueError("Data contains infinite or NaN values")

        # Get unique gene symbols and assign colors
        gene_symbols = metadata['gene'].unique()
        n_colors = len(gene_symbols)
        colors = plt.cm.rainbow(np.linspace(0, 1, n_colors))
        color_map = dict(zip(gene_symbols, colors))
    
        # Create a color list for each row
        row_colors = metadata['gene'].map(color_map).tolist()
    
        # Create a clustermap with adjusted parameters
        g = sns.clustermap(
            latents,
            row_cluster=True,
            col_cluster=True,
            cmap='viridis',
            figsize=(20, 20),
            cbar_kws={'label': 'Standardized Value'},
            xticklabels=False,
            yticklabels=False,
            row_colors=row_colors
        )
    
        # Adjust colorbar limits to show more variation
        vmin, vmax = np.percentile(latents.values, [5, 95])
        g.ax_heatmap.collections[0].set_clim(vmin, vmax)
    
        # Add title
        plt.suptitle("Feature Clustermap", fontsize=16, y=1.02)
    
        # Create a custom legend for gene symbols
        legend_elements = [plt.Rectangle((0, 0), 1, 1, fc=color_map[gene], label=gene) for gene in gene_symbols]
        plt.legend(handles=legend_elements, title='Gene Symbols', 
                loc='center left', bbox_to_anchor=(2, 0.5), ncol=1)
    
        # Save the figure
        plt.savefig(os.path.join(self.eval_output_dir, f'{split}_clustermap.png'), bbox_inches='tight')
        plt.close()

    def _save_split_results(self, split, split_results):
        # Save VAE metrics
        vae_metrics_df = pd.DataFrame({
            'loss': [split_results['loss']],
            'recon': [split_results['recon']],
            'kld': [split_results['kld']]
        })
        vae_metrics_df.to_csv(os.path.join(self.eval_output_dir, f'{split}_vae_metrics.csv'), index=False)

        # Calculate AUC for each class
        y_true = split_results['clf_results']['true_labels']
        y_pred_proba = split_results['clf_results']['probabilities']
        auc_scores = {}
        for i, class_name in enumerate(self.class_names):
            auc_scores[class_name] = roc_auc_score((np.array(y_true) == class_name).astype(int), 
                                                [prob[i] for prob in y_pred_proba])

        # Save classifier metrics
        clf_metrics_df = pd.DataFrame({
            'loss': [split_results['clf_results']['loss']],
            'accuracy': [split_results['clf_results']['accuracy']],
            **{f'auc_{class_name}': [auc_scores[class_name]] for class_name in self.class_names},
            'macro_avg_auc': [np.mean(list(auc_scores.values()))]
        })
        clf_metrics_df.to_csv(os.path.join(self.eval_output_dir, f'{split}_clf_metrics.csv'), index=False)

        # Save latent vectors with metadata
        combined_df = pd.concat([split_results['metadata'], split_results['latent_df']], axis=1)
        combined_df.to_csv(os.path.join(self.eval_output_dir, f'{split}_latent_vectors.csv'), index=False)

        # Generate and save confusion matrix
        conf_matrix = split_results['clf_results']['confusion_matrix']
        self._plot_confusion_matrix(conf_matrix, split)

        # Generate and save classification report
        class_report = split_results['clf_results']['classification_report']
        with open(os.path.join(self.eval_output_dir, f'{split}_classification_report.txt'), 'w') as f:
            f.write(class_report)

        # Generate and save ROC curve
        self._plot_roc_curve(split_results['clf_results']['true_labels'], 
                            split_results['clf_results']['probabilities'], 
                            split)

    def _summarize_results(self, results):
        summary = {split: {
            'vae_loss': results[split]['loss'], 
            'vae_recon': results[split]['recon'], 
            'vae_kld': results[split]['kld'],
            'clf_loss': results[split]['clf_results']['loss'],
            'clf_accuracy': results[split]['clf_results']['accuracy']
        } for split in results}
        summary_df = pd.DataFrame(summary)
        summary_df.to_csv(os.path.join(self.eval_output_dir, 'summary_results.csv'))

    def _generate_latent_space_visualizations(self, results):
        # Reset index for all dataframes before concatenation
        all_metadata = pd.concat([
            results[split]['metadata'].reset_index(drop=True).assign(split=split) 
            for split in ['train', 'val', 'test']
        ], ignore_index=True)
        
        all_latents = pd.concat([
            results[split]['latent_df'].reset_index(drop=True) 
            for split in ['train', 'val', 'test']
        ], ignore_index=True)

        # Create color maps for unique gene and barcode values
        unique_genes = all_metadata['gene'].unique()
        unique_barcode = all_metadata['barcode'].unique()
        gene_color_map = plt.cm.get_cmap('viridis')(np.linspace(0, 1, len(unique_genes)))
        barcode_color_map = plt.cm.get_cmap('plasma')(np.linspace(0, 1, len(unique_barcode)))
        gene_color_dict = dict(zip(unique_genes, gene_color_map))
        barcode_color_dict = dict(zip(unique_barcode, barcode_color_map))

        # Fit PCA and UMAP on all data
        scaler = StandardScaler()
        all_latents_scaled = scaler.fit_transform(all_latents)
        pca = PCA(n_components=2)
        pca_coords = pca.fit_transform(all_latents_scaled)

        # Create PCA dataframe with metadata
        pca_df = pd.DataFrame(pca_coords, columns=['PCA1', 'PCA2'])
        pca_df = pd.concat([all_metadata.reset_index(drop=True), pca_df], axis=1)
        pca_df.to_csv(os.path.join(self.eval_output_dir, 'pca_coordinates.csv'), index=False)

        print(f"PCA shape: {pca_df.shape}")
        print(f"Metadata shape: {all_metadata.shape}")
        print(f"First few rows of PCA data:")
        print(pca_df.head())

        try:
            reducer = umap.UMAP(random_state=42, n_jobs=1)
            umap_coords = reducer.fit_transform(all_latents_scaled)
            
            # Create UMAP dataframe with metadata
            umap_df = pd.DataFrame(umap_coords, columns=['UMAP1', 'UMAP2'])
            umap_df = pd.concat([all_metadata.reset_index(drop=True), umap_df], axis=1)
            umap_df.to_csv(os.path.join(self.eval_output_dir, 'umap_coordinates.csv'), index=False)

            print(f"UMAP shape: {umap_df.shape}")
            print(f"First few rows of UMAP data:")
            print(umap_df.head())
        except ImportError:
            print("UMAP not installed. Skipping UMAP visualization.")
            reducer = None

        # Generate visualizations
        for method, data, viz_df in [
            ('PCA', pca_coords, pca_df), 
            ('UMAP', umap_coords if reducer else None, umap_df if reducer else None)
        ]:
            if data is None:
                continue

            # 1. Plots colored by gene
            self._plot_latent_space(data, viz_df, gene_color_dict, 'gene', method)

            # 2. Plots colored by barcode
            self._plot_latent_space(data, viz_df, barcode_color_dict, 'barcode', method)

            # 3. Kernel density plots
            self._plot_kernel_density(data, viz_df, method)

    def _plot_latent_space(self, data, metadata, color_dict, color_by, method):
        legend_elements = [plt.Line2D([0], [0], marker='o', color='w', 
                            label=value, markerfacecolor=color, markersize=10)
                            for value, color in color_dict.items()]

        fig, axes = plt.subplots(1, 3, figsize=(24, 8))
        fig.suptitle(f'2D {method} of Latent Space - All Splits (Colored by {color_by.capitalize()})', fontsize=16)

        for idx, split in enumerate(['train', 'val', 'test']):
            split_mask = metadata['split'] == split
            split_data = data[split_mask]
            split_colors = metadata.loc[split_mask, color_by].map(color_dict)

            axes[idx].scatter(split_data[:, 0], split_data[:, 1], c=split_colors, alpha=0.6)
            axes[idx].set_title(f'{split.capitalize()}')
            axes[idx].set_xlabel(f'First {method} Component')
            axes[idx].set_ylabel(f'Second {method} Component')

        fig.legend(handles=legend_elements, title=color_by.capitalize(), loc='center right', bbox_to_anchor=(0.98, 0.5))
        plt.tight_layout()
        plt.savefig(os.path.join(self.eval_output_dir, f'combined_latent_space_{method.lower()}_{color_by}.png'), bbox_inches='tight')
        plt.close()

    def _plot_kernel_density(self, data, metadata, method):
        try:
            df = pd.DataFrame(data, columns=['x', 'y'])
            df['gene'] = metadata['gene'].values
            df['barcode'] = metadata['barcode'].values

            print(f"Data shape: {df.shape}")
            print(f"Unique genes: {df['gene'].nunique()}")
            print(f"Unique barcodes: {df['barcode'].nunique()}")

            for category in ['gene', 'barcode']:
                try:
                    g = sns.jointplot(
                        data=df,
                        x='x',
                        y='y',
                        hue=category,
                        kind='kde',
                        height=10
                    )
                    g.plot_joint(sns.scatterplot, s=30, alpha=0.4)
                    g.fig.suptitle(f'2D {method} of Latent Space - Kernel Density (Colored by {category.capitalize()})', fontsize=16)
                    g.fig.tight_layout()
                    
                    output_path = os.path.join(self.eval_output_dir, f'kernel_density_{method.lower()}_{category}.png')
                    print(f"Saving plot to: {output_path}")
                    g.savefig(output_path, dpi=300, bbox_inches='tight')
                    plt.close(g.fig)
                    print(f"Successfully saved {category} plot")
                except Exception as e:
                    print(f"Error plotting {category}: {str(e)}")
        except Exception as e:
            print(f"Error in _plot_kernel_density: {str(e)}")
            
    def _save_image(self, data, recon, metadata, split):
        image_idx = np.random.randint(data.shape[0])
        original = data[image_idx].cpu().numpy()
        reconstruction = recon[image_idx].cpu().numpy()
        image_metadata = {key: value[image_idx] for key, value in zip(self.config['metadata_keys'], metadata)}
        fig, axes = plt.subplots(2, 4, figsize=(20, 10))
        
        channel_names = self.config['channel_names']
        
        for i in range(len(channel_names)):
            # Original image
            im = axes[0, i].imshow(original[i], cmap='viridis')
            axes[0, i].set_title(f'Original {channel_names[i]}', fontsize=12)
            axes[0, i].axis('off')
            fig.colorbar(im, ax=axes[0, i], fraction=0.046, pad=0.04)
            
            # Reconstructed image
            im = axes[1, i].imshow(reconstruction[i], cmap='viridis')
            axes[1, i].set_title(f'Reconstructed {channel_names[i]}', fontsize=12)
            axes[1, i].axis('off')
            fig.colorbar(im, ax=axes[1, i], fraction=0.046, pad=0.04)

        # Create filename
        filename = f"{split}_comparison.png"

        # Add gene name to the title
        fig.suptitle(f"Original vs. Reconstructed Images - Perturbation: {image_metadata}", fontsize=16)
        
        # Save the image
        plt.savefig(os.path.join(self.eval_output_dir, filename), dpi=300, bbox_inches='tight')
        plt.close(fig)  

    def _plot_roc_curve(self, true_labels, probabilities, split):
        fpr = dict()
        tpr = dict()
        roc_auc = dict()
        for i, class_name in enumerate(self.class_names):
            fpr[i], tpr[i], _ = roc_curve((np.array(true_labels) == class_name).astype(int), 
                                        [prob[i] for prob in probabilities])
            roc_auc[i] = auc(fpr[i], tpr[i])
        
        plt.figure(figsize=(10, 8))
        for i, class_name in enumerate(self.class_names):
            plt.plot(fpr[i], tpr[i], label=f'{class_name} (AUC = {roc_auc[i]:.2f})')
        plt.plot([0, 1], [0, 1], 'k--')
        plt.xlim([0.0, 1.0])
        plt.ylim([0.0, 1.05])
        plt.xlabel('False Positive Rate')
        plt.ylabel('True Positive Rate')
        plt.title(f'ROC Curve - {split.capitalize()}')
        plt.legend(loc="lower right")
        plt.tight_layout()
        plt.savefig(os.path.join(self.eval_output_dir, f'{split}_roc_curve.png'))
        plt.close()

    def _plot_confusion_matrix(self, conf_matrix, split):
        conf_matrix_percent = conf_matrix.astype('float') / conf_matrix.sum(axis=1)[:, np.newaxis] * 100
        
        plt.figure(figsize=(12, 10))
        sns.heatmap(conf_matrix_percent, annot=True, fmt='.1f', cmap='Blues',
                    xticklabels=self.class_names, yticklabels=self.class_names)
        plt.title(f'Confusion Matrix (%) - {split.capitalize()}')
        plt.ylabel('True label')
        plt.xlabel('Predicted label')
        
        cbar = plt.gca().collections[0].colorbar
        cbar.set_label('Percentage (%)')
        
        plt.tight_layout()
        plt.savefig(os.path.join(self.eval_output_dir, f'{split}_confusion_matrix_percent.png'))
        plt.close()

class LatentSpaceVisualizer:
    def __init__(self, eval_output_dir, config):
        """
        Initialize visualization dashboard with pre-computed VAE evaluation results
        
        Args:
            eval_output_dir: Directory containing evaluation results (should already include epoch)
            config: Original configuration used for evaluation
        """
        self.eval_dir = eval_output_dir  # This path should already include the epoch
        print(f"Using evaluation directory: {self.eval_dir}")
        
        self.config = config
        self.app = Dash(__name__)
        
        # Load cached results
        self.cached_data = self._load_evaluation_results()
        if not self.cached_data:
            raise ValueError(f"No data found in {self.eval_dir}. Has the model been evaluated?")
            
        self.dataset_train = self._initialize_dataset(split='train')
        self.dataset_val = self._initialize_dataset(split='val')
        self.dataset_test = self._initialize_dataset(split='test')
        
        self.setup_layout()
        self.setup_callbacks()
    
    def _load_evaluation_results(self):
        """Load pre-computed UMAP and PCA coordinates"""
        print(f"Loading visualization data from: {self.eval_dir}")
        
        umap_file = os.path.join(self.eval_dir, 'umap_coordinates.csv')
        pca_file = os.path.join(self.eval_dir, 'pca_coordinates.csv')
        
        data = {}
        if os.path.exists(umap_file):
            data['umap'] = pd.read_csv(umap_file)
            print(f"Loaded UMAP data with shape: {data['umap'].shape}")
        
        if os.path.exists(pca_file):
            data['pca'] = pd.read_csv(pca_file)
            print(f"Loaded PCA data with shape: {data['pca'].shape}")
            
        if not data:
            raise ValueError(f"No coordinate data found in {self.eval_dir}")
            
        return data
    
    def _initialize_dataset(self, split):
        """Initialize datasets for all splits"""
        dataset_mean, dataset_std = read_config(self.config['yaml_file'])
        normalizations = v2.Compose([v2.CenterCrop(self.config['crop_size'])])
        
        dataset = ZarrCellDataset(
                parent_dir=self.config['parent_dir'],
                csv_file=self.config['csv_file'],
                split=split,
                channels=self.config['channels'],
                mask=self.config['transform'],
                normalizations=normalizations,
                interpolations=None,
                mean=dataset_mean,
                std=dataset_std
            )
        return dataset

    def setup_layout(self):
        self.app.layout = html.Div([
            html.H1('Dimensionality Reduction Visualization', style={'textAlign': 'center'}),
            
            # Controls
            html.Div([
                html.Div([
                    html.Label('Visualization Type:'),
                    dcc.Dropdown(
                        id='viz-type-dropdown',
                        options=[
                            {'label': 'UMAP', 'value': 'umap'},
                            {'label': 'PCA', 'value': 'pca'}
                        ],
                        value='umap'
                    )
                ], style={'width': '200px', 'margin': '10px'}),
                
                html.Div([
                    html.Label('Color By:'),
                    dcc.Dropdown(
                        id='color-dropdown',
                        options=[
                            {'label': 'Gene', 'value': 'gene'},
                            {'label': 'Barcode', 'value': 'barcode'},
                            {'label': 'Split', 'value': 'split'}
                        ],
                        value='gene'
                    )
                ], style={'width': '200px', 'margin': '10px'}),
                html.Div([
                    html.Label('Show Images:'),
                    dcc.Checklist(
                        id='show-images-switch',
                        options=[{'label': '', 'value': True}],
                        value=[True]
                    )
                ], style={'width': '200px', 'margin': '10px'})
            ], style={'display': 'flex', 'justifyContent': 'center'}),
            
            # Main visualization
            html.Div([
                dcc.Graph(
                    id='dim-reduction-plot',
                    style={'width': '1200px', 'height': '600px'}
                ),
                dcc.Tooltip(id='hover-tooltip')
            ], style={'display': 'flex', 'justifyContent': 'center'})
        ])
        
    def setup_callbacks(self):
        @self.app.callback(
            Output('dim-reduction-plot', 'figure'),
            [Input('viz-type-dropdown', 'value'),
            Input('color-dropdown', 'value')]
        )
        def update_graph(viz_type, color_by):
            df = self.cached_data[viz_type]
            coord_cols = {'umap': ['UMAP1', 'UMAP2'], 
                        'pca': ['PCA1', 'PCA2']}[viz_type]
            
            return px.scatter(
                df,
                x=coord_cols[0],
                y=coord_cols[1],
                color=color_by,
                hover_data=['gene', 'barcode', 'split', 'cell_idx']
            ).update_layout(
                width=1250,  
                height=1000
            ).update_traces(
                hovertemplate=None,
                hoverinfo='none'
            )

        @self.app.callback(
            [Output("hover-tooltip", "show"),
            Output("hover-tooltip", "bbox"),
            Output("hover-tooltip", "children")],
            [Input("dim-reduction-plot", "hoverData"),
            Input("viz-type-dropdown", "value"),
            Input("color-dropdown", "value"),
            Input("show-images-switch", "value")]
        )
        def show_hover_info(hover_data, viz_type, color_by, show_images):
            if not hover_data:
                return False, no_update, no_update

            pt = hover_data['points'][0]
            gene, barcode, split, cell_idx = pt['customdata']
            point_color = pt.get('marker.color', '#ffffff')
            
            base_style = {
                'padding': '0px', 
                'borderRadius': '0px',
                'backgroundColor': point_color,
                'color': 'white' if point_color != '#ffffff' else 'black',
                'border': '0px solid #ddd'
            }
            
            if not show_images:
                children = html.Div([
                    html.P(f"gene: {gene}"),
                    html.P(f"barcode: {barcode}"),
                    html.P(f"split: {split}"),
                    html.P(f"cell_idx: {cell_idx}")
                ], style=base_style)
                return True, pt['bbox'], children

            try:
                if split == 'train':
                    cell_data = self.dataset_train[cell_idx]
                elif split == 'val':
                    cell_data = self.dataset_val[cell_idx]
                elif split == 'test':
                    cell_data = self.dataset_test[cell_idx]

                images = [self._normalize_and_encode_image(cell_data["cell_image"][i]) 
                        for i in range(len(self.config['channels']))]
                
                children = html.Div([
                    html.Div([
                        html.Div([
                            html.Img(src=img, style={'width': '75px', 'height': '75px', 'margin': '2px'}),
                            html.P(f'{self.config["channel_names"][i]}', 
                                    style={'margin': '0', 'fontSize': '12px', 'textAlign': 'center'})
                        ], style={'display': 'inline-block'}) for i, img in enumerate(images)
                    ], style={'display': 'flex', 'flexWrap': 'wrap', 'gap': '2px', 'justifyContent': 'center'}),
                    
                    html.Div([
                        html.P(f"gene: {gene}", style={'margin': '2px', 'fontSize': '12px'}),
                        html.P(f"barcode: {barcode}", style={'margin': '2px', 'fontSize': '12px'}),
                        html.P(f"split: {split}", style={'margin': '2px', 'fontSize': '12px'}),
                        html.P(f"cell_idx: {cell_idx}", style={'margin': '2px', 'fontSize': '12px'})
                    ], style={'marginTop': '5px'})
                ], style={'padding': '0px', 'backgroundColor': 'white', 'border': '0px solid #ddd'})
                        
                return True, pt['bbox'], children
            except IndexError as e:
                print(f"Error accessing dataset: {str(e)}")
                return False, no_update, no_update       
     
    def _normalize_and_encode_image(self, image_array):
        """Convert tensor to base64 encoded string"""
        # Convert to numpy and normalize
        image_array = image_array.cpu().numpy()
        normalized = (image_array - image_array.min()) / (image_array.max() - image_array.min())
        image = Image.fromarray((normalized * 255).astype('uint8'))
        buffered = io.BytesIO()
        image.save(buffered, format="PNG")
        return f'data:image/png;base64,{base64.b64encode(buffered.getvalue()).decode()}'

    def run(self, debug=True):
        self.app.run(debug=debug)

class VIEWSAnalyzer:
    def __init__(self, latent_vectors_path: str, config: dict, split: str, n_top_dims: Optional[int] = 25):
        """Initialize VIEWS analyzer with latent vectors data and dataset
        
        Args:
            latent_vectors_path: Path to CSV containing latent vectors
            config: Configuration dictionary
            split: Dataset split to use
            n_top_dims: Number of top latent dimensions to consider for constraints.
                       If None, uses all dimensions.
        """
        # Load latent vectors
        self.data = pd.read_csv(latent_vectors_path)
        self.latent_cols = [col for col in self.data.columns if col.startswith('latent_')]
        self.metadata_cols = [col for col in self.data.columns if not col.startswith('latent_')]
        
        # Store full latent vectors but also get top N if specified
        self.latent_vectors_full = self.data[self.latent_cols].values
        if n_top_dims is not None:
            # Ensure n_top_dims doesn't exceed available dimensions
            self.n_top_dims = min(n_top_dims, len(self.latent_cols))
            # Take only the first n_top_dims columns
            self.latent_vectors = self.latent_vectors_full[:, :self.n_top_dims]
        else:
            self.n_top_dims = len(self.latent_cols)
            self.latent_vectors = self.latent_vectors_full
        
        # Initialize dataset
        dataset_mean, dataset_std = read_config(config['yaml_file'])
        normalizations = v2.Compose([v2.CenterCrop(config['crop_size'])])
        
        self.dataset = ZarrCellDataset(
            parent_dir=config['parent_dir'],
            csv_file=config['csv_file'],
            split=split,
            channels=config['channels'],
            mask=config['transform'],
            normalizations=normalizations,
            interpolations=None,
            mean=dataset_mean,
            std=dataset_std
        )
        
        self.config = config

        # Create custom colormaps
        self.create_custom_colormaps()

    def create_custom_colormaps(self):
        """Create custom colormaps that transition from black to each color"""
        self.channel_colormaps = []
        
        # Define colors and their RGB values
        colors = [
            ('blue', (0, 0, 1)),
            ('green', (0, 1, 0)),
            ('red', (1, 0, 0)),
            ('cyan', (0, 1, 1))
        ]
        
        for color_name, rgb in colors:
            # Create array of color positions
            colors_array = np.zeros((256, 3))
            for i in range(256):
                # Linear interpolation from black (0,0,0) to target color
                colors_array[i] = np.array([c * i / 255 for c in rgb])
            
            self.channel_colormaps.append(
                LinearSegmentedColormap.from_list(f'black_to_{color_name}', colors_array)
            )

    def find_constrained_samples(self, 
                            target_dim: int,
                            n_samples_per_percentile: int = 3,
                            percentiles: List[int] = [1, 15, 50, 85, 99],
                            constraint_percentile: float = 10) -> List[List[int]]:
        """
        Find n_samples_per_percentile samples for each specified percentile
        that are close to mean in all dimensions except target_dim,
        considering only the top N dimensions if specified.
        
        Args:
            target_dim: Index of the dimension to analyze (must be < n_top_dims)
            n_samples_per_percentile: Number of samples to find for each percentile
            percentiles: List of percentiles to sample at
            constraint_percentile: Percentile threshold for considering points "close to mean"
        
        Returns:
            List of lists, where each inner list contains indices for a specific percentile
        """
        if target_dim >= self.n_top_dims:
            raise ValueError(f"target_dim ({target_dim}) must be less than n_top_dims ({self.n_top_dims})")
            
        # Get other dimensions (only from the top N dimensions we're considering)
        other_dims = list(range(self.latent_vectors.shape[1]))
        other_dims.remove(target_dim)
        
        # Calculate mean vector for other dimensions
        mean_vector = np.mean(self.latent_vectors[:, other_dims], axis=0)
        
        # Calculate distances to mean for other dimensions
        distances = cdist(self.latent_vectors[:, other_dims], [mean_vector]).flatten()
        
        # Find points close to mean in other dimensions
        distance_threshold = np.percentile(distances, constraint_percentile)
        close_to_mean = distances <= distance_threshold
        
        # Get target dimension values for close points
        target_values = self.latent_vectors[close_to_mean, target_dim]
        
        # For each percentile, find n_samples_per_percentile closest samples
        selected_indices = []
        for percentile in percentiles:
            threshold = np.percentile(target_values, percentile)
            
            # Find valid indices and sort by distance to threshold
            valid_indices = np.where(close_to_mean)[0]
            distances_to_threshold = np.abs(self.latent_vectors[valid_indices, target_dim] - threshold)
            closest_indices = valid_indices[np.argsort(distances_to_threshold)[:n_samples_per_percentile]]
            
            selected_indices.append(closest_indices.tolist())
        
        return selected_indices

    def get_cell_images(self, idx: int, lower_percentile: float = 1, upper_percentile: float = 99.9) -> np.ndarray:
        """Get normalized cell images for a given index with percentile-based normalization
        
        Args:
            idx: Index of the cell to retrieve
            lower_percentile: Lower percentile for normalization clipping (default: 1)
            upper_percentile: Upper percentile for normalization clipping (default: 99.9)
        """
        cell_data = self.dataset[idx]
        images = cell_data["cell_image"]  # This should be a tensor
        
        # Convert to numpy and normalize each channel
        normalized_images = []
        for i in range(len(self.config['channels'])):
            img = images[i].numpy()
            
            # Get percentile values for this image
            min_val = np.percentile(img, lower_percentile)
            max_val = np.percentile(img, upper_percentile)
            
            # Clip and normalize
            img_clipped = np.clip(img, min_val, max_val)
            norm_img = (img_clipped - min_val) / (max_val - min_val)
            
            # Ensure the range is exactly [0, 1]
            norm_img = np.clip(norm_img, 0, 1)
            
            normalized_images.append(norm_img)
            
        return normalized_images
    
    def visualize_dimension(self,
                        target_dim: int,
                        selected_indices: List[List[int]],  
                        output_path: Optional[str] = None,
                        target_percentiles: List[int] = [1, 15, 50, 85, 99]) -> None:
        """
        Create visualization for selected samples along a dimension with cell images
        """
        n_percentiles = len(selected_indices)
        
        # Create figure
        fig = plt.figure(figsize=(15, 10))
        
        # Create main GridSpec with space for density plot and 3 rows of image sets
        outer_gs = plt.GridSpec(4, n_percentiles, 
                        height_ratios=[1, 3, 3, 3],
                        hspace=0.15,
                        wspace=0.1)
        
        # Plot density at top
        ax_density = fig.add_subplot(outer_gs[0, :])
        values = self.latent_vectors[:, target_dim]
        
        # Get extended range percentile values
        extended_range = np.percentile(values, [0.05, 99.95])
        percentile_values = np.percentile(values, target_percentiles)
        
        # Create x_range based on extended percentiles
        x_range = np.linspace(extended_range[0], extended_range[1], 200)
        
        # Compute KDE
        kernel = stats.gaussian_kde(values)
        density = kernel(x_range)
        density_normalized = density / density.max()
        
        # Plot the normalized density
        ax_density.fill_between(x_range, density_normalized, color='purple', alpha=0.3)
        ax_density.plot(x_range, density_normalized, color='purple')
        
        # Set up the density plot
        ax_density.set_ylim(0, 1.1)
        ax_density.set_xlim(extended_range[0], extended_range[1])
        ax_density.set_xticks(percentile_values)
        ax_density.set_xticklabels([str(p) for p in target_percentiles], fontsize=8)
        ax_density.set_yticks([])
        ax_density.set_xlabel('')
        ax_density.set_ylabel('Density')
        ax_density.spines['top'].set_visible(False)
        ax_density.spines['right'].set_visible(False)
        ax_density.spines['left'].set_visible(False)

        # Create legend patches
        channel_names = ['DAPI', 'Tubulin', 'γH2AX', 'Actin']
        colors = ['blue', 'green', 'red', 'cyan']
        legend_patches = []
        for color, name in zip(colors, channel_names):
            patch = plt.Rectangle((0, 0), 1, 1, fc=color, alpha=0.8)
            legend_patches.append(patch)

        ax_density.legend(legend_patches, channel_names, 
                        loc='upper right', 
                        bbox_to_anchor=(1, 1.5), 
                        ncol=1,
                        frameon=False,
                        fontsize=8)
        
        # Get percentiles for all samples
        percentiles_by_group = []
        for group_indices in selected_indices:
            group_percentiles = []
            for idx in group_indices:
                value = self.latent_vectors[idx, target_dim]
                percentile = stats.percentileofscore(values, value)
                group_percentiles.append(percentile)
            percentiles_by_group.append(group_percentiles)
        
        print(f"Percentiles by group: {percentiles_by_group}")
        
        # Plot cell images for each position
        for row in range(3):  # 3 rows of sampled images
            for col in range(n_percentiles):
                # Create a 2x2 GridSpec for the 4 channels within this cell
                inner_gs = outer_gs[row+1, col].subgridspec(2, 2, hspace=0.05, wspace=0.05)
                
                sample_idx = selected_indices[col][row]
                cell_idx = self.data.iloc[sample_idx]['cell_idx']
                images = self.get_cell_images(cell_idx)
                
                # Plot each channel
                for i, channel_img in enumerate(images):
                    ax = fig.add_subplot(inner_gs[i//2, i%2])
                    ax.imshow(channel_img, cmap=self.channel_colormaps[i])
                    ax.axis('off')

                # Add percentile labels below bottom row
                if row == 2:
                    # Create a fake axis for the label
                    ax_label = fig.add_subplot(outer_gs[row+1, col])
                    ax_label.set_xticks([])
                    ax_label.set_yticks([])
                    ax_label.set_xlabel(f'{target_percentiles[col]}th\nPercentile', 
                                    fontsize=10, color='black',
                                    labelpad=5)
                    ax_label.set_frame_on(False)
        
        # Add title
        plt.suptitle(f'Latent Dimension {target_dim}', y=0.95, color='black')
        
        plt.tight_layout()
        
        if output_path:
            plt.savefig(output_path, dpi=300, bbox_inches='tight',
                    facecolor='white')
        plt.close()

    def analyze_dimension_distribution(self, target_dim: int) -> pd.DataFrame:
        """Analyze the statistical distribution of values in the target dimension"""
        values = self.latent_vectors[:, target_dim]
        stats = {
            'mean': np.mean(values),
            'std': np.std(values),
            'min': np.min(values),
            'max': np.max(values),
            'skew': pd.Series(values).skew(),
            'kurtosis': pd.Series(values).kurtosis()
        }
        return pd.DataFrame([stats])