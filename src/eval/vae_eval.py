import os
import sys
import numpy as np
import torch
from torch.utils.data import DataLoader
from torch.nn import functional as F
from torchvision.transforms import v2
from torchvision.utils import save_image
import matplotlib.pyplot as plt
import pandas as pd
import yaml
import seaborn as sns
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline
from sklearn.model_selection import cross_val_score
from sklearn.metrics import accuracy_score, log_loss, confusion_matrix, classification_report, roc_curve, auc, roc_auc_score
import umap
from matplotlib.colors import ListedColormap
import piq
import seaborn as sns

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..')))
from src.data.dataset import ZarrCellDataset
from src.data.dataloader import collate_wrapper
from src.data.utils import read_config
from src.models.VAE_resnet18 import VAEResNet18

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
        # Prepare the combined data
        all_metadata = pd.concat([results[split]['metadata'].assign(split=split) for split in ['train', 'val', 'test']])
        all_latents = pd.concat([results[split]['latent_df'] for split in ['train', 'val', 'test']])

        # Create color maps for unique gene and barcode values
        unique_genes = all_metadata['gene'].unique()
        unique_barcode = all_metadata['barcode'].unique()
        gene_color_map = plt.cm.get_cmap('viridis')(np.linspace(0, 1, len(unique_genes)))
        barcode_color_map = plt.cm.get_cmap('plasma')(np.linspace(0, 1, len(unique_barcode)))
        gene_color_dict = dict(zip(unique_genes, gene_color_map))
        barcode_color_dict = dict(zip(unique_barcode, barcode_color_map))

        # Fit PCA and UMAP on all data
        pca = PCA(n_components=2)
        scaler = StandardScaler()
        all_latents_scaled = scaler.fit_transform(all_latents)
        all_latents_pca = pca.fit_transform(all_latents_scaled)

        try:
            reducer = umap.UMAP(random_state=42, n_jobs=1)
            all_latents_umap = reducer.fit_transform(all_latents_scaled)
        except ImportError:
            print("UMAP not installed. Skipping UMAP visualization.")
            reducer = None

        # Generate visualizations
        for method, data in [('PCA', all_latents_pca), ('UMAP', all_latents_umap if reducer else None)]:
            if data is None:
                continue

            # 1. Plots colored by gene
            self._plot_latent_space(data, all_metadata, gene_color_dict, 'gene', method)

            # 2. Plots colored by barcode
            self._plot_latent_space(data, all_metadata, barcode_color_dict, 'barcode', method)

            # 3. Kernel density plots
            self._plot_kernel_density(data, all_metadata, method)

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
        
        channel_names = ['dapi', 'tubulin', 'gh2ax', 'actin']  
        
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