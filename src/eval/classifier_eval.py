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
from matplotlib.colors import ListedColormap
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import confusion_matrix, classification_report, roc_curve, auc

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..')))
from src.data.dataset import ZarrCellDataset
from src.data.dataloader import collate_wrapper
from src.data.utils import read_config
from src.models.vgg2d import Vgg2D
from src.models.VAE_resnet18 import ResNet18Classifier

class Classifier_Evaluator:
    def __init__(self, config):
        self.config = config
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.class_names = config['class_names']
        self.base_output_dir = config['output_dir']
        self.eval_output_dir = os.path.join(self.base_output_dir, config['checkpoint_epoch'])
        self.output_dir = self._create_output_dirs()
        self._plot_epoch_log_metrics()
        self.model = self._init_model()
        self.criterion = torch.nn.CrossEntropyLoss()

    def _create_output_dirs(self):
        os.makedirs(self.base_output_dir, exist_ok=True)
        os.makedirs(self.eval_output_dir, exist_ok=True)
    
    def _init_model(self):
        if self.config['model_name'] == 'VGG2D':
            model = Vgg2D(
                input_size=(self.config['crop_size'], self.config['crop_size']),
                input_fmaps=len(self.config['channels']),
                output_classes=len(self.class_names),
            )
        elif self.config['model_name'] == 'ResNet18':
            model = ResNet18Classifier(
                num_classes=len(self.class_names),
                nc=len(self.config['channels'])
            )

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
        print(f"Train Loss: {checkpoint['train_loss']:.4f}, Train Accuracy: {checkpoint['train_accuracy']:.4f}")
        print(f"Train Balanced Accuracy: {checkpoint['train_balanced_accuracy']:.4f}, Train AUC: {checkpoint['train_auc']:.4f}")
        print(f"Val Loss: {checkpoint['val_loss']:.4f}, Val Accuracy: {checkpoint['val_accuracy']:.4f}")
        print(f"Val Balanced Accuracy: {checkpoint['val_balanced_accuracy']:.4f}, Val AUC: {checkpoint['val_auc']:.4f}")
        return model, checkpoint['epoch']

    def _plot_epoch_log_metrics(self):
        # Read the epoch log CSV file
        epoch_log_path = os.path.join(self.config['log_dir'], 'epoch_log.csv')
        if not os.path.exists(epoch_log_path):
            print(f"Epoch log file not found at {epoch_log_path}. Skipping epoch log plot.")
            return

        df = pd.read_csv(epoch_log_path)

        metrics = ['loss', 'accuracy', 'balanced_accuracy', 'auc']
        fig, axes = plt.subplots(2, 2, figsize=(10, 7.5))
        fig.suptitle('Training and Validation Metrics Over Epochs', fontsize=16)

        def smooth(y, box_pts):
            box = np.ones(box_pts)/box_pts
            y_smooth = np.convolve(y, box, mode='same')
            return y_smooth

        for i, metric in enumerate(metrics):
            ax = axes[i // 2, i % 2]
            
            # Smooth the lines
            train_smooth = smooth(df[f'train_{metric}'], 5)
            val_smooth = smooth(df[f'val_{metric}'], 5)
            
            ax.plot(df['epoch'], train_smooth, label=f'Train {metric.capitalize()}')
            ax.plot(df['epoch'], val_smooth, label=f'Validation {metric.capitalize()}')
            
            ax.set_xlabel('Epoch')
            ax.set_ylabel(metric.capitalize())
            ax.set_title(f'{metric.capitalize()} vs. Epoch')
            ax.legend()
            
            # Remove grid
            ax.grid(False)
            
            # Set y-axis limits for accuracy, balanced_accuracy, and auc
            if metric in ['accuracy', 'balanced_accuracy', 'auc']:
                ax.set_ylim(0, 1)
            
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
    
    def _evaluate_model(self, dataloader):
        self.model.eval()
        total_loss = 0
        correct = 0
        total = 0
        all_predictions = []
        all_true_labels = []
        all_probabilities = []
        
        with torch.no_grad():
            for batch in dataloader:
                data = batch['cell_image'].to(self.device)
                labels = torch.tensor([self.class_names.index(label) for label in batch[self.config['label_type']]]).to(self.device)

                outputs = self.model(data)
                loss = self.criterion(outputs, labels)

                total_loss += loss.item()
                probabilities = F.softmax(outputs, dim=1)
                _, predicted = outputs.max(1)
                total += labels.size(0)
                correct += predicted.eq(labels).sum().item()

                all_predictions.extend(predicted.cpu().numpy())
                all_true_labels.extend(labels.cpu().numpy())
                all_probabilities.extend(probabilities.cpu().numpy())

        avg_loss = total_loss / len(dataloader)
        accuracy = 100. * correct / total

        return avg_loss, accuracy, all_predictions, all_true_labels, all_probabilities


    def evaluate(self):
        results = {}

        for split in ['train', 'val', 'test']:
            dataloader = self._create_dataloader(split, drop_last=False)
            print(f"Evaluating on {split} data...")

            loss, accuracy, predictions, true_labels, probabilities = self._evaluate_model(dataloader)

            print(f"{split.capitalize()} - Loss: {loss:.4f}, Accuracy: {accuracy:.2f}%")

            results[split] = {
                'loss': loss,
                'accuracy': accuracy,
                'predictions': predictions,
                'true_labels': true_labels,
                'probabilities': probabilities
            }

            # Generate and save split-specific results
            self._save_split_results(split, results[split])

        # Summarize results across all splits
        self._summarize_results(results)

        return results

    def _save_split_results(self, split, split_results):
        # Save metrics
        metrics_df = pd.DataFrame({
            'loss': [split_results['loss']],
            'accuracy': [split_results['accuracy']]
        })
        metrics_df.to_csv(os.path.join(self.eval_output_dir, f'{split}_metrics.csv'), index=False)

        # Generate and save confusion matrix
        conf_matrix = confusion_matrix(split_results['true_labels'], split_results['predictions'])
        self._plot_confusion_matrix(conf_matrix, split)

        # Generate and save classification report
        class_report = classification_report(split_results['true_labels'], split_results['predictions'], target_names=self.class_names)
        with open(os.path.join(self.eval_output_dir, f'{split}_classification_report.txt'), 'w') as f:
            f.write(class_report)

        # Generate and save ROC curve
        self._plot_roc_curve(split_results['true_labels'], split_results['probabilities'], split)

    def _summarize_results(self, results):
        summary = {split: {'loss': results[split]['loss'], 'accuracy': results[split]['accuracy']} for split in results}
        summary_df = pd.DataFrame(summary)
        summary_df.to_csv(os.path.join(self.eval_output_dir, 'summary_results.csv'))

    def _plot_confusion_matrix(self, conf_matrix, split):
        conf_matrix_percent = conf_matrix.astype('float') / conf_matrix.sum(axis=1)[:, np.newaxis] * 100
        
        plt.figure(figsize=(12, 10))
        sns.heatmap(conf_matrix_percent, annot=True, fmt='.1f', cmap='Blues',
                    xticklabels=self.class_names, yticklabels=self.class_names)
        plt.title(f'Confusion Matrix (%) - {split.capitalize()}')
        plt.ylabel('True label')
        plt.xlabel('Predicted label')
        
        # Add color bar
        cbar = plt.gca().collections[0].colorbar
        cbar.set_label('Percentage (%)')
        
        plt.tight_layout()
        plt.savefig(os.path.join(self.eval_output_dir, f'{split}_confusion_matrix_percent.png'))
        plt.close()


    def _plot_roc_curve(self, true_labels, probabilities, split):
        n_classes = len(self.class_names)
        fpr = dict()
        tpr = dict()
        roc_auc = dict()

        for i in range(n_classes):
            fpr[i], tpr[i], _ = roc_curve((np.array(true_labels) == i).astype(int), 
                                        [prob[i] for prob in probabilities])
            roc_auc[i] = auc(fpr[i], tpr[i])

        plt.figure(figsize=(10, 8))
        for i in range(n_classes):
            plt.plot(fpr[i], tpr[i], label=f'{self.class_names[i]} (AUC = {roc_auc[i]:.2f})')
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