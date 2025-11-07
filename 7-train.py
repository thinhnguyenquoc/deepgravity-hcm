import json
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader, TensorDataset
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import r2_score, mean_squared_error, mean_absolute_error
import warnings
warnings.filterwarnings('ignore')

# Set random seeds for reproducibility
torch.manual_seed(42)
np.random.seed(42)

# Device configuration
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
# print(f"Using device: {device}")

class EnhancedMigrationDataset(Dataset):
    def __init__(self):
        self.feature_names = [
            'population_a', 'population_b', 'distance', 
            'amenity_a', 'amenity_b', 'shop_a', 'shop_b',
            'tourism_a', 'tourism_b', 'leisure_a', 'leisure_b',
            'office_a', 'office_b', 'public_transport_a', 'public_transport_b'
        ]
        self.scaler = StandardScaler()
        self._generate_data()
        self._prepare_data()
    
    def _generate_data(self):
        """Generate synthetic migration data with urban features"""
        filename = "./aggregated_data.json"
        with open(filename, 'r') as file:
            data_aggregate = json.load(file)
        # np.random.seed(42)
        self.n_samples = len(data_aggregate)
        pop_a_list = []
        pop_b_list = []
        dist_list = []
        amenity_a_list = []
        amenity_b_list = []
        shop_a_list = []
        shop_b_list = []
        tourism_a_list = []
        tourism_b_list = []
        leisure_a_list = []
        leisure_b_list = []
        office_a_list = []
        office_b_list = []
        public_transport_a_list = []
        public_transport_b_list = []
        probability_list = []
        for item in data_aggregate:
            pop_a_list.append(item["population_from"])
            pop_b_list.append(item["population_to"])
            dist_list.append(item["distance_km"])
            amenity_a_list.append(item["pois_amenity_from"])
            amenity_b_list.append(item["pois_amenity_to"])
            shop_a_list.append(item["pois_shop_from"])
            shop_b_list.append(item["pois_shop_to"])
            tourism_a_list.append(item["pois_tourism_from"])
            tourism_b_list.append(item["pois_tourism_to"])
            leisure_a_list.append(item["pois_leisure_from"])
            leisure_b_list.append(item["pois_leisure_to"])
            office_a_list.append(item["pois_office_from"])
            office_b_list.append(item["pois_office_to"])
            public_transport_a_list.append(item["pois_public_transport_from"])
            public_transport_b_list.append(item["pois_public_transport_to"])
            probability_list.append(item["probability_move"])

        pop_a = np.array(pop_a_list)
        pop_b = np.array(pop_b_list)
        distance = np.array(dist_list)
        amenity_a = np.array(amenity_a_list)
        amenity_b = np.array(amenity_b_list)
        shop_a = np.array(shop_a_list)
        shop_b = np.array(shop_b_list)
        tourism_a = np.array(tourism_a_list)
        tourism_b = np.array(tourism_b_list)
        leisure_a = np.array(leisure_a_list)
        leisure_b = np.array(leisure_b_list)
        office_a = np.array(office_a_list)
        office_b = np.array(office_b_list)
        public_transport_a = np.array(public_transport_a_list)
        public_transport_b = np.array(public_transport_b_list)
        
        # Create feature matrix
        features = np.column_stack([
            pop_a, pop_b, distance,
            amenity_a, amenity_b, shop_a, shop_b,
            tourism_a, tourism_b, leisure_a, leisure_b,
            office_a, office_b, public_transport_a, public_transport_b
        ])
        
        
        probability = np.array(probability_list)
        
        self.raw_features = torch.tensor(features, dtype=torch.float32)
        self.raw_targets = torch.tensor(probability, dtype=torch.float32)
    
    def _prepare_data(self):
        """Prepare and scale the data"""
        # Log transform features (handle zeros by adding 1)
        features_log = torch.log(self.raw_features + 1)
        
        # Scale features
        features_scaled = self.scaler.fit_transform(features_log.numpy())
        
        self.features = torch.tensor(features_scaled, dtype=torch.float32)
        self.targets = self.raw_targets
    
    def __len__(self):
        return self.n_samples
    
    def __getitem__(self, idx):
        return self.features[idx], self.targets[idx]
    
    def get_feature_names(self):
        return self.feature_names
    
    def get_raw_data(self):
        return self.raw_features, self.raw_targets

# Create dataset
# dataset = EnhancedMigrationDataset(n_samples=10000)
# features, targets = dataset.get_raw_data()

# print("Enhanced Dataset Info:")
# print(f"Features shape: {features.shape}")
# print(f"Targets shape: {targets.shape}")
# print(f"\nFeature names: {dataset.get_feature_names()}")
# print(f"Probability range: [{targets.min():.4f}, {targets.max():.4f}]")

class EnhancedMigrationPredictor(nn.Module):
    def __init__(self, input_dim=15, hidden_dims=[512, 256, 128, 64, 32], 
                 dropout_rate=0.3, use_batch_norm=True):
        super(EnhancedMigrationPredictor, self).__init__()
        
        self.input_dim = input_dim
        self.hidden_dims = hidden_dims
        
        # Build hidden layers
        layers = []
        prev_dim = input_dim
        
        for i, hidden_dim in enumerate(hidden_dims):
            layers.append(nn.Linear(prev_dim, hidden_dim))
            
            if use_batch_norm:
                layers.append(nn.BatchNorm1d(hidden_dim))
            
            layers.append(nn.ReLU())
            layers.append(nn.Dropout(dropout_rate))
            prev_dim = hidden_dim
        
        self.feature_extractor = nn.Sequential(*layers)
        
        # Output layer
        self.output_layer = nn.Sequential(
            nn.Linear(prev_dim, 1),
            nn.Sigmoid()  # Output between 0 and 1
        )
        
        # Initialize weights
        self._initialize_weights()
    
    def _initialize_weights(self):
        """Initialize weights using appropriate methods"""
        for module in self.modules():
            if isinstance(module, nn.Linear):
                nn.init.kaiming_uniform_(module.weight, nonlinearity='relu')
                if module.bias is not None:
                    nn.init.constant_(module.bias, 0)
    
    def forward(self, x):
        features = self.feature_extractor(x)
        output = self.output_layer(features)
        return output.squeeze()
    
    def get_feature_importance(self, data_loader):
        """Calculate approximate feature importance using permutation"""
        self.eval()
        baseline_metrics = self.evaluate(data_loader)
        baseline_mse = baseline_metrics['mse']
        
        feature_importance = []
        
        for feature_idx in range(self.input_dim):
            # Create modified data loader with shuffled feature
            modified_features = []
            modified_targets = []
            
            for batch_X, batch_y in data_loader:
                # Shuffle the specific feature
                shuffled_X = batch_X.clone()
                shuffled_X[:, feature_idx] = shuffled_X[torch.randperm(shuffled_X.size(0)), feature_idx]
                
                modified_features.append(shuffled_X)
                modified_targets.append(batch_y)
            
            # Calculate MSE with shuffled feature
            all_preds = []
            all_targets = []
            
            with torch.no_grad():
                for batch_X, batch_y in zip(modified_features, modified_targets):
                    preds = self(batch_X)
                    all_preds.append(preds)
                    all_targets.append(batch_y)
            
            all_preds = torch.cat(all_preds)
            all_targets = torch.cat(all_targets)
            shuffled_mse = nn.MSELoss()(all_preds, all_targets).item()
            
            # Importance is the increase in MSE
            importance = shuffled_mse - baseline_mse
            feature_importance.append(importance)
        
        return np.array(feature_importance)

# Test model
model = EnhancedMigrationPredictor(input_dim=15)
model.to(device)

print("Enhanced Model Architecture:")
print(model)
print(f"Total parameters: {sum(p.numel() for p in model.parameters()):,}")

class AdvancedMigrationTrainer:
    def __init__(self, model, train_loader, val_loader, learning_rate=0.001):
        self.model = model
        self.train_loader = train_loader
        self.val_loader = val_loader
        self.device = next(model.parameters()).device
        
        # Loss function with regularization
        self.criterion = nn.MSELoss()
        self.optimizer = optim.AdamW(model.parameters(), lr=learning_rate, weight_decay=1e-4)
        
        # Learning rate scheduler
        self.scheduler = optim.lr_scheduler.ReduceLROnPlateau(
            self.optimizer, mode='min', patience=15, factor=0.5
        )
        
        self.train_losses = []
        self.val_losses = []
        self.learning_rates = []
        self.best_val_loss = float('inf')
        self.best_model_state = None
        self.patience_counter = 0
        self.max_patience = 30
    
    def train_epoch(self):
        """Train for one epoch"""
        self.model.train()
        epoch_loss = 0.0
        
        for batch_X, batch_y in self.train_loader:
            batch_X, batch_y = batch_X.to(self.device), batch_y.to(self.device)
            
            self.optimizer.zero_grad()
            predictions = self.model(batch_X)
            loss = self.criterion(predictions, batch_y)
            
            loss.backward()
            # Gradient clipping
            torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=1.0)
            self.optimizer.step()
            
            epoch_loss += loss.item()
        
        return epoch_loss / len(self.train_loader)
    
    def validate(self):
        """Validate the model"""
        self.model.eval()
        val_loss = 0.0
        
        with torch.no_grad():
            for batch_X, batch_y in self.val_loader:
                batch_X, batch_y = batch_X.to(self.device), batch_y.to(self.device)
                predictions = self.model(batch_X)
                loss = self.criterion(predictions, batch_y)
                val_loss += loss.item()
        
        return val_loss / len(self.val_loader)
    
    def train(self, epochs=300):
        """Complete training loop with early stopping"""
        print("Starting advanced training...")
        
        for epoch in range(epochs):
            # Training phase
            train_loss = self.train_epoch()
            self.train_losses.append(train_loss)
            
            # Validation phase
            val_loss = self.validate()
            self.val_losses.append(val_loss)
            
            # Learning rate tracking
            current_lr = self.optimizer.param_groups[0]['lr']
            self.learning_rates.append(current_lr)
            
            # Early stopping check
            if val_loss < self.best_val_loss:
                self.best_val_loss = val_loss
                self.best_model_state = self.model.state_dict().copy()
                self.patience_counter = 0
            else:
                self.patience_counter += 1
            
            # Learning rate scheduling
            self.scheduler.step(val_loss)
            
            # Print progress
            if epoch % 25 == 0:
                print(f'Epoch {epoch:3d}/{epochs}: '
                      f'Train Loss = {train_loss:.6f}, '
                      f'Val Loss = {val_loss:.6f}, '
                      f'LR = {current_lr:.2e}, '
                      f'Patience = {self.patience_counter}/{self.max_patience}')
            
            # Early stopping
            if self.patience_counter >= self.max_patience:
                print(f"\nEarly stopping at epoch {epoch}")
                break
        
        # Load best model
        self.model.load_state_dict(self.best_model_state)
        print(f"\nTraining completed. Best validation loss: {self.best_val_loss:.6f}")
        
        return self.train_losses, self.val_losses

def calculate_comprehensive_metrics(model, data_loader):
    """Calculate comprehensive evaluation metrics"""
    model.eval()
    device = next(model.parameters()).device
    
    all_predictions = []
    all_targets = []
    
    with torch.no_grad():
        for batch_X, batch_y in data_loader:
            batch_X, batch_y = batch_X.to(device), batch_y.to(device)
            predictions = model(batch_X)
            all_predictions.append(predictions.cpu())
            all_targets.append(batch_y.cpu())
    
    predictions = torch.cat(all_predictions)
    targets = torch.cat(all_targets)
    
    # Calculate metrics
    mse = nn.MSELoss()(predictions, targets).item()
    mae = nn.L1Loss()(predictions, targets).item()
    rmse = np.sqrt(mse)
    
    # R-squared
    ss_res = torch.sum((targets - predictions) ** 2)
    ss_tot = torch.sum((targets - torch.mean(targets)) ** 2)
    r_squared = 1 - ss_res / ss_tot
    
    # Mean Absolute Percentage Error
    mape = torch.mean(torch.abs((targets - predictions) / (targets + 1e-8))) * 100
    
    return {
        'mse': mse,
        'mae': mae,
        'rmse': rmse,
        'r_squared': r_squared.item(),
        'mape': mape.item(),
        'predictions': predictions.numpy(),
        'targets': targets.numpy()
    }

def run_enhanced_training_pipeline():
    # Create enhanced dataset
    print("Creating enhanced dataset...")
    dataset = EnhancedMigrationDataset()
    
    # Split data
    train_size = int(0.7 * len(dataset))
    val_size = int(0.15 * len(dataset))
    test_size = len(dataset) - train_size - val_size
    
    train_dataset, val_dataset, test_dataset = torch.utils.data.random_split(
        dataset, [train_size, val_size, test_size]
    )
    
    # Create data loaders
    train_loader = DataLoader(train_dataset, batch_size=128, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=128, shuffle=False)
    test_loader = DataLoader(test_dataset, batch_size=128, shuffle=False)
    
    # Initialize model
    model = EnhancedMigrationPredictor(
        input_dim=15, 
        hidden_dims=[512, 256, 128, 64, 32], 
        dropout_rate=0.2,
        use_batch_norm=True
    )
    model.to(device)
    
    print(f"\nModel Parameters: {sum(p.numel() for p in model.parameters()):,}")
    
    # Initialize trainer
    trainer = AdvancedMigrationTrainer(model, train_loader, val_loader, learning_rate=0.001)
    
    # Train model
    train_losses, val_losses = trainer.train(epochs=300)
    
    # Evaluate on test set
    test_metrics = calculate_comprehensive_metrics(model, test_loader)
    
    print(f"\nTest Set Performance:")
    print(f"MSE:  {test_metrics['mse']:.6f}")
    print(f"MAE:  {test_metrics['mae']:.6f}")
    print(f"RMSE: {test_metrics['rmse']:.6f}")
    print(f"R²:   {test_metrics['r_squared']:.4f}")
    print(f"MAPE: {test_metrics['mape']:.2f}%")
    
    return model, dataset, trainer, test_metrics, train_losses, val_losses

# Run training
model, dataset, trainer, test_metrics, train_losses, val_losses = run_enhanced_training_pipeline()

def plot_enhanced_results(train_losses, val_losses, test_metrics, dataset):
    fig, axes = plt.subplots(2, 3, figsize=(10, 8))
    
    # Plot training history
    axes[0, 0].plot(train_losses, label='Training Loss', alpha=0.7, linewidth=2)
    axes[0, 0].plot(val_losses, label='Validation Loss', alpha=0.7, linewidth=2)
    axes[0, 0].set_xlabel('Epoch')
    axes[0, 0].set_ylabel('Loss (MSE)')
    axes[0, 0].set_title('Training History')
    axes[0, 0].legend()
    axes[0, 0].grid(True, alpha=0.3)
    axes[0, 0].set_yscale('log')
    
    # Plot predictions vs actual
    predictions = test_metrics['predictions']
    targets = test_metrics['targets']
    
    scatter = axes[0, 1].scatter(targets, predictions, alpha=0.3, s=10, c=targets, cmap='viridis')
    min_val = min(targets.min(), predictions.min())
    max_val = max(targets.max(), predictions.max())
    axes[0, 1].plot([min_val, max_val], [min_val, max_val], 'r--', alpha=0.8, linewidth=2)
    axes[0, 1].set_xlabel('Actual Probability')
    axes[0, 1].set_ylabel('Predicted Probability')
    axes[0, 1].set_title(f'Predictions vs Actual (R² = {test_metrics["r_squared"]:.4f})')
    axes[0, 1].grid(True, alpha=0.3)
    plt.colorbar(scatter, ax=axes[0, 1], label='Actual Probability')
    
    # Plot residuals
    residuals = targets - predictions
    axes[0, 2].scatter(predictions, residuals, alpha=0.3, s=10, c=targets, cmap='viridis')
    axes[0, 2].axhline(y=0, color='r', linestyle='--', alpha=0.8)
    axes[0, 2].set_xlabel('Predicted Probability')
    axes[0, 2].set_ylabel('Residuals')
    axes[0, 2].set_title('Residual Plot')
    axes[0, 2].grid(True, alpha=0.3)
    plt.colorbar(scatter, ax=axes[0, 2], label='Actual Probability')
    
    # Plot distribution
    axes[1, 0].hist(targets, bins=50, alpha=0.7, label='Actual', density=True, color='blue')
    axes[1, 0].hist(predictions, bins=50, alpha=0.7, label='Predicted', density=True, color='orange')
    axes[1, 0].set_xlabel('Probability')
    axes[1, 0].set_ylabel('Density')
    axes[1, 0].set_title('Distribution of Probabilities')
    axes[1, 0].legend()
    axes[1, 0].grid(True, alpha=0.3)
    
    # Feature correlation heatmap (sample)
    features_sample, _ = dataset.get_raw_data()
    features_df = pd.DataFrame(features_sample.numpy()[:1000], columns=dataset.get_feature_names())
    correlation_matrix = features_df.corr()
    
    im = axes[1, 1].imshow(correlation_matrix, cmap='coolwarm', aspect='auto', vmin=-1, vmax=1)
    axes[1, 1].set_xticks(range(len(dataset.get_feature_names())))
    axes[1, 1].set_xticklabels(dataset.get_feature_names(), rotation=45, ha='right')
    axes[1, 1].set_yticks(range(len(dataset.get_feature_names())))
    axes[1, 1].set_yticklabels(dataset.get_feature_names())
    axes[1, 1].set_title('Feature Correlations')
    plt.colorbar(im, ax=axes[1, 1])
    
    # Error distribution
    axes[1, 2].hist(residuals, bins=50, alpha=0.7, color='green', density=True)
    axes[1, 2].axvline(x=0, color='r', linestyle='--', alpha=0.8)
    axes[1, 2].set_xlabel('Prediction Error')
    axes[1, 2].set_ylabel('Density')
    axes[1, 2].set_title('Error Distribution')
    axes[1, 2].grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.show()

plot_enhanced_results(train_losses, val_losses, test_metrics, dataset)

class EnhancedMigrationPredictor(nn.Module):
    def __init__(self, input_dim=15, hidden_dims=[512, 256, 128, 64, 32], 
                 dropout_rate=0.3, use_batch_norm=True):
        super(EnhancedMigrationPredictor, self).__init__()
        
        self.input_dim = input_dim
        self.hidden_dims = hidden_dims
        
        # Build hidden layers
        layers = []
        prev_dim = input_dim
        
        for i, hidden_dim in enumerate(hidden_dims):
            layers.append(nn.Linear(prev_dim, hidden_dim))
            
            if use_batch_norm:
                layers.append(nn.BatchNorm1d(hidden_dim))
            
            layers.append(nn.ReLU())
            layers.append(nn.Dropout(dropout_rate))
            prev_dim = hidden_dim
        
        self.feature_extractor = nn.Sequential(*layers)
        
        # Output layer
        self.output_layer = nn.Sequential(
            nn.Linear(prev_dim, 1),
            nn.Sigmoid()  # Output between 0 and 1
        )
        
        # Initialize weights
        self._initialize_weights()
    
    def _initialize_weights(self):
        """Initialize weights using appropriate methods"""
        for module in self.modules():
            if isinstance(module, nn.Linear):
                nn.init.kaiming_uniform_(module.weight, nonlinearity='relu')
                if module.bias is not None:
                    nn.init.constant_(module.bias, 0)
    
    def forward(self, x):
        features = self.feature_extractor(x)
        output = self.output_layer(features)
        return output.squeeze()
    
    def evaluate(self, data_loader, criterion=None):
        """Evaluate the model on given data loader"""
        if criterion is None:
            criterion = nn.MSELoss()
            
        self.eval()
        device = next(self.parameters()).device
        
        total_loss = 0.0
        all_predictions = []
        all_targets = []
        
        with torch.no_grad():
            for batch_X, batch_y in data_loader:
                batch_X, batch_y = batch_X.to(device), batch_y.to(device)
                predictions = self(batch_X)
                loss = criterion(predictions, batch_y)
                
                total_loss += loss.item()
                all_predictions.append(predictions.cpu())
                all_targets.append(batch_y.cpu())
        
        # Concatenate all batches
        all_predictions = torch.cat(all_predictions)
        all_targets = torch.cat(all_targets)
        
        # Calculate additional metrics
        mse = criterion(all_predictions, all_targets).item()
        mae = nn.L1Loss()(all_predictions, all_targets).item()
        rmse = np.sqrt(mse)
        
        # R-squared
        ss_res = torch.sum((all_targets - all_predictions) ** 2)
        ss_tot = torch.sum((all_targets - torch.mean(all_targets)) ** 2)
        r_squared = 1 - ss_res / ss_tot
        
        # Mean Absolute Percentage Error
        mape = torch.mean(torch.abs((all_targets - all_predictions) / (all_targets + 1e-8))) * 100
        
        metrics = {
            'loss': total_loss / len(data_loader),
            'mse': mse,
            'mae': mae,
            'rmse': rmse,
            'r_squared': r_squared.item(),
            'mape': mape.item(),
            'predictions': all_predictions.numpy(),
            'targets': all_targets.numpy()
        }
        
        return metrics
    
    def get_feature_importance(self, data_loader, n_permutations=5):
        """Calculate approximate feature importance using permutation"""
        self.eval()
        baseline_metrics = self.evaluate(data_loader)
        baseline_mse = baseline_metrics['mse']
        
        feature_importance = np.zeros(self.input_dim)
        
        for feature_idx in range(self.input_dim):
            permutation_mses = []
            
            for _ in range(n_permutations):
                # Create modified data with shuffled feature
                modified_features = []
                modified_targets = []
                
                for batch_X, batch_y in data_loader:
                    # Shuffle the specific feature
                    shuffled_X = batch_X.clone()
                    shuffle_idx = torch.randperm(shuffled_X.size(0))
                    shuffled_X[:, feature_idx] = shuffled_X[shuffle_idx, feature_idx]
                    
                    modified_features.append(shuffled_X)
                    modified_targets.append(batch_y)
                
                # Calculate MSE with shuffled feature
                all_preds = []
                all_targets = []
                
                with torch.no_grad():
                    for batch_X, batch_y in zip(modified_features, modified_targets):
                        preds = self(batch_X)
                        all_preds.append(preds)
                        all_targets.append(batch_y)
                
                all_preds = torch.cat(all_preds)
                all_targets = torch.cat(all_targets)
                shuffled_mse = nn.MSELoss()(all_preds, all_targets).item()
                permutation_mses.append(shuffled_mse)
            
            # Average importance over permutations
            avg_shuffled_mse = np.mean(permutation_mses)
            importance = avg_shuffled_mse - baseline_mse
            feature_importance[feature_idx] = importance
        
        return feature_importance
    
# Create predictor
predictor = EnhancedMigrationPredictor(model, dataset)

# Example predictions
print("Enhanced Migration Probability Predictions:")
print("=" * 90)

examples = [
    # Format: (pop_a, pop_b, dist, amenity_a, amenity_b, shop_a, shop_b, tourism_a, tourism_b, 
    #          leisure_a, leisure_b, office_a, office_b, public_transport_a, public_transport_b)
    (1000000, 800000, 50, 120, 100, 250, 200, 60, 50, 80, 70, 120, 100, 50, 40),
    (500000, 300000, 200, 60, 40, 120, 80, 25, 20, 40, 30, 60, 40, 25, 15),
    (50000, 30000, 500, 8, 5, 15, 10, 3, 2, 6, 4, 8, 5, 3, 2),
    (100000, 100000, 1000, 15, 15, 30, 30, 8, 8, 12, 12, 15, 15, 6, 6),
    (5000000, 5000000, 10, 600, 600, 1200, 1200, 250, 250, 400, 400, 600, 600, 200, 200),
]

feature_labels = [
    "Pop A", "Pop B", "Dist", "Amt A", "Amt B", "Shop A", "Shop B", 
    "Tour A", "Tour B", "Leis A", "Leis B", "Off A", "Off B", "PT A", "PT B"
]

header = " | ".join(f"{label:>8}" for label in feature_labels) + " | Probability"
print(header)
print("-" * len(header))

for example in examples:
    prob = predictor.predict(*example)
    values = " | ".join(f"{val:8,d}" if i < 3 else f"{val:8,d}" for i, val in enumerate(example))
    print(f"{values} | {prob:>11.4f}")

# Analyze feature importance
feature_importance = predictor.analyze_feature_importance(n_samples=500)
print("\nTop 5 Most Important Features:")
for feature, importance in sorted(feature_importance.items(), key=lambda x: x[1], reverse=True)[:5]:
    print(f"  {feature}: {importance:.6f}")

def save_enhanced_model(model, dataset, filepath='enhanced_migration_predictor.pth'):
    """Save the trained model and preprocessing parameters"""
    torch.save({
        'model_state_dict': model.state_dict(),
        'scaler_mean': dataset.scaler.mean_,
        'scaler_scale': dataset.scaler.scale_,
        'feature_names': dataset.get_feature_names(),
        'model_config': {
            'input_dim': 15,
            'hidden_dims': [512, 256, 128, 64, 32],
            'dropout_rate': 0.2,
            'use_batch_norm': True
        }
    }, filepath)
    print(f"Enhanced model saved to {filepath}")

def load_enhanced_model(filepath='enhanced_migration_predictor.pth'):
    """Load the trained model and preprocessing parameters"""
    checkpoint = torch.load(filepath, map_location='cpu')
    
    # Create model architecture
    model = EnhancedMigrationPredictor(**checkpoint['model_config'])
    model.load_state_dict(checkpoint['model_state_dict'])
    
    # Create dummy dataset for scaler and feature names
    dataset = EnhancedMigrationDataset(n_samples=1)
    dataset.scaler.mean_ = checkpoint['scaler_mean']
    dataset.scaler.scale_ = checkpoint['scaler_scale']
    
    predictor = EnhancedMigrationPredictor(model, dataset)
    print(f"Enhanced model loaded from {filepath}")
    
    return predictor

# Save the model
save_enhanced_model(model, dataset)

# Example of loading the model
# loaded_predictor = load_enhanced_model()