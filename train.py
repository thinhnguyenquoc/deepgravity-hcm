import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.optim as optim
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
import matplotlib.pyplot as plt
import seaborn as sns

class MigrationDataGenerator:
    def __init__(self, n_samples=10000):
        self.n_samples = n_samples
        self.scaler = StandardScaler()
    
    def generate_data(self):
        """Generate synthetic migration data based on gravity model principles"""
        np.random.seed(42)
        
        # Generate population data (log-normal distribution)
        pop_a = np.random.lognormal(10, 1, self.n_samples)  # Population destination A
        pop_b = np.random.lognormal(10, 1, self.n_samples)  # Population destination B
        distance = np.random.lognormal(5, 1, self.n_samples)  # Distance between A and B
        
        # Gravity model: migration ∝ (pop_a * pop_b) / distance^2
        # Add some noise and normalize to probabilities
        gravity_factor = (pop_a * pop_b) / (distance**2 + 1)
        noise = np.random.normal(0, 0.1, self.n_samples)
        
        # Convert to probabilities using sigmoid
        raw_probability = gravity_factor + noise
        probability = 1 / (1 + np.exp(-raw_probability))
        
        # Ensure probabilities are between 0 and 1
        probability = np.clip(probability, 0, 0.5)  # Cap at 0.5 since migration is rare
        
        data = pd.DataFrame({
            'population_a': pop_a,
            'population_b': pop_b,
            'distance': distance,
            'move_probability': probability
        })
        
        return data
    
    def prepare_features(self, data):
        """Prepare and scale features"""
        features = data[['population_a', 'population_b', 'distance']]
        
        # Log transform populations and distance (common in gravity models)
        features_log = np.log(features + 1)
        
        # Scale features
        features_scaled = self.scaler.fit_transform(features_log)
        
        return features_scaled, data['move_probability'].values

# Generate data
# data_gen = MigrationDataGenerator(n_samples=10000)
# data = data_gen.generate_data()

# print("Data sample:")
# print(data.head())
# print(f"\nData shape: {data.shape}")
# print(f"Probability range: [{data['move_probability'].min():.3f}, {data['move_probability'].max():.3f}]")

class MigrationPredictor(nn.Module):
    def __init__(self, input_dim=3, hidden_dims=[64, 32, 16], dropout_rate=0.2):
        super(MigrationPredictor, self).__init__()
        
        layers = []
        prev_dim = input_dim
        
        # Create hidden layers
        for hidden_dim in hidden_dims:
            layers.extend([
                nn.Linear(prev_dim, hidden_dim),
                nn.BatchNorm1d(hidden_dim),
                nn.ReLU(),
                nn.Dropout(dropout_rate)
            ])
            prev_dim = hidden_dim
        
        self.feature_layers = nn.Sequential(*layers)
        self.output_layer = nn.Sequential(
            nn.Linear(prev_dim, 1),
            nn.Sigmoid()  # Output probability between 0 and 1
        )
    
    def forward(self, x):
        features = self.feature_layers(x)
        output = self.output_layer(features)
        return output.squeeze()

class MigrationTrainer:
    def __init__(self, model, learning_rate=0.001):
        self.model = model
        self.criterion = nn.MSELoss()  # Mean Squared Error for regression
        self.optimizer = optim.Adam(model.parameters(), lr=learning_rate)
        self.scheduler = optim.lr_scheduler.ReduceLROnPlateau(self.optimizer, patience=5)
    
    def train(self, train_loader, val_loader, epochs=100):
        train_losses = []
        val_losses = []
        
        for epoch in range(epochs):
            # Training phase
            self.model.train()
            train_loss = 0.0
            
            for batch_X, batch_y in train_loader:
                self.optimizer.zero_grad()
                predictions = self.model(batch_X)
                loss = self.criterion(predictions, batch_y)
                loss.backward()
                self.optimizer.step()
                train_loss += loss.item()
            
            # Validation phase
            self.model.eval()
            val_loss = 0.0
            with torch.no_grad():
                for batch_X, batch_y in val_loader:
                    predictions = self.model(batch_X)
                    loss = self.criterion(predictions, batch_y)
                    val_loss += loss.item()
            
            train_loss /= len(train_loader)
            val_loss /= len(val_loader)
            
            train_losses.append(train_loss)
            val_losses.append(val_loss)
            
            self.scheduler.step(val_loss)
            
            if epoch % 10 == 0:
                print(f'Epoch {epoch:3d}: Train Loss = {train_loss:.6f}, Val Loss = {val_loss:.6f}')
        
        return train_losses, val_losses
    
def run_complete_pipeline():
    # Generate and prepare data
    print("Generating data...")
    data_gen = MigrationDataGenerator(n_samples=10000)
    data = data_gen.generate_data()
    
    X, y = data_gen.prepare_features(data)
    
    # Split data
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42
    )
    X_train, X_val, y_train, y_val = train_test_split(
        X_train, y_train, test_size=0.25, random_state=42
    )
    
    # Convert to PyTorch tensors
    X_train_tensor = torch.FloatTensor(X_train)
    y_train_tensor = torch.FloatTensor(y_train)
    X_val_tensor = torch.FloatTensor(X_val)
    y_val_tensor = torch.FloatTensor(y_val)
    X_test_tensor = torch.FloatTensor(X_test)
    y_test_tensor = torch.FloatTensor(y_test)
    
    # Create data loaders
    train_dataset = torch.utils.data.TensorDataset(X_train_tensor, y_train_tensor)
    val_dataset = torch.utils.data.TensorDataset(X_val_tensor, y_val_tensor)
    test_dataset = torch.utils.data.TensorDataset(X_test_tensor, y_test_tensor)
    
    train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=32, shuffle=True)
    val_loader = torch.utils.data.DataLoader(val_dataset, batch_size=32, shuffle=False)
    test_loader = torch.utils.data.DataLoader(test_dataset, batch_size=32, shuffle=False)
    
    # Initialize model and trainer
    model = MigrationPredictor(input_dim=3, hidden_dims=[128, 64, 32, 16], dropout_rate=0.3)
    trainer = MigrationTrainer(model, learning_rate=0.001)
    
    print(f"Model architecture:\n{model}")
    print(f"Total parameters: {sum(p.numel() for p in model.parameters()):,}")
    
    # Train model
    print("\nStarting training...")
    train_losses, val_losses = trainer.train(train_loader, val_loader, epochs=20)
    
    # Evaluate model
    model.eval()
    with torch.no_grad():
        test_predictions = model(X_test_tensor)
        test_loss = trainer.criterion(test_predictions, y_test_tensor)
        
        # Calculate R-squared
        ss_res = torch.sum((y_test_tensor - test_predictions) ** 2)
        ss_tot = torch.sum((y_test_tensor - torch.mean(y_test_tensor)) ** 2)
        r_squared = 1 - ss_res / ss_tot
    
    print(f"\nFinal Test Loss: {test_loss:.6f}")
    print(f"R-squared: {r_squared:.4f}")
    
    return model, data_gen, train_losses, val_losses, test_predictions, y_test_tensor

# Run the complete pipeline
model, data_gen, train_losses, val_losses, test_pred, test_true = run_complete_pipeline()

def plot_results(train_losses, val_losses, test_pred, test_true):
    fig, axes = plt.subplots(2, 2, figsize=(10, 8))
    
    # Plot training history
    axes[0, 0].plot(train_losses, label='Training Loss')
    axes[0, 0].plot(val_losses, label='Validation Loss')
    axes[0, 0].set_xlabel('Epoch')
    axes[0, 0].set_ylabel('Loss')
    axes[0, 0].set_title('Training History')
    axes[0, 0].legend()
    axes[0, 0].grid(True, alpha=0.3)
    
    # Plot predictions vs actual
    axes[0, 1].scatter(test_true.numpy(), test_pred.numpy(), alpha=0.5)
    axes[0, 1].plot([0, test_true.max()], [0, test_true.max()], 'r--', alpha=0.8)
    axes[0, 1].set_xlabel('Actual Probability')
    axes[0, 1].set_ylabel('Predicted Probability')
    axes[0, 1].set_title('Predictions vs Actual')
    axes[0, 1].grid(True, alpha=0.3)
    
    # Plot residuals
    residuals = test_true.numpy() - test_pred.numpy()
    axes[1, 0].scatter(test_pred.numpy(), residuals, alpha=0.5)
    axes[1, 0].axhline(y=0, color='r', linestyle='--')
    axes[1, 0].set_xlabel('Predicted Probability')
    axes[1, 0].set_ylabel('Residuals')
    axes[1, 0].set_title('Residual Plot')
    axes[1, 0].grid(True, alpha=0.3)
    
    # Plot distribution of predictions
    axes[1, 1].hist(test_pred.numpy(), bins=50, alpha=0.7, label='Predicted')
    axes[1, 1].hist(test_true.numpy(), bins=50, alpha=0.7, label='Actual')
    axes[1, 1].set_xlabel('Probability')
    axes[1, 1].set_ylabel('Frequency')
    axes[1, 1].set_title('Distribution of Probabilities')
    axes[1, 1].legend()
    axes[1, 1].grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.show()

# plot_results(train_losses, val_losses, test_pred, test_true)

def predict_migration_probability(model, data_gen, population_a, population_b, distance):
    """
    Predict migration probability for new data
    
    Args:
        model: Trained model
        data_gen: Data generator with fitted scaler
        population_a: Population of destination A
        population_b: Population of destination B  
        distance: Distance between A and B
    
    Returns:
        probability: Predicted migration probability
    """
    model.eval()
    
    # Prepare input
    input_data = np.array([[population_a, population_b, distance]])
    input_log = np.log(input_data + 1)
    input_scaled = data_gen.scaler.transform(input_log)
    
    # Convert to tensor and predict
    input_tensor = torch.FloatTensor(input_scaled)
    
    with torch.no_grad():
        probability = model(input_tensor).item()
    
    return probability

# Example predictions
print("Example Predictions:")
print("-" * 50)

examples = [
    (100000, 50000, 100),   # Large populations, short distance
    (10000, 8000, 500),     # Medium populations, medium distance  
    (5000, 3000, 1000),     # Small populations, long distance
    (1000000, 1000000, 50)  # Very large populations, very short distance
]

for pop_a, pop_b, dist in examples:
    prob = predict_migration_probability(model, data_gen, pop_a, pop_b, dist)
    print(f"Pop A: {pop_a:8,d}, Pop B: {pop_b:8,d}, Distance: {dist:4} km -> Probability: {prob:.4f}")