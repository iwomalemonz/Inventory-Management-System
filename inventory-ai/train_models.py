"""Optional: retrain the models by hand and print their test scores.

    python train_models.py

The app also trains automatically the first time it starts, and saves the
result to models/demand_models.joblib. Delete that file (or run this script)
after changing the dataset.
"""
from inventory.models import DemandModels

if __name__ == "__main__":
    DemandModels.load_or_train(force=True)
    print("Saved to models/demand_models.joblib")
