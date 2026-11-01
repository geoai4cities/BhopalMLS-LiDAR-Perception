# Bhopal Class Mapping Files

This directory contains two class mapping files for the Bhopal V13 dataset.

## Files

### 1. `Bhopal_class_mapping_V13.csv` (COMPLETE MAPPING)

**Purpose**: Complete reference mapping for all classes in the dataset

**Usage**:
- Data organization and folder structure
- Reference for understanding original labeling
- Documentation purposes

**Details**:
- Total classes: 71 (codes 0-70)
- Includes ALL classes (training + non-training)
- XX=0 classes (codes 0-8): Non-training classes
  - Unlabeled, Noise Points, Moving objects, etc.
  - Not used for model training
- XX≠0 classes (codes 9-70): Training classes
  - 62 classes used for model training

**Columns**:
- `XX`: Major class category (0-12)
- `ZZ`: Sub-class within category (0-14)
- `class_name`: Human-readable class name
- `class_code`: Original sequential class code (0-70)

---

### 2. `Bhopal_class_mapping_V13_TRAINING.csv` (TRAINING MAPPING)

**Purpose**: Optimized mapping for neural network training

**Usage**:
- **Use this file for model training**
- Neural network class predictions
- Loss calculation and evaluation
- Model architecture definition

**Details**:
- Total classes: 62 (codes 0-61)
- Only includes training classes (XX≠0)
- Sequential codes from 0-61
- No gaps in class codes
- Efficient for neural networks

**Columns**:
- `training_class_code`: New sequential code for training (0-61)
- `XX`: Major class category (1-12)
- `ZZ`: Sub-class within category (0-14)
- `class_name`: Human-readable class name
- `original_class_code`: Original code from V13 mapping (9-70)

---

## Why Two Mappings?

### Problem with Original Mapping
- Neural networks would need 71 output neurons
- 9 neurons (codes 0-8) would be unused
- Wasted model capacity
- Less efficient training

### Solution with Training Mapping
- Neural networks need only 62 output neurons
- All neurons are used efficiently
- Sequential codes (0-61) are easier to work with
- Clear separation of training vs non-training classes

---

## Usage Examples

### Loading Training Mapping (Python)

```python
import pandas as pd

# Load training mapping
mapping = pd.read_csv('Bhopal_class_mapping_V13_TRAINING.csv')

# Get number of classes for model
num_classes = len(mapping)  # 62

# Convert original class code to training code
def original_to_training(original_code):
    row = mapping[mapping['original_class_code'] == original_code]
    if len(row) > 0:
        return row['training_class_code'].iloc[0]
    return None  # XX=0 class, not used for training

# Convert training code to class name
def training_to_name(training_code):
    row = mapping[mapping['training_class_code'] == training_code]
    return row['class_name'].iloc[0] if len(row) > 0 else 'Unknown'

# Example usage
print(f"Number of training classes: {num_classes}")
print(f"Class 0: {training_to_name(0)}")  # Ground
print(f"Original code 17 -> Training code: {original_to_training(17)}")  # Building -> 8
```

### Model Architecture

```python
import torch.nn as nn

# Load mapping to get number of classes
mapping = pd.read_csv('Bhopal_class_mapping_V13_TRAINING.csv')
num_classes = len(mapping)  # 62

# Define model output layer
output_layer = nn.Linear(hidden_size, num_classes)  # 62 outputs, not 71!
```

---

## Class Distribution

### By Major Category (XX)

| XX | Category | Number of Classes |
|----|----------|-------------------|
| 1  | Ground   | 4 |
| 2  | Vegetation | 4 |
| 3  | Building | 5 |
| 4  | Road | 1 |
| 5  | Wire | 1 |
| 6  | Pole and Tower | 4 |
| 7  | Wire-Structure Connector | 1 |
| 8  | Vehicle | 15 |
| 9  | Board | 3 |
| 10 | Fencing | 5 |
| 11 | Human and Animal | 7 |
| 12 | Road Side Object | 12 |
| **Total** | | **62** |

---

## Excluded Classes (XX=0)

The following classes are **NOT** included in training mapping:

1. Unlabeled (code 0)
2. Noise Points (code 1)
3. High Noise Points (code 2)
4. Moving Vehicle (code 3)
5. Moving Humans (code 4)
6. Moving Female (code 5)
7. Moving Male (code 6)
8. Moving Animal (code 7)
9. Mobike Rider Moving (code 8)

**Reason**: These are temporary/noise/dynamic objects not useful for static scene understanding.

---

## Important Notes

1. **Always use `Bhopal_class_mapping_V13_TRAINING.csv` for model training**
2. When loading data, filter out files with XX=0 classes
3. Training class codes are 0-indexed (0-61)
4. Original class codes (9-70) can be mapped back using the `original_class_code` column
5. Keep both mappings for reference and compatibility

---

## File Statistics

### Complete Mapping (V13)
- Total classes: 71
- Non-training (XX=0): 9 classes (codes 0-8)
- Training (XX≠0): 62 classes (codes 9-70)

### Training Mapping (V13_TRAINING)
- Total classes: 62
- Training class codes: 0-61
- All classes are for training (XX≠0)

---

## Created
- Complete mapping: Original Bhopal V13 dataset
- Training mapping: 2025-11-03
- Purpose: Optimized neural network training
