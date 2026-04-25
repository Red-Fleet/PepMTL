from model import *
from sklearn.metrics import matthews_corrcoef, accuracy_score, precision_score, recall_score, f1_score, roc_auc_score, confusion_matrix
import json
# This code is for generating test results, it reads the model weight and dataset
# Then splits dataset in train, val and test with Random State = 42
# Random State = 42 --- this is important to make sure test set does not gets dirty with train data
DEVICE    = 'cuda' if torch.cuda.is_available() else 'cpu'
MODEL_PATH = 'best_model.pt'
THRESHOLD_PATH = 'best_model.json'
DATASET_PATH = 'cleaned_clustered_dataset.json'

with open(DATASET_PATH) as f:
    clustered_dataset = json.load(f)

train_val_test_ratio = [0.7, 0.1, 0.2]
random_state = 42

if len(train_val_test_ratio) != 3:
    raise ValueError("train_val_test_ratio must contain [train, val, test].")

ratio_sum = sum(train_val_test_ratio)
if ratio_sum <= 0:
    raise ValueError("Sum of train_val_test_ratio must be > 0.")

train_ratio, val_ratio, test_ratio = [r / ratio_sum for r in train_val_test_ratio]

all_seqs_count = sum(len(cluster) for cluster in clustered_dataset)
print('all_seqs_count', all_seqs_count)

train_size = int(all_seqs_count * train_ratio)
val_size = int(all_seqs_count * val_ratio)
test_size = all_seqs_count - train_size - val_size

train_seqs = []
val_seqs = []
test_seqs = []

train_cluster_count = 0
val_cluster_count = 0
test_cluster_count = 0

all_cluster_ids = [i for i in range(len(clustered_dataset))]
rng = random.Random(random_state)
rng.shuffle(all_cluster_ids)

for cluster_id in all_cluster_ids:
    cluster = clustered_dataset[cluster_id]
    if len(train_seqs) < train_size:
        train_seqs.extend(cluster)
        train_cluster_count += 1
    elif len(val_seqs) < val_size:
        val_seqs.extend(cluster)
        val_cluster_count += 1
    else:
        test_seqs.extend(cluster)
        test_cluster_count += 1

print('train_seqs', len(train_seqs))
print('val_seqs', len(val_seqs))
print('test_seqs', len(test_seqs))

print('train_cluster_count', train_cluster_count)
print('val_cluster_count', val_cluster_count)
print('test_cluster_count', test_cluster_count)

# other-functional is present with known functionl class then remove other-functional
for item in test_seqs:
    if 'other-functional' in item['classes'] and len(item['classes']) > 1:
        item['classes'].remove('other-functional')
        if len(item['classes']) == 0: raise Exception("no class left")

test_df_rows = []
for entry in test_seqs:
    row = [False]*len(endpoints)
    if len(set(entry['classes']).intersection(endpoints_set)) > 0: 

        for c in entry['classes']:
            if c in endpoints:
                row[endpoint_index[c]] = True

        row = [entry['sequence']] + row
        test_df_rows.append(row)

test_df = pd.DataFrame(test_df_rows, columns=['sequence'] + endpoints)


class PeptideDataset(Dataset):

    def __init__(
        self,
        dataframe,
        tokenizer,
        label_columns,
        max_length: int = 128
    ):
        """
        Args:
            dataframe (pd.DataFrame): DataFrame containing the data.
                                      Must have a 'sequence' column.
            tokenizer (AutoTokenizer): A Hugging Face tokenizer (e.g., for ESM-2).
            label_columns (List[str]): A list of column names that represent the labels.
            max_length (int): Maximum sequence length for padding/truncation.
        """
        self.df = dataframe
        self.tokenizer = tokenizer
        self.sequences = self.df['sequence'].values
        self.labels = self.df[label_columns].values
        self.label_columns = label_columns
        self.max_length = max_length

        functional_idxs = []
        non_functional_idxs = []
        for i, label in enumerate(self.labels):
            if label[endpoint_index['non-functional']] == True:
                non_functional_idxs.append(i)
            else:
                functional_idxs.append(i)
        
        non_functional_idxs = [non_functional_idxs[i: i+1] for i in range(0, len(non_functional_idxs), 1)]
        non_functional_idxs = [i for i in non_functional_idxs]

        functional_idxs = [functional_idxs[i: i+1] for i in range(0, len(functional_idxs), 1)]
        self.all_idxs = non_functional_idxs + functional_idxs

        # Pre-tokenize ALL sequences once
        sequences = dataframe['sequence'].tolist()
        encodings = tokenizer(
            sequences,
            add_special_tokens=True,
            max_length=max_length,
            padding='max_length',
            truncation=True,
            return_attention_mask=True,
            return_tensors='pt'
        )
        self.input_ids = encodings['input_ids']
        self.attention_masks = encodings['attention_mask']
        

    def __len__(self) -> int:
        """Returns the total number of samples in the dataset."""
        return len(self.all_idxs)

    def __getitem__(self, index: int):
        """
        Retrieves a single sample from the dataset.

        Args:
            index (int): The index of the sample to retrieve.

        Returns:
            A dictionary containing:
            - 'input_ids': Token IDs of the sequence.
            - 'attention_mask': Mask to avoid performing attention on padding tokens.
            - 'labels': A multi-hot encoded tensor of labels.
            - 'img_input': A dummy tensor to match the MultiModelNetwork's forward signature.
        """
        idx = random.choice(self.all_idxs[index])

        return {
            'input_ids': self.input_ids[idx],
            'attention_mask': self.attention_masks[idx],
            'labels': torch.FloatTensor(self.labels[idx])
        }

LABEL_COLUMNS = endpoints
MAX_LEN = 100
esm_tokenizer = AutoTokenizer.from_pretrained("facebook/esm2_t6_8M_UR50D")
test_dataset = PeptideDataset(test_df, esm_tokenizer, LABEL_COLUMNS, MAX_LEN)
test_loader = DataLoader(test_dataset, batch_size=64, shuffle=True)

# --- HELPER FUNCTIONS ---
def calculate_complete_metrics(y_true, y_pred_probs, thresholds, class_names):
    results = {}
    for i in range(y_true.shape[1]):
        y_pred_cls = (y_pred_probs[:, i] >= thresholds[i]).astype(int)
        mcc = matthews_corrcoef(y_true[:, i], y_pred_cls)
        acc = accuracy_score(y_true[:, i], y_pred_cls)
        prec = precision_score(y_true[:, i], y_pred_cls, zero_division=0)
        rec = recall_score(y_true[:, i], y_pred_cls, zero_division=0)
        f1 = f1_score(y_true[:, i], y_pred_cls, zero_division=0)
        try:
            auc_val = roc_auc_score(y_true[:, i], y_pred_probs[:, i])
        except ValueError:
            auc_val = 0.5
        tn, fp, fn, tp = confusion_matrix(y_true[:, i], y_pred_cls, labels=[0, 1]).ravel()
        results[class_names[i]] = {
            "mcc": float(mcc),
            "threshold": float(thresholds[i]),
            "accuracy": float(acc),
            "precision": float(prec),
            "recall": float(rec),
            "f1_score": float(f1),
            "roc_auc": float(auc_val),
            "tp": int(tp),
            "tn": int(tn),
            "fp": int(fp),
            "fn": int(fn),
        }
    return results

def find_optimal_thresholds(y_true, y_pred_probs):
    optimal_thresholds = []
    for i in range(y_true.shape[1]):
        best_mcc = -1
        best_threshold = 0.5
        for threshold in np.arange(0.1, 0.95, 0.05):
            y_pred_class = (y_pred_probs[:, i] >= threshold).astype(int)
            mcc = matthews_corrcoef(y_true[:, i], y_pred_class)
            if mcc > best_mcc:
                best_mcc = mcc
                best_threshold = threshold
        optimal_thresholds.append(best_threshold)
    return optimal_thresholds

def calculate_mcc(tp, tn, fp, fn):
    num = (tp * tn) - (fp * fn)
    den_sq = (tp + fp) * (tp + fn) * (tn + fp) * (tn + fn)
    return num / math.sqrt(den_sq) if den_sq > 0 else 0.0

def aggregate_mcc(*metric_groups):
    tp, tn, fp, fn = 0, 0, 0, 0
    for metrics in metric_groups:
        for values in metrics.values():
            tp += values['tp']
            tn += values['tn']
            fp += values['fp']
            fn += values['fn']
    return calculate_mcc(tp, tn, fp, fn)

def feature_mixup(features, labels, alpha=0.4):
    batch_size = features.size(0)
    lam = np.random.beta(alpha, alpha) if alpha > 0 else 1.0
    index = torch.randperm(batch_size, device=features.device)
    mixed_features = lam * features + (1 - lam) * features[index]
    mixed_labels = lam * labels + (1 - lam) * labels[index]
    return mixed_features, mixed_labels

def rdrop_kl_loss(logits1, logits2):
    kl1 = F.binary_cross_entropy_with_logits(logits1, torch.sigmoid(logits2).detach(), reduction='mean')
    kl2 = F.binary_cross_entropy_with_logits(logits2, torch.sigmoid(logits1).detach(), reduction='mean')
    return (kl1 + kl2) / 2

def enable_dropout(model):
    """
    Specifically tailored for PeptideNetwork to activate all sources of dropout
    during Test-Time Augmentation (TTA), including functional dropouts hidden 
    inside complex PyTorch modules.
    """
    for module in model.modules():
        class_name = module.__class__.__name__
        
        # 1. Standard explicit dropout layers (Dropout, Dropout1d, Dropout2d)
        if class_name.startswith('Dropout'):
            module.train()
            
        # 2. Cross-Attention functional dropout
        elif class_name == 'MultiheadAttention':
            module.train()
            
        # 3. Task Query Decoder functional dropout
        elif class_name in ['TransformerDecoder', 'TransformerDecoderLayer']:
            module.train()
            
        # 4. GRU functional dropout (applied between internal layers)
        elif class_name == 'GRU':
            module.train()


model = PeptideNetwork(num_classes=13, mask_token_id=32)
if os.path.exists(MODEL_PATH):
    print(f"Found {MODEL_PATH} ! Loading weights.")
    state_dict = torch.load(MODEL_PATH, map_location="cpu")
else:
    raise("Wrong model path")

model.load_state_dict(state_dict, strict=True)
model = model.to(DEVICE)

# reading threasholds 
with open(THRESHOLD_PATH) as f:
    data = json.load(f)
    print('reading thresholds from 72th epoch, as it gave the best val mcc')
    threshold_binary = data[71]['thresholds']['binary']
    threshold_functional = data[71]['thresholds']['functional']

func_names = [index_endpoint[i] for i in FUNC_INDICES]
bin_names = ["NON-FUNCTIONAL"]

mixup_alpha = 0.4
rdrop_alpha = 1.0
tta_passes = 5

# --- TEST (TTA) ---
total_test_loss = 0.0
test_bin_y = []
test_spec_y = []
test_tta_bin_preds = [[] for _ in range(tta_passes)]
test_tta_spec_preds = [[] for _ in range(tta_passes)]

with torch.no_grad():
    for data in test_loader:
        input_ids = data['input_ids'].to(DEVICE)
        attention_mask = data['attention_mask'].to(DEVICE)
        labels = data['labels'].to(DEVICE)

        target_bin = labels[:, NON_FUNC_IDX].unsqueeze(1).float()
        target_spec = labels[:, FUNC_INDICES]

        pred_bin, pred_spec = model(input_ids, attention_mask)
        
        test_bin_y.append(target_bin.cpu().numpy())
        test_spec_y.append(target_spec.cpu().numpy())
        
        # First pass predictions
        test_tta_bin_preds[0].append(torch.sigmoid(pred_bin).detach().cpu().numpy())
        test_tta_spec_preds[0].append(torch.sigmoid(pred_spec).detach().cpu().numpy())

        # Additional TTA passes
        if tta_passes > 1:
            enable_dropout(model)
            for t in range(1, tta_passes):
                pred_bin_t, pred_spec_t = model(input_ids, attention_mask)
                test_tta_bin_preds[t].append(torch.sigmoid(pred_bin_t).detach().cpu().numpy())
                test_tta_spec_preds[t].append(torch.sigmoid(pred_spec_t).detach().cpu().numpy())
            model.eval()

y_true_bin_test = np.concatenate(test_bin_y)
y_pred_bin_test = np.mean([np.concatenate(pred_list) for pred_list in test_tta_bin_preds], axis=0)

y_true_spec_test = np.concatenate(test_spec_y)
raw_pred_spec_test = np.mean([np.concatenate(pred_list) for pred_list in test_tta_spec_preds], axis=0)

# SOFT GATING for Test
y_pred_spec_test = raw_pred_spec_test * (1 - y_pred_bin_test)

test_bin_metrics = calculate_complete_metrics(y_true_bin_test, y_pred_bin_test, threshold_binary, bin_names)
test_spec_metrics = calculate_complete_metrics(y_true_spec_test, y_pred_spec_test, threshold_functional, func_names)
test_total_mcc = aggregate_mcc(test_bin_metrics, test_spec_metrics)

test_func_mccs = [metrics['mcc'] for metrics in test_spec_metrics.values() if not np.isnan(metrics['mcc'])]
test_avg_func_mcc = sum(test_func_mccs) / len(test_func_mccs) if test_func_mccs else 0.0


import pandas as pd

# Combine both dictionaries
all_metrics = {**test_bin_metrics, **test_spec_metrics}

# Format into a DataFrame
df_metrics = pd.DataFrame.from_dict(all_metrics, orient='index')

# Select only the specific columns you want
columns_to_show = ['accuracy', 'precision', 'recall', 'f1_score', 'mcc', 'roc_auc']
df_table = df_metrics[columns_to_show].copy()

# Rename the columns to match your requested format
df_table.rename(columns={
    'accuracy': 'Accuracy',
    'precision': 'Precision',
    'recall': 'Recall',
    'f1_score': 'F1-Score',
    'mcc': 'MCC',
    'roc_auc': 'ROC AUC'
}, inplace=True)

# Round to 3 decimal places
df_table = df_table.round(3)

# Print the formatted table
print(df_table.to_markdown())