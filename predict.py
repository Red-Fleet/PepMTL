from model import *
from sklearn.metrics import matthews_corrcoef, accuracy_score, precision_score, recall_score, f1_score, roc_auc_score, confusion_matrix
import json
# This code is for generating test results, it reads the model weight and dataset
# Then splits dataset in train, val and test with Random State = 42
# Random State = 42 --- this is important to make sure test set does not gets dirty with train data
DEVICE    = 'cuda' if torch.cuda.is_available() else 'cpu'
MODEL_PATH = 'best_model.pt'
THRESHOLD_PATH = 'best_model.json'


model = PeptideNetwork(num_classes=13, mask_token_id=32)
if os.path.exists(MODEL_PATH):
    print(f"Found {MODEL_PATH} ! Loading weights.")
    state_dict = torch.load(MODEL_PATH, map_location="cpu")
else:
    raise("Wrong model path")

esm_tokenizer = AutoTokenizer.from_pretrained("facebook/esm2_t6_8M_UR50D")
model.load_state_dict(state_dict, strict=True)
model = model.to(DEVICE)

func_names = [index_endpoint[i] for i in FUNC_INDICES]
bin_names = ["NON-FUNCTIONAL"]

# reading threasholds 
with open(THRESHOLD_PATH) as f:
    data = json.load(f)
    print('reading thresholds from 72th epoch, as it gave the best val mcc')
    threshold_binary = data[71]['thresholds']['binary']
    threshold_functional = data[71]['thresholds']['functional']




sequences = input("Enter Space Seprated Sequences: ")
sequences = sequences.split(' ')

BATCH_SIZE = 64
tta_passes = 5

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

test_tta_bin_preds = [[] for _ in range(tta_passes)]
test_tta_spec_preds = [[] for _ in range(tta_passes)]
for i in range(0, len(sequences), BATCH_SIZE):
    batch = sequences[i: i+BATCH_SIZE]
    encodings = esm_tokenizer(sequences, add_special_tokens=True, max_length=128,
                                padding='max_length', truncation=True, return_attention_mask=True, return_tensors='pt')

    input_ids = encodings['input_ids'].to(DEVICE)
    attention_mask = encodings['attention_mask'].to(DEVICE)


    pred_bin, pred_spec = model(input_ids, attention_mask)

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

y_pred_bin_test = np.mean([np.concatenate(pred_list) for pred_list in test_tta_bin_preds], axis=0)

raw_pred_spec_test = np.mean([np.concatenate(pred_list) for pred_list in test_tta_spec_preds], axis=0)

# SOFT GATING for Test
y_pred_spec_test = raw_pred_spec_test * (1 - y_pred_bin_test)

for i in range(len(sequences)):
    print(f"\n{'='*60}")
    print(f"Sequence: {sequences[i]}")
    print(f"{'='*60}")
    
    # Table Header
    print(f"{'Category':<25} | {'Probability':<11} | {'Status'}")
    print(f"{'-'*25}-+-{'-'*11}-+-{'-'*8}")
    
    present_classes = []
    
    # Check binary (NON-FUNCTIONAL) prediction
    prob_bin = float(y_pred_bin_test[i][0])
    if prob_bin >= threshold_binary[0]: 
        print(f"{bin_names[0]:<25} | {prob_bin:<11.4f} | Present")
        present_classes.append(bin_names[0])
    else:
        print(f"{bin_names[0]:<25} | {prob_bin:<11.4f} | -")

    # Check functional predictions
    for j in range(len(y_pred_spec_test[i])):
        prob_spec = float(y_pred_spec_test[i][j])
        if prob_spec >= threshold_functional[j]:
            print(f"{func_names[j]:<25} | {prob_spec:<11.4f} | Present")
            present_classes.append(func_names[j])
        else:
            print(f"{func_names[j]:<25} | {prob_spec:<11.4f} | -")
            
    # Print the summary of all present classes
    print(f"{'-'*60}")
    if present_classes:
        print(f"Detected Classes: {', '.join(present_classes)}")
    else:
        print(f"Detected Classes: None")
    print(f"{'='*60}")