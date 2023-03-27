import argparse
from tqdm import tqdm
from collections import defaultdict

import torch
from torch import nn
from torch.utils.data import Dataset, DataLoader
from transformers import AutoModel, AutoTokenizer
from sklearn.model_selection import train_test_split

# TODO:
# - support several different training modes:
# PRIORITY 1
# --- controllable downsampling (what ratio of positive to negative examples do we train on?  should not affect validation)
# LOWER PRIORITY
# --- the number of changepoints predicted in the document so far
# --- in addition to downsampling, support different weights for 0 vs 1 examples in the loss
#       https://pytorch.org/docs/stable/generated/torch.nn.BCEWithLogitsLoss.html
# - it should have early stopping guided by average precision on the validation set

# utterance1<NA1 utterance2<NV2 utterance3<CP,NV2 utterance4 utterance5

class ChangepointNormsDataset(Dataset):
    def __init__(self, split, utterances_before, utterances_after):
        # PRIORITY 1 (once unblocked)
        # TODO: load the LDC dataset using the LDC loader functionality
        # TODO: load the associated UIUC predicted norms for the file_ids in the split
        self.split = split
        self.utterances_before = utterances_before
        self.utterances_after = utterances_after

        self.examples = [{ "file_id": "1",
              "timestamp": "12:05",
              "utterance": "hello waddup you are",
              "norms":"ADHERED:GREETING, VIOLATED:APOLOGY" ,
              "label": 1},{ "file_id": "2",
              "timestamp": "12:06",
              "utterance": "ciao kid no don't",
              "norms":"ADHERED:GREETING, VIOLATED:APOLOGY" ,
              "label": 1},{ "file_id": "3",
              "timestamp": "12:06",
              "utterance": "ciao kid no don't",
              "norms":"ADHERED:GREETING, VIOLATED:APOLOGY" ,
              "label": 1},{ "file_id": "4",
              "timestamp": "12:06",
              "utterance": "ciao kid no don't",
              "norms":"ADHERED:GREETING, VIOLATED:APOLOGY" ,
              "label": 1},{ "file_id": "5",
              "timestamp": "12:06",
              "utterance": "ciao kid no don't",
              "norms":"ADHERED:GREETING, VIOLATED:APOLOGY" ,
              "label": 1},{ "file_id": "6",
              "timestamp": "12:06",
              "utterance": "ciao kid no don't",
              "norms":"ADHERED:GREETING, VIOLATED:APOLOGY" ,
              "label": 1},{ "file_id": "7",
              "timestamp": "12:06",
              "utterance": "ciao kid no don't",
              "norms":"ADHERED:GREETING, VIOLATED:APOLOGY" ,
              "label": 0},{ "file_id": "8",
              "timestamp": "12:06",
              "utterance": "ciao kid no don't",
              "norms":"ADHERED:GREETING, VIOLATED:APOLOGY" ,
              "label": 0},{ "file_id": "9",
              "timestamp": "12:06",
              "utterance": "ciao kid no don't",
              "norms":"ADHERED:GREETING" ,
              "label": 0},{ "file_id": "10",
              "timestamp": "12:06",
              "utterance": "ciao kid no don't",
              "norms":"VIOLATED:APOLOGY" ,
              "label": 0}]
        
        self.train, self.valid = train_test_split(self.examples, test_size=0.07)


    def __len__(self):
        if self.split=='train':
            return len(self.train)
        else:
            return len(self.valid)

    def __getitem__(self, idx):
        # PRIORITY 1 (implement with dummy data so can test code)

        # assume that each individual example looks like this:
        # {
        #       "file_id": the source LDC file,
        #       "timestamp": , # the timestamp for the central utterance
        #       "utterance": (join the utterances before, the central utterance, and the utterances after), (string)
        #       "norms": (list of norm names that are adhered / violated across the utterances)
        #               "ADHERED:GREETING, VIOLATED:APOLOGY, etc" (string)
        #       "label": whether or not there's a changepoint (integer, 0 or 1)
        # }
        if self.split=='train':
            return self.train[idx]
        else:
            return self.valid[idx]


class ChangepointNormsClassifier(nn.Module):
    def __init__(self, encoder):
        super().__init__()
        self.model = AutoModel.from_pretrained(encoder)
        # TODO: make the complexity of the classifier configurable (eg, more layers, etc)
        self.classifier = nn.Linear(self.model.config.hidden_size, 1)

    def forward(self, inputs):
        outputs = self.model(**inputs)
        # pass CLS token representation through classifier
        logits = self.classifier(outputs.last_hidden_state[:, 0, :])
        return logits

def tokenize(batch, tokenizer, args):
    if args.include_utterance:
        return tokenizer(
            batch['norms'],
            batch['utterance'],
            return_tensors='pt',
            padding=True,
            truncation=True
        )
    else:
        return tokenizer(
            batch['norms'],
            return_tensors='pt',
            padding=True,
            truncation=True
        )


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--variant', type=str)
    parser.add_argument('--utterances-before', type=int, default=1)
    parser.add_argument('--utterances-after', type=int, default=1)
    parser.add_argument('--downsample', type=float, default=1.0)
    parser.add_argument('--include-utterance', action='store_true')
    parser.add_argument('--encoder', type=str, default='xlm-roberta-base')
    # TODO: support different regularization parameters like weight decay & dropout
    # TODO: support different learning rates for pretrained encoder and randomly initialized classificatin head (should be higher)
    parser.add_argument('--learning-rate', '--lr', type=float, default=1e-5)
    parser.add_argument('--batch-size', type=int, default=4)
    parser.add_argument('--epochs', type=int, default=10)
    parser.add_argument('--device', type=str, default='cpu')
    parser.add_argument('--checkpoint', action='store_true')
    parser.add_argument('--seed', type=int, default=42)
    args = parser.parse_args()
    print(args)

    # PRIORITY 1
    # TODO: create output directory for experiment results using variant name, store argument configs there as config.json

    # PRIORITY 1
    # TODO: seed everything using the random seed (look at PyTorch documents for reproducibility)

    if args.device == 'cuda':
        assert torch.cuda.is_available()
    device = torch.device(args.device)

    model = ChangepointNormsClassifier(args.encoder).to(device)
    tokenizer = AutoTokenizer.from_pretrained(args.encoder)
    # # TODO: support configurable weight decay, higher LR for classification head
    # # TODO: support learning rate scheduler (linear?)
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.learning_rate)

    dataset_args = [args.utterances_before, args.utterances_after]
    train, valid = ChangepointNormsDataset('train', *dataset_args), \
                   ChangepointNormsDataset('valid', *dataset_args)
    train_loader, valid_loader = DataLoader(train, batch_size=args.batch_size, shuffle=True), \
        DataLoader(valid, batch_size=2 * args.batch_size)

    # PRIORITY 1
    # TODO:
    # - keep track of lowest validation loss, best validation F1, best validation AP
    # - log it to a results CSV file in your output directory (what epoch, train metrics, valid metrics)

    #train loop
    for epoch in range(1):
        model = model.train()
        t_tot_loss, t_preds, t_labels = 0, [], []
        for t_batch in tqdm(train_loader, desc='validating epoch %s' % epoch, leave=False):
            optimizer.zero_grad()
            print(t_batch)
            t_tokenized = tokenize(
                t_batch, tokenizer, args
            ).to(device)
            t_logits = model(t_tokenized)
            crazy = t_batch['label']
            crazy = crazy.unsqueeze(1)
            t_loss = nn.BCEWithLogitsLoss()(
                t_logits.float(), crazy.float().to(device))
            
            t_loss.backward()

            optimizer.step()
            # TODO: step the scheduler

            t_tot_loss += t_loss.item()
            t_preds.extend(t_logits.detach().cpu().numpy())
            t_labels.extend(t_batch['label'].numpy())



       # print something (end of epoch, total training loss, precision, recall, f1, etc)
        print(t_loss)
        print(t_labels)
        model = model.eval()
        v_tot_loss, v_preds, v_labels = 0, defaultdict(list), defaultdict(list)
        for v_batch in tqdm(valid_loader, desc='validating epoch %s' % epoch, leave=False):
            print("------")
            print(v_batch)
            v_tokenized = tokenize(
                v_batch, tokenizer, args
            ).to(device)
            v_logits = model(v_tokenized)
            crazy = v_batch['label']
            crazy = crazy.unsqueeze(1)
            v_loss = nn.BCEWithLogitsLoss()(
                v_logits.float(), crazy.float().to(device)
            )

            v_tot_loss += v_loss.item()
            for v_file, v_pred, v_label in zip(
                v_batch['file_id'], v_logits.detach().cpu().numpy(), v_batch['label'].numpy()
            ):
                v_preds[v_file].append(v_pred)
                v_labels[v_file].append(v_label)

        # PRIORITY 1
        # TODO: compute validation metrics (precision, recall, f1, etc)
        # TODO: log new metrics to your results CSV
        # TODO: save model weights if we have a new best validation metric (if args.checkpoint)
        # TODO: figure out how best to calculate average precision (use helper function that Amith provides), blocked for now
        #     will require:
        #           - confidence scores for the predictions (can just use the logits post sigmoid)
        #           - require timestamps for the predictions
        #           - the timestamps for the LDC annotated changepoints