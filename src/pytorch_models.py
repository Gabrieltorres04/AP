import torch
import torch.nn as nn

def select_device():
    if torch.cuda.is_available():
        return torch.device("cuda")
    elif torch.backends.mps.is_available():
        return torch.device("mps")
    else:        
        return torch.device("cpu")    

class MLPClassifier(nn.Module):
    def __init__(self, input_size, hidden_size, num_classes, dropout=0.2):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(input_size, hidden_size),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_size, num_classes)
        )

    def forward(self, x):
        return self.net(x)
    
class RNNClassifier(nn.Module):
    def __init__(
        self,
        vocab_size,
        embed_dim,
        hidden_dim,
        num_layers,
        num_classes,
        rnn_type="gru",
        padding_idx=0,
        dropout=0.2,
    ):
        super().__init__()
        self.rnn_type = rnn_type.lower()
        self.embedding = nn.Embedding(vocab_size, embed_dim, padding_idx=padding_idx)

        rnn_dropout = dropout if num_layers > 1 else 0.0
        if self.rnn_type == "rnn":
            self.rnn = nn.RNN(embed_dim, hidden_dim, num_layers=num_layers, batch_first=True, dropout=rnn_dropout)
        elif self.rnn_type == "lstm":
            self.rnn = nn.LSTM(embed_dim, hidden_dim, num_layers=num_layers, batch_first=True, dropout=rnn_dropout)
        elif self.rnn_type == "gru":
            self.rnn = nn.GRU(embed_dim, hidden_dim, num_layers=num_layers, batch_first=True, dropout=rnn_dropout)
        else:
            raise ValueError("rnn_type must be one of: rnn, lstm, gru")

        self.dropout = nn.Dropout(dropout)
        self.fc = nn.Linear(hidden_dim, num_classes)

    def forward(self, x):
        x = self.embedding(x)
        if self.rnn_type == "lstm":
            _, (h_n, _) = self.rnn(x)
        else:
            _, h_n = self.rnn(x)
        last_hidden = h_n[-1]
        return self.fc(self.dropout(last_hidden))


def build_model(model_name, config):
    name = model_name.lower()

    if name == "mlp_tfidf":
        return MLPClassifier(
            input_size=config["input_size"],
            hidden_size=config.get("hidden_size", 128),
            num_classes=config["num_classes"],
            dropout=config.get("dropout", 0.2),
        )

    if name in ["rnn", "lstm", "gru"]:
        return RNNClassifier(
            vocab_size=config["vocab_size"],
            embed_dim=config.get("embed_dim", 128),
            hidden_dim=config.get("hidden_dim", 128),
            num_layers=config.get("num_layers", 1),
            num_classes=config["num_classes"],
            rnn_type=name,
            padding_idx=config.get("padding_idx", 0),
            dropout=config.get("dropout", 0.2),
        )

    raise ValueError("Unknown model_name")