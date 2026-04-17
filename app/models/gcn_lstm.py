import torch
import torch.nn as nn

class Gcnlstm(nn.Module):
    def __init__(self, seq, n_fea, adj_dense, node=307, gcn_out=32, gcn_layers=1, lstm_hidden_dim=256, lstm_layers=2,
                 hidden_dim=32):
        super(Gcnlstm, self).__init__()

        self.nodes = node
        self.seq_len = seq
        self.num_feat = n_fea
        self.lstm_hidden_dim = lstm_hidden_dim
        self.lstm_layers = lstm_layers
        self.gcn_out = gcn_out
        self.gcn_layers = gcn_layers
        self.hidden_dim = hidden_dim

        # Initialize GCN layers
        self.gcn_layers_list = nn.ModuleList()
        for i in range(gcn_layers):
            in_dim = seq * n_fea if i == 0 else gcn_out
            self.gcn_layers_list.append(nn.Linear(in_dim, gcn_out))
        self.act = nn.ReLU()

        self.encoder = nn.Conv2d(self.nodes, self.nodes, (1, n_fea))
        # Initialize LSTM layer
        self.lstm = nn.LSTM(input_size=n_fea, hidden_size=self.lstm_hidden_dim, num_layers=self.lstm_layers,
                            batch_first=True)
        self.decoder = nn.Linear(seq + self.gcn_out + self.lstm_hidden_dim, 1)
        # Calculate A_delta matrix
        deg = torch.sum(adj_dense, dim=0)
        deg = torch.diag(deg)
        deg_delta = torch.linalg.inv(torch.sqrt(deg))
        a_delta = torch.matmul(torch.matmul(deg_delta, adj_dense), deg_delta)
        self.A = a_delta

    def forward(self, occ, extra_feat=None):
        x = occ.clone().unsqueeze(-1)  # Add feature dimension
        # [修改点]：同时排除 Python 的 None 对象和字符串 'None'
        if extra_feat is not None and extra_feat != 'None':
            x = torch.cat([occ.unsqueeze(-1), extra_feat], dim=-1)
        assert x.shape[-1] == self.num_feat, f"Number of features ({x.shape[-1]}) does not match n_fea ({self.num_feat})."
        # Create a copy of occ to avoid modifying the original data
        x = self.encoder(x)
        x_copy = x.clone().unsqueeze(-1)
        batch_size = x_copy.size(0)

        # Process all timesteps with LSTM
        x_lstm = x_copy.view(batch_size * self.nodes, self.seq_len, self.num_feat)
        lstm_out, _ = self.lstm(x_lstm)
        lstm_out = lstm_out.view(batch_size, self.nodes, self.seq_len,
                                 self.lstm_hidden_dim)  # Shape: (batch, node, seq, lstm_hidden_dim)

        # Process with GCN layers
        gcn_out = x.view(batch_size,self.nodes,self.seq_len * self.num_feat)
        for gcn_layer in self.gcn_layers_list:
            gcn_out = gcn_layer(gcn_out)
            gcn_out = torch.matmul(self.A, gcn_out)
            gcn_out = self.act(gcn_out)

        # Concatenate LSTM and GCN outputs
        combined_out = torch.cat((occ,lstm_out[:,:,-1,:], gcn_out), dim=-1)  # Shape: (batch, node, seq, lstm_hidden_dim + gcn_out)

        x = self.decoder(combined_out)
        x = torch.squeeze(x)
        return x
