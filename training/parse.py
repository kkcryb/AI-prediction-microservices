import argparse


def parse_args():
    parser = argparse.ArgumentParser(description="Go Spatio-temporal EV Charging Demand Prediction!")

    parser.add_argument('--device', type=int, default=0, help="CUDA.")
    parser.add_argument('--seed', type=int, default=42, help="Random seed.")
    parser.add_argument('--seq_len', type=int, default=12, help="The sequence length of input data.")
    parser.add_argument('--bs', type=int, default=32, help="The batch size of fine-tuning.")
    parser.add_argument('--epoch', type=int, default=20, help="The max epoch of the training process.")
    parser.add_argument('--total_fold', type=int, default=6, help="The fold used for spliting data in cross-validation")

    parser.add_argument('--model', type=str, default='GCN', help="The used model")
    parser.add_argument('--pred_len', type=int, default=1, help="The length of prediction interval.")
    parser.add_argument('--add_feat', type=str, default='None', help="Whether to use additional features for prediction")
    parser.add_argument('--fold', type=int, default=0, help="The current fold number for training data")
    parser.add_argument('--pred_type', type=str, default='region', help="Prediction at node or regional level")
    parser.add_argument('--feat', type=str, default='occ', help="Which feature to use for prediction")
    parser.add_argument('--is_train', action='store_true', default=True)

    # parser.add_argument('--is_train', action='store_true', default=False)

    #新增stgmamba
    parser.add_argument('--mamba_features', type=int, default=300,help="Hidden features for STG-Mamba (d_model/features)")
    parser.add_argument('--mamba_layers', type=int, default=4, help="Number of Mamba layers")
    parser.add_argument('--mamba_K', type=int, default=3, help="K value for KFGN in STG-Mamba")

    #新增st-llm
    parser.add_argument('--llm_layers', type=int, default=3, help='GPT2 layers to use')
    parser.add_argument('--llm_U', type=int, default=1, help='Number of unfrozen layers in ST-LLM')

    parser.add_argument('--dropout', type=float, default=0.3, help="Dropout rate.")
    parser.add_argument('--blocks', type=int, default=2, help="Number of blocks for TCN/STGCN based models.")
    parser.add_argument('--layers', type=int, default=2, help="Number of layers per block.")
    parser.add_argument('--rnn_units', type=int, default=32, help="Hidden units for PDG2Seq RNN")

    return parser.parse_args()
