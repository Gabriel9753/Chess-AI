from torch.utils.data import Dataset, DataLoader
import re
import numpy as np
import torch

class ChessDataset(Dataset):
    def __init__(self, df):
        self.fens = list(df["FEN"])
        self.evals = list(df["Normalized Evaluation"])
        self._len = len(self.evals)
        
    def fen_to_bit_vector(self, fen):
        parts = re.split(" ", fen)
        piece_placement = re.split("/", parts[0])
        active_color = parts[1]
        castling_rights = parts[2]
        en_passant = parts[3]
        bit_vector = np.zeros((13, 8, 8), dtype=np.uint8)
        # piece to layer structure taken from reference [1]
        piece_to_layer = {'R': 1,'N': 2,'B': 3,'Q': 4,'K': 5,'P': 6,'p': 7,'k': 8,'q': 9,'b': 10,'n': 11,'r': 12}

        castling = {'K': (7,7),'Q': (7,0),'k': (0,7),'q': (0,0),}
        for r, row in enumerate(piece_placement):
            c = 0
            for piece in row:
                if piece in piece_to_layer:
                    bit_vector[piece_to_layer[piece], r, c] = 1
                    c += 1
                else:
                    c += int(piece)
        if en_passant != '-':
            bit_vector[0, ord(en_passant[0]) - ord('a'), int(en_passant[1]) - 1] = 1
        if castling_rights != '-':
            for char in castling_rights:
                bit_vector[0, castling[char][0], castling[char][1]] = 1
        if active_color == 'w':
            bit_vector[0, 7, 4] = 1
        else:
            bit_vector[0, 0, 4] = 1
        return bit_vector

    def __len__(self):
        return self._len

    def __getitem__(self, index):
        fen = torch.from_numpy(np.array([self.fen_to_bit_vector(self.fens[index])], dtype=np.uint8))
        eval = torch.Tensor([self.evals[index]])
        return fen, eval
