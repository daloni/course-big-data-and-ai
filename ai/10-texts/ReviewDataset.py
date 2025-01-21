import torch
from torch.utils.data import Dataset

class ReviewDataset(Dataset):
    def __init__(self, reviews, labels, vocab, max_len=100):
        self.reviews = reviews
        self.labels = labels
        self.vocab = vocab
        self.max_len = max_len
    
    def __len__(self):
        return len(self.reviews)
    
    def __getitem__(self, idx):
        review = self.reviews[idx]
        label = self.labels[idx]

        review_indices = [self.vocab.get(word, self.vocab['<UNK>']) for word in review.split()]
        review_indices = review_indices[:self.max_len]
        review_indices = torch.tensor(review_indices, dtype=torch.long)

        return review_indices, label
