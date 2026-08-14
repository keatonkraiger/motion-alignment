import torch
import torch.nn as nn
import torch.nn.functional as F

class CrossCorrLoss(nn.Module):
    def forward(self, A, B):
        """Cross Correlation Loss

        Parameters
        ----------
        A : torch.Tensor
            Embedding vectors from augmentation A. Shape (B, D).
        B : torch.Tensor
            Embedding vectors from augmentation B. Shape (B, D).
        
        Returns
        -------
        troch.Tensor
            Loss value.
        """
        Anorm = F.normalize(A, p=2) # (B, D)
        Bnorm = F.normalize(B, p=2) # (B, D)

        Cmat = Anorm @ Bnorm.T # (B, B)
        cdiff = (Cmat - torch.eye(Cmat.shape[0], device=A.device)).pow(2) # (B, B)
        lambd = 1 / (cdiff.shape[0] - 1)
        sumdiag = cdiff.diag().sum()
        sumoffdiag = cdiff.sum() - sumdiag
        loss = sumdiag + lambd * sumoffdiag
        return loss
