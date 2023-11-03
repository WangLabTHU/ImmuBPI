import torch
import torch.nn as nn

class LogitNLLLoss(nn.Module):
    def __init__(self, **kwargs):
        super().__init__()
        if kwargs.get('weight', None):
            if isinstance(kwargs['weight'], list):
                kwargs['weight'] = torch.tensor(kwargs['weight'])
        self.nll_loss = nn.NLLLoss(**kwargs)
    
    def forward(self, input: torch.Tensor, target: torch.Tensor, **kwargs) -> torch.Tensor:
        if 'weight' not in kwargs:
            log_prob = input.log_softmax(dim=-1)
            return self.nll_loss(log_prob, target)
        else:
            log_prob = input.log_softmax(dim=-1)
            terms = - kwargs['weight'] * log_prob.gather(dim=-1, index=target.unsqueeze(1)).squeeze(1)
            return torch.mean(terms)

if __name__ == "__main__":
    B, C = 2, 3
    logit = torch.zeros(B, C)
    target = torch.tensor([0, 1])
    #TODO bug or feature?
    weight = torch.tensor([0.2] + [1.] * (C - 1))
    #weight = [0.2] + [1.] * (C - 1)
    criterion = LogitNLLLoss(weight=weight)
    print(criterion(logit, target)) # (w[0] / w[0] + w[1]) * log C + (w[1] / w[0] + w[1]) * log C
    print(torch.log(torch.tensor(C)))
