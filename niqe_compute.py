import torch
from torch import nn

class NIQE(nn.Module):
    
    def __init__(self, p=None, stabilitiy_scale=0.001, args=None):
        super(NIQE, self).__init__()
        self.mu_r = None
        self.sigma_r = None
        self.I_stability = None
        self.stabilitiy_scale = stabilitiy_scale
        self.args = args
        self.compute_pristine(p.unsqueeze(0))       ## [1, num_patches, feature_length]
            
    
    def forward(self, x):
        mu_t = torch.mean(x, dim=-2, keepdim=True)
        sigma_t = self.batch_covariance(x, mu_t)
        
        mean_diff = self.mu_r - mu_t
        
        cov_sum = ((self.sigma_r + sigma_t) / 2) + self.I_stability       ## Identity matrix added for getting a positive definite matrix
        cov_sum_inv = torch.linalg.inv(cov_sum)
        
        # print(torch.dist((self.sigma_r + sigma_t) / 2, torch.tensor(0)))
        # print(torch.dist(cov_sum, torch.tensor(0)))
        # print(torch.dist((self.sigma_r + sigma_t) / 2, cov_sum))
        # print(torch.dist(torch.matmul(cov_sum, cov_sum_inv), torch.eye(cov_sum.size(-1)).unsqueeze(0).to(x.device)))
        # exit()

        fit = torch.matmul(torch.matmul(mean_diff, cov_sum_inv), torch.transpose(mean_diff, -2, -1))

        return torch.sqrt(fit).squeeze()
    
    def compute_pristine(self, p):
        self.mu_r = torch.mean(p, dim=-2, keepdim=True)
        self.sigma_r = self.batch_covariance(p, self.mu_r)
        self.I_stability = self.stabilitiy_scale * torch.eye(p.size(-1), device=p.device).unsqueeze(0)
        
    def batch_covariance(self, tensor, mu, bias=False):
        tensor = tensor - mu
        factor = 1 / (tensor.shape[-2] - int(not bool(bias)))
        return factor * tensor.transpose(-1, -2) @ tensor.conj()

if __name__ == '__main__':

    p = torch.randn((422,2048)).cuda()      ## corpus of pristine patches (422 features of length 2048)
    x = torch.randn((64,48,2048)).cuda()    ## 64 images in a batch each patchified into 48 patches and passed through feature encoder

    niqe_model = NIQE(p).cuda() 
    
    score = niqe_model(x)
    print(score)