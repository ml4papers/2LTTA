import time
from copy import deepcopy
import torch
torch.cuda.memory_summary(device=None, abbreviated=False)
import torch.nn as nn
import torch.jit
from torch.cuda.amp import autocast,GradScaler
import math




class TLTTA(nn.Module):
    """Tent adapts a model by entropy minimization during testing.
    Once tented, a model adapts itself by updating on every forward.
    """
    def __init__(self, model, optimizer, device, args, steps=1, episodic=False):
        super().__init__()
        self.model = model
        self.optimizer = optimizer
        self.steps = steps
        assert steps > 0, "tent requires >= 1 step(s) to forward and update"
        self.episodic = episodic
        self.args = args
        self.scaler = GradScaler()
        self.device = device

    def forward(self, x, adapt_flag):
        for _ in range(self.steps):
            if adapt_flag:
                outputs, loss = forward_and_adapt(x, self.model, self.optimizer, self.args, self.scaler)
            else:
                outputs, _ = self.model.module.forward_eval(a=x[0], v=x[1], mode=self.args.testmode)
                loss = (0, 0)
                outputs = (outputs, outputs)

        return outputs, loss


@torch.jit.script
def softmax_entropy(x: torch.Tensor) -> torch.Tensor:
    """Entropy of softmax distribution from logits."""
    return -(x.softmax(1) * x.log_softmax(1)).sum(1)


@torch.enable_grad()  # ensure grads in possible no grad context for testing
def forward_and_adapt(x, model, optimizer, args, scaler):
    """Forward and adapt model on batch of data.
    Compute loss function (Eq. 7) based on the model prediction, take gradients, and update params.
    """
    with autocast():
        # forward
        outputs, _ = model.module.forward_eval(a=x[0], v=x[1], mode=args.testmode)
        v_outputs, _ = model.module.forward(a=x[0], v=x[1], mode=args.testmode)
    # adapt
    """
    Shannon Entropy with weighting function for intra-modality alignment
    """
    entropys = softmax_entropy(v_outputs)
    coeff = 1 / (torch.exp(entropys.clone().detach() - 0.4*math.log(args.n_class)))
    entropys = entropys.mul(coeff)
    loss = entropys.mean()
    """
    diversity-promoting loss for reducing cross-modality bias
    """
    msoftmax = nn.Softmax(dim=1)(outputs).mean(dim=0) # divergence
    loss += 0.5*torch.sum(msoftmax * torch.log(msoftmax))
    
    optimizer.zero_grad()

    scaler.scale(loss).backward()
    scaler.step(optimizer)
    scaler.update()

    with torch.no_grad():
        with autocast():
        # forward
            outputs2, _ = model.module.forward_eval(a=x[0], v=x[1], mode=args.testmode)

    return (outputs, outputs2), loss


def collect_params(model):
    """
    Walk the model's modules and collect qkv parameters of the fusion attn module.
    Return the parameters and their names.
    Note: other choices of parameterization are possible!
    """
    params = []
    names = []
    for nm, m in model.named_modules():
        # if nm == 'module.blocks_u.0.attn.qkv' or nm == 'module.blocks_u.0.attn.q' or nm == 'module.blocks_u.0.attn.k' or nm == 'module.blocks_u.0.attn.v':
        #     for np, p in m.named_parameters():
        #         if np in ['weight', 'bias']:
        #             params.append(p)
        #             names.append(f"{nm}.{np}")
        if isinstance(m, (nn.BatchNorm2d, nn.GroupNorm, nn.LayerNorm, nn.Conv2d, nn.Linear)):
            for np, p in m.named_parameters():
                if np in ['weight', 'bias']:  # weight is scale, bias is shift
                    params.append(p)
                    names.append(f"{nm}.{np}")
    return params, names


def copy_model_and_optimizer(model, optimizer):
    """Copy the model and optimizer states for resetting after adaptation."""
    model_state = deepcopy(model.state_dict())
    optimizer_state = deepcopy(optimizer.state_dict())
    return model_state, optimizer_state


def load_model_and_optimizer(model, optimizer, model_state, optimizer_state):
    """Restore the model and optimizer states from copies."""
    model.load_state_dict(model_state, strict=True)
    optimizer.load_state_dict(optimizer_state)

def configure_model(model):
    """Configure model for use with Renata."""
    # train mode, but no grad
    # print(model)
    """
    fine-tune the visual modality
    """
    model.train()
    model.requires_grad_(False)
    model.module.blocks_v[0].attn.proj.weight.requires_grad = True
    model.module.blocks_v[1].attn.proj.weight.requires_grad = True
    model.module.blocks_v[2].attn.proj.weight.requires_grad = True
    model.module.blocks_v[3].attn.proj.weight.requires_grad = True
    model.module.blocks_v[4].attn.proj.weight.requires_grad = True
    for nm in model.module.blocks_v:
        for m in nm.modules():
            if isinstance(m, (nn.BatchNorm2d, nn.GroupNorm, nn.LayerNorm)):
                m.requires_grad_(True)
                # force use of batch stats in train and eval modes
                m.track_running_stats = False
                m.running_mean = None
                m.running_var = None
    """
    fine-tune the audio modality
    """
    # model.module.blocks_a[0].attn.proj.weight.requires_grad = True
    # model.module.blocks_a[1].attn.proj.weight.requires_grad = True
    # model.module.blocks_a[2].attn.proj.weight.requires_grad = True
    # model.module.blocks_a[3].attn.proj.weight.requires_grad = True
    # model.module.blocks_a[4].attn.proj.weight.requires_grad = True
    for nm in model.module.blocks_a:
        for m in nm.modules():
            if isinstance(m, (nn.BatchNorm2d, nn.GroupNorm, nn.LayerNorm)):
                m.requires_grad_(True)
                # force use of batch stats in train and eval modes
                m.track_running_stats = False
                m.running_mean = None
                m.running_var = None
    """
    fine-tune the fusion module
    """
    model.module.blocks_u[0].attn.proj.weight.requires_grad = True
    for nm, m in model.named_modules():
        if nm == 'module.blocks_u.0.attn.q' or nm == 'module.blocks_u.0.attn.k' or nm == 'module.blocks_u.0.attn.v':
            m.requires_grad_(True)
    for nm in model.module.blocks_u:
        for m in nm.modules():
            if isinstance(m, (nn.BatchNorm2d, nn.GroupNorm, nn.LayerNorm)):
                m.requires_grad_(True)
                # force use of batch stats in train and eval modes
                m.track_running_stats = False
                m.running_mean = None
                m.running_var = None
    return model
