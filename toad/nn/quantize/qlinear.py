import torch
import torch.nn.functional as F
# from quanto import QModuleMixin, register_qmodule, quantize_activation



class QLinear(torch.nn.Linear):
    @classmethod
    def qcreate(
        cls, module, weights = None, activations = None, optimizer = None
    ):
        return cls(
            module.in_features,
            module.out_features,
            module.bias is not None,
            dtype=module.weight.dtype,
            device=module.weight.device,
        )
    

    def quantize(self):
        qweight = torch.randint(256, size=self.weight.shape, dtype=torch.uint8)
        self.qweight = qweight
        return self.qweight
    

    def freeze(self):
        self.weight = torch.nn.Parameter(self.pack_weight(self.qweight))

    def pack_weight(self, weight):
        return weight.view(torch.float32)

    def unpack_weight(self, weight):
        return weight.view(torch.uint8)
    
    def forward(self, input: torch.Tensor) -> torch.Tensor:
        qweight = self.unpack_weight(self.weight).to(torch.float32)
        print("forward qweight shape", qweight.shape)
        return F.linear(input, qweight, bias=self.bias)
    
