import torch
import torch.nn as nn

class Test(nn.Module):
    def __init__(self) -> None:
        super(Test, self).__init__()
        self.dropout = nn.Dropout()
    def forward(self, x):
        return self.dropout(x)

test = Test()
test.eval()
print(test(torch.randn((5))))
test.train()
print(test(torch.randn((5))))

test.eval()
traced = torch.jit.script(test)

traced.eval()
print(traced(torch.randn((5))))
traced.train()
print(traced(torch.randn((5))))