import torch
import torch.fx
import torch.nn as nn

class MyModule(nn.Module):
    def __init__(self):
        super().__init__()
        self.linear1 = nn.Linear(512, 512)
        self.linear2 = nn.Linear(512, 512)

    def forward(self, x):
        x = self.linear1(x)
        x = torch.relu(x)
        x = self.linear2(x)
        return x

# Trace the model to create a torch.fx.Graph
model = MyModule()
traced_model = torch.fx.symbolic_trace(model)

# Mark requires_grad for the model parameters
for param in model.parameters():
    param.requires_grad = True

# Identify nodes that require grad
def find_requires_grad_nodes(graph: torch.fx.GraphModule, model: nn.Module):
    grad_nodes = []
    for node in graph.graph.nodes:
        # Check if the node is an operation that could involve gradients
        if node.op == "call_module":
            # Get the module corresponding to this node
            module = dict(model.named_modules()).get(node.target)
            if module and any(param.requires_grad for param in module.parameters()):
                grad_nodes.append(node)

        elif node.op == "call_function":
            # Check if the function's output would require grad using a sample tensor
            input_example = torch.randn(1, 512, requires_grad=True)
            try:
                output_example = node.target(input_example)
                if isinstance(output_example, torch.Tensor) and output_example.requires_grad:
                    grad_nodes.append(node)
            except Exception:
                pass  # Skip nodes where testing fails

    return grad_nodes

# Find nodes that require grad
grad_nodes = find_requires_grad_nodes(traced_model, model)
print("\nNodes that require gradients:")
for node in grad_nodes:
    print(node)
