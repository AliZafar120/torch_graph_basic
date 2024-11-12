import torch
import torch.fx
from torch.fx.passes.graph_drawer import FxGraphDrawer

# Function to draw the graph
def draw_graph(graph: torch.fx.Graph, filepath):
    g = FxGraphDrawer(graph, filepath)
    with open(filepath, "wb") as file:
        file.write(g.get_dot_graph().create_svg())

# Model definition
class MyModule(torch.nn.Module):
    def __init__(self, do_activation: bool = False):
        super().__init__()
        self.do_activation = do_activation
        self.linear = torch.nn.Linear(512, 512)

    def forward(self, x):
        x = self.linear(x)
        if self.do_activation:
            x = torch.relu(x)
        return x

# Function to find and print all activation nodes in the graph
def find_activation_nodes(graph: torch.fx.Graph):
    activation_nodes = []
    activations = {torch.relu, torch.sigmoid, torch.tanh}  # Known activations to check for

    for node in graph.nodes:
        # Check if the node's target is in the known activations
        if node.op == "call_function" and node.target in activations:
            activation_nodes.append(node)

    return activation_nodes

# Instantiate models
without_activation = MyModule(do_activation=False)
with_activation = MyModule(do_activation=True)

# Trace models
traced_without_activation = torch.fx.symbolic_trace(without_activation)
traced_with_activation = torch.fx.symbolic_trace(with_activation)

# Print and visualize graph with and without activation
print("Without Activation:")
print(traced_without_activation.code)
draw_graph(traced_without_activation, "./woactivation.svg")

print("\nWith Activation:")
print(traced_with_activation.code)
draw_graph(traced_with_activation, "./activation.svg")

# Find and print activation nodes
activation_nodes = find_activation_nodes(traced_with_activation.graph)
print("\nActivation Nodes:")
for node in activation_nodes:
    print(node)
