import torch
import torch.fx
from torch.fx.passes.graph_drawer import FxGraphDrawer
def draw_graph(graph:torch.fx.graph,filepath):
    g = FxGraphDrawer(graph, filepath)
    with open(filepath, "wb") as file:
        file.write(g.get_dot_graph().create_svg())
class MyModule(torch.nn.Module):
    def __init__(self, do_activation : bool = False):
        super().__init__()
        self.do_activation = do_activation
        self.linear = torch.nn.Linear(512, 512)

    def forward(self, x):
        x = self.linear(x)
        # This if-statement is so-called static control flow.
        # Its condition does not depend on any input values
        if self.do_activation:
            x = torch.relu(x)
        return x

without_activation = MyModule(do_activation=False)
with_activation = MyModule(do_activation=True)

traced_without_activation = torch.fx.symbolic_trace(without_activation)
print(traced_without_activation.code)
draw_graph(traced_without_activation,"./woactivation.svg")


traced_with_activation = torch.fx.symbolic_trace(with_activation)
print(traced_with_activation.code)
draw_graph(traced_with_activation,"./activation.svg")
