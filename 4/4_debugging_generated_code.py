import torch
import torch.fx
import torchvision.models as models
from torch.fx.passes.graph_drawer import FxGraphDrawer
def draw_graph(graph:torch.fx.graph,filepath):
    g = FxGraphDrawer(graph, filepath)
    with open(filepath, "wb") as file:
        file.write(g.get_dot_graph().create_svg())
def my_pass(inp: torch.nn.Module, tracer_class : type = torch.fx.Tracer) -> torch.nn.Module:
    graph = tracer_class().trace(inp)
    # Transformation logic here
    # <...>

    # Return new Module
    return torch.fx.GraphModule(inp, graph)

my_module = models.resnet18()
my_module_transformed = my_pass(my_module)

input_value = torch.randn(5, 3, 224, 224)

# When this line is executed at runtime, we will be dropped into an
# interactive `pdb` prompt. We can use the `step` or `s` command to
# step into the execution of the next line
import pdb; pdb.set_trace()
draw_graph(my_module_transformed,"./before.jog")
my_module_transformed(input_value)
draw_graph(my_module_transformed,"./after.jog")