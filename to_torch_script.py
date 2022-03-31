import torch
from torch.utils.mobile_optimizer import optimize_for_mobile

from model.MyMoblenet3 import MyMobileNetv2

model_s = MyMobileNetv2()
model_s.load_state_dict(torch.load('./weights/spatial_weights/spacial_model.th', map_location=torch.device('cpu')))
script_module = torch.jit.script(model_s)
script_module._save_for_lite_interpreter("./model_script_s.ptl")
optimized_script_model = optimize_for_mobile(script_module)
optimized_script_model._save_for_lite_interpreter("./model_script_optimize_s.ptl")
