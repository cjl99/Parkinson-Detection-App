import os

import torch
from PIL import Image
from torch.autograd import Variable
from torch.utils.mobile_optimizer import optimize_for_mobile
from torchvision import transforms

from model.VGGV import VGGV


def sumfigure(img_path):
    transform = transforms.Compose([
        transforms.CenterCrop((648, 570)),
        transforms.Resize((216, 190)),
        transforms.ToTensor()]
    )

    img = Image.open(img_path)
    img = transform(img)
    print(img.shape)
    return img


def sum_data(sourcepath, picfile):
    pics = os.listdir(sourcepath + picfile)
    pics.sort()
    i = 0
    for pic in pics:
        i += 1
        if i % 2 == 0:
            continue
        if i == 1:
            videoseg = sumfigure(sourcepath + picfile + '/' + pic)
        elif i == 3:
            videoseg = torch.cat((videoseg[None], sumfigure(sourcepath + picfile + '/' + pic)[None]), 0)
        else:
            videoseg = torch.cat((videoseg, sumfigure(sourcepath + picfile + '/' + pic)[None]), 0)
        if i >= 64:
            break
    return videoseg


model_s = VGGV()
model_s.load_state_dict(torch.load('./weights/spatial_weights/spacial_model.th', map_location=torch.device('cpu')))
sourcepath_spatial = './dataset/crop_face/Testset/'  # 空间流输入数据
picfile = 'IMG_0462Seg_300'
videoseg_s = sum_data(sourcepath_spatial, picfile).permute(1, 0, 2, 3)

# x = Variable(torch.unsqueeze(videoseg_s, dim=0).float(), requires_grad=False)
# model_s(x)

# traced_script_module = torch.jit.trace(model_s, x)
# traced_script_module._save_for_lite_interpreter("./model_trace_s.ptl")
# optimize_trace_model = optimize_for_mobile(traced_script_module)
# optimize_trace_model._save_for_lite_interpreter("./model_trace_optimize_s.ptl")
#
script_module = torch.jit.script(model_s)
script_module._save_for_lite_interpreter("./model_script_s.ptl")
optimized_script_model = optimize_for_mobile(script_module)
optimized_script_model._save_for_lite_interpreter("./model_script_optimize_s.ptl")
