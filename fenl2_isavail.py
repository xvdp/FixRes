# Copyright (c) Facebook, Inc. and its affiliates.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of https://github.com/facebookresearch/FixRes
#
# from torchbench.image_classification import ImageNet
import torchvision.transforms as transforms
import PIL
import urllib.request
from timm import create_model
import torchvision.transforms.functional as F
from torchvision import transforms
from PIL import Image
import torch 

def get_transforms_v2(test_size=224, backbone='EfficientNetL2',crop_ptc=1.0,mean_type=False):
    mean, std = [0.485, 0.456, 0.406], [0.229, 0.224, 0.225]
    Rs_size=int((256 / 224) * test_size)
    itrpl=2
    if backbone is not None and backbone in ['EfficientNetL2']:
        Rs_size=int((1.0/crop_ptc)*test_size)
        itrpl=3
        if mean_type:
            mean, std = [0.5, 0.5, 0.5], [0.5, 0.5, 0.5]
    transformations = {}
    transformations['val_test'] = transforms.Compose(
        [transforms.Resize(Rs_size, interpolation=itrpl),
         transforms.CenterCrop(test_size),
         transforms.ToTensor(),
         transforms.Normalize(mean, std)])

    return transformations



def bench(model_name="FixEfficientNet_L2", data_root="/home/z/data/ImageNet/ILSVRC2012"):
    transforms_list = ['torch', 'full']
    test_sizes=[320,384,420,472,472,576,680,632,600,320,384,420,472,512,576,576,632,800]
    mean_types=[False,False,False,False,False,False,False,False,False,True,True,True,True,True,True,True,True,True]
    model_names =['fixEfficientNet_b0_ns','fixEfficientNet_b1_ns','fixEfficientNet_b2_ns','fixEfficientNet_b3_ns','fixEfficientNet_b4_ns','fixEfficientNet_b5_ns','fixEfficientNet_b6_ns','fixEfficientNet_b7_ns','FixEfficientNet_L2','fixEfficientNet_b0','fixEfficientNet_b1','fixEfficientNet_b2','fixEfficientNet_b3','fixEfficientNet_b4','fixEfficientNet_b5','fixEfficientNet_b6','fixEfficientNet_b7','fixEfficientNet_b8']
    batch_sizes=[64,32,32,32,32,16,16,16,16,16,64,32,32,32,32,32,16,16]
    architecture_names=['tf_efficientnet_b0_ns','tf_efficientnet_b1_ns','tf_efficientnet_b2_ns','tf_efficientnet_b3_ns','tf_efficientnet_b4_ns','tf_efficientnet_b5_ns','tf_efficientnet_b6_ns','tf_efficientnet_b7_ns','tf_efficientnet_l2_ns_475','tf_efficientnet_b0_ap','tf_efficientnet_b1_ap','tf_efficientnet_b2_ap','tf_efficientnet_b3_ap','tf_efficientnet_b4_ap','tf_efficientnet_b5_ap','tf_efficientnet_b6_ap','tf_efficientnet_b7_ap','tf_efficientnet_b8_ap']

    print(len(test_sizes))
    print(len(mean_types))
    print(len(model_names))
    print(len(batch_sizes))
    print(len(architecture_names))
    print(model_names.index(model_name))
    idx = model_names.index(model_name)

    test_size = test_sizes[idx]
    mean_type = mean_types[idx]
    batch_size = batch_sizes[idx]
    architecture_name = architecture_names[idx]

    print(architecture_name, model_name, test_size, mean_type, batch_size)

    input_transform = get_transforms_v2(test_size=test_sizes[idx], backbone='EfficientNetL2',mean_type=mean_types[idx],crop_ptc=1.0)['val_test']
    urllib.request.urlretrieve('https://dl.fbaipublicfiles.com/FixRes_data/FixRes_Pretrained_Models/FixEfficientNet/'+str(model_names[idx])+'.pth', str(model_names[idx])+'.pth')


    pretrained_dict=torch.load(str(model_name)+'.pth',map_location='cpu')
    model = create_model(architecture_name, pretrained=False)
    model_dict = model.state_dict()
    count=0
    count2=0
    for k in model_dict.keys():
        count=count+1.0
        if(k in pretrained_dict.keys()):
            count2=count2+1.0
            model_dict[k]=pretrained_dict.get(k)
    model.load_state_dict(model_dict)
    print("load "+str(count2*100/count)+" %")
    model.eval()
    model.require_grad=False
    
    # Run the benchmark
    
    ImageNet.benchmark(
        model=model,
        model_description='FixRes',
        paper_model_name=model_name,
        data_root=data_root,
        paper_arxiv_id='1906.06423',
        input_transform=input_transform,
        batch_size=batch_size,
        num_gpu=1
    )
    torch.cuda.empty_cache()


if __name__ == "__main__":
    bench()

# for test_size,mean_type,model_name,batch_size,architecture_name in zip(test_sizes,mean_types,model_names,batch_sizes,architecture_names):
#     input_transform = get_transforms_v2(test_size=test_size, backbone='EfficientNetL2',mean_type=mean_type,crop_ptc=1.0)['val_test']
#     urllib.request.urlretrieve('https://dl.fbaipublicfiles.com/FixRes_data/FixRes_Pretrained_Models/FixEfficientNet/'+str(model_name)+'.pth', str(model_name)+'.pth')

       


#     pretrained_dict=torch.load(str(model_name)+'.pth',map_location='cpu')
#     model = create_model(architecture_name, pretrained=False)
#     model_dict = model.state_dict()
#     count=0
#     count2=0
#     for k in model_dict.keys():
#         count=count+1.0
#         if(k in pretrained_dict.keys()):
#             count2=count2+1.0
#             model_dict[k]=pretrained_dict.get(k)
#     model.load_state_dict(model_dict)
#     print("load "+str(count2*100/count)+" %")
#     model.eval()
#     model.require_grad=False
    
#     # Run the benchmark
    
#     ImageNet.benchmark(
#         model=model,
#         model_description='FixRes',
#         paper_model_name=model_name,
#         paper_arxiv_id='1906.06423',
#         input_transform=input_transform,
#         batch_size=batch_size,
#         num_gpu=1
#     )
#     torch.cuda.empty_cache()

