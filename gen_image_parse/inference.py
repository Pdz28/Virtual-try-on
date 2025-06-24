from PIL import Image
import torch
import timeit
import numpy as np
import cv2
import os
import argparse
import sys
from datetime import datetime
from collections import OrderedDict
sys.path.append('./')
# PyTorch includes
from torch.autograd import Variable
from torchvision import transforms


# Custom includes
from networks_f import deeplab_xception_transfer, graph
from dataloaders import custom_transforms as tr

#

label_colours = [(0,0,0)
                , (128,0,0), (255,0,0), (0,85,0), (170,0,51), (255,85,0), (0,0,85), (0,119,221), (85,85,0), (0,85,85), (85,51,0), (52,86,128), (0,128,0)
                , (0,0,255), (51,170,221), (0,255,255), (85,255,170), (170,255,85), (255,255,0), (255,170,0)]


def flip(x, dim):
    indices = [slice(None)] * x.dim()
    indices[dim] = torch.arange(x.size(dim) - 1, -1, -1,
                                dtype=torch.long, device=x.device)
    return x[tuple(indices)]

def flip_cihp(tail_list):
    '''

    :param tail_list: tail_list size is 1 x n_class x h x w
    :return:
    '''
    # tail_list = tail_list[0]
    tail_list_rev = [None] * 20
    for xx in range(14):
        tail_list_rev[xx] = tail_list[xx].unsqueeze(0)
    tail_list_rev[14] = tail_list[15].unsqueeze(0)
    tail_list_rev[15] = tail_list[14].unsqueeze(0)
    tail_list_rev[16] = tail_list[17].unsqueeze(0)
    tail_list_rev[17] = tail_list[16].unsqueeze(0)
    tail_list_rev[18] = tail_list[19].unsqueeze(0)
    tail_list_rev[19] = tail_list[18].unsqueeze(0)
    return torch.cat(tail_list_rev,dim=0)


def decode_labels(mask, num_images=1, num_classes=20):
    """Decode batch of segmentation masks.

    Args:
      mask: result of inference after taking argmax.
      num_images: number of images to decode from the batch.
      num_classes: number of classes to predict (including background).

    Returns:
      A batch with num_images RGB images of the same size as the input.
    """
    n, h, w = mask.shape
    assert (n >= num_images), 'Batch size %d should be greater or equal than number of images to save %d.' % (
    n, num_images)
    outputs = np.zeros((num_images, h, w, 3), dtype=np.uint8)
    for i in range(num_images):
        img = Image.new('RGB', (len(mask[i, 0]), len(mask[i])))
        pixels = img.load()
        for j_, j in enumerate(mask[i, :, :]):
            for k_, k in enumerate(j):
                if k < num_classes:
                    pixels[k_, j_] = label_colours[k]
        outputs[i] = np.array(img)
    return outputs

def read_img(img_path):
    _img = Image.open(img_path).convert('RGB')  # return is RGB pic
    return _img

def img_transform(img, transform=None):
    sample = {'image': img, 'label': 0}

    sample = transform(sample)
    return sample

def inference(net, img_path='', output_path='./', output_name='f', use_gpu=False):
    '''

    :param net:
    :param img_path:
    :param output_path:
    :return:
    '''
    # Tạo thư mục đầu ra nếu chưa tồn tại
    os.makedirs(output_path, exist_ok=True)
    
    # adj
    adj2_ = torch.from_numpy(graph.cihp2pascal_nlp_adj).float()
    if use_gpu:
        adj2_test = adj2_.unsqueeze(0).unsqueeze(0).expand(1, 1, 7, 20).cuda().transpose(2, 3)
    else:
        adj2_test = adj2_.unsqueeze(0).unsqueeze(0).expand(1, 1, 7, 20).transpose(2, 3)

    adj1_ = Variable(torch.from_numpy(graph.preprocess_adj(graph.pascal_graph)).float())
    if use_gpu:
        adj3_test = adj1_.unsqueeze(0).unsqueeze(0).expand(1, 1, 7, 7).cuda()
    else:
        adj3_test = adj1_.unsqueeze(0).unsqueeze(0).expand(1, 1, 7, 7)

    cihp_adj = graph.preprocess_adj(graph.cihp_graph)
    adj3_ = Variable(torch.from_numpy(cihp_adj).float())
    if use_gpu:
        adj1_test = adj3_.unsqueeze(0).unsqueeze(0).expand(1, 1, 20, 20).cuda()
    else:
        adj1_test = adj3_.unsqueeze(0).unsqueeze(0).expand(1, 1, 20, 20)

    # multi-scale
    scale_list = [1, 0.5, 0.75, 1.25, 1.5, 1.75]
    img = read_img(img_path)
    testloader_list = []
    testloader_flip_list = []
    for pv in scale_list:
        composed_transforms_ts = transforms.Compose([
            tr.Scale_only_img(pv),
            tr.Normalize_xception_tf_only_img(),
            tr.ToTensor_only_img()])

        tensor_in = img_transform(img, composed_transforms_ts)['image']
        testloader_list.append(tensor_in)
        # flip
        if use_gpu:
            testloader_flip_list.append(flip(tensor_in.cuda(), 2))
        else:
            testloader_flip_list.append(flip(tensor_in, 2))

    start_time = timeit.default_timer()
    # One testing epoch
    net.eval()
    # 1 0.5 0.75 1.25 1.5 1.75 ; flip:
    
    outputs_all = []
    outputs_f_all = []
    
    for iii, sample_batched in enumerate(zip(testloader_list, testloader_flip_list)):
        inputs, inputs_f = sample_batched
        if use_gpu:
            inputs = inputs.unsqueeze(0).cuda()
            inputs_f = inputs_f.unsqueeze(0).cuda()
            with torch.no_grad():
                outputs = net.forward(inputs, adj1_test, adj3_test, adj2_test)
                outputs_f = net.forward(inputs_f, adj1_test, adj3_test, adj2_test)
        else:
            inputs = inputs.unsqueeze(0)
            inputs_f = inputs_f.unsqueeze(0)
            with torch.no_grad():
                outputs = net.forward(inputs, adj1_test, adj3_test, adj2_test)
                outputs_f = net.forward(inputs_f, adj1_test, adj3_test, adj2_test)
                
        outputs = outputs[0]
        outputs_f = outputs_f[0]
        outputs_f = flip(outputs_f, 2)
        outputs = outputs.cpu()
        outputs_f = outputs_f.cpu()
        
        # Lưu kết quả
        outputs_all.append(outputs)
        outputs_f_all.append(outputs_f)
    
    # Chọn kích thước cơ sở để điều chỉnh tất cả các đầu ra về cùng kích thước
    base_output = outputs_all[0]  # Dùng output của scale đầu tiên làm cơ sở
    base_size = base_output.shape[-2:]  # Lấy chiều cao và rộng
    
    # Khởi tạo outputs_final với kích thước đúng
    outputs_final = torch.zeros_like(base_output)
    
    # Kết hợp tất cả các đầu ra đã điều chỉnh kích thước
    for i in range(len(outputs_all)):
        output = outputs_all[i]
        output_f = outputs_f_all[i]
        
        # Kiểm tra kích thước và điều chỉnh nếu cần
        if output.shape[-2:] != base_size:
            output = torch.nn.functional.interpolate(
                output.unsqueeze(0),
                size=base_size,
                mode='bilinear',
                align_corners=True
            )[0]
        
        if output_f.shape[-2:] != base_size:
            output_f = torch.nn.functional.interpolate(
                output_f.unsqueeze(0),
                size=base_size,
                mode='bilinear',
                align_corners=True
            )[0]
        
        # Cộng vào outputs_final
        outputs_final += output + output_f
    
    ################ plot pic
    predictions = torch.max(outputs_final, 0)[1]
    results = predictions.cpu().numpy()
    results = results.reshape(1, results.shape[0], results.shape[1])  # Thêm chiều batch
    vis_res = decode_labels(results)

    parsing_im = Image.fromarray(vis_res[0])
    # parsing_im.save(output_path+'/{}.png'.format(output_name))
    cv2.imwrite(output_path+'/{}.png'.format(output_name), results[0, :, :])

    end_time = timeit.default_timer()
    print('time used for the multi-scale image inference' + ' is :' + str(end_time - start_time))

if __name__ == '__main__':
    '''argparse begin'''
    parser = argparse.ArgumentParser()
    # parser.add_argument('--loadmodel',default=None,type=str)
    parser.add_argument('--loadmodel', default='', type=str)
    parser.add_argument('--img_path', default='', type=str)
    parser.add_argument('--output_path', default='', type=str)
    parser.add_argument('--output_name', default='', type=str)
    parser.add_argument('--use_gpu', default=0, type=int)
    opts = parser.parse_args()

    net = deeplab_xception_transfer.deeplab_xception_transfer_projection_savemem(n_classes=20,
                                                                                 hidden_layers=128,
                                                                                 source_classes=7, )
    if not opts.loadmodel == '':
        x = torch.load(opts.loadmodel, map_location=torch.device('cpu'))
        net.load_state_dict(x)
        print('Load model:', opts.loadmodel)
    else:
        print('No model is loaded!')

    use_gpu = False
    if opts.use_gpu > 0 and torch.cuda.is_available():
        net.cuda()
        use_gpu = True
        print('Using GPU for inference')
    else:
        use_gpu = False
        print('Using CPU for inference')

    inference(net=net, img_path=opts.img_path, output_path=opts.output_path, output_name=opts.output_name, use_gpu=use_gpu)

