# coding=utf-8
import torch
import torch.nn as nn
import torch.nn.functional as F
import argparse
import os
import time
import glob
import shutil
import cv2
import numpy as np
from cp_dataset import CPDataset, CPDataLoader
from networks import GMM, UnetGenerator, load_checkpoint
from gen_data import process_and_prepare_data
from tensorboardX import SummaryWriter
from visualization import board_add_images, save_images


def check_input_directory(input_dir="./input"):
    """Kiểm tra thư mục input"""
    if not os.path.exists(input_dir):
        print(f"Thư mục input không tồn tại: {input_dir}")
        return False
    
    person_dir = os.path.join(input_dir, "person_images")
    cloth_dir = os.path.join(input_dir, "cloth_images")
    
    if not os.path.exists(person_dir):
        print(f"Thiếu thư mục: {person_dir}")
        return False
    
    if not os.path.exists(cloth_dir):
        print(f"Thiếu thư mục: {cloth_dir}")
        return False
    
    person_files = [f for f in os.listdir(person_dir) if f.lower().endswith(('.jpg', '.jpeg', '.png', '.bmp'))]
    cloth_files = [f for f in os.listdir(cloth_dir) if f.lower().endswith(('.jpg', '.jpeg', '.png', '.bmp'))]
    
    if not person_files or not cloth_files:
        print("Không tìm thấy ảnh trong thư mục")
        return False
    
    print(f"Tìm thấy {len(person_files)} ảnh người và {len(cloth_files)} ảnh quần áo")
    return True


def create_input_structure(input_dir="./input"):
    """Tạo cấu trúc thư mục input"""
    person_dir = os.path.join(input_dir, "person_images")
    cloth_dir = os.path.join(input_dir, "cloth_images")
    
    os.makedirs(person_dir, exist_ok=True)
    os.makedirs(cloth_dir, exist_ok=True)
    
    print(f"Đã tạo cấu trúc thư mục:")
    print(f"  {person_dir}")
    print(f"  {cloth_dir}")


# def check_data_availability(opt):
#     """Kiểm tra dữ liệu test"""
#     data_list_path = os.path.join(opt.dataroot, opt.datamode, opt.data_list)
    
#     if not os.path.exists(data_list_path):
#         print(f"Không tìm thấy {data_list_path}")
#         return False
    
    # with open(data_list_path, 'r') as f:
    #     lines = f.readlines()
    
    # total_pairs = len([line for line in lines if line.strip()])
    # print(f"Tổng số pairs: {total_pairs}")
    # return True


def upscale_with_opencv(image_path, output_path="upscaled_opencv.jpg"):
    """Upscale với OpenCV (thuật toán khác)"""
    
    try:
        print("🔬 Upscale với OpenCV...")
        
        # Đọc ảnh với OpenCV
        img = cv2.imread(image_path)
        if img is None:
            print("❌ Không thể đọc ảnh với OpenCV")
            return None
        
        height, width = img.shape[:2]
        print(f"📐 Kích thước gốc: {width}x{height}")
        
        # Tính tỷ lệ scale
        target_width = 1920
        target_height = 1080
        
        scale_x = target_width / width
        scale_y = target_height / height
        scale = min(scale_x, scale_y)
        
        new_width = int(width * scale)
        new_height = int(height * scale)
        
        print(f"📏 Scale: {scale:.2f}x -> {new_width}x{new_height}")
        
        # Upscale với INTER_CUBIC (chất lượng cao)
        img_resized = cv2.resize(img, (new_width, new_height), interpolation=cv2.INTER_CUBIC)
        
        # Tạo canvas 1920x1080
        canvas = np.zeros((target_height, target_width, 3), dtype=np.uint8)
        
        # Center ảnh
        x = (target_width - new_width) // 2
        y = (target_height - new_height) // 2
        
        canvas[y:y+new_height, x:x+new_width] = img_resized
        
        # Áp dụng sharpening filter
        kernel = np.array([[-1,-1,-1], [-1,9,-1], [-1,-1,-1]])
        canvas = cv2.filter2D(canvas, -1, kernel)
        
        # Lưu ảnh
        cv2.imwrite(output_path, canvas, [cv2.IMWRITE_JPEG_QUALITY, 95])
        print(f"✅ OpenCV upscale đã lưu: {output_path}")
        
        return output_path
        
    except Exception as e:
        print(f"❌ Lỗi OpenCV: {e}")
        return None


def upscale_results(result_dir):
    """Upscale tất cả kết quả trong thư mục và trả về thư mục final_result"""
    if not os.path.exists(result_dir):
        print(f"Thư mục kết quả không tồn tại: {result_dir}")
        return None
    
    # Tạo thư mục final_result
    final_result_dir = "final_result"
    os.makedirs(final_result_dir, exist_ok=True)
    
    # Tìm tất cả file ảnh
    image_files = []
    for ext in ['*.jpg', '*.jpeg', '*.png', '*.bmp']:
        image_files.extend(glob.glob(os.path.join(result_dir, ext)))
    
    if not image_files:
        print("Không tìm thấy ảnh để upscale")
        return None
    
    print(f"🚀 Bắt đầu upscale {len(image_files)} ảnh...")
    
    for i, image_path in enumerate(image_files):
        filename = os.path.basename(image_path)
        name, ext = os.path.splitext(filename)
        output_path = os.path.join(final_result_dir, f"{name}_1080p{ext}")
        
        print(f"[{i+1}/{len(image_files)}] Xử lý: {filename}")
        result = upscale_with_opencv(image_path, output_path)
        
        if result:
            print(f"✅ Hoàn thành: {output_path}")
        else:
            print(f"❌ Thất bại: {filename}")
    
    print(f"🎉 Upscale hoàn tất! Kết quả trong: {final_result_dir}")
    return final_result_dir


def clean_temp_directories():
    """Xóa các thư mục tạm sau khi upscale"""
    temp_dirs = ["data/test", "result"]
    
    for temp_dir in temp_dirs:
        if os.path.exists(temp_dir):
            try:
                shutil.rmtree(temp_dir)
                print(f"🗑️ Đã xóa thư mục tạm: {temp_dir}")
            except Exception as e:
                print(f"❌ Không thể xóa {temp_dir}: {e}")


def get_opt():
    parser = argparse.ArgumentParser()
    parser.add_argument("--name", default="GMM")
    parser.add_argument("--gpu_ids", default="")
    parser.add_argument('-j', '--workers', type=int, default=1)
    parser.add_argument('-b', '--batch-size', type=int, default=4)
    parser.add_argument("--dataroot", default="data")
    parser.add_argument("--datamode", default="test")
    parser.add_argument("--stage", default="BOTH")
    parser.add_argument("--data_list", default="test_pairs.txt")
    parser.add_argument("--fine_width", type=int, default=192)
    parser.add_argument("--fine_height", type=int, default=256)
    parser.add_argument("--radius", type=int, default=5)
    parser.add_argument("--grid_size", type=int, default=5)
    parser.add_argument('--gen_data', action='store_true', default=True, help='Tạo dữ liệu (mặc định bật)')
    parser.add_argument('--source_dir', type=str, default='./input')
    parser.add_argument('--output_dir', type=str, default='./data/test')
    parser.add_argument('--tensorboard_dir', type=str, default='tensorboard')
    parser.add_argument('--result_dir_gmm', type=str, default='data/test')
    parser.add_argument('--result_dir_tom', type=str, default='result')
    parser.add_argument('--gmm_checkpoint', type=str, default='checkpoints/GMM/gmm_final.pth')
    parser.add_argument('--tom_checkpoint', type=str, default='checkpoints/TOM/tom_final.pth')
    parser.add_argument("--display_count", type=int, default=1)
    parser.add_argument("--shuffle", action='store_true')
    parser.add_argument("--use_cuda", action='store_true')
    parser.add_argument("--upscale", action='store_true', default=True, help='Upscale kết quả lên 1920x1080 (mặc định bật)')
    parser.add_argument("--keep_temp", action='store_true', help='Giữ lại thư mục tạm (mặc định xóa)')

    opt = parser.parse_args()
    opt.device = torch.device('cuda' if opt.gpu_ids and torch.cuda.is_available() and opt.use_cuda else 'cpu')
    print(f"Sử dụng thiết bị: {opt.device}")
    return opt


def test_gmm(opt, test_loader, model, board):
    model = model.to(opt.device)
    model.eval()
    
    save_dir = os.path.join(opt.result_dir_gmm)
    os.makedirs(save_dir, exist_ok=True)
    
    dirs = ['warp_cloth', 'warp_mask', 'result_dir', 'overlayed_TPS', 'warped_grid']
    for dir_name in dirs:
        os.makedirs(os.path.join(save_dir, dir_name), exist_ok=True)
    
    print("Đang xử lý GMM...")
    for step, inputs in enumerate(test_loader.data_loader):
        iter_start_time = time.time()
        
        # Move inputs to device
        for key in ['image', 'pose_image', 'head', 'shape', 'agnostic', 'cloth', 'cloth_mask', 'parse_cloth', 'grid_image']:
            if key in inputs:
                inputs[key] = inputs[key].to(opt.device)
        
        grid, theta = model(inputs['agnostic'], inputs['cloth_mask'])
        warped_cloth = F.grid_sample(inputs['cloth'], grid, padding_mode='border')
        warped_mask = F.grid_sample(inputs['cloth_mask'], grid, padding_mode='zeros')
        warped_grid = F.grid_sample(inputs['grid_image'], grid, padding_mode='zeros')
        
        # Save results
        save_images(warped_cloth, inputs['im_name'], os.path.join(save_dir, 'warp_cloth'))
        save_images(warped_mask * 2 - 1, inputs['im_name'], os.path.join(save_dir, 'warp_mask'))
        save_images(warped_grid, inputs['im_name'], os.path.join(save_dir, 'warped_grid'))
        
        if (step+1) % opt.display_count == 0:
            t = time.time() - iter_start_time
            print(f'step: {step+1:8d}, time: {t:.3f}', flush=True)
    
    return save_dir


def test_tom(opt, test_loader, model, board):
    model = model.to(opt.device)
    model.eval()
    
    save_dir = os.path.join(opt.result_dir_tom)
    os.makedirs(save_dir, exist_ok=True)
    
    dirs = ['try_on', 'p_rendered', 'm_composite', 'im_pose', 'shape', 'im_h']
    for dir_name in dirs:
        os.makedirs(os.path.join(save_dir, dir_name), exist_ok=True)
    
    print("Đang xử lý TOM...")
    for step, inputs in enumerate(test_loader.data_loader):
        iter_start_time = time.time()
        
        # Move inputs to device
        inputs['image'] = inputs['image'].to(opt.device)
        inputs['agnostic'] = inputs['agnostic'].to(opt.device)
        inputs['cloth'] = inputs['cloth'].to(opt.device)
        inputs['cloth_mask'] = inputs['cloth_mask'].to(opt.device)
        
        outputs = model(torch.cat([inputs['agnostic'], inputs['cloth'], inputs['cloth_mask']], 1))
        p_rendered, m_composite = torch.split(outputs, 3, 1)
        p_rendered = F.tanh(p_rendered)
        m_composite = F.sigmoid(m_composite)
        
        p_tryon = inputs['cloth'] * m_composite + p_rendered * (1 - m_composite)
        
        # Save results
        save_images(p_tryon, inputs['im_name'], os.path.join(save_dir, 'try_on'))
        save_images(inputs['head'], inputs['im_name'], os.path.join(save_dir, 'im_h'))
        save_images(inputs['shape'], inputs['im_name'], os.path.join(save_dir, 'shape'))
        save_images(inputs['pose_image'], inputs['im_name'], os.path.join(save_dir, 'im_pose'))
        save_images(m_composite, inputs['im_name'], os.path.join(save_dir, 'm_composite'))
        save_images(p_rendered, inputs['im_name'], os.path.join(save_dir, 'p_rendered'))
        
        if (step+1) % opt.display_count == 0:
            t = time.time() - iter_start_time
            print(f'step: {step+1:8d}, time: {t:.3f}', flush=True)
    
    return os.path.join(save_dir, 'try_on')


def main():
    opt = get_opt()
    print(f"Cấu hình: Stage={opt.stage}, Device={opt.device}")
    print(f"Gen data: {opt.gen_data}, Upscale: {opt.upscale}")
    
    os.makedirs(opt.tensorboard_dir, exist_ok=True)
    
    # Tạo dữ liệu (mặc định bật)
    if opt.gen_data:
        print("=== TẠO DỮ LIỆU ===")
        create_input_structure(opt.source_dir)
        
        if not check_input_directory(opt.source_dir):
            print("Vui lòng đặt ảnh vào thư mục input/ với cấu trúc:")
            print("  input/person_images/")
            print("  input/cloth_images/")
            return
        
        try:
            process_and_prepare_data(opt.source_dir, opt.output_dir)
            print("Tạo dữ liệu hoàn tất!")
        except Exception as e:
            print(f"Lỗi tạo dữ liệu: {e}")
            return
    
    # # Kiểm tra dữ liệu
    # if not check_data_availability(opt):
    #     print("Chạy với --gen_data để tạo dữ liệu")
    #     return
    
    tom_output_dir = None
    
    # Chạy GMM
    if opt.stage in ['GMM', 'BOTH']:
        print("=== GMM ===")
        opt_gmm = argparse.Namespace(**vars(opt))
        opt_gmm.stage = 'GMM'
        
        test_dataset_gmm = CPDataset(opt_gmm)
        test_loader_gmm = CPDataLoader(opt_gmm, test_dataset_gmm)
        board_gmm = SummaryWriter(logdir=os.path.join(opt.tensorboard_dir, 'GMM'))
        
        model_gmm = GMM(opt_gmm)
        load_checkpoint(model_gmm, opt.gmm_checkpoint)
        
        with torch.no_grad():
            gmm_output_dir = test_gmm(opt_gmm, test_loader_gmm, model_gmm, board_gmm)
        
        board_gmm.close()
        print(f"GMM hoàn thành: {gmm_output_dir}")
    
    # Chạy TOM
    if opt.stage in ['TOM', 'BOTH']:
        print("=== TOM ===")
        opt_tom = argparse.Namespace(**vars(opt))
        opt_tom.stage = 'TOM'
        
        test_dataset_tom = CPDataset(opt_tom)
        test_loader_tom = CPDataLoader(opt_tom, test_dataset_tom)
        board_tom = SummaryWriter(logdir=os.path.join(opt.tensorboard_dir, 'TOM'))
        
        model_tom = UnetGenerator(26, 4, 6, ngf=64, norm_layer=nn.InstanceNorm2d)
        load_checkpoint(model_tom, opt.tom_checkpoint)
        
        with torch.no_grad():
            tom_output_dir = test_tom(opt_tom, test_loader_tom, model_tom, board_tom)
        
        board_tom.close()
        print(f"TOM hoàn thành: {tom_output_dir}")
    
    # Upscale kết quả (mặc định bật)
    if opt.upscale and tom_output_dir:
        print("=== UPSCALE KẾT QUẢ ===")
        final_result_dir = upscale_results(tom_output_dir)
        
        if final_result_dir:
            print(f"✅ Kết quả cuối cùng: {final_result_dir}")
            
            # Xóa thư mục tạm (trừ khi có flag --keep_temp)
            if not opt.keep_temp:
                print("=== DỌN DẸP THƯƠNG MỤC TẠM ===")
                clean_temp_directories()
        else:
            print("❌ Upscale thất bại")
    elif opt.upscale:
        print("❌ Không có kết quả TOM để upscale")
    
    print("=== HOÀN THÀNH ===")


if __name__ == "__main__":
    main()