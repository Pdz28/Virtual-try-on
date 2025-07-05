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
from visualization import board_add_image, board_add_images, save_images


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


def upscale_try_on_results(try_on_dir, upscale_dir="upscaled_results"):
    """Upscale tất cả ảnh try-on results lên 1920x1080"""
    
    if not os.path.exists(try_on_dir):
        print(f"❌ Thư mục try-on không tồn tại: {try_on_dir}")
        return
    
    # Tạo thư mục upscale
    if not os.path.exists(upscale_dir):
        os.makedirs(upscale_dir)
    
    print(f"\n🚀 Bắt đầu upscale tất cả ảnh từ {try_on_dir}...")
    
    # Lấy tất cả file ảnh
    image_files = []
    for ext in ['*.jpg', '*.jpeg', '*.png', '*.bmp']:
        image_files.extend(glob.glob(os.path.join(try_on_dir, ext)))
    
    if not image_files:
        print("❌ Không tìm thấy ảnh nào để upscale")
        return
    
    print(f"📁 Tìm thấy {len(image_files)} ảnh để upscale")
    
    successful_upscales = 0
    
    for i, image_path in enumerate(image_files):
        print(f"\n📸 Đang xử lý ({i+1}/{len(image_files)}): {os.path.basename(image_path)}")
        
        # Tên file output
        filename = os.path.basename(image_path)
        name, ext = os.path.splitext(filename)
        output_path = os.path.join(upscale_dir, f"{name}_1080p{ext}")
        
        # Upscale
        result = upscale_with_opencv(image_path, output_path)
        
        if result:
            successful_upscales += 1
            file_size = os.path.getsize(result) / (1024 * 1024)  # MB
            print(f"✅ Hoàn thành: {os.path.basename(result)} ({file_size:.1f} MB)")
        else:
            print(f"❌ Thất bại: {filename}")
    
    print(f"\n🎉 HOÀN THÀNH UPSCALE!")
    print(f"✅ Thành công: {successful_upscales}/{len(image_files)} ảnh")
    print(f"📁 Kết quả được lưu trong: {upscale_dir}")


def cleanup_directories():
    """
    Dọn dẹp các thư mục sau khi tạo kết quả:
    - Xóa tất cả thư mục trong result/ trừ try_on/
    - Xóa tất cả nội dung trong data/test/
    """
    
    # Đường dẫn gốc
    base_path = os.path.dirname(os.path.abspath(__file__))
    result_path = os.path.join(base_path, 'result')
    data_test_path = os.path.join(base_path, 'data', 'test')
    
    print("🧹 Bắt đầu dọn dẹp thư mục...")
    
    # 1. Dọn dẹp thư mục result (giữ lại try_on)
    if os.path.exists(result_path):
        print(f"📁 Dọn dẹp thư mục: {result_path}")
        
        for item in os.listdir(result_path):
            item_path = os.path.join(result_path, item)
            
            # Bỏ qua thư mục try_on
            if item == 'try_on':
                print(f"Giữ lại: {item}")
                continue
            
            # Xóa các thư mục/file khác
            if os.path.isdir(item_path):
                try:
                    shutil.rmtree(item_path)
                    print(f"Đã xóa thư mục: {item}")
                except Exception as e:
                    print(f"Lỗi khi xóa {item}: {e}")
            elif os.path.isfile(item_path):
                try:
                    os.remove(item_path)
                    print(f"Đã xóa file: {item}")
                except Exception as e:
                    print(f"Lỗi khi xóa {item}: {e}")
    
    # 2. Dọn dẹp thư mục data/test
    if os.path.exists(data_test_path):
        print(f"📁 Dọn dẹp thư mục: {data_test_path}")
        
        for item in os.listdir(data_test_path):
            item_path = os.path.join(data_test_path, item)
            
            if os.path.isdir(item_path):
                try:
                    shutil.rmtree(item_path)
                    print(f"Đã xóa thư mục: {item}")
                except Exception as e:
                    print(f"Lỗi khi xóa {item}: {e}")
            elif os.path.isfile(item_path):
                try:
                    os.remove(item_path)
                    print(f"Đã xóa file: {item}")
                except Exception as e:
                    print(f"Lỗi khi xóa {item}: {e}")
    
    print("Hoàn thành dọn dẹp!")


def get_opt():
    parser = argparse.ArgumentParser()

    parser.add_argument("--name", default="GMM")
    parser.add_argument("--gpu_ids", default="")
    parser.add_argument('-j', '--workers', type=int, default=1)
    parser.add_argument('-b', '--batch-size', type=int, default=4)
    parser.add_argument("--dataroot", default="data")
    parser.add_argument("--datamode", default="test")
    parser.add_argument("--stage", default="BOTH")  # GMM, TOM hoặc BOTH
    parser.add_argument("--data_list", default="test_pairs.txt")
    parser.add_argument("--fine_width", type=int, default=192)
    parser.add_argument("--fine_height", type=int, default=256)
    parser.add_argument("--radius", type=int, default=5)
    parser.add_argument("--grid_size", type=int, default=5)

    parser.add_argument('--gen_data', action='store_true', help='Generate data before testing')
    parser.add_argument('--source_dir', type=str, default='./input', help='Directory with source images')
    parser.add_argument('--output_dir', type=str, default='./data/test', help='Output directory for processed data')

    parser.add_argument('--tensorboard_dir', type=str,
                        default='tensorboard', help='save tensorboard infos')
    
    parser.add_argument('--result_dir_gmm', type=str,
                        default='data/test', help='save GMM result infos')
    parser.add_argument('--result_dir_tom', type=str,
                        default='result', help='save result infos')
    
    parser.add_argument('--gmm_checkpoint', type=str, default='checkpoints/GMM/gmm_final.pth', help='GMM model checkpoint')
    parser.add_argument('--tom_checkpoint', type=str, default='checkpoints/TOM/tom_final.pth', help='TOM model checkpoint')

    parser.add_argument("--display_count", type=int, default=1)
    parser.add_argument("--shuffle", action='store_true',
                        help='shuffle input data')
    parser.add_argument("--use_cuda", action='store_true',
                        help='use cuda')

    opt = parser.parse_args()
    # Xác định thiết bị (CPU hoặc GPU)
    opt.device = torch.device('cuda' if opt.gpu_ids and torch.cuda.is_available() and opt.use_cuda else 'cpu')
    print(f"Sử dụng thiết bị: {opt.device}")
    return opt


def test_gmm(opt, test_loader, model, board):
    model = model.to(opt.device)
    model.eval()

    name = "GMM"
    save_dir = os.path.join(opt.result_dir_gmm)
    if not os.path.exists(save_dir):
        os.makedirs(save_dir)
    warp_cloth_dir = os.path.join(save_dir, 'warp_cloth')
    if not os.path.exists(warp_cloth_dir):
        os.makedirs(warp_cloth_dir)
    warp_mask_dir = os.path.join(save_dir, 'warp_mask')
    if not os.path.exists(warp_mask_dir):
        os.makedirs(warp_mask_dir)
    result_dir1 = os.path.join(save_dir, 'result_dir')
    if not os.path.exists(result_dir1):
        os.makedirs(result_dir1)
    overlayed_TPS_dir = os.path.join(save_dir, 'overlayed_TPS')
    if not os.path.exists(overlayed_TPS_dir):
        os.makedirs(overlayed_TPS_dir)
    warped_grid_dir = os.path.join(save_dir, 'warped_grid')
    if not os.path.exists(warped_grid_dir):
        os.makedirs(warped_grid_dir)
    
    print("Đang xử lý GMM...")
    for step, inputs in enumerate(test_loader.data_loader):
        iter_start_time = time.time()

        c_names = inputs['c_name']
        im_names = inputs['im_name']
        im = inputs['image'].to(opt.device)
        im_pose = inputs['pose_image'].to(opt.device)
        im_h = inputs['head'].to(opt.device)
        shape = inputs['shape'].to(opt.device)
        agnostic = inputs['agnostic'].to(opt.device)
        c = inputs['cloth'].to(opt.device)
        cm = inputs['cloth_mask'].to(opt.device)
        im_c = inputs['parse_cloth'].to(opt.device)
        im_g = inputs['grid_image'].to(opt.device)
        shape_ori = inputs['shape_ori']  # original body shape without blurring

        grid, theta = model(agnostic, cm)
        warped_cloth = F.grid_sample(c, grid, padding_mode='border')
        warped_mask = F.grid_sample(cm, grid, padding_mode='zeros')
        warped_grid = F.grid_sample(im_g, grid, padding_mode='zeros')
        overlay = 0.7 * warped_cloth + 0.3 * im

        visuals = [[im_h, shape, im_pose],
                   [c, warped_cloth, im_c],
                   [warped_grid, (warped_cloth+im)*0.5, im]]

        save_images(warped_cloth, im_names, warp_cloth_dir)
        save_images(warped_mask * 2 - 1, im_names, warp_mask_dir)
        save_images(shape_ori.to(opt.device) * 0.2 + warped_cloth *
                    0.8, im_names, result_dir1)
        save_images(warped_grid, im_names, warped_grid_dir)
        save_images(overlay, im_names, overlayed_TPS_dir)

        if (step+1) % opt.display_count == 0:
            board_add_images(board, 'combine', visuals, step+1)
            t = time.time() - iter_start_time
            print('step: %8d, time: %.3f' % (step+1, t), flush=True)
    
    return save_dir


def test_tom(opt, test_loader, model, board, gmm_outputs_dir=None):
    model = model.to(opt.device)
    model.eval()

    name = "TOM"
    save_dir = os.path.join(opt.result_dir_tom)
    if not os.path.exists(save_dir):
        os.makedirs(save_dir)
    try_on_dir = os.path.join(save_dir, 'try_on')
    if not os.path.exists(try_on_dir):
        os.makedirs(try_on_dir)
    p_rendered_dir = os.path.join(save_dir, 'p_rendered')
    if not os.path.exists(p_rendered_dir):
        os.makedirs(p_rendered_dir)
    m_composite_dir = os.path.join(save_dir, 'm_composite')
    if not os.path.exists(m_composite_dir):
        os.makedirs(m_composite_dir)
    im_pose_dir = os.path.join(save_dir, 'im_pose')
    if not os.path.exists(im_pose_dir):
        os.makedirs(im_pose_dir)
    shape_dir = os.path.join(save_dir, 'shape')
    if not os.path.exists(shape_dir):
        os.makedirs(shape_dir)
    im_h_dir = os.path.join(save_dir, 'im_h')
    if not os.path.exists(im_h_dir):
        os.makedirs(im_h_dir)  # for test data

    print('Dataset size: %05d!' % (len(test_loader.dataset)), flush=True)
    print("Đang xử lý TOM...")
    
    for step, inputs in enumerate(test_loader.data_loader):
        iter_start_time = time.time()

        im_names = inputs['im_name']
        im = inputs['image'].to(opt.device)
        im_pose = inputs['pose_image']
        im_h = inputs['head']
        shape = inputs['shape']

        agnostic = inputs['agnostic'].to(opt.device)
        c = inputs['cloth'].to(opt.device)
        cm = inputs['cloth_mask'].to(opt.device)
        
        # Nếu có protected mask, sử dụng nó
        # protected_mask = None
        # if 'protected_mask' in inputs:
        #     protected_mask = inputs['protected_mask'].to(opt.device)

        outputs = model(torch.cat([agnostic, c, cm], 1))  # CP-VTON+
        p_rendered, m_composite = torch.split(outputs, 3, 1)
        p_rendered = F.tanh(p_rendered)
        m_composite = F.sigmoid(m_composite)
        
        # Nếu có protected mask, áp dụng nó để tránh áo dính lên tóc và cổ
        # if protected_mask is not None:
        #     m_composite = m_composite * (1 - protected_mask)
            
        p_tryon = c * m_composite + p_rendered * (1 - m_composite)

        visuals = [[im_h, shape, im_pose],
                   [c, 2*cm-1, m_composite],
                   [p_rendered, p_tryon, im]]

        save_images(p_tryon, im_names, try_on_dir)
        save_images(im_h, im_names, im_h_dir)
        save_images(shape, im_names, shape_dir)
        save_images(im_pose, im_names, im_pose_dir)
        save_images(m_composite, im_names, m_composite_dir)
        save_images(p_rendered, im_names, p_rendered_dir)  # For test data

        if (step+1) % opt.display_count == 0:
            board_add_images(board, 'combine', visuals, step+1)
            t = time.time() - iter_start_time
            print('step: %8d, time: %.3f' % (step+1, t), flush=True)
    
    return try_on_dir


def main():
    opt = get_opt()
    print(opt)
    
    # Tạo thư mục tensorboard nếu chưa có
    if not os.path.exists(opt.tensorboard_dir):
        os.makedirs(opt.tensorboard_dir)

    # Nếu cần, tạo dữ liệu trước khi chạy    
    if opt.gen_data:
        print("Generating data before testing...")
        try:
            process_and_prepare_data(opt.source_dir, opt.output_dir)
            print("Data generation completed.")
        except Exception as e:
            print(f"Lỗi khi tạo dữ liệu: {str(e)}")
            import traceback
            traceback.print_exc()

    tom_output_dir = None
    
    # Chạy GMM
    if opt.stage == 'GMM' or opt.stage == 'BOTH':
        print("=== Đang khởi tạo giai đoạn GMM ===")
        # Cấu hình cho GMM
        opt_gmm = argparse.Namespace(**vars(opt))
        opt_gmm.stage = 'GMM'
        
        # Tạo dataset cho GMM
        test_dataset_gmm = CPDataset(opt_gmm)
        test_loader_gmm = CPDataLoader(opt_gmm, test_dataset_gmm)
        
        # Tạo board cho GMM
        board_gmm = SummaryWriter(logdir=os.path.join(opt.tensorboard_dir, 'GMM'))
        
        # Load model GMM
        model_gmm = GMM(opt_gmm)
        load_checkpoint(model_gmm, opt.gmm_checkpoint)
        
        # Chạy GMM
        with torch.no_grad():
            gmm_output_dir = test_gmm(opt_gmm, test_loader_gmm, model_gmm, board_gmm)
        
        print(f"Hoàn thành GMM, kết quả được lưu vào {gmm_output_dir}")
        
        # Đóng board GMM khi xong
        board_gmm.close()
    
    # Chạy TOM
    if opt.stage == 'TOM' or opt.stage == 'BOTH':
        print("\n=== Đang khởi tạo giai đoạn TOM ===")
        # Cấu hình cho TOM
        opt_tom = argparse.Namespace(**vars(opt))
        opt_tom.stage = 'TOM'
        
        # Tạo dataset mới cho TOM - quan trọng vì logic dataset khác nhau cho GMM và TOM
        test_dataset_tom = CPDataset(opt_tom)
        test_loader_tom = CPDataLoader(opt_tom, test_dataset_tom)
        
        # Tạo board cho TOM
        board_tom = SummaryWriter(logdir=os.path.join(opt.tensorboard_dir, 'TOM'))
        
        # Load model TOM
        model_tom = UnetGenerator(26, 4, 6, ngf=64, norm_layer=nn.InstanceNorm2d)  # CP-VTON+
        load_checkpoint(model_tom, opt.tom_checkpoint)
        
        # Chạy TOM
        with torch.no_grad():
            tom_output_dir = test_tom(opt_tom, test_loader_tom, model_tom, board_tom)
        
        print(f"Hoàn thành TOM, kết quả được lưu vào {tom_output_dir}")
        
        # Đóng board TOM khi xong
        board_tom.close()
    
    print('\nQuá trình xử lý đã hoàn tất!')
    
    # Upscale tự động sau khi hoàn thành TOM
    if tom_output_dir:
        print('\n🚀 Bắt đầu upscale kết quả lên 1920x1080...')
        try:
            upscale_dir = "upscaled_results"
            upscale_try_on_results(tom_output_dir, upscale_dir)
            print(f'✅ Hoàn thành upscale! Kết quả được lưu trong: {upscale_dir}')
        except Exception as e:
            print(f'❌ Lỗi khi upscale: {e}')
    
    # Dọn dẹp tự động sau khi hoàn thành
    try:
        print('\n🧹 Đang dọn dẹp thư mục...')
        cleanup_directories()
    except Exception as e:
        print(f'\n❌ Lỗi khi dọn dẹp: {e}')


if __name__ == "__main__":
    main()
