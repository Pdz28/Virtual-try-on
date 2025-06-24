import os
import cv2
import numpy as np
import mediapipe as mp
from PIL import Image
import json
import sys
from pathlib import Path
import subprocess
import torch

# Add gen_image_parse directory to path if needed
sys.path.append(os.path.join(os.path.dirname(os.path.abspath(__file__)), 'gen_image_parse'))

def create_directories(base_dir="./data/test"):
    """Create necessary directories if they don't exist"""
    directories = [
        os.path.join(base_dir, "image"),
        os.path.join(base_dir, "image_parse"),
        os.path.join(base_dir, "cloth"),
        os.path.join(base_dir, "cloth_mask"),
        os.path.join(base_dir, "image_mask"),
        os.path.join(base_dir, "pose"),
       
    ]
    
    for directory in directories:
        if not os.path.exists(directory):
            os.makedirs(directory)
            print(f"Created directory: {directory}")
    
    return base_dir

def resize_image(img, width=192, height=256):
    """Resize image to target dimensions"""
    return cv2.resize(img, (width, height), interpolation=cv2.INTER_AREA)

def create_cloth_mask(cloth_img):
    """Create mask for clothing image using GrabCut"""
    # Convert to RGB if needed
    if len(cloth_img.shape) == 2:
        cloth_img = cv2.cvtColor(cloth_img, cv2.COLOR_GRAY2RGB)
    
    # Create initial mask
    mask = np.zeros(cloth_img.shape[:2], np.uint8)
    
    # Setup GrabCut parameters
    rect = (10, 10, cloth_img.shape[1]-20, cloth_img.shape[0]-20)
    bgd_model = np.zeros((1, 65), np.float64)
    fgd_model = np.zeros((1, 65), np.float64)
    
    # Apply GrabCut
    cv2.grabCut(cloth_img, mask, rect, bgd_model, fgd_model, 5, cv2.GC_INIT_WITH_RECT)
    
    # Create binary mask where 0 and 2 are background, 1 and 3 are foreground
    mask2 = np.where((mask==2)|(mask==0), 0, 1).astype('uint8')
    
    # Clean the mask with morphological operations
    kernel = np.ones((5,5), np.uint8)
    mask2 = cv2.morphologyEx(mask2, cv2.MORPH_OPEN, kernel)
    mask2 = cv2.morphologyEx(mask2, cv2.MORPH_CLOSE, kernel)
    
    # Multiply by 255 for display
    binary_cloth_mask = mask2 * 255
    
    return binary_cloth_mask

def create_person_mask(person_img):
    """Create mask for person image using GrabCut"""
    # Convert to RGB if needed
    if len(person_img.shape) == 2:
        person_img = cv2.cvtColor(person_img, cv2.COLOR_GRAY2RGB)
    
    # Create initial mask
    mask = np.zeros(person_img.shape[:2], np.uint8)
    
    # Setup GrabCut parameters - use a more conservative rectangle for person
    # Assume person is in center of image with some margin
    h, w = person_img.shape[:2]
    margin_x = int(w * 0.1)  # 10% margin on each side
    margin_y = int(h * 0.05)  # 5% margin on top and bottom
    rect = (margin_x, margin_y, w - 2*margin_x, h - 2*margin_y)
    
    bgd_model = np.zeros((1, 65), np.float64)
    fgd_model = np.zeros((1, 65), np.float64)
    
    # Apply GrabCut with more iterations for better segmentation
    cv2.grabCut(person_img, mask, rect, bgd_model, fgd_model, 8, cv2.GC_INIT_WITH_RECT)
    
    # Create binary mask where 0 and 2 are background, 1 and 3 are foreground
    mask2 = np.where((mask==2)|(mask==0), 0, 1).astype('uint8')
    
    # Clean the mask with morphological operations
    kernel = np.ones((5,5), np.uint8)
    mask2 = cv2.morphologyEx(mask2, cv2.MORPH_OPEN, kernel)
    mask2 = cv2.morphologyEx(mask2, cv2.MORPH_CLOSE, kernel)
    
    # Multiply by 255 for display
    binary_person_mask = mask2 * 255
    
    return binary_person_mask

def convert_mediapipe_to_openpose_format(image_path, output_path):
    """Convert MediaPipe pose estimation to OpenPose format"""
    # Initialize MediaPipe modules
    mp_pose = mp.solutions.pose
    
    # Read image
    image = cv2.imread(image_path)
    if image is None:
        print(f"Could not read image from {image_path}")
        return None
    
    # Initialize MediaPipe Pose
    with mp_pose.Pose(
        static_image_mode=True,
        model_complexity=2,
        enable_segmentation=True,
        min_detection_confidence=0.5) as pose:
        
        # Convert image to RGB (MediaPipe requires RGB input)
        image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        
        # Process image
        results = pose.process(image_rgb)
        
        if not results.pose_landmarks:
            print(f"No pose detected in image {image_path}")
            return None
        
        # Mapping from MediaPipe Pose to OpenPose COCO-18
        # MediaPipe has 33 landmarks, COCO-18 has 18 keypoints
        mediapipe_to_coco = {
            # Format: COCO_INDEX: MediaPipe_INDEX
            0: 0,    # Nose
            1: 11,   # Neck (MediaPipe: mid-shoulder)
            2: 12,   # Right Shoulder
            3: 14,   # Right Elbow
            4: 16,   # Right Wrist
            5: 11,   # Left Shoulder
            6: 13,   # Left Elbow
            7: 15,   # Left Wrist
            8: 24,   # Right Hip
            9: 26,   # Right Knee
            10: 28,  # Right Ankle
            11: 23,  # Left Hip
            12: 25,  # Left Knee
            13: 27,  # Left Ankle
            14: 2,   # Right Eye
            15: 5,   # Left Eye
            16: 8,   # Right Ear
            17: 7    # Left Ear
        }
        
        # Initialize pose_keypoints array
        pose_keypoints = []
        
        # Get image dimensions
        height, width, _ = image.shape
        
        # Process each keypoint
        for coco_idx in range(18):
            if coco_idx in mediapipe_to_coco:
                # Get corresponding mediapipe index
                mp_idx = mediapipe_to_coco[coco_idx]
                
                # Get the landmark
                landmark = results.pose_landmarks.landmark[mp_idx]
                
                # Convert normalized coordinates to pixel coordinates
                x = landmark.x * width
                y = landmark.y * height
                
                # Use visibility as confidence
                confidence = landmark.visibility
                
                # Add to pose_keypoints
                pose_keypoints.extend([float(x), float(y), float(confidence)])
            else:
                # For missing mappings, add zeros
                pose_keypoints.extend([0.0, 0.0, 0.0])
        
        # Create JSON in OpenPose format
        openpose_json = {
            "version": 1.0,
            "people": [{
                "face_keypoints": [],
                "pose_keypoints": pose_keypoints,
                "hand_right_keypoints": [],
                "hand_left_keypoints": []
            }]
        }
        
        # Save JSON file
        with open(output_path, 'w') as f:
            json.dump(openpose_json, f)
        
        print(f"JSON file saved to {output_path}")
        
        
        return openpose_json


def generate_parse_image(img_path, output_path, model_path=None):
    """Generate human parsing image using pre-trained model"""
    # Default model path if not provided
    if model_path is None:
        model_path = "checkpoints/inference.pth"
    
    # Ensure model path exists
    if not os.path.exists(model_path):
        print(f"Error: Model file not found at {model_path}")
        return None
    
    # Create output directory if needed
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    
    # If img_path is a numpy array, save it temporarily
    if isinstance(img_path, np.ndarray):
        temp_path = "temp_image_for_parse.jpg"
        cv2.imwrite(temp_path, img_path)
        img_path = temp_path
        temp_created = True
    else:
        temp_created = False
    
    # Get the filename without extension for output naming
    filename = Path(output_path).stem
    output_dir = str(Path(output_path).parent)
    
    try:
        # Import inference module
        from gen_image_parse.inference import inference
        from networks_f.deeplab_xception_transfer import deeplab_xception_transfer_projection_savemem
        
        # Load model
        net = deeplab_xception_transfer_projection_savemem(n_classes=20, hidden_layers=128, source_classes=7)
        x = torch.load(model_path, map_location=torch.device('cpu'))
        net.load_state_dict(x)
        
        # Check if CUDA is available
        use_gpu = torch.cuda.is_available()
        if use_gpu:
            net.cuda()
        
        # Run inference
        inference(net=net, img_path=img_path, output_path=output_dir, output_name=filename, use_gpu=use_gpu)
        
        # Return the path to the generated parse image
        parse_img_path = os.path.join(output_dir, f"{filename}.png")
        if os.path.exists(parse_img_path):
            parse_img = cv2.imread(parse_img_path)
        else:
            parse_img = None
            print(f"Warning: Parse image was not generated at {parse_img_path}")
        
        # Clean up temporary file
        if temp_created and os.path.exists(temp_path):
            os.remove(temp_path)
        
        return parse_img
        
    except ModuleNotFoundError:
        print("Could not import inference module. Using command line fallback.")
        # Fallback: Use command-line script if the module import fails
        cmd = [
            "python", 
            "gen_image_parse/inference.py",
            "--loadmodel", model_path,
            "--img_path", img_path,
            "--output_path", output_dir,
            "--output_name", filename
        ]
        
        try:
            subprocess.run(cmd, check=True)
            parse_img_path = os.path.join(output_dir, f"{filename}.png")
            
            # Clean up temporary file
            if temp_created and os.path.exists(temp_path):
                os.remove(temp_path)
                
            if os.path.exists(parse_img_path):
                return cv2.imread(parse_img_path)
            else:
                print(f"Warning: Parse image was not generated at {parse_img_path}")
                return None
        except subprocess.CalledProcessError as e:
            print(f"Error running parsing script: {e}")
            return None

def read_source_images(source_dir):
    """Read all images from source directory"""
    person_images = []
    cloth_images = []
    
    # Person images
    person_dir = os.path.join(source_dir, "person_images")
    if os.path.exists(person_dir):
        for file in os.listdir(person_dir):
            if file.lower().endswith(('.png', '.jpg', '.jpeg', '.webp')):
                person_images.append(os.path.join(person_dir, file))
    
    # Clothing images
    cloth_dir = os.path.join(source_dir, "cloth_images")
    if os.path.exists(cloth_dir):
        for file in os.listdir(cloth_dir):
            if file.lower().endswith(('.png', '.jpg', '.jpeg', '.webp')):
                cloth_images.append(os.path.join(cloth_dir, file))
    
    return person_images, cloth_images

def create_test_pairs(person_images, cloth_images, output_file="./data/test_pairs.txt"):
    """Create test_pairs.txt file to pair person and clothing images"""
    os.makedirs(os.path.dirname(output_file), exist_ok=True)
    
    with open(output_file, 'w') as f:
        for person_img in person_images:
            person_name = os.path.splitext(os.path.basename(person_img))[0]
            for cloth_img in cloth_images:
                cloth_name = os.path.splitext(os.path.basename(cloth_img))[0]
                f.write(f"{person_name}.jpg {cloth_name}.jpg\n")
    
    print(f"Test pairs file created: {output_file}")

def ensure_valid_image_format(input_dir):
    """
    Kiểm tra và chuyển đổi tất cả ảnh trong thư mục sang định dạng JPG
    nếu chúng không phải là JPG hoặc PNG
    
    Args:
        input_dir: Thư mục chứa ảnh cần kiểm tra
    
    Returns:
        list: Danh sách các file đã được chuyển đổi
    """
    converted_files = []
    valid_extensions = ['.jpg', '.jpeg', '.png']
    
    # Các thư mục cần kiểm tra
    check_dirs = [
        os.path.join(input_dir, "person_images"),
        os.path.join(input_dir, "cloth_images")
    ]
    
    print("Kiểm tra định dạng ảnh đầu vào...")
    
    for dir_path in check_dirs:
        if not os.path.exists(dir_path):
            print(f"Thư mục {dir_path} không tồn tại, bỏ qua")
            continue
        
        print(f"Kiểm tra thư mục: {dir_path}")
        
        # Lấy danh sách tất cả các file trong thư mục
        files = [f for f in os.listdir(dir_path) if os.path.isfile(os.path.join(dir_path, f))]
        
        for file in files:
            file_path = os.path.join(dir_path, file)
            file_ext = os.path.splitext(file)[1].lower()
            
            # Nếu không phải JPG hoặc PNG, chuyển đổi sang JPG
            if file_ext not in valid_extensions:
                print(f"Phát hiện file không hỗ trợ: {file_path}")
                try:
                    # Thử đọc bằng PIL
                    from PIL import Image
                    try:
                        img = Image.open(file_path)
                        
                        # Tạo tên file mới với đuôi .jpg
                        new_path = os.path.splitext(file_path)[0] + ".jpg"
                        
                        # Đảm bảo sử dụng chế độ RGB cho ảnh JPG
                        if img.mode != 'RGB':
                            img = img.convert('RGB')
                            
                        # Lưu ảnh với định dạng JPG
                        img.save(new_path, "JPEG", quality=95)
                        print(f"Đã chuyển đổi thành công: {file_path} -> {new_path}")
                        
                        # Xóa file gốc nếu chuyển đổi thành công
                        os.remove(file_path)
                        
                        converted_files.append((file_path, new_path))
                    except Exception as img_error:
                        print(f"Lỗi khi xử lý ảnh với PIL: {str(img_error)}")
                        
                        # Thử dùng OpenCV nếu PIL thất bại
                        try:
                            img = cv2.imread(file_path)
                            if img is not None:
                                new_path = os.path.splitext(file_path)[0] + ".jpg"
                                cv2.imwrite(new_path, img, [cv2.IMWRITE_JPEG_QUALITY, 95])
                                os.remove(file_path)
                                print(f"Đã chuyển đổi bằng OpenCV: {file_path} -> {new_path}")
                                converted_files.append((file_path, new_path))
                            else:
                                print(f"Không thể đọc ảnh bằng OpenCV: {file_path}")
                        except Exception as cv_error:
                            print(f"Lỗi khi xử lý ảnh với OpenCV: {str(cv_error)}")
                
                except ImportError:
                    print("Không thể import PIL. Vui lòng cài đặt: pip install pillow")
            else:
                # Kiểm tra xem file có phải là ảnh hợp lệ không
                try:
                    img = cv2.imread(file_path)
                    if img is None:
                        print(f"Cảnh báo: File {file_path} có đuôi hợp lệ nhưng không thể đọc được")
                        
                        # Thử đọc bằng PIL
                        try:
                            from PIL import Image
                            img_pil = Image.open(file_path)
                            # Nếu đọc được, chuyển đổi để đảm bảo định dạng chuẩn
                            new_path = file_path  # Giữ nguyên tên nếu đuôi đã hợp lệ
                            if img_pil.mode != 'RGB':
                                img_pil = img_pil.convert('RGB')
                            img_pil.save(new_path, quality=95)
                            print(f"Đã sửa chữa file ảnh: {file_path}")
                        except Exception as e:
                            print(f"Không thể sửa chữa file ảnh: {str(e)}")
                except Exception:
                    print(f"Lỗi khi kiểm tra file {file_path}")
    
    return converted_files

def process_and_prepare_data(source_dir="./input", output_dir="./data/test"):
    """Process images and prepare data"""
    # Ensure valid image formats
    ensure_valid_image_format(source_dir)
    # Create directories
    output_dir = create_directories(output_dir)
    
    # Read images
    person_images_paths, cloth_images_paths = read_source_images(source_dir)
    
    # Check person images
    if not person_images_paths:
        print("No person images found. Please create 'person_images' directory and add images.")
        return
    
    # Check clothing images
    if not cloth_images_paths:
        print("No clothing images found. Please create 'cloth_images' directory and add images.")
        return
    
    # Process person images
    processed_person_filenames = []
    for img_path in person_images_paths:
        filename = os.path.splitext(os.path.basename(img_path))[0]
        processed_person_filenames.append(filename)
        
        # Read and resize
        img = cv2.imread(img_path)
        if img is None:
            print(f"Could not read image: {img_path}")
            continue
            
        resized_img = resize_image(img)
        
        # Save resized image
        person_output_path = os.path.join(output_dir, "image", f"{filename}.jpg")
        os.makedirs(os.path.dirname(person_output_path), exist_ok=True)
        cv2.imwrite(person_output_path, resized_img)
        print(f"Processed person image: {person_output_path}")
        
        # Generate keypoints JSON using MediaPipe
        pose_output_path = os.path.join(output_dir, "pose", f"{filename}_keypoints.json")
        os.makedirs(os.path.dirname(pose_output_path), exist_ok=True)  # Đảm bảo thư mục tồn tại
        temp_img_path = os.path.join(os.path.dirname(output_dir), "temp_resized.jpg")
        cv2.imwrite(temp_img_path, resized_img)
        img_resized_path = temp_img_path  # Lưu đường dẫn tạm thời

        # Sử dụng đường dẫn tạm thời này để gọi hàm
        pose_data = convert_mediapipe_to_openpose_format(img_resized_path, pose_output_path)

        # Xóa file tạm sau khi sử dụng xong
        if os.path.exists(temp_img_path):
            os.remove(temp_img_path)
                
        if pose_data is not None:
            print(f"Pose data created: {pose_output_path}")
        else:
            # Tạo dữ liệu pose rỗng nếu không nhận diện được
            empty_pose_data = {
                "version": 1.0,
                "people": [{
                    "face_keypoints": [],
                    "pose_keypoints": [0.0] * 54,  # 18 keypoints, mỗi keypoint có 3 giá trị (x, y, confidence)
                    "hand_right_keypoints": [],
                    "hand_left_keypoints": []
                }]
            }
            
            # Lưu pose dữ liệu rỗng
            with open(pose_output_path, 'w') as f:
                json.dump(empty_pose_data, f)
            print(f"Empty pose data created: {pose_output_path}")
            
        
        # Generate and save parse image
        parse_output_path = os.path.join(output_dir, "image_parse", f"{filename}.png")
        parse_img = generate_parse_image(resized_img, parse_output_path)
        if parse_img is not None:
            print(f"Parse image created: {parse_output_path}")

        # Generate and save person mask
        person_mask = create_person_mask(parse_img)
        mask_output_path = os.path.join(output_dir, "image_mask", f"{filename}.png")
        os.makedirs(os.path.dirname(mask_output_path), exist_ok=True)
        cv2.imwrite(mask_output_path, person_mask)
        print(f"Person mask created: {mask_output_path}")
    
    # Process clothing images
    processed_cloth_filenames = []
    for img_path in cloth_images_paths:
        filename = os.path.splitext(os.path.basename(img_path))[0]
        processed_cloth_filenames.append(filename)
        
        # Read and resize
        img = cv2.imread(img_path)
        if img is None:
            print(f"Could not read image: {img_path}")
            continue
            
        resized_img = resize_image(img)
        
        # Save resized clothing image
        cloth_output_path = os.path.join(output_dir, "cloth", f"{filename}.jpg")
        os.makedirs(os.path.dirname(cloth_output_path), exist_ok=True)
        cv2.imwrite(cloth_output_path, resized_img)
        print(f"Processed clothing image: {cloth_output_path}")
        
        # Create and save clothing mask
        cloth_mask = create_cloth_mask(resized_img)
        mask_output_path = os.path.join(output_dir, "cloth_mask", f"{filename}.jpg")
        os.makedirs(os.path.dirname(mask_output_path), exist_ok=True)
        cv2.imwrite(mask_output_path, cloth_mask)
        print(f"Clothing mask created: {mask_output_path}")
    
    # Create test_pairs.txt
    create_test_pairs(processed_person_filenames, processed_cloth_filenames)
    
    print("Data preparation complete!")

if __name__ == "__main__":
    # Import torch only if needed to avoid dependency issues
    try:
        import torch
    except ImportError:
        print("PyTorch not found. Using command line for parsing generation.")
    
    # Create source directories if they don't exist
    if not os.path.exists("./input/person_images"):
        os.makedirs("./input/person_images")
    if not os.path.exists("./input/cloth_images"):
        os.makedirs("./input/cloth_images")
        
    print("Please place person images in ./input/person_images")
    print("Please place clothing images in ./input/cloth_images")
    
    # Check if source directories have images
    person_dir = "./input/person_images"
    cloth_dir = "./input/cloth_images"
    
    # Kiểm tra và chuyển đổi định dạng ảnh trước khi xử lý
    ensure_valid_image_format("./input")
    
    if (os.path.exists(person_dir) and len([f for f in os.listdir(person_dir) if f.lower().endswith(('.png', '.jpg', '.jpeg'))])) and \
       (os.path.exists(cloth_dir) and len([f for f in os.listdir(cloth_dir) if f.lower().endswith(('.png', '.jpg', '.jpeg'))])):
        process_and_prepare_data()
    else:
        print("No images found in source directories. Please add images and run again.")