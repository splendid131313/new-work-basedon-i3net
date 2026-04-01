import numpy as np
from pathlib import Path
from PIL import Image


def normalize_to_uint8(data):
    """
    将数据归一化到0-255范围（uint8）
    
    参数:
        data: 输入numpy数组
    
    返回:
        归一化后的uint8数组
    """
    data = data.astype(np.float64)
    data_min = data.min()
    data_max = data.max()
    
    if data_min >= 0 and data_max <= 255:
        return np.clip(data, 0, 255).astype(np.uint8)
    
    # 如果数据范围超出0-255，进行归一化
    if data_max > data_min:
        data = (data - data_min) / (data_max - data_min) * 255
    else:
        # 所有值都相同的情况
        data = np.zeros_like(data)
    
    return data.astype(np.uint8)


def save_slice_as_image(slice_data, output_path, image_format='PNG', normalize=True):
    """
    将切片保存为图片
    
    参数:
        slice_data: 2D numpy数组
        output_path: 输出文件路径
        image_format: 图片格式（PNG, JPEG等）
        normalize: 是否归一化数据
    """
    if normalize:
        slice_data = normalize_to_uint8(slice_data)
    else:
        if slice_data.dtype != np.uint8:
            slice_data = np.clip(slice_data, 0, 255).astype(np.uint8)
    
    # 创建PIL图像
    image = Image.fromarray(slice_data, mode='L')  # 'L'表示灰度图
    image.save(output_path, format=image_format)


def slice_volume(npy_file, output_dir=None, save_slices=True, save_images=True, image_format='PNG'):
    """
    从3D体积中按 XY 方向切片
    
    参数:
        npy_file: .npy 文件路径
        output_dir: 输出目录；如果为 None，会在数据所在目录下的 `xy/<volume_name>` 中保存
        save_slices: 是否保存 .npy 切片文件
        save_images: 是否保存图片格式的切片
        image_format: 图片格式（PNG, JPEG 等）
    
    返回:
        xy_slices: XY 方向切片列表（沿 z 轴）
    """
    # 加载3D体积数据
    volume = np.load(npy_file)
    print(f"load: {npy_file}")
    print(f"shape: {volume.shape}")

    # 如果是4D数据，取第2个通道/时间点（索引1）
    if volume.ndim == 4:
        volume = volume[:, :, :, 1]
        print(f"4D -> 3D by [:,:,:,1], new shape: {volume.shape}")
    
    # 假设体积形状为 (depth, height, width) 或 (z, y, x)
    if len(volume.shape) != 3:
        raise ValueError(f"hopes 3D array, but got shape: {volume.shape}")
    
    x_size, y_size, z_size = volume.shape
    print(f"size: X={x_size}, Y={y_size}, Z={z_size}")
    
    # XY方向切片（沿z轴，固定z值，得到xy平面）
    xy_slices = []
    for z_idx in range(z_size):
        xy_slice = volume[:, :, z_idx]
        xy_slices.append(xy_slice)
    
    print(f"xy slices number: {len(xy_slices)}")

    npy_path = Path(npy_file)
    if output_dir is None:
        base_xy_dir = npy_path.parent / "xy"
        output_dir = base_xy_dir / npy_path.stem
    else:
        output_dir = Path(output_dir)
    
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # 保存XY切片
    xy_dir = output_dir
    xy_dir.mkdir(exist_ok=True)
    for idx, slice_data in enumerate(xy_slices):
        if save_slices:
            slice_file = xy_dir / f"{idx:04d}.npy"
            np.save(slice_file, slice_data)
        if save_images:
            image_file = xy_dir / f"{idx:04d}.{image_format.lower()}"
            save_slice_as_image(slice_data, image_file, image_format=image_format)
    
    saved_types = []
    if save_slices:
        saved_types.append(".npy")
    if save_images:
        saved_types.append(f"{image_format}")
    print(f"Saved to: {output_dir} ({', '.join(saved_types)})")
    
    return xy_slices


def batch_slice_volumes(input_dir, output_base_dir=None, pattern="*.npy", 
                       save_slices=True, save_images=True, image_format='PNG'):
    """
    批量处理目录中的所有 .npy 文件
    
    参数:
        input_dir: 输入目录
        output_base_dir: 输出基础目录；如果为 None，则为 input_dir/xy
        pattern: 文件匹配模式
        save_slices: 是否保存 .npy 切片文件
        save_images: 是否保存图片格式的切片
        image_format: 图片格式（PNG, JPEG 等）
    """
    input_dir = Path(input_dir)
    if output_base_dir is None:
        output_base_dir = input_dir.parent / "xy"
    else:
        output_base_dir = Path(output_base_dir)
    
    npy_files = list(input_dir.glob(pattern))
    print(f"Find {len(npy_files)}  .npy files")
    
    for npy_file in npy_files:
        print(f"\nSlicing: {npy_file.name}")
        try:
            output_dir = output_base_dir / npy_file.stem
            slice_volume(npy_file, output_dir, save_slices=save_slices,
                            save_images=save_images, image_format=image_format)
        except Exception as e:
            print(f"Slicing {npy_file.name} went error: {e}")
            continue
    
    print("\nDone!")


if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description="从3D体积中按 XY 方向切片（只切 XY）")
    parser.add_argument("input", help="输入.npy文件路径或目录")
    parser.add_argument("-o", "--output", help="输出目录", default=None)
    parser.add_argument("-b", "--batch", action="store_true", 
                       help="批量处理模式（输入为目录）")
    parser.add_argument("--no-npy", action="store_true", 
                       help="不保存.npy文件，只保存图片")
    parser.add_argument("--no-images", action="store_true", 
                       help="不保存图片，只保存.npy文件")
    parser.add_argument("--format", choices=['PNG', 'JPEG', 'TIFF'], 
                       default='PNG', help="图片格式（默认：PNG）")
    
    args = parser.parse_args()
    
    save_slices = not args.no_npy
    save_images = not args.no_images
    
    if args.batch:
        # 批量处理模式
        batch_slice_volumes(args.input, args.output, save_slices=save_slices, 
                           save_images=save_images, image_format=args.format)
    else:
        # 单文件处理模式
        slice_volume(args.input, args.output, save_slices=save_slices, 
                        save_images=save_images, image_format=args.format)

