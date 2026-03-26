import numpy as np
import nibabel as nib
import os
from pathlib import Path


def _get_npy_name_from_nii(path_obj: Path, out_dir: Path) -> Path:
    """
    根据 .nii / .nii.gz 文件生成对应的 .npy 路径：
    - x.nii    -> x.npy
    - x.nii.gz -> x.npy
    其它后缀则退回到 stem.npy
    """
    name = path_obj.name
    if name.endswith(".nii.gz"):
        base = name[: -len(".nii.gz")]
    elif name.endswith(".nii"):
        base = name[: -len(".nii")]
    else:
        base = path_obj.stem
    return out_dir / f"{base}.npy"

def convert_nii_to_npy(input_path, output_path=None):
    """
    将.nii文件转换为.npy格式

    参数:
        input_path: 输入的.nii文件路径
        output_path: 输出的.npy文件路径（如果为None，则自动生成）

    返回:
        转换后的numpy数组
    """
    # 读取.nii文件
    print(f"Loading file: {input_path}")
    nii_img = nib.load(input_path)
    data = nii_img.get_fdata()

    # 检查原始尺寸
    original_shape = data.shape
    print(f"Original shape: {original_shape}")

    # 生成输出路径（正确处理 .nii / .nii.gz）
    if output_path is None:
        input_path_obj = Path(input_path)
        output_path = _get_npy_name_from_nii(input_path_obj)

    # 保存为.npy文件
    print(f"Saving to: {output_path}")
    np.save(output_path, data)

    return data


def batch_convert(input_dir, output_dir=None, pattern="auto"):
    """
    批量转换目录中的所有.nii/.nii.gz 文件

    参数:
        input_dir: 输入目录路径
        output_dir: 输出目录路径（如果为None，则在输入目录中创建npy_output文件夹）
        pattern: 文件匹配模式；
                 - 默认 \"auto\"：同时匹配 *.nii 和 *.nii.gz
                 - 也可以传入如 \"*.nii\"、\"*.nii.gz\" 等自定义模式
    """
    input_path = Path(input_dir)

    if output_dir is None:
        output_path = input_path / "npy_output"
    else:
        output_path = Path(output_dir)

    output_path.mkdir(parents=True, exist_ok=True)

    # 查找所有 .nii / .nii.gz 文件
    if pattern == "auto":
        nii_files = list(input_path.glob("*.nii")) + list(input_path.glob("*.nii.gz"))
    else:
        nii_files = list(input_path.glob(pattern))

    if len(nii_files) == 0:
        print(f"No files found in {input_dir} matching {pattern}")
        return

    print(f"Found {len(nii_files)} .nii/.nii.gz files")

    for nii_file in nii_files:
        print(f"Processing file: {nii_file.name}")

        # 正确处理 .nii / .nii.gz 输出名
        output_file = _get_npy_name_from_nii(nii_file, output_path)
        try:
            convert_nii_to_npy(str(nii_file), str(output_file))
        except Exception as e:
            print(f"Error processing {nii_file.name}: {str(e)}")
            continue


if __name__ == "__main__":
    import sys

    if len(sys.argv) > 1:
        input_file = sys.argv[1]
        output_file = sys.argv[2] if len(sys.argv) > 2 else None

        if os.path.isfile(input_file):
            convert_nii_to_npy(input_file, output_file)
        elif os.path.isdir(input_file):
            batch_convert(input_file, output_file)
   

        
