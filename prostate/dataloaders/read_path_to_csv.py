import os
import glob
import pandas as pd
import re

def find_matching_pairs(data_dir):
    """
    Search for matching image and mask file pairs
    assumption：
    - image file：Case00.nii.gz, Case01.nii.gz, ...
    - Mask file：Case00_segmentation.nii.gz, Case01_segmentation.nii.gz, ...
    """
    # Obtain all .nii.gz files
    all_files = sorted(glob.glob(os.path.join(data_dir, "*.nii.gz")))
    
    # Create two lists to store the paths of the images and masks respectively
    case_dict = {}  # A mapping used to store case numbers and file paths
    
    for file_path in all_files:
        filename = os.path.basename(file_path)
        
        # Use regular expressions to extract the case numbers (such as 00, 01, etc.)
        match = re.match(r'Case(\d+)(_segmentation|_Segmentation)?\.nii\.gz', filename)
        
        if match:
            case_number = match.group(1)  # Extract the numerical part, such as "00"
            
            if '_segmentation' in filename.lower():
                # This is a mask file.
                case_dict.setdefault(case_number, {})['mask'] = '/'.join(file_path.split('/')[-2:])
            else:
                # This is an image file.
                case_dict.setdefault(case_number, {})['image'] = '/'.join(file_path.split('/')[-2:])
    
    # Build a pairing list
    pairs = []
    for case_number in sorted(case_dict.keys()):
        if 'image' in case_dict[case_number] and 'mask' in case_dict[case_number]:
            pairs.append({
                # 'case_id': f"Case{case_number}",
                'image': case_dict[case_number]['image'],
                'mask': case_dict[case_number]['mask']
            })
        else:
            print(f"warning: Case{case_number} Missing image or mask file")
    
    return pairs

def save_paths_to_csv(data_dir, output_csv='nifti_paths.csv'):
    """Search for the file and save it as CSV"""
    # Search for matching file pairs
    pairs = find_matching_pairs(data_dir)
    
    if not pairs:
        print(f"No matching image-mask pairs were found in the directory {data_dir}.")
        return None
    
    # Create DataFrame
    df = pd.DataFrame(pairs)
    
    # Save as CSV
    df.to_csv(output_csv, index=False)
    
    print(f"Successfully found {len(pairs)} pairs of files:")
    
    print(f"\n Saved to: {output_csv}")
    return df

# Usage example
if __name__ == "__main__":
    # Directly specify the data directory
    root = "mainproject_prostate/Processed_data_ni/"
    dir_names = [name for name in os.listdir(root)
             if os.path.isdir(os.path.join(root, name))]
    for dir_name in dir_names:
        data_directory = os.path.join(root, dir_name)
        print(data_directory)
        save_paths_to_csv(data_directory, f"mainproject_prostate/Processed_data_ni/{dir_name}.csv")