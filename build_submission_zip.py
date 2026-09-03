import os
import zipfile

def create_submission_zip():
    zip_filename = "sonnyariady-submission.zip"
    
    # List of files and directories to include in zip root
    files_to_include = [
        "sonnyariady-submission.ipynb",
        "sonnyariady-testing.ipynb",
        "transform.py",
        "tuner.py",
        "trainer.py",
        "format-dokumentasi.md",
        "README.md",
        "requirements.txt",
        "Dockerfile"
    ]
    
    dirs_to_include = [
        "sonnyariady-pipeline",
        "serving_model",
        "data"
    ]
    
    print(f"Creating submission zip: {zip_filename}...")
    with zipfile.ZipFile(zip_filename, 'w', zipfile.ZIP_DEFLATED) as zipf:
        # Include individual files
        for filename in files_to_include:
            if os.path.exists(filename):
                zipf.write(filename, arcname=filename)
                print(f"  Added file: {filename}")
            else:
                print(f"  WARNING: File {filename} not found!")
                
        # Include directories recursively
        for dir_name in dirs_to_include:
            if os.path.exists(dir_name):
                for root, dirs, files in os.walk(dir_name):
                    # Skip __pycache__ or temporary cache dirs
                    if '__pycache__' in root or '.ipynb_checkpoints' in root:
                        continue
                    for file in files:
                        file_path = os.path.join(root, file)
                        zipf.write(file_path, arcname=file_path)
                print(f"  Added directory: {dir_name}")
            else:
                print(f"  WARNING: Directory {dir_name} not found!")

    size_mb = os.path.getsize(zip_filename) / (1024 * 1024)
    print(f"Zip created successfully: {zip_filename} ({size_mb:.2f} MB)")

if __name__ == '__main__':
    create_submission_zip()
