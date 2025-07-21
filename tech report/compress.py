import subprocess
import os
import sys

def compress_pdf(input_path, output_path, max_size_mb=18):
    max_size_bytes = max_size_mb * 1024 * 1024
    
    # Compression settings from light to heavy
    settings = [
        ['-dPDFSETTINGS=/prepress', '-dColorImageResolution=1000'],
    ]
    
    for i, setting in enumerate(settings):
        cmd = [
            'gs', '-dNOPAUSE', '-dBATCH', '-sDEVICE=pdfwrite',
            '-dCompatibilityLevel=1.4'
        ] + setting + ['-sOutputFile=' + output_path, input_path]
        
        try:
            subprocess.run(cmd, check=True, capture_output=True)
            file_size = os.path.getsize(output_path)
            size_mb = file_size / (1024 * 1024)
            
            print(f"Compression level {i+1}: {size_mb:.2f}MB")
            
            if file_size <= max_size_bytes:
                print(f"Success! Final file size: {size_mb:.2f}MB")
                return True
                
        except subprocess.CalledProcessError as e:
            print(f"Compression failed: {e}")
            continue
    
    print("Warning: Unable to compress to target size")
    return False

if __name__ == "__main__":
    if len(sys.argv) != 2:
        print("Usage: python compress_pdf.py <input_pdf>")
        sys.exit(1)
    
    input_file = sys.argv[1]
    output_file = input_file.replace('.pdf', '_18mb.pdf')
    
    compress_pdf(input_file, output_file)