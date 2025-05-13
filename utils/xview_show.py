import os
import numpy as np
import matplotlib.pyplot as plt
from PIL import Image, ImageDraw
import glob
import re

plt.rcParams['font.family'] = 'STHeiti'
plt.rcParams['axes.unicode_minus'] = False  # Fix negative sign display issue

def load_fragments(train_folder):
    """Load all subimages starting with 5__1__ and correctly parse x, y coordinates"""
    fragments = {}
    # Regex pattern to match 5__1__{x}___{y}.png format
    # First capture group (\d+) is x coordinate
    # Second capture group (\d+) is y coordinate
    pattern = re.compile(r'5__1__(\d+)___(\d+)\.png') 
    
    print(f"Searching for files matching '5__1__*___*.png' in {train_folder}...")
    found_files = glob.glob(os.path.join(train_folder, '5__1__*.png'))
    print(f"Found {len(found_files)} potentially matching files.")

    for img_path in found_files:
        img_name = os.path.basename(img_path)
        match = pattern.match(img_name)
        if match:
            # First number is x, second is y
            x = int(match.group(1)) 
            y = int(match.group(2))
            try:
                img = Image.open(img_path)
                # Store key as (x, y) for later use
                fragments[(x, y)] = img 
                print(f"  Loaded fragment: {img_name} -> coordinates x={x}, y={y}")
            except Exception as e:
                print(f"  Cannot load image {img_name}: {e}")
        else:
             print(f"  Filename {img_name} does not match pattern 5__1__(\\d+)___(\\d+).png")
    
    if not fragments:
        print(f"Warning: No fragments matching '5__1__*___*.png' found or loaded in {train_folder}.")
        print("Please check 'train' directory contents and filename format.")
        # List some .png files for debugging
        all_pngs = glob.glob(os.path.join(train_folder, '*.png'))
        print(f"Sample PNG files in directory: {all_pngs[:5]}")

    return fragments

def reconstruct_image(fragments, overlap=200):
    """Reconstruct complete image from fragments"""
    if not fragments:
        print("No fragments found")
        return None
    
    # Determine reconstructed image dimensions
    # fragments keys are (x, y)
    max_x = max(pos[0] for pos in fragments.keys())
    max_y = max(pos[1] for pos in fragments.keys())
    
    # Get dimensions of first fragment (assuming all fragments are same size)
    try:
        first_img = next(iter(fragments.values()))
        frag_width, frag_height = first_img.size
    except StopIteration:
        print("Error: Cannot get fragment dimensions because fragments is empty.")
        return None
    except Exception as e:
        print(f"Error: Failed to get fragment dimensions: {e}")
        return None

    full_width = max_x + frag_width
    full_height = max_y + frag_height
    
    print(f"Reconstructed image dimensions: width={full_width}, height={full_height}")
    print(f"Based on max_x={max_x}, max_y={max_y}, frag_width={frag_width}, frag_height={frag_height}")

    # Create blank canvas
    reconstructed = Image.new('RGB', (full_width, full_height))
    
    # Place each fragment
    for (x, y), img in fragments.items():
        try:
            reconstructed.paste(img, (x, y))
        except Exception as e:
            print(f"Error: Failed to paste fragment (x={x}, y={y}): {e}")

    return reconstructed

def load_labels(label_path):
    """Load label file"""
    boxes = []
    if os.path.exists(label_path):
        with open(label_path, 'r') as f:
            for line in f:
                parts = line.strip().split()
                if len(parts) >= 5:
                    try:
                        class_id = int(parts[0])
                        x, y, w, h = map(float, parts[1:5])
                        boxes.append((class_id, x, y, w, h))
                    except ValueError:
                        print(f"Warning: Could not parse line in label file {label_path}: {line.strip()}")
    return boxes

def load_fragment_labels(label_folder, fragment_keys):
    """Load corresponding label files for each fragment"""
    fragment_labels = {}
    print(f"Searching for label files in {label_folder}...")
    
    # fragment_keys are (x, y) coordinate tuples
    for x, y in fragment_keys:
        # Label filename format should also be 5__1__{x}___{y}.txt
        label_file = os.path.join(label_folder, f"5__1__{x}___{y}.txt")
        if os.path.exists(label_file):
            try:
                labels = load_labels(label_file)
                fragment_labels[(x, y)] = labels
                print(f"  Loaded label file: {os.path.basename(label_file)} -> {len(labels)} annotations for (x={x}, y={y})")
            except Exception as e:
                 print(f"  Cannot load label file {os.path.basename(label_file)}: {e}")
        else:
            pass
            
    print(f"Loaded labels for {len(fragment_labels)} fragments.")
    return fragment_labels

def draw_boxes(image, labels, color='cyan', width=1):
    """Draw bounding boxes on the image"""
    draw = ImageDraw.Draw(image)
    img_width, img_height = image.size
    
    for class_id, x, y, w, h in labels:
        # Convert to pixel coordinates
        x1 = int((x - w/2) * img_width)
        y1 = int((y - h/2) * img_height)
        x2 = int((x + w/2) * img_width)
        y2 = int((y + h/2) * img_height)
        
        # Use the provided color tuple or string
        draw.rectangle([x1, y1, x2, y2], outline=color, width=width)
    
    return image

def show_overlay_grid(original_image_with_boxes, reconstructed_image, fragments, fragment_labels, overlap=200):
    """Display comparison of original and reconstructed images with fragment boundaries and target boxes"""
    # Create figure and axes explicitly
    fig, axes = plt.subplots(1, 2, figsize=(15, 7)) # Use figsize from main
    ax1, ax2 = axes.ravel()

    # Show original image with bounding boxes
    if original_image_with_boxes:
        ax1.imshow(np.array(original_image_with_boxes))
        ax1.set_title("Original Image with Bounding Boxes")
    else:
        ax1.set_title("Original Image (Not Loaded)")
    ax1.axis('off')

    # Show reconstructed image
    handles, labels = [], [] # Initialize legend items list
    if reconstructed_image:
        ax2.imshow(np.array(reconstructed_image)) # Re-enabled reconstructed image display
        ax2.set_title("Sliced Image with Bounding Boxes") # Updated title

        # Keep track if legend items have been added
        added_legend = {'Slice Boundary': False, 'Horizontal Overlap': False, 'Vertical Overlap': False, 'Bounding Box': False}

        # Draw fragment boundaries and target boxes
        for (x, y), img in fragments.items():
            width, height = img.size
            # Draw fragment boundary (red)
            label_boundary = None
            if not added_legend['Slice Boundary']:
                label_boundary = 'Slice Boundary'
                added_legend['Slice Boundary'] = True
            rect = plt.Rectangle((x, y), width, height, linewidth=1, edgecolor='r', facecolor='none', label=label_boundary)
            ax2.add_patch(rect)
            if label_boundary: handles.append(rect)

            # Add text to identify fragment
            ax2.text(x + width/2, y + height/2, f'({x},{y})',
                     color='Yellow', fontsize=10, ha='center', va='center')

            # Draw overlap regions - using transparent colors
            if x > 0:  # Left overlap (yellow)
                label_overlap_h = None
                if not added_legend['Horizontal Overlap']:
                    label_overlap_h = 'Horizontal Overlap'
                    added_legend['Horizontal Overlap'] = True
                rect_overlap_x = plt.Rectangle((x, y), overlap, height, linewidth=1, edgecolor='yellow', facecolor='yellow', alpha=0.2, label=label_overlap_h)
                ax2.add_patch(rect_overlap_x)
                if label_overlap_h: handles.append(rect_overlap_x)

            if y > 0:  # Top overlap (green)
                label_overlap_v = None
                if not added_legend['Vertical Overlap']:
                     label_overlap_v = 'Vertical Overlap'
                     added_legend['Vertical Overlap'] = True
                rect_overlap_y = plt.Rectangle((x, y), width, overlap, linewidth=1, edgecolor='green', facecolor='green', alpha=0.2, label=label_overlap_v)
                ax2.add_patch(rect_overlap_y)
                if label_overlap_v: handles.append(rect_overlap_y)

            # Draw target boxes in this fragment (cyan, width=1)
            if (x, y) in fragment_labels:
                for i, (class_id, box_x, box_y, box_w, box_h) in enumerate(fragment_labels[(x, y)]):
                    box_abs_x = box_x * width
                    box_abs_y = box_y * height
                    box_abs_w = box_w * width
                    box_abs_h = box_h * height
                    box_x1_local = box_abs_x - box_abs_w / 2
                    box_y1_local = box_abs_y - box_abs_h / 2
                    box_x1_global = x + box_x1_local
                    box_y1_global = y + box_y1_local
                    
                    label_bbox = None
                    if not added_legend['Bounding Box']:
                        label_bbox = 'Bounding Box'
                        added_legend['Bounding Box'] = True
                    # Ensure linewidth is 1 for fragment bounding boxes
                    rect_label = plt.Rectangle((box_x1_global, box_y1_global), box_abs_w, box_abs_h,
                                        linewidth=0.5, edgecolor='cyan', facecolor='none', label=label_bbox) 
                    ax2.add_patch(rect_label)
                    if label_bbox: handles.append(rect_label)
        
        # Get labels from handles for the legend
        labels = [h.get_label() for h in handles]

    else:
        ax2.set_title("Reconstructed Image (Not Generated)") # Keep this title if reconstruction failed
    ax2.axis('off')

    # Don't call tight_layout or legend here, do it in main
    return fig, handles, labels # Return handles and labels for external legend


def main():
    # File paths
    current_dir = os.path.dirname(os.path.abspath(__file__))
    train_folder = os.path.join(current_dir, "train")  # train directory
    labels_folder = os.path.join(current_dir, "label")  # label directory
    original_img_path = os.path.join(current_dir, '5.png')  # original image at same level as train/label
    original_label_path = os.path.join(current_dir, '5.txt')  # original label at same level as image
    
    original_image_with_boxes = None 
    original_image = None
    # Load original image and labels
    if os.path.exists(original_img_path):
        try:
            original_image = Image.open(original_img_path)
            print(f"Loaded original image: {original_img_path}")
            original_image_with_boxes = original_image.copy() 
        except Exception as e:
            print(f"Error: Cannot load original image {original_img_path}: {e}")
    else:
        print(f"Original image not found: {original_img_path}")
        
    original_labels = load_labels(original_label_path)
    if original_labels:
        print(f"Loaded original labels, {len(original_labels)} objects total")
        if original_image_with_boxes: 
            try:
                # Draw boxes on original image with CYAN color and width 1
                original_image_with_boxes = draw_boxes(original_image_with_boxes, original_labels, color='cyan', width=5) 
            except Exception as e:
                 print(f"Error: Failed to draw bounding boxes on original image: {e}")
    else:
        print(f"Original label file not found or could not be loaded: {original_label_path}")

    # Load and reconstruct fragments
    fragments = load_fragments(train_folder)
    print(f"Successfully loaded {len(fragments)} fragments")
    
    # Load labels corresponding to fragments
    fragment_labels = load_fragment_labels(labels_folder, fragments.keys())

    reconstructed_image = None 
    if fragments:
        reconstructed_image = reconstruct_image(fragments, overlap=200)
        if reconstructed_image:
             print("Image reconstruction successful.")
        else:
             print("Image reconstruction failed.")
    else:
        print("No fragments loaded, cannot reconstruct image.")

    # Display results
    print("Preparing to display images...")
    try:
        # Create the figure using show_overlay_grid, get handles/labels back
        fig, handles, labels = show_overlay_grid(
            original_image_with_boxes, 
            reconstructed_image, 
            fragments, 
            fragment_labels
        )
        
        # Add the legend at the bottom, horizontally
        if handles and labels: # Check if there's anything to add to the legend
            fig.legend(handles, labels, loc='lower center', ncol=len(handles), bbox_to_anchor=(0.5, 0.01)) # Adjust y in bbox_to_anchor if needed
            # Adjust layout to prevent legend overlapping plots
            plt.subplots_adjust(bottom=0.15) # Increase bottom margin
        else:
             plt.tight_layout() # Use tight_layout if no legend

        output_comparison_path = os.path.join(current_dir, 'split_comparison.png')
        plt.savefig(output_comparison_path, bbox_inches='tight', dpi=300)
        print(f"Comparison image saved to: {output_comparison_path}")
        
        #plt.show() 
        print("Image display completed.")

    except Exception as e:
        print(f"Error: Failed to display or save image: {e}")

    if reconstructed_image:
        try:
            output_reconstructed_path = os.path.join(current_dir, 'reconstructed_5.png')
            reconstructed_image.save(output_reconstructed_path)
            print(f"Reconstructed image saved to: {output_reconstructed_path}")
        except Exception as e:
            print(f"Error: Failed to save reconstructed image: {e}")
    else:
        print("No reconstructed image to save.")

if __name__ == "__main__":
    main()