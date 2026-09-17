import os
import glob
import hashlib
from collections import Counter
from PIL import Image
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from src import config

def analyze_dataset(data_dir=config.DATA_DIR, save_plots=True):

    
    classes = config.CLASSES
    class_counts = {}
    image_paths = {c: [] for c in classes}
    corrupted_files = []
    formats = set()
    dimensions = []
    aspect_ratios = []
    md5_hashes = {}
    duplicates = []
    prefix_counts = Counter()

    total_images = 0

    for c in classes:
        c_dir = os.path.join(data_dir, c)
        if not os.path.isdir(c_dir):
            print(f"[WARNING] Directory missing for class {c}: {c_dir}")
            continue
        
        files = glob.glob(os.path.join(c_dir, "*"))
        valid_files = [f for f in files if os.path.isfile(f) and not os.path.basename(f).startswith('.')]
        class_counts[c] = len(valid_files)
        total_images += len(valid_files)
        image_paths[c] = valid_files

        for fpath in valid_files:
            fname = os.path.basename(fpath)
            prefix = fname.split('_')[0] if '_' in fname else fname.split('.')[0]
            prefix_counts[prefix] += 1
            
            try:
                with Image.open(fpath) as img:
                    formats.add(img.format)
                    w, h = img.size
                    dimensions.append((w, h))
                    aspect_ratios.append(w / h)
            except Exception as e:
                corrupted_files.append((fpath, str(e)))

            try:
                with open(fpath, "rb") as f:
                    file_hash = hashlib.md5(f.read()).hexdigest()
                    if file_hash in md5_hashes:
                        duplicates.append((fpath, md5_hashes[file_hash]))
                    else:
                        md5_hashes[file_hash] = fpath
            except Exception:
                pass

    max_count = max(class_counts.values()) if class_counts else 1
    min_count = min(class_counts.values()) if class_counts else 1
    imbalance_ratio = max_count / min_count

    print(f"Total Detected Images : {total_images}")
    print(f"Image Formats         : {list(formats)}")
    print(f"Corrupted Files       : {len(corrupted_files)}")
    print(f"Exact MD5 Duplicates  : {len(duplicates)}")
    print(f"Class Imbalance Ratio : {imbalance_ratio:.2f} (max/min)")
    print("\nClass Distribution:")
    for c, count in class_counts.items():
        pct = (count / total_images) * 100 if total_images > 0 else 0
        print(f"  - {c:<12}: {count:>5} images ({pct:.2f}%)")

    top_dims = Counter(dimensions).most_common(5)
    print(f"\nTop Image Dimensions (W, H): {top_dims}")
    print(f"Filename Prefix Patterns   : {prefix_counts.most_common(8)}")

    if save_plots:
        _plot_dataset_summary(class_counts, dimensions, image_paths)
        _plot_sample_grid(image_paths)

    stats = {
        "total_images": total_images,
        "class_counts": class_counts,
        "imbalance_ratio": imbalance_ratio,
        "corrupted_count": len(corrupted_files),
        "duplicate_count": len(duplicates),
        "formats": list(formats),
        "image_paths": image_paths,
    }
    print("==================================================\n")
    return stats

def _plot_dataset_summary(class_counts, dimensions, image_paths):
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))

    # Class counts bar chart
    classes = list(class_counts.keys())
    counts = list(class_counts.values())
    colors = sns.color_palette("viridis", len(classes))
    
    bars = axes[0].bar(classes, counts, color=colors, edgecolor='black', alpha=0.85)
    axes[0].set_title("Class Frequency Distribution", fontsize=14, fontweight='bold')
    axes[0].set_ylabel("Count", fontsize=12)
    axes[0].grid(axis='y', linestyle='--', alpha=0.6)
    
    for bar in bars:
        height = bar.get_height()
        axes[0].annotate(f'{height}',
                         xy=(bar.get_x() + bar.get_width() / 2, height),
                         xytext=(0, 3),  
                         textcoords="offset points",
                         ha='center', va='bottom', fontweight='bold')

    widths = [d[0] for d in dimensions]
    heights = [d[1] for d in dimensions]
    axes[1].scatter(widths, heights, alpha=0.3, color='crimson', edgecolors='none')
    axes[1].set_title("Image Resolution Distribution (Width vs Height)", fontsize=14, fontweight='bold')
    axes[1].set_xlabel("Width (px)", fontsize=12)
    axes[1].set_ylabel("Height (px)", fontsize=12)
    axes[1].grid(True, linestyle='--', alpha=0.6)

    plt.tight_layout()
    out_path = os.path.join(config.PLOTS_DIR, "dataset_analysis.png")
    plt.savefig(out_path, dpi=300, bbox_inches='tight')
    plt.close()
    print(f"[PLOT] Dataset analysis chart saved to {out_path}")

def _plot_sample_grid(image_paths, samples_per_class=4):
    classes = config.CLASSES
    fig, axes = plt.subplots(len(classes), samples_per_class, figsize=(3 * samples_per_class, 3 * len(classes)))

    for i, c in enumerate(classes):
        paths = image_paths.get(c, [])
        sample_paths = np.random.choice(paths, size=min(samples_per_class, len(paths)), replace=False)
        for j in range(samples_per_class):
            ax = axes[i, j]
            if j < len(sample_paths):
                img = Image.open(sample_paths[j]).convert('RGB')
                ax.imshow(img)
                ax.set_title(f"{c.capitalize()}", fontsize=11, fontweight='bold')
            ax.axis('off')

    plt.tight_layout()
    out_path = os.path.join(config.PLOTS_DIR, "dataset_samples.png")
    plt.savefig(out_path, dpi=300, bbox_inches='tight')
    plt.close()
    print(f"[PLOT] Representative MRI grid saved to {out_path}")

if __name__ == "__main__":
    analyze_dataset()
