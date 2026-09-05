import os
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.patches as patches

docs_dir = os.path.dirname(os.path.abspath(__file__))

def create_dfd_level0():
    fig, ax = plt.subplots(figsize=(10, 5), dpi=150)
    fig.patch.set_facecolor('#ffffff')
    ax.set_facecolor('#ffffff')
    ax.axis('off')

    plt.title("Data Flow Diagram (DFD Level 0 — Context Diagram)", color="#0f172a",
              fontsize=16, weight="bold", pad=20)

    # User Box (External Entity)
    user_box = patches.FancyBboxPatch((0.5, 1.8), 2.5, 1.6, boxstyle="round,pad=0.2",
                                      fc="#e0f2fe", ec="#0284c7", lw=2.5)
    ax.add_patch(user_box)
    ax.text(1.75, 2.6, "User / Radiologist\n(External Entity)", color="#0369a1", weight="bold",
            fontsize=13, ha="center", va="center")

    # System Circle
    sys_circle = patches.Circle((7.5, 2.6), 1.5, fc="#f0f9ff", ec="#0284c7", lw=3.0)
    ax.add_patch(sys_circle)
    ax.text(7.5, 2.6, "0.0\nBrain Tumor\nClassification &\nExplainability\nPipeline", 
            color="#0f172a", weight="bold", fontsize=12, ha="center", va="center")

    # Top Arrow (User -> System)
    ax.annotate("", xy=(5.9, 3.2), xytext=(3.1, 3.2),
                arrowprops=dict(arrowstyle="-|>", color="#0284c7", lw=2.5, mutation_scale=20))
    ax.text(4.5, 3.45, "1. Upload MRI Image Scan (PNG/JPG)\n2. HTTP GET Download PDF Report Request", 
            color="#0369a1", fontsize=11, weight="bold", ha="center", va="bottom")

    # Bottom Arrow (System -> User)
    ax.annotate("", xy=(3.1, 2.0), xytext=(5.9, 2.0),
                arrowprops=dict(arrowstyle="-|>", color="#16a34a", lw=2.5, mutation_scale=20))
    ax.text(4.5, 1.75, "1. Render Diagnostic Result View (Label, Conf %)\n2. Patient PDF Diagnostic Report Stream", 
            color="#15803d", fontsize=11, weight="bold", ha="center", va="top")

    output_path = os.path.join(docs_dir, "dfd_level0.png")
    plt.tight_layout()
    plt.savefig(output_path, facecolor=fig.get_facecolor(), bbox_inches='tight')
    plt.close()
    print(f"Generated: {output_path}")

def create_dfd_level1():
    fig, ax = plt.subplots(figsize=(12, 7.5), dpi=150)
    fig.patch.set_facecolor('#ffffff')
    ax.set_facecolor('#ffffff')
    ax.axis('off')

    plt.title("Data Flow Diagram (DFD Level 1 — Core Pipeline & Data Stores)", color="#0f172a",
              fontsize=16, weight="bold", pad=20)

    # Actor (User)
    actor_box = patches.FancyBboxPatch((0.3, 3.5), 1.8, 1.2, boxstyle="round,pad=0.15", fc="#e0f2fe", ec="#0284c7", lw=2.5)
    ax.add_patch(actor_box)
    ax.text(1.2, 4.1, "User /\nRadiologist", color="#0369a1", weight="bold", fontsize=12, ha="center", va="center")

    # Core Processes (P1 to P6)
    proc_coords = {
        "P1": (3.4, 5.7, "1.0 Validation Guard"),
        "P2": (6.2, 5.7, "2.0 Image Preprocessing"),
        "P3": (9.0, 5.7, "3.0 Model Inference"),
        "P4": (9.0, 2.5, "4.0 Grad-CAM Engine"),
        "P5": (6.2, 2.5, "5.0 PDF Report Generator"),
        "P6": (3.4, 2.5, "6.0 View Renderer"),
    }

    for key, (x, y, label) in proc_coords.items():
        box = patches.FancyBboxPatch((x-1.1, y-0.55), 2.2, 1.1, boxstyle="round,pad=0.15", fc="#f0f9ff", ec="#0284c7", lw=2.5)
        ax.add_patch(box)
        ax.text(x, y, label, color="#0f172a", weight="bold", fontsize=11, ha="center", va="center")

    # Data Stores
    ds_coords = {
        "DS1": (4.8, 4.1, "DS1: Upload Store"),
        "DS2": (11.5, 5.7, "DS2: Model Weights"),
        "DS3": (11.5, 2.5, "DS3: Heatmaps Store"),
        "DS4": (6.2, 0.6, "DS4: PDF Reports"),
    }

    for key, (x, y, label) in ds_coords.items():
        box = patches.Rectangle((x-1.0, y-0.45), 2.0, 0.9, fc="#faf5ff", ec="#7e22ce", lw=2.2, linestyle="-")
        ax.add_patch(box)
        ax.text(x, y, label, color="#6b21a8", weight="bold", fontsize=10.5, ha="center", va="center")

    # Helper function for arrows
    def draw_arrow(start, end, text="", color="#0284c7"):
        ax.annotate("", xy=end, xytext=start,
                    arrowprops=dict(arrowstyle="-|>", color=color, lw=2.2, mutation_scale=16))
        if text:
            mid_x = (start[0] + end[0]) / 2
            mid_y = (start[1] + end[1]) / 2 + 0.18
            ax.text(mid_x, mid_y, text, color=color, fontsize=10, weight="bold", ha="center")

    draw_arrow((2.2, 4.4), (2.4, 5.7), "Upload Scan")
    draw_arrow((2.4, 5.7), (3.8, 4.1), "Valid Scan")
    draw_arrow((4.8, 4.6), (5.2, 5.7), "Raw Path")
    draw_arrow((7.2, 5.7), (8.0, 5.7), "Tensor")
    draw_arrow((10.6, 5.7), (10.0, 5.7), "Load .keras", color="#7e22ce")
    draw_arrow((9.0, 5.1), (9.0, 3.1), "Prob & Tensor")
    draw_arrow((10.0, 2.5), (10.6, 2.5), "Heatmaps", color="#7e22ce")
    draw_arrow((8.0, 2.5), (7.2, 2.5), "Overlays")
    draw_arrow((6.2, 1.9), (6.2, 1.1), "Report PDF", color="#7e22ce")
    draw_arrow((5.2, 2.5), (4.4, 2.5), "Result Props", color="#15803d")
    draw_arrow((3.4, 3.1), (2.2, 3.9), "Render View", color="#15803d")

    output_path = os.path.join(docs_dir, "dfd_level1.png")
    plt.tight_layout()
    plt.savefig(output_path, facecolor=fig.get_facecolor(), bbox_inches='tight')
    plt.close()
    print(f"Generated: {output_path}")

def create_class_architecture():
    fig, ax = plt.subplots(figsize=(12, 8.5), dpi=150)
    fig.patch.set_facecolor('#ffffff')
    ax.set_facecolor('#ffffff')
    ax.axis('off')

    plt.title("High-Level System Class Architecture Diagram", color="#0f172a",
              fontsize=16, weight="bold", pad=20)

    # 1. Config & App Factory
    factory_box = patches.FancyBboxPatch((0.5, 6.2), 4.8, 1.9, boxstyle="round,pad=0.2", fc="#eff6ff", ec="#1d4ed8", lw=2.5)
    ax.add_patch(factory_box)
    ax.text(2.9, 7.6, "Config & Application Factory", color="#1e40af", weight="bold", fontsize=13, ha="center")
    ax.text(2.9, 6.7, "Config (IMG_SIZE, MODEL_PATH, THRESHOLD=0.3)\ncreate_app(Config) -> Flask App Instance\nModel Singleton Boot Initialization\nDirectory & Metric Synchronizer",
            color="#1e293b", fontsize=10.5, ha="center")

    # 2. Controllers / Blueprints
    routes_box = patches.FancyBboxPatch((6.2, 6.2), 5.3, 1.9, boxstyle="round,pad=0.2", fc="#fffbeb", ec="#b45309", lw=2.5)
    ax.add_patch(routes_box)
    ax.text(8.85, 7.6, "Controller Blueprints (app/routes)", color="#92400e", weight="bold", fontsize=13, ha="center")
    ax.text(8.85, 6.7, "• PredictRoutes (/predict, /download_report)\n• MainRoutes (/ index upload UI)\n• StatsRoutes (/stats test benchmarks)\n• HealthRoutes (/health status check)",
            color="#1e293b", fontsize=10.5, ha="center")

    # 3. Services Layer
    services = [
        ("ValidationService", "is_allowed_file()\nis_valid_file_size()", (0.5, 3.4), "#15803d", "#f0fdf4"),
        ("ImageService", "preprocess_image()\nResize (224,224), Float32", (4.3, 3.4), "#1d4ed8", "#eff6ff"),
        ("ModelService (Singleton)", "load_model()\npredict() -> (label, conf %)\nThreshold = 0.3", (8.1, 3.4), "#be185d", "#fdf2f8"),
        ("GradCAMService", "generate_gradcam()\nOtsu Mask, JET Overlay\nROI Top 50% HUD Highlight", (0.5, 0.7), "#6b21a8", "#faf5ff"),
        ("PDFService", "generate_patient_pdf_report()\nReportLab Story Builder\nDiagnostic Card & Grid", (4.3, 0.7), "#c2410c", "#fff7ed"),
        ("StatsService", "ensure_directories_and_graphs()\nget_performance_metrics()", (8.1, 0.7), "#0f766e", "#f0fdfa"),
    ]

    for title, desc, (x, y), border_col, bg_col in services:
        box = patches.FancyBboxPatch((x, y), 3.4, 2.1, boxstyle="round,pad=0.15", fc=bg_col, ec=border_col, lw=2.5)
        ax.add_patch(box)
        ax.text(x + 1.7, y + 1.6, title, color=border_col, weight="bold", fontsize=11.5, ha="center")
        ax.text(x + 1.7, y + 0.8, desc, color="#1e293b", fontsize=10, ha="center")

    # Connections
    ax.annotate("", xy=(6.2, 7.15), xytext=(5.3, 7.15),
                arrowprops=dict(arrowstyle="-|>", color="#1d4ed8", lw=2.5, mutation_scale=16))

    output_path = os.path.join(docs_dir, "class_architecture.png")
    plt.tight_layout()
    plt.savefig(output_path, facecolor=fig.get_facecolor(), bbox_inches='tight')
    plt.close()
    print(f"Generated: {output_path}")

def create_sequence_diagram():
    fig, ax = plt.subplots(figsize=(12, 6.5), dpi=150)
    fig.patch.set_facecolor('#ffffff')
    ax.set_facecolor('#ffffff')
    ax.axis('off')

    plt.title("End-to-End Component Execution Sequence Diagram", color="#0f172a",
              fontsize=16, weight="bold", pad=20)

    lifelines = ["User", "PredictRoutes", "Validation", "ImageService", "ModelService", "GradCAM", "PDFService"]
    x_positions = [1.0, 2.8, 4.6, 6.4, 8.2, 10.0, 11.8]

    for x, name in zip(x_positions, lifelines):
        ax.plot([x, x], [0.8, 5.4], color="#94a3b8", linestyle="--", lw=2)
        box = patches.FancyBboxPatch((x-0.75, 5.4), 1.5, 0.6, boxstyle="round,pad=0.1", fc="#e0f2fe", ec="#0284c7", lw=2)
        ax.add_patch(box)
        ax.text(x, 5.7, name, color="#0369a1", weight="bold", fontsize=11, ha="center")

    steps = [
        (1.0, 2.8, 4.9, "1. POST /predict (file)", "#0284c7"),
        (2.8, 4.6, 4.3, "2. is_allowed_file & size check", "#15803d"),
        (2.8, 6.4, 3.7, "3. preprocess_image(path)", "#1d4ed8"),
        (2.8, 8.2, 3.1, "4. predict(model, processed_img)", "#be185d"),
        (2.8, 10.0, 2.5, "5. generate_gradcam(...)", "#6b21a8"),
        (2.8, 11.8, 1.9, "6. generate_patient_pdf_report(...)", "#c2410c"),
        (2.8, 1.0, 1.3, "7. Render result.html HTML Response", "#15803d"),
    ]

    for x1, x2, y, label, color in steps:
        ax.annotate("", xy=(x2, y), xytext=(x1, y),
                    arrowprops=dict(arrowstyle="-|>", color=color, lw=2.2, mutation_scale=16))
        mid_x = (x1 + x2) / 2
        ax.text(mid_x, y + 0.15, label, color=color, fontsize=10, weight="bold", ha="center")

    output_path = os.path.join(docs_dir, "sequence_diagram.png")
    plt.tight_layout()
    plt.savefig(output_path, facecolor=fig.get_facecolor(), bbox_inches='tight')
    plt.close()
    print(f"Generated: {output_path}")

if __name__ == "__main__":
    create_dfd_level0()
    create_dfd_level1()
    create_class_architecture()
    create_sequence_diagram()
