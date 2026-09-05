import os
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.patches as patches

docs_dir = os.path.dirname(os.path.abspath(__file__))

def generate_dfd_level0():
    fig, ax = plt.subplots(figsize=(10, 5.2), dpi=200)
    fig.patch.set_facecolor('#ffffff')
    ax.set_facecolor('#ffffff')
    ax.axis('off')

    plt.title("Figure 4.1 (a): Data Flow Diagram (DFD Level 0 — Context Diagram)", color="#0f172a",
              fontsize=15, weight="bold", pad=20)

    user_box = patches.FancyBboxPatch((0.5, 1.7), 2.7, 1.8, boxstyle="round,pad=0.25",
                                      fc="#e0f2fe", ec="#0284c7", lw=2.5)
    ax.add_patch(user_box)
    ax.text(1.85, 2.6, "User / Radiologist\n/ Clinician\n(External Entity)", color="#0369a1", weight="bold",
            fontsize=12, ha="center", va="center", linespacing=1.4)

    sys_box = patches.FancyBboxPatch((6.8, 1.5), 3.4, 2.2, boxstyle="round,pad=0.3",
                                     fc="#f0f9ff", ec="#0284c7", lw=3.0)
    ax.add_patch(sys_box)
    ax.text(8.5, 2.6, "0.0\nBrain Tumor Classification\n& Explainability Pipeline\n(Flask + Keras CNN)", 
            color="#0f172a", weight="bold", fontsize=12, ha="center", va="center", linespacing=1.4)

    ax.annotate("", xy=(6.6, 3.1), xytext=(3.4, 3.1),
                arrowprops=dict(arrowstyle="-|>", color="#0284c7", lw=2.5, mutation_scale=18))
    ax.text(5.0, 3.35, "1. Upload MRI Image Scan File (PNG / JPG)\n2. Request Patient PDF Diagnostic Report", 
            color="#0369a1", fontsize=10.5, weight="bold", ha="center", va="bottom", linespacing=1.3)

    ax.annotate("", xy=(3.4, 2.1), xytext=(6.6, 2.1),
                arrowprops=dict(arrowstyle="-|>", color="#16a34a", lw=2.5, mutation_scale=18))
    ax.text(5.0, 1.85, "1. Render Diagnostic Result View (Label, Conf %)\n2. Stream Downloadable Patient PDF Report", 
            color="#15803d", fontsize=10.5, weight="bold", ha="center", va="top", linespacing=1.3)

    output_path = os.path.join(docs_dir, "dfd_level0.png")
    plt.savefig(output_path, facecolor=fig.get_facecolor(), bbox_inches='tight')
    plt.close()
    print(f"Successfully generated: {output_path}")

def generate_dfd_level1():
    fig, ax = plt.subplots(figsize=(13.5, 8.0), dpi=200)
    fig.patch.set_facecolor('#ffffff')
    ax.set_facecolor('#ffffff')
    ax.axis('off')

    plt.title("Figure 4.1 (b): Data Flow Diagram (DFD Level 1 — Core Pipeline & Data Stores)", color="#0f172a",
              fontsize=15, weight="bold", pad=20)

    actor_box = patches.FancyBboxPatch((0.4, 3.6), 2.0, 1.4, boxstyle="round,pad=0.2", fc="#e0f2fe", ec="#0284c7", lw=2.5)
    ax.add_patch(actor_box)
    ax.text(1.4, 4.3, "User /\nRadiologist", color="#0369a1", weight="bold", fontsize=12, ha="center", va="center", linespacing=1.3)

    proc_coords = {
        "P1": (3.8, 6.0, "1.0 Validation Guard\n(File Extension & Size)"),
        "P2": (7.0, 6.0, "2.0 Image Preprocessing\n(224x224 RGB Float Tensor)"),
        "P3": (10.2, 6.0, "3.0 Model Inference\n(Keras Binary CNN Classifier)"),
        "P4": (10.2, 2.5, "4.0 Grad-CAM Engine\n(Autograd & HUD Region Box)"),
        "P5": (7.0, 2.5, "5.0 PDF Generator\n(ReportLab Flowable Service)"),
        "P6": (3.8, 2.5, "6.0 View Renderer\n(HTML Result Dashboard)"),
    }

    for key, (x, y, label) in proc_coords.items():
        box = patches.FancyBboxPatch((x-1.3, y-0.6), 2.6, 1.2, boxstyle="round,pad=0.2", fc="#f0f9ff", ec="#0284c7", lw=2.5)
        ax.add_patch(box)
        ax.text(x, y, label, color="#0f172a", weight="bold", fontsize=10.5, ha="center", va="center", linespacing=1.3)

    ds_coords = {
        "DS1": (5.4, 4.2, "DS1: Scan Uploads Store\n(app/static/uploads/)"),
        "DS2": (12.8, 6.0, "DS2: Keras Model Weights\n(bestModel.keras)"),
        "DS3": (12.8, 2.5, "DS3: Explainability Store\n(app/static/heatmaps/)"),
        "DS4": (7.0, 0.7, "DS4: PDF Patient Reports\n(app/static/reports/)"),
    }

    for key, (x, y, label) in ds_coords.items():
        box = patches.Rectangle((x-1.15, y-0.5), 2.3, 1.0, fc="#faf5ff", ec="#7e22ce", lw=2.2)
        ax.add_patch(box)
        ax.text(x, y, label, color="#6b21a8", weight="bold", fontsize=10, ha="center", va="center", linespacing=1.3)

    def draw_flow(start, end, text="", color="#0284c7"):
        ax.annotate("", xy=end, xytext=start,
                    arrowprops=dict(arrowstyle="-|>", color=color, lw=2.2, mutation_scale=16))
        if text:
            mid_x = (start[0] + end[0]) / 2
            mid_y = (start[1] + end[1]) / 2 + 0.18
            ax.text(mid_x, mid_y, text, color=color, fontsize=9.5, weight="bold", ha="center")

    draw_flow((2.4, 4.6), (2.6, 6.0), "Upload MRI Scan")
    draw_flow((2.6, 6.0), (4.1, 4.2), "Valid File Stream")
    draw_flow((5.4, 4.7), (5.8, 6.0), "Saved Scan Path")
    draw_flow((8.3, 6.0), (8.9, 6.0), "Tensor (1,224,224,3)")
    draw_flow((11.6, 6.0), (11.5, 6.0), "Load .keras Model", color="#7e22ce")
    draw_flow((10.2, 5.4), (10.2, 3.1), "Label & Activation Tensor")
    draw_flow((11.5, 2.5), (11.6, 2.5), "Heatmap / HUD Box", color="#7e22ce")
    draw_flow((8.9, 2.5), (8.3, 2.5), "Visual Overlay Paths")
    draw_flow((7.0, 1.9), (7.0, 1.2), "Build Patient PDF", color="#7e22ce")
    draw_flow((5.7, 2.5), (5.1, 2.5), "Result Data Props", color="#15803d")
    draw_flow((3.8, 3.1), (2.4, 4.0), "Render Result UI", color="#15803d")

    output_path = os.path.join(docs_dir, "dfd_level1.png")
    plt.savefig(output_path, facecolor=fig.get_facecolor(), bbox_inches='tight')
    plt.close()
    print(f"Successfully generated: {output_path}")

def generate_er_diagram():
    fig, ax = plt.subplots(figsize=(14, 8.0), dpi=200)
    fig.patch.set_facecolor('#ffffff')
    ax.set_facecolor('#ffffff')
    ax.axis('off')

    plt.title("Entity-Relationship (E-R) Diagram — Brain Tumor Diagnostic System", color="#0f172a",
              fontsize=15, weight="bold", pad=20)

    entities = [
        ("USER", "User_ID (PK)\nName / Role\nSession_ID", (0.8, 4.5), "#e0f2fe", "#0284c7"),
        ("MRI_SCAN", "Scan_ID (PK)\nFilename, File_Size\nUpload_Timestamp", (4.2, 4.5), "#f0fdf4", "#15803d"),
        ("MODEL_PREDICTION", "Prediction_ID (PK)\nLabel ('Tumor' / 'No Tumor')\nConfidence_%, Probability\nThreshold (0.3)", (7.6, 4.5), "#fdf2f8", "#be185d"),
        ("VISUAL_EXPLAINABILITY", "Artifact_ID (PK)\nHeatmap_Path\nHUD_Highlight_Path\nROI_Bounding_Box", (7.6, 0.8), "#faf5ff", "#6b21a8"),
        ("PATIENT_PDF_REPORT", "Report_ID (PK)\nPDF_Filename\nReport_Timestamp", (11.4, 4.5), "#fff7ed", "#c2410c"),
    ]

    for name, attrs, (x, y), bg_col, border_col in entities:
        box = patches.FancyBboxPatch((x, y), 2.5, 2.2, boxstyle="round,pad=0.2", fc=bg_col, ec=border_col, lw=2.5)
        ax.add_patch(box)
        ax.text(x + 1.25, y + 1.8, name, color=border_col, weight="bold", fontsize=11, ha="center")
        ax.text(x + 1.25, y + 0.8, attrs, color="#1e293b", fontsize=9.5, ha="center", linespacing=1.3)

    rel_coords = [
        ("Uploads\n(1 : N)", (3.5, 5.6)),
        ("Processes\n(1 : 1)", (6.9, 5.6)),
        ("Generates\n(1 : 1)", (8.85, 3.5)),
        ("Produces\n(1 : 1)", (10.3, 5.6)),
    ]

    for label, (x, y) in rel_coords:
        diamond = patches.RegularPolygon((x, y), numVertices=4, radius=0.55, fc="#fffbeb", ec="#b45309", lw=2)
        ax.add_patch(diamond)
        ax.text(x, y, label, color="#92400e", weight="bold", fontsize=8.5, ha="center", va="center")

    ax.plot([3.3, 3.5, 4.2], [5.6, 5.6, 5.6], color="#0284c7", lw=2)
    ax.plot([6.7, 6.9, 7.6], [5.6, 5.6, 5.6], color="#15803d", lw=2)
    ax.plot([8.85, 8.85], [4.5, 3.0], color="#be185d", lw=2)
    ax.plot([10.1, 10.3, 11.4], [5.6, 5.6, 5.6], color="#be185d", lw=2)

    output_path = os.path.join(docs_dir, "er_diagram.png")
    plt.savefig(output_path, facecolor=fig.get_facecolor(), bbox_inches='tight')
    plt.close()
    print(f"Successfully generated: {output_path}")

def generate_class_architecture():
    fig, ax = plt.subplots(figsize=(13.5, 9.0), dpi=200)
    fig.patch.set_facecolor('#ffffff')
    ax.set_facecolor('#ffffff')
    ax.axis('off')

    plt.title("Figure 4.2: High-Level System Class Architecture Diagram", color="#0f172a",
              fontsize=15, weight="bold", pad=20)

    factory_box = patches.FancyBboxPatch((0.5, 6.8), 5.4, 2.0, boxstyle="round,pad=0.25", fc="#eff6ff", ec="#1d4ed8", lw=2.5)
    ax.add_patch(factory_box)
    ax.text(3.2, 8.3, "Config & Application Factory", color="#1e40af", weight="bold", fontsize=12.5, ha="center")
    ax.text(3.2, 7.4, "Config (IMG_SIZE, MODEL_PATH, THRESHOLD=0.3)\ncreate_app(Config) -> Flask App Instance\nPre-loads Singleton Model on Startup\nInitializes Upload Folders & Graphs",
            color="#1e293b", fontsize=10.5, ha="center", linespacing=1.3)

    routes_box = patches.FancyBboxPatch((6.6, 6.8), 6.4, 2.0, boxstyle="round,pad=0.25", fc="#fffbeb", ec="#b45309", lw=2.5)
    ax.add_patch(routes_box)
    ax.text(9.8, 8.3, "Controller Blueprints (app/routes)", color="#92400e", weight="bold", fontsize=12.5, ha="center")
    ax.text(9.8, 7.4, "• PredictRoutes (/predict POST upload, /download_report GET)\n• MainRoutes (/ index drag-and-drop UI)\n• StatsRoutes (/stats test benchmark metrics & ROC curves)\n• HealthRoutes (/health runtime JSON check)",
            color="#1e293b", fontsize=10.5, ha="center", linespacing=1.3)

    services = [
        ("ValidationService", "is_allowed_file()\nis_valid_file_size()", (0.5, 3.6), "#15803d", "#f0fdf4"),
        ("ImageService", "preprocess_image()\nResize (224,224), Float32", (4.8, 3.6), "#1d4ed8", "#eff6ff"),
        ("ModelService (Singleton)", "load_model()\npredict() -> (label, conf %)\nThreshold = 0.3", (9.1, 3.6), "#be185d", "#fdf2f8"),
        ("GradCAMService", "generate_gradcam()\nOtsu Mask, JET Overlay\nROI Top 50% HUD Highlight", (0.5, 0.6), "#6b21a8", "#faf5ff"),
        ("PDFService", "generate_patient_pdf_report()\nReportLab Story Builder\nDiagnostic Card & Grid", (4.8, 0.6), "#c2410c", "#fff7ed"),
        ("StatsService", "ensure_directories_and_graphs()\nget_performance_metrics()", (9.1, 0.6), "#0f766e", "#f0fdfa"),
    ]

    for title, desc, (x, y), border_col, bg_col in services:
        box = patches.FancyBboxPatch((x, y), 3.9, 2.2, boxstyle="round,pad=0.2", fc=bg_col, ec=border_col, lw=2.5)
        ax.add_patch(box)
        ax.text(x + 1.95, y + 1.65, title, color=border_col, weight="bold", fontsize=11.5, ha="center")
        ax.text(x + 1.95, y + 0.8, desc, color="#1e293b", fontsize=10, ha="center", linespacing=1.3)

    ax.annotate("", xy=(6.6, 7.8), xytext=(5.9, 7.8),
                arrowprops=dict(arrowstyle="-|>", color="#1d4ed8", lw=2.5, mutation_scale=18))

    output_path = os.path.join(docs_dir, "class_architecture.png")
    plt.savefig(output_path, facecolor=fig.get_facecolor(), bbox_inches='tight')
    plt.close()
    print(f"Successfully generated: {output_path}")

def generate_object_interaction_diagram():
    fig, ax = plt.subplots(figsize=(13.5, 7.5), dpi=200)
    fig.patch.set_facecolor('#ffffff')
    ax.set_facecolor('#ffffff')
    ax.axis('off')

    plt.title("Section 4.7.5: Object Interaction Diagram", color="#0f172a",
              fontsize=15, weight="bold", pad=20)

    objects = [
        ("client : Browser", (0.8, 5.8), "#e0f2fe", "#0284c7"),
        ("predict_bp : Blueprint", (3.2, 5.8), "#fffbeb", "#b45309"),
        ("validator : Validation", (5.6, 5.8), "#f0fdf4", "#15803d"),
        ("preprocessor : Image", (8.0, 5.8), "#eff6ff", "#1d4ed8"),
        ("model : ModelSingleton", (10.4, 5.8), "#fdf2f8", "#be185d"),
        ("gradcam : GradCAM", (10.4, 1.2), "#faf5ff", "#6b21a8"),
        ("pdfEngine : PDFService", (5.6, 1.2), "#fff7ed", "#c2410c"),
    ]

    for label, (x, y), bg_col, border_col in objects:
        box = patches.FancyBboxPatch((x-1.0, y-0.45), 2.0, 0.9, boxstyle="round,pad=0.15", fc=bg_col, ec=border_col, lw=2.2)
        ax.add_patch(box)
        ax.text(x, y, label, color=border_col, weight="bold", fontsize=10, ha="center", va="center")

    calls = [
        ((0.8, 5.35), (3.2, 5.35), "1: handle_predict(file)", "#0284c7"),
        ((3.2, 5.8), (4.6, 5.8), "2: is_allowed_file()", "#15803d"),
        ((3.2, 5.35), (7.0, 5.35), "3: preprocess_image()", "#1d4ed8"),
        ((3.2, 4.9), (9.4, 4.9), "4: predict(model, img)", "#be185d"),
        ((3.2, 4.45), (10.4, 2.1), "5: generate_gradcam()", "#6b21a8"),
        ((3.2, 4.0), (5.6, 2.1), "6: generate_pdf_report()", "#c2410c"),
        ((3.2, 3.5), (0.8, 3.5), "7: render result.html", "#15803d"),
    ]

    for start, end, msg, col in calls:
        ax.annotate("", xy=end, xytext=start,
                    arrowprops=dict(arrowstyle="-|>", color=col, lw=2.0, mutation_scale=14))
        mid_x = (start[0] + end[0]) / 2
        mid_y = (start[1] + end[1]) / 2 + 0.15
        ax.text(mid_x, mid_y, msg, color=col, fontsize=9.5, weight="bold", ha="center")

    output_path = os.path.join(docs_dir, "object_interaction_diagram.png")
    plt.savefig(output_path, facecolor=fig.get_facecolor(), bbox_inches='tight')
    plt.close()
    print(f"Successfully generated: {output_path}")

def generate_control_flow_diagram():
    fig, ax = plt.subplots(figsize=(13.5, 9.5), dpi=200)
    fig.patch.set_facecolor('#ffffff')
    ax.set_facecolor('#ffffff')
    ax.axis('off')

    plt.title("Section 4.8.3: Control Flow Diagram (CFD)", color="#0f172a",
              fontsize=15, weight="bold", pad=25)

    # Rectangles (States/Processes) & Diamonds (Decisions)
    nodes = [
        ("Start: load_model()", (6.75, 8.8), "#e0f2fe", "#0284c7", "ellipse"),
        ("Check: bestModel.keras Exists?", (6.75, 7.8), "#fffbeb", "#b45309", "diamond"),
        ("Error: Raise FileNotFoundError", (2.0, 7.8), "#fef2f2", "#ef4444", "rect"),
        ("HTTP Listener: POST /predict", (6.75, 6.8), "#f0f9ff", "#0284c7", "rect"),
        ("Check: File Format (.png/.jpg) & Size (<=10MB)", (6.75, 5.7), "#fffbeb", "#b45309", "diamond"),
        ("Error: Render error.html (HTTP 400)", (2.0, 5.7), "#fef2f2", "#ef4444", "rect"),
        ("Preprocess: Resize 224x224 RGB, Scale / 255.0", (6.75, 4.6), "#eff6ff", "#1d4ed8", "rect"),
        ("Model Feedforward: predict()", (6.75, 3.6), "#fdf2f8", "#be185d", "rect"),
        ("Decision: Probability >= 0.3 Threshold?", (6.75, 2.5), "#fffbeb", "#b45309", "diamond"),
        ("Label: Tumor & Conf %", (3.8, 1.4), "#fdf2f8", "#be185d", "rect"),
        ("Label: No Tumor & Conf %", (9.7, 1.4), "#f0fdf4", "#15803d", "rect"),
        ("Grad-CAM Overlay & PDF Synthesis", (6.75, 0.4), "#faf5ff", "#6b21a8", "rect"),
    ]

    for label, (x, y), bg_col, border_col, n_type in nodes:
        if n_type == "ellipse":
            box = patches.FancyBboxPatch((x-1.5, y-0.35), 3.0, 0.7, boxstyle="round,pad=0.2", fc=bg_col, ec=border_col, lw=2.2)
        elif n_type == "diamond":
            box = patches.RegularPolygon((x, y), numVertices=4, radius=0.65, fc=bg_col, ec=border_col, lw=2.2)
        else:
            box = patches.FancyBboxPatch((x-1.8, y-0.35), 3.6, 0.7, boxstyle="round,pad=0.15", fc=bg_col, ec=border_col, lw=2.2)
        ax.add_patch(box)
        ax.text(x, y, label, color=border_col, weight="bold", fontsize=9.5, ha="center", va="center")

    def draw_arrow(start, end, text="", col="#0284c7"):
        ax.annotate("", xy=end, xytext=start,
                    arrowprops=dict(arrowstyle="-|>", color=col, lw=2.0, mutation_scale=14))
        if text:
            mid_x = (start[0] + end[0]) / 2 + 0.15
            mid_y = (start[1] + end[1]) / 2 + 0.1
            ax.text(mid_x, mid_y, text, color=col, fontsize=9.0, weight="bold", ha="center")

    draw_arrow((6.75, 8.45), (6.75, 8.45))
    draw_arrow((6.75, 8.45), (6.75, 8.45))

    output_path = os.path.join(docs_dir, "control_flow_diagram.png")
    plt.savefig(output_path, facecolor=fig.get_facecolor(), bbox_inches='tight')
    plt.close()
    print(f"Successfully generated: {output_path}")

def generate_sequence_diagram():
    fig, ax = plt.subplots(figsize=(13.0, 7.2), dpi=200)
    fig.patch.set_facecolor('#ffffff')
    ax.set_facecolor('#ffffff')
    ax.axis('off')

    plt.title("Component Interaction & Execution Sequence Diagram", color="#0f172a",
              fontsize=15, weight="bold", pad=20)

    lifelines = ["User", "PredictRoutes", "Validation", "ImageService", "ModelService", "GradCAM", "PDFService"]
    x_positions = [1.0, 2.9, 4.8, 6.7, 8.6, 10.5, 12.4]

    for x, name in zip(x_positions, lifelines):
        ax.plot([x, x], [0.8, 6.0], color="#94a3b8", linestyle="--", lw=2)
        box = patches.FancyBboxPatch((x-0.8, 6.0), 1.6, 0.7, boxstyle="round,pad=0.15", fc="#e0f2fe", ec="#0284c7", lw=2)
        ax.add_patch(box)
        ax.text(x, 6.35, name, color="#0369a1", weight="bold", fontsize=11, ha="center")

    steps = [
        (1.0, 2.9, 5.4, "1. POST /predict (file upload)", "#0284c7"),
        (2.9, 4.8, 4.7, "2. is_allowed_file & size validation", "#15803d"),
        (2.9, 6.7, 4.0, "3. preprocess_image(upload_path)", "#1d4ed8"),
        (2.9, 8.6, 3.3, "4. predict(model_instance, processed_img)", "#be185d"),
        (2.9, 10.5, 2.6, "5. generate_gradcam(model, processed_img...)", "#6b21a8"),
        (2.9, 12.4, 1.9, "6. generate_patient_pdf_report(...)", "#c2410c"),
        (2.9, 1.0, 1.2, "7. Render result.html HTML Response", "#15803d"),
    ]

    for x1, x2, y, label, color in steps:
        ax.annotate("", xy=(x2, y), xytext=(x1, y),
                    arrowprops=dict(arrowstyle="-|>", color=color, lw=2.2, mutation_scale=16))
        mid_x = (x1 + x2) / 2
        ax.text(mid_x, y + 0.16, label, color=color, fontsize=10, weight="bold", ha="center")

    output_path = os.path.join(docs_dir, "sequence_diagram.png")
    plt.savefig(output_path, facecolor=fig.get_facecolor(), bbox_inches='tight')
    plt.close()
    print(f"Successfully generated: {output_path}")

if __name__ == "__main__":
    generate_dfd_level0()
    generate_dfd_level1()
    generate_er_diagram()
    generate_class_architecture()
    generate_object_interaction_diagram()
    generate_control_flow_diagram()
    generate_sequence_diagram()
