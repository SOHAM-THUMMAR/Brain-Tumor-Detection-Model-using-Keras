# Brain Tumor Classification System — Architecture & Data Flow Diagrams

> 📸 **Screenshot & Viewing Instructions for RK University Project Report:**
> - **Option 1 (Interactive HTML Diagram Viewer)**: Open [docs/view_diagrams.html](file:///d:/collage%20project/Brain-Tumor-Detection-Model-using-Keras/docs/view_diagrams.html) in Chrome/Edge to view pixel-perfect Mermaid diagrams rendered on a clean white background.
> - **Option 2 (Markdown Preview)**: Open this document ([docs/diagrams.md](file:///d:/collage%20project/Brain-Tumor-Detection-Model-using-Keras/docs/diagrams.md)) in your Markdown Previewer to capture Mermaid diagrams or high-resolution PNG diagram figures directly.

---

## Table of Contents
1. [Data Flow Diagrams (DFD)](#1-data-flow-diagrams-dfd)
   - [1.1 Context Level Data Flow Diagram (DFD Level 0 — Figure 4.1a)](#11-context-level-data-flow-diagram-dfd-level-0--figure-41a)
   - [1.2 Level 1 Data Flow Diagram (DFD Level 1 — Figure 4.1b)](#12-level-1-data-flow-diagram-dfd-level-1--figure-41b)
   - [1.3 Level 2 Data Flow Diagram (DFD Level 2 — Sub-Process Breakdown)](#13-level-2-data-flow-diagram-dfd-level-2--sub-process-breakdown)
   - [1.4 Data Dictionary & Data Flow Mapping](#14-data-dictionary--data-flow-mapping)
2. [High-Level System Class Architecture Diagram](#2-high-level-system-class-architecture-diagram)
   - [2.1 UML Class Architecture Diagram (Figure 4.2)](#21-uml-class-architecture-diagram-figure-42)
   - [2.2 Detailed Class Responsibilities & Design Patterns](#22-detailed-class-responsibilities--design-patterns)
   - [2.3 Component Interaction & Execution Sequence](#23-component-interaction--execution-sequence)

---

## 1. Data Flow Diagrams (DFD)

The Data Flow Diagrams describe how data enters the NeuroScan AI system, transforms across preprocessing, neural network inference, visual explainability engines, report synthesis, and is delivered to the user.

### 1.1 Context Level Data Flow Diagram (DFD Level 0 — Figure 4.1a)

The Level 0 Context Diagram establishes the boundary between the primary external actor (**User / Radiologist / Clinician**) and the single high-level system boundary (**Brain Tumor Classification Pipeline**).

#### 📊 Mermaid Diagram (For Screenshots & Markdown Renderers)

```mermaid
graph TD
    User["👤 External User / Radiologist / Clinician"]
    System["🧠 0.0 Brain Tumor Classification & Explainability Pipeline"]

    User -- "1. Upload MRI Image Scan File (PNG / JPG)\n2. HTTP GET Download PDF Report Request" --> System
    System -- "1. Render Diagnostic Result View (Label, Confidence %)\n2. Stream Patient PDF Diagnostic Report" --> User

    style User fill:#e0f2fe,stroke:#0284c7,stroke-width:2px,color:#0369a1
    style System fill:#f0f9ff,stroke:#0284c7,stroke-width:3px,color:#0f172a
```

#### 🖼️ Visual Diagram Image

![Data Flow Diagram Level 0](file:///d:/collage%20project/Brain-Tumor-Detection-Model-using-Keras/docs/dfd_level0.png)

#### 📝 Plain-Text ASCII Diagram

```
+-----------------------------------------------------------------+
|                    USER / RADIOLOGIST                           |
|                    (External Entity)                            |
+-----------------------------------------------------------------+
       |                                                   ^
       | 1. Upload MRI Image (PNG/JPG)                     | 1. Render Result View
       | 2. Request PDF Report                             | 2. Stream PDF Report
       v                                                   |
+-----------------------------------------------------------------+
|     0.0 BRAIN TUMOR CLASSIFICATION & EXPLAINABILITY PIPELINE     |
|   (Flask App + Keras CNN Model + Grad-CAM + ReportLab PDF Engine)|
+-----------------------------------------------------------------+
```

---

### 1.2 Level 1 Data Flow Diagram (DFD Level 1 — Figure 4.1b)

The Level 1 DFD decomposes the system into six core functional processes, highlighting five distinct persistent data stores.

#### 📊 Mermaid Diagram (For Screenshots & Markdown Renderers)

```mermaid
flowchart TB
    subgraph External_Entity ["External Entity"]
        User["👤 User / Radiologist"]
    end

    subgraph Data_Stores ["Persistent Data Stores"]
        DS1[("DS1: Scan Uploads Store\n(app/static/uploads/)")]
        DS2[("DS2: Keras Model Artifact\n(bestModel.keras)")]
        DS3[("DS3: Explainability Artifacts\n(app/static/heatmaps/)")]
        DS4[("DS4: Patient PDF Reports\n(app/static/reports/)")]
        DS5[("DS5: Metric Plot Store\n(app/static/graphs/)")]
    end

    subgraph System_Processes ["System Core Processes"]
        P1["1.0 Request Handling & Input Validation Guard"]
        P2["2.0 Image Preprocessing & Normalization"]
        P3["3.0 Model Inference & Decision Classification"]
        P4["4.0 Grad-CAM Explainability & Region Highlighting"]
        P5["5.0 Patient PDF Diagnostic Report Generation"]
        P6["6.0 Result Rendering & Metric Presentation"]
    end

    User -- "Upload Image File" --> P1
    P1 -- "Validated Scan Stream" --> DS1
    DS1 -- "Raw Saved Scan Path" --> P2
    P2 -- "Normalized Tensor (1, 224, 224, 3)" --> P3
    
    DS2 -- "Pre-loaded Model Weights" --> P3
    P3 -- "Prediction Label & Confidence Score" --> P6
    
    P3 -- "Model Instance & Processed Tensor" --> P4
    P4 -- "JET Heatmap & HUD Contour Images" --> DS3
    
    DS1 -- "Original Image Path" --> P5
    DS3 -- "Heatmap & Highlight Paths" --> P5
    P3 -- "Classification Label & Confidence" --> P5
    P5 -- "Generated Patient PDF File" --> DS4
    
    DS4 -- "Stream Download PDF File" --> User
    DS5 -- "Evaluation Metric Graphs" --> P6
    P6 -- "Render HTML Dashboard View" --> User

    style User fill:#e0f2fe,stroke:#0284c7,stroke-width:2px,color:#0369a1
    style DS1 fill:#faf5ff,stroke:#7e22ce,stroke-width:2px,color:#6b21a8
    style DS2 fill:#faf5ff,stroke:#7e22ce,stroke-width:2px,color:#6b21a8
    style DS3 fill:#faf5ff,stroke:#7e22ce,stroke-width:2px,color:#6b21a8
    style DS4 fill:#faf5ff,stroke:#7e22ce,stroke-width:2px,color:#6b21a8
    style DS5 fill:#faf5ff,stroke:#7e22ce,stroke-width:2px,color:#6b21a8

    style P1 fill:#f0f9ff,stroke:#0284c7,stroke-width:2px,color:#0f172a
    style P2 fill:#f0f9ff,stroke:#0284c7,stroke-width:2px,color:#0f172a
    style P3 fill:#f0f9ff,stroke:#0284c7,stroke-width:2px,color:#0f172a
    style P4 fill:#f0f9ff,stroke:#0284c7,stroke-width:2px,color:#0f172a
    style P5 fill:#f0f9ff,stroke:#0284c7,stroke-width:2px,color:#0f172a
    style P6 fill:#f0f9ff,stroke:#0284c7,stroke-width:2px,color:#0f172a
```

#### 🖼️ Visual Diagram Image

![Data Flow Diagram Level 1](file:///d:/collage%20project/Brain-Tumor-Detection-Model-using-Keras/docs/dfd_level1.png)

---

### 1.3 Level 2 Data Flow Diagram (DFD Level 2 — Sub-Process Breakdown)

The Level 2 DFD details the deep learning calculation pipeline, visual feature extraction, and report construction sub-processes.

```mermaid
flowchart TD
    subgraph P3_Sub ["Process 3.0: Model Inference & Scoring"]
        P31["3.1 Feedforward Conv/Dense Pass"]
        P32["3.2 Logit Calculation (pen_eval @ dense_w + dense_b)"]
        P33["3.3 Threshold Evaluator (prob >= 0.3)"]
    end

    subgraph P4_Sub ["Process 4.0: Grad-CAM XAI Engine"]
        P41["4.1 Extract Conv2D (conv2d_7) & Penultimate Outputs"]
        P42["4.2 Backpropagate Logit Gradients via tf.GradientTape"]
        P43["4.3 Channel Pooled Gradient Weighting & ReLU"]
        P44["4.4 Otsu Thresholding Head Tissue Masking"]
        P45["4.5 JET Colormap Fusion Overlay (heatmap_*.jpg)"]
        P46["4.6 Top 50% Activation ROI Contour & HUD Box (highlight_*.jpg)"]
    end

    subgraph P5_Sub ["Process 5.0: PDF Diagnostic Report Engine"]
        P51["5.1 ReportLab Flowable Story Builder"]
        P52["5.2 Header Banner & Meta Table Formatting"]
        P53["5.3 Diagnostic Result Card & Color Badge"]
        P54["5.4 3-Image Comparison Grid Layout"]
        P55["5.5 Benchmark Summary & Disclaimer Injection"]
        P56["5.6 Document Build & File Writer"]
    end

    %% Connections
    P31 --> P32 --> P33
    P31 --> P41 --> P42 --> P43 --> P44 --> P45 --> P46
    P33 --> P53
    P45 --> P54
    P46 --> P54
    P51 --> P52 --> P53 --> P54 --> P55 --> P56
```

---

### 1.4 Data Dictionary & Data Flow Mapping

| Data Element Name | Data Type | Source Process / Entity | Destination Process / Store | Description |
| :--- | :--- | :--- | :--- | :--- |
| `raw_image_file` | File Stream / Bytes | External User | `1.0 Validation Guard` | Binary image payload uploaded via HTTP POST (`multipart/form-data`). |
| `saved_filename` | String | `1.0 Validation Guard` | `DS1 Uploads Store` | Unique file name string (`<uuid8>_<filename>`). |
| `processed_img` | `numpy.ndarray` (float32) | `2.0 Image Preprocessing` | `3.0 Model Inference` / `4.0 Grad-CAM Engine` | Tensor of shape `(1, 224, 224, 3)` with pixel values scaled to `[0, 1]`. |
| `orig_pil_img` | `PIL.Image` (RGB) | `2.0 Image Preprocessing` | `4.0 Grad-CAM Engine` | Unscaled original RGB image used for visual overlay resizing. |
| `prediction_label` | String | `3.0 Model Inference` | `5.0 PDF Engine` / `6.0 Result View` | Binary diagnosis result: `"Tumor"` or `"No Tumor"`. |
| `confidence` | Float (2 decimal places) | `3.0 Model Inference` | `5.0 PDF Engine` / `6.0 Result View` | Prediction confidence percentage (`0.0%` to `100.0%`). |
| `heatmap_relative_path`| String | `4.0 Grad-CAM Engine` | `DS3 Heatmap Store` / `6.0 Result View` | Relative path to JET colormap overlay (`heatmaps/heatmap_*.jpg`). |
| `highlight_relative_path`| String | `4.0 Grad-CAM Engine` | `DS3 Heatmap Store` / `6.0 Result View` | Relative path to HUD bounding box/contour highlight (`heatmaps/highlight_*.jpg`). |
| `pdf_relative_path` | String | `5.0 PDF Report Engine` | `DS4 Reports Store` / `User` | Relative path to downloadable patient diagnostic PDF (`reports/report_*.pdf`). |

---

## 2. High-Level System Class Architecture Diagram

The system employs a **Modular Service-Oriented Web Architecture (MVC pattern)** built on Flask, with pre-loaded singletons for the deep learning runtime.

### 2.1 UML Class Architecture Diagram (Figure 4.2)

#### 📊 Mermaid Diagram (For Screenshots & Markdown Renderers)

```mermaid
classDiagram
    direction TB

    class Config {
        +str BASE_DIR
        +str MODEL_PATH
        +tuple IMG_SIZE = (224, 224)
        +set ALLOWED_EXTENSIONS
        +int MAX_FILE_SIZE_MB = 10
        +float CLASSIFICATION_THRESHOLD = 0.3
    }

    class ApplicationFactory {
        +create_app(config_class) Flask
    }

    class PredictRoutes {
        +handle_predict() HTTPResponse
        +download_report(filename) HTTPResponse
    }

    class ValidationService {
        +is_allowed_file(filename) bool
        +is_valid_file_size(file_path) bool
    }

    class ImageService {
        +preprocess_image(image_input) tuple
    }

    class ModelService {
        -_MODEL_INSTANCE
        +load_model() Model
        +predict(model, img) tuple
    }

    class GradCAMService {
        +generate_gradcam(...) tuple
    }

    class PDFService {
        +generate_patient_pdf_report(...) str
    }

    ApplicationFactory ..> Config : uses
    ApplicationFactory --> PredictRoutes : registers blueprint
    ApplicationFactory ..> ModelService : pre-loads model singleton

    PredictRoutes ..> ValidationService : validates
    PredictRoutes ..> ImageService : resizes/normalizes
    PredictRoutes ..> ModelService : infers
    PredictRoutes ..> GradCAMService : visual overlays
    PredictRoutes ..> PDFService : generates PDF
```

#### 🖼️ Visual Diagram Image

![High-Level System Class Architecture Diagram](file:///d:/collage%20project/Brain-Tumor-Detection-Model-using-Keras/docs/class_architecture.png)

#### 📝 System Architecture Layer Breakdown (ASCII Box Map)

```
+-----------------------------------------------------------------------------------+
|                           APPLICATION CONFIG & FACTORY                            |
|  - Config (IMG_SIZE, MODEL_PATH, CLASSIFICATION_THRESHOLD=0.3, UPLOAD_FOLDER)     |
|  - create_app(Config) -> Flask App Instance (Model pre-loaded on startup)         |
+-----------------------------------------------------------------------------------+
                                         |
                                         v
+-----------------------------------------------------------------------------------+
|                         CONTROLLER BLUEPRINTS (app/routes)                        |
|  - PredictRoutes (/predict, /download_report/<filename>)                          |
|  - MainRoutes (/ index page upload form)                                          |
|  - StatsRoutes (/stats test dataset benchmark metrics & ROC curves)               |
|  - HealthRoutes (/health runtime diagnostics JSON endpoint)                       |
+-----------------------------------------------------------------------------------+
                                         |
       +---------------------------------+---------------------------------+
       |                                 |                                 |
       v                                 v                                 v
+-------------------+          +-------------------+          +-------------------+
| ValidationService |          |   ImageService    |          |   ModelService    |
| - is_allowed_file |          | - preprocess_image|          | - load_model()    |
| - is_valid_size   |          |   (224x224 RGB)   |          | - predict()       |
+-------------------+          +-------------------+          +-------------------+
                                                                           |
       +-------------------------------------------------------------------+
       |                                 |
       v                                 v
+-------------------+          +-------------------+
|  GradCAMService   |          |    PDFService     |
| - JET Heatmap     |          | - ReportLab PDF   |
| - HUD Contour Box |          |   Patient Report  |
+-------------------+          +-------------------+
```

---

### 2.2 Detailed Class Responsibilities & Design Patterns

#### 1. Configuration & Factory (`Config`, `create_app`)
- **Design Pattern**: Application Factory Pattern & Centralized Config.
- **Responsibilities**:
  - `Config`: Centralizes environment configurations, thresholds (`CLASSIFICATION_THRESHOLD = 0.3`), image input shape `(224, 224)`, and folder path structures.
  - `create_app()`: Instantiates the Flask web app, initializes storage directories, pre-loads the Keras model instance into memory on app boot, and registers modular HTTP blueprints.

#### 2. Model Service (`ModelService`)
- **Design Pattern**: Singleton Pattern.
- **Responsibilities**:
  - Maintains `_MODEL_INSTANCE` at module-level scope to avoid re-loading the 67MB `.keras` binary file on every request.
  - `predict()`: Computes raw sigmoid probability output, applies the medical safety margin threshold (`0.3`), and returns formatted classification strings (`"Tumor"` / `"No Tumor"`) and percentage confidence scores.

#### 3. Visual Explainability Engine (`GradCAMService`)
- **Design Pattern**: Strategy & Image Processing Service.
- **Responsibilities**:
  - Constructs a functional model mapping model inputs to the final convolutional layer (`conv2d_7`) and penultimate layer output (`dense_3`).
  - Utilizes `tf.GradientTape` to compute activation map gradients w.r.t linear logits.
  - Calculates channel-pooled weights, applies ReLU, and normalizes feature focus maps.
  - Performs Otsu binary thresholding on original images to generate skull/head tissue masks, preventing background artifact noise.
  - Blends colorized OpenCV JET colormaps for heatmaps (`heatmap_*.jpg`).
  - Isolates top 50% peak focus activation ROIs, generates semi-transparent red highlights, yellow contour boundaries, outer HUD framing brackets, crosshair target reticles, and model attention badges (`highlight_*.jpg`).

#### 4. Diagnostic Report Generator (`PDFService`)
- **Design Pattern**: Flowable Builder Pattern.
- **Responsibilities**:
  - Synthesizes professional single-page patient diagnostic reports using ReportLab (`SimpleDocTemplate`).
  - Assembles styled story elements: Header banner, metadata grid (Report ID, analysis timestamp, model engine), diagnostic result status card (color-coded red/green), 3-image side-by-side comparison grid (Original, Highlight, Heatmap), test performance summary (98% accuracy, 100% recall, 0 false negatives), and medical liability disclaimer.

#### 5. Controller Blueprints (`PredictRoutes`, `StatsRoutes`, `MainRoutes`, `HealthRoutes`)
- **Design Pattern**: MVC Controller / Modular Blueprint Pattern.
- **Responsibilities**:
  - Orchestrates client HTTP requests, form parsing, validation guards, multi-service execution, and template rendering (`index.html`, `result.html`, `stats.html`, `error.html`).

---

### 2.3 Component Interaction & Execution Sequence

#### 📊 Mermaid Sequence Diagram (For Screenshots & Markdown Renderers)

```mermaid
sequenceDiagram
    autonumber
    actor User as 👤 User / Radiologist
    participant Web as 🌐 PredictRoutes
    participant Val as 🛡️ ValidationService
    participant Img as 🖼️ ImageService
    participant Model as 🧠 ModelService
    participant XAI as 🔍 GradCAMService
    participant PDF as 📄 PDFService
    participant Store as 💾 Storage

    User->>Web: POST /predict (file upload)
    
    Web->>Val: is_allowed_file(filename)
    Val-->>Web: True
    Web->>Store: Save file to app/static/uploads/
    Web->>Val: is_valid_file_size(upload_path)
    Val-->>Web: True

    Web->>Img: preprocess_image(upload_path)
    Img-->>Web: (processed_img tensor, orig_pil_img)

    Web->>Model: predict(model_instance, processed_img)
    Model-->>Web: (prediction_label, confidence)

    Web->>XAI: generate_gradcam(...)
    XAI->>Store: Write heatmap_*.jpg & highlight_*.jpg
    XAI-->>Web: (heatmap_path, highlight_path)

    Web->>PDF: generate_patient_pdf_report(...)
    PDF->>Store: Write report_*.pdf
    PDF-->>Web: pdf_path

    Web-->>User: Render result.html HTML Response
```
