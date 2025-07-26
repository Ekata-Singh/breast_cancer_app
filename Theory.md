# 📚 Theoretical Background

Understanding the medical and diagnostic context behind the dataset enhances both model performance and its real-world relevance.

---

## 🔬 Tumor Types

### Benign Tumor
- Non-cancerous
- Encapsulated and non-invasive
- Slow-growing
- Does **not metastasize** (spread)
- Cells are **normal in shape and size**

### Malignant Tumor
- **Cancerous**
- Non-capsulated, **invasive**
- Fast-growing
- Can **metastasize** to other parts of the body
- Cells often have **large, dark nuclei** and **irregular shapes**

> 🎯 **Goal of the model**: Accurately classify whether a tumor is **benign** or **malignant** based on measurable features.

---

## 💉 What is Fine Needle Aspiration (FNA)?

**Fine Needle Aspiration** is a type of biopsy technique where:
- A **thin needle** is inserted into the suspected tumor
- **Cell samples** are drawn and **digitally imaged**
- Features such as **radius, texture, concavity**, etc., are extracted from these images

> 🧠 The dataset used in this project is derived from **cell images obtained via FNA**.

---

## 📊 Feature Explanation: Breast Cancer Dataset Columns

| Column(s) | Description |
|-----------|-------------|
| `id` | Unique identifier (not used in model) |
| `diagnosis` | Target variable: `M` = Malignant, `B` = Benign |
| `*_mean` | Mean value of the feature (first 10 columns) |
| `*_se` | Standard Error of the feature |
| `*_worst` | Worst (maximum) value of the feature |

### 🔍 Feature Types

| Feature | Meaning |
|--------|---------|
| `radius` | Distance from center to perimeter |
| `texture` | Variation in pixel intensity |
| `perimeter` | Circumference of the cell |
| `area` | Total area covered by the nucleus |
| `smoothness` | Edge smoothness (local variation) |
| `compactness` | Shape compactness (perimeter²/area - 1) |
| `concavity` | Depth of concave portions |
| `concave points` | Number of concave edges |
| `symmetry` | How symmetrical the cell is |
| `fractal_dimension` | Complexity of the border (like in fractals) |

> These features are critical in distinguishing cancerous vs non-cancerous cell behavior.

---

## 💡 Summary

- This theoretical understanding ensures that model decisions are rooted in medical relevance.
- Knowing what each feature means allows for better **feature selection**, **interpretation**, and possibly **explainable AI (XAI)** extensions in the future.
