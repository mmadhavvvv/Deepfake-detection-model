from fpdf import FPDF
import datetime

class PDF(FPDF):
    def header(self):
        self.set_font('Arial', 'B', 15)
        self.cell(0, 10, 'Deepfake Detection using Attention-Driven CNN and TTA', 0, 1, 'C')
        self.ln(5)

    def footer(self):
        self.set_y(-15)
        self.set_font('Arial', 'I', 8)
        self.cell(0, 10, 'Page ' + str(self.page_no()) + '/{nb}', 0, 0, 'C')

def create_pdf():
    pdf = PDF()
    pdf.alias_nb_pages()
    pdf.add_page()
    pdf.set_auto_page_break(auto=True, margin=15)
    pdf.set_font("Times", size=12)

    # Title and Author
    pdf.set_font("Times", 'B', 16)
    pdf.cell(0, 10, "Robust Deepfake Detection via ResNet-18 with", 0, 1, 'C')
    pdf.cell(0, 10, "Explainable Grad-CAM and Test-Time Augmentation", 0, 1, 'C')
    pdf.ln(5)
    pdf.set_font("Times", 'I', 12)
    pdf.cell(0, 10, "Vasu Tandon", 0, 1, 'C')
    pdf.cell(0, 10, datetime.date.today().strftime("%B %d, %Y"), 0, 1, 'C')
    pdf.ln(10)

    # Abstract
    pdf.set_font("Times", 'B', 12)
    pdf.cell(0, 10, "Abstract", 0, 1, 'L')
    pdf.set_font("Times", '', 12)
    pdf.multi_cell(0, 6, 
        "The proliferation of hyper-realistic deepfake content poses a significant threat "
        "to digital media integrity. This paper presents a lightweight yet robust deepfake "
        "detection system based on the ResNet-18 architecture, optimized for resource-constrained "
        "environments. Our approach integrates Grad-CAM (Gradient-weighted Class Activation Mapping) "
        "for model explainability and employs a Test-Time Augmentation (TTA) strategy involving "
        "horizontal flips and center crops to enhance prediction reliability. We introduce an "
        "'Aggressive Detection' thresholding mechanism to minimize false negatives in high-stakes "
        "scenarios. Trained on a balanced subset of 40,000 images from the '140k Real and Fake Faces' "
        "dataset, the model demonstrates high efficiency and reliable classification performance."
    )
    pdf.ln(5)

    # 1. Introduction
    pdf.set_font("Times", 'B', 12)
    pdf.cell(0, 10, "1. Introduction", 0, 1, 'L')
    pdf.set_font("Times", '', 12)
    pdf.multi_cell(0, 6, 
        "Deep learning advancements have democratized the creation of synthetic media, necessitating "
        "equally advanced detection mechanisms. While large transformer-based models offer high accuracy, "
        "they often require significant computational resources. This research focuses on optimizing "
        "a convolutional neural network (CNN) for efficiency without compromising detection capabilities. "
        "We utilize a modified ResNet-18 backbone and introduce a resource-aware training loop that "
        "includes active cooling pauses to maintain hardware stability during training on standard laptops. "
        "Furthermore, we address the 'black box' nature of CNNs by embedding Grad-CAM visualization directly "
        "into the inference pipeline, allowing users to see which facial features triggered a 'fake' classification."
    )
    pdf.ln(5)

    # 2. Methodology
    pdf.set_font("Times", 'B', 12)
    pdf.cell(0, 10, "2. Methodology", 0, 1, 'L')
    pdf.set_font("Times", 'I', 12)
    pdf.cell(0, 10, "2.1 Dataset and Preprocessing", 0, 1, 'L')
    pdf.set_font("Times", '', 12)
    pdf.multi_cell(0, 6, 
        "We utilized the '140k Real and Fake Faces' dataset, curating a balanced subset of 20,000 real "
        "and 20,000 fake images for training. Images were resized to 224x224 pixels and normalized using "
        "ImageNet standards (mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]). A custom "
        "'FastDeepfakeDataset' class was implemented to handle efficient data loading with error resilience."
    )
    pdf.ln(5)

    pdf.set_font("Times", 'I', 12)
    pdf.cell(0, 10, "2.2 Model Architecture", 0, 1, 'L')
    pdf.set_font("Times", '', 12)
    pdf.multi_cell(0, 6, 
        "The core architecture is a ResNet-18 model, chosen for its balance between depth and computational "
        "cost. The final fully connected layer was adapted for binary classification (Real vs. Fake). "
        "We utilized Binary Cross Entropy with Logits Loss (BCEWithLogitsLoss) and the AdamW optimizer "
        "(lr=5e-5) for stable convergence."
    )
    pdf.ln(5)

    pdf.set_font("Times", 'I', 12)
    pdf.cell(0, 10, "2.3 Test-Time Augmentation (TTA)", 0, 1, 'L')
    pdf.set_font("Times", '', 12)
    pdf.multi_cell(0, 6, 
        "To improve inference robustness, we implemented TTA. For every input image, the model predicts on "
        "three variations: (1) the original image, (2) a horizontally flipped version, and (3) a 90% center-cropped "
        "zoom. The final confidence score is the average of these three probabilities. This ensemble-like approach "
        "reduces susceptibility to noise and pose variations."
    )
    pdf.ln(5)

    pdf.set_font("Times", 'I', 12)
    pdf.cell(0, 10, "2.4 Explainability via Grad-CAM", 0, 1, 'L')
    pdf.set_font("Times", '', 12)
    pdf.multi_cell(0, 6, 
        "We integrated Grad-CAM by hooking into the final convolutional layer (layer4[-1]) of the ResNet backbone. "
        "This generates a heatmap highlighting the regions most influential in the model's decision, "
        "providing interpretability for end-users."
    )
    pdf.ln(5)

    # 3. Experimental Results
    pdf.set_font("Times", 'B', 12)
    pdf.cell(0, 10, "3. Experimental Results & Discussion", 0, 1, 'L')
    pdf.set_font("Times", '', 12)
    pdf.multi_cell(0, 6, 
        "The model was trained for 4 epochs with a batch size of 16. To mitigate overfitting and ensure "
        "high detection rates for manipulated content, we implemented an 'Aggressive Mode' during inference. "
        "Any image with a fake probability score greater than 0.30 via TTA is flagged as 'DEEPFAKE'. "
        "This lowers the threshold for detection, prioritizing recall for security-critical applications. "
        "During validation, the model achieved consistent accuracy improvements, demonstrating the effectiveness "
        "of the TTA strategy in refining borderline predictions."
    )
    pdf.ln(5)

    # 4. Conclusion
    pdf.set_font("Times", 'B', 12)
    pdf.cell(0, 10, "4. Conclusion", 0, 1, 'L')
    pdf.set_font("Times", '', 12)
    pdf.multi_cell(0, 6, 
        "We successfully developed a deepfake detection system that balances performance with interpretability. "
        "By combining a lightweight ResNet-18 backbone with Test-Time Augmentation and Grad-CAM, the system "
        "provides reliable and explainable predictions suitable for deployment on consumer hardware. Future work "
        "will explore temporal analysis for video-based deepfake detection."
    )
    pdf.ln(5)

    # References
    pdf.set_font("Times", 'B', 12)
    pdf.cell(0, 10, "References", 0, 1, 'L')
    pdf.set_font("Times", '', 10)
    pdf.multi_cell(0, 5, 
        "[1] K. He, X. Zhang, S. Ren, and J. Sun, 'Deep Residual Learning for Image Recognition,' CVPR, 2016.\n"
        "[2] R. R. Selvaraju et al., 'Grad-CAM: Visual Explanations from Deep Networks via Gradient-based Localization,' ICCV, 2017.\n"
        "[3] Xhlulu, '140k Real and Fake Faces Dataset,' Kaggle, 2020."
    )

    pdf.output("Deepfake_Detection_Research_Paper.pdf")
    print("PDF Generated Successfully: Deepfake_Detection_Research_Paper.pdf")

if __name__ == "__main__":
    create_pdf()
