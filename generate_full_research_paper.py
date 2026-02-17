
from fpdf import FPDF
import datetime
import os
import random

# Import the content from our content module
import paper_content as content

class ResearchPaperPDF(FPDF):
    def header(self):
        if self.page_no() > 1:  # No header on title page
            self.set_font('Times', 'I', 10)
            self.set_text_color(128, 128, 128)
            self.cell(0, 10, 'Deepfake Detection using ResNet-18 and TTA', 0, 0, 'L')
            self.cell(0, 10, f'Page {self.page_no()}', 0, 0, 'R')
            self.ln(15)
            self.set_text_color(0, 0, 0)

    def footer(self):
        self.set_y(-15)
        self.set_font('Times', 'I', 8)
        self.set_text_color(128, 128, 128)
        self.cell(0, 10, 'Major Project Report - Deepfake Detection System', 0, 0, 'C')
        self.set_text_color(0, 0, 0)

    def chapter_title(self, num, label):
        self.set_font('Times', 'B', 18)
        self.set_fill_color(240, 240, 240)
        self.cell(0, 12, f"Chapter {num}: {label}", 0, 1, 'L', fill=True)
        self.ln(10)

    def section_title(self, label):
        self.set_font('Times', 'B', 14)
        self.cell(0, 10, label, 0, 1, 'L')
        self.ln(2)

    def subsection_title(self, label):
        self.set_font('Times', 'B', 12)
        self.cell(0, 10, label, 0, 1, 'L')
        self.ln(1)

    def chapter_body(self, text):
        self.set_font('Times', '', 12)
        # 1.5 line spacing approx (6mm height * 1.5)
        self.multi_cell(0, 8, text)
        self.ln(5)

    def add_figure(self, image_path, title, width=150):
        if os.path.exists(image_path):
            self.ln(10)
            # Center image
            # Get current page width
            page_width = self.w - 2*self.l_margin
            
            # Check if image fits within page width, if not scale down
            if width > page_width:
                width = page_width
                
            x = (self.w - width) / 2
            
            # Check if enough space on page, else add page
            if self.get_y() + 100 > self.h - 20: # Rough estimate for image height
                self.add_page()
                
            self.image(image_path, x=x, w=width)
            self.ln(5)
            self.set_font('Times', 'BI', 11)
            self.cell(0, 10, title, 0, 1, 'C')
            self.ln(10)

def generate_pdf():
    pdf = ResearchPaperPDF()
    pdf.set_auto_page_break(auto=True, margin=20)
    pdf.set_left_margin(25)
    pdf.set_right_margin(25)
    
    # --- Title Page ---
    pdf.add_page()
    pdf.ln(30)
    
    # University Logo Placeholder (Optional)
    # pdf.image('university_logo.png', x=90, w=30)
    # pdf.ln(10)

    pdf.set_font('Times', 'B', 22)
    pdf.multi_cell(0, 12, "Robust Deepfake Detection via Attention-Driven CNN and Test-Time Augmentation", 0, 'C')
    pdf.ln(20)
    
    pdf.set_font('Times', '', 16)
    pdf.cell(0, 10, "A Major Project Report Submitted", 0, 1, 'C')
    pdf.cell(0, 10, "in Partial Fulfillment of the Requirements for the Degree of", 0, 1, 'C')
    pdf.ln(5)
    
    pdf.set_font('Times', 'B', 16)
    pdf.cell(0, 10, "Bachelor of Technology", 0, 1, 'C')
    pdf.ln(5)
    pdf.set_font('Times', '', 16)
    pdf.cell(0, 10, "in", 0, 1, 'C')
    pdf.set_font('Times', 'B', 16)
    pdf.cell(0, 10, "Computer Science & Engineering", 0, 1, 'C')
    
    pdf.ln(20)
    pdf.set_font('Times', '', 14)
    pdf.cell(0, 10, "By", 0, 1, 'C')
    pdf.ln(5)
    pdf.set_font('Times', 'B', 16)
    pdf.cell(0, 10, "Vasu Tandon", 0, 1, 'C')
    
    pdf.ln(25)
    pdf.set_font('Times', 'B', 14)
    pdf.cell(0, 10, "Department of Computer Science & Engineering", 0, 1, 'C')
    pdf.set_font('Times', '', 14)
    pdf.cell(0, 10, datetime.date.today().strftime("%B, %Y"), 0, 1, 'C')
    
    # --- Certificate / Declaration Page (Standard in Project Reports) ---
    pdf.add_page()
    pdf.ln(10)
    pdf.set_font('Times', 'B', 18)
    pdf.cell(0, 10, "DECLARATION", 0, 1, 'C')
    pdf.ln(20)
    pdf.set_font('Times', '', 12)
    declaration_text = (
        "I hereby declare that the work presented in this major project report entitled "
        "'Robust Deepfake Detection via Attention-Driven CNN and Test-Time Augmentation' "
        "is an authentic record of my own work carried out under the guidance of my supervisor. "
        "The matter embodied in this report has not been submitted by me for the award of any other "
        "degree or diploma to any other University or Institute."
    )
    pdf.multi_cell(0, 10, declaration_text)
    pdf.ln(40)
    pdf.cell(0, 10, "(Vasu Tandon)", 0, 1, 'R')
    pdf.cell(0, 10, "Date: " + datetime.date.today().strftime("%d-%m-%Y"), 0, 1, 'L')

    # --- Acknowledgement ---
    pdf.add_page()
    pdf.ln(10)
    pdf.set_font('Times', 'B', 18)
    pdf.cell(0, 10, "ACKNOWLEDGEMENT", 0, 1, 'C')
    pdf.ln(20)
    pdf.set_font('Times', '', 12)
    ack_text = (
        "I would like to express my sincere gratitude to my supervisor, for their invaluable guidance, "
        "continuous encouragement, and support throughout the course of this project.\n\n"
        "I also wish to thank the Head of the Department, Computer Science & Engineering, for providing "
        "the necessary facilities and environment to carry out this work.\n\n"
        "Finally, I am grateful to my family and friends for their unwavering support and motivation."
    )
    pdf.multi_cell(0, 10, ack_text)

    # --- Abstract ---
    pdf.add_page()
    pdf.ln(10)
    pdf.set_font('Times', 'B', 18)
    pdf.cell(0, 10, "ABSTRACT", 0, 1, 'C')
    pdf.ln(10)
    pdf.set_font('Times', '', 12)
    pdf.multi_cell(0, 8, content.ABSTRACT)
    
    # --- Table of Contents ---
    pdf.add_page()
    pdf.ln(10)
    pdf.set_font('Times', 'B', 18)
    pdf.cell(0, 10, "TABLE OF CONTENTS", 0, 1, 'C')
    pdf.ln(10)
    
    # Simple manual TOC for typical report
    pdf.set_font('Times', '', 12)
    toc = [
        ("1. Introduction", "1"),
        ("   1.1 Problem Statement", ""),
        ("   1.2 Motivation", ""),
        ("   1.3 Scope of the Project", ""),
        ("   1.4 Objectives", ""),
        ("2. Literature Review", ""),
        ("   2.1 Early Detection Methods", ""),
        ("   2.2 CNN-Based Approaches", ""),
        ("   2.3 Frequency Domain Analysis", ""),
        ("   2.4 Transformers & Attention", ""),
        ("   2.5 Explainability Gaps", ""),
        ("3. Proposed Methodology", ""),
        ("   3.1 Dataset Selection", ""),
        ("   3.2 Preprocessing Pipeline", ""),
        ("   3.3 Model Architecture (ResNet-18)", ""),
        ("   3.4 Loss Function", ""),
        ("   3.5 Test-Time Augmentation", ""),
        ("   3.6 Grad-CAM Integration", ""),
        ("4. System Architecture", ""),
        ("   4.1 High-Level Design", ""),
        ("   4.2 Component Architecture", ""),
        ("   4.3 Data Flow Pipeline", ""),
        ("5. Implementation Details", ""),
        ("   5.1 Software Stack", ""),
        ("   5.2 Hardware Specifications", ""),
        ("   5.3 Training Configuration", ""),
        ("6. Results & Analysis", ""),
        ("   6.1 Evaluation Metrics", ""),
        ("   6.2 Training Performance", ""),
        ("   6.3 Confusion Matrix Analysis", ""),
        ("   6.4 Impact of TTA", ""),
        ("   6.5 Comparison with SOTA", ""),
        ("   6.6 Grad-CAM Analysis", ""),
        ("7. Conclusion & Future Work", ""),
        ("8. References", "")
    ]
    
    for title, page in toc:
        if not page:
            pdf.cell(10) # Indent subsections
        pdf.cell(0, 8, title, 0, 1, 'L')
        # Note: We are not calculating exact page numbers here as it's dynamic
        # but in a real report you'd do a 2-pass generation or estimate.

    # --- Chapter 1: Introduction ---
    pdf.add_page()
    pdf.chapter_title(1, "Introduction")
    pdf.chapter_body(content.INTRO_1)
    
    pdf.add_figure("paper_assets/deepfake_example_placeholder.png", "Figure 1.1: Example of High-Quality Deepfake vs Real Face") # This file might not exist, placeholder logic in add_figure handles it safely or we add a real one? We generated charts, not "deepfake example".
    
    pdf.chapter_body(content.INTRO_2)
    pdf.chapter_body(content.INTRO_3)

    # --- Chapter 2: Literature Review ---
    pdf.add_page()
    pdf.chapter_title(2, "Literature Review")
    pdf.chapter_body(content.LIT_REVIEW_1)
    pdf.add_figure("paper_assets/model_comparison.png", "Figure 2.1: Accuracy Comparison with Existing Baselines")
    pdf.chapter_body(content.LIT_REVIEW_2)
    pdf.chapter_body(content.LIT_REVIEW_3)

    # --- Chapter 3: Methodology ---
    pdf.add_page()
    pdf.chapter_title(3, "Proposed Methodology")
    pdf.chapter_body(content.METHODOLOGY_1)
    pdf.add_figure("paper_assets/dataset_dist.png", "Figure 3.1: Dataset Distribution (Balanced Classes)")
    pdf.chapter_body(content.METHODOLOGY_2)
    pdf.chapter_body(content.METHODOLOGY_3)
    
    # We can add a placeholder block diagram here if we had one
    # pdf.add_figure("paper_assets/resnet_arch.png", "Figure 3.2: ResNet-18 Architecture")

    pdf.chapter_body(content.METHODOLOGY_4)

    # --- Chapter 4: System Architecture ---
    pdf.add_page()
    pdf.chapter_title(4, "System Architecture")
    pdf.chapter_body(content.SYS_ARCH)
    
    # --- Chapter 5: Implementation ---
    pdf.add_page()
    pdf.chapter_title(5, "Implementation Details")
    pdf.chapter_body(content.IMPLEMENTATION)

    # --- Chapter 6: Results ---
    pdf.add_page()
    pdf.chapter_title(6, "Results & Analysis")
    pdf.chapter_body(content.RESULTS_1)
    
    pdf.add_figure("paper_assets/accuracy_plot.png", "Figure 6.1: Training and Validation Accuracy over 4 Epochs")
    
    pdf.chapter_body(content.RESULTS_2)
    
    pdf.add_figure("paper_assets/loss_plot.png", "Figure 6.2: Training and Validation Loss Convergence")
    
    pdf.chapter_body(content.RESULTS_3)
    
    pdf.add_figure("paper_assets/confusion_matrix.png", "Figure 6.3: Confusion Matrix Evaluation on Test Set")
    
    # --- Chapter 7: Conclusion ---
    pdf.add_page()
    pdf.chapter_title(7, "Conclusion & Future Work")
    pdf.chapter_body(content.CONCLUSION)

    # --- Chapter 8: References ---
    pdf.add_page()
    pdf.chapter_title(8, "References")
    
    pdf.set_font('Times', '', 11)
    for ref in content.REFERENCES:
        pdf.multi_cell(0, 8, ref)
        pdf.ln(3)

    output_filename = "Deepfake_Detection_Major_Project_Report.pdf"
    pdf.output(output_filename)
    print(f"✅ Generated Full Report: {output_filename}")

if __name__ == "__main__":
    generate_pdf()
