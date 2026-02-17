
from fpdf import FPDF
import datetime
import os
import paper_content as content

class CondensedResearchPaperPDF(FPDF):
    def header(self):
        if self.page_no() > 1:
            self.set_font('Times', 'I', 9)
            self.cell(0, 10, 'Deepfake Detection - Major Project Report', 0, 0, 'L')
            self.cell(0, 10, f'{self.page_no()}', 0, 0, 'R')
            self.ln(12)

    def footer(self):
        self.set_y(-15)
        self.set_font('Times', '', 8)
        self.cell(0, 10, 'Department of Computer Science & Engineering', 0, 0, 'C')

    def chapter_title(self, label):
        self.set_font('Times', 'B', 14)
        self.cell(0, 8, label.upper(), 0, 1, 'L')
        self.ln(2)

    def section_title(self, label):
        self.set_font('Times', 'B', 12)
        self.cell(0, 6, label, 0, 1, 'L')
        self.ln(1)

    def body_text(self, text):
        self.set_font('Times', '', 11)
        self.multi_cell(0, 5, text) # Tighter line spacing
        self.ln(3)

    def add_figure(self, image_path, title, height=60):
        if os.path.exists(image_path):
            self.ln(2)
            # Center image
            x = (210 - 100) / 2 # Approx center for 100mm width
            self.image(image_path, x=x, h=height)
            self.ln(1)
            self.set_font('Times', 'I', 9)
            self.cell(0, 5, title, 0, 1, 'C')
            self.ln(4)

def generate_pdf():
    pdf = CondensedResearchPaperPDF()
    pdf.set_auto_page_break(auto=True, margin=15)
    pdf.set_margins(20, 20, 20)
    
    # --- Title Page (Page 1) ---
    pdf.add_page()
    pdf.ln(40)
    pdf.set_font('Times', 'B', 20)
    pdf.multi_cell(0, 10, "Robust Deepfake Detection via Attention-Driven CNN and Test-Time Augmentation", 0, 'C')
    pdf.ln(15)
    
    pdf.set_font('Times', '', 14)
    pdf.cell(0, 10, "A Major Project Report", 0, 1, 'C')
    pdf.ln(20)
    
    pdf.set_font('Times', 'B', 16)
    pdf.cell(0, 10, "Vasu Tandon", 0, 1, 'C')
    pdf.ln(10)
    
    pdf.set_font('Times', '', 12)
    pdf.cell(0, 10, "Department of Computer Science & Engineering", 0, 1, 'C')
    pdf.cell(0, 10, datetime.date.today().strftime("%B %Y"), 0, 1, 'C')
    
    pdf.ln(30)
    pdf.set_font('Times', 'B', 12)
    pdf.multi_cell(0, 6, "ABSTRACT\n\n", 0, 'C')
    pdf.set_font('Times', '', 11)
    pdf.multi_cell(0, 5, content.ABSTRACT)
    
    # --- Introduction (Page 2) ---
    pdf.add_page()
    pdf.chapter_title("1. Introduction")
    pdf.body_text(content.INTRO_1)
    pdf.section_title("1.1 Problem Statement")
    pdf.body_text(content.INTRO_2) # Contains problem statement text
    
    # --- Lit Review (Page 3-4) ---
    pdf.chapter_title("2. Literature Review")
    pdf.body_text(content.LIT_REVIEW_1)
    
    # Place graphic here to save space
    pdf.add_figure("paper_assets/model_comparison.png", "Fig 1: Accuracy Comparison", height=50)
    
    pdf.body_text(content.LIT_REVIEW_2)
    pdf.body_text(content.LIT_REVIEW_3)

    # --- Methodology (Page 5-7) ---
    pdf.chapter_title("3. Proposed Methodology")
    pdf.section_title("3.1 Dataset & Preprocessing")
    pdf.body_text(content.METHODOLOGY_1)
    pdf.add_figure("paper_assets/dataset_dist.png", "Fig 2: Dataset Classes", height=45)
    pdf.body_text(content.METHODOLOGY_2)
    
    pdf.section_title("3.2 Model Architecture (ResNet-18)")
    pdf.body_text(content.METHODOLOGY_3)
    
    pdf.section_title("3.3 Test-Time Augmentation (TTA)")
    pdf.body_text(content.METHODOLOGY_4)

    # --- System Architecture (Page 8) ---
    pdf.chapter_title("4. System Architecture")
    pdf.body_text(content.SYS_ARCH)
    
    # --- Implementation (Page 9) ---
    pdf.chapter_title("5. Implementation Details")
    pdf.body_text(content.IMPLEMENTATION)

    # --- Results (Page 10-11) ---
    pdf.chapter_title("6. Results & Analysis")
    pdf.body_text(content.RESULTS_1)
    
    # Side-by-side plots logic (simulated by sequential small plots)
    pdf.add_figure("paper_assets/accuracy_plot.png", "Fig 3: Accuracy Curve", height=50)
    pdf.body_text(content.RESULTS_2)
    
    pdf.add_figure("paper_assets/confusion_matrix.png", "Fig 4: Confusion Matrix", height=50)
    pdf.body_text(content.RESULTS_3)
    
    # --- Conclusion (Page 11-12) ---
    pdf.chapter_title("7. Conclusion")
    pdf.body_text(content.CONCLUSION)

    # --- References (Page 12) ---
    pdf.chapter_title("8. References")
    pdf.set_font('Times', '', 9)
    for i, ref in enumerate(content.REFERENCES):
        if i > 12: break # Limit refs to fit 12 pages if needed
        pdf.multi_cell(0, 4, ref)
        pdf.ln(1)

    output_filename = "Deepfake_Detection_Report_12Pages.pdf"
    pdf.output(output_filename)
    print(f"✅ Generated 12-Page Report: {output_filename}")

if __name__ == "__main__":
    generate_pdf()
