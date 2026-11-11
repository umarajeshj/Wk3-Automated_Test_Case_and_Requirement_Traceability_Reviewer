from pypdf import PdfWriter
from reportlab.lib.pagesizes import letter
from reportlab.platypus import SimpleDocTemplate, Paragraph, Spacer
from reportlab.lib.styles import getSampleStyleSheet, ParagraphStyle

def generate_sample_pdf(filename="sample_prd.pdf"):
    doc = SimpleDocTemplate(filename, pagesize=letter)
    styles = getSampleStyleSheet()
    story = []

    title_style = ParagraphStyle(
        'TitleStyle',
        parent=styles['Heading1'],
        fontSize=18,
        spaceAfter=12
    )
    
    heading_style = ParagraphStyle(
        'HeadingStyle',
        parent=styles['Heading2'],
        fontSize=14,
        spaceBefore=12,
        spaceAfter=6
    )
    
    body_style = styles['Normal']

    # Content
    story.append(Paragraph("Product Requirement Document & Test Suite", title_style))
    story.append(Paragraph("Feature Module: User Authentication & Payment Gateway", heading_style))
    story.append(Paragraph("Module Priority: P0 (Critical)", body_style))
    story.append(Spacer(1, 10))

    story.append(Paragraph("1. Requirements Overview", heading_style))
    story.append(Paragraph("• REQ-01: System must allow secure user login via OAuth2. (Priority: P0)", body_style))
    story.append(Paragraph("• REQ-02: Multi-factor authentication (MFA) must be enforced for admin roles. (Priority: P0)", body_style))
    story.append(Paragraph("• REQ-03: Payment processing via Stripe must support tokenized credit cards. (Priority: P0)", body_style))
    story.append(Paragraph("• REQ-04: System shall log all failed login attempts for security auditing. (Priority: P1)", body_style))
    story.append(Spacer(1, 10))

    story.append(Paragraph("2. Test Cases & Traceability Matrix", heading_style))
    story.append(Paragraph("• TC-101 (Maps to REQ-01): Verify successful login with valid credentials. (Type: Positive)", body_style))
    story.append(Paragraph("• TC-102 (Maps to REQ-01): Verify login failure with invalid password. (Type: Negative)", body_style))
    story.append(Paragraph("• TC-103 (Maps to REQ-02): Verify MFA challenge prompt on admin login. (Type: Positive)", body_style))
    story.append(Paragraph("• TC-104 (Maps to REQ-03): Verify successful payment execution with Stripe. (Type: Positive)", body_style))
    story.append(Paragraph("• TC-105 (Maps to REQ-04): Verify security audit logs record lockouts correctly. (Type: Negative)", body_style))
    story.append(Spacer(1, 10))

    story.append(Paragraph("3. Summary Statistics", heading_style))
    story.append(Paragraph("Total Requirements: 4", body_style))
    story.append(Paragraph("Total Test Cases: 5", body_style))
    story.append(Paragraph("Mapped Requirements: 4", body_style))
    story.append(Paragraph("Critical (P0) Requirements: 3", body_style))
    story.append(Paragraph("Unmapped Critical Requirements: 0", body_style))
    story.append(Paragraph("Negative Test Cases: 2", body_style))

    doc.build(story)
    print(f"Successfully generated {filename}")

if __name__ == "__main__":
    generate_sample_pdf()