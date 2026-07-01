import os
from reportlab.lib.pagesizes import letter
from reportlab.platypus import SimpleDocTemplate, Paragraph, Spacer
from reportlab.lib.styles import getSampleStyleSheet

def main():
    pdf_path = "./data/test_scripture.pdf"
    os.makedirs("./data", exist_ok=True)
    
    print(f"Creating test PDF at: {pdf_path}")
    doc = SimpleDocTemplate(pdf_path, pagesize=letter)
    story = []
    
    styles = getSampleStyleSheet()
    title_style = styles["Heading1"]
    body_style = styles["BodyText"]
    
    # Page 1 Title
    story.append(Paragraph("The Story of Rama and the Bow", title_style))
    story.append(Spacer(1, 20))
    
    # Story text rich in ontology terms
    paragraphs = [
        "In the ancient days, there was a prince known as Rama, the pride of Raghu's race, lotus-eyed and strong of arm. "
        "Accompanied by his loyal brother Lakshmana, Rama journeyed through the hermitage of Gautama and freed Ahalya from her long curse. "
        "Shatánanda, the eldest son of Gautam, praised Rama for this glorious deed. Flowers rained from the sky and the holy hermit "
        "Gautama was reconciled with his wife, who returned to share his austerities.",
        
        "The mighty saint Vishvamitra, son of Gádhi of Kusha's line, acted as their guide on the journey to Mithilá. "
        "Mithilá, also known as Visálá, was ruled by Janaka, the best of kings, high-souled and lofty-souled. Janaka had planned a great "
        "sacrifice, an enclosed area prepared with thousands of Bráhmans and noble visitors.",
        
        "At Janak's court, there was a Famous bow of wondrous virtue. Many kings had tried to lift this weapon and failed. "
        "Janaka had declared that whoever could bend the Famous bow would win the hand of Sita, the Maithil dame, princess of Videha. "
        "Rama, with his peerless strength, lifted the bow, bent it, and broke it in twain. Thus Sita became the wife of Rama, "
        "and their wedding was celebrated on the sacrificial ground with great joy."
    ]
    
    for para in paragraphs:
        story.append(Paragraph(para, body_style))
        story.append(Spacer(1, 15))
        
    doc.build(story)
    print("Test PDF created successfully.")

if __name__ == "__main__":
    main()
