"""
Document Processor — extracts text from PDFs and defines the structured
strategic objectives and action items for HHS Vision 2030 & QIP 2025-26.

This module serves as the DATA FOUNDATION of the ISPS system.

Key Functions:
1. Define data structures (StrategicObjective, ActionItem dataclasses)
2. Hard-code strategic plan content (Vision 2030)
3. Hard-code action plan content (QIP 2025-26)
4. Provide utility functions for PDF extraction and text chunking

Why Hard-Code Instead of PDF Parsing?
    - Reliability: PDF parsing is fragile (tables, multi-column layouts break)
    - Accuracy: Manual extraction ensures correct interpretation
    - Structure: Allows adding semantic metadata (codes, sections, relationships)
    - Performance: No runtime PDF parsing overhead
    - Maintainability: Easy to update if documents change

Trade-off:
    Initial effort to manually extract data, but eliminates ongoing parsing issues.
"""

from dataclasses import dataclass, field
from pypdf import PdfReader


# ────────────────────────────────────────────────────────────────────
# Data Classes (Structured Representation)
# ────────────────────────────────────────────────────────────────────

@dataclass
class StrategicObjective:
    """
    Represents a high-level strategic aim from HHS Vision 2030.
    
    Attributes:
        code: Short identifier (e.g., "LEAD", "BUILD")
        name: Full name of the strategic objective
        description: One-sentence summary of what it aims to achieve
        focus_areas: List of specific focus points under this objective
    
    Example:
        StrategicObjective(
            code="LEAD",
            name="Lead in Research & Innovation",
            description="We will lead in research...",
            focus_areas=["Bringing new discoveries...", "Embedding research..."]
        )
    """
    code: str
    name: str
    description: str
    focus_areas: list[str] = field(default_factory=list)


@dataclass
class ActionItem:
    """
    Represents a concrete action from HHS QIP 2025-26.
    
    Attributes:
        id: Unique identifier (e.g., "QIP-1", "AF-2")
        section: Category (e.g., "QIP Priority", "Access & Flow")
        title: Short title of the action
        description: Full description of what the action entails
        details: Optional list of sub-actions or implementation details
    
    Example:
        ActionItem(
            id="QIP-1",
            section="QIP Priority",
            title="Reduce workplace violence",
            description="Implement strategies to reduce...",
            details=["Training programs", "Reporting system"]
        )
    """
    id: str
    section: str
    title: str
    description: str
    details: list[str] = field(default_factory=list)


# ────────────────────────────────────────────────────────────────────
# HHS Vision 2030 — Strategic Plan (Hard-Coded from PDF)
# ────────────────────────────────────────────────────────────────────

VISION = (
    """Hamilton Health Sciences' aspirational statement of their desired future state."""
    "To shape the future of health by leading in care, discovery, "
    "and learning, while advancing equity and regional growth."
)

MISSION = (
    """Concise statement of HHS's core purpose."""
    "Leading care, driven by discovery."
)

# HHS's four core values that guide organizational culture and decisions.
# These values inform all strategic objectives and operational decisions.
VALUES = {
    "Collaboration": "We work together to create the best results.",
    "Compassion": "We ensure every person feels seen and supported.",
    "Advancement": "We continuously seek better ways to work and care.",
    "Accountability": "We are accountable for the strength of our decisions.",
}

# The four main strategic aims of HHS Vision 2030.
# 
# These are manually extracted from the Vision 2030 strategic plan PDF
# (downloaded from hamiltonhealthsciences.ca/vision2030/).
# 
# Each objective includes:
# - A memorable code (LEAD, BUILD, SERVE, LEARN)
# - A descriptive name
# - A summary description
# - 2 focus areas that operationalize the objective
# 
# These objectives form the "strategic" side of the synchronization analysis.
# The system evaluates how well the QIP action items support these goals.
STRATEGIC_OBJECTIVES: list[StrategicObjective] = [
    StrategicObjective(
        code="LEAD",
        name="Lead in Research & Innovation",
        description="We will lead in research and innovation to continually set new standards of care in our areas of expertise.",
        focus_areas=[
            "Bringing new discoveries to patients faster and ensuring all patients have an opportunity to benefit from emerging evidence.",
            "Embedding research, innovation and continuous improvement in our decision-making.",
        ],
    ),
    StrategicObjective(
        code="BUILD",
        name="Build Regional Opportunities",
        description="We will build new opportunities in our region for health and growth.",
        focus_areas=[
            "Advancing our region's growing life sciences hub.",
            "Partnering to create the infrastructure needed to advance care and research.",
        ],
    ),
    StrategicObjective(
        code="SERVE",
        name="Serve with Quality & Equity",
        description="We will serve every patient and caregiver's diverse needs with high quality, accessible and equitable care.",
        focus_areas=[
            "Delivering the highest standard of evidence-based care, every time.",
            "Working with partners to ensure seamless, equitable care across our system.",
        ],
    ),
    StrategicObjective(
        code="LEARN",
        name="Learn & Grow Together",
        description="We will learn and grow together to become the top Canadian hospital to build an impactful career.",
        focus_areas=[
            "Advancing our unique role in health education and research.",
            "Developing our people to create a sustainable and skilled workforce.",
        ],
    ),
]

# Three cross-cutting commitments that underpin all strategic objectives.
# 
# These represent HHS's fundamental principles that apply across all aims.
# They are treated as pseudo-strategic objectives in the knowledge graph
# and evaluation to assess their alignment with actions.
FOUNDATIONAL_COMMITMENTS = [
    {"name": "Health Equity", "code": "FC-HEA",
     "description": "We will build an inclusive system that ensures accessible, high-quality, and compassionate care for all."},
    {"name": "Indigenous Truth & Reconciliation", "code": "FC-IND",
     "description": "We will build genuine, respectful relationships with Indigenous communities grounded in trust, understanding, and shared healing."},
    {"name": "Research & Innovation", "code": "FC-RES",
     "description": "We will accelerate research and innovation to improve lives."},
]

# Supporting strategic plans that operationalize Vision 2030.
CORE_PLANS = [
    "Capital Development Plan", "Clinical Service Plan", "Digital Health Plan",
    "Equity, Diversity, and Inclusion Plan", "Indigenous Health Plan",
    "People Plan", "Quality and Safety Plan", "Research and Innovation Plan",
]


# ────────────────────────────────────────────────────────────────────
# QIP 2025-26 — Action Plan (Hard-Coded from PDF)
# ────────────────────────────────────────────────────────────────────

ACTION_ITEMS: list[ActionItem] = [
    # ── QIP Priorities ──
    ActionItem("QIP-1", "QIP Priority", "Maintain Reduced Rate of Sepsis",
               "Maintain the organisation's reduced rate of sepsis through continued monitoring and evidence-based interventions.",
               ["Focus on decreasing sepsis and contributing infections",
                "Central line infections prevention",
                "Catheter-associated urinary tract infections prevention",
                "Surgical site infections prevention"]),
    ActionItem("QIP-2", "QIP Priority", "Reduce Deaths Following Major Surgery",
               "Reduce mortality rates following major surgical procedures through quality improvement initiatives.",
               ["Evidence-based surgical protocols",
                "Post-operative monitoring improvements",
                "Surgical safety checklist compliance"]),
    ActionItem("QIP-3", "QIP Priority", "Reduce Incidence of Pressure Injuries",
               "Reduce pressure injuries with equity-informed skin assessment and prevention strategies.",
               ["Participation in Never Events reporting to Ontario Health",
                "Equity-informed skin assessment tool for darker skin tones",
                "Innovative solutions for pressure injury prevention"]),
    ActionItem("QIP-4", "QIP Priority", "Maintain Improved Hand Hygiene Rates",
               "Reduce risk of hospital-acquired infections through maintaining improved hand hygiene compliance.",
               ["Ongoing hand hygiene monitoring",
                "Staff education and awareness campaigns",
                "Infection prevention and control measures"]),
    ActionItem("QIP-5", "QIP Priority", "Expand EDI CARE Data Initiative",
               "Expand the Collecting Accurate and Robust Equity data initiative to improve health equity insights.",
               ["Expansion to over 260 outpatient/ambulatory clinics",
                "Equity data collection standardisation",
                "Use of data to identify and address health disparities"]),
    ActionItem("QIP-6", "QIP Priority", "Reduce Overdue Discharge Summaries",
               "Reduce the number of overdue discharge summaries and operative notes.",
               ["Discharge documentation process improvement",
                "Accountability measures for timely completion",
                "System-wide tracking and monitoring"]),
    ActionItem("QIP-7", "QIP Priority", "Collect Race/Ethnicity Data in Critical Incidents",
               "Start collecting race, ethnicity and language data as part of critical incident reviews.",
               ["Integration of demographic data into safety reviews",
                "Identification of disproportionate safety impacts",
                "Equity-informed safety improvement actions"]),

    # ── Access & Flow ──
    ActionItem("AF-1", "Access & Flow", "Improve ED Efficiency & Wait Times",
               "Implement initiatives to improve Emergency Department efficiency and reduce patient wait times.",
               ["Triage Assessment Teams at Juravinski and Hamilton General",
                "Early assessment and management of patients",
                "Treatment at Triage process improvements"]),
    ActionItem("AF-2", "Access & Flow", "Improve Inpatient Flow & Transitions",
               "Reduce length of stay and improve transitions from hospital to community care.",
               ["Discharge planning optimisation",
                "Rapid outpatient follow-up programmes",
                "Virtual care for improving outcomes at home"]),
    ActionItem("AF-3", "Access & Flow", "Innovate Models of Care",
               "Develop and implement innovative care delivery models across HHS sites.",
               ["Expanded step-down models",
                "Hospitalist models",
                "Short stay surgical models",
                "Care coordination enhancements through Epic system"]),

    # ── Equity & Indigenous Health ──
    ActionItem("EI-1", "Equity & Indigenous Health", "EDI Training & Capacity Building",
               "Comprehensive equity, diversity and inclusion training and organisational culture change.",
               ["Introductory EDI training for all formal leaders (74 % completion)",
                "Ethics & Diversity Grand Rounds",
                "5-year EDI Plan 2023-2028 implementation"]),
    ActionItem("EI-2", "Equity & Indigenous Health", "Black Health Initiatives",
               "Implement Black health initiatives at McMaster Children's Hospital.",
               ["Enhancing culturally responsive clinical care",
                "Community partner integration for Black populations",
                "Innovative solutions for darker skin tone assessment",
                "Race-neutral equation for pulmonary function testing"]),
    ActionItem("EI-3", "Equity & Indigenous Health", "Indigenous Health Plan Development",
               "Develop a comprehensive Indigenous Health Plan through community engagement.",
               ["Led by Indigenous Strategic Advisor",
                "Aligned with Truth & Reconciliation Calls to Action",
                "United Nations Declaration on Rights of Indigenous Peoples"]),

    # ── Patient Experience ──
    ActionItem("PE-1", "Patient Experience", "Patient & Family Advisory Programme",
               "Strengthen patient and family engagement through advisory councils and partnerships.",
               ["Roster of ~100 patient and family advisors",
                "5 patient and family advisory councils",
                "Over 1 740 hours contributed (Apr 2024 – Feb 2025)"]),
    ActionItem("PE-2", "Patient Experience", "Electronic Patient Experience Surveying",
               "Implement electronic surveying across emergency and inpatient areas.",
               ["Electronic surveys for real-time feedback",
                "Plans to expand to ambulatory services"]),
    ActionItem("PE-3", "Patient Experience", "Video Interpretation Services",
               "Deploy video interpretation at bedside for 200+ languages including ASL.",
               ["Available within seconds at bedside",
                "All inpatient and emergency units covered"]),

    # ── Provider Experience ──
    ActionItem("PX-1", "Provider Experience", "Care Transformation Initiative",
               "Organisation-wide initiative to standardise care models and optimise team roles.",
               ["Standardised criteria for care models across sites",
                "Skill mix model optimisation",
                "Clear role definition and supportive education",
                "Continual evaluation of patient and staff outcomes"]),

    # ── Safety ──
    ActionItem("S-1", "Safety", "Safety Huddles & Proactive Safety",
               "Spread standardised safety huddles and proactive safety culture across HHS.",
               ["Standardised safety huddles across all sites",
                "Daily review of safety risks",
                "Solutions for Patient Safety collaborative (paediatric)"]),
    ActionItem("S-2", "Safety", "Rethinking Patient Safety Programme",
               "Healthcare Excellence Canada pilot broadening the view of hospital harm.",
               ["Proactive approach to patient safety",
                "Supported by four clinical units and patient advisors",
                "Incorporating EDI metrics into safety work"]),

    # ── Palliative Care ──
    ActionItem("PC-1", "Palliative Care", "Palliative Care Excellence",
               "Maintain and expand palliative care services as a leader in quality improvement.",
               ["Expert end-of-life care at St. Peter's Hospital",
                "Patient and family-driven decision-making",
                "Direct access to inpatient care 7 days/week"]),
    ActionItem("PC-2", "Palliative Care", "Keaton's House Children's Hospice",
               "Partner with Kemp Care Network to build the region's first children's hospice.",
               ["Region's first children's hospice",
                "Care for children with life-limiting conditions",
                "Home-like, child-friendly setting"]),

    # ── Population Health Management ──
    ActionItem("PH-1", "Population Health", "Greater Hamilton Health Network (GHHN)",
               "Partner with 40+ organisations in the Ontario Health Team to transform regional healthcare.",
               ["40+ partner organisations",
                "One of 12 OHTs with advanced work plan",
                "Co-design patient-centred health system",
                "Integrated primary care planning"]),
    ActionItem("PH-2", "Population Health", "HealthPathways & Community Wellness Hubs",
               "Launch evidence-based clinical pathways and community wellness hubs.",
               ["HealthPathways for clinicians (real-time, evidence-based)",
                "Two Community Wellness Hubs launched fall 2024",
                "Integrated care model for older adults"]),
]


# ────────────────────────────────────────────────────────────────────
# Utility Functions
# ────────────────────────────────────────────────────────────────────

def extract_text_from_pdf(pdf_path: str) -> str:
    """
    Extract all text content from a PDF file.
    
    Args:
        pdf_path: File path to the PDF
    
    Returns:
        str: Concatenated text from all pages, separated by double newlines
    
    Implementation:
        Uses pypdf library to read pages and extract text.
        Skips empty pages automatically.
    
    Limitations:
        - May not handle complex layouts (multi-column, tables) perfectly
        - Image-based PDFs (scanned documents) won't extract text
        - For this reason, we hard-code the content instead of relying on this
    
    Use Case:
        Available for extensibility (adding supplementary documents),
        but not used for primary strategic/action plan content.
    """
    reader = PdfReader(pdf_path)
    return "\n\n".join(
        page.extract_text() for page in reader.pages if page.extract_text()
    )


def chunk_text(text: str, size: int = 400, overlap: int = 80) -> list[str]:
    """
    Split text into overlapping chunks for better context in RAG.
    
    Chunking is a critical RAG technique: it breaks long documents into
    smaller pieces that can be:
    1. Individually embedded (each chunk gets a vector)
    2. Retrieved independently (return only relevant chunks, not whole doc)
    3. Fit within LLM context windows (chunks are smaller than full docs)
    
    Args:
        text: Input text to chunk
        size: Number of words per chunk (default 400 ≈ 500-600 characters)
        overlap: Number of words to overlap between chunks (default 80)
    
    Returns:
        list[str]: List of text chunks
    
    Overlap Rationale:
        Overlap ensures that information near chunk boundaries isn't lost.
        Example: If a sentence spans two chunks, overlap captures it in both.
    
    Chunk Size Trade-offs:
        - Too small (100 words): Loses context, fragments ideas
        - Too large (1000 words): Less precise retrieval, exceeds some embeddings
        - 400 words: Good balance for semantic understanding
    
    Algorithm:
        1. Split text into words
        2. Take windows of `size` words, stepping by (size - overlap)
        3. Rejoin words with spaces
        4. Skip empty chunks
    """
    words = text.split()
    chunks = []
    # Step through text with overlap
    # range(start, stop, step) where step = size - overlap
    for i in range(0, len(words), size - overlap):
        # Extract window of words
        chunk = " ".join(words[i : i + size])
        if chunk.strip():  # Skip empty chunks
            chunks.append(chunk)
    return chunks


def get_strategic_texts() -> list[dict]:
    """
    Prepare strategic items for embedding.
    
    Combines STRATEGIC_OBJECTIVES and FOUNDATIONAL_COMMITMENTS into a
    single list with consistent structure for embedding.
    
    Returns:
        list[dict]: Each dict contains:
            - id: Strategic code (e.g., "LEAD", "FC-HEA")
            - text: Full concatenated text for embedding
            - type: "strategic_aim" or "foundational_commitment"
    
    Text Construction:
        Concatenates name + description + focus areas into a single string.
        This gives the embedding model full context to understand the objective.
        
        Example:
            "Lead in Research & Innovation: We will lead in research and 
             innovation to continually set new standards of care... Bringing 
             new discoveries to patients faster..."
    
    Why Flatten to List?
        The embedding model expects a flat list of strings, not nested objects.
        This function transforms the structured data into embedding-ready format.
    """
    items: list[dict] = []
    
    # Add strategic objectives
    for obj in STRATEGIC_OBJECTIVES:
        items.append({
            "id": obj.code,
            # Concatenate all text fields for comprehensive embedding
            "text": f"{obj.name}: {obj.description} " + " ".join(obj.focus_areas),
            "type": "strategic_aim",
        })
    
    # Add foundational commitments (treated as additional strategic items)
    for fc in FOUNDATIONAL_COMMITMENTS:
        items.append({
            "id": fc["code"],
            "text": f"{fc['name']}: {fc['description']}",
            "type": "foundational_commitment",
        })
    
    return items


def get_action_texts() -> list[dict]:
    """
    Prepare action items for embedding.
    
    Converts ACTION_ITEMS list into a flat structure suitable for embedding.
    
    Returns:
        list[dict]: Each dict contains:
            - id: Action ID (e.g., "QIP-1", "AF-2")
            - text: Full concatenated text for embedding
            - section: Category (e.g., "QIP Priority", "Access & Flow")
            - type: "action_item"
    
    Text Construction:
        Concatenates title + description + details into a single string.
        This captures the full intent and implementation of each action.
        
        Example:
            "Reduce workplace violence: Implement strategies to reduce 
             workplace violence... Training programs Reporting system..."
    
    Section Field:
        Preserved for filtering and grouping in visualizations.
        Allows analyzing alignment by QIP section.
    """
    return [
        {
            "id": item.id,
            # Concatenate all text fields
            "text": f"{item.title}: {item.description} " + " ".join(item.details),
            "section": item.section,
            "type": "action_item",
        }
        for item in ACTION_ITEMS
    ]
