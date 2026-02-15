# Intelligent Strategic Plan Synchronization System (ISPS)

## Hamilton Health Sciences Vision 2030 vs Quality Improvement Plan 2025-26

**Programme:** MSc Data Science  
**Module:** Information Retrieval  
**Submission Type:** Individual Coursework Report  
**Student:** _[Your Name]_  
**Student ID:** _[Your ID]_  
**Date:** _[Submission Date]_  
**Live Application Link:** _[Add Streamlit/HuggingFace URL]_  
**GitHub Repository:** _[Add GitHub URL]_  

---

## Abstract

Modern organizations produce high-level strategic plans and lower-level action plans, but alignment between these documents is often assessed manually, which is slow, subjective, and difficult to scale. This report presents an Intelligent Strategic Plan Synchronization System (ISPS) that automatically evaluates alignment between a strategic plan and an action plan using Natural Language Processing (NLP), vector embeddings, retrieval-augmented analysis, and visual analytics.

The implemented case study uses public documents from Hamilton, Ontario: Hamilton Health Sciences (HHS) Vision 2030 (strategic plan) and HHS Quality Improvement Plan (QIP) 2025-26 (action plan). The system computes semantic similarity between strategic objectives and action items, provides strategy-level and overall synchronization metrics, identifies alignment gaps, and offers AI-generated improvement recommendations. It also includes an interactive knowledge graph and a testing/evaluation module with precision, recall, F1-score, and threshold optimization.

The current prototype reports an overall synchronization score of **51.0%**, indicating moderate alignment with identifiable gaps. At a default threshold of 0.45, the system achieves precision 0.625, recall 0.125, and F1 0.208; threshold tuning shows optimal F1 at 0.35 (F1 0.50). These outcomes demonstrate that embedding-based methods can provide practical, explainable support for strategic decision-making while preserving a human-in-the-loop governance model.

---

## 1. Introduction and Background

Organizations in healthcare increasingly depend on formal strategic documents to guide long-term outcomes and operational plans to drive near-term execution. A recurring challenge is strategic drift: action-level activities can gradually become misaligned with strategic intent due to resource constraints, policy changes, and local optimization decisions. Traditional review methods depend on committees and periodic manual audits, which are time-intensive and can miss semantic patterns across large text corpora.

This project addresses that challenge by building an AI-powered dashboard that measures synchronization between:

- **Strategic Plan:** Hamilton Health Sciences Vision 2030.
- **Action Plan:** Hamilton Health Sciences Quality Improvement Plan (QIP) 2025-26.

The key objective is to operationalize strategic alignment analysis through:

1. Embedding-based semantic matching.
2. Strategy-wise traceability from goals to actions.
3. Retrieval-augmented and LLM-assisted improvement suggestions.
4. Explainable visual outputs for decision-makers.

The system is implemented as a web dashboard for practical use by quality teams, planning officers, and healthcare executives. It is designed to remain transparent, lightweight, and easy to deploy.

---

## 2. Problem Statement

The core problem is: **How well does the operational action plan (QIP 2025-26) support the strategic intent articulated in Vision 2030?**

This problem has three dimensions:

- **Coverage:** Are strategic priorities represented by concrete actions?
- **Depth:** Is each strategic objective supported strongly enough?
- **Quality:** Are there meaningful actions missing that should be introduced?

A robust solution must therefore provide both quantitative and qualitative outputs: numerical alignment metrics plus actionable recommendations.

---

## 3. Objectives

The project objectives were defined to align directly with coursework requirements:

1. Build an end-to-end system that ingests strategic and action plan content.
2. Compute overall and objective-level synchronization metrics using NLP.
3. Detect weak alignment zones and visualize them.
4. Generate intelligent improvement suggestions for poorly aligned objectives.
5. Introduce innovative AI features (agentic reasoning + knowledge graph).
6. Evaluate system performance against a manually curated ground truth.
7. Provide deployment-ready architecture and practical security guidance.

---

## 4. Literature Review

### 4.1 Information Retrieval and Semantic Matching

Classical IR approaches based on bag-of-words and TF-IDF are effective for lexical overlap but weaker for semantic equivalence when different wording expresses similar intent. In strategic planning texts, this is common (e.g., “health equity,” “inclusive access,” and “care disparities” may refer to related concepts). Semantic embeddings therefore provide stronger matching than exact token overlap.

### 4.2 Sentence Embeddings

Sentence-BERT introduced efficient sentence-level embeddings suitable for semantic similarity tasks and clustering (Reimers and Gurevych, 2019). For this project, a compact model (`all-MiniLM-L6-v2`) was selected due to:

- strong semantic performance,
- low inference cost,
- practical local execution.

### 4.3 Retrieval-Augmented Generation (RAG)

RAG combines retrieval with generation by grounding LLM outputs in retrieved document context, reducing generic or unsupported responses (Lewis et al., 2020). This is especially useful for governance contexts where recommendations must remain tied to source documents.

### 4.4 Vector Databases

Vector stores such as ChromaDB support nearest-neighbor retrieval over embeddings. They are useful for:

- mapping queries to most relevant strategic/action fragments,
- explainability (showing retrieved context),
- powering chat and recommendation modules.

### 4.5 Knowledge Graphs

Knowledge graphs provide structured, interpretable representations of relationships among entities and concepts. In planning analysis, they improve explainability by showing which action items support which strategic objectives, including commitments and values.

### 4.6 LLMs in Decision Support

Large language models can assist with summarization, suggestion generation, and alternative scenario reasoning. However, reliability constraints require human validation and contextual grounding. This system therefore uses LLM outputs as advisory artifacts, not final decisions.

---

## 5. Dataset and Domain Context

### 5.1 Domain Choice

Healthcare was selected due to:

- high societal impact,
- rich public strategic documentation,
- clear operational quality indicators.

### 5.2 Input Documents

1. **HHS Vision 2030 Booklet** (Strategic Plan)
2. **HHS QIP Narrative 2025-26** (Action Plan)

Both are public documents from Hamilton Health Sciences and satisfy coursework expectations for realistic organizational context.

### 5.3 Structured Representation

From the strategic plan, the system models:

- 4 strategic aims: **LEAD, BUILD, SERVE, LEARN**
- 3 foundational commitments
- organizational values

From the action plan, the system models 23 action items across themes such as:

- QIP priorities,
- access and flow,
- equity and Indigenous health,
- patient/provider experience,
- safety,
- palliative care,
- population health.

---

## 6. Methodology and System Architecture

### 6.1 Pipeline Overview

The implemented pipeline is:

1. **Document processing**  
   PDF text extraction and normalization.

2. **Text representation**  
   Conversion of strategic and action entries into sentence embeddings.

3. **Similarity analysis**  
   Cosine similarity matrix between strategic items and action items.

4. **Vector indexing**  
   ChromaDB storage for retrieval and downstream RAG features.

5. **Analytics + visualization**  
   Per-strategy metrics, heatmaps, radar plots, gap analysis.

6. **Intelligent layer**  
   LLM suggestions, executive summaries, agentic reasoning, interactive chat.

7. **Evaluation layer**  
   Ground-truth comparison and threshold tuning.

### 6.2 Core Mathematical Formulation

Let:

- \( S = \{s_1, s_2, ..., s_n\} \) be strategic items,
- \( A = \{a_1, a_2, ..., a_m\} \) be action items.

Each text is embedded into vector space:

- \( e(s_i) \in \mathbb{R}^d \)
- \( e(a_j) \in \mathbb{R}^d \)

Similarity matrix:

\[
M_{ij} = \cos(e(s_i), e(a_j))
\]

Overall synchronization score:

\[
\text{Overall} = \frac{1}{n}\sum_{i=1}^{n}\max_j(M_{ij}) \times 100
\]

Strategy-level score uses top-k averages (k=3) to reduce noise from single matches.

### 6.3 Technology Stack

- **Frontend/UI:** Streamlit + Plotly
- **NLP Embeddings:** sentence-transformers
- **Vector DB:** ChromaDB
- **LLM Layer:** Google Gemini (free tier)
- **Graph Layer:** NetworkX + PyVis
- **Evaluation:** scikit-learn

### 6.4 Architecture Diagram (Textual)

Strategic Plan + Action Plan -> Document Processor -> Embeddings -> ChromaDB + Similarity Matrix -> Dashboard analytics + RAG + Agentic reasoning + Evaluation.

**Figure 1. System Architecture Diagram (replace with exported image before final PDF):**

![Figure 1 - ISPS Architecture Diagram](./figures/figure1-isps-architecture.png)

*Caption guidance:* The diagram should show end-to-end flow from input PDFs to NLP embedding, vector retrieval, similarity scoring, LLM recommendation layer, dashboard visualization, and evaluation outputs.  
*Action needed:* Export your architecture from draw.io/Lucidchart and save it as `./figures/figure1-isps-architecture.png`.

### 6.5 Detailed Methodology Decisions and Justification

This section documents why specific technical choices were made, what alternatives were considered, and the trade-offs involved.

#### 6.5.1 Why sentence-level embeddings instead of keyword matching

Keyword overlap methods (for example TF-IDF with cosine similarity) were considered for the baseline because they are lightweight and easy to interpret. However, the planning domain often uses semantically related but lexically different wording. For instance, strategic language may mention "equitable care pathways" while operational actions mention "collect race, ethnicity and language data during incident reviews." Lexical systems under-score such pairs, while sentence embeddings provide better semantic generalization.

Embedding-based matching was therefore selected as the primary mechanism, with the option of adding TF-IDF as an additional baseline in future work. This decision improved semantic coverage at the cost of model download size and slightly higher startup time.

#### 6.5.2 Why `all-MiniLM-L6-v2` model

The project needed a model that:

- is free and practical on a student laptop,
- has strong semantic textual similarity performance,
- works offline for core analysis.

`all-MiniLM-L6-v2` was selected as a strong balance of quality and efficiency. Larger models may improve nuanced reasoning but would reduce responsiveness and complicate deployment.

#### 6.5.3 Why ChromaDB instead of managed vector services

ChromaDB was used because it is lightweight, local-first, and simple to integrate in Streamlit without additional cloud infrastructure. Managed vector databases such as Pinecone were considered but were not necessary for this scale (single-case-study document set). Local ChromaDB improved reproducibility and reduced operational complexity, while still satisfying the assignment requirement to use vector-database-style retrieval.

#### 6.5.4 Why threshold-based gap detection

The assignment requires identifying weak strategic-action fit. A threshold approach is transparent and easy for decision-makers to understand:

- if similarity >= threshold -> aligned,
- if similarity < threshold -> potential gap.

This also allows governance flexibility: teams can tune threshold based on risk appetite. For exploratory planning, recall may be prioritized (lower threshold). For conservative reporting, precision may be prioritized (higher threshold).

#### 6.5.5 Why top-3 average for strategy scoring

A single highest similarity can overestimate alignment due to one strong but narrow action. Using top-3 average captures broader support and reduces sensitivity to outliers. This choice gave a more stable representation of strategy coverage, especially for objectives with multiple relevant operational actions.

#### 6.5.6 Why optional LLM layer

The solution was intentionally designed so core analysis works without any API key. LLM functionality is optional and used for:

- suggestion synthesis,
- executive summarization,
- exploratory Q&A.

This design keeps costs low, improves accessibility for coursework demonstrations, and avoids making the whole system dependent on paid APIs.

#### 6.5.7 Why include knowledge graph despite small dataset

Even with a moderate number of entities, the knowledge graph provides an explainability layer. Stakeholders can visually inspect strategic coverage and conceptual relationships (aims, commitments, values, and actions). This supports trust and communication beyond raw metrics.

#### 6.5.8 Risk and bias handling choices

Methodologically, the project includes safeguards:

- explicit reminder that LLM outputs are advisory,
- evaluation against manually curated ground truth,
- threshold tuning to expose precision-recall trade-offs,
- discussion of potential annotation subjectivity.

These choices improve methodological transparency and align with responsible AI practice.

---

## 7. Implementation Details

### 7.1 Application Modules

- `src/document_processor.py`  
  PDF extraction and canonical strategic/action structures.

- `src/analyzer.py`  
  Embedding generation, similarity computation, gap analysis, vector queries.

- `src/llm_helper.py`  
  Gemini integration for recommendations, executive reports, chat, and agentic reasoning.

- `src/knowledge_graph.py`  
  Entity/relationship graph construction and interactive rendering.

- `src/evaluation.py`  
  Precision/recall/F1 calculations and threshold sweep.

### 7.2 Dashboard Pages

1. **Home:** headline metrics and quick profile.
2. **Synchronization Analysis:** heatmap, radar, strategy scores, threshold gap checks.
3. **Strategy Deep Dive:** top supporting actions and weakly aligned actions.
4. **Improvement Suggestions:** LLM recommendations and executive report generation.
5. **Knowledge Graph:** interactive strategic-action map.
6. **Evaluation:** confusion matrix and threshold optimization.
7. **Chat With Plans:** RAG-grounded question answering.

### 7.3 Simplicity Design Choices

To keep the system simple and submission-friendly:

- local embedding model (no paid dependency for core features),
- one-click Streamlit app execution,
- optional API key only for advanced AI features,
- modular but minimal codebase.

### 7.4 Development Phases and Screenshot Placeholders

This subsection should be retained in the final PDF with actual screenshots inserted. The coursework rubric explicitly expects development evidence with visuals.

#### Phase 1: Problem Scoping and Document Selection

**What to describe:** Why Hamilton Health Sciences was selected; why Vision 2030 and QIP 2025-26 are suitable and realistic.  
**Add screenshot placeholder:**

![Figure 2 - Input documents in project folder](./figures/figure2-input-documents.png)

*Caption template:* "Strategic and action plan source documents loaded into the project workspace."

#### Phase 2: Data Structuring and Preprocessing

**What to describe:** PDF extraction process, strategic objective structuring, action item extraction and normalization.  
**Add screenshot placeholder:**

![Figure 3 - Structured strategic and action records](./figures/figure3-data-structuring.png)

*Caption template:* "Structured representation of strategic aims and action items used for embedding."

#### Phase 3: Embedding and Similarity Engine

**What to describe:** sentence-transformer loading, embedding generation, cosine similarity matrix calculation.  
**Add screenshot placeholder:**

![Figure 4 - Similarity computation output](./figures/figure4-similarity-engine.png)

*Caption template:* "Similarity engine producing objective-action alignment scores."

#### Phase 4: Vector Database and Retrieval Layer

**What to describe:** ChromaDB collections for strategic and action corpora; nearest-neighbor retrieval behavior.  
**Add screenshot placeholder:**

![Figure 5 - ChromaDB retrieval example](./figures/figure5-vector-retrieval.png)

*Caption template:* "Vector retrieval results for an objective-level query."

#### Phase 5: Dashboard Interface Implementation

**What to describe:** Home metrics, strategy deep dive, heatmap/radar visualization, gap diagnostics.  
**Add screenshot placeholders:**

![Figure 6 - Home dashboard](./figures/figure6-home-dashboard.png)

![Figure 7 - Synchronization heatmap](./figures/figure7-heatmap.png)

![Figure 8 - Strategy deep dive](./figures/figure8-deep-dive.png)

*Caption template:* "Interactive dashboard views for executive and analyst users."

#### Phase 6: Intelligent Features

**What to describe:** RAG suggestions, agentic analysis chain, interactive knowledge graph, chat interface.  
**Add screenshot placeholders:**

![Figure 9 - AI suggestions panel](./figures/figure9-ai-suggestions.png)

![Figure 10 - Knowledge graph visualization](./figures/figure10-knowledge-graph.png)

![Figure 11 - Chat with plans interface](./figures/figure11-chat.png)

*Caption template:* "Intelligent modules supporting recommendation generation and explainability."

#### Phase 7: Testing and Evaluation

**What to describe:** confusion matrix, threshold sweep, precision/recall trade-offs, selected operating threshold.  
**Add screenshot placeholder:**

![Figure 12 - Evaluation metrics and threshold tuning](./figures/figure12-evaluation.png)

*Caption template:* "Performance evaluation against manually curated alignment ground truth."

---

## 8. Results and Analysis

### 8.1 Overall Synchronization

- **Overall synchronization score:** **51.0%**

Interpretation: alignment is moderate; the action plan addresses several strategic areas, but coverage depth is uneven across objectives.

### 8.2 Strategy-Level Snapshot

Observed top-3 average alignment scores:

- LEAD: 45.4%
- BUILD: 45.9%
- SERVE: 49.1%
- LEARN: 41.6%
- FC-HEA: 48.6%
- FC-IND: 38.3% (with high best-match outlier)
- FC-RES: 31.6%

The model indicates stronger operational expression for service and equity-related areas than for learning/research translation.

### 8.3 Gap Detection

At default threshold 0.45:

- Gaps detected: **2**
- Strength areas: **5**

This indicates not total misalignment, but targeted areas where strategic intent has weaker operational expression in current action items.

---

## 9. Testing and Evaluation

### 9.1 Ground Truth Strategy

A manually curated alignment map (strategic item -> expected supporting actions) was prepared to evaluate algorithmic predictions.

### 9.2 Default Threshold Performance (0.45)

- Precision: **0.625**
- Recall: **0.125**
- F1-score: **0.208**

Interpretation:

- High precision relative to recall means the model is conservative: predicted links are usually relevant, but many true links are missed.

### 9.3 Threshold Optimization

Threshold sweep results identify best F1 at:

- **Threshold = 0.35**
- Precision: 0.458
- Recall: 0.550
- F1: **0.500**

Implication: lowering threshold improves recall substantially and produces better balance for exploratory planning analysis. A practical deployment can expose threshold as a governance control.

### 9.4 Evaluation Conclusion

The system is suitable as a decision-support tool, not an autonomous replacement for expert review. It helps teams prioritize where to inspect alignment manually.

### 9.5 Extended Error Analysis

To improve transparency, it is useful to examine common error patterns in predicted mappings.

#### 9.5.1 False positives

False positives usually occur when the model detects thematic similarity at a broad level but misses operational specificity. Example pattern:

- Strategic objective includes "innovation in care delivery."
- Action item references "digital care coordination" in a narrow context.

The model may mark this as aligned due to shared semantic neighborhood, even if the action does not materially advance the strategic objective at portfolio scale.

#### 9.5.2 False negatives

False negatives typically arise when strategic wording is abstract and action wording is highly procedural. Example pattern:

- Strategy: "advance equity across the system."
- Action: "collect race/ethnicity/language in incident reviews."

If wording overlap is limited and embedding distance is just below threshold, a meaningful link may be missed.

#### 9.5.3 Section-level imbalance effects

Some QIP sections contain dense operational language (for example safety and flow improvements), which can dominate similarity distributions. This may bias ranking behavior if not normalized by section frequency. A future enhancement would apply section-aware reweighting to reduce dominance effects.

#### 9.5.4 Recommendations from error analysis

1. Add a hybrid similarity score combining embeddings with keyword overlap.
2. Introduce section-aware calibration (for example weighted top-k matching).
3. Include sentence-level evidence snippets in every predicted link.
4. Use multi-rater expert validation to refine ground-truth mappings.
5. Report confidence bands for borderline threshold cases.

### 9.6 Human-in-the-Loop Review Workflow

For practical deployment, this project recommends a structured review workflow:

1. **Automated pass:** System computes candidate mappings and gap list.
2. **Analyst pass:** Planning analyst reviews low-confidence and high-impact links.
3. **Clinical/operational pass:** Domain experts verify strategic relevance.
4. **Governance pass:** Leadership approves revised alignment actions/KPIs.
5. **Re-run cycle:** Updated action proposals are re-evaluated before final plan sign-off.

This review structure balances automation speed with accountable decision-making. It also creates an auditable trail for how recommendations were accepted, modified, or rejected.

---

## 10. Intelligent Features and Innovation

### 10.1 Agentic AI Reasoning

The agentic module performs structured multi-step reasoning:

1. Strategic intent interpretation.
2. Action relevance mapping.
3. Missing element identification.
4. Root cause analysis.
5. Recommendation synthesis.
6. Risk assessment.

This extends beyond static similarity by providing strategic narrative reasoning.

### 10.2 RAG-Based Recommendations

The recommendation engine retrieves relevant chunks and generates:

- gap diagnosis,
- actionable improvements,
- KPI suggestions,
- implementation timelines.

### 10.3 Knowledge Graph

The graph layer improves explainability by exposing structural relationships among:

- strategic aims,
- values,
- commitments,
- action items.

Graph statistics (current build):

- nodes: 34
- edges: 68
- strategic aims: 4
- action items: 23

### 10.4 Ontology Mapping Artifact (Concrete)

To satisfy ontology-oriented innovation requirements, a lightweight conceptual ontology was defined for the case study. This is not a full OWL deployment, but it provides explicit concept classes and relations that can be extended into RDF/OWL in future work.

#### 10.4.1 Concept classes

| Class | Description | Example |
|---|---|---|
| `StrategicAim` | Long-term organizational strategic objective | `LEAD`, `SERVE` |
| `ActionItem` | Operational initiative in QIP | `QIP-1`, `AF-2` |
| `Commitment` | Foundational organizational commitment | `FC-HEA`, `FC-IND` |
| `Value` | Organizational value guiding behavior | `Collaboration` |
| `KPI` | Measurable outcome indicator | Sepsis rate, hand hygiene compliance |
| `Risk` | Negative consequence if gaps persist | Delayed improvement in equity outcomes |

#### 10.4.2 Relation schema

| Relation | Domain -> Range | Meaning |
|---|---|---|
| `supports` | `ActionItem -> StrategicAim` | Action operationally supports an objective |
| `underpins` | `Commitment -> StrategicAim` | Commitment is foundational to objective delivery |
| `guidedBy` | `StrategicAim -> Value` | Objective should be executed under value principles |
| `measuredBy` | `StrategicAim -> KPI` | KPI tracks objective progress |
| `mitigates` | `ActionItem -> Risk` | Action reduces operational or strategic risk |

#### 10.4.3 Example RDF-style triples

```ttl
@prefix isps: <http://example.org/isps#> .

isps:LEAD a isps:StrategicAim ;
    isps:guidedBy isps:Advancement ;
    isps:measuredBy isps:ResearchTranslationRate .

isps:QIP_1 a isps:ActionItem ;
    isps:supports isps:SERVE ;
    isps:mitigates isps:HospitalAcquiredInfectionRisk .

isps:FC_HEA a isps:Commitment ;
    isps:underpins isps:SERVE .

isps:QIP_5 a isps:ActionItem ;
    isps:supports isps:FC_HEA ;
    isps:supports isps:SERVE .
```

#### 10.4.4 Practical use in this project

The ontology layer supports three practical outcomes:

1. **Explainability:** relations can be rendered in graph form for stakeholder interpretation.
2. **Consistency checks:** each strategic aim can be checked for missing `supports` relations.
3. **Future semantic reasoning:** rule-based checks can flag unsupported aims or commitments.

---

## 11. Security, Privacy, and Ethical Considerations

### 11.1 Security Controls in Prototype

- No hardcoded credentials.
- LLM API key is optional and user-provided at runtime.
- Core analytics run offline without API access.
- No sensitive personal health records are processed.

### 11.2 Privacy Considerations

Input documents are public planning documents; nonetheless:

- deployment should enforce HTTPS,
- runtime secrets should be stored in platform secret manager,
- logs should avoid storing raw sensitive prompts in production settings.

### 11.3 Responsible AI Considerations

- LLM outputs can hallucinate or overgeneralize.
- Recommendations should be validated by domain experts before use.
- Evaluation metrics and evidence traces should always accompany AI suggestions.

---

## 12. Hosting Architecture Proposal

### 12.1 Recommended Deployment

- **Frontend + backend app:** Streamlit Cloud (single service).
- **Vector index:** local/in-app ChromaDB for lightweight usage.
- **LLM:** Gemini API via secure secrets.
- **Monitoring:** app logs + usage metrics.

### 12.2 Alternative Enterprise Deployment

For higher compliance:

- Deploy on Azure/AWS container service.
- Managed secret vault for API keys.
- Optional local/private LLM for data sovereignty.
- Role-based access and audit logging.

### 12.3 Data Protection and GDPR-Oriented Notes

Although the case study uses public planning documents rather than personal clinical data, a production-ready version should still implement privacy-by-design controls:

- **Data minimization:** only store necessary text fragments and derived embeddings.
- **Purpose limitation:** restrict use to strategy-alignment analysis and planning support.
- **Access control:** enforce role-based permissions for administrators, analysts, and viewers.
- **Key management:** store API credentials in managed secret vaults, not source code.
- **Auditability:** log access and recommendation generation events for compliance review.
- **Retention policy:** define deletion schedules for uploaded documents and generated outputs.

If adapted to internal documents containing personal or sensitive organizational information, a DPIA-style assessment and legal review should be completed before deployment.

---

## 13. Discussion

The implemented system successfully demonstrates the central coursework thesis: IR and modern NLP methods can make strategic/action plan synchronization measurable and explainable.

Key strengths:

- clear objective-to-action traceability,
- practical dashboard UX,
- integration of multiple intelligent components (RAG, agentic analysis, KG),
- built-in evaluation and threshold governance.

Observed limitations:

- embedding model may miss domain nuance without fine-tuning,
- ground truth is manually curated and may include subjective assumptions,
- LLM quality depends on prompt engineering and API behavior.

Overall, the prototype is suitable as a practical planning intelligence assistant and a strong foundation for enterprise extension.

### 13.1 Deeper Interpretation of Findings

The observed overall score (51.0%) should be interpreted as moderate operational alignment rather than failure. In planning environments, complete alignment is rare because action plans are constrained by budget cycles, compliance requirements, and short-term operational pressures. A mid-range score indicates that strategic intent is partially expressed but unevenly distributed across objectives.

The best-performing areas show that patient-facing and equity-related objectives have stronger action support. This is coherent with healthcare governance realities, where quality and safety indicators are often mandated and therefore well represented in annual plans. Lower-scoring objectives likely reflect either long-horizon ambitions (which are harder to operationalize quickly) or strategic themes that rely on multi-year infrastructure and workforce initiatives.

### 13.2 Precision-Recall Trade-off in Real Governance Use

The evaluation metrics show a clear operational trade-off:

- at threshold 0.45, precision is higher but recall is low (conservative mode),
- at threshold 0.35, recall increases significantly and F1 is stronger (exploratory mode).

For executive planning reviews, a higher-recall setting may be preferable because missing a true gap is often more harmful than reviewing a few additional false positives. For formal reporting with limited review capacity, a higher-precision setting can reduce analyst workload. This demonstrates that threshold should be a governance parameter, not a hard-coded constant.

### 13.3 Methodological Limitations (Expanded)

#### 13.3.1 Document abstraction bias

The system models strategic and action content as summarized entries. While this improves clarity and speed, summarization may compress nuance from full document text (for example, contextual qualifiers around target populations or implementation dependencies).

#### 13.3.2 Semantic ambiguity

Embedding models capture contextual semantics but may conflate closely related healthcare concepts. For example, "quality improvement," "patient safety," and "clinical effectiveness" overlap semantically but can represent distinct governance programs.

#### 13.3.3 Ground-truth subjectivity

Manual mapping is necessary for evaluation but introduces rater bias. Different experts may disagree about what counts as "support" versus "partial support." This impacts measured precision/recall and should be reported transparently.

#### 13.3.4 LLM output variability

Suggestion quality depends on prompt framing, model version, and retrieved context quality. Recommendations can be useful but should not be interpreted as policy without human validation and feasibility assessment.

#### 13.3.5 Limited temporal modeling

The current prototype is cross-sectional (single strategic plan + single annual QIP). It does not model temporal drift, delayed impact, or inter-year strategic momentum.

### 13.4 Validity Threats and Mitigation

| Threat Type | Description | Mitigation Used | Further Improvement |
|---|---|---|---|
| Construct validity | "Alignment" may be interpreted inconsistently | Explicit similarity and threshold definitions | Add expert rubric for alignment categories |
| Internal validity | Results depend on chosen embedding model | Reproducible model and code | Compare with at least one alternative model |
| External validity | Single organization case study | Real-world public documents used | Replicate across additional organizations |
| Reliability | LLM suggestions can vary | Retrieval grounding + fixed prompt structure | Add deterministic evaluation prompts |

### 13.5 Practical Implications for Decision-Makers

From an organizational perspective, the system is most useful as a planning support instrument in quarterly and annual review cycles. Decision-makers can use it to:

- identify where action portfolios are under-supporting strategic commitments,
- prioritize strategy refresh workshops,
- generate candidate KPIs for weakly mapped objectives,
- improve communication between strategic and operational teams.

The tool should be positioned as an augmentation mechanism, not an autonomous evaluator. Human oversight remains central, especially where recommendations may influence resource allocation or service redesign.

### 13.6 Expert Validation Plan (Section to Complete Before Final Submission)

To satisfy evaluation rigor, include a lightweight expert validation exercise:

1. Recruit 2-3 reviewers (for example classmates, supervisor, or healthcare analyst).
2. Provide each reviewer with top 20 predicted strategy-action links.
3. Ask them to label each as `Aligned`, `Partially Aligned`, or `Not Aligned`.
4. Compute inter-rater agreement and summarize qualitative comments.
5. Compare reviewer labels with model predictions.

**Placeholder table to complete:**

| Reviewer | Background | Links Reviewed | Agreement with Model (%) | Key Comments |
|---|---|---:|---:|---|
| Reviewer A | _Add_ | _Add_ | _Add_ | _Add_ |
| Reviewer B | _Add_ | _Add_ | _Add_ | _Add_ |
| Reviewer C | _Optional_ | _Add_ | _Add_ | _Add_ |

This addition will materially strengthen the "Testing and Evaluation" marks.

---

## 14. Conclusion

This project designed and implemented an Intelligent Strategic Plan Synchronization System for Hamilton Health Sciences documents, combining semantic similarity, vector retrieval, AI-assisted recommendations, and explainable analytics in a single dashboard.

The prototype achieved a moderate synchronization score (51.0%) and demonstrated actionable diagnostics, including gap identification and objective-level analysis. Evaluation results confirm the importance of threshold calibration, with balanced performance at 0.35 (F1 0.50). The system meets the assignment goals for IR/NLP integration, intelligent features, dashboard usability, and testing rigor.

The work shows that strategic governance can be augmented by AI while maintaining transparency and human oversight.

---

## 15. Future Work

1. Fine-tune embeddings on healthcare strategic corpora.
2. Add ontology-backed concept normalization using RDF/OWL.
3. Introduce human annotation workflow in-app for active learning.
4. Add temporal trend analysis across multiple annual QIPs.
5. Support comparative analysis across hospitals/regions.
6. Add stronger explainability (salient sentence-level rationales).
7. Package as an API for CI/governance pipelines.

---

## 16. Reproducibility Appendix

### 16.1 Run Steps

```bash
python3 -m venv venv
source venv/bin/activate
pip install -r requirements.txt
streamlit run app.py
```

### 16.2 Files of Interest

- `app.py`
- `pages/1_Synchronisation_Analysis.py`
- `pages/2_Strategy_Deep_Dive.py`
- `pages/3_Improvement_Suggestions.py`
- `pages/4_Knowledge_Graph.py`
- `pages/5_Evaluation.py`
- `pages/6_Chat_With_Plans.py`
- `src/document_processor.py`
- `src/analyzer.py`
- `src/llm_helper.py`
- `src/knowledge_graph.py`
- `src/evaluation.py`

---

## 17. Suggested Figure List for Final PDF Export

When converting this markdown into final PDF submission, include screenshots for:

1. Home dashboard with overall score.
2. Synchronization heatmap and radar chart.
3. Strategy deep dive page.
4. AI improvement suggestions output.
5. Interactive knowledge graph.
6. Evaluation page (confusion matrix + threshold chart).
7. Chat-with-plans example Q&A.

### 17.1 Exact Placeholder Blocks You Can Keep in the Report

Copy these blocks where needed; they are pre-formatted for markdown to PDF export.

```md
![Figure X - Title](./figures/figureX-file-name.png)
*Figure X. One-line caption that explains what the reader should notice.*
```

Recommended figure insertion points:

- After Section 5.2: Input document evidence.
- After Section 6.4: Architecture diagram.
- After Section 7.4: Development phase screenshots.
- Inside Section 8: Results charts (heatmap, radar, per-strategy bars).
- Inside Section 9: Confusion matrix and threshold optimization chart.
- Inside Section 10: Knowledge graph and suggestion outputs.

### 17.2 Submission Checklist Placeholders (Fill before export)

- **Live Application URL:** `[INSERT LIVE URL]`
- **GitHub Repository URL:** `[INSERT REPOSITORY URL]`
- **Slides URL:** `[INSERT SLIDES LINK]`
- **Presentation Recording URL:** `[INSERT RECORDING LINK]`
- **Appendix (screenshots folder):** `[INSERT CLOUD/DRIVE LINK IF REQUIRED]`

---

## References (Harvard Style)

Google DeepMind (2024) *Gemini API documentation*. Available at: https://ai.google.dev/ (Accessed: 10 February 2026).

Hamilton Health Sciences (2025) *Quality Improvement Plan (QIP) Narrative 2025/26*. Available at: https://www.hamiltonhealthsciences.ca/ (Accessed: 10 February 2026).

Hamilton Health Sciences (2026) *Vision 2030 booklet*. Available at: https://www.hamiltonhealthsciences.ca/ (Accessed: 10 February 2026).

Lewis, P. et al. (2020) ‘Retrieval-Augmented Generation for Knowledge-Intensive NLP Tasks’, *Advances in Neural Information Processing Systems*, 33. Available at: https://arxiv.org/abs/2005.11401 (Accessed: 10 February 2026).

Mikolov, T. et al. (2013) ‘Efficient Estimation of Word Representations in Vector Space’, arXiv preprint arXiv:1301.3781.

Pedregosa, F. et al. (2011) ‘Scikit-learn: Machine Learning in Python’, *Journal of Machine Learning Research*, 12, pp. 2825-2830.

Plotly Technologies Inc. (2024) *Plotly Python open source graphing library*. Available at: https://plotly.com/python/ (Accessed: 10 February 2026).

Reimers, N. and Gurevych, I. (2019) ‘Sentence-BERT: Sentence Embeddings using Siamese BERT-Networks’, *Proceedings of EMNLP-IJCNLP*. Available at: https://arxiv.org/abs/1908.10084 (Accessed: 10 February 2026).

Streamlit (2024) *Streamlit documentation*. Available at: https://docs.streamlit.io/ (Accessed: 10 February 2026).

Van Rossum, G. and Drake, F.L. (2009) *Python 3 Reference Manual*. Scotts Valley, CA: CreateSpace.

---

## Declaration

This report documents the design and implementation work completed for the Information Retrieval coursework. AI tools were used to assist software development and drafting; final technical interpretation, validation, and submission responsibility remain with the student.
