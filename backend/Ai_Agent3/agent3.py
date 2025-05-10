from langchain_openai import ChatOpenAI, OpenAIEmbeddings
from langchain.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain_community.vectorstores import Chroma
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.document_loaders import PyPDFLoader
from langgraph.graph import StateGraph, END
import os
from dotenv import load_dotenv
import re
from typing import List, Dict, Any, Optional, TypedDict, Literal
import tempfile
from reportlab.lib.pagesizes import letter
from reportlab.platypus import SimpleDocTemplate, Paragraph, Spacer, Table, TableStyle
from reportlab.lib.styles import getSampleStyleSheet, ParagraphStyle
from reportlab.lib import colors
from reportlab.lib.units import inch

# Load environment variables
# load_dotenv()
OPENAI_API_KEY = "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"


# Define state structure
class StandardEnhancementState(TypedDict):
    standard_id: str
    standard_content: Optional[str]
    analysis: Optional[str]
    enhancements: Optional[str]
    validations: Optional[str]
    final_report: Optional[str]
    error: Optional[str]
    status: Literal["in_progress", "completed", "failed"]
    current_step: Literal[
        "retrieve", "analyze", "enhance", "validate", "report", "complete", "error"
    ]


# Function to convert text to PDF using ReportLab
def text_to_pdf(content, output_filename):
    """Convert text content to PDF file using ReportLab"""
    doc = SimpleDocTemplate(
        output_filename,
        pagesize=letter,
        rightMargin=72,
        leftMargin=72,
        topMargin=72,
        bottomMargin=72,
    )

    # Define styles
    styles = getSampleStyleSheet()
    title_style = styles["Heading1"]
    heading2_style = styles["Heading2"]
    heading3_style = styles["Heading3"]
    normal_style = styles["Normal"]

    # Create custom styles
    styles.add(
        ParagraphStyle(
            name="CodeBlock",
            parent=styles["Normal"],
            fontName="Courier",
            fontSize=9,
            leftIndent=36,
            rightIndent=36,
            spaceAfter=12,
            spaceBefore=12,
            backColor=colors.lightgrey,
        )
    )

    # Process content
    story = []

    # Split content into lines and process
    lines = content.split("\n")
    i = 0
    while i < len(lines):
        line = lines[i].strip()

        # Process headers
        if line.startswith("# "):
            story.append(Paragraph(line[2:], title_style))
        elif line.startswith("## "):
            story.append(Paragraph(line[3:], heading2_style))
        elif line.startswith("### "):
            story.append(Paragraph(line[4:], heading3_style))
        # Process code blocks
        elif line.startswith("```"):
            code_block = []
            i += 1
            while i < len(lines) and not lines[i].startswith("```"):
                code_block.append(lines[i])
                i += 1
            if code_block:
                code_text = "\n".join(code_block)
                story.append(Paragraph(code_text, styles["CodeBlock"]))
        # Process bullet points
        elif line.startswith("- "):
            story.append(Paragraph("• " + line[2:], normal_style))
        # Process numbered lists
        elif re.match(r"^\d+\.\s", line):
            story.append(Paragraph(line, normal_style))
        # Process normal text
        elif line:
            story.append(Paragraph(line, normal_style))
        # Add spacing between paragraphs
        if line:
            story.append(Spacer(1, 0.1 * inch))

        i += 1

    # Build the PDF
    doc.build(story)


# Function to load and process PDF files
def load_fas_standards_from_pdfs(pdf_paths):
    documents = []

    for pdf_path in pdf_paths:
        # Extract standard ID from filename
        match = re.search(r"FAS(\d+)", os.path.basename(pdf_path))
        if match:
            standard_num = match.group(1)
            standard_id = f"FAS {standard_num}"
        else:
            standard_id = os.path.basename(pdf_path).replace(".pdf", "")

        try:
            # Load PDF
            loader = PyPDFLoader(pdf_path)
            pdf_docs = loader.load()

            # Add standard ID to metadata
            for doc in pdf_docs:
                doc.metadata["standard"] = standard_id
                doc.metadata["page"] = doc.metadata.get("page_number", "unknown")
                documents.append(doc)
        except Exception as e:
            print(f"Error loading {pdf_path}: {e}")

    return documents


# Create embedding database using Chroma
def create_embedding_db(pdf_paths):
    # Initialize embeddings
    embeddings = OpenAIEmbeddings()

    # Load documents from PDFs
    documents = load_fas_standards_from_pdfs(pdf_paths)

    if not documents:
        raise Exception("No documents were loaded from PDFs")

    # Use optimized chunking strategy
    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=500, chunk_overlap=100, separators=["\n\n", "\n", ". ", " ", ""]
    )

    split_docs = text_splitter.split_documents(documents)

    # Create Chroma vector store
    vector_store = Chroma.from_documents(
        documents=split_docs,
        embedding=embeddings,
        collection_name="aaoifi_fas_standards",
    )

    return vector_store


# Create the nodes for the LangGraph
def retrieve_standard_content(
    state: StandardEnhancementState, vector_store
) -> StandardEnhancementState:
    """
    Retrieve content for a standard
    """
    standard_id = state["standard_id"]

    try:
        # Normalize standard_id format
        if not standard_id.startswith("FAS "):
            standard_id = f"FAS {standard_id.replace('FAS', '').strip()}"

        # Retrieve all chunks for this standard
        results = vector_store.similarity_search(
            query=f"{standard_id} full content",
            k=50,  # Get a large number of chunks
            filter={"standard": standard_id},
        )

        # Combine all chunks
        combined_text = "\n\n".join([doc.page_content for doc in results])

        # Create output directory if it doesn't exist
        os.makedirs("outputs", exist_ok=True)

        # Save content to file in outputs directory
        with open(f"outputs/{standard_id.replace(' ', '_')}_Content.txt", "w") as file:
            file.write(combined_text)

        return {
            **state,
            "standard_content": combined_text,
            "current_step": "analyze",
            "status": "in_progress",
        }
    except Exception as e:
        error_msg = f"Error retrieving content for {standard_id}: {str(e)}"
        print(f"Error: {error_msg}")
        return {
            **state,
            "error": error_msg,
            "current_step": "error",
            "status": "failed",
        }


def analyze_standard(state: StandardEnhancementState) -> StandardEnhancementState:
    """
    Analyze a standard and extract its structure and key elements
    """
    standard_id = state["standard_id"]
    standard_content = state["standard_content"]

    # Create prompt for analysis
    analysis_prompt = ChatPromptTemplate.from_template(
        """You are an expert Islamic finance scholar specializing in AAOIFI Financial Accounting Standards (FAS).
Your task is to analyze the following standard and extract its key elements, structure, and purpose.

Standard ID: {standard_id}
Standard Content:{standard_content}
Please provide a comprehensive analysis that includes:

1. OVERVIEW
   - Title and scope of the standard
   - Main purpose and objectives
   - Key stakeholders affected

2. STRUCTURE
   - Major sections and organization
   - Key definitions
   - Main requirements

3. KEY ELEMENTS
   - Core accounting principles established
   - Recognition requirements
   - Measurement guidelines
   - Disclosure requirements
   - Implementation guidance

4. CRITICAL ANALYSIS
   - Areas that could benefit from clarification
   - Potential challenges in implementation
   - Comparison with conventional accounting standards (if relevant)

Your analysis should be thorough, well-structured, and focused on the most important aspects of the standard.
"""
    )

    # Create analysis chain
    llm = ChatOpenAI(model="gpt-4", temperature=0)
    analysis_chain = analysis_prompt | llm | StrOutputParser()

    # Run analysis
    try:
        analysis = analysis_chain.invoke(
            {"standard_id": standard_id, "standard_content": standard_content}
        )

        # Create output directory if it doesn't exist
        os.makedirs("outputs", exist_ok=True)

        # Save analysis to text file
        txt_filename = f"outputs/{standard_id.replace(' ', '_')}_Analysis.txt"
        with open(txt_filename, "w") as file:
            file.write(analysis)

        # Convert text to PDF
        pdf_filename = f"outputs/{standard_id.replace(' ', '_')}_Analysis.pdf"
        text_to_pdf(analysis, pdf_filename)

        return {
            **state,
            "analysis": analysis,
            "current_step": "enhance",
            "status": "in_progress",
        }
    except Exception as e:
        error_msg = f"Error analyzing {standard_id}: {str(e)}"
        print(f"Error: {error_msg}")
        return {
            **state,
            "error": error_msg,
            "current_step": "error",
            "status": "failed",
        }


def generate_enhancements(state: StandardEnhancementState) -> StandardEnhancementState:
    """
    Generate enhancement suggestions based on standard analysis
    """
    standard_id = state["standard_id"]
    analysis = state["analysis"]

    # Create prompt for enhancements
    enhancement_prompt = ChatPromptTemplate.from_template(
        """You are an expert Islamic finance consultant with deep knowledge of AAOIFI Financial Accounting Standards.
Your task is to propose thoughtful, valuable enhancements to improve the clarity, applicability, and effectiveness of the standard.

Standard ID: {standard_id}

Standard Analysis:{analysis}
Please propose 5-7 specific enhancements that would meaningfully improve this standard. For each enhancement:

1. ELEMENT TO ENHANCE
   - Identify the specific section, definition, requirement, or guidance to be enhanced
   - Quote or clearly reference the original text

2. PROPOSED ENHANCEMENT
   - Provide specific revised wording or new content
   - Be precise and concrete in your suggestions

3. RATIONALE
   - Explain why this enhancement is valuable
   - Discuss how it addresses ambiguity, implementation challenges, or emerging issues
   - Reference relevant Islamic finance principles or practices

4. EXPECTED IMPACT
   - Describe the practical benefits for financial institutions
   - Indicate the significance (High/Medium/Low)

IMPORTANT GUIDELINES:
- Focus on substantive improvements, not minor editorial changes
- Ensure all suggestions maintain Shariah compliance
- Consider practical implementation challenges
- Ensure compatibility with other AAOIFI standards
- Address emerging trends in Islamic finance
- Propose enhancements that add genuine value

Your suggestions should be presented in a clear, structured format with each enhancement clearly numbered and separated.
"""
    )

    # Create enhancement chain
    llm = ChatOpenAI(model="gpt-4", temperature=0.2)
    enhancement_chain = enhancement_prompt | llm | StrOutputParser()

    # Run enhancement generation
    try:
        enhancements = enhancement_chain.invoke(
            {"standard_id": standard_id, "analysis": analysis}
        )

        # Create output directory if it doesn't exist
        os.makedirs("outputs", exist_ok=True)

        # Save enhancements to text file
        txt_filename = f"outputs/{standard_id.replace(' ', '_')}_Enhancements.txt"
        with open(txt_filename, "w") as file:
            file.write(enhancements)

        # Convert text to PDF
        pdf_filename = f"outputs/{standard_id.replace(' ', '_')}_Enhancements.pdf"
        text_to_pdf(enhancements, pdf_filename)

        return {
            **state,
            "enhancements": enhancements,
            "current_step": "validate",
            "status": "in_progress",
        }
    except Exception as e:
        error_msg = f"Error generating enhancements for {standard_id}: {str(e)}"
        print(f"Error: {error_msg}")
        return {
            **state,
            "error": error_msg,
            "current_step": "error",
            "status": "failed",
        }


def validate_enhancements(state: StandardEnhancementState) -> StandardEnhancementState:
    """
    Validate enhancement suggestions
    """
    standard_id = state["standard_id"]
    analysis = state["analysis"]
    enhancements = state["enhancements"]

    # Create prompt for validation
    validation_prompt = ChatPromptTemplate.from_template(
        """You are a Shariah scholar and AAOIFI standards expert responsible for validating proposed enhancements to AAOIFI Financial Accounting Standards.
Your task is to evaluate each enhancement suggestion and determine if it meets the necessary criteria for approval.

Standard ID: {standard_id}

Standard Analysis:{analysis}
Proposed Enhancements:{enhancements}
Please conduct a thorough validation of each enhancement suggestion and provide the following for each:

1. ENHANCEMENT SUMMARY
   - Briefly summarize the enhancement

2. VALIDATION ASSESSMENT
   - Shariah Compliance: Assess whether the enhancement complies with Shariah principles (Yes/No)
   - Practical Applicability: Evaluate if the enhancement is practically implementable (Yes/No)
   - Consistency: Determine if it maintains consistency with other AAOIFI standards (Yes/No)
   - Value Addition: Assess if it provides meaningful improvement (Yes/No)

3. APPROVAL DECISION
   - Approved: Yes/No
   - If not approved, provide a revised version that would be acceptable

4. RATIONALE
   - Provide detailed reasoning for your decision
   - Reference specific Shariah principles, accounting concepts, or implementation considerations

CRITICAL GUIDELINES:
- Shariah compliance is non-negotiable - reject any enhancement that compromises Islamic principles
- Be mindful of practical implementation challenges
- Consider the global context of Islamic finance
- Ensure enhancements maintain the original intent of the standard
- Apply the highest standards of scrutiny

Organize your validation by enhancement number and provide a clear approval decision for each.
"""
    )

    # Create validation chain
    llm = ChatOpenAI(model="gpt-4", temperature=0)
    validation_chain = validation_prompt | llm | StrOutputParser()

    # Run validation
    try:
        validations = validation_chain.invoke(
            {
                "standard_id": standard_id,
                "analysis": analysis,
                "enhancements": enhancements,
            }
        )

        # Create output directory if it doesn't exist
        os.makedirs("outputs", exist_ok=True)

        # Save validations to text file
        txt_filename = f"outputs/{standard_id.replace(' ', '_')}_Validations.txt"
        with open(txt_filename, "w") as file:
            file.write(validations)

        # Convert text to PDF
        pdf_filename = f"outputs/{standard_id.replace(' ', '_')}_Validations.pdf"
        text_to_pdf(validations, pdf_filename)

        return {
            **state,
            "validations": validations,
            "current_step": "report",
            "status": "in_progress",
        }
    except Exception as e:
        error_msg = f"Error validating enhancements for {standard_id}: {str(e)}"
        print(f"Error: {error_msg}")
        return {
            **state,
            "error": error_msg,
            "current_step": "error",
            "status": "failed",
        }


def generate_final_report(state: StandardEnhancementState) -> StandardEnhancementState:
    """
    Generate a final report summarizing the enhancement process
    """
    standard_id = state["standard_id"]
    analysis = state["analysis"]
    enhancements = state["enhancements"]
    validations = state["validations"]

    # Create prompt for report
    report_prompt = ChatPromptTemplate.from_template(
        """You are an expert in Islamic finance and AAOIFI standards.
Your task is to create a comprehensive, professional report summarizing the enhancement process for {standard_id}.

Standard Analysis:{analysis}
Proposed Enhancements:{enhancements}
Validation Results:{validations}
Please create a well-structured report with the following sections:

# AAOIFI Standard Enhancement Report: {standard_id}

## Executive Summary
[Concise summary of the standard, enhancement process, and key outcomes]

## 1. Standard Overview
[Summary of the standard's purpose, scope, and key requirements]

## 2. Enhancement Methodology
[Description of the multi-agent approach used in the enhancement process]

## 3. Key Findings from Analysis
[Summary of the critical analysis findings]

## 4. Enhancement Recommendations
[Detailed presentation of approved enhancements, including:
- Original text
- Enhanced text
- Rationale
- Expected impact]

## 5. Implementation Considerations
[Discussion of practical considerations for implementing these enhancements]

## 6. Conclusion
[Summary of the value added by these enhancements]

Make the report professional, well-organized, and focused on actionable insights.
"""
    )

    # Create report chain
    llm = ChatOpenAI(model="gpt-4", temperature=0)
    report_chain = report_prompt | llm | StrOutputParser()

    # Generate report
    try:
        report = report_chain.invoke(
            {
                "standard_id": standard_id,
                "analysis": analysis,
                "enhancements": enhancements,
                "validations": validations,
            }
        )

        # Create output directory if it doesn't exist
        os.makedirs("outputs", exist_ok=True)

        # Save report to text file
        txt_filename = f"outputs/{standard_id.replace(' ', '_')}_Final_Report.txt"
        with open(txt_filename, "w") as file:
            file.write(report)

        # Convert text to PDF
        pdf_filename = f"outputs/{standard_id.replace(' ', '_')}_Final_Report.pdf"
        text_to_pdf(report, pdf_filename)

        return {
            **state,
            "final_report": report,
            "current_step": "complete",
            "status": "completed",
        }
    except Exception as e:
        error_msg = f"Error generating final report for {standard_id}: {str(e)}"
        print(f"Error: {error_msg}")
        return {
            **state,
            "error": error_msg,
            "current_step": "error",
            "status": "failed",
        }


def handle_error(state: StandardEnhancementState) -> StandardEnhancementState:
    """
    Handle errors in the process
    """
    standard_id = state["standard_id"]
    error = state["error"]

    # Try to generate an error report with what we have
    try:
        error_report = f"""# Error Report for {standard_id}

## Error Details
{error}

## Process Status
- Standard Content: {"Retrieved" if state.get("standard_content") else "Not retrieved"}
- Analysis: {"Completed" if state.get("analysis") else "Not completed"}
- Enhancements: {"Generated" if state.get("enhancements") else "Not generated"}
- Validation: {"Completed" if state.get("validations") else "Not completed"}
- Final Report: {"Generated" if state.get("final_report") else "Not generated"}

## Partial Results

"""
        # Add any partial results we have
        if state.get("analysis"):
            error_report += "\n### Analysis Preview\n"
            error_report += state["analysis"][:500] + "...\n"

        if state.get("enhancements"):
            error_report += "\n### Enhancement Suggestions Preview\n"
            error_report += state["enhancements"][:500] + "...\n"

        if state.get("validations"):
            error_report += "\n### Validation Results Preview\n"
            error_report += state["validations"][:500] + "...\n"

        # Create output directory if it doesn't exist
        os.makedirs("outputs", exist_ok=True)

        # Save error report to text file
        txt_filename = f"outputs/{standard_id.replace(' ', '_')}_Error_Report.txt"
        with open(txt_filename, "w") as file:
            file.write(error_report)

        # Convert text to PDF
        pdf_filename = f"outputs/{standard_id.replace(' ', '_')}_Error_Report.pdf"
        text_to_pdf(error_report, pdf_filename)

        return {**state, "final_report": error_report, "status": "failed"}
    except Exception as e:
        print(f"Error generating error report: {str(e)}")
        return {**state, "status": "failed"}


# Define the LangGraph workflow
def build_enhancement_workflow(vector_store):
    """
    Build the LangGraph workflow for standard enhancement
    """
    # Define the workflow
    workflow = StateGraph(StandardEnhancementState)

    # Add nodes
    workflow.add_node(
        "retrieve", lambda state: retrieve_standard_content(state, vector_store)
    )
    workflow.add_node("analyze", analyze_standard)
    workflow.add_node("enhance", generate_enhancements)
    workflow.add_node("validate", validate_enhancements)
    workflow.add_node("report", generate_final_report)
    workflow.add_node("handle_error", handle_error)

    # Add conditional edges
    workflow.add_conditional_edges(
        "retrieve",
        lambda state: (
            "analyze" if state["current_step"] == "analyze" else "handle_error"
        ),
    )

    workflow.add_conditional_edges(
        "analyze",
        lambda state: (
            "enhance" if state["current_step"] == "enhance" else "handle_error"
        ),
    )

    workflow.add_conditional_edges(
        "enhance",
        lambda state: (
            "validate" if state["current_step"] == "validate" else "handle_error"
        ),
    )

    workflow.add_conditional_edges(
        "validate",
        lambda state: "report" if state["current_step"] == "report" else "handle_error",
    )

    # Add terminal edges
    workflow.add_edge("report", END)
    workflow.add_edge("handle_error", END)

    # Set entry point
    workflow.set_entry_point("retrieve")

    # Compile the workflow
    return workflow.compile()


# Main function
def enhance_standard_with_graph(standard_id, vector_store):
    """
    Enhance a standard using the LangGraph workflow

    Args:
        standard_id: ID of the standard to enhance
        vector_store: The vector store

    Returns:
        The final state of the workflow
    """
    # Initialize the workflow
    workflow = build_enhancement_workflow(vector_store)

    # Set initial state
    initial_state = {
        "standard_id": standard_id,
        "standard_content": None,
        "analysis": None,
        "enhancements": None,
        "validations": None,
        "final_report": None,
        "error": None,
        "status": "in_progress",
        "current_step": "retrieve",
    }

    print(f"Processing {standard_id}...")

    # Execute the workflow
    final_output = None
    for output in workflow.stream(initial_state):
        final_output = output

    # Get the final state
    final_state = final_output.get("state", {})

    return final_state


# Main function
def main():
    # Create outputs directory
    os.makedirs("outputs", exist_ok=True)

    pdf_paths = ["FAS4.pdf", "FAS10.pdf", "FAS32.pdf"]

    # Standards to enhance
    standards_to_enhance = ["FAS 4", "FAS 10", "FAS 32"]

    print("AAOIFI Standards Enhancement System")
    print("----------------------------------")

    print("Initializing system...")
    vector_store = create_embedding_db(pdf_paths)

    # Process each standard
    results = {}
    for standard_id in standards_to_enhance:
        try:
            results[standard_id] = enhance_standard_with_graph(
                standard_id, vector_store
            )
        except Exception as e:
            print(f"Error processing {standard_id}")
            results[standard_id] = {"status": "failed", "error": str(e)}

    # Generate summary report
    summary_report = """# AAOIFI Standards Enhancement Summary Report

## Process Overview
This report summarizes the enhancement process for AAOIFI Financial Accounting Standards.

## Standards Processed
"""

    for standard_id, result in results.items():
        status = result.get("status", "Unknown")
        status_display = "Completed" if status == "completed" else "Failed"
        summary_report += f"\n### {standard_id}\n- Status: {status_display}\n"

        if status == "completed":
            summary_report += "- All processing steps completed successfully\n"
            summary_report += f"- Final report available at: outputs/{standard_id.replace(' ', '_')}_Final_Report.pdf\n"
        else:
            summary_report += f"- Error: {result.get('error', 'Unknown error')}\n"
            summary_report += f"- Error report available at: outputs/{standard_id.replace(' ', '_')}_Error_Report.pdf\n"

    summary_report += "\n## Output Files\n"
    summary_report += "All output files are available in the 'outputs' directory."

    # Save summary report
    with open("outputs/Enhancement_Summary_Report.txt", "w") as file:
        file.write(summary_report)

    # Convert summary to PDF
    text_to_pdf(summary_report, "outputs/Enhancement_Summary_Report.pdf")

    print("Enhancement process completed for all standards.")
    print("All reports generated in PDF format in the 'outputs' directory.")


if __name__ == "__main__":
    main()
