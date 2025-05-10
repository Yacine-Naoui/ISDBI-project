from langchain_openai import ChatOpenAI, OpenAIEmbeddings
from langchain.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain_community.vectorstores import Chroma
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_core.documents import Document
from langchain_community.document_loaders import PyPDFLoader
import os
from dotenv import load_dotenv
import re
import json
from typing import List, Dict, Any

# Load environment variables
load_dotenv()


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

        print(f"Loading {pdf_path} as {standard_id}...")

        try:
            # Load PDF
            loader = PyPDFLoader(pdf_path)
            pdf_docs = loader.load()

            print(f"Successfully loaded {len(pdf_docs)} pages from {pdf_path}")

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

    print(f"Loaded {len(documents)} total pages from all PDFs")

    # Use optimized chunking strategy
    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=500, chunk_overlap=100, separators=["\n\n", "\n", ". ", " ", ""]
    )

    split_docs = text_splitter.split_documents(documents)
    print(f"Split into {len(split_docs)} chunks for embedding")

    # Create Chroma vector store
    vector_store = Chroma.from_documents(
        documents=split_docs,
        embedding=embeddings,
        collection_name="aaoifi_fas_standards",
    )

    return vector_store


# Enhanced semantic search function
def semantic_search(vector_store, standard_id, query, k=3):
    """Perform enhanced semantic search with filtering"""
    # Normalize standard_id format
    if not standard_id.startswith("FAS "):
        standard_id = f"FAS {standard_id.replace('FAS', '').strip()}"

    try:
        # First try with standard filter
        results = vector_store.similarity_search(
            query=query, k=k, filter={"standard": standard_id}
        )

        # If no results, try without filter
        if not results:
            results = vector_store.similarity_search(
                query=f"{standard_id} {query}", k=k
            )

        return results
    except Exception as e:
        print(f"Search error for {standard_id} - {query}: {e}")
        return []


# Helper function to query standards
def query_standards(vector_store, transaction_type):
    """
    Query relevant AAOIFI standards for a transaction type

    Args:
        vector_store: The vector store
        transaction_type: Type of transaction

    Returns:
        Dictionary of relevant standards with descriptions
    """
    print(f"Querying standards for: {transaction_type}")

    # Get relevant results for all our target standards
    standards = ["FAS 4", "FAS 7", "FAS 10", "FAS 28", "FAS 32"]
    results = {}

    # Search specialized query terms for the transaction type
    query_terms = [
        transaction_type,
        f"{transaction_type} accounting",
        f"{transaction_type} journal entries",
    ]

    # Add additional terms for partner exit/buyout
    if (
        "buyout" in transaction_type.lower()
        or "exit" in transaction_type.lower()
        or "equity" in transaction_type.lower()
    ):
        query_terms.extend(
            [
                "partner exit buyout",
                "equity stake acquisition",
                "partnership termination",
            ]
        )

    # For each standard, search with these terms
    for std in standards:
        std_results = []

        for query in query_terms:
            docs = semantic_search(vector_store, std, query, k=1)
            std_results.extend(docs)

        if std_results:
            # Combine content from all results for this standard
            content = "\n\n".join([doc.page_content for doc in std_results[:2]])
            results[std] = content

    return results


# Helper function to analyze journal entry
def analyze_journal_entry(journal_entry):
    """
    Analyze the journal entry to identify key accounts and transaction type

    Args:
        journal_entry: The journal entries for the transaction

    Returns:
        Analysis of the journal entry
    """
    # Extract debits and credits
    debits = []
    credits = []

    for line in journal_entry.split("\n"):
        line = line.strip()
        if line.startswith("Dr.") or line.startswith("DR.") or line.startswith("dr."):
            parts = line.split()
            if len(parts) >= 2:
                account = parts[1:-1] if len(parts) > 2 else [parts[1]]
                debits.append(" ".join(account))
        elif line.startswith("Cr.") or line.startswith("CR.") or line.startswith("cr."):
            parts = line.split()
            if len(parts) >= 2:
                account = parts[1:-1] if len(parts) > 2 else [parts[1]]
                credits.append(" ".join(account))

    # Determine transaction indicators
    indicators = {
        "involves_equity": any("equity" in acct.lower() for acct in debits + credits),
        "involves_cash": any("cash" in acct.lower() for acct in debits + credits),
        "involves_wip": any(
            ("work" in acct.lower() and "progress" in acct.lower())
            for acct in debits + credits
        ),
        "involves_payable": any("payable" in acct.lower() for acct in debits + credits),
        "involves_liability": any(
            "liability" in acct.lower() for acct in debits + credits
        ),
    }

    # Identify transaction type indicators
    tx_indicators = []

    if indicators["involves_equity"] and any(
        "cash" in acct.lower() for acct in credits
    ):
        tx_indicators.append("partner_exit")

    if indicators["involves_equity"] and indicators["involves_cash"]:
        tx_indicators.append("equity_transfer")

    if indicators["involves_wip"] and indicators["involves_payable"]:
        tx_indicators.append("contract_reversal")

    return {
        "debits": debits,
        "credits": credits,
        "indicators": indicators,
        "transaction_indicators": tx_indicators,
    }


# Helper function to determine standard weights
def determine_standard_weights(transaction_type, journal_analysis, standards_content):
    """
    Determine standard weights based on transaction and journal entries

    Args:
        transaction_type: Type of transaction
        journal_analysis: Analysis of journal entries
        standards_content: Content of applicable standards

    Returns:
        Weighted probabilities for standards
    """
    # Initialize weights
    weights = {}
    reasoning = {}

    # Get key indicators
    indicators = journal_analysis.get("indicators", {})
    tx_indicators = journal_analysis.get("transaction_indicators", [])

    # Base weights on indicators
    if "partner_exit" in tx_indicators or "equity_transfer" in tx_indicators:
        weights["FAS 4"] = 70
        reasoning["FAS 4"] = (
            "Primary standard for partnership exits and equity transfers in Musharaka arrangements"
        )

        weights["FAS 7"] = 20
        reasoning["FAS 7"] = (
            "Secondary standard for investments accounting but less specific to partnership exits"
        )

        weights["FAS 32"] = 5
        reasoning["FAS 32"] = (
            "Minor relevance if the arrangement has lease-like characteristics"
        )

        weights["FAS 28"] = 5
        reasoning["FAS 28"] = (
            "Minor relevance for deferred payment aspects if applicable"
        )

        weights["FAS 10"] = 0
        reasoning["FAS 10"] = "Not applicable to partnership/equity transactions"

    elif "contract_reversal" in tx_indicators:
        weights["FAS 10"] = 75
        reasoning["FAS 10"] = (
            "Primary standard for construction contracts and contract modifications"
        )

        weights["FAS 28"] = 15
        reasoning["FAS 28"] = (
            "Secondary relevance for deferred payment aspects of contracts"
        )

        weights["FAS 4"] = 5
        reasoning["FAS 4"] = (
            "Minor relevance if there's a partnership aspect to the contract"
        )

        weights["FAS 7"] = 5
        reasoning["FAS 7"] = "Minor relevance for investment accounting aspects"

        weights["FAS 32"] = 0
        reasoning["FAS 32"] = "Not applicable to contract reversals"

    else:
        # Generic weights based on transaction type
        if "buyout" in transaction_type.lower() or "equity" in transaction_type.lower():
            weights["FAS 4"] = 60
            reasoning["FAS 4"] = "Most relevant for equity transactions"

            weights["FAS 7"] = 25
            reasoning["FAS 7"] = "Relevant for investment accounting aspects"

            weights["FAS 28"] = 10
            reasoning["FAS 28"] = "Minor relevance for payment terms"

            weights["FAS 32"] = 5
            reasoning["FAS 32"] = "Low relevance unless leasing involved"

            weights["FAS 10"] = 0
            reasoning["FAS 10"] = "Not applicable to equity transactions"

        elif "contract" in transaction_type.lower():
            weights["FAS 10"] = 70
            reasoning["FAS 10"] = "Most relevant for contract transactions"

            weights["FAS 28"] = 20
            reasoning["FAS 28"] = "Relevant for payment aspects"

            weights["FAS 4"] = 5
            reasoning["FAS 4"] = "Low relevance unless partnership involved"

            weights["FAS 7"] = 5
            reasoning["FAS 7"] = "Low relevance for investment aspects"

            weights["FAS 32"] = 0
            reasoning["FAS 32"] = "Not applicable to contracts"
        else:
            # Fallback - equal weights
            for std in standards_content:
                weights[std] = 20
                reasoning[std] = (
                    "Equal weight assigned due to insufficient transaction indicators"
                )

    result = {}
    for std in standards_content:
        if std in weights:
            result[std] = {
                "weight": weights[std],
                "reasoning": reasoning.get(std, "No specific reasoning provided"),
            }

    return result


# Main analysis function
def analyze_transaction(vector_store, transaction_details):
    """
    Analyze a transaction and identify applicable AAOIFI standards

    Args:
        vector_store: The vector store
        transaction_details: The transaction details

    Returns:
        Analysis results
    """
    # Extract transaction type and journal entries
    lines = transaction_details.strip().split("\n")
    context_lines = []
    journal_entry_lines = []
    transaction_type = ""

    in_journal = False
    for line in lines:
        if "Journal Entry" in line:
            in_journal = True
            continue

        if in_journal:
            journal_entry_lines.append(line)
        else:
            context_lines.append(line)

    context = "\n".join(context_lines)
    journal_entry = "\n".join(journal_entry_lines)

    # Extract transaction type
    if "buyout" in context.lower() or "exits" in context.lower():
        transaction_type = "equity buyout"
    elif "change order" in context.lower() or "revert" in context.lower():
        transaction_type = "contract reversal"
    else:
        transaction_type = "general transaction"

    print(f"Analyzing transaction type: {transaction_type}")
    print(f"Journal entry: {journal_entry}")

    # Step 1: Analyze journal entry
    journal_analysis = analyze_journal_entry(journal_entry)
    print("Journal analysis completed")

    # Step 2: Query standards
    standards_content = query_standards(vector_store, transaction_type)
    print("Standards query completed")

    # Step 3: Determine standard weights
    weighted_standards = determine_standard_weights(
        transaction_type, journal_analysis, standards_content
    )
    print("Weights determined")

    # Step 4: Generate the final analysis
    llm = ChatOpenAI(model="gpt-4", temperature=0)

    prompt_template = """Analyze this Islamic finance transaction and determine which AAOIFI FAS standards apply.

Transaction details:
{transaction}

Journal entry analysis:
{journal_analysis}

Applicable standards with preliminary weights:
{weighted_standards}

Format your response as follows:
1. Transaction Type: [Brief description]
2. Journal Entry Analysis: [Analysis of the specific journal entries]
3. Applicable Standards (with weighted probabilities):
   - FAS X (xx%): [Brief reasoning]
   - FAS Y (yy%): [Brief reasoning]
4. Detailed Explanation: [In-depth analysis of why these standards apply]
"""

    prompt = ChatPromptTemplate.from_template(prompt_template)

    chain = prompt | llm | StrOutputParser()

    result = chain.invoke(
        {
            "transaction": transaction_details,
            "journal_analysis": json.dumps(journal_analysis, indent=2),
            "weighted_standards": json.dumps(weighted_standards, indent=2),
        }
    )

    return result


# Main function
def main():
    pdf_paths = ["FAS4.pdf", "FAS7.pdf", "FAS10.pdf", "FAS28.pdf", "FAS32.pdf"]

    print("AAOIFI FAS Classification Agent")
    print("------------------------------")

    print("\nInitializing vector store from PDFs...")
    vector_store = create_embedding_db(pdf_paths)

    # Define the transaction to analyze
    transaction = """
 The client cancels the change order, reverting to the original contract terms. 
Adjustments:  
Revised Contract Value: Back to $5,000,000 
Timeline Restored: 2 years 
Accounting Treatment:  
Adjustment of revenue and cost projections 
Reversal of additional cost accruals 
Journal Entry for Cost Reversal: 
Dr. Accounts Payable   $1,000,000   
Cr. Work-in-Progress  $1,000,000   
This restores the original contract cost. 
    """

    print("\nAnalyzing Partner Buyout Transaction...\n")
    try:
        result = analyze_transaction(vector_store, transaction)

        print("\nResults:")
        print(result)
    except Exception as e:
        print(f"Error analyzing transaction: {str(e)}")
        import traceback

        traceback.print_exc()


if __name__ == "__main__":
    main()

# main
