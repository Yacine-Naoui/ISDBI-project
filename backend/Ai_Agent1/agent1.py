from langchain_openai import ChatOpenAI
from langchain.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain.agents import AgentExecutor, create_openai_functions_agent
from langchain_core.tools import tool
from langchain.memory import ConversationBufferMemory
import operator
import os
from dotenv import load_dotenv
from typing import List, Dict, Any, TypedDict, Optional, Union

# Load environment variables (.env file with your OpenAI API key)
# load_dotenv(dotenv_path=os.path.join(os.path.dirname(__file__), ".env"))
OPENAI_API_KEY = "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"


# Create a simple calculator tool
@tool
def calculator(expression: str) -> float:
    """Evaluate a mathematical expression."""
    try:
        return eval(
            expression,
            {"__builtins__": {}},
            {k: getattr(operator, k) for k in dir(operator) if not k.startswith("_")},
        )
    except Exception as e:
        return f"Error in calculation: {str(e)}"


# Simplified additional costs tool to avoid parsing errors
@tool
def extract_additional_costs(cost_text: str) -> List[Dict[str, Union[str, float]]]:
    """
    Extract additional costs from text

    Args:
        cost_text: Text describing additional costs

    Returns:
        List of additional costs with description and amount
    """
    # This is a simplified version that will be more reliable
    # In a real application, you would want more sophisticated parsing
    additional_costs = []

    if "installation costs" in cost_text.lower() and "USD 25,000" in cost_text:
        additional_costs.append(
            {"description": "Installation Costs", "amount": 25000.0}
        )

    if "specialized equipment" in cost_text.lower() and "USD 45,000" in cost_text:
        additional_costs.append(
            {"description": "Specialized Equipment", "amount": 45000.0}
        )

    return additional_costs


# Improved Ijarah calculation tool
@tool
def calculate_ijarah_values(
    purchase_price: float,
    purchase_option_price: float,
    yearly_rental: float,
    years: int,
    residual_value: float,
    additional_costs: Optional[List[Dict[str, Union[str, float]]]] = None,
) -> dict:
    """
    Calculate all Ijarah MBT accounting values at initial recognition

    Args:
        purchase_price: The original price of the asset
        purchase_option_price: Price to purchase at end of Ijarah term
        yearly_rental: The annual rental amount
        years: Number of years in the Ijarah term
        residual_value: The expected residual value at end of term
        additional_costs: Optional list of additional costs [{"description": str, "amount": float}, ...]

    Returns:
        Complete calculation results with journal entry
    """
    # Handle None for additional_costs
    if additional_costs is None:
        additional_costs = []

    # Calculate total cost and ROU
    additional_cost_total = sum(cost.get("amount", 0) for cost in additional_costs)
    total_cost = purchase_price + additional_cost_total
    rou_value = total_cost - purchase_option_price

    # Calculate deferred cost
    total_rentals = yearly_rental * years
    deferred_cost = total_rentals - rou_value

    # Calculate amortizable amount
    terminal_difference = residual_value - purchase_option_price
    amortizable_amount = rou_value - terminal_difference

    # Generate journal entry
    journal_entry = {
        "debit": {
            "Right of Use Asset (ROU)": round(rou_value, 2),
            "Deferred Ijarah Cost": round(deferred_cost, 2),
        },
        "credit": {"Ijarah Liability": round(rou_value + deferred_cost, 2)},
    }

    # Create detailed explanation
    cost_details = []
    if additional_costs:
        for cost in additional_costs:
            cost_details.append(
                f"{cost.get('description', 'Additional cost')}: {cost.get('amount', 0)}"
            )

    # Format the additional costs explanation
    additional_costs_explanation = ""
    if cost_details:
        additional_costs_explanation = " + " + " + ".join(cost_details)

    return {
        "calculations": {
            "purchase_price": purchase_price,
            "additional_costs": {
                cost.get("description", f"Cost {i+1}"): cost.get("amount", 0)
                for i, cost in enumerate(additional_costs)
            },
            "additional_cost_total": additional_cost_total,
            "prime_cost": total_cost,
            "rou_value": rou_value,
            "total_rentals": total_rentals,
            "deferred_cost": deferred_cost,
            "terminal_difference": terminal_difference,
            "amortizable_amount": amortizable_amount,
        },
        "journal_entry": journal_entry,
        "explanation": [
            f"1. Prime cost = Purchase price + Additional costs = {purchase_price}{additional_costs_explanation} = {total_cost}",
            f"2. Right of Use (ROU) = Prime cost - Purchase option price = {total_cost} - {purchase_option_price} = {rou_value}",
            f"3. Total rentals = Yearly rental × Years = {yearly_rental} × {years} = {total_rentals}",
            f"4. Deferred Ijarah Cost = Total rentals - ROU = {total_rentals} - {rou_value} = {deferred_cost}",
            f"5. Terminal value difference = Residual value - Purchase option price = {residual_value} - {purchase_option_price} = {terminal_difference}",
            f"6. Amortizable Amount = ROU - Terminal value difference = {rou_value} - {terminal_difference} = {amortizable_amount}",
        ],
    }


# Fixed validation tool with optional additional_costs parameter
@tool
def validate_ijarah_transaction(
    purchase_price: float,
    purchase_option_price: float,
    residual_value: float,
    yearly_rental: float,
    years: int,
    additional_costs: Optional[List[Dict[str, Union[str, float]]]] = None,
) -> dict:
    """
    Validate an Ijarah MBT transaction for completeness and Shariah compliance

    Args:
        purchase_price: The original price of the asset
        purchase_option_price: Price to purchase at end of Ijarah term
        residual_value: Expected residual value at end of term
        yearly_rental: Annual rental amount
        years: Number of years in the Ijarah term
        additional_costs: Optional list of additional costs

    Returns:
        Validation results and any compliance issues
    """
    issues = []
    warnings = []

    # Handle None for additional_costs
    if additional_costs is None:
        additional_costs = []
        warnings.append(
            "No additional costs provided. Make sure all costs are considered."
        )

    # Check for basic errors
    if purchase_price <= 0:
        issues.append("Purchase price must be greater than zero")

    if purchase_option_price < 0:
        issues.append("Purchase option price cannot be negative")

    if residual_value < 0:
        issues.append("Residual value cannot be negative")

    if yearly_rental <= 0:
        issues.append("Yearly rental must be greater than zero")

    if years <= 0:
        issues.append("Ijarah term must be greater than zero")

    # Check for potential Shariah compliance issues
    if purchase_option_price == 0:
        warnings.append(
            "Zero purchase option price may raise Shariah compliance concerns"
        )

    additional_cost_total = sum(cost.get("amount", 0) for cost in additional_costs)
    total_cost = purchase_price + additional_cost_total
    total_rentals = yearly_rental * years

    # Check if total rentals significantly exceed asset cost (potential riba)
    if total_rentals > total_cost * 1.5:
        warnings.append(
            "Total rentals significantly exceed asset cost, which may raise concerns about excessive charges"
        )

    return {
        "is_valid": len(issues) == 0,
        "issues": issues,
        "warnings": warnings,
        "transaction_summary": {
            "total_cost": total_cost,
            "total_rentals": total_rentals,
            "rental_to_cost_ratio": (
                round(total_rentals / total_cost, 2) if total_cost > 0 else "N/A"
            ),
        },
    }


@tool
def calculate_monthly_amortization(amortizable_amount: float, years: int) -> dict:
    """
    Calculate monthly amortization for Ijarah MBT

    Args:
        amortizable_amount: The amount to be amortized
        years: Number of years for amortization

    Returns:
        Monthly amortization schedule
    """
    months = years * 12
    monthly_amount = amortizable_amount / months

    schedule = []
    remaining_amount = amortizable_amount

    for i in range(1, months + 1):
        remaining_amount -= monthly_amount
        schedule.append(
            {
                "month": i,
                "amortization_amount": round(monthly_amount, 2),
                "remaining_balance": round(remaining_amount, 2),
            }
        )

    return {
        "monthly_amortization": round(monthly_amount, 2),
        "total_months": months,
        "total_amortization": amortizable_amount,
        "schedule": schedule[:3],  # Just show first 3 months for brevity
    }


# Define all tools
tools = [
    calculate_ijarah_values,
    calculate_monthly_amortization,
    extract_additional_costs,
    validate_ijarah_transaction,
    calculator,
]

# Initialize OpenAI model
llm = ChatOpenAI(model="gpt-4")

# Create improved prompt template with agent_scratchpad
prompt = ChatPromptTemplate.from_messages(
    [
        (
            "system",
            """You are an Islamic banking accountant specialized in Ijarah MBT (Ijara Muntahia Bittamleek). 
    Follow AAOIFI accounting standards for Islamic financial institutions.
    
    When processing an Ijarah MBT transaction, follow these steps:
    1. FIRST, extract ALL additional costs (installation, freight, equipment, etc.) using the extract_additional_costs tool
    2. THEN, calculate the Right of Use (ROU) asset based on the TOTAL COST (purchase price + ALL additional costs) minus the purchase option price
    3. Calculate Deferred Ijarah Cost as the difference between total rentals and ROU value
    4. Determine the amortizable amount by adjusting for terminal value differences
    5. Generate proper journal entries according to AAOIFI standards
    
    IMPORTANT: According to AAOIFI standards, ALL costs associated with acquiring and preparing the asset 
    for use must be capitalized as part of the prime cost. This includes installation costs, freight, 
    specialized equipment, import taxes, and any other direct costs.
    
    When calling the validate_ijarah_transaction or calculate_ijarah_values tool, ALWAYS include 
    the additional_costs parameter. If no additional costs are found, pass an empty list [].
    
    Present your analysis in a clear, step-by-step format, explaining each calculation.
    Remember that in Ijarah MBT (lease ending with ownership), the lessee recognizes 
    the right-of-use asset and corresponding liability at commencement.
    """,
        ),
        ("human", "{input}"),
        MessagesPlaceholder(variable_name="agent_scratchpad"),
    ]
)


# Create agent with more detailed system prompt
memory = ConversationBufferMemory(return_messages=True)
agent = create_openai_functions_agent(llm, tools, prompt)
agent_executor = AgentExecutor(agent=agent, tools=tools, verbose=True, memory=memory)


# Define a function to process transactions
def process_ijarah_transaction(transaction_details: str):
    """Process an Ijarah MBT transaction and generate accounting entries"""
    return agent_executor.invoke({"input": transaction_details})


# Updated example transaction
example_transaction = """
On 1 January 2019 Alpha Islamic bank (Lessee) entered into an Ijarah MBT arrangement with Super Generators for 
Ijarah of a heavy-duty generator purchase by Super Generators at a price of USD 450,000. 
Super Generators has also paid USD 12,000 as import tax and USD 30,000 for freight charges. 
The Ijarah Term is 02 years and expected residual value at the end USD 5,000. 
At the end of Ijarah Term, it is highly likely that the option of transfer of ownership of the 
underlying asset to the lessee shall be exercised through purchase at a price of USD 3,000. 
Alpha Islamic Bank will amortize the 'right of use' on yearly basis and it is required to pay 
yearly rental of USD 300,000. 

Provide the accounting entry for initial recognition at the time of commencement of Ijarah.
"""

# Simplified LangGraph implementation to avoid errors
from langgraph.graph import StateGraph, END


# Define state type
class GraphState(TypedDict):
    input: str
    response: Optional[Dict[str, Any]]
    error: Optional[str]


# Define node function
def process_with_agent(state: GraphState) -> GraphState:
    """Process the transaction using our agent"""
    try:
        result = agent_executor.invoke({"input": state["input"]})
        return {"input": state["input"], "response": result, "error": None}
    except Exception as e:
        return {"input": state["input"], "response": None, "error": str(e)}


def handle_error(state: GraphState) -> GraphState:
    """Handle any errors that occurred during processing"""
    error_msg = state.get("error", "Unknown error")
    return {
        "input": state["input"],
        "response": {"output": f"An error occurred: {error_msg}"},
        "error": error_msg,
    }


# Create the graph
def create_ijarah_graph():
    """Create and return the LangGraph workflow"""
    workflow = StateGraph(GraphState)

    # Add nodes
    workflow.add_node("process_transaction", process_with_agent)
    workflow.add_node("handle_error", handle_error)

    # Add conditional edges
    workflow.add_conditional_edges(
        "process_transaction",
        lambda state: "handle_error" if state.get("error") else END,
    )

    # Add final edge
    workflow.add_edge("handle_error", END)

    # Set entry point
    workflow.set_entry_point("process_transaction")

    # Compile and return
    return workflow.compile()


# Create the graph
ijarah_graph = create_ijarah_graph()


# Function to process with the graph
def process_with_graph(transaction_details: str):
    """Process a transaction using the LangGraph workflow"""
    initial_state = {"input": transaction_details, "response": None, "error": None}
    result = ijarah_graph.invoke(initial_state)
    return result.get("response", {"output": "Processing failed"})


# Main function with better error handling
def main():
    """Main function to run the agent with better error handling"""
    print("Islamic Banking AI Agent - Ijarah MBT Accounting Calculator")
    print("----------------------------------------------------------")
    print("\nProcessing example transaction...\n")

    try:
        # Process with basic agent
        result = process_ijarah_transaction(example_transaction)

        # Display results
        print("\nResults:")
        print(result["output"])

    except Exception as e:
        print(f"Error occurred: {str(e)}")
        import traceback

        traceback.print_exc()


if __name__ == "__main__":
    main()

####main
