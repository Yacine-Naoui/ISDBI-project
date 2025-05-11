from langchain_openai import ChatOpenAI, OpenAIEmbeddings
from langchain.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser, JsonOutputParser
from langchain_core.pydantic_v1 import BaseModel, Field, validator
from langchain_community.vectorstores import Chroma
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.document_loaders import PyPDFLoader
from langgraph.graph import StateGraph, END
import numpy as np
import matplotlib.pyplot as plt
import math
import os
from dotenv import load_dotenv
import re
import json
from typing import List, Dict, Any, Optional, TypedDict, Literal, Union, Tuple
import pandas as pd
import seaborn as sns
from io import BytesIO
import base64

# Load environment variables
load_dotenv()

# Define state structure
class OptimizationState(TypedDict):
    contract_type: str
    contract_details: Dict[str, Any]
    analysis: Optional[Dict[str, Any]]
    optimization_results: Optional[Dict[str, Any]]
    visualization: Optional[Dict[str, Any]]
    shariah_validation: Optional[Dict[str, Any]]
    recommendation: Optional[Dict[str, Any]]
    error: Optional[str]
    status: Literal["in_progress", "completed", "failed"]
    current_step: Literal["analyze", "optimize", "validate", "recommend", "complete", "error"]

# Define schema for contract parameters
class ContractParameters(BaseModel):
    contract_type: str = Field(..., description="Type of Islamic contract (Musharaka, Murabaha, Salam, Istisna'a, Ijarah)")
    capital: float = Field(..., description="Initial investment amount")
    tenor: float = Field(..., description="Contract duration in years")
    return_rates: List[float] = Field(..., description="Expected return rates for each period")
    loss_share: float = Field(..., description="Fraction of losses borne by the investor (0-1)")
    guarantee_level: float = Field(..., description="Fraction of capital or profit guaranteed (0-1)")
    compliance_score: float = Field(1.0, description="Measure of Shariah/AAOIFI compliance (0-1)")
    volatility: float = Field(..., description="Measure of unpredictability/risk")
    benchmark_return: float = Field(0.12, description="Reference rate to penalize interest-like deals")
    loss_cap: float = Field(0.2, description="Maximum negative return allowed (as positive number)")
    
    @validator('loss_share', 'guarantee_level', 'compliance_score')
    def must_be_between_zero_and_one(cls, v, values, **kwargs):
        if not 0 <= v <= 1:
            raise ValueError('Value must be between 0 and 1')
        return v

# Function to calculate RV score
def calculate_rv(params):
    """
    Calculate Risk-Reward Value (RV) based on the formula from Tawfid AI Platform
    
    RVraw = C × L × (Σ(max(rt, -δ) - G*rmax) / (σ + ε))
    RV = 1 / (1 + e^(-k*RVraw - m))
    """
    # Extract parameters
    C = params["compliance_score"]
    L = params["loss_share"]
    returns = params["return_rates"]
    G = params["guarantee_level"]
    rmax = params["benchmark_return"]
    sigma = params["volatility"]
    delta = params["loss_cap"]
    
    # Constants
    epsilon = 0.01  # Small constant to avoid division by zero
    k = 10  # Scaling parameter
    m = 0.2  # Shift parameter
    
    # Calculate time weights
    T = params["tenor"]
    if isinstance(returns, list) and len(returns) > 1:
        # If we have multiple return periods
        alpha = [1/len(returns)] * len(returns)
    else:
        # If we have a single return
        alpha = [1]
        if not isinstance(returns, list):
            returns = [returns]
    
    # Calculate the numerator term
    numerator = 0
    for i, r in enumerate(returns):
        # Ensure r doesn't go below -delta
        adjusted_r = max(r, -delta)
        numerator += alpha[i] * (adjusted_r - G * rmax)
    
    # Calculate RVraw
    denominator = sigma + epsilon
    RVraw = C * L * (numerator / denominator)
    
    # Apply logistic function to map to [0,1]
    RV = 1 / (1 + math.exp(-k * RVraw - m))
    
    return {
        "RVraw": RVraw,
        "RV": RV,
        "components": {
            "compliance": C,
            "risk_sharing": L,
            "return_component": numerator,
            "risk_component": denominator
        }
    }

# Function to analyze contract parameters
def analyze_contract(state: OptimizationState) -> OptimizationState:
    """
    Analyze the contract parameters and calculate initial RV
    """
    contract_type = state["contract_type"]
    contract_details = state["contract_details"]
    
    print(f"Analyzing {contract_type} contract...")
    
    try:
        # Calculate RV for the current parameters
        rv_result = calculate_rv(contract_details)
        
        # Create analysis
        analysis = {
            "contract_type": contract_type,
            "parameters": contract_details,
            "rv_score": rv_result["RV"],
            "raw_score": rv_result["RVraw"],
            "components": rv_result["components"],
            "interpretation": interpret_rv(rv_result["RV"]),
            "shariah_concerns": identify_shariah_concerns(contract_details),
            "risk_assessment": assess_risk(contract_details)
        }
        
        print(f"Initial RV score: {rv_result['RV']:.4f}")
        
        return {
            **state,
            "analysis": analysis,
            "current_step": "optimize",
            "status": "in_progress"
        }
    except Exception as e:
        error_msg = f"Error analyzing contract: {str(e)}"
        print(error_msg)
        return {
            **state,
            "error": error_msg,
            "current_step": "error",
            "status": "failed"
        }

# Function to interpret RV score
def interpret_rv(rv_score):
    """Provide interpretation of RV score"""
    if rv_score >= 0.8:
        return "Excellent - Highly Shariah-compliant with optimal risk-reward balance"
    elif rv_score >= 0.6:
        return "Good - Strong Shariah compliance with reasonable risk-reward"
    elif rv_score >= 0.4:
        return "Acceptable - Adequate Shariah compliance but risk-reward could be improved"
    elif rv_score >= 0.2:
        return "Concerning - Shariah compliance or risk-reward issues need addressing"
    else:
        return "Poor - Significant Shariah compliance or risk-reward problems"

# Function to identify Shariah concerns
def identify_shariah_concerns(params):
    """Identify potential Shariah compliance concerns"""
    concerns = []
    
    # Check for guaranteed returns (resembles interest)
    if params["guarantee_level"] > 0.5:
        concerns.append({
            "issue": "High guarantee level",
            "description": "Guarantees exceeding 50% may resemble conventional interest-based contracts",
            "recommendation": "Reduce guarantee level to enhance risk-sharing"
        })
    
    # Check for limited risk-sharing
    if params["loss_share"] < 0.3:
        concerns.append({
            "issue": "Limited loss-sharing",
            "description": "Low loss-sharing reduces genuine partnership aspect",
            "recommendation": "Increase loss-sharing to align with Shariah principles"
        })
    
    # Check for high benchmark return alignment
    if params["benchmark_return"] > 0.15:
        concerns.append({
            "issue": "High benchmark return",
            "description": "Benchmarking to high conventional rates may indicate interest-like structure",
            "recommendation": "Consider alternative benchmarks or profit-sharing mechanisms"
        })
    
    return concerns

# Function to assess risk
def assess_risk(params):
    """Provide risk assessment"""
    risk_level = "Medium"  # Default
    
    # Assess based on volatility
    if params["volatility"] > 0.15:
        risk_level = "High"
    elif params["volatility"] < 0.08:
        risk_level = "Low"
    
    # Adjust based on loss-sharing
    if params["loss_share"] > 0.7:
        risk_level = "High" if risk_level != "High" else "Very High"
    elif params["loss_share"] < 0.3:
        risk_level = "Low" if risk_level != "Low" else "Very Low"
    
    # Create assessment
    assessment = {
        "risk_level": risk_level,
        "volatility_impact": f"Volatility of {params['volatility']} contributes to {risk_level} risk",
        "loss_sharing_impact": f"Loss-sharing of {params['loss_share']} impacts investor exposure"
    }
    
    return assessment

# Function to optimize contract parameters
def optimize_contract(state: OptimizationState) -> OptimizationState:
    """
    Optimize contract parameters to maximize RV score
    """
    contract_type = state["contract_type"]
    contract_details = state["contract_details"]
    analysis = state["analysis"]
    
    print(f"Optimizing {contract_type} contract parameters...")
    
    try:
        # Create a grid of possible parameter values
        parameter_grid = {
            "loss_share": [round(x, 2) for x in np.linspace(0.3, 0.7, 5)],
            "guarantee_level": [round(x, 2) for x in np.linspace(0, 0.5, 6)],
            "volatility": [round(x, 2) for x in np.linspace(max(0.05, contract_details["volatility"]*0.7), 
                                                 min(0.2, contract_details["volatility"]*1.3), 4)]
        }
        
        # Generate all combinations
        results = []
        for loss_share in parameter_grid["loss_share"]:
            for guarantee_level in parameter_grid["guarantee_level"]:
                for volatility in parameter_grid["volatility"]:
                    # Create parameter set
                    test_params = {**contract_details}
                    test_params["loss_share"] = loss_share
                    test_params["guarantee_level"] = guarantee_level
                    test_params["volatility"] = volatility
                    
                    # Calculate RV
                    rv_result = calculate_rv(test_params)
                    
                    # Store result
                    results.append({
                        "loss_share": loss_share,
                        "guarantee_level": guarantee_level,
                        "volatility": volatility,
                        "rv_score": rv_result["RV"],
                        "raw_score": rv_result["RVraw"]
                    })
        
        # Convert to DataFrame for analysis
        results_df = pd.DataFrame(results)
        
        # Find optimal parameters
        optimal_result = results_df.loc[results_df["rv_score"].idxmax()]
        
        # Generate optimization report
        optimization_results = {
            "original_parameters": {
                "loss_share": contract_details["loss_share"],
                "guarantee_level": contract_details["guarantee_level"],
                "volatility": contract_details["volatility"]
            },
            "original_rv": analysis["rv_score"],
            "optimal_parameters": {
                "loss_share": optimal_result["loss_share"],
                "guarantee_level": optimal_result["guarantee_level"],
                "volatility": optimal_result["volatility"]
            },
            "optimal_rv": optimal_result["rv_score"],
            "improvement": optimal_result["rv_score"] - analysis["rv_score"],
            "grid_search_results": results
        }
        
        print(f"Optimization complete. Optimal RV: {optimal_result['rv_score']:.4f}")
        
        return {
            **state,
            "optimization_results": optimization_results,
            "current_step": "validate",
            "status": "in_progress"
        }
    except Exception as e:
        error_msg = f"Error optimizing contract: {str(e)}"
        print(error_msg)
        return {
            **state,
            "error": error_msg,
            "current_step": "error",
            "status": "failed"
        }

# Function to create visualizations
def visualize_results(state: OptimizationState) -> OptimizationState:
    """
    Create visualizations of the optimization results
    """
    print("Creating visualizations...")
    
    try:
        optimization_results = state["optimization_results"]
        
        # Convert grid search results to DataFrame if not already
        if not isinstance(optimization_results["grid_search_results"], pd.DataFrame):
            results_df = pd.DataFrame(optimization_results["grid_search_results"])
        else:
            results_df = optimization_results["grid_search_results"]
        
        # Create heatmap of RV scores for loss_share vs guarantee_level
        plt.figure(figsize=(10, 8))
        
        # Create pivot table
        pivot_table = results_df.pivot_table(
            values='rv_score', 
            index='loss_share', 
            columns='guarantee_level', 
            aggfunc='mean'
        )
        
        # Create heatmap
        heatmap = sns.heatmap(pivot_table, annot=True, cmap='viridis', fmt=".3f")
        plt.title('RV Score by Loss Share and Guarantee Level')
        plt.tight_layout()
        
        # Convert to base64 for embedding
        buffer = BytesIO()
        plt.savefig(buffer, format='png')
        buffer.seek(0)
        heatmap_img = base64.b64encode(buffer.getvalue()).decode()
        plt.close()
        
        # Create 3D surface plot
        fig = plt.figure(figsize=(10, 8))
        ax = fig.add_subplot(111, projection='3d')
        
        # Get unique values
        loss_shares = sorted(results_df['loss_share'].unique())
        guarantee_levels = sorted(results_df['guarantee_level'].unique())
        
        # Create meshgrid
        X, Y = np.meshgrid(guarantee_levels, loss_shares)
        Z = np.zeros(X.shape)
        
        # Fill Z values
        for i, ls in enumerate(loss_shares):
            for j, gl in enumerate(guarantee_levels):
                subset = results_df[(results_df['loss_share'] == ls) & (results_df['guarantee_level'] == gl)]
                if not subset.empty:
                    Z[i, j] = subset['rv_score'].mean()
        
        # Create surface plot
        surf = ax.plot_surface(X, Y, Z, cmap='viridis', edgecolor='none', alpha=0.8)
        
        # Add labels
        ax.set_xlabel('Guarantee Level')
        ax.set_ylabel('Loss Share')
        ax.set_zlabel('RV Score')
        ax.set_title('3D Surface of RV Score')
        
        # Add color bar
        fig.colorbar(surf, ax=ax, shrink=0.5, aspect=5)
        
        # Convert to base64 for embedding
        buffer = BytesIO()
        plt.savefig(buffer, format='png')
        buffer.seek(0)
        surface_img = base64.b64encode(buffer.getvalue()).decode()
        plt.close()
        
        # Create parameter comparison chart
        plt.figure(figsize=(10, 6))
        
        # Parameters to compare
        params = ['loss_share', 'guarantee_level', 'volatility']
        original = [optimization_results["original_parameters"][p] for p in params]
        optimal = [optimization_results["optimal_parameters"][p] for p in params]
        
        x = np.arange(len(params))
        width = 0.35
        
        # Create bars
        plt.bar(x - width/2, original, width, label='Original')
        plt.bar(x + width/2, optimal, width, label='Optimal')
        
        # Add labels and title
        plt.xlabel('Parameter')
        plt.ylabel('Value')
        plt.title('Comparison of Original vs Optimal Parameters')
        plt.xticks(x, params)
        plt.legend()
        
        # Convert to base64 for embedding
        buffer = BytesIO()
        plt.savefig(buffer, format='png')
        buffer.seek(0)
        comparison_img = base64.b64encode(buffer.getvalue()).decode()
        plt.close()
        
        # Store visualizations
        visualization = {
            "heatmap": heatmap_img,
            "surface_plot": surface_img,
            "parameter_comparison": comparison_img
        }
        
        print("Visualizations created successfully")
        
        return {
            **state,
            "visualization": visualization,
            "current_step": "recommend",
            "status": "in_progress"
        }
    except Exception as e:
        error_msg = f"Error creating visualizations: {str(e)}"
        print(error_msg)
        return {
            **state,
            "error": error_msg,
            "current_step": "error",
            "status": "failed"
        }


# Function to validate Shariah compliance
def validate_shariah_compliance(state: OptimizationState) -> OptimizationState:
    """
    Validate Shariah compliance of the optimized contract
    """
    contract_type = state["contract_type"]
    optimization_results = state["optimization_results"]
    
    print(f"Validating Shariah compliance for optimized {contract_type} contract...")
    
    try:
        # Get original and optimized parameters
        original_params = state["contract_details"]
        
        # Create optimized parameters
        optimized_params = {**original_params}
        for param, value in optimization_results["optimal_parameters"].items():
            optimized_params[param] = value
        
        # Get the contract-specific validation logic
        validation_result = get_shariah_validation(contract_type, optimized_params)
        
        # Apply visualizations
        visualization_results = visualize_results(state)
        
        # Check for visualization errors
        if visualization_results.get("error"):
            return visualization_results
        
        return {
            **visualization_results,
            "shariah_validation": validation_result,
            "current_step": "recommend",
            "status": "in_progress"
        }
    except Exception as e:
        error_msg = f"Error validating Shariah compliance: {str(e)}"
        print(error_msg)
        return {
            **state,
            "error": error_msg,
            "current_step": "error",
            "status": "failed"
        }

# Function to get Shariah validation based on contract type
def get_shariah_validation(contract_type, params):
    """
    Get Shariah validation specific to contract type
    """
    # Base validation
    validation = {
        "is_compliant": True,
        "compliance_score": params["compliance_score"],
        "issues": [],
        "recommendations": []
    }
    
    # Common validations
    if params["guarantee_level"] > 0.5:
        validation["is_compliant"] = False
        validation["issues"].append({
            "severity": "High",
            "description": "Excessive guarantee level resembles interest-based financing",
            "AAOIFI_reference": "Shariah Standard 8, Clause 4/1"
        })
        validation["recommendations"].append("Reduce guarantee level below 50%")
    
    if params["loss_share"] < 0.3 and contract_type.lower() in ["musharaka", "mudaraba"]:
        validation["is_compliant"] = False
        validation["issues"].append({
            "severity": "High",
            "description": "Insufficient loss sharing violates partnership principles",
            "AAOIFI_reference": "Shariah Standard 12, Clause 3/1/3"
        })
        validation["recommendations"].append("Increase loss sharing to at least 30%")
    
    # Contract-specific validations
    if contract_type.lower() == "murabaha":
        if not params.get("asset_ownership", True):
            validation["is_compliant"] = False
            validation["issues"].append({
                "severity": "Critical",
                "description": "IFI must own asset before selling in Murabaha",
                "AAOIFI_reference": "Shariah Standard 8, Clause 3/1/1"
            })
    
    elif contract_type.lower() == "ijarah":
        if not params.get("maintenance_responsibility", True):
            validation["is_compliant"] = False
            validation["issues"].append({
                "severity": "High",
                "description": "Major maintenance must be lessor's responsibility",
                "AAOIFI_reference": "Shariah Standard 9, Clause 5/1/7"
            })
    
    elif contract_type.lower() == "salam":
        if params.get("parallel_salam_dependent", False):
            validation["is_compliant"] = False
            validation["issues"].append({
                "severity": "High",
                "description": "Parallel Salam cannot be contingent on original Salam",
                "AAOIFI_reference": "Shariah Standard 10, Clause 4/2"
            })
    
    return validation

# Function to generate final recommendations
def generate_recommendations(state: OptimizationState) -> OptimizationState:
    """
    Generate final recommendations based on optimization and validation
    """
    contract_type = state["contract_type"]
    analysis = state["analysis"]
    optimization_results = state["optimization_results"]
    shariah_validation = state["shariah_validation"]
    
    print(f"Generating recommendations for {contract_type} contract...")
    
    try:
        # Create optimized parameters
        original_params = state["contract_details"]
        optimized_params = {**original_params}
        for param, value in optimization_results["optimal_parameters"].items():
            optimized_params[param] = value
        
        # Calculate journal entries for the optimized contract
        journal_entries = generate_journal_entries(contract_type, optimized_params)
        
        # Generate recommendations
        recommendations = {
            "summary": f"Optimized {contract_type} contract with RV score of {optimization_results['optimal_rv']:.4f}",
            "parameter_changes": [
                {
                    "parameter": param,
                    "original": optimization_results["original_parameters"][param],
                    "optimized": value,
                    "change_percentage": ((value - optimization_results["original_parameters"][param]) / 
                                         optimization_results["original_parameters"][param] * 100) 
                                         if optimization_results["original_parameters"][param] != 0 else float('inf')
                }
                for param, value in optimization_results["optimal_parameters"].items()
            ],
            "shariah_status": "Compliant" if shariah_validation["is_compliant"] else "Non-Compliant",
            "risk_profile": {
                "original": analysis["risk_assessment"]["risk_level"],
                "optimized": assess_risk(optimized_params)["risk_level"]
            },
            "implementation_steps": generate_implementation_steps(contract_type, optimized_params),
            "journal_entries": journal_entries,
            "additional_notes": generate_additional_notes(contract_type, optimized_params)
        }
        
        print("Recommendations generated successfully")
        
        return {
            **state,
            "recommendation": recommendations,
            "current_step": "complete",
            "status": "completed"
        }
    except Exception as e:
        error_msg = f"Error generating recommendations: {str(e)}"
        print(error_msg)
        return {
            **state,
            "error": error_msg,
            "current_step": "error",
            "status": "failed"
        }

# Function to generate journal entries
def generate_journal_entries(contract_type, params):
    """
    Generate journal entries for the optimized contract
    """
    entries = []
    
    # Common formatting for monetary values
    def format_amount(amount):
        return f"{amount:,.2f}"
    
    if contract_type.lower() == "musharaka":
        # Extract parameters
        capital = params["capital"]
        returns = params["return_rates"]
        tenor = params["tenor"]
        
        # Calculate IFI and customer contributions
        if params.get("ifi_share", None) is not None:
            ifi_share = params["ifi_share"]
        else:
            ifi_share = 0.7  # Default
        
        ifi_capital = capital * ifi_share
        customer_capital = capital * (1 - ifi_share)
        
        # Initial financing
        entries.append({
            "date": "Day 1",
            "description": "Initial Financing",
            "entries": [
                {"account": "Musharaka Financing", "debit": format_amount(ifi_capital), "credit": ""},
                {"account": "Cash", "debit": "", "credit": format_amount(ifi_capital)},
                {"account": "Customer Capital Contribution", "debit": format_amount(customer_capital), "credit": ""},
                {"account": "Cash", "debit": "", "credit": format_amount(customer_capital)}
            ]
        })
        
        # Calculate profit/loss for each period
        for i, r in enumerate(returns):
            period_result = capital * r
            ifi_share_amount = period_result * ifi_share
            
            if period_result >= 0:  # Profit
                entries.append({
                    "date": f"Period {i+1}",
                    "description": "Profit Distribution",
                    "entries": [
                        {"account": "Cash", "debit": format_amount(period_result), "credit": ""},
                        {"account": "Musharaka Profit", "debit": "", "credit": format_amount(ifi_share_amount)},
                        {"account": "Partner's Share", "debit": "", "credit": format_amount(period_result - ifi_share_amount)}
                    ]
                })
            else:  # Loss
                loss = abs(period_result)
                ifi_loss = loss * params["loss_share"] * ifi_share
                customer_loss = loss - ifi_loss
                
                entries.append({
                    "date": f"Period {i+1}",
                    "description": "Loss Recognition",
                    "entries": [
                        {"account": "Musharaka Loss", "debit": format_amount(ifi_loss), "credit": ""},
                        {"account": "Partner's Loss", "debit": format_amount(customer_loss), "credit": ""},
                        {"account": "Musharaka Financing", "debit": "", "credit": format_amount(loss)}
                    ]
                })
    
    elif contract_type.lower() == "murabaha":
        # Extract parameters
        capital = params["capital"]
        tenor = params["tenor"]
        
        # Calculate markup based on return rate
        if len(params["return_rates"]) > 0:
            return_rate = params["return_rates"][0]
        else:
            return_rate = 0.15  # Default
        
        cost_basis = capital
        markup = cost_basis * return_rate
        sale_price = cost_basis + markup
        
        # Asset purchase
        entries.append({
            "date": "Day 1",
            "description": "Asset Purchase",
            "entries": [
                {"account": "Murabaha Inventory", "debit": format_amount(cost_basis), "credit": ""},
                {"account": "Cash", "debit": "", "credit": format_amount(cost_basis)}
            ]
        })
        
        # Sale to customer
        entries.append({
            "date": "Day 1",
            "description": "Sale to Customer",
            "entries": [
                {"account": "Murabaha Receivable", "debit": format_amount(sale_price), "credit": ""},
                {"account": "Inventory", "debit": "", "credit": format_amount(cost_basis)},
                {"account": "Deferred Profit", "debit": "", "credit": format_amount(markup)}
            ]
        })
        
        # Profit recognition (simplified)
        monthly_profit = markup / (tenor * 12)
        entries.append({
            "date": "Monthly",
            "description": "Profit Recognition",
            "entries": [
                {"account": "Deferred Profit", "debit": format_amount(monthly_profit), "credit": ""},
                {"account": "Income", "debit": "", "credit": format_amount(monthly_profit)}
            ]
        })
    
    elif contract_type.lower() == "ijarah":
        # Extract parameters
        capital = params["capital"]
        tenor = params["tenor"]
        
        # Calculate rental
        if len(params["return_rates"]) > 0:
            return_rate = params["return_rates"][0]
        else:
            return_rate = 0.08  # Default
        
        annual_rental = capital * return_rate
        
        # Asset acquisition
        entries.append({
            "date": "Day 1",
            "description": "Asset Acquisition",
            "entries": [
                {"account": "Ijarah Asset", "debit": format_amount(capital), "credit": ""},
                {"account": "Cash", "debit": "", "credit": format_amount(capital)}
            ]
        })
        
        # Rental payments (annual)
        entries.append({
            "date": "Annual",
            "description": "Rental Payment",
            "entries": [
                {"account": "Cash", "debit": format_amount(annual_rental), "credit": ""},
                {"account": "Ijarah Income", "debit": "", "credit": format_amount(annual_rental)}
            ]
        })
        
        # Depreciation
        annual_depreciation = capital / tenor
        entries.append({
            "date": "Annual",
            "description": "Depreciation",
            "entries": [
                {"account": "Depreciation Expense", "debit": format_amount(annual_depreciation), "credit": ""},
                {"account": "Accumulated Depreciation", "debit": "", "credit": format_amount(annual_depreciation)}
            ]
        })
    
    # Add more contract types as needed...
    
    return entries

# Function to generate implementation steps
def generate_implementation_steps(contract_type, params):
    """
    Generate implementation steps for the optimized contract
    """
    steps = []
    
    # Common steps
    steps.append({
        "step": 1,
        "description": "Review and approve optimized contract parameters",
        "stakeholders": ["Shariah Board", "Risk Management", "Business Unit"]
    })
    
    steps.append({
        "step": 2,
        "description": "Document parameter changes and expected RV improvement",
        "stakeholders": ["Risk Management", "Finance"]
    })
    
    # Contract-specific steps
    if contract_type.lower() == "musharaka":
        steps.append({
            "step": 3,
            "description": f"Update loss-sharing ratio to {params['loss_share']:.2f}",
            "stakeholders": ["Legal", "Business Unit"]
        })
        steps.append({
            "step": 4,
            "description": f"Adjust guarantee provisions to {params['guarantee_level']:.2f}",
            "stakeholders": ["Legal", "Risk Management"]
        })
        
        steps.append({
            "step": 5,
            "description": "Update documentation to reflect optimized partnership structure",
            "stakeholders": ["Legal", "Shariah Compliance"]
        })
    
    elif contract_type.lower() == "murabaha":
        steps.append({
            "step": 3,
            "description": "Ensure asset ownership prior to customer sale",
            "stakeholders": ["Operations", "Legal"]
        })
        
        steps.append({
            "step": 4,
            "description": f"Adjust markup calculation to reflect optimized parameters",
            "stakeholders": ["Finance", "Business Unit"]
        })
        
        steps.append({
            "step": 5,
            "description": "Update offer and acceptance documentation",
            "stakeholders": ["Legal", "Sales"]
        })
    
    elif contract_type.lower() == "ijarah":
        steps.append({
            "step": 3,
            "description": "Verify lessor responsibility for major maintenance",
            "stakeholders": ["Operations", "Legal"]
        })
        
        steps.append({
            "step": 4,
            "description": f"Adjust rental calculations and periodicity",
            "stakeholders": ["Finance", "Business Unit"]
        })
        
        steps.append({
            "step": 5,
            "description": "Update lease and purchase undertaking documentation",
            "stakeholders": ["Legal", "Shariah Compliance"]
        })
    
    # Final common steps
    steps.append({
        "step": len(steps) + 1,
        "description": "Obtain final Shariah approval on revised structure",
        "stakeholders": ["Shariah Board", "Compliance"]
    })
    
    steps.append({
        "step": len(steps) + 1,
        "description": "Implement monitoring system for actual vs. optimized performance",
        "stakeholders": ["Risk Management", "IT", "Finance"]
    })
    
    return steps

# Function to generate additional notes
def generate_additional_notes(contract_type, params):
    """
    Generate additional notes for the optimized contract
    """
    notes = []
    
    # Common notes
    notes.append({
        "topic": "RV Framework",
        "content": "The optimization relies on the Risk-Reward Value framework which balances Shariah compliance, risk-sharing, and economic returns."
    })
    
    notes.append({
        "topic": "Parameter Sensitivity",
        "content": f"The loss-sharing parameter ({params['loss_share']:.2f}) has significant impact on both Shariah compliance and risk profile."
    })
    
    # Contract-specific notes
    if contract_type.lower() == "musharaka":
        notes.append({
            "topic": "Loss Recognition",
            "content": "Ensure accounting systems can properly record loss-sharing according to the optimized parameters."
        })
        
        notes.append({
            "topic": "Exit Strategy",
            "content": "Consider documenting an exit mechanism aligned with the optimized parameters."
        })
    
    elif contract_type.lower() == "murabaha":
        notes.append({
            "topic": "Asset Risk",
            "content": "The optimization assumes the asset is available and can be acquired before customer sale."
        })
        
        notes.append({
            "topic": "Deferred Payment",
            "content": "The optimized structure reflects deferred payment considerations in compliance with AAOIFI FAS 28."
        })
    
    elif contract_type.lower() == "ijarah":
        notes.append({
            "topic": "Asset Ownership",
            "content": "The optimization assumes the lessor maintains ownership and major maintenance responsibilities."
        })
        
        notes.append({
            "topic": "Lease Term",
            "content": "The optimized parameters are based on a term of " + str(params["tenor"]) + " years."
        })
    
    return notes

# Function to handle errors
def handle_error(state: OptimizationState) -> OptimizationState:
    """
    Handle errors in the process
    """
    error = state["error"]
    print(f"Error occurred: {error}")
    
    # Create error report
    error_report = {
        "error": error,
        "step": state["current_step"],
        "recommendation": "Please review parameters and try again",
        "debug_info": {k: v for k, v in state.items() if k not in ["error", "current_step", "status"]}
    }
    
    return {
        **state,
        "recommendation": error_report,
        "status": "failed"
    }

# Define the LangGraph workflow
def build_optimization_workflow():
    """
    Build the LangGraph workflow for contract optimization
    """
    # Define the workflow
    workflow = StateGraph(OptimizationState)
    
    # Add nodes
    workflow.add_node("analyze", analyze_contract)
    workflow.add_node("optimize", optimize_contract)
    workflow.add_node("validate", validate_shariah_compliance)
    workflow.add_node("recommend", generate_recommendations)
    workflow.add_node("handle_error", handle_error)
    
    # Add conditional edges with clearer logic
    workflow.add_conditional_edges(
        "analyze",
        lambda state: "optimize" if state.get("current_step") == "optimize" and not state.get("error") else "handle_error"
    )
    
    workflow.add_conditional_edges(
        "optimize",
        lambda state: "validate" if state.get("current_step") == "validate" and not state.get("error") else "handle_error"
    )
    
    workflow.add_conditional_edges(
        "validate",
        lambda state: "recommend" if state.get("current_step") == "recommend" and not state.get("error") else "handle_error"
    )
    
    # Add terminal edges
    workflow.add_edge("recommend", END)
    workflow.add_edge("handle_error", END)
    
    # Set entry point
    workflow.set_entry_point("analyze")
    
    # Compile the workflow
    return workflow.compile()

# Function to run optimization
def optimize_contract_parameters(contract_details):
    """
    Optimize contract parameters using the LangGraph workflow
    
    Args:
        contract_details: Details of the contract to optimize
        
    Returns:
        Optimization results
    """
    # Initialize the workflow
    workflow = build_optimization_workflow()
    
    # Extract contract type
    contract_type = contract_details["contract_type"]
    
    # Set initial state
    initial_state = {
        "contract_type": contract_type,
        "contract_details": contract_details,
        "analysis": None,
        "optimization_results": None,
        "visualization": None,
        "shariah_validation": None,
        "recommendation": None,
        "error": None,
        "status": "in_progress",
        "current_step": "analyze"
    }
    
    # Execute the workflow
    print(f"\n{'='*20} Optimizing {contract_type} Contract {'='*20}")
    
    # Track and display progress
    final_state = None
    
    try:
        # Using invoke instead of stream to get the final state directly
        final_state = workflow.invoke(initial_state)
    except Exception as e:
        print(f"Error during workflow execution: {str(e)}")
        # Create a fallback state with error info if the workflow fails
        final_state = {
            "contract_type": contract_type,
            "contract_details": contract_details,
            "error": str(e),
            "status": "failed",
            "current_step": "error"
        }
    
    # Return optimization results
    return final_state

# Function to create a PDF report
def create_optimization_report(optimization_results, filename="Contract_Optimization_Report.pdf"):
    """
    Create a PDF report of the optimization results
    
    Args:
        optimization_results: Results of the optimization
        filename: Output filename
    """
    from fpdf import FPDF
    
    # Create PDF
    pdf = FPDF()
    pdf.add_page()
    
    # Set font
    pdf.set_font("Arial", "B", 16)
    
    # Title
    contract_type = optimization_results["contract_type"]
    pdf.cell(0, 10, f"Shariah-Compliant {contract_type} Optimization Report", 0, 1, "C")
    pdf.line(10, 30, 200, 30)
    
    # Add content
    pdf.set_font("Arial", "", 12)
    pdf.ln(10)
    
    # Summary
    pdf.set_font("Arial", "B", 14)
    pdf.cell(0, 10, "Executive Summary", 0, 1)
    pdf.set_font("Arial", "", 12)
    
    recommendation = optimization_results.get("recommendation", {})
    summary = recommendation.get("summary", "No summary available")
    pdf.multi_cell(0, 10, summary)
    pdf.ln(5)
    
    # Parameter changes
    pdf.set_font("Arial", "B", 14)
    pdf.cell(0, 10, "Optimized Parameters", 0, 1)
    pdf.set_font("Arial", "", 12)
    
    parameter_changes = recommendation.get("parameter_changes", [])
    if parameter_changes:
        # Create table header
        pdf.set_fill_color(200, 220, 255)
        pdf.cell(50, 10, "Parameter", 1, 0, "C", True)
        pdf.cell(40, 10, "Original", 1, 0, "C", True)
        pdf.cell(40, 10, "Optimized", 1, 0, "C", True)
        pdf.cell(50, 10, "Change (%)", 1, 1, "C", True)
        
        # Add data rows
        pdf.set_fill_color(255, 255, 255)
        for change in parameter_changes:
            pdf.cell(50, 10, change["parameter"], 1, 0, "L")
            pdf.cell(40, 10, f"{change['original']:.4f}", 1, 0, "C")
            pdf.cell(40, 10, f"{change['optimized']:.4f}", 1, 0, "C")
            pdf.cell(50, 10, f"{change['change_percentage']:.2f}%", 1, 1, "C")
    
    pdf.ln(5)
    
    # Shariah validation
    pdf.set_font("Arial", "B", 14)
    pdf.cell(0, 10, "Shariah Compliance", 0, 1)
    pdf.set_font("Arial", "", 12)
    
    shariah_status = recommendation.get("shariah_status", "Unknown")
    pdf.cell(0, 10, f"Status: {shariah_status}", 0, 1)
    
    shariah_validation = optimization_results.get("shariah_validation", {})
    issues = shariah_validation.get("issues", [])
    
    if issues:
        pdf.ln(5)
        pdf.cell(0, 10, "Compliance Issues:", 0, 1)
        for issue in issues:
            pdf.multi_cell(0, 10, f"- {issue.get('description', '')}")
    else:
        pdf.multi_cell(0, 10, "No Shariah compliance issues identified.")
    
    pdf.ln(5)
    
    # Implementation steps
    pdf.set_font("Arial", "B", 14)
    pdf.cell(0, 10, "Implementation Steps", 0, 1)
    pdf.set_font("Arial", "", 12)
    
    implementation_steps = recommendation.get("implementation_steps", [])
    for step in implementation_steps:
        pdf.multi_cell(0, 10, f"{step['step']}. {step['description']}")
        stakeholders = ", ".join(step.get("stakeholders", []))
        pdf.multi_cell(0, 10, f"   Stakeholders: {stakeholders}")
    
    pdf.ln(5)
    
    # Add visualizations if available
    visualization = optimization_results.get("visualization", {})
    if visualization:
        pdf.add_page()
        pdf.set_font("Arial", "B", 14)
        pdf.cell(0, 10, "Visualizations", 0, 1)
        
        # Heatmap
        heatmap_img = visualization.get("heatmap")
        if heatmap_img:
            pdf.set_font("Arial", "B", 12)
            pdf.cell(0, 10, "RV Score Heatmap", 0, 1)
            
            # Save base64 image to temp file
            import base64
            import tempfile
            
            with tempfile.NamedTemporaryFile(delete=False, suffix='.png') as temp:
                temp.write(base64.b64decode(heatmap_img))
                temp_path = temp.name
            
            pdf.image(temp_path, x=10, y=None, w=180)
            
            # Clean up
            import os
            os.unlink(temp_path)
        
        # Parameter comparison
        pdf.add_page()
        comparison_img = visualization.get("parameter_comparison")
        if comparison_img:
            pdf.set_font("Arial", "B", 12)
            pdf.cell(0, 10, "Parameter Comparison", 0, 1)
            
            with tempfile.NamedTemporaryFile(delete=False, suffix='.png') as temp:
                temp.write(base64.b64decode(comparison_img))
                temp_path = temp.name
            
            pdf.image(temp_path, x=10, y=None, w=180)
            
            # Clean up
            os.unlink(temp_path)
    
    # Save PDF
    pdf.output(filename)
    print(f"PDF report saved as {filename}")

# Example usage
def run_example_scenario():
    """
    Run an example scenario to demonstrate the system
    """
    # Define example Musharaka contract
    musharaka_contract = {
        "contract_type": "Musharaka",
        "capital": 1000000,  # Initial investment (USD)
        "tenor": 2,  # Years
        "return_rates": [0.08, -0.03],  # Expected returns for each period
        "loss_share": 0.3,  # Initial loss sharing ratio
        "guarantee_level": 0.2,  # Initial guarantee level
        "compliance_score": 1.0,  # Assuming fully compliant
        "volatility": 0.12,  # Volatility measure
        "benchmark_return": 0.12,  # Benchmark return rate
        "loss_cap": 0.2,  # Maximum acceptable loss
        "ifi_share": 0.6  # IFI's share of the capital
    }
    
    # Optimize the contract
    optimization_results = optimize_contract_parameters(musharaka_contract)
    
    # Create PDF report
    if optimization_results.get("status") == "completed":
        try:
            create_optimization_report(optimization_results, "Musharaka_Optimization_Report.pdf")
        except Exception as e:
            print(f"Error creating PDF report: {str(e)}")
    
    # Print key results
    print("\n" + "="*50)
    print("Optimization Results Summary")
    print("="*50)
    
    # Make sure to use .get() to avoid KeyError
    print(f"Status: {optimization_results.get('status', 'Unknown')}")
    
    if optimization_results.get("status") == "completed":
        recommendation = optimization_results.get("recommendation", {})
        
        print(f"Summary: {recommendation.get('summary', 'No summary available')}")
        
        print("\nParameter Changes:")
        for change in recommendation.get("parameter_changes", []):
            print(f"  {change['parameter']}: {change['original']:.4f} -> {change['optimized']:.4f} ({change['change_percentage']:.2f}%)")
        
        print(f"\nShariah Status: {recommendation.get('shariah_status', 'Unknown')}")
        
        print("\nRisk Profile:")
        risk_profile = recommendation.get("risk_profile", {})
        print(f"  Original: {risk_profile.get('original', 'Unknown')}")
        print(f"  Optimized: {risk_profile.get('optimized', 'Unknown')}")
    else:
        print(f"Error: {optimization_results.get('error', 'Unknown error')}")

# Main function
def main():
    print("Shariah-Compliant Investment Optimization System")
    print("===============================================")
    
    try:
        # Run example scenario
        run_example_scenario()
    except Exception as e:
        print(f"Error running example scenario: {str(e)}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    main()