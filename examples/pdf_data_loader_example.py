"""
PDF Data Loader Example

This example demonstrates how to use the extended DataLoaderToolsAgent
to extract data from PDF documents for the Fetch AI platform.

This example shows:
1. Loading structured data from PDF tables
2. Extracting text content from PDFs
3. Using the smart extraction method
4. Integrating with the existing data science workflow

Usage:
    python examples/pdf_data_loader_example.py
"""

import os
import sys
import pandas as pd
from dotenv import load_dotenv
from langchain_openai import ChatOpenAI

# Add the src directory to the path
sys.path.append(os.path.join(os.path.dirname(__file__), '..', 'src'))

from agents.data_loader_tools_agent import DataLoaderToolsAgent
from tools.pdf_processor import (
    extract_pdf_text,
    extract_pdf_tables,
    smart_extract_data_from_pdf,
    get_pdf_info
)

# Load environment variables
load_dotenv()

def create_sample_pdf_with_table():
    """Create a sample PDF with table data for testing."""
    print("📄 Creating sample PDF with table data...")
    
    # Create sample data
    sample_data = pd.DataFrame({
        'Product': ['Widget A', 'Widget B', 'Widget C', 'Widget D'],
        'Price': [10.99, 15.99, 8.99, 12.99],
        'Quantity': [100, 150, 200, 75],
        'Revenue': [1099.00, 2398.50, 1798.00, 974.25]
    })
    
    # Note: In a real scenario, you would create a PDF with table data
    # For this example, we'll create a CSV that we can then reference
    os.makedirs("test_data", exist_ok=True)
    sample_data.to_csv("test_data/sample_table_data.csv", index=False)
    
    print("✅ Sample data created (CSV format for testing)")
    return "test_data/sample_table_data.csv"

def demonstrate_pdf_info_extraction():
    """Demonstrate PDF information extraction."""
    print("\n📊 Demonstrating PDF Information Extraction")
    print("=" * 60)
    
    # Note: This would work with actual PDF files
    # For demonstration, we'll show what the function would return
    print("🔍 PDF Info Extraction Features:")
    print("• File metadata (size, pages, creation date)")
    print("• Content analysis (text vs tables)")
    print("• Extraction recommendations")
    print("• Structure detection")
    
    # Show what get_pdf_info would return
    mock_pdf_info = {
        "file_path": "sample_report.pdf",
        "file_size_mb": 2.5,
        "file_name": "sample_report.pdf",
        "total_pages": 15,
        "content_analysis": {
            "pages_with_text": 12,
            "pages_with_tables": 3,
            "likely_content_type": "data_document",
            "extraction_recommendation": "table_extraction"
        }
    }
    
    print("\n📋 Example PDF Info Result:")
    for key, value in mock_pdf_info.items():
        print(f"  {key}: {value}")

def demonstrate_pdf_text_extraction():
    """Demonstrate PDF text extraction capabilities."""
    print("\n📝 Demonstrating PDF Text Extraction")
    print("=" * 60)
    
    print("🔍 PDF Text Extraction Features:")
    print("• Multi-page text extraction")
    print("• Page-by-page processing")
    print("• Formatting preservation")
    print("• Error handling for corrupted files")
    
    # Show what extract_pdf_text would return
    mock_text_result = {
        "text": "Sample extracted text from PDF document...",
        "total_pages": 5,
        "pages_processed": 5,
        "text_length": 1250,
        "extraction_method": "pdfplumber",
        "has_text": True
    }
    
    print("\n📋 Example Text Extraction Result:")
    for key, value in mock_text_result.items():
        print(f"  {key}: {value}")

def demonstrate_pdf_table_extraction():
    """Demonstrate PDF table extraction capabilities."""
    print("\n📊 Demonstrating PDF Table Extraction")
    print("=" * 60)
    
    print("🔍 PDF Table Extraction Features:")
    print("• Automatic table detection")
    print("• Multiple extraction methods (tabula, pdfplumber)")
    print("• DataFrame conversion")
    print("• Quality assessment")
    
    # Show what extract_pdf_tables would return
    mock_table_result = {
        "tables": [
            {
                "table_id": 1,
                "page_number": 3,
                "shape": [4, 4],
                "columns": ["Product", "Price", "Quantity", "Revenue"],
                "extraction_method": "tabula"
            }
        ],
        "total_tables": 1,
        "success": True
    }
    
    print("\n📋 Example Table Extraction Result:")
    for key, value in mock_table_result.items():
        print(f"  {key}: {value}")

def demonstrate_smart_pdf_extraction():
    """Demonstrate smart PDF extraction capabilities."""
    print("\n🧠 Demonstrating Smart PDF Extraction")
    print("=" * 60)
    
    print("🔍 Smart PDF Extraction Features:")
    print("• Automatic method selection")
    print("• Quality scoring")
    print("• Fallback strategies")
    print("• Structured data detection")
    
    # Show what smart_extract_data_from_pdf would return
    mock_smart_result = {
        "extraction_strategy": "comprehensive",
        "structured_data": [
            {
                "table_id": 1,
                "data": {"Product": ["Widget A", "Widget B"], "Price": [10.99, 15.99]},
                "shape": [2, 2]
            }
        ],
        "text_content": "Additional text content...",
        "extraction_summary": {
            "tables_found": 1,
            "text_extracted": True,
            "quality_score": 0.9,
            "recommended_use": "data_analysis",
            "primary_data_type": "tabular"
        }
    }
    
    print("\n📋 Example Smart Extraction Result:")
    for key, value in mock_smart_result.items():
        if key == "structured_data":
            print(f"  {key}: {len(value)} tables found")
        elif key == "extraction_summary":
            print(f"  {key}:")
            for sub_key, sub_value in value.items():
                print(f"    {sub_key}: {sub_value}")
        else:
            print(f"  {key}: {value}")

def demonstrate_agent_integration():
    """Demonstrate integration with DataLoaderToolsAgent."""
    print("\n🤖 Demonstrating Agent Integration")
    print("=" * 60)
    
    # Check if we have API key
    api_key = os.getenv("OPENAI_API_KEY")
    if not api_key:
        print("⚠️ No OpenAI API key found. Showing mock demonstration.")
        
        print("\n🔍 Agent Integration Features:")
        print("• Natural language PDF processing instructions")
        print("• Automatic method selection")
        print("• Integration with existing data pipeline")
        print("• Error handling and fallbacks")
        
        print("\n📋 Example Agent Instructions:")
        example_instructions = [
            "Load data from the financial_report.pdf file",
            "Extract all tables from the quarterly_results.pdf document",
            "Get a summary of the content in the research_paper.pdf file",
            "Find and load the revenue data from the annual_report.pdf"
        ]
        
        for i, instruction in enumerate(example_instructions, 1):
            print(f"  {i}. {instruction}")
        
        return
    
    try:
        # Initialize the agent
        llm = ChatOpenAI(model="gpt-4o-mini", api_key=api_key)
        agent = DataLoaderToolsAgent(model=llm)
        
        print("✅ DataLoaderToolsAgent initialized with PDF support")
        
        # Test with a sample file (CSV for now, but shows the workflow)
        sample_file = create_sample_pdf_with_table()
        
        print(f"\n🧪 Testing with sample file: {sample_file}")
        
        # Test agent with file loading
        agent.invoke_agent(
            user_instructions=f"Load the data from {sample_file} and analyze its structure"
        )
        
        # Get results
        loaded_data = agent.get_artifacts(as_dataframe=True)
        if isinstance(loaded_data, pd.DataFrame):
            print(f"✅ Successfully loaded data: {loaded_data.shape[0]} rows × {loaded_data.shape[1]} columns")
            print(f"📊 Columns: {list(loaded_data.columns)}")
        else:
            print(f"⚠️ Data loading result: {loaded_data}")
        
        # Show tool calls
        tool_calls = agent.get_tool_calls()
        print(f"🔧 Tools used: {tool_calls}")
        
        # Show AI response
        ai_response = agent.get_ai_message()
        print(f"🤖 Agent response: {ai_response[:200]}...")
        
    except Exception as e:
        print(f"❌ Error during agent testing: {str(e)}")

def demonstrate_fetch_ai_integration():
    """Demonstrate integration with Fetch AI platform."""
    print("\n🌐 Demonstrating Fetch AI Platform Integration")
    print("=" * 60)
    
    print("🔍 Fetch AI Platform Benefits:")
    print("• PDF file upload directly to platform")
    print("• Automatic data extraction and processing")
    print("• Agent-to-agent communication with PDF data")
    print("• Integration with existing data science workflows")
    
    print("\n📋 Typical Fetch AI PDF Workflow:")
    workflow_steps = [
        "1. User uploads PDF file to Fetch AI chat platform",
        "2. DataLoaderToolsAgent receives PDF processing request",
        "3. Agent analyzes PDF structure and selects best extraction method",
        "4. Data is extracted and converted to structured format",
        "5. Processed data is passed to DataCleaningAgent for cleaning",
        "6. Final clean dataset is ready for analysis or visualization"
    ]
    
    for step in workflow_steps:
        print(f"  {step}")
    
    print("\n🎯 Key Advantages:")
    advantages = [
        "• Seamless integration with existing agent ecosystem",
        "• Automatic format detection and processing",
        "• Robust error handling and fallback strategies",
        "• Support for various PDF types (reports, forms, data sheets)",
        "• Maintains data quality and structure"
    ]
    
    for advantage in advantages:
        print(f"  {advantage}")

def main():
    """Main demonstration function."""
    print("🚀 PDF Data Loader Extension Demonstration")
    print("=" * 60)
    print("This example shows how the DataLoaderToolsAgent has been extended")
    print("to support PDF files for the Fetch AI platform.")
    
    # Run all demonstrations
    demonstrate_pdf_info_extraction()
    demonstrate_pdf_text_extraction()
    demonstrate_pdf_table_extraction()
    demonstrate_smart_pdf_extraction()
    demonstrate_agent_integration()
    demonstrate_fetch_ai_integration()
    
    print("\n🎉 Demonstration Complete!")
    print("\n📝 Next Steps:")
    print("1. Install PDF dependencies: pip install pdfplumber tabula-py PyPDF2")
    print("2. Test with actual PDF files containing tables")
    print("3. Deploy to Fetch AI platform for testing")
    print("4. Integrate with existing data science workflow")
    
    # Cleanup
    try:
        import shutil
        if os.path.exists("test_data"):
            shutil.rmtree("test_data")
        print("🧹 Cleaned up test files")
    except:
        pass

if __name__ == "__main__":
    main() 