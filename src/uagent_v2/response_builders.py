"""
Response building module for the enhanced uAgent implementation.

This module contains all response building functions extracted from the original
enhanced_uagent.py file for better code organization and maintainability.
"""

import logging
from typing import Dict, Any, List, Optional, Union

from .config import UAgentConfig
from .exceptions import handle_analysis_error

logger = logging.getLogger(__name__)


class ResponseBuilder:
    """Build various types of responses for the enhanced uAgent."""
    
    def __init__(self, config: UAgentConfig):
        self.config = config
    
    def create_no_data_response(self) -> str:
        """Create response when no cleaned data is available."""
        return """
🚫 **No Recent Data Found**

I don't have any recently processed data to deliver. Please first run a data cleaning task, for example:

"Clean and analyze https://raw.githubusercontent.com/datasciencedojo/datasets/master/titanic.csv"

Then I can provide your cleaned data in various formats.
"""
    
    def create_expired_session_response(self) -> str:
        """Create response when data session has expired."""
        return f"""
🕐 **Data Session Expired**

Your cleaned data session has expired (older than {self.config.session_timeout_hours} hour{'s' if self.config.session_timeout_hours > 1 else ''}). 
Please re-run your data cleaning task to get fresh results.
"""
    
    def create_analysis_error_response(self, error: Exception) -> str:
        """Create response for analysis errors."""
        error_response = handle_analysis_error(error, "analysis")
        
        return f"""
🚫 **Analysis Error**

Sorry, I encountered an issue: {str(error)}

**Common solutions:**
1. Include a direct CSV URL in your request (e.g., https://example.com/data.csv)
2. Be specific about what analysis you want
3. Example: "Clean and analyze https://raw.githubusercontent.com/datasciencedojo/datasets/master/titanic.csv for survival prediction"

**Need help?** Try: "Analyze https://raw.githubusercontent.com/mwaskom/seaborn-data/master/iris.csv for species classification"

**Technical details:**
{error_response}
"""
    
    def create_processing_started_response(self, query: str) -> str:
        """Create response when processing starts."""
        return f"""
🚀 **Processing Started**

Your request: "{query[:200]}{'...' if len(query) > 200 else ''}"

I'm now:
1. 🔍 Analyzing your request with AI
2. 📊 Loading and validating your data
3. 🧹 Cleaning and preparing your dataset
4. 🤖 Running the requested analysis
5. 📈 Generating comprehensive results

Please wait while I process your data...
"""
    
    def create_security_warning_response(self, issue: str) -> str:
        """Create response for security warnings."""
        return f"""
⚠️ **Security Warning**

{issue}

**For your security:**
- Only use trusted CSV URLs
- Avoid files with suspicious extensions
- Check file sizes before processing
- Be cautious with personal data

**Need help?** Try using a publicly available dataset like:
- Titanic: https://raw.githubusercontent.com/datasciencedojo/datasets/master/titanic.csv
- Iris: https://raw.githubusercontent.com/mwaskom/seaborn-data/master/iris.csv
"""
    
    def create_rate_limit_response(self) -> str:
        """Create response when rate limit is exceeded."""
        return f"""
🚫 **Rate Limit Exceeded**

You've exceeded the rate limit for data processing. Please wait {self.config.rate_limit_cooldown_seconds} seconds before making another request.

**Rate limits are in place to:**
- Ensure fair usage for all users
- Prevent system overload
- Maintain service quality

**Try again in a few moments!**
"""
    
    def create_file_too_large_response(self, file_size_mb: float) -> str:
        """Create response when file is too large."""
        return f"""
📁 **File Too Large**

Your file is {file_size_mb:.1f} MB, which exceeds the maximum allowed size of {self.config.max_file_size_mb} MB.

**Suggestions:**
1. Use a smaller sample of your data
2. Filter your data before processing
3. Split large files into smaller chunks
4. Use data compression techniques

**For large datasets, consider:**
- Sampling techniques to reduce size
- Using cloud-based data processing services
- Local processing with appropriate tools
"""
    
    def create_unsupported_format_response(self, file_format: str) -> str:
        """Create response for unsupported file formats."""
        return f"""
📄 **Unsupported Format**

The file format "{file_format}" is not supported. 

**Supported formats:**
- CSV (.csv) - Comma-separated values
- TSV (.tsv) - Tab-separated values
- Excel (.xlsx, .xls) - Microsoft Excel files
- JSON (.json) - JavaScript Object Notation

**To convert your file:**
1. Open in Excel or similar software
2. Save as CSV format
3. Use the CSV URL in your request

**Need help?** Try: "How do I convert {file_format} to CSV format?"
"""
    
    def create_network_error_response(self, url: str) -> str:
        """Create response for network errors."""
        return f"""
🌐 **Network Error**

Unable to access the URL: {url}

**Common causes:**
- URL is not accessible or doesn't exist
- Network connectivity issues
- Server is temporarily down
- URL requires authentication

**Troubleshooting:**
1. Check that the URL is correct
2. Try accessing the URL in your browser
3. Use a different data source
4. Check your internet connection

**Example working URLs:**
- https://raw.githubusercontent.com/datasciencedojo/datasets/master/titanic.csv
- https://raw.githubusercontent.com/mwaskom/seaborn-data/master/iris.csv
"""
    
    def create_success_summary_response(self, summary: Dict[str, Any]) -> str:
        """Create success summary response."""
        return f"""
✅ **Processing Complete**

**Summary:**
- Dataset: {summary.get('dataset_name', 'Unknown')}
- Rows processed: {summary.get('rows_processed', 0):,}
- Columns processed: {summary.get('columns_processed', 0)}
- Processing time: {summary.get('processing_time_seconds', 0):.2f} seconds
- Data quality: {summary.get('data_quality_score', 0):.2f}/1.0

**What's ready:**
- ✅ Cleaned dataset
- ✅ Analysis results
- ✅ Download links
- ✅ Insights and recommendations

**Next steps:**
Ask for your cleaned data: "Send me my cleaned dataset"
"""
    
    def create_help_response(self) -> str:
        """Create help response with usage instructions."""
        return """
🤖 **AI Data Science Agent - Help Guide**

**What I can do:**
- 🧹 Clean and prepare your data
- 🔧 Engineer features for better analysis
- 🤖 Build machine learning models
- 📊 Generate insights and visualizations
- 📁 Provide downloadable results

**How to use me:**
1. **Basic Analysis:** "Analyze https://your-data-url.csv"
2. **Specific Task:** "Clean and analyze https://data.csv for prediction"
3. **ML Focus:** "Build ML model using https://data.csv to predict target_column"

**Example requests:**
- "Clean and analyze https://raw.githubusercontent.com/datasciencedojo/datasets/master/titanic.csv for survival prediction"
- "Perform feature engineering on https://raw.githubusercontent.com/mwaskom/seaborn-data/master/iris.csv"
- "Build classification model for https://your-data.csv"

**Follow-up requests:**
- "Send me my cleaned data"
- "Provide my data in 5 chunks"
- "Show me rows 1-100 of my data"
- "Send me columns 1-5"

**Tips:**
- Use direct CSV URLs for best results
- Be specific about what you want to analyze
- Ask for help if you need guidance
- Request your data in chunks for large datasets

**Need more help?** Ask: "What can you do with my data?"
"""


class ErrorResponseBuilder:
    """Build error responses with proper formatting."""
    
    def __init__(self, config: UAgentConfig):
        self.config = config
    
    def build_generic_error_response(self, error: Exception, context: str = "processing") -> str:
        """Build a generic error response."""
        return f"""
❌ **Error During {context.title()}**

An error occurred: {str(error)}

**What you can try:**
1. Check if your data URL is accessible
2. Verify the file format is supported (CSV, Excel, JSON)
3. Try a different dataset
4. Ask for help with specific error details

**Need assistance?** Provide the error details and I'll help troubleshoot.
"""
    
    def build_validation_error_response(self, validation_errors: List[str]) -> str:
        """Build response for validation errors."""
        return f"""
⚠️ **Validation Errors**

Your request has the following issues:

{chr(10).join(f"• {error}" for error in validation_errors)}

**To fix these issues:**
1. Check your data URL format
2. Verify file accessibility
3. Ensure proper file format
4. Review your request for clarity

**Need help?** Ask: "How should I format my request?"
"""
    
    def build_timeout_error_response(self, timeout_seconds: int) -> str:
        """Build response for timeout errors."""
        return f"""
⏱️ **Processing Timeout**

Your request took longer than {timeout_seconds} seconds to process.

**This might happen when:**
- Dataset is very large
- Complex analysis is requested
- Network is slow
- Server is under heavy load

**What you can try:**
1. Use a smaller dataset
2. Simplify your analysis request
3. Try again in a few minutes
4. Split large requests into smaller parts

**For large datasets:** Consider using data sampling or chunking techniques.
""" 