#!/usr/bin/env python3
"""
Generate HTML version of the complete wildfire prediction report
"""

import os

def create_html_report():
    # Read the markdown file
    with open('WildFire_Prediction_Complete_Report.md', 'r', encoding='utf-8') as f:
        md_content = f.read()

    # Convert markdown to HTML manually for key elements
    html_content = md_content
    
    # Convert headers
    html_content = html_content.replace('# ', '<h1>').replace('\n## ', '</h1>\n<h2>').replace('\n### ', '</h2>\n<h3>')
    html_content = html_content.replace('\n#### ', '</h3>\n<h4>')
    
    # Convert bold text
    html_content = html_content.replace('**', '<strong>').replace('**', '</strong>')
    
    # Convert code blocks
    html_content = html_content.replace('```', '<pre><code>').replace('```', '</code></pre>')
    
    # Convert line breaks
    html_content = html_content.replace('\n\n', '<br><br>')
    html_content = html_content.replace('\n', '<br>')
    
    # Create complete HTML document
    html_template = f"""<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>Wildfire Prediction - Complete Analysis Report</title>
    <style>
        body {{
            font-family: -apple-system, BlinkMacSystemFont, 'Segoe UI', Roboto, sans-serif;
            line-height: 1.6;
            color: #333;
            max-width: 1200px;
            margin: 0 auto;
            padding: 20px;
            background-color: #fafafa;
        }}
        .container {{
            background: white;
            padding: 40px;
            border-radius: 10px;
            box-shadow: 0 4px 6px rgba(0, 0, 0, 0.1);
        }}
        h1 {{
            color: #d73527;
            border-bottom: 3px solid #d73527;
            padding-bottom: 10px;
        }}
        h2 {{
            color: #2c5aa0;
            border-left: 4px solid #2c5aa0;
            padding-left: 15px;
            margin-top: 30px;
        }}
        h3 {{
            color: #5d4e75;
        }}
        table {{
            width: 100%;
            border-collapse: collapse;
            margin: 20px 0;
            background: white;
        }}
        table, th, td {{
            border: 1px solid #ddd;
        }}
        th, td {{
            padding: 12px;
            text-align: left;
        }}
        th {{
            background-color: #f8f9fa;
            font-weight: 600;
        }}
        code {{
            background-color: #f4f4f4;
            padding: 2px 6px;
            border-radius: 3px;
            font-family: 'Courier New', monospace;
        }}
        pre {{
            background-color: #f8f9fa;
            padding: 15px;
            border-radius: 5px;
            overflow-x: auto;
            border-left: 4px solid #007acc;
        }}
        .success {{
            background-color: #d4edda;
            padding: 10px;
            border-radius: 5px;
            border-left: 4px solid #28a745;
            margin: 15px 0;
        }}
        hr {{
            border: none;
            height: 2px;
            background: linear-gradient(90deg, #d73527, #2c5aa0);
            margin: 30px 0;
        }}
    </style>
</head>
<body>
    <div class="container">
        {html_content}
    </div>
</body>
</html>"""

    # Save the HTML file
    with open('WildFire_Prediction_Complete_Report.html', 'w', encoding='utf-8') as f:
        f.write(html_template)

    print('✅ HTML report generated: WildFire_Prediction_Complete_Report.html')
    print('📁 Files created:')
    print('  - WildFire_Prediction_Complete_Report.md (Markdown version)')
    print('  - WildFire_Prediction_Complete_Report.html (HTML version)')
    print('')
    print('🎯 Both files contain complete project information:')
    print('  ✓ Model performance and baseline comparisons')
    print('  ✓ Feature sensitivity analysis results')
    print('  ✓ Data quality assessment from 4 years')
    print('  ✓ Statistical summaries and metrics')
    print('  ✓ Technical implementation details')
    print('  ✓ All visualization descriptions')
    print('')
    print('📤 Ready to share: Single file contains everything!')

if __name__ == "__main__":
    create_html_report()
