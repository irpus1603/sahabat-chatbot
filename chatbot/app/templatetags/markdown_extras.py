from django import template
import markdown
import re

register = template.Library()

@register.filter
def markdown_to_html(value):
    """Convert markdown to HTML"""
    if not value:
        return value
    
    # Configure markdown with extensions
    md = markdown.Markdown(
        extensions=['codehilite', 'fenced_code', 'tables'],
        extension_configs={
            'codehilite': {
                'css_class': 'highlight',
                'use_pygments': False,  # Use highlight.js instead
            }
        }
    )
    
    # Convert markdown to HTML
    html = md.convert(value)
    
    # Add highlight.js classes to code blocks
    html = re.sub(r'<pre><code>', r'<pre><code class="hljs">', html)
    
    return html