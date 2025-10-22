/* RAG Admin Custom JavaScript */

// Initialize RAG Admin functionality
document.addEventListener('DOMContentLoaded', function() {
    console.log('RAG Admin JavaScript loaded');
    
    // Initialize status indicators
    updateStatusIndicators();
    
    // Initialize tooltips
    initializeTooltips();
    
    // Initialize action buttons
    initializeActionButtons();
    
    // Initialize stats
    updateStats();
});

// Update status indicators with proper styling
function updateStatusIndicators() {
    const statusElements = document.querySelectorAll('[data-status]');
    
    statusElements.forEach(element => {
        const status = element.getAttribute('data-status');
        element.classList.add(`status-${status}`);
    });
}

// Initialize tooltips
function initializeTooltips() {
    const tooltipElements = document.querySelectorAll('[title]');
    
    tooltipElements.forEach(element => {
        const title = element.getAttribute('title');
        if (title) {
            element.classList.add('rag-tooltip');
            element.setAttribute('data-tooltip', title);
        }
    });
}

// Initialize action buttons
function initializeActionButtons() {
    // Add click handlers for action buttons
    document.querySelectorAll('.rag-action-button').forEach(button => {
        button.addEventListener('click', function(e) {
            const action = this.getAttribute('data-action');
            const itemId = this.getAttribute('data-item-id');
            
            if (action && itemId) {
                handleAction(action, itemId, this);
            }
        });
    });
}

// Handle various actions
function handleAction(action, itemId, button) {
    switch(action) {
        case 'reprocess':
            reprocessDocument(itemId, button);
            break;
        case 'delete':
            deleteItem(itemId, button);
            break;
        case 'view-chunks':
            viewChunks(itemId);
            break;
        default:
            console.log('Unknown action:', action);
    }
}

// Reprocess document
function reprocessDocument(documentId, button) {
    if (!confirm('Are you sure you want to reprocess this document?')) {
        return;
    }
    
    const originalText = button.textContent;
    button.textContent = 'Processing...';
    button.disabled = true;
    
    fetch(`/rag/documents/${documentId}/reprocess/`, {
        method: 'POST',
        headers: {
            'X-CSRFToken': getCookie('csrftoken'),
            'Content-Type': 'application/json',
        }
    })
    .then(response => response.json())
    .then(data => {
        if (data.success) {
            showNotification('Document reprocessing started successfully', 'success');
            setTimeout(() => {
                location.reload();
            }, 2000);
        } else {
            showNotification('Error: ' + data.error, 'error');
        }
    })
    .catch(error => {
        console.error('Error:', error);
        showNotification('Error reprocessing document', 'error');
    })
    .finally(() => {
        button.textContent = originalText;
        button.disabled = false;
    });
}

// Delete item
function deleteItem(itemId, button) {
    if (!confirm('Are you sure you want to delete this item? This action cannot be undone.')) {
        return;
    }
    
    const originalText = button.textContent;
    button.textContent = 'Deleting...';
    button.disabled = true;
    
    fetch(`/rag/documents/${itemId}/delete/`, {
        method: 'POST',
        headers: {
            'X-CSRFToken': getCookie('csrftoken'),
            'Content-Type': 'application/json',
        }
    })
    .then(response => response.json())
    .then(data => {
        if (data.success) {
            showNotification('Item deleted successfully', 'success');
            setTimeout(() => {
                location.reload();
            }, 1000);
        } else {
            showNotification('Error: ' + data.error, 'error');
        }
    })
    .catch(error => {
        console.error('Error:', error);
        showNotification('Error deleting item', 'error');
    })
    .finally(() => {
        button.textContent = originalText;
        button.disabled = false;
    });
}

// View chunks
function viewChunks(documentId) {
    const chunksUrl = `/admin/chat_rag/ragdocumentchunk/?document__id=${documentId}`;
    window.open(chunksUrl, '_blank');
}

// Show notification
function showNotification(message, type = 'info') {
    const notification = document.createElement('div');
    notification.className = `rag-notification rag-notification-${type}`;
    notification.textContent = message;
    
    // Style the notification
    notification.style.cssText = `
        position: fixed;
        top: 20px;
        right: 20px;
        padding: 15px 20px;
        border-radius: 5px;
        color: white;
        font-weight: bold;
        z-index: 9999;
        animation: slideIn 0.3s ease;
    `;
    
    // Set background color based on type
    switch(type) {
        case 'success':
            notification.style.backgroundColor = '#28a745';
            break;
        case 'error':
            notification.style.backgroundColor = '#dc3545';
            break;
        case 'warning':
            notification.style.backgroundColor = '#ffc107';
            notification.style.color = '#333';
            break;
        default:
            notification.style.backgroundColor = '#007cba';
    }
    
    document.body.appendChild(notification);
    
    // Remove notification after 5 seconds
    setTimeout(() => {
        notification.style.animation = 'slideOut 0.3s ease';
        setTimeout(() => {
            document.body.removeChild(notification);
        }, 300);
    }, 5000);
}

// Update stats
function updateStats() {
    const statsElements = document.querySelectorAll('.rag-stat-number');
    
    statsElements.forEach(element => {
        const value = parseInt(element.textContent);
        if (!isNaN(value)) {
            animateNumber(element, 0, value, 1000);
        }
    });
}

// Animate number counting
function animateNumber(element, start, end, duration) {
    const startTime = performance.now();
    
    function updateNumber(currentTime) {
        const elapsed = currentTime - startTime;
        const progress = Math.min(elapsed / duration, 1);
        
        const currentValue = Math.floor(progress * (end - start) + start);
        element.textContent = currentValue;
        
        if (progress < 1) {
            requestAnimationFrame(updateNumber);
        }
    }
    
    requestAnimationFrame(updateNumber);
}

// Get CSRF token
function getCookie(name) {
    let cookieValue = null;
    if (document.cookie && document.cookie !== '') {
        const cookies = document.cookie.split(';');
        for (let i = 0; i < cookies.length; i++) {
            const cookie = cookies[i].trim();
            if (cookie.substring(0, name.length + 1) === (name + '=')) {
                cookieValue = decodeURIComponent(cookie.substring(name.length + 1));
                break;
            }
        }
    }
    return cookieValue;
}

// Add CSS animations
const style = document.createElement('style');
style.textContent = `
    @keyframes slideIn {
        from {
            transform: translateX(100%);
            opacity: 0;
        }
        to {
            transform: translateX(0);
            opacity: 1;
        }
    }
    
    @keyframes slideOut {
        from {
            transform: translateX(0);
            opacity: 1;
        }
        to {
            transform: translateX(100%);
            opacity: 0;
        }
    }
`;
document.head.appendChild(style);

// Enhanced search functionality
function initializeSearch() {
    const searchInputs = document.querySelectorAll('input[name="q"]');
    
    searchInputs.forEach(input => {
        input.addEventListener('input', function(e) {
            const query = e.target.value.toLowerCase();
            const rows = document.querySelectorAll('.results tbody tr');
            
            rows.forEach(row => {
                const text = row.textContent.toLowerCase();
                if (text.includes(query)) {
                    row.style.display = '';
                } else {
                    row.style.display = 'none';
                }
            });
        });
    });
}

// Initialize search after DOM is loaded
document.addEventListener('DOMContentLoaded', initializeSearch);

// Auto-refresh functionality for processing status
function startAutoRefresh() {
    const processingElements = document.querySelectorAll('.status-processing');
    
    if (processingElements.length > 0) {
        setInterval(() => {
            fetch(window.location.href)
                .then(response => response.text())
                .then(html => {
                    const parser = new DOMParser();
                    const doc = parser.parseFromString(html, 'text/html');
                    const newProcessingElements = doc.querySelectorAll('.status-processing');
                    
                    if (newProcessingElements.length !== processingElements.length) {
                        location.reload();
                    }
                });
        }, 10000); // Check every 10 seconds
    }
}

// Start auto-refresh
document.addEventListener('DOMContentLoaded', startAutoRefresh);

// Export functions for global use
window.reprocessDocument = reprocessDocument;
window.deleteItem = deleteItem;
window.viewChunks = viewChunks;
window.showNotification = showNotification;