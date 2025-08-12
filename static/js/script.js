// DOM Elements
const uploadArea = document.getElementById('uploadArea');
const fileInput = document.getElementById('fileInput');
const uploadBtn = document.getElementById('uploadBtn');
const urlInput = document.getElementById('urlInput');
const loadUrlBtn = document.getElementById('loadUrlBtn');
const classifyBtn = document.getElementById('classifyBtn');
const previewContainer = document.getElementById('previewContainer');
const resultsContainer = document.getElementById('resultsContainer');
const generateReportBtn = document.getElementById('generateReportBtn');
const reportDisplay = document.getElementById('reportDisplay');
const reportContent = document.getElementById('reportContent');
const downloadReportBtn = document.getElementById('downloadReportBtn');
const printReportBtn = document.getElementById('printReportBtn');
const sampleItems = document.querySelectorAll('.sample-item');

// Global variables
let currentImageFile = null;
let currentImageUrl = null;
let lastAnalysisResults = null;
let currentReportData = null;

// Initialize event listeners
document.addEventListener('DOMContentLoaded', function() {
    initializeEventListeners();
    checkServerHealth();
});

function initializeEventListeners() {
    // File upload listeners
    uploadArea.addEventListener('click', () => fileInput.click());
    uploadArea.addEventListener('dragover', handleDragOver);
    uploadArea.addEventListener('dragleave', handleDragLeave);
    uploadArea.addEventListener('drop', handleDrop);
    fileInput.addEventListener('change', handleFileSelect);
    
    // URL input listeners
    loadUrlBtn.addEventListener('click', handleUrlLoad);
    urlInput.addEventListener('keypress', (e) => {
        if (e.key === 'Enter') handleUrlLoad();
    });
    
    // Classify button listener
    classifyBtn.addEventListener('click', handleClassification);
    
    // Report generation listeners
    generateReportBtn.addEventListener('click', handleReportGeneration);
    downloadReportBtn.addEventListener('click', handleReportDownload);
    printReportBtn.addEventListener('click', handleReportPrint);
    
    // Sample images listeners
    sampleItems.forEach(item => {
        item.addEventListener('click', () => {
            const sampleType = item.getAttribute('data-sample');
            const imgSrc = item.querySelector('img').src;
            loadSampleImage(imgSrc, sampleType);
        });
    });
}

// Check server health on load
async function checkServerHealth() {
    try {
        const response = await fetch('/api/health');
        const health = await response.json();
        console.log('Server health:', health);
        
        if (!health.models_loaded.knot_model || 
            !health.models_loaded.leaf_model || 
            !health.models_loaded.disease_model) {
            showInfo('Running in demo mode - some models not loaded');
        }
        
        if (!health.openrouter_configured) {
            showInfo('AI report generation may be limited - OpenRouter not configured');
        }
    } catch (error) {
        console.warn('Could not check server health:', error);
    }
}

// File upload handlers
function handleDragOver(e) {
    e.preventDefault();
    uploadArea.classList.add('dragover');
}

function handleDragLeave(e) {
    e.preventDefault();
    uploadArea.classList.remove('dragover');
}

function handleDrop(e) {
    e.preventDefault();
    uploadArea.classList.remove('dragover');
    const files = e.dataTransfer.files;
    if (files.length > 0) {
        handleFile(files[0]);
    }
}

function handleFileSelect(e) {
    const file = e.target.files[0];
    if (file) {
        handleFile(file);
    }
}

function handleFile(file) {
    if (!file.type.startsWith('image/')) {
        showError('Please select a valid image file');
        return;
    }
    
    if (file.size > 16 * 1024 * 1024) {
        showError('File size must be less than 16MB');
        return;
    }
    
    currentImageFile = file;
    currentImageUrl = null;
    displayImagePreview(file);
    enableClassifyButton();
    clearResults();
}

// URL loading handler
function handleUrlLoad() {
    const url = urlInput.value.trim();
    if (!url) {
        showError('Please enter a valid image URL');
        return;
    }
    
    if (!isValidImageUrl(url)) {
        showError('Please enter a valid image URL');
        return;
    }
    
    currentImageUrl = url;
    currentImageFile = null;
    displayImagePreview(url);
    enableClassifyButton();
    clearResults();
}

// Sample image handler
function loadSampleImage(imgSrc, sampleType) {
    currentImageUrl = imgSrc;
    currentImageFile = null;
    displayImagePreview(imgSrc);
    enableClassifyButton();
    clearResults();
    
    // Optional: Auto-classify sample images for demo
    setTimeout(() => {
        handleClassification();
    }, 800);
}

// Image preview display
function displayImagePreview(source) {
    const img = document.createElement('img');
    img.className = 'preview-image';
    img.alt = 'Preview image';
    
    if (typeof source === 'string') {
        // URL
        img.src = source;
    } else {
        // File
        const reader = new FileReader();
        reader.onload = (e) => {
            img.src = e.target.result;
        };
        reader.readAsDataURL(source);
    }
    
    img.onload = () => {
        previewContainer.innerHTML = '';
        previewContainer.appendChild(img);
    };
    
    img.onerror = () => {
        showError('Failed to load image');
        resetPreview();
    };
}

// Classification handler
async function handleClassification() {
    if (!currentImageFile && !currentImageUrl) {
        showError('Please select an image first');
        return;
    }
    
    setClassifyButtonLoading(true);
    clearResults();
    hideReportDisplay();
    
    try {
        const result = await classifyImage();
        lastAnalysisResults = result;
        displayResults(result);
    } catch (error) {
        console.error('Classification error:', error);
        showError('Failed to classify image. Please try again.');
    } finally {
        setClassifyButtonLoading(false);
    }
}

// Main classification function
async function classifyImage() {
    const formData = new FormData();
    
    if (currentImageFile) {
        formData.append('image', currentImageFile);
    } else if (currentImageUrl) {
        formData.append('image_url', currentImageUrl);
    }
    
    // Backend API call
    const response = await fetch('/api/classify', {
        method: 'POST',
        body: formData
    });
    
    if (!response.ok) {
        const errorData = await response.json().catch(() => ({}));
        throw new Error(errorData.message || `HTTP error! status: ${response.status}`);
    }
    
    const result = await response.json();
    return result;
}

// Display classification results
function displayResults(result) {
    console.log('Displaying results:', result); // Debug log
    
    const predictedClass = document.getElementById('predictedClass');
    const confidence = document.getElementById('confidence');
    const healthStatus = document.getElementById('healthStatus');
    
    // Clear any existing detailed results first
    const existingDetails = document.getElementById('detailedResults');
    const existingGallery = document.getElementById('imageGallery');
    if (existingDetails) existingDetails.remove();
    if (existingGallery) existingGallery.remove();
    
    // Display primary prediction
    predictedClass.textContent = result.predicted_class || 'Unknown';
    confidence.textContent = result.confidence ? `${(result.confidence * 100).toFixed(1)}%` : 'N/A';
    
    // Handle analysis results (the main detection data)
    if (result.analysis) {
        const analysis = result.analysis;
        console.log('Analysis data:', analysis); // Debug log
        
        // Update health status
        let statusText = analysis.health_status || 'Unknown';
        let statusClass = 'unknown';
        
        if (statusText === 'Healthy') {
            statusClass = 'healthy';
        } else if (statusText.includes('Disease') || statusText === 'Diseased') {
            statusClass = 'diseased';
        } else if (statusText.includes('Mostly')) {
            statusClass = 'mostly-healthy';
        }
        
        healthStatus.textContent = statusText;
        healthStatus.className = `result-value health-status ${statusClass}`;
        
        // Add detailed analysis information
        addDetailedResults(analysis);
        
        // Display processed images if available
        if (result.processed_images) {
            displayProcessedImages(result.processed_images);
        }
        
        // Update main classification display with meaningful data
        if (analysis.knot_count > 0 || analysis.leaf_count > 0) {
            const mainDescription = `${analysis.leaf_count} leaves, ${analysis.knot_count} knots detected`;
            predictedClass.textContent = mainDescription;
        }
        
    } else {
        // Fallback for simple classification
        const isHealthy = result.predicted_class?.toLowerCase().includes('healthy');
        const statusText = isHealthy ? 'Healthy' : 'Disease Detected';
        const statusClass = isHealthy ? 'healthy' : 'diseased';
        
        healthStatus.textContent = statusText;
        healthStatus.className = `result-value health-status ${statusClass}`;
    }
    
    // Show demo mode indicator if applicable
    if (result.demo_mode) {
        addDemoModeIndicator();
    }
    
    // Show results container
    resultsContainer.style.display = 'block';
    
    // Scroll to results
    resultsContainer.scrollIntoView({ behavior: 'smooth' });
}

// Fixed report generation handler
async function handleReportGeneration() {
    if (!lastAnalysisResults || !lastAnalysisResults.analysis) {
        showError('Please classify an image first to generate a report');
        return;
    }
    
    setGenerateReportButtonLoading(true);
    
    try {
        // Prepare the data for report generation
        const reportData = {
            analysis: lastAnalysisResults.analysis,
            // Also include the main prediction data for context
            predicted_class: lastAnalysisResults.predicted_class,
            confidence: lastAnalysisResults.confidence,
            demo_mode: lastAnalysisResults.demo_mode || false,
            timestamp: new Date().toISOString()
        };
        
        console.log('Sending report data:', reportData); // Debug log
        
        const reportResult = await generateReport(reportData);
        currentReportData = reportResult.report;
        displayReport(reportResult.report);
        showSuccess('🎉 Professional report generated successfully!');
    } catch (error) {
        console.error('Report generation error:', error);
        showError('Failed to generate report. Please try again or check your connection.');
    } finally {
        setGenerateReportButtonLoading(false);
    }
}

// Fixed main report generation function
async function generateReport(analysisResults) {
    console.log('Generating report for:', analysisResults); // Debug log
    
    const response = await fetch('/api/generate-report', {
        method: 'POST',
        headers: {
            'Content-Type': 'application/json'
        },
        body: JSON.stringify(analysisResults)
    });
    
    if (!response.ok) {
        const errorData = await response.json().catch(() => ({}));
        throw new Error(errorData.message || `Report generation failed: ${response.status}`);
    }
    
    const result = await response.json();
    console.log('Report generation result:', result);
    return result;
}

// Enhanced report formatting with better structure
function formatReportForDisplay(reportData) {
    const timestamp = new Date(reportData.timestamp).toLocaleString();
    const isAI = !reportData.is_fallback;
    
    // Safely get analysis summary data
    const analysisSummary = reportData.analysis_summary || {};
    const leafCount = analysisSummary.leaf_count || 0;
    const knotCount = analysisSummary.knot_count || 0;
    const healthStatus = analysisSummary.health_status || 'Unknown';
    const confidenceAvg = analysisSummary.confidence_avg || 0;
    const diseaseBreakdown = analysisSummary.disease_summary || {};
    
    return `
        <div class="report-metadata">
            <div class="metadata-item">
                <strong>📋 Report ID:</strong> 
                <span>${reportData.report_id}</span>
            </div>
            <div class="metadata-item">
                <strong>📅 Generated:</strong> 
                <span>${timestamp}</span>
            </div>
            <div class="metadata-item">
                <strong>🤖 Analysis Type:</strong> 
                <span>${isAI ? '🤖 AI-Generated Professional Report' : '📋 Template-Based Report'}</span>
            </div>
            <div class="metadata-item">
                <strong>🔬 Model Used:</strong> 
                <span>${reportData.model_used || 'Standard Analysis'}</span>
            </div>
        </div>
        
        <div class="report-body">
            ${formatReportContent(reportData.ai_generated_report)}
        </div>
        
        <div class="analysis-summary">
            <h3>📊 Detailed Analysis Summary</h3>
            <div class="summary-grid">
                <div class="summary-item">
                    <span class="summary-label">🍃 Leaves Analyzed:</span>
                    <span class="summary-value">${leafCount}</span>
                </div>
                <div class="summary-item">
                    <span class="summary-label">🔴 Knots Detected:</span>
                    <span class="summary-value">${knotCount}</span>
                </div>
                <div class="summary-item">
                    <span class="summary-label">🏥 Health Status:</span>
                    <span class="summary-value ${getHealthStatusClass(healthStatus)}">${healthStatus}</span>
                </div>
                <div class="summary-item">
                    <span class="summary-label">🎯 Confidence:</span>
                    <span class="summary-value">${(confidenceAvg * 100).toFixed(1)}%</span>
                </div>
            </div>
            
            ${Object.keys(diseaseBreakdown).length > 0 ? `
                <div class="disease-distribution">
                    <h4>🦠 Disease Distribution</h4>
                    <div class="disease-chart">
                        ${Object.entries(diseaseBreakdown).map(([disease, count]) => {
                            const percentage = leafCount > 0 ? ((count / leafCount) * 100).toFixed(1) : 0;
                            const isHealthy = disease.toLowerCase() === 'healthy';
                            return `
                                <div class="disease-stat ${isHealthy ? 'healthy' : 'diseased'}">
                                    <span class="disease-name">${disease}</span>
                                    <span class="disease-data">
                                        <span class="count">${count} leaves</span>
                                        <span class="percentage">(${percentage}%)</span>
                                    </span>
                                </div>
                            `;
                        }).join('')}
                    </div>
                </div>
            ` : ''}
        </div>
        
        <div class="report-footer">
            ${reportData.is_fallback ? 
                '<div class="fallback-notice">⚠️ This report was generated using a basic template due to AI service limitations. For more detailed analysis, please ensure the AI service is properly configured.</div>' : 
                '<div class="ai-notice">🤖 This comprehensive report was generated using advanced AI analysis based on computer vision detection results.</div>'
            }
        </div>
    `;
}

// Enhanced report content formatting
function formatReportContent(content) {
    if (!content) return '<p>No report content available.</p>';
    
    return content
        // Handle markdown-style formatting
        .replace(/\*\*(.*?)\*\*/g, '<strong>$1</strong>')
        .replace(/\*(.*?)\*/g, '<em>$1</em>')
        
        // Handle headers
        .replace(/^### (.*$)/gim, '<h3>$1</h3>')
        .replace(/^## (.*$)/gim, '<h2>$1</h2>')
        .replace(/^# (.*$)/gim, '<h1>$1</h1>')
        
        // Handle lists
        .replace(/^[-•] (.*$)/gim, '<li>$1</li>')
        .replace(/^(\d+)\. (.*$)/gim, '<li>$2</li>')
        
        // Wrap consecutive list items in ul tags
        .replace(/((<li>.*<\/li>\s*)+)/g, '<ul>$1</ul>')
        
        // Handle paragraphs
        .split('\n\n')
        .map(paragraph => {
            paragraph = paragraph.trim();
            if (!paragraph) return '';
            if (paragraph.startsWith('<h') || paragraph.startsWith('<ul') || paragraph.startsWith('<li')) {
                return paragraph;
            }
            return `<p>${paragraph}</p>`;
        })
        .join('')
        
        // Clean up empty paragraphs and fix nested elements
        .replace(/<p><\/p>/g, '')
        .replace(/<p>(<h[1-6]>)/g, '$1')
        .replace(/(<\/h[1-6]>)<\/p>/g, '$1')
        .replace(/<p>(<ul>)/g, '$1')
        .replace(/(<\/ul>)<\/p>/g, '$1');
}

// Get health status CSS class
function getHealthStatusClass(status) {
    if (!status) return 'unknown';
    if (status.toLowerCase().includes('healthy')) return 'healthy';
    if (status.toLowerCase().includes('disease')) return 'diseased';
    return 'unknown';
}

// Report download handler
async function handleReportDownload() {
    if (!currentReportData) {
        showError('No report to download');
        return;
    }
    
    try {
        const response = await fetch(`/api/download-report/${currentReportData.report_id}`);
        if (!response.ok) {
            throw new Error('Download failed');
        }
        
        const blob = await response.blob();
        const url = window.URL.createObjectURL(blob);
        const a = document.createElement('a');
        a.href = url;
        a.download = `olive_tree_report_${currentReportData.report_id}.json`;
        document.body.appendChild(a);
        a.click();
        window.URL.revokeObjectURL(url);
        document.body.removeChild(a);
        
        showSuccess('Report downloaded successfully!');
    } catch (error) {
        console.error('Download error:', error);
        showError('Failed to download report');
    }
}

// Report print handler
function handleReportPrint() {
    if (!currentReportData) {
        showError('No report to print');
        return;
    }
    
    const printWindow = window.open('', '_blank');
    const printContent = `
        <!DOCTYPE html>
        <html>
        <head>
            <title>Olive Tree Health Report - ${currentReportData.report_id}</title>
            <style>
                body { font-family: Arial, sans-serif; margin: 40px; line-height: 1.6; }
                h1, h2, h3 { color: #2c5e3f; }
                .metadata-item { margin-bottom: 10px; }
                .summary-grid { display: grid; grid-template-columns: repeat(2, 1fr); gap: 20px; margin: 20px 0; }
                .summary-item { padding: 10px; background: #f5f5f5; border-radius: 5px; }
                .summary-label { font-weight: bold; }
                .healthy { color: #4caf50; }
                .diseased { color: #f44336; }
                .unknown { color: #ff9800; }
                @media print { body { margin: 20px; } }
            </style>
        </head>
        <body>
            ${reportContent.innerHTML}
        </body>
        </html>
    `;
    
    printWindow.document.write(printContent);
    printWindow.document.close();
    printWindow.focus();
    printWindow.print();
}

// Add detailed analysis results
function addDetailedResults(analysis) {
    // Create detailed results container
    let detailedResults = document.createElement('div');
    detailedResults.id = 'detailedResults';
    detailedResults.className = 'detailed-results';
    
    // Create detailed analysis HTML
    let detailedHTML = '<h4>🔍 Detailed Analysis</h4>';
    
    // Detection counts with better formatting
    detailedHTML += `
        <div class="analysis-section">
            <h5>📊 Detection Summary</h5>
            <div class="detection-stats">
                <div class="stat-item">
                    <div class="stat-number">${analysis.leaf_count || 0}</div>
                    <div class="stat-label">Leaves Detected</div>
                </div>
                <div class="stat-item">
                    <div class="stat-number">${analysis.knot_count || 0}</div>
                    <div class="stat-label">Knots Detected</div>
                </div>
                <div class="stat-item">
                    <div class="stat-number">${(analysis.confidence_avg * 100 || 0).toFixed(1)}%</div>
                    <div class="stat-label">Avg Confidence</div>
                </div>
            </div>
        </div>
    `;
    
    // Disease summary
    if (analysis.disease_summary && Object.keys(analysis.disease_summary).length > 0) {
        detailedHTML += `
        <div class="analysis-section">
            <h5>🦠 Disease Distribution</h5>
            <div class="disease-breakdown">
        `;
        
        const totalLeaves = analysis.leaf_count || 1;
        for (const [disease, count] of Object.entries(analysis.disease_summary)) {
            const percentage = (count / totalLeaves * 100).toFixed(1);
            const statusClass = disease.toLowerCase() === 'healthy' ? 'healthy' : 'diseased';
            
            detailedHTML += `
                <div class="disease-item ${statusClass}">
                    <span class="disease-name">${disease}</span>
                    <span class="disease-stats">
                        <span class="disease-count">${count} leaves</span>
                        <span class="disease-percentage">(${percentage}%)</span>
                    </span>
                </div>
            `;
        }
        
        detailedHTML += `</div></div>`;
    }
    
    // Individual leaf results
    if (analysis.diseases_detected && analysis.diseases_detected.length > 0) {
        detailedHTML += `
        <div class="analysis-section">
            <h5>🍃 Individual Leaf Analysis</h5>
            <div class="leaf-results">
        `;
        
        analysis.diseases_detected.slice(0, 10).forEach(leaf => { // Limit to first 10 for display
            const confidencePercent = (leaf.confidence * 100).toFixed(1);
            const statusClass = leaf.disease.toLowerCase() === 'healthy' ? 'healthy' : 'diseased';
            
            detailedHTML += `
                <div class="leaf-item ${statusClass}">
                    <span class="leaf-info">
                        <span class="leaf-id">Leaf ${leaf.leaf_id}</span>
                        <span class="leaf-disease">${leaf.disease}</span>
                    </span>
                    <span class="leaf-confidence">${confidencePercent}% confidence</span>
                </div>
            `;
        });
        
        if (analysis.diseases_detected.length > 10) {
            detailedHTML += `
                <div class="leaf-item more-items">
                    <span class="leaf-info">
                        <span class="leaf-id">+ ${analysis.diseases_detected.length - 10} more leaves...</span>
                    </span>
                </div>
            `;
        }
        
        detailedHTML += `</div></div>`;
    }
    
    detailedResults.innerHTML = detailedHTML;
    resultsContainer.appendChild(detailedResults);
}


// Display processed images
function displayProcessedImages(processedImages) {
    let imageGallery = document.createElement('div');
    imageGallery.id = 'imageGallery';
    imageGallery.className = 'image-gallery';
    
    let galleryHTML = '<h4>📸 Analysis Visualization</h4><div class="image-grid">';
    
    if (processedImages.knots) {
        galleryHTML += `
            <div class="gallery-item">
                <img src="${processedImages.knots}" alt="Knot Detection" class="result-image" onclick="openImageModal(this.src, 'Knot Detection Results')">
                <p class="image-caption">🔴 Knot Detection</p>
                <p class="image-description">Red boxes show detected olive knots</p>
            </div>
        `;
    }
    
    if (processedImages.combined) {
        galleryHTML += `
            <div class="gallery-item">
                <img src="${processedImages.combined}" alt="Combined Detection" class="result-image" onclick="openImageModal(this.src, 'Combined Detection Results')">
                <p class="image-caption">🟢🔴 Combined Detection</p>
                <p class="image-description">Red: knots, Green: leaves</p>
            </div>
        `;
    }
    
    if (processedImages.final) {
        galleryHTML += `
            <div class="gallery-item">
                <img src="${processedImages.final}" alt="Final Analysis" class="result-image" onclick="openImageModal(this.src, 'Disease Classification Results')">
                <p class="image-caption">🔬 Disease Classification</p>
                <p class="image-description">Final analysis with disease labels</p>
            </div>
        `;
    }
    
    galleryHTML += '</div>';
    imageGallery.innerHTML = galleryHTML;
    resultsContainer.appendChild(imageGallery);
}


// Add demo mode indicator
function addDemoModeIndicator() {
    const demoIndicator = document.createElement('div');
    demoIndicator.className = 'demo-indicator';
    demoIndicator.innerHTML = `
        <div style="background: linear-gradient(135deg, #ff9800, #f57c00); color: white; padding: 0.8rem 1rem; border-radius: 8px; margin-bottom: 1rem; text-align: center; font-weight: 600;">
            🧪 Demo Mode - Models not loaded, showing sample results
        </div>
    `;
    
    resultsContainer.insertBefore(demoIndicator, resultsContainer.firstChild.nextSibling);
}

// Open image in modal (simple implementation)
function openImageModal(src, title = 'Analysis Result') {
    const modal = document.createElement('div');
    modal.style.cssText = `
        position: fixed;
        top: 0;
        left: 0;
        width: 100%;
        height: 100%;
        background: rgba(0, 0, 0, 0.9);
        display: flex;
        flex-direction: column;
        align-items: center;
        justify-content: center;
        z-index: 10000;
        cursor: pointer;
        padding: 20px;
    `;
    
    const titleEl = document.createElement('h3');
    titleEl.textContent = title;
    titleEl.style.cssText = `
        color: white;
        margin-bottom: 20px;
        font-size: 1.5rem;
        text-align: center;
    `;
    
    const img = document.createElement('img');
    img.src = src;
    img.style.cssText = `
        max-width: 90%;
        max-height: 80%;
        border-radius: 12px;
        box-shadow: 0 10px 30px rgba(0, 0, 0, 0.5);
    `;
    
    const closeText = document.createElement('p');
    closeText.textContent = 'Click anywhere to close';
    closeText.style.cssText = `
        color: white;
        margin-top: 15px;
        opacity: 0.8;
        font-size: 1rem;
    `;
    
    modal.appendChild(titleEl);
    modal.appendChild(img);
    modal.appendChild(closeText);
    document.body.appendChild(modal);
    
    modal.addEventListener('click', () => {
        document.body.removeChild(modal);
    });
}

// Utility functions
function enableClassifyButton() {
    classifyBtn.disabled = false;
}

function setClassifyButtonLoading(loading) {
    if (loading) {
        classifyBtn.disabled = true;
        classifyBtn.innerHTML = `<div class="loading"></div> Classifying...`;
    } else {
        classifyBtn.disabled = false;
        classifyBtn.innerHTML = `<span class="btn-icon">🔍</span> Classify Image`;
    }
}

function setGenerateReportButtonLoading(loading) {
    if (loading) {
        generateReportBtn.disabled = true;
        generateReportBtn.innerHTML = `<div class="loading"></div> Generating Report...`;
    } else {
        generateReportBtn.disabled = false;
        generateReportBtn.innerHTML = `<span class="btn-icon">🤖</span> Generate AI Report`;
    }
}

function hideReportDisplay() {
    reportDisplay.style.display = 'none';
    currentReportData = null;
}

function clearResults() {
    resultsContainer.style.display = 'none';
    
    // Remove any existing detailed results or galleries
    const existingDetails = document.getElementById('detailedResults');
    const existingGallery = document.getElementById('imageGallery');
    const demoIndicator = document.querySelector('.demo-indicator');
    
    if (existingDetails) existingDetails.remove();
    if (existingGallery) existingGallery.remove();
    if (demoIndicator) demoIndicator.remove();
}

function resetPreview() {
    previewContainer.innerHTML = `
        <div class="preview-placeholder">
            <div class="placeholder-icon">🖼️</div>
            <p>No image selected</p>
        </div>
    `;
    currentImageFile = null;
    currentImageUrl = null;
    classifyBtn.disabled = true;
    clearResults();
}

function showError(message) {
    showMessage(message, 'error');
}

function showInfo(message) {
    showMessage(message, 'info');
}

function showSuccess(message) {
    showMessage(message, 'success');
}

function showMessage(message, type = 'error') {
    // Remove existing messages
    const existingMessages = document.querySelectorAll('.message');
    existingMessages.forEach(msg => msg.remove());
    
    // Create and show message
    const messageDiv = document.createElement('div');
    messageDiv.className = `message ${type}-message`;
    
    let bgColor;
    switch (type) {
        case 'error':
            bgColor = 'linear-gradient(135deg, #f44336, #d32f2f)';
            break;
        case 'success':
            bgColor = 'linear-gradient(135deg, #4caf50, #388e3c)';
            break;
        case 'info':
        default:
            bgColor = 'linear-gradient(135deg, #2196f3, #1976d2)';
            break;
    }
    
    messageDiv.style.cssText = `
        position: fixed;
        top: 20px;
        right: 20px;
        background: ${bgColor};
        color: white;
        padding: 1rem 1.5rem;
        border-radius: 12px;
        box-shadow: 0 6px 20px rgba(0, 0, 0, 0.3);
        z-index: 1000;
        animation: slideInRight 0.4s ease;
        font-weight: 500;
        max-width: 300px;
        cursor: pointer;
    `;
    
    messageDiv.textContent = message;
    document.body.appendChild(messageDiv);
    
    // Auto remove after 5 seconds
    setTimeout(() => {
        if (messageDiv.parentNode) {
            messageDiv.style.animation = 'slideOutRight 0.4s ease';
            setTimeout(() => {
                if (messageDiv.parentNode) {
                    messageDiv.remove();
                }
            }, 400);
        }
    }, 5000);
    
    // Remove on click
    messageDiv.addEventListener('click', () => {
        messageDiv.style.animation = 'slideOutRight 0.4s ease';
        setTimeout(() => {
            if (messageDiv.parentNode) {
                messageDiv.remove();
            }
        }, 400);
    });
}

function isValidImageUrl(url) {
    try {
        const urlObj = new URL(url);
        return /\.(jpg|jpeg|png|gif|bmp|webp)$/i.test(urlObj.pathname);
    } catch {
        return false;
    }
}

// Add additional CSS animations for messages
const additionalStyles = document.createElement('style');
additionalStyles.textContent = `
    @keyframes slideInRight {
        from { 
            transform: translateX(100%); 
            opacity: 0;
        }
        to { 
            transform: translateX(0); 
            opacity: 1;
        }
    }
    
    @keyframes slideOutRight {
        from { 
            transform: translateX(0); 
            opacity: 1;
        }
        to { 
            transform: translateX(100%); 
            opacity: 0;
        }
    }
    
    .loading {
        width: 16px;
        height: 16px;
        border: 2px solid #ffffff;
        border-top: 2px solid transparent;
        border-radius: 50%;
        animation: spin 1s linear infinite;
        display: inline-block;
        margin-right: 8px;
    }
    
    @keyframes spin {
        0% { transform: rotate(0deg); }
        100% { transform: rotate(360deg); }
    }
`;
document.head.appendChild(additionalStyles);