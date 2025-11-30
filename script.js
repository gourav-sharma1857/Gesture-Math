// DOM Elements
const statusBadge = document.getElementById('statusBadge');
const statusText = document.getElementById('statusText');
const canvasImage = document.getElementById('canvasImage');
const canvasPlaceholder = document.getElementById('canvasPlaceholder');
const drawingStatus = document.getElementById('drawingStatus');
const modeBadge = document.getElementById('modeBadge');
const problemText = document.getElementById('problemText');
const typeBadge = document.getElementById('typeBadge');
const solutionDisplay = document.getElementById('solutionDisplay');
const errorBanner = document.getElementById('errorBanner');

// Format math text - remove LaTeX symbols
function formatMathText(text) {
    if (!text) return '';
    
    let formatted = text;
    
    // Remove $$ and $ delimiters
    formatted = formatted.replace(/\$\$/g, '');
    formatted = formatted.replace(/\$/g, '');
    
    // Convert LaTeX fractions: \frac{a}{b} → (a)/(b)
    formatted = formatted.replace(/\\frac\{([^}]+)\}\{([^}]+)\}/g, '($1)/($2)');
    
    // Convert square roots: \sqrt{x} → √(x)
    formatted = formatted.replace(/\\sqrt\{([^}]+)\}/g, '√($1)');
    formatted = formatted.replace(/\\sqrt/g, '√');
    
    // Convert powers
    formatted = formatted.replace(/\^{2}/g, '²');
    formatted = formatted.replace(/\^2/g, '²');
    formatted = formatted.replace(/\^{3}/g, '³');
    formatted = formatted.replace(/\^3/g, '³');
    formatted = formatted.replace(/\^{([^}]+)}/g, '^($1)');
    
    // Convert common LaTeX symbols
    formatted = formatted.replace(/\\times/g, '×');
    formatted = formatted.replace(/\\cdot/g, '·');
    formatted = formatted.replace(/\\div/g, '÷');
    formatted = formatted.replace(/\\pm/g, '±');
    formatted = formatted.replace(/\\neq/g, '≠');
    formatted = formatted.replace(/\\leq/g, '≤');
    formatted = formatted.replace(/\\geq/g, '≥');
    formatted = formatted.replace(/\\approx/g, '≈');
    formatted = formatted.replace(/\\infty/g, '∞');
    formatted = formatted.replace(/\\pi/g, 'π');
    formatted = formatted.replace(/\\theta/g, 'θ');
    formatted = formatted.replace(/\\alpha/g, 'α');
    formatted = formatted.replace(/\\beta/g, 'β');
    formatted = formatted.replace(/\\sum/g, 'Σ');
    formatted = formatted.replace(/\\int/g, '∫');
    formatted = formatted.replace(/\\partial/g, '∂');
    formatted = formatted.replace(/\\rightarrow/g, '→');
    formatted = formatted.replace(/\\Rightarrow/g, '⇒');
    formatted = formatted.replace(/\\therefore/g, '∴');
    
    // Remove remaining backslashes from LaTeX commands
    formatted = formatted.replace(/\\[a-zA-Z]+/g, '');
    
    // Clean up braces
    formatted = formatted.replace(/\{/g, '(');
    formatted = formatted.replace(/\}/g, ')');
    formatted = formatted.replace(/\s+/g, ' ');
    
    return formatted.trim();
}

// Parse solution into steps
function parseSolutionSteps(text) {
    if (!text) return [];
    
    const formatted = formatMathText(text);
    const lines = formatted.split(/\n/);
    const steps = [];
    let currentStep = null;
    
    for (const line of lines) {
        const trimmed = line.trim();
        if (!trimmed) continue;
        
        const stepMatch = trimmed.match(/^(Step\s*\d+|Solution|Answer|Result|Final\s*Answer|Given|Find|Therefore|Thus|Hence|So|We\s*have|First|Second|Third|Next|Finally)[:.]?\s*/i);
        
        if (stepMatch) {
            if (currentStep) steps.push(currentStep);
            currentStep = {
                label: stepMatch[1],
                content: trimmed.substring(stepMatch[0].length).trim()
            };
        } else if (currentStep) {
            currentStep.content += ' ' + trimmed;
        } else {
            steps.push({ label: '', content: trimmed });
        }
    }
    
    if (currentStep) steps.push(currentStep);
    return steps;
}

// Render solution steps
function renderSolution(solutionText) {
    const steps = parseSolutionSteps(solutionText);
    
    if (steps.length === 0) {
        return `
            <div class="empty-solution">
                <div class="empty-icon">👍</div>
                <p>Make a thumbs-up gesture to solve</p>
                <p class="hint">Draw your equation first, then hold thumb up</p>
            </div>
        `;
    }
    
    let html = '<div class="solution-steps">';
    for (const step of steps) {
        html += '<div class="step-item">';
        if (step.label) {
            html += `<div class="step-label">${step.label}</div>`;
        }
        html += `<div class="step-content">${step.content}</div>`;
        html += '</div>';
    }
    html += '</div>';
    
    return html;
}

// Update UI with data from Python
function updateUI(data) {
    // Update status badge
    statusBadge.className = 'status-badge';
    if (data.analyzing) {
        statusBadge.classList.add('status-analyzing');
        statusText.textContent = 'ANALYZING... Processing your equation';
    } else if (data.cooldown) {
        statusBadge.classList.add('status-cooldown');
        statusText.textContent = 'COOLDOWN - Wait before next gesture';
    } else {
        statusBadge.classList.add('status-ready');
        statusText.textContent = `READY - Drawing: ${data.drawing_active ? 'ON' : 'OFF'}`;
    }
    
    // Update canvas image
    if (data.drawing_canvas_b64) {
        canvasImage.src = `data:image/png;base64,${data.drawing_canvas_b64}`;
        canvasImage.style.display = 'block';
        canvasPlaceholder.style.display = 'none';
    }
    
    // Update drawing status
    if (data.drawing_active) {
        drawingStatus.className = 'drawing-on';
        drawingStatus.textContent = '● Drawing Mode ON';
    } else {
        drawingStatus.className = 'drawing-off';
        drawingStatus.textContent = '○ Drawing Paused (Press S)';
    }
    
    // Update mode badge
    modeBadge.textContent = data.operation_mode || 'SOLVE';
    
    // Update problem text
    problemText.textContent = formatMathText(data.problem);
    
    // Update solution type badge
    if (data.solution_type) {
        typeBadge.textContent = data.solution_type;
        typeBadge.style.display = 'inline-block';
    } else {
        typeBadge.style.display = 'none';
    }
    
    // Update solution display
    solutionDisplay.innerHTML = renderSolution(data.solution_text);
    
    // Hide error banner
    errorBanner.style.display = 'none';
}

// Show error state
function showError() {
    statusBadge.className = 'status-badge status-error';
    statusText.textContent = 'ERROR: Cannot connect to backend';
    errorBanner.style.display = 'block';
}

// Poll for updates from Python backend
async function pollForUpdates() {
    try {
        const response = await fetch('http://localhost:5000/api/status');
        const data = await response.json();
        updateUI(data);
    } catch (error) {
        showError();
        console.error('Connection error:', error);
    }
}

// Start polling
setInterval(pollForUpdates, 100);
pollForUpdates();

console.log('Math Solver UI loaded - waiting for Python backend on port 5000');