const drawStatus = document.getElementById('drawStatus');
const problemText = document.getElementById('problemText');
const analyzingOverlay = document.getElementById('analyzingOverlay');
const solutionContent = document.getElementById('solutionContent');
const solutionTypeBadge = document.getElementById('solutionTypeBadge');
const statusDot = document.getElementById('statusDot');

// ===== STATE TRACKING =====
let previousState = {
    drawing_active: null,
    problem: null,
    analyzing: null,
    solution_type: null,
    solution_text: null,
    error: null
};

let isCalculating = false;
let shouldUpdateSolution = false;

// ===== LATEX CLEANING FUNCTION =====
function cleanLatexText(text) {
    if (!text) return '';
    
    return text
        // Remove all dollar signs (single and double)
        .replace(/\$+/g, '')
        // Replace LaTeX fractions with readable format
        .replace(/\\frac\{([^}]+)\}\{([^}]+)\}/g, '($1)/($2)')
        .replace(/\\frac/g, '')
        // Replace common LaTeX math operators
        .replace(/\\times/g, '×')
        .replace(/\\div/g, '÷')
        .replace(/\\cdot/g, '·')
        .replace(/\\pm/g, '±')
        .replace(/\\ne/g, '≠')
        .replace(/\\le/g, '≤')
        .replace(/\\ge/g, '≥')
        .replace(/\\int/g, '∫')
        .replace(/\\infty/g, '∞')
        .replace(/\\pi/g, 'π')
        .replace(/\\theta/g, 'θ')
        .replace(/\\alpha/g, 'α')
        .replace(/\\beta/g, 'β')
        .replace(/\\gamma/g, 'γ')
        // Replace square roots
        .replace(/\\sqrt\{([^}]+)\}/g, '√($1)')
        .replace(/\\sqrt/g, '√')
        // Remove LaTeX text formatting
        .replace(/\\text\{([^}]+)\}/g, '$1')
        .replace(/\\left|\\right/g, '')
        // Remove all remaining LaTeX commands
        .replace(/\\[a-zA-Z]+/g, '')
        // Clean up braces and other symbols
        .replace(/[{}]/g, '')
        .replace(/\^/g, '^')
        // Remove percentage symbols that aren't part of the answer
        .replace(/%(?!\s*$)/g, '')
        // Remove extra spaces and trim
        .replace(/\s+/g, ' ')
        .trim();
}

// ===== UPDATE FUNCTIONS =====
function updateDrawingStatus(isActive) {
    // Only update if changed
    if (previousState.drawing_active === isActive) return;
    
    drawStatus.textContent = `Drawing: ${isActive ? 'ON' : 'OFF'}`;
    
    if (isActive) {
        statusDot.style.background = '#10b981';
        statusDot.classList.add('pulse');
    } else {
        statusDot.style.background = '#94a3b8';
        statusDot.classList.remove('pulse');
    }
    
    previousState.drawing_active = isActive;
}

function updateProblem(problem) {
    // Only update if changed
    if (previousState.problem === problem) return;
    
    const cleaned = cleanLatexText(problem);
    problemText.textContent = cleaned;
    
    // Add smooth fade animation
    problemText.style.opacity = '0';
    problemText.style.transform = 'translateY(10px)';
    
    setTimeout(() => {
        problemText.style.opacity = '1';
        problemText.style.transform = 'translateY(0)';
    }, 50);
    
    previousState.problem = problem;
}

function showAnalyzing() {
    // Only update if changed
    if (previousState.analyzing === true) return;
    
    analyzingOverlay.classList.add('active');
    isCalculating = true;
    previousState.analyzing = true;
    
    console.log('🔄 Started analyzing...');
}

function hideAnalyzing() {
    // Only update if changed
    if (previousState.analyzing === false) return;
    
    analyzingOverlay.classList.remove('active');
    
    // Mark that we should update solution when data arrives
    if (isCalculating) {
        shouldUpdateSolution = true;
        isCalculating = false;
        console.log('✅ Analysis complete - ready to display result');
    }
    
    previousState.analyzing = false;
}

function formatSolutionText(text) {
    if (!text) return '<div class="empty-state"><svg class="empty-icon" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="1.5"><path d="M9.663 17h4.673M12 3v1m6.364 1.636l-.707.707M21 12h-1M4 12H3m3.343-5.657l-.707-.707m2.828 9.9a5 5 0 117.072 0l-.548.547A3.374 3.374 0 0014 18.469V19a2 2 0 11-4 0v-.531c0-.895-.356-1.754-.988-2.386l-.548-.547z"/></svg><p class="empty-text">Make a thumbs-up gesture to solve the problem</p></div>';

    // Clean LaTeX from the solution text
    const cleaned = cleanLatexText(text);
    
    // Split by newlines and filter empty lines
    const lines = cleaned.split('\n').map(line => line.trim()).filter(line => line.length > 0);
    
    let formatted = '';
    let inSubSteps = false;
    
    lines.forEach((line, index) => {
        // Check if this is a main step header (Step 1:, Step 2:, etc.)
        if (line.match(/^Step \d+:/i)) {
            formatted += `<div class="step-header">${line}</div>`;
            inSubSteps = true;
        }
        // Check if this is a solution/answer/result header
        else if (line.match(/^(Solution|Answer|Result|Final Answer|Therefore|Conclusion):/i)) {
            formatted += `<div class="solution-header">${line}</div>`;
            inSubSteps = false;
        }
        // Check if this is a sub-step (starts with -, •, or is indented)
        else if (line.match(/^[-•]\s*/) || (inSubSteps && line.match(/^\s+/))) {
            const cleanedLine = line.replace(/^[-•]\s*/, '');
            formatted += `<div class="sub-step">${cleanedLine}</div>`;
        }
        // Check if this line contains an equation (has = sign)
        else if (line.includes('=') && !line.match(/^(Step|Solution|Answer|Result)/i)) {
            formatted += `<div class="equation-line">${line}</div>`;
        }
        // Regular explanation line
        else {
            formatted += `<div class="explanation-line">${line}</div>`;
        }
    });
    
    return `<div class="solution-text">${formatted}</div>`;
}

function updateSolution(solutionType, solutionText) {
    // ONLY update if we just finished calculating (shouldUpdateSolution flag is true)
    if (!shouldUpdateSolution) {
        console.log('⏸️ Skipping solution update - waiting for calculation to complete');
        return;
    }
    
    // Check if this is actually a new solution
    if (previousState.solution_type === solutionType && 
        previousState.solution_text === solutionText) {
        return;
    }
    
    console.log('📊 Displaying solution:', solutionType);
    
    // Update badge
    if (solutionType && solutionType !== 'Error') {
        solutionTypeBadge.textContent = solutionType;
        solutionTypeBadge.classList.add('visible');
    } else {
        solutionTypeBadge.classList.remove('visible');
    }
    
    // Update content with smooth transition
    const formattedText = formatSolutionText(solutionText);
    
    // Fade out
    solutionContent.style.opacity = '0';
    
    setTimeout(() => {
        solutionContent.innerHTML = formattedText;
        // Fade in
        solutionContent.style.opacity = '1';
    }, 200);
    
    previousState.solution_type = solutionType;
    previousState.solution_text = solutionText;
    
    // Reset the flag after displaying
    shouldUpdateSolution = false;
}

function showError(message) {
    // Only show error if we were calculating
    if (!shouldUpdateSolution && previousState.error === message) return;
    
    console.log('❌ Error:', message);
    
    solutionContent.innerHTML = `
        <div class="empty-state">
            <svg class="empty-icon error-icon" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="1.5">
                <path d="M12 8v4m0 4h.01M21 12a9 9 0 11-18 0 9 9 0 0118 0z"/>
            </svg>
            <p class="empty-text error-text">${message}</p>
        </div>
    `;
    solutionTypeBadge.classList.remove('visible');
    
    previousState.error = message;
    shouldUpdateSolution = false;
}

function clearSolution() {
    solutionContent.innerHTML = `
        <div class="empty-state">
            <svg class="empty-icon" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="1.5">
                <path d="M9.663 17h4.673M12 3v1m6.364 1.636l-.707.707M21 12h-1M4 12H3m3.343-5.657l-.707-.707m2.828 9.9a5 5 0 117.072 0l-.548.547A3.374 3.374 0 0014 18.469V19a2 2 0 11-4 0v-.531c0-.895-.356-1.754-.988-2.386l-.548-.547z"/>
            </svg>
            <p class="empty-text">Make a thumbs-up gesture to solve the problem</p>
        </div>
    `;
    solutionTypeBadge.classList.remove('visible');
    previousState.solution_type = null;
    previousState.solution_text = null;
    shouldUpdateSolution = false;
}

// ===== MAIN UPDATE FUNCTION =====
function updateUI(data) {
    // Update drawing status
    if (data.hasOwnProperty('drawing_active')) {
        updateDrawingStatus(data.drawing_active);
    }
    
    // Update problem text
    if (data.hasOwnProperty('problem')) {
        updateProblem(data.problem);
    }
    
    // Handle analyzing state changes
    if (data.hasOwnProperty('analyzing')) {
        if (data.analyzing && !previousState.analyzing) {
            // Just started analyzing
            showAnalyzing();
        } else if (!data.analyzing && previousState.analyzing) {
            // Just finished analyzing
            hideAnalyzing();
        }
    }
    
    // Only update solution if calculation just completed
    if (shouldUpdateSolution && data.hasOwnProperty('solution_type') && data.hasOwnProperty('solution_text')) {
        if (data.solution_text && data.solution_text !== '') {
            updateSolution(data.solution_type, data.solution_text);
        } else {
            clearSolution();
        }
    }
    
    // Handle errors
    if (data.hasOwnProperty('error') && data.error) {
        showError(data.error);
    }
}

// ===== POLLING FOR UPDATES =====
function pollForUpdates() {
    console.log('🔄 Starting to poll for updates...');
    
    setInterval(async () => {
        try {
            const response = await fetch('http://localhost:5000/api/status');
            if (!response.ok) {
                throw new Error(`HTTP error! status: ${response.status}`);
            }
            const data = await response.json();
            updateUI(data);
        } catch (error) {
            console.error('❌ Polling error:', error);
        }
    }, 500); // Poll every 500ms
}

// ===== INITIALIZATION =====
document.addEventListener('DOMContentLoaded', () => {
    console.log('✅ AI Math Solver UI loaded');
    console.log('🌐 Starting connection to Python backend...');
    
    // Test connection first
    fetch('http://localhost:5000/api/status')
        .then(response => response.json())
        .then(data => {
            console.log('✅ Successfully connected to Python backend!');
            console.log('Initial data:', data);
            updateUI(data);
            // Start polling
            pollForUpdates();
        })
        .catch(error => {
            console.error('❌ Cannot connect to Python backend:', error);
            console.log('Make sure your Python script is running!');
            alert('Cannot connect to Python backend. Make sure your Python script is running on port 5000.');
        });
});

console.log('💡 TIP: Open browser console (F12) to see connection status');
console.log('💡 Make sure Python backend is running on http://localhost:5000');
console.log('👍 Solution will ONLY update when you complete the thumbs-up gesture!');