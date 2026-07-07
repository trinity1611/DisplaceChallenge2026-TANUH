/**
 * DISPLACE MedAI – Main Application Controller
 */

document.addEventListener('DOMContentLoaded', () => {
    
    // DOM Elements
    const uploadZone = document.getElementById('upload-zone');
    const fileInput = document.getElementById('file-input');
    const fileNameDisplay = document.getElementById('file-name');
    const btnProcess = document.getElementById('btn-process');
    const langSelect = document.getElementById('language-select');
    
    const outputContent = document.getElementById('output-content');
    const btnCopy = document.getElementById('btn-copy');
    const btnDownload = document.getElementById('btn-download');
    
    let selectedFile = null;
    let pollInterval = null;
    let lastProcessedResults = null;

    // --- Tab Handling ---
    const navAudioSummary = document.getElementById('nav-audio-summary');
    const navUserGuide = document.getElementById('nav-user-guide');
    const tabAudioSummary = document.getElementById('tab-content-audio-summary');
    const tabUserGuide = document.getElementById('tab-content-user-guide');

    function switchTab(activeNav, inactiveNav, activeTab, inactiveTab) {
        if (!activeNav || !inactiveNav || !activeTab || !inactiveTab) return;
        activeNav.classList.add('active');
        inactiveNav.classList.remove('active');
        activeTab.style.display = 'block';
        inactiveTab.style.display = 'none';
    }

    if (navAudioSummary && navUserGuide) {
        navAudioSummary.addEventListener('click', () => {
            switchTab(navAudioSummary, navUserGuide, tabAudioSummary, tabUserGuide);
        });

        navUserGuide.addEventListener('click', () => {
            switchTab(navUserGuide, navAudioSummary, tabUserGuide, tabAudioSummary);
        });
    }

    // --- File Upload Handling ---
    
    uploadZone.addEventListener('click', () => {
        if (!btnProcess.disabled || btnProcess.textContent === 'Process') {
            fileInput.click();
        }
    });

    fileInput.addEventListener('change', (e) => {
        if (e.target.files.length > 0) {
            selectedFile = e.target.files[0];
            fileNameDisplay.textContent = selectedFile.name;
            btnProcess.disabled = false;
        }
    });

    // --- Processing Logic ---

    btnProcess.addEventListener('click', async () => {
        if (!selectedFile) {
            showToast('Please select an audio file first.', 'error');
            return;
        }

        // UI updates
        btnProcess.disabled = true;
        btnProcess.textContent = 'Processing...';
        outputContent.innerHTML = '';
        logToOutput('Starting process...', 'progress-log');
        lastProcessedResults = null;

        try {
            // 1. Upload File
            logToOutput(`Uploading ${selectedFile.name}...`, 'progress-log');
            const formData = new FormData();
            formData.append('file', selectedFile);

            const uploadResp = await fetch('/api/upload', {
                method: 'POST',
                body: formData,
            });

            if (!uploadResp.ok) throw new Error('Upload failed');
            const uploadData = await uploadResp.json();
            const jobId = uploadData.job_id;
            
            logToOutput(`Upload complete. Job ID: ${jobId}`, 'progress-log');

            // 2. Start Processing
            const lang = langSelect.value;
            const processResp = await fetch(`/api/process/${jobId}`, {
                method: 'POST',
                headers: { 'Content-Type': 'application/json' },
                body: JSON.stringify({ language: lang }),
            });

            if (!processResp.ok) throw new Error('Failed to start pipeline');
            
            logToOutput(`Pipeline queued for ${lang.toUpperCase()} audio.`, 'progress-log');
            logToOutput(`Waiting for GPU resources...`, 'progress-log');

            // 3. Poll Status
            let lastMessage = '';
            pollInterval = setInterval(async () => {
                try {
                    const statResp = await fetch(`/api/jobs/${jobId}`);
                    if (!statResp.ok) return;
                    
                    const statData = await statResp.json();
                    
                    if (statData.stage_message && statData.stage_message !== lastMessage) {
                        lastMessage = statData.stage_message;
                        logToOutput(`[${statData.progress}%] ${lastMessage}`, 'progress-log');
                    }

                    if (statData.status === 'COMPLETED') {
                        clearInterval(pollInterval);
                        logToOutput(`Pipeline finished successfully! Fetching results...`, 'progress-log');
                        fetchAndDisplayResults(jobId);
                    } else if (statData.status === 'FAILED') {
                        clearInterval(pollInterval);
                        logToOutput(`Pipeline Failed: ${statData.error_message}`, 'error-log');
                        resetProcessButton();
                    }
                } catch (e) {
                    console.error('Poll error', e);
                }
            }, 2000);

        } catch (err) {
            console.error(err);
            logToOutput(`Error: ${err.message}`, 'error-log');
            showToast(err.message, 'error');
            resetProcessButton();
        }
    });

    async function fetchAndDisplayResults(jobId) {
        try {
            const resp = await fetch(`/api/jobs/${jobId}/results`);
            if (!resp.ok) throw new Error('Failed to fetch results');
            
            const results = await resp.json();
            lastProcessedResults = results;
            
            outputContent.innerHTML = ''; // clear logs
            
            // Format nice output
            let outputHtml = '';
            
            // Summary
            outputHtml += `<div class="result-section">=== CLINICAL SUMMARY ===</div>`;
            outputHtml += `<div>${results.summary.replace(/\n/g, '<br>')}</div>\n`;
            
            // Transcript
            outputHtml += `<div class="result-section">=== TRANSCRIPT ===</div>`;
            results.transcript.forEach(seg => {
                outputHtml += `<div>[${formatTime(seg.start_time)} - ${formatTime(seg.end_time)}] <strong>${seg.speaker_id}:</strong> ${seg.text}</div>`;
            });
            
            outputContent.innerHTML = outputHtml;
            
            showToast('Processing complete!', 'success');
            btnProcess.textContent = 'Process Complete';
            
            // Show the Done button
            const btnDone = document.getElementById('btn-done');
            if (btnDone) {
                btnDone.style.display = 'inline-block';
                // Remove existing listener to prevent duplicates if any, then add
                btnDone.onclick = () => {
                    window.location.reload();
                };
            }
            
        } catch (err) {
            logToOutput(`Error fetching results: ${err.message}`, 'error-log');
            resetProcessButton();
        }
    }

    function resetProcessButton() {
        btnProcess.textContent = 'Process';
        btnProcess.disabled = false;
    }

    function logToOutput(msg, className = '') {
        const div = document.createElement('div');
        div.className = className;
        div.textContent = msg;
        outputContent.appendChild(div);
        outputContent.scrollTop = outputContent.scrollHeight;
    }

    function formatTime(seconds) {
        const m = Math.floor(seconds / 60);
        const s = Math.floor(seconds % 60);
        return `${m}:${s.toString().padStart(2, '0')}`;
    }

    // --- Actions ---

    btnCopy.addEventListener('click', () => {
        if (!lastProcessedResults) return;
        const text = outputContent.innerText;
        navigator.clipboard.writeText(text).then(() => {
            showToast('Results copied to clipboard');
        });
    });

    btnDownload.addEventListener('click', () => {
        if (!lastProcessedResults) return;
        
        // Save as Text
        const text = outputContent.innerText;
        const dataStr = "data:text/plain;charset=utf-8," + encodeURIComponent(text);
        const a = document.createElement('a');
        a.href = dataStr;
        a.download = "medical_analysis_results.txt";
        a.click();
    });

    // --- Toast UI ---

    function showToast(message, type = 'info') {
        const container = document.getElementById('toast-container');
        const toast = document.createElement('div');
        toast.className = 'toast';
        toast.style.borderLeftColor = type === 'error' ? '#ef4444' : '#2a8f8f';
        toast.textContent = message;
        container.appendChild(toast);

        setTimeout(() => {
            toast.style.opacity = '0';
            setTimeout(() => toast.remove(), 300);
        }, 3000);
    }
});
