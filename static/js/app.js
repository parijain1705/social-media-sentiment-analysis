document.addEventListener('DOMContentLoaded', () => {
  const postInput = document.getElementById('post-input');
  const modelSelect = document.getElementById('model-select');
  const predictBtn = document.getElementById('predict-btn');
  const btnSpinner = document.getElementById('btn-spinner');
  const resultPlaceholder = document.getElementById('result-placeholder');
  const resultContainer = document.getElementById('result-container');
  
  const sentimentBadge = document.getElementById('sentiment-badge');
  const sentimentIcon = document.getElementById('sentiment-icon');
  const sentimentText = document.getElementById('sentiment-text');

  const probValPos = document.getElementById('prob-val-positive');
  const probValNeu = document.getElementById('prob-val-neutral');
  const probValNeg = document.getElementById('prob-val-negative');

  const barPos = document.getElementById('bar-positive');
  const barNeu = document.getElementById('bar-neutral');
  const barNeg = document.getElementById('bar-negative');

  const traceClean = document.getElementById('trace-clean');
  const traceFeatures = document.getElementById('trace-features');

  // Preset buttons
  document.querySelectorAll('.preset-btn').forEach(btn => {
    btn.addEventListener('click', () => {
      postInput.value = btn.getAttribute('data-text');
      triggerPrediction();
    });
  });

  // Predict button
  predictBtn.addEventListener('click', triggerPrediction);

  async function triggerPrediction() {
    const text = postInput.value.trim();
    if (!text) {
      alert('Please enter a post or select a sample first.');
      postInput.focus();
      return;
    }

    const selectedModel = modelSelect.value;

    // Loading UI state
    predictBtn.disabled = true;
    btnSpinner.style.display = 'inline-block';

    try {
      const response = await fetch('/api/predict', {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({
          text: text,
          model: selectedModel
        })
      });

      const data = await response.json();

      if (!response.ok) {
        alert(data.error || 'Failed to predict sentiment.');
        return;
      }

      displayResult(data);
    } catch (err) {
      console.error(err);
      alert('Error communicating with the backend server.');
    } finally {
      predictBtn.disabled = false;
      btnSpinner.style.display = 'none';
    }
  }

  function displayResult(data) {
    resultPlaceholder.classList.add('hidden');
    resultContainer.classList.remove('hidden');

    const sentiment = data.sentiment.toLowerCase();

    // Reset badge classes
    sentimentBadge.className = 'sentiment-badge';
    if (sentiment === 'positive') {
      sentimentBadge.classList.add('badge-positive');
      sentimentIcon.textContent = '😊';
      sentimentText.textContent = 'Positive';
    } else if (sentiment === 'negative') {
      sentimentBadge.classList.add('badge-negative');
      sentimentIcon.textContent = '😡';
      sentimentText.textContent = 'Negative';
    } else {
      sentimentBadge.classList.add('badge-neutral');
      sentimentIcon.textContent = '😐';
      sentimentText.textContent = 'Neutral';
    }

    // Update probabilities
    const conf = data.confidence || {};
    const posPct = conf.positive !== undefined ? conf.positive : 0;
    const neuPct = conf.neutral !== undefined ? conf.neutral : 0;
    const negPct = conf.negative !== undefined ? conf.negative : 0;

    probValPos.textContent = `${posPct}%`;
    probValNeu.textContent = `${neuPct}%`;
    probValNeg.textContent = `${negPct}%`;

    barPos.style.width = `${posPct}%`;
    barNeu.style.width = `${neuPct}%`;
    barNeg.style.width = `${negPct}%`;

    // Trace details
    traceClean.textContent = data.clean_text || '(none)';

    traceFeatures.innerHTML = '';
    if (data.detected_features && data.detected_features.length > 0) {
      data.detected_features.forEach(feat => {
        const span = document.createElement('span');
        span.className = 'tag';
        span.textContent = feat;
        traceFeatures.appendChild(span);
      });
    } else {
      traceFeatures.innerHTML = '<span style="color: #94a3b8; font-size: 0.8rem;">No known vocabulary features matched</span>';
    }
  }

  // Tabs for Visualizations
  const tabBtns = document.querySelectorAll('.tab-btn');
  const tabPanes = document.querySelectorAll('.tab-pane');

  tabBtns.forEach(btn => {
    btn.addEventListener('click', () => {
      tabBtns.forEach(b => b.classList.remove('active'));
      tabPanes.forEach(p => p.classList.remove('active'));

      btn.classList.add('active');
      const targetId = btn.getAttribute('data-tab');
      const targetPane = document.getElementById(targetId);
      if (targetPane) {
        targetPane.classList.add('active');
      }
    });
  });
});
