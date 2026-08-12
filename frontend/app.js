const form = document.querySelector('#verify-form');
const claimInput = document.querySelector('#claim');
const submitButton = document.querySelector('#submit-button');
const exampleButton = document.querySelector('#example-button');
const charCount = document.querySelector('#char-count');
const message = document.querySelector('#message');
const results = document.querySelector('#results');
const systemStatus = document.querySelector('#system-status');
const statusDot = document.querySelector('.status-dot');

const exampleClaim = 'انتشر خبر يزعم أن قرارًا جديدًا دخل حيز التنفيذ هذا الأسبوع.';

function escapeHtml(value) {
  return String(value ?? '').replace(/[&<>'"]/g, (char) => ({
    '&': '&amp;', '<': '&lt;', '>': '&gt;', "'": '&#39;', '"': '&quot;'
  }[char]));
}

function showMessage(text, type = '') {
  message.textContent = text;
  message.className = `message ${type}`.trim();
}

function setLoading(loading) {
  submitButton.disabled = loading;
  submitButton.querySelector('span:first-child').textContent = loading ? 'جاري التحليل…' : 'حلّل الادعاء';
}

function formatPercent(value) {
  return `${(Number(value || 0) * 100).toFixed(1)}%`;
}

function verdictClass(verdict) {
  if (verdict === 'False') return 'negative';
  if (verdict === 'Partly-false' || verdict === 'Unverifiable' || verdict === 'insufficient_evidence') return 'neutral';
  return '';
}

function renderProbabilities(probabilities = {}) {
  const container = document.querySelector('#probabilities');
  const entries = Object.entries(probabilities).sort((a, b) => b[1] - a[1]);
  container.innerHTML = entries.map(([label, value]) => `
    <div class="probability">
      <div class="probability-head"><span>${escapeHtml(label)}</span><strong>${formatPercent(value)}</strong></div>
      <div class="probability-track"><span style="width:${Math.max(2, Number(value) * 100)}%"></span></div>
    </div>
  `).join('') || '<p class="muted">لا توجد احتمالات متاحة.</p>';
}

function candidatePreview(candidate) {
  return candidate.content_excerpt || candidate.content || `${candidate.label || 'تصنيف غير متاح'} · ${candidate.claim_id || 'معرّف غير متاح'}`;
}

function renderCandidates(candidates = []) {
  const container = document.querySelector('#candidates');
  if (!candidates.length) {
    container.innerHTML = '<p class="muted">لم يُسترجع مرشحون كافون؛ راجع الادعاء أو استخدم مصادر خارجية.</p>';
    return;
  }
  container.innerHTML = candidates.map((candidate, index) => `
    <div class="candidate">
      <span class="candidate-rank">0${index + 1}</span>
      <span class="candidate-title" title="${escapeHtml(candidatePreview(candidate))}">${escapeHtml(candidatePreview(candidate))}</span>
      <span class="candidate-meta">${escapeHtml(candidate.source || 'مصدر غير متاح')} · ${Number(candidate.score || 0).toFixed(3)}</span>
    </div>
  `).join('');
}

function renderResult(data) {
  const verdict = data.verdict || 'غير حاسم';
  const confidence = Number(data.confidence || 0);
  const evidence = data.evidence || {};
  const sentence = evidence.sentences?.[0]?.text || 'لا توجد جملة دليلية كافية للعرض.';

  document.querySelector('#verdict').textContent = verdict;
  document.querySelector('#verdict').className = `verdict-value ${verdictClass(verdict)}`.trim();
  document.querySelector('#verdict-note').textContent = data.evidence_status === 'sufficient'
    ? 'النتيجة مدعومة بمرشح دليل قابل للمراجعة، ولا تعني أنها بديل عن قراءة المصدر الأصلي.'
    : 'امتنع النظام عن تقديم حكم قطعي لأن صلة الدليل أو الثقة أقل من العتبة الآمنة.';
  document.querySelector('#confidence').textContent = formatPercent(confidence);
  document.querySelector('#confidence-meter').style.width = `${Math.max(2, confidence * 100)}%`;
  document.querySelector('#evidence-status').textContent = data.evidence_status === 'sufficient' ? 'دليل متاح' : 'دليل غير كافٍ';
  document.querySelector('#model-version').textContent = data.model_version || 'Mizan model';
  document.querySelector('#retrieval-score').textContent = data.retrieval_score ? `صلة ${Number(data.retrieval_score).toFixed(3)}` : 'لا توجد صلة';
  document.querySelector('#evidence-source').textContent = evidence.source || 'مصدر غير متاح';
  document.querySelector('#evidence-meta').textContent = evidence.date ? `تاريخ المصدر: ${evidence.date}` : 'مصدر مسترجع من الفهرس المحلي';
  document.querySelector('#evidence-text').textContent = sentence;
  renderProbabilities(data.probabilities);
  renderCandidates(data.candidates);
  results.classList.remove('hidden');
  results.scrollIntoView({ behavior: 'smooth', block: 'start' });
}

async function checkHealth() {
  try {
    const response = await fetch('/health');
    const data = await response.json();
    if (response.ok && data.model_loaded === 'true') {
      systemStatus.textContent = 'النموذج جاهز للتجربة';
      statusDot.classList.add('ok');
    } else {
      systemStatus.textContent = 'الخدمة تعمل دون artifacts';
    }
  } catch (_error) {
    systemStatus.textContent = 'تعذر الاتصال بالخدمة';
  }
}

claimInput.addEventListener('input', () => {
  charCount.textContent = `${claimInput.value.length} / 2000`;
});

exampleButton.addEventListener('click', () => {
  claimInput.value = exampleClaim;
  claimInput.dispatchEvent(new Event('input'));
  claimInput.focus();
});

claimInput.addEventListener('keydown', (event) => {
  if ((event.ctrlKey || event.metaKey) && event.key === 'Enter') {
    form.requestSubmit();
  }
});

form.addEventListener('submit', async (event) => {
  event.preventDefault();
  const claim = claimInput.value.trim();
  if (claim.length < 3) {
    showMessage('اكتب ادعاءً أطول قليلًا حتى يمكن تحليله.', 'error');
    return;
  }
  setLoading(true);
  message.className = 'message hidden';
  try {
    const response = await fetch('/verify', {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify({ claim, top_k: 3 }),
    });
    const payload = await response.json();
    if (!response.ok) throw new Error(payload.detail || 'تعذر تنفيذ الطلب.');
    renderResult(payload);
  } catch (error) {
    showMessage(error.message || 'حدث خطأ غير متوقع. تأكد من تشغيل النموذج.', 'error');
  } finally {
    setLoading(false);
  }
});

checkHealth();
