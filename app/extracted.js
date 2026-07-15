
  // ── Slider sync ──────────────────────────────────────
  function syncSlider(id, val, suffix) {
    document.getElementById(id).textContent = val + suffix;
  }

  // ── Mode toggle ──────────────────────────────────────
  function setMode(mode) {
    const isInd = mode === 'individual';
    document.getElementById('panel-individual').style.display = isInd ? '' : 'none';
    document.getElementById('panel-bulk').style.display       = isInd ? 'none' : '';
    document.getElementById('mode-individual').classList.toggle('active', isInd);
    document.getElementById('mode-bulk').classList.toggle('active', !isInd);
    // Hide bulk results when switching back to individual
    if (isInd) document.getElementById('bulk-results-wrap').classList.remove('visible');
    // Hide individual results panel when switching to bulk
    const ph = document.getElementById('placeholder-card');
    if (!isInd && ph) { /* leave placeholder */ }
  }

  // ── CSV field definitions ─────────────────────────────
  // Only the fields the backend actually uses
  const CSV_FIELDS = [
    'student_name','attendance_rate','exam_score','grade_level',
    'fee_payment_status','distance_km','gender'
  ];

  // ── Template download ─────────────────────────────────
  function downloadTemplate() {
    const header = CSV_FIELDS.join(',');
    const ex = 'Kwame Asante,75,62,Grade 7,Fully Paid,3.5,Male';
    const blob = new Blob([header + '\n' + ex], { type: 'text/csv' });
    const a = document.createElement('a'); a.href = URL.createObjectURL(blob);
    a.download = 'edutrace_template.csv'; a.click();
  }

  // ── CSV file input ────────────────────────────────────
  const csvInput  = document.getElementById('csv-file-input');
  const dropZone  = document.getElementById('drop-zone');
  const dropFname = document.getElementById('drop-fname');
  const bulkPaste = document.getElementById('bulk-paste');

  csvInput.addEventListener('change', () => {
    const file = csvInput.files[0];
    if (!file) return;
    dropFname.textContent = '📄 ' + file.name;
    const reader = new FileReader();
    reader.onload = e => {
      const lines = e.target.result.split('\n').map(l => l.trim()).filter(Boolean);
      // Strip header if present
      const first = lines[0].toLowerCase();
      const hasHeader = CSV_FIELDS.some(f => first.includes(f));
      const dataLines = hasHeader ? lines.slice(1) : lines;
      bulkPaste.value = dataLines.join('\n');
      updateRowCount();
    };
    reader.readAsText(file);
  });

  // Drag & drop
  dropZone.addEventListener('dragover',  e => { e.preventDefault(); dropZone.classList.add('drag-over'); });
  dropZone.addEventListener('dragleave', () => dropZone.classList.remove('drag-over'));
  dropZone.addEventListener('drop', e => {
    e.preventDefault(); dropZone.classList.remove('drag-over');
    const file = e.dataTransfer.files[0];
    if (!file) return;
    dropFname.textContent = '📄 ' + file.name;
    const reader = new FileReader();
    reader.onload = ev => {
      const lines = ev.target.result.split('\n').map(l => l.trim()).filter(Boolean);
      const first = lines[0].toLowerCase();
      const hasHeader = CSV_FIELDS.some(f => first.includes(f));
      bulkPaste.value = (hasHeader ? lines.slice(1) : lines).join('\n');
      updateRowCount();
    };
    reader.readAsText(file);
  });

  bulkPaste.addEventListener('input', updateRowCount);

  function updateRowCount() {
    const rows = parsePasteRows();
    document.getElementById('bulk-row-count').textContent =
      rows.length === 0 ? '0 rows detected' : `${rows.length} student${rows.length > 1 ? 's' : ''} detected`;
  }

  function parsePasteRows() {
    return bulkPaste.value.split('\n')
      .map(l => l.trim()).filter(Boolean)
      .map(line => {
        const parts = line.split(',').map(v => v.trim());
        if (parts.length < CSV_FIELDS.length) return null;
        const obj = {};
        CSV_FIELDS.forEach((f, i) => obj[f] = parts[i]);
        return obj;
      }).filter(Boolean);
  }

  // ── Bulk prediction ───────────────────────────────────
  let bulkResultsData = [];
  let sortCol = -1, sortAsc = true;

  async function runBulkPrediction() {
    const rows = parsePasteRows();
    if (rows.length === 0) { showError('No valid rows found. Please upload a CSV or paste data.'); return; }

    const btn    = document.getElementById('bulk-run-btn');
    const spin   = document.getElementById('bulk-spinner');
    const label  = document.getElementById('bulk-btn-label');
    const progWrap = document.getElementById('bulk-progress-wrap');
    const progBar  = document.getElementById('bulk-progress-bar');
    const progMsg  = document.getElementById('bulk-processing-msg');

    btn.disabled = true; spin.style.display = 'block'; label.textContent = 'Processing…';
    progWrap.style.display = 'block'; progBar.style.width = '0%'; progMsg.textContent = '';

    bulkResultsData = [];
    const results = [];
    const BATCH = 5;

    for (let i = 0; i < rows.length; i += BATCH) {
      const chunk = rows.slice(i, i + BATCH);
      const chunkResults = await Promise.all(chunk.map(r => fetchPredict(r)));
      results.push(...chunkResults);
      const pct = Math.round(((i + chunk.length) / rows.length) * 100);
      progBar.style.width = pct + '%';
      progMsg.textContent = `Analysed ${Math.min(i + BATCH, rows.length)} / ${rows.length} students…`;
    }

    bulkResultsData = results;
    progMsg.textContent = `✅ Done — ${results.length} students analysed.`;
    btn.disabled = false; spin.style.display = 'none'; label.textContent = '⚡ Run Bulk Assessment';

    renderBulkTable(results, rows);
  }

  async function fetchPredict(data) {
    try {
      // Send only numeric types for numeric fields
      const payload = {
        student_name:       data.student_name || '',
        attendance_rate:    parseFloat(data.attendance_rate),
        exam_score:         parseFloat(data.exam_score),
        grade_level:        data.grade_level,
        fee_payment_status: data.fee_payment_status,
        distance_km:        parseFloat(data.distance_km),
        gender:             data.gender,
      };
      const res = await fetch('https://edutrace-backend-4mvu.onrender.com/explain', {
        method: 'POST', headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify(payload)
      });
      const json = await res.json();
      if (!res.ok) return { _ok: false, error: json.error || json.detail || `Error ${res.status}`, _input: data };
      // Normalise
      const prob      = json.probabilities ? json.probabilities.dropout_risk : (json.probability || 0);
      const riskLevel = prob >= 0.66 ? 'High' : prob >= 0.33 ? 'Moderate' : 'Low';
      return { _ok: true, prediction: json.prediction, probability: prob, risk_level: riskLevel, shap_ranked: json.shap_ranked, _input: data };
    } catch (e) {
      return { _ok: false, error: 'Network error', _input: data };
    }
  }

  function renderBulkTable(results, inputRows) {
    const wrap = document.getElementById('bulk-results-wrap');
    const tbody = document.getElementById('bulk-table-body');
    tbody.innerHTML = '';
    sortCol = -1;

    const counts = { High: 0, Moderate: 0, Low: 0, Error: 0 };
    results.forEach(r => {
      if (!r._ok) counts.Error++;
      else counts[r.risk_level] = (counts[r.risk_level] || 0) + 1;
    });

    document.getElementById('bulk-count-badge').textContent = `— ${results.length} students`;
    const badgesEl = document.getElementById('bulk-stats-badges');
    badgesEl.innerHTML = '';
    if (counts.High)     badgesEl.innerHTML += `<span class="b-badge high">🚨 ${counts.High} High</span>`;
    if (counts.Moderate) badgesEl.innerHTML += `<span class="b-badge moderate">⚠️ ${counts.Moderate} Moderate</span>`;
    if (counts.Low)      badgesEl.innerHTML += `<span class="b-badge low">✅ ${counts.Low} Low</span>`;

    results.forEach((r, i) => {
      const inp = r._input || {};
      const row = document.createElement('tr');
      if (!r._ok) {
        row.innerHTML = `<td>${i+1}</td><td>${escHtml(inp.student_name||'—')}</td><td colspan="6" style="color:var(--red);font-size:0.8rem;">⚠ ${escHtml(r.error||'Failed')}</td>`;
      } else {
        const lvl   = r.risk_level.toLowerCase();
        const pct   = Math.round(r.probability * 100);
        const att   = Math.round(parseFloat(inp.attendance_rate||0));
        const score = Math.round(parseFloat(inp.exam_score||0));
        const feeS  = inp.fee_payment_status || '—';
        row.innerHTML = `
          <td style="color:var(--text-muted);">${i+1}</td>
          <td style="font-weight:600;">${escHtml(inp.student_name||'Student '+(i+1))}</td>
          <td><span class="tbl-badge ${lvl}">${lvl==='low'?'✅':lvl==='moderate'?'⚠️':'🚨'} ${r.risk_level}</span></td>
          <td>
            <div class="prob-bar-wrap">
              <div class="prob-bar-track"><div class="prob-bar-fill ${lvl}" style="width:${pct}%"></div></div>
              <span class="prob-pct" style="color:${lvl==='high'?'var(--red)':lvl==='moderate'?'var(--orange)':'var(--green)'}">${pct}%</span>
            </div>
          </td>
          <td style="color:${r.prediction===1?'var(--red)':'var(--green)'}; font-weight:600;">${r.prediction===1?'At Risk':'Safe'}</td>
          <td>${att}%</td>
          <td>${score}/100</td>
          <td>${escHtml(feeS)}</td>
        `;
      }
      tbody.appendChild(row);
    });

    wrap.classList.add('visible');
    wrap.scrollIntoView({ behavior: 'smooth', block: 'start' });
  }

  // ── Table sort ────────────────────────────────────────
  function sortTable(col) {
    if (sortCol === col) sortAsc = !sortAsc; else { sortCol = col; sortAsc = true; }
    document.querySelectorAll('.bulk-table th').forEach((th, i) => th.classList.toggle('sorted', i === col));

    const tbody = document.getElementById('bulk-table-body');
    const rows  = Array.from(tbody.querySelectorAll('tr'));
    rows.sort((a, b) => {
      const va = a.cells[col]?.textContent.trim() || '';
      const vb = b.cells[col]?.textContent.trim() || '';
      const na = parseFloat(va), nb = parseFloat(vb);
      const cmp = isNaN(na) || isNaN(nb) ? va.localeCompare(vb) : na - nb;
      return sortAsc ? cmp : -cmp;
    });
    rows.forEach(r => tbody.appendChild(r));
  }

  // ── Export results CSV ────────────────────────────────
  function downloadResultsCSV() {
    if (!bulkResultsData.length) return;
    const header = 'student_name,risk_level,dropout_probability,classification,attendance_rate,exam_score,fee_payment_status,gender,grade_level';
    const lines = bulkResultsData.map((r, i) => {
      const inp = r._input || {};
      if (!r._ok) return `${inp.student_name||'Student '+(i+1)},ERROR,,,,,,,`;
      const pct = (r.probability * 100).toFixed(1);
      return [
        inp.student_name||'Student '+(i+1),
        r.risk_level, pct+'%',
        r.prediction===1?'At Risk':'Safe',
        inp.attendance_rate, inp.exam_score,
        inp.fee_payment_status, inp.gender, inp.grade_level
      ].join(',');
    });
    const blob = new Blob([header + '\n' + lines.join('\n')], { type: 'text/csv' });
    const a = document.createElement('a'); a.href = URL.createObjectURL(blob);
    a.download = 'edutrace_results.csv'; a.click();
  }

  // ── Form submit ──────────────────────────────────────
  document.getElementById('predict-form').addEventListener('submit', async (e) => {
    e.preventDefault();
    await runPrediction();
  });

  async function runPrediction() {
    const btn     = document.getElementById('predict-btn');
    const spinner = document.getElementById('spinner');
    const label   = document.getElementById('btn-label');

    btn.disabled = true;
    spinner.style.display = 'block';
    label.textContent = 'Analysing…';

    try {
      const form = document.getElementById('predict-form');
      const raw = {};
      new FormData(form).forEach((v, k) => { raw[k] = v; });

      // Send only the fields the backend uses; convert numeric sliders to numbers
      const data = {
        student_name:       raw.student_name || '',
        attendance_rate:    parseFloat(raw.attendance_rate),
        exam_score:         parseFloat(raw.exam_score),
        grade_level:        raw.grade_level,
        fee_payment_status: raw.fee_payment_status,
        distance_km:        parseFloat(raw.distance_km),
        gender:             raw.gender,
      };

      const res = await fetch('https://edutrace-backend-4mvu.onrender.com/explain', {
        method:  'POST',
        headers: { 'Content-Type': 'application/json' },
        body:    JSON.stringify(data),
      });

      if (!res.ok) {
        const errJson = await res.json().catch(() => ({}));
        showError(errJson.error || errJson.detail || `Server error (${res.status})`);
        return;
      }

      const json = await res.json();
      const prob      = json.probabilities ? json.probabilities.dropout_risk : (json.probability || 0);
      const riskLevel = prob >= 0.66 ? 'High' : prob >= 0.33 ? 'Moderate' : 'Low';
      const normalised = {
        prediction:  json.prediction,
        probability: prob,
        risk_level:  riskLevel,
        shap_ranked: json.shap_ranked,
        _inputData:  data,
      };

      renderResults(normalised, data.student_name || 'Student');

    } catch (err) {
      showError('Could not reach the prediction server. Please try again shortly.');
    } finally {
      btn.disabled = false;
      spinner.style.display = 'none';
      label.textContent = '⚡ Run Risk Assessment';
    }
  }

  // ── Render results ────────────────────────────────────
  function renderResults(data, name) {
    const panel = document.getElementById('results-panel');

    // Remove placeholder
    const ph = document.getElementById('placeholder-card');
    if (ph) ph.remove();

    panel.innerHTML = '';

    // 1. Risk Gauge
    panel.appendChild(buildGauge(data));

    // 2. Feature Contributions (SHAP)
    if (data.shap_ranked) panel.appendChild(buildFeatureBars(data.shap_ranked));

    // 3. Input summary card
    if (data._inputData) panel.appendChild(buildInputSummary(data._inputData));

    // 4. SMS Action Card (only if High/Moderate risk)
    if (data.risk_level === 'High' || data.risk_level === 'Moderate') {
        panel.appendChild(buildSMSAction(data._inputData));
    }

    // Scroll into view on mobile
    if (window.innerWidth < 900) {
      panel.scrollIntoView({ behavior: 'smooth', block: 'start' });
    }
  }

  // ── Gauge card ───────────────────────────────────────
  function buildGauge(data) {
    const pct   = Math.round(data.probability * 100);
    const level = data.risk_level.toLowerCase();
    const colorMap = { low: '#22c55e', moderate: '#f59e0b', high: '#ef4444' };
    const col  = colorMap[level] || '#3b82f6';

    const R     = 80;
    const total = Math.PI * R;
    const fill  = total * (pct / 100);
    const offset = total - fill;

    const div = document.createElement('div');
    div.className = `gauge-card risk-${level}`;
    div.id = 'gauge-card';
    div.innerHTML = `
      <div class="gauge-title">Dropout Risk Score</div>
      <div class="gauge-svg-wrap">
        <svg viewBox="0 0 200 110" xmlns="http://www.w3.org/2000/svg">
          <path d="M 20 90 A 80 80 0 0 1 180 90" fill="none" stroke="#1e2d47" stroke-width="12" stroke-linecap="round" />
          <path id="gauge-fill" d="M 20 90 A 80 80 0 0 1 180 90" fill="none" stroke="${col}" stroke-width="12" stroke-linecap="round" stroke-dasharray="${total}" stroke-dashoffset="${total}" style="transition: stroke-dashoffset 1s cubic-bezier(0.4,0,0.2,1)" />
        </svg>
        <div class="gauge-value" style="color:${col}" id="gauge-num">0%</div>
      </div>
      <div><span class="risk-badge ${level}"><span>${level === 'low' ? '✅' : level === 'moderate' ? '⚠️' : '🚨'}</span>${data.risk_level} Risk</span></div>
      <div class="divider" style="margin-top:1.2rem;"></div>
      <div class="stats-row" style="margin-top:0.75rem;">
        <div class="stat-box"><div class="stat-val">${pct}%</div><div class="stat-lbl">Dropout Prob.</div></div>
        <div class="stat-box"><div class="stat-val">${data.prediction === 1 ? 'At Risk' : 'Safe'}</div><div class="stat-lbl">Classification</div></div>
      </div>
    `;

    setTimeout(() => {
      const arc = div.querySelector('#gauge-fill');
      const num = div.querySelector('#gauge-num');
      if (arc) arc.style.strokeDashoffset = offset;
      if (num) animateNum(num, 0, pct, 900, '%');
    }, 80);

    return div;
  }

  function animateNum(el, from, to, dur, suffix) {
    const start = performance.now();
    function step(ts) {
      const t = Math.min((ts - start) / dur, 1);
      el.textContent = Math.round(from + (to - from) * easeOut(t)) + suffix;
      if (t < 1) requestAnimationFrame(step);
    }
    requestAnimationFrame(step);
  }

  function easeOut(t) { return 1 - Math.pow(1 - t, 3); }

  // ── Input summary card ──────────────────────────────
  function buildInputSummary(inp) {
    const div = document.createElement('div');
    div.className = 'features-card';
    const rows = [
      ['Attendance Rate',    inp.attendance_rate + '%'],
      ['Exam Score',         inp.exam_score + ' / 100'],
      ['Grade Level',        inp.grade_level],
      ['Fee Payment Status', inp.fee_payment_status],
      ['Distance from School', inp.distance_km + ' km'],
      ['Gender',             inp.gender],
    ];
    div.innerHTML = `
      <div class="card-title">
        <div class="icon" style="background:rgba(139,92,246,0.15);">📊</div>
        Input Summary
      </div>
      ${rows.map(([label, val]) => `
        <div class="feat-item">
          <div class="feat-header">
            <span class="feat-name">${label}</span>
            <span class="feat-val">${val}</span>
          </div>
        </div>
      `).join('')}
    `;
    return div;
  }

  // ── SHAP Feature bars ─────────────────────────────────
  function buildFeatureBars(shap) {
    const div = document.createElement('div');
    div.className = 'features-card';
    const bars = shap.slice(0, 5).map(([feat, val]) => {
      const isPushing = val < 0; 
      const color = isPushing ? 'var(--red)' : 'var(--green)';
      const width = Math.min(Math.abs(val) * 30, 100);
      return `
        <div class="feat-item">
          <div class="feat-header">
            <span class="feat-name">${escHtml(feat.replace(/_/g, ' '))}</span>
            <span class="feat-val" style="color:${color}">${isPushing ? '+ Risk' : '- Risk'}</span>
          </div>
          <div class="prob-bar-track">
            <div class="prob-bar-fill" style="width:${width}%; background:${color}; margin-${isPushing ? 'left' : 'right'}:auto;"></div>
          </div>
        </div>
      `;
    }).join('');

    div.innerHTML = `
      <div class="card-title">
        <div class="icon" style="background:rgba(59,130,246,0.15);">🔍</div>
        Top Risk Factors
      </div>
      ${bars}
    `;
    return div;
  }

  // ── SMS Action Card ───────────────────────────────────
  function buildSMSAction(inputData) {
    const div = document.createElement('div');
    div.className = 'sms-card';
    div.innerHTML = `
      <div class="card-title">
        <div class="icon" style="background:rgba(34,197,94,0.15);">📱</div>
        Send SMS Alert to Guardian
      </div>
      <div class="sms-body" style="font-size:0.95rem; margin-bottom:1rem; color:var(--text-muted);">
        This student is flagged as at risk. You can automatically send an SMS alert to their guardian (via the teacher) containing a summary of the risk factors.
      </div>
      <div style="display:flex; gap:0.5rem; margin-bottom:0.5rem;">
        <input type="text" id="teacher-phone" placeholder="233xxxxxxxxx" style="flex:1; padding:0.5rem; border:1px solid var(--border); border-radius:4px; background:var(--bg); color:var(--text);" />
        <button id="send-sms-btn" class="btn" style="padding:0.5rem 1rem;">Send SMS</button>
      </div>
      <div id="sms-status" style="font-size:0.85rem; font-weight:500; margin-top:0.5rem;"></div>
    `;

    const btn = div.querySelector('#send-sms-btn');
    btn.onclick = async () => {
      const phone = div.querySelector('#teacher-phone').value.trim();
      const status = div.querySelector('#sms-status');
      if (!phone) { status.textContent = 'Please enter a phone number.'; status.style.color = 'var(--red)'; return; }
      
      btn.disabled = true;
      btn.textContent = 'Sending...';
      status.textContent = '';
      
      try {
        const payload = { ...inputData, teacher_phone: phone };
        const res = await fetch('https://edutrace-backend-4mvu.onrender.com/alert', {
          method: 'POST', headers: { 'Content-Type': 'application/json' },
          body: JSON.stringify(payload)
        });
        const json = await res.json();
        
        if (json.alert_triggered || json.message) {
          status.textContent = '✅ SMS Drafted: ' + json.message + (json.sms_response && json.sms_response.error ? ' (API Error: '+json.sms_response.error+')' : '');
          status.style.color = json.sms_response && json.sms_response.error ? 'var(--orange)' : 'var(--green)';
        } else if (json.error) {
          status.textContent = '❌ ' + json.error;
          status.style.color = 'var(--red)';
        } else {
          status.textContent = '❌ Failed to send SMS. Risk may not meet threshold.';
          status.style.color = 'var(--red)';
        }
      } catch(e) {
        status.textContent = '❌ Network error.';
        status.style.color = 'var(--red)';
      } finally {
        btn.disabled = false;
        btn.textContent = 'Send SMS';
      }
    };
    return div;
  }

  // ── Helpers ──────────────────────────────────────────
  function escHtml(s) {
    return s.replace(/&/g,'&amp;').replace(/</g,'&lt;').replace(/>/g,'&gt;');
  }

  function showError(msg) {
    const t = document.getElementById('error-toast');
    t.textContent = '⚠️  ' + msg;
    t.style.display = 'block';
    setTimeout(() => { t.style.display = 'none'; }, 5000);
  }
