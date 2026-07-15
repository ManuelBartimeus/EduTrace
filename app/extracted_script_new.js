
  // â”€â”€ Slider sync â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€
  function syncSlider(id, val, suffix) {
    document.getElementById(id).textContent = val + suffix;
  }

  // â”€â”€ Mode toggle â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€
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

  // â”€â”€ CSV field definitions â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€
  // Only the fields the backend actually uses
  const CSV_FIELDS = [
    'student_name','attendance_rate','exam_score','grade_level',
    'fee_payment_status','distance_km','gender'
  ];

  // â”€â”€ Template download â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€
  function downloadTemplate() {
    const header = CSV_FIELDS.join(',');
    const ex = 'Kwame Asante,75,62,Grade 7,Fully Paid,3.5,Male';
    const blob = new Blob([header + '\n' + ex], { type: 'text/csv' });
    const a = document.createElement('a'); a.href = URL.createObjectURL(blob);
    a.download = 'edutrace_template.csv'; a.click();
  }

  // â”€â”€ CSV file input â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€
  const csvInput  = document.getElementById('csv-file-input');
  const dropZone  = document.getElementById('drop-zone');
  const dropFname = document.getElementById('drop-fname');
  const bulkPaste = document.getElementById('bulk-paste');

  csvInput.addEventListener('change', () => {
    const file = csvInput.files[0];
    if (!file) return;
    dropFname.textContent = 'ðŸ“„ ' + file.name;
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
    dropFname.textContent = 'ðŸ“„ ' + file.name;
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

  // â”€â”€ Bulk prediction â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€
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

    btn.disabled = true; spin.style.display = 'block'; label.textContent = 'Processingâ€¦';
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
      progMsg.textContent = `Analysed ${Math.min(i + BATCH, rows.length)} / ${rows.length} studentsâ€¦`;
    }

    bulkResultsData = results;
    progMsg.textContent = `âœ… Done â€” ${results.length} students analysed.`;
    btn.disabled = false; spin.style.display = 'none'; label.textContent = 'âš¡ Run Bulk Assessment';

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
      const res = await fetch('https://edutrace-backend-4mvu.onrender.com/predict', {
        method: 'POST', headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify(payload)
      });
      const json = await res.json();
      if (!res.ok) return { _ok: false, error: json.error || json.detail || `Error ${res.status}`, _input: data };
      // Normalise
      const prob      = json.probabilities ? json.probabilities.dropout_risk : (json.probability || 0);
      const riskLevel = prob >= 0.66 ? 'High' : prob >= 0.33 ? 'Moderate' : 'Low';
      return { _ok: true, prediction: json.prediction, probability: prob, risk_level: riskLevel, _input: data };
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

    document.getElementById('bulk-count-badge').textContent = `â€” ${results.length} students`;
    const badgesEl = document.getElementById('bulk-stats-badges');
    badgesEl.innerHTML = '';
    if (counts.High)     badgesEl.innerHTML += `<span class="b-badge high">ðŸš¨ ${counts.High} High</span>`;
    if (counts.Moderate) badgesEl.innerHTML += `<span class="b-badge moderate">âš ï¸ ${counts.Moderate} Moderate</span>`;
    if (counts.Low)      badgesEl.innerHTML += `<span class="b-badge low">âœ… ${counts.Low} Low</span>`;

    results.forEach((r, i) => {
      const inp = r._input || {};
      const row = document.createElement('tr');
      if (!r._ok) {
        row.innerHTML = `<td>${i+1}</td><td>${escHtml(inp.student_name||'â€”')}</td><td colspan="6" style="color:var(--red);font-size:0.8rem;">âš  ${escHtml(r.error||'Failed')}</td>`;
      } else {
        const lvl   = r.risk_level.toLowerCase();
        const pct   = Math.round(r.probability * 100);
        const att   = Math.round(parseFloat(inp.attendance_rate||0));
        const score = Math.round(parseFloat(inp.exam_score||0));
        const feeS  = inp.fee_payment_status || 'â€”';
        row.innerHTML = `
          <td style="color:var(--text-muted);">${i+1}</td>
          <td style="font-weight:600;">${escHtml(inp.student_name||'Student '+(i+1))}</td>
          <td><span class="tbl-badge ${lvl}">${lvl==='low'?'âœ…':lvl==='moderate'?'âš ï¸':'ðŸš¨'} ${r.risk_level}</span></td>
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

  // â”€â”€ Table sort â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€
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

  // â”€â”€ Export results CSV â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€
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

  // â”€â”€ Form submit â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€
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
    label.textContent = 'Analysingâ€¦';

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

      const res = await fetch('https://edutrace-backend-4mvu.onrender.com/predict', {
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
        _inputData:  data,
      };

      renderResults(normalised, data.student_name || 'Student');

    } catch (err) {
      showError('Could not reach the prediction server. Please try again shortly.');
    } finally {
      btn.disabled = false;
      spinner.style.display = 'none';
      label.textContent = 'âš¡ Run Risk Assessment';
    }
  }

  // â”€â”€ Render results â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€
  function renderResults(data, name) {
    const panel = document.getElementById('results-panel');

    // Remove placeholder
    const ph = document.getElementById('placeholder-card');
    if (ph) ph.remove();

    panel.innerHTML = '';

    // 1. Risk Gauge
    panel.appendChild(buildGauge(data));

    // 2. Input summary card
    if (data._inputData) panel.appendChild(buildInputSummary(data._inputData));

    // Scroll into view on mobile
    if (window.innerWidth < 900) {
      panel.scrollIntoView({ behavior: 'smooth', block: 'start' });
    }
  }

  // â”€â”€ Gauge card â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€
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
      <div><span class="risk-badge ${level}"><span>${level === 'low' ? 'âœ…' : level === 'moderate' ? 'âš ï¸' : 'ðŸš¨'}</span>${data.risk_level} Risk</span></div>
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

  // â”€â”€ Input summary card â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€
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
        <div class="icon" style="background:rgba(139,92,246,0.15);">ðŸ“Š</div>
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

  // â”€â”€ SMS card â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€
  function buildSMS(msg) {
    const div = document.createElement('div');
    div.className = 'sms-card';
    div.innerHTML = `
      <div class="card-title">
        <div class="icon" style="background:rgba(34,197,94,0.15);">ðŸ“±</div>
        SHAPtoSMS Alert Preview
      </div>
      <div class="sms-body">${escHtml(msg)}</div>
      <div class="sms-char-count">${msg.length} / 160 characters</div>
    `;
    return div;
  }

  // â”€â”€ Helpers â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€
  function escHtml(s) {
    return s.replace(/&/g,'&amp;').replace(/</g,'&lt;').replace(/>/g,'&gt;');
  }

  function showError(msg) {
    const t = document.getElementById('error-toast');
    t.textContent = 'âš ï¸  ' + msg;
    t.style.display = 'block';
    setTimeout(() => { t.style.display = 'none'; }, 5000);
  }

