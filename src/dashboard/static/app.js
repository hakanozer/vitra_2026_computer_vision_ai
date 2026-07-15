// src/dashboard/static/app.js
// Dashboard gercek zamanli veri, kamera secimi ve kamera-basi model atama

const API = '';
const PRODUCTION_BINDING = '__production__';

let selectedCameraId = null;
let cameraStats = {};
let cameraBindings = {};
let discoveredUsbCameras = [];
let registryModels = [];
let productionModelInfo = null;
let controlStateLoaded = false;
let analysisFilter = 'all';

function setControlFeedback(message, isError = false) {
  const el = document.getElementById('control-feedback');
  if (!el) return;
  el.textContent = message;
  el.style.color = isError ? 'var(--color-nok)' : 'var(--text-muted)';
}

function formatUptime(seconds) {
  const h = Math.floor(seconds / 3600);
  const m = Math.floor((seconds % 3600) / 60);
  const s = seconds % 60;
  return `${h}s ${m}d ${s}s`;
}

function updateLiveStream() {
  const liveStream = document.getElementById('live-stream');
  if (!liveStream || !selectedCameraId) return;
  liveStream.src = `${API}/api/camera/stream/${encodeURIComponent(selectedCameraId)}?annotated=true&t=${Date.now()}`;
}

function populateCameraSelect() {
  const select = document.getElementById('camera-select');
  if (!select) return;

  const cameraIds = Object.keys(cameraStats);
  if (!cameraIds.length) {
    select.innerHTML = '<option value="">Kamera yok</option>';
    selectedCameraId = null;
    return;
  }

  if (!selectedCameraId || !cameraStats[selectedCameraId]) {
    selectedCameraId = cameraIds[0];
  }

  select.innerHTML = cameraIds.map(cameraId => {
    const stats = cameraStats[cameraId] || {};
    const suffix = stats.camera_alive ? 'aktif' : 'pasif';
    return `<option value="${cameraId}" ${cameraId === selectedCameraId ? 'selected' : ''}>${cameraId} (${suffix})</option>`;
  }).join('');
}

function populateUsbCameraSelect() {
  const select = document.getElementById('usb-camera-select');
  if (!select) return;

  if (!discoveredUsbCameras.length) {
    select.innerHTML = '<option value="">Eklenebilir USB kamera bulunamadi</option>';
    return;
  }

  select.innerHTML = [
    '<option value="">USB kamera secin</option>',
    ...discoveredUsbCameras.map(camera => {
    const resolution = camera.width && camera.height ? ` ${camera.width}x${camera.height}` : '';
    return `<option value="${camera.source}">${camera.label}${resolution}</option>`;
  }),
  ].join('');
}

function populateModelSelect() {
  const select = document.getElementById('model-select');
  if (!select) return;

  const productionLabel = productionModelInfo?.version
    ? `Production (${productionModelInfo.version})`
    : 'Production (atanmamis)';

  const options = [
    `<option value="${PRODUCTION_BINDING}">${productionLabel}</option>`,
    ...registryModels.map(model => {
      const version = model.version;
      const map50 = Number(model.metrics?.mAP50 ?? 0).toFixed(3);
      return `<option value="${version}">${version} | mAP50=${map50} | dataset=${model.dataset_version}</option>`;
    }),
  ];

  select.innerHTML = options.join('');

  if (selectedCameraId) {
    const assignment = cameraBindings[selectedCameraId];
    const binding = assignment?.binding || PRODUCTION_BINDING;
    select.value = binding;
  }
}

function applyAnalysisFilter(items) {
  if (analysisFilter === 'defect') {
    return items.filter(item => item.status === 'defect');
  }
  return items;
}

async function fetchControlState({ includeDiscovery = false, preserveFeedback = false, silent = false } = {}) {
  const requests = [
    fetch(`${API}/api/camera/list`).then(r => r.json()),
    fetch(`${API}/api/model/camera-bindings`).then(r => r.json()),
    fetch(`${API}/api/model/registry`).then(r => r.json()),
    fetch(`${API}/api/model/production`).then(async r => (r.ok ? r.json() : null)),
  ];

  if (includeDiscovery) {
    requests.splice(1, 0, fetch(`${API}/api/camera/discover?max_index=6`).then(r => r.json()));
  }

  const responses = await Promise.all(requests);
  const cameraRes = responses[0];
  const discoverRes = includeDiscovery ? responses[1] : null;
  const bindingsRes = includeDiscovery ? responses[2] : responses[1];
  const registryRes = includeDiscovery ? responses[3] : responses[2];
  const productionRes = includeDiscovery ? responses[4] : responses[3];

  cameraStats = cameraRes || {};
  if (includeDiscovery) {
    discoveredUsbCameras = discoverRes?.cameras || [];
  }
  cameraBindings = bindingsRes?.bindings || {};
  registryModels = registryRes?.versions || [];
  productionModelInfo = productionRes;

  populateCameraSelect();
  populateUsbCameraSelect();
  populateModelSelect();
  updateLiveStream();

  if (!preserveFeedback && !silent) {
    setControlFeedback(includeDiscovery
      ? 'Kamera ve model listesi guncellendi.'
      : 'Kamera durumlari guncellendi.');
  }

  controlStateLoaded = true;
}

async function fetchStats() {
  if (!selectedCameraId) {
    return;
  }

  try {
    const cameraQuery = `camera_id=${encodeURIComponent(selectedCameraId)}`;
    const [stats, labels, recent] = await Promise.all([
      fetch(`${API}/api/dashboard/stats?${cameraQuery}`).then(r => r.json()),
      fetch(`${API}/api/dashboard/labeling-summary`).then(r => r.json()),
      fetch(`${API}/api/dashboard/recent-detections?limit=20&${cameraQuery}`).then(r => r.json()),
    ]);

    document.getElementById('total-frames').textContent = stats.total_frames_processed.toLocaleString();
    document.getElementById('total-defects').textContent = stats.total_defects_detected.toLocaleString();
    document.getElementById('pending-labels').textContent = labels.pending ?? '—';
    document.getElementById('approved-labels').textContent = labels.approved ?? '—';

    const uptime = Math.round(stats.uptime_seconds);
    document.getElementById('uptime-badge').textContent = `Uptime: ${formatUptime(uptime)}`;

    const modelBadge = document.getElementById('model-badge');
    const selectedCameraModel = stats.selected_camera_model;
    const assignedVersion = selectedCameraModel?.model_metadata?.version || 'Production';
    const badgeText = `Kamera: ${stats.selected_camera_id || '-'} | Model: ${assignedVersion}`;
    if (stats.model_loaded || selectedCameraModel?.binding !== PRODUCTION_BINDING) {
      modelBadge.textContent = badgeText;
      modelBadge.className = 'badge badge-green';
    } else {
      modelBadge.textContent = `${badgeText} | Yuklu Degil`;
      modelBadge.className = 'badge badge-red';
    }

    const statusCard = document.getElementById('status-card');
    const partStatus = document.getElementById('part-status');
    const partStatusSub = document.getElementById('part-status-sub');
    const lfs = stats.latest_frame_status || {};

    if (lfs.status === 'defect') {
      partStatus.textContent = 'HATALI (NOK)';
      partStatus.style.color = 'var(--color-nok)';
      statusCard.className = 'card status-card pulse';
      statusCard.style.borderColor = 'var(--color-nok)';

      const topDefect = (lfs.detections || []).find(d => d.class_name.startsWith('defect')) || {};
      const conf = topDefect.confidence ? `${(topDefect.confidence * 100).toFixed(1)}%` : '';
      partStatusSub.textContent = `Kamera ${selectedCameraId}: ${topDefect.class_name || 'defect'} ${conf}`.trim();
    } else if (lfs.status === 'ok') {
      partStatus.textContent = 'TEMIZ (OK)';
      partStatus.style.color = 'var(--color-ok)';
      statusCard.className = 'card status-card pulse-ok';
      statusCard.style.borderColor = 'var(--color-ok)';
      partStatusSub.textContent = `Kamera ${selectedCameraId}: OK sinifi tespit edildi.`;
    } else if (lfs.status === 'ok_implicit') {
      partStatus.textContent = 'Analiz Ediliyor';
      partStatus.style.color = 'var(--text-muted)';
      statusCard.className = 'card status-card';
      statusCard.style.borderColor = 'var(--border-color)';
      partStatusSub.textContent = `Kamera ${selectedCameraId}: esik ustu tespit yok.`;
    } else {
      partStatus.textContent = 'Sinyal Yok';
      partStatus.style.color = 'var(--text-muted)';
      statusCard.className = 'card status-card';
      statusCard.style.borderColor = 'var(--border-color)';
      partStatusSub.textContent = `Kamera ${selectedCameraId}: frame bekleniyor...`;
    }

    const tbody = document.getElementById('detections-body');
    const filteredDetections = applyAnalysisFilter(recent.detections || []);
    if (!filteredDetections.length) {
      const message = analysisFilter === 'defect'
        ? 'Secili kamera icin defect tespiti yok'
        : 'Secili kamera icin henuz analiz kaydi yok';
      tbody.innerHTML = `<tr><td colspan="5" style="text-align:center;color:var(--text-muted)">${message}</td></tr>`;
    } else {
      tbody.innerHTML = filteredDetections.map(d => {
        const det = d.detections[0] || {};
        const ts = new Date(d.timestamp * 1000).toLocaleTimeString('tr-TR');
        const conf = det.confidence ? `${(det.confidence * 100).toFixed(1)}%` : '—';
        const cls = det.class_name || (d.status === 'ok_implicit' ? 'Tespit Yok' : 'Analiz Edildi');
        const colorClass = d.status === 'defect' ? 'defect' : 'ok';
        return `<tr>
          <td>${ts}</td>
          <td>${d.camera_id}</td>
          <td class="${colorClass}">${cls}</td>
          <td>${conf}</td>
          <td>${d.inference_ms.toFixed(1)}</td>
        </tr>`;
      }).join('');
    }
  } catch (e) {
    console.warn('Dashboard fetch error:', e);
    setControlFeedback(`Dashboard veri okuma hatasi: ${e.message}`, true);
  }
}

async function applyModelBinding() {
  if (!selectedCameraId) {
    setControlFeedback('Once bir kamera secin.', true);
    return;
  }

  const modelSelect = document.getElementById('model-select');
  const modelVersion = modelSelect?.value || PRODUCTION_BINDING;
  const response = await fetch(`${API}/api/model/camera-bindings/${encodeURIComponent(selectedCameraId)}`, {
    method: 'PUT',
    headers: { 'Content-Type': 'application/json' },
    body: JSON.stringify({ model_version: modelVersion }),
  });

  const payload = await response.json();
  if (!response.ok) {
    throw new Error(payload.detail || 'Model atamasi basarisiz.');
  }

  cameraBindings[selectedCameraId] = payload.assignment;
  populateModelSelect();
  setControlFeedback(`${selectedCameraId} icin model atandi: ${payload.assignment.binding}`);
  await fetchStats();
}

async function addDiscoveredCamera() {
  const usbSelect = document.getElementById('usb-camera-select');
  const sourceValue = usbSelect?.value;
  const cameraIdInput = document.getElementById('new-camera-id');
  const cameraId = (cameraIdInput?.value || '').trim() || `camera-${sourceValue}`;

  if (!sourceValue) {
    setControlFeedback('Eklenecek USB kamera bulunamadi.', true);
    return;
  }

  const response = await fetch(`${API}/api/camera/add`, {
    method: 'POST',
    headers: { 'Content-Type': 'application/json' },
    body: JSON.stringify({
      camera_id: cameraId,
      source: Number(sourceValue),
      queue_size: 30,
    }),
  });

  const payload = await response.json();
  if (!response.ok) {
    throw new Error(payload.detail || 'Kamera eklenemedi.');
  }

  if (payload.status === 'exists') {
    selectedCameraId = cameraId;
    setControlFeedback(`Kamera zaten kayitli: ${cameraId}`);
    await fetchControlState({ includeDiscovery: true, preserveFeedback: true });
    updateLiveStream();
    await fetchStats();
    return;
  }

  setControlFeedback(`Kamera eklendi: ${cameraId} (source=${sourceValue})`);
  await fetchControlState({ includeDiscovery: true, preserveFeedback: true });
  selectedCameraId = cameraId;
  populateCameraSelect();
  updateLiveStream();
  await fetchStats();
}

async function refreshControls(includeDiscovery = false, silent = false) {
  try {
    await fetchControlState({ includeDiscovery, silent });
  } catch (e) {
    console.warn('Control fetch error:', e);
    setControlFeedback(`Kontrol listesi yuklenemedi: ${e.message}`, true);
  }
}

function registerEvents() {
  const cameraSelect = document.getElementById('camera-select');
  const usbSelect = document.getElementById('usb-camera-select');
  const analysisFilterSelect = document.getElementById('analysis-filter');

  cameraSelect?.addEventListener('change', async event => {
    selectedCameraId = event.target.value || null;
    populateModelSelect();
    updateLiveStream();
    await fetchStats();
  });

  usbSelect?.addEventListener('change', event => {
    const cameraIdInput = document.getElementById('new-camera-id');
    if (cameraIdInput) {
      cameraIdInput.value = event.target.value !== '' ? `camera-${event.target.value}` : '';
    }
  });

  analysisFilterSelect?.addEventListener('change', async event => {
    analysisFilter = event.target.value || 'all';
    await fetchStats();
  });

  document.getElementById('apply-model-btn')?.addEventListener('click', async () => {
    try {
      await applyModelBinding();
    } catch (e) {
      setControlFeedback(e.message, true);
    }
  });

  document.getElementById('add-camera-btn')?.addEventListener('click', async () => {
    try {
      await addDiscoveredCamera();
    } catch (e) {
      setControlFeedback(e.message, true);
    }
  });

  document.getElementById('refresh-cameras-btn')?.addEventListener('click', async () => {
    await refreshControls(true);
    await fetchStats();
  });
}

async function init() {
  registerEvents();
  await refreshControls(true);
  await fetchStats();
  setInterval(fetchStats, 3000);
  setInterval(async () => {
    if (!controlStateLoaded) {
      return;
    }
    await refreshControls(false, true);
  }, 10000);
}

init();
