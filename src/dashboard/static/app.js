// src/dashboard/static/app.js
// Dashboard gerçek zamanlı veri çekme

const API = '';  // Aynı origin

async function fetchStats() {
  try {
    const [stats, labels, recent] = await Promise.all([
      fetch(`${API}/api/dashboard/stats`).then(r => r.json()),
      fetch(`${API}/api/dashboard/labeling-summary`).then(r => r.json()),
      fetch(`${API}/api/dashboard/recent-detections?limit=20`).then(r => r.json()),
    ]);

    // Stats kartları
    document.getElementById('total-frames').textContent = stats.total_frames_processed.toLocaleString();
    document.getElementById('total-defects').textContent = stats.total_defects_detected.toLocaleString();
    document.getElementById('pending-labels').textContent = labels.pending ?? '—';
    document.getElementById('approved-labels').textContent = labels.approved ?? '—';

    const uptime = Math.round(stats.uptime_seconds);
    document.getElementById('uptime-badge').textContent = `Uptime: ${formatUptime(uptime)}`;

    const modelBadge = document.getElementById('model-badge');
    if (stats.model_loaded) {
      modelBadge.textContent = `Model: ${stats.model_version}`;
      modelBadge.className = 'badge badge-green';
    } else {
      modelBadge.textContent = 'Model: Yüklü Değil';
      modelBadge.className = 'badge badge-red';
    }

    // Son tespitler tablosu
    const tbody = document.getElementById('detections-body');
    if (recent.detections.length === 0) {
      tbody.innerHTML = '<tr><td colspan="5" style="text-align:center;color:#64748b">Henüz tespit yok</td></tr>';
    } else {
      tbody.innerHTML = recent.detections.map(d => {
        const det = d.detections[0] || {};
        const ts = new Date(d.timestamp * 1000).toLocaleTimeString('tr-TR');
        const conf = det.confidence ? (det.confidence * 100).toFixed(1) + '%' : '—';
        const cls = det.class_name || '—';
        const colorClass = cls.startsWith('defect') ? 'defect' : 'ok';
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
  }
}

function formatUptime(seconds) {
  const h = Math.floor(seconds / 3600);
  const m = Math.floor((seconds % 3600) / 60);
  const s = seconds % 60;
  return `${h}s ${m}d ${s}s`;
}

// 3 saniyede bir güncelle
fetchStats();
setInterval(fetchStats, 3000);
