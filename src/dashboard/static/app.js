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

    // Parça Durumu Güncelleme
    const statusCard = document.getElementById('status-card');
    const partStatus = document.getElementById('part-status');
    const partStatusSub = document.getElementById('part-status-sub');
    
    if (stats.latest_frame_status) {
      const lfs = stats.latest_frame_status;
      if (lfs.status === 'defect') {
        partStatus.textContent = 'HATALI (NOK)';
        partStatus.style.color = 'var(--color-nok)';
        statusCard.className = 'card status-card pulse';
        statusCard.style.borderColor = 'var(--color-nok)';
        
        const topDefect = lfs.detections.find(d => d.class_name.startsWith('defect')) || {};
        const conf = topDefect.confidence ? (topDefect.confidence * 100).toFixed(1) + '%' : '';
        partStatusSub.textContent = `Hata Tespit Edildi: ${topDefect.class_name || 'defect'} (${conf})`;
      } else if (lfs.status === 'ok') {
        // Sadece modelin açıkça "ok" sınıfı tespit ettiği durumda YESIL göster
        partStatus.textContent = 'TEMİZ (OK)';
        partStatus.style.color = 'var(--color-ok)';
        statusCard.className = 'card status-card pulse-ok';
        statusCard.style.borderColor = 'var(--color-ok)';
        partStatusSub.textContent = 'Parça başarıyla doğrulandı (OK sınıfı tespit edildi).';
      } else if (lfs.status === 'ok_implicit') {
        // Model HİÇBİR şey tespit etmedi — tarafsız (gri) göster
        partStatus.textContent = 'Analiz Ediliyor';
        partStatus.style.color = 'var(--text-muted)';
        statusCard.className = 'card status-card';
        statusCard.style.borderColor = 'var(--border-color)';
        partStatusSub.textContent = 'Kamerada eşik üzeri tespit yok — nesneyi kameraya daha yakın tutun.';
      } else {
        partStatus.textContent = 'Sinyal Yok';
        partStatus.style.color = 'var(--text-muted)';
        statusCard.className = 'card status-card';
        statusCard.style.borderColor = 'var(--border-color)';
        partStatusSub.textContent = 'Kamera görüntüsü işlenmesi bekleniyor...';
      }
    }

    // Son tespitler tablosu
    const tbody = document.getElementById('detections-body');
    if (recent.detections.length === 0) {
      tbody.innerHTML = '<tr><td colspan="5" style="text-align:center;color:var(--text-muted)">Henüz tespit yok</td></tr>';
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
