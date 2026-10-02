/**
 * SURVILLENCE TRAFFIC — Shared Shell (Sidebar + Topbar)
 * Call renderShell(pageTitle) inside DOMContentLoaded on every protected page.
 * Includes:
 *  - Premium sidebar navigation
 *  - Topbar with global processing indicator
 *  - User avatar + logout
 *  - Mobile responsiveness
 */
function renderShell(pageTitle = 'Dashboard') {
  const sidebarHTML = `
    <div id="sidebar-overlay" class="sidebar-overlay"></div>
    <nav id="sidebar" class="sidebar" aria-label="Main navigation">
      <div class="sidebar-logo">
        <div class="sidebar-logo-icon">
          <svg viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="1.8">
            <path d="M15 10l4.553-2.069A1 1 0 0121 8.866V19a2 2 0 01-2 2H5a2 2 0 01-2-2V5a2 2 0 012-2h3.5"/>
            <circle cx="9" cy="9" r="3"/>
          </svg>
        </div>
        <div class="sidebar-logo-text">SURVILLENCE<span>Traffic Intelligence</span></div>
        <button id="sidebar-close-btn" class="sidebar-close-btn" aria-label="Close navigation menu">
          <svg viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2" style="width:16px;height:16px"><line x1="18" y1="6" x2="6" y2="18"/><line x1="6" y1="6" x2="18" y2="18"/></svg>
        </button>
      </div>

      <nav class="sidebar-nav" aria-label="Sections">
        <div class="sidebar-section-label">Overview</div>
        <a class="nav-item" href="/dashboard" id="nav-dashboard">
          <svg viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2"><rect x="3" y="3" width="7" height="7" rx="1"/><rect x="14" y="3" width="7" height="7" rx="1"/><rect x="14" y="14" width="7" height="7" rx="1"/><rect x="3" y="14" width="7" height="7" rx="1"/></svg>
          Dashboard
        </a>
        <a class="nav-item" href="/surveillance" id="nav-surveillance">
          <svg viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2"><path d="M15 10l4.553-2.069A1 1 0 0121 8.866V19a2 2 0 01-2 2H5a2 2 0 01-2-2V5a2 2 0 012-2h3.5"/><circle cx="9" cy="9" r="3"/></svg>
          Surveillance
        </a>
        <a class="nav-item" href="/live" id="nav-live">
          <svg viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2"><polygon points="23 7 16 12 23 17 23 7"/><rect x="1" y="5" width="15" height="14" rx="2"/></svg>
          Live Monitoring
        </a>
        <a class="nav-item" href="/cameras" id="nav-cameras">
          <svg viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2"><path d="M23 7l-7 5 7 5V7z"/><rect x="1" y="5" width="15" height="14" rx="2" ry="2"/></svg>
          CCTV Connectivity
        </a>

        <div class="sidebar-section-label" style="margin-top:var(--space-4)">Analytics</div>
        <a class="nav-item" href="/analytics" id="nav-analytics">
          <svg viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2"><polyline points="22 12 18 12 15 21 9 3 6 12 2 12"/></svg>
          Vehicle Analytics
        </a>
        <a class="nav-item" href="/command-center" id="nav-command-center">
          <svg viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2"><path d="M12 2v20M2 12h20"/><circle cx="12" cy="12" r="7"/><path d="M12 8v4l3 2"/></svg>
          Command Center
        </a>
        <a class="nav-item" href="/traffic-ai" id="nav-traffic-ai">
          <svg viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2"><path d="M12 2L2 7l10 5 10-5-10-5z"/><path d="M2 17l10 5 10-5"/><path d="M2 12l10 5 10-5"/></svg>
          Traffic AI
        </a>
        <a class="nav-item" href="/vehicle-search" id="nav-vehicle-search">
          <svg viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2"><circle cx="11" cy="11" r="7"/><path d="m20 20-4-4"/><path d="M8 11h6"/></svg>
          Vehicle Search
        </a>

        <div class="sidebar-section-label" style="margin-top:var(--space-4)">Records</div>
        <a class="nav-item" href="/history" id="nav-history">
          <svg viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2"><circle cx="12" cy="12" r="10"/><polyline points="12 6 12 12 16 14"/></svg>
          History
        </a>
        <a class="nav-item" href="/reports" id="nav-reports">
          <svg viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2"><path d="M14 2H6a2 2 0 00-2 2v16a2 2 0 002 2h12a2 2 0 002-2V8z"/><polyline points="14 2 14 8 20 8"/><line x1="16" y1="13" x2="8" y2="13"/><line x1="16" y1="17" x2="8" y2="17"/></svg>
          Reports
        </a>
        <a class="nav-item" href="/settings" id="nav-settings">
          <svg viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2"><circle cx="12" cy="12" r="3"/><path d="M19.4 15a1.65 1.65 0 00.33 1.82l.06.06a2 2 0 010 2.83 2 2 0 01-2.83 0l-.06-.06a1.65 1.65 0 00-1.82-.33 1.65 1.65 0 00-1 1.51V21a2 2 0 01-2 2 2 2 0 01-2-2v-.09A1.65 1.65 0 009 19.4a1.65 1.65 0 00-1.82.33l-.06.06a2 2 0 01-2.83 0 2 2 0 010-2.83l.06-.06A1.65 1.65 0 004.68 15a1.65 1.65 0 00-1.51-1H3a2 2 0 01-2-2 2 2 0 012-2h.09A1.65 1.65 0 004.6 9a1.65 1.65 0 00-.33-1.82l-.06-.06a2 2 0 010-2.83 2 2 0 012.83 0l.06.06A1.65 1.65 0 009 4.68a1.65 1.65 0 001-1.51V3a2 2 0 012-2 2 2 0 012 2v.09a1.65 1.65 0 001 1.51 1.65 1.65 0 001.82-.33l.06-.06a2 2 0 012.83 0 2 2 0 010 2.83l-.06.06A1.65 1.65 0 0019.4 9a1.65 1.65 0 001.51 1H21a2 2 0 012 2 2 2 0 01-2 2h-.09a1.65 1.65 0 00-1.51 1z"/></svg>
          Settings
        </a>
      </nav>

      <div class="sidebar-footer">
        <div class="sidebar-user">
          <div class="sidebar-avatar" data-user-initials>--</div>
          <div class="sidebar-user-info">
            <div class="sidebar-user-name" data-user-name>User</div>
            <div class="sidebar-user-role" data-user-email>Operator</div>
          </div>
          <button class="btn btn-ghost btn-icon btn-sm" data-logout title="Sign out" aria-label="Sign out">
            <svg viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2" style="width:15px;height:15px"><path d="M9 21H5a2 2 0 01-2-2V5a2 2 0 012-2h4"/><polyline points="16 17 21 12 16 7"/><line x1="21" y1="12" x2="9" y2="12"/></svg>
          </button>
        </div>
      </div>
    </nav>

    <header class="topbar" role="banner">
      <div class="topbar-left">
        <button type="button" id="sidebar-toggle" class="topbar-btn sidebar-toggle-btn" aria-label="Toggle navigation menu">
          <svg viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2" style="width:18px;height:18px"><line x1="3" y1="12" x2="21" y2="12"/><line x1="3" y1="6" x2="21" y2="6"/><line x1="3" y1="18" x2="21" y2="18"/></svg>
        </button>

        <div class="topbar-breadcrumb">
          <span class="topbar-brand">SURVILLENCE TRAFFIC</span>
          <span class="topbar-sep">/</span>
          <h1 class="topbar-page">${pageTitle}</h1>
        </div>
      </div>

      <!-- Global Processing Indicator -->
      <div id="global-proc-indicator" class="global-proc-indicator" style="display:none">
        <div class="gpi-pulse-ring"></div>
        <div class="gpi-icon">
          <svg viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2" style="width:12px;height:12px"><circle cx="12" cy="12" r="10"/><polyline points="12 6 12 12 16 14"/></svg>
        </div>
        <div class="gpi-info">
          <div class="gpi-label">AI ANALYSIS</div>
          <div class="gpi-filename">Processing...</div>
        </div>
        <div class="gpi-status-wrap">
          <span class="gpi-status">Detecting vehicles...</span>
        </div>
        <a href="/surveillance" class="gpi-view-link" title="View progress">
          <svg viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2" style="width:13px;height:13px"><path d="M5 12h14M12 5l7 7-7 7"/></svg>
        </a>
      </div>

      <div class="topbar-actions">
        <button class="alert-bell-btn" id="global-alert-btn" aria-label="Traffic alerts" title="Traffic alerts">
          <span class="alert-bell-dot" id="global-alert-count">0</span>
          <svg viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="1.8"><path d="M18 8a6 6 0 0 0-12 0c0 7-3 7-3 9h18c0-2-3-2-3-9"/><path d="M10 21h4"/></svg>
        </button>
        <button class="theme-cycle-btn" data-theme-cycle aria-label="Switch theme" title="Switch theme">
          <svg viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="1.8" style="width:16px;height:16px"><path d="M12 3v2M12 19v2M3 12h2M19 12h2M5.64 5.64l1.42 1.42M16.94 16.94l1.42 1.42M5.64 18.36l1.42-1.42M16.94 7.06l1.42-1.42"/><circle cx="12" cy="12" r="4"/></svg>
        </button>
        <div class="server-status-dot" id="server-status-dot" title="Server status"></div>
        <div class="topbar-avatar" data-user-initials style="cursor:pointer" title="Account">--</div>
      </div>
    </header>
  `;

  // Insert sidebar and topbar directly into document.body without wrapper div
  document.body.insertAdjacentHTML('afterbegin', sidebarHTML);
  injectGlobalTrafficAssistant();

  // Active nav highlight
  const currentPath = window.location.pathname.split('/').pop();
  document.querySelectorAll('.nav-item[href]').forEach(a => {
    const aPath = a.getAttribute('href').split('/').pop();
    if (aPath === currentPath) a.classList.add('active');
  });

  // Mobile drawer handlers (click + touchstart support)
  const toggleBtn = document.getElementById('sidebar-toggle');
  const closeBtn  = document.getElementById('sidebar-close-btn');
  const sidebar   = document.getElementById('sidebar');
  const overlay   = document.getElementById('sidebar-overlay');

  function openDrawer(e) {
    if (e && typeof e.stopPropagation === 'function') e.stopPropagation();
    const sb = document.getElementById('sidebar');
    const ov = document.getElementById('sidebar-overlay');
    if (sb) sb.classList.add('mobile-open');
    if (ov) ov.classList.add('visible');
    document.body.classList.add('drawer-open');
  }

  function closeDrawer(e) {
    if (e && typeof e.stopPropagation === 'function') e.stopPropagation();
    const sb = document.getElementById('sidebar');
    const ov = document.getElementById('sidebar-overlay');
    if (sb) sb.classList.remove('mobile-open');
    if (ov) ov.classList.remove('visible');
    document.body.classList.remove('drawer-open');
  }

  function toggleDrawer(e) {
    if (e) {
      if (typeof e.preventDefault === 'function') e.preventDefault();
      if (typeof e.stopPropagation === 'function') e.stopPropagation();
    }
    const sb = document.getElementById('sidebar');
    if (sb && sb.classList.contains('mobile-open')) {
      closeDrawer(e);
    } else {
      openDrawer(e);
    }
  }

  if (toggleBtn) {
    toggleBtn.onclick = toggleDrawer;
  }
  if (closeBtn) {
    closeBtn.onclick = closeDrawer;
  }
  if (overlay) {
    overlay.onclick = closeDrawer;
  }
  document.querySelectorAll('.nav-item').forEach(el => el.addEventListener('click', () => {
    if (window.innerWidth <= 900) closeDrawer();
  }));
  document.addEventListener('keydown', e => {
    if (e.key === 'Escape') closeDrawer();
  });

  initShell();

  // Server status check (non-blocking)
  setTimeout(() => {
    const dot = document.getElementById('server-status-dot');
    if (dot) checkServerStatus(dot);
  }, 1000);
}


function escText(v){return String(v??'').replace(/[&<>"']/g,c=>({'&':'&amp;','<':'&lt;','>':'&gt;','"':'&quot;',"'":'&#039;'}[c]));}

function injectGlobalTrafficAssistant() {
  if (document.getElementById('traffic-assistant')) return;
  document.body.insertAdjacentHTML('beforeend', `
    <div id="traffic-alert-popover" class="traffic-popover" aria-hidden="true">
      <div class="traffic-popover-head"><div><strong>Traffic Alerts</strong><div class="text-xs text-muted">Latest intelligent incidents</div></div><button class="btn btn-ghost btn-icon btn-sm" id="alerts-close">×</button></div>
      <div id="global-alert-list" class="traffic-alert-list"><div class="text-sm text-muted" style="padding:12px">No alerts yet.</div></div>
    </div>
    <button id="traffic-assistant" class="traffic-assistant-fab" aria-label="Open Traffic Assistant" title="Traffic Assistant">
      <span class="assistant-pulse"></span>
      <span style="font-size:20px">✦</span>
    </button>
    <div id="traffic-assistant-panel" class="traffic-assistant-panel" aria-hidden="true">
      <div class="assistant-head"><div><strong>Traffic Copilot</strong><div class="text-xs text-muted" id="assistant-context-label">Local project-aware intelligence</div></div><button class="btn btn-ghost btn-icon btn-sm" id="assistant-close">×</button></div>
      <div id="assistant-quick-actions" class="assistant-quick-actions"></div>
      <div id="assistant-messages" class="assistant-messages">
        <div class="assistant-msg bot">Hi! I can explain the AI pipeline, interpret your latest analysis, troubleshoot accuracy, and navigate the system.</div>
      </div>
      <form id="assistant-form" class="assistant-input-row">
        <input id="assistant-input" class="form-input" placeholder="Ask: “summarize this analysis”…” autocomplete="off">
        <button class="btn btn-primary btn-icon" aria-label="Send">➤</button>
      </form>
    </div>`);

  const fab=document.getElementById('traffic-assistant');
  const panel=document.getElementById('traffic-assistant-panel');
  const close=document.getElementById('assistant-close');
  const form=document.getElementById('assistant-form');
  const input=document.getElementById('assistant-input');
  const quick=document.getElementById('assistant-quick-actions');
  const contextLabel=document.getElementById('assistant-context-label');
  const sessionId = new URLSearchParams(window.location.search).get('session') || new URLSearchParams(window.location.search).get('sessionId') || '';
  const addMsg=(text,who)=>{ const el=document.createElement('div'); el.className=`assistant-msg ${who}`; el.textContent=text; document.getElementById('assistant-messages').appendChild(el); document.getElementById('assistant-messages').scrollTop=99999; return el; };
  const showQuick=(items=[])=>{ if(!quick) return; quick.innerHTML=(items||[]).slice(0,4).map(x=>`<button type="button" class="assistant-chip">${x}</button>`).join(''); quick.querySelectorAll('.assistant-chip').forEach(b=>b.onclick=()=>{input.value=b.textContent; form.requestSubmit();}); };
  showQuick(['Summarize this analysis','How can I improve accuracy?','Which lane is busiest?','How is speed calculated?']);
  const toggle=()=>{ const open=panel.classList.toggle('open'); panel.setAttribute('aria-hidden',String(!open)); if(open){ input.focus(); if(contextLabel) contextLabel.textContent=sessionId?'Analysis-aware Copilot':'Latest-report aware Copilot'; } };
  fab.onclick=toggle; close.onclick=toggle;
  const askCopilot=(message)=>{ if(!message) return; if(!panel.classList.contains('open')) toggle(); input.value=message; form.requestSubmit(); };
  window.TrafficCopilotAsk=askCopilot;
  form.onsubmit=async e=>{ e.preventDefault(); const msg=input.value.trim(); if(!msg) return; input.value=''; addMsg(msg,'user'); const pending=addMsg('Thinking…','bot pending'); try { const res=await ApiClient.chat(msg,sessionId); pending.remove(); addMsg(res.answer || 'I could not answer that yet.','bot'); if(res.action){ const a=document.createElement('a'); a.className='assistant-action'; a.href=res.action.url; a.textContent=res.action.label; document.getElementById('assistant-messages').appendChild(a); } showQuick(res.suggestions || []); } catch(err) { pending.remove(); addMsg(err.message || 'Assistant unavailable.','bot'); } };

  const alertBtn=document.getElementById('global-alert-btn'); const pop=document.getElementById('traffic-alert-popover'); const alertClose=document.getElementById('alerts-close');
  const loadAlerts=async()=>{ try { const alerts=await ApiClient.getAlerts(); const unread=alerts.filter(a=>!a.acknowledged).length; const badge=document.getElementById('global-alert-count'); badge.textContent=unread>99?'99+':unread; badge.style.display=unread?'flex':'none'; document.getElementById('global-alert-list').innerHTML=alerts.length?alerts.slice(0,12).map(a=>`<div class="traffic-alert-item ${String(a.severity||'').toLowerCase()}"><div><strong>${escText(a.type||'INCIDENT')}</strong><div class="text-xs" style="margin-top:3px">${escText(a.message||'')}</div><div class="text-xs text-dim" style="margin-top:4px">${a.createdAt?formatDate(a.createdAt):''}</div></div>${a.acknowledged?'':'<button class="btn btn-ghost btn-sm alert-ack" data-id="'+a.id+'">Clear</button>'}</div>`).join(''):'<div class="text-sm text-muted" style="padding:12px">No alerts yet.</div>'; document.querySelectorAll('.alert-ack').forEach(b=>b.onclick=async()=>{await ApiClient.acknowledgeAlert(b.dataset.id); loadAlerts();}); } catch {} };
  alertBtn.onclick=()=>{ pop.classList.toggle('open'); loadAlerts(); }; alertClose.onclick=()=>pop.classList.remove('open'); loadAlerts(); setInterval(loadAlerts,15000);
}

window.renderShell = renderShell;
