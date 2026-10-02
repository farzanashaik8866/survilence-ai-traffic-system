/* ============================================================
   SURVILLENCE TRAFFIC — Visual Enhancements Controller v2
   Theme engine + motion polish. No framework dependency.
   ============================================================ */
(function () {
  'use strict';

  const THEMES = ['dark', 'light', 'emerald', 'violet'];
  const THEME_LABELS = { dark: 'Dark', light: 'Light', emerald: 'Emerald', violet: 'Violet' };
  const THEME_ICONS = {
    dark: '<path d="M12 3a7 7 0 1 0 6.7 9A7.1 7.1 0 0 1 12 3Z"/>',
    light: '<circle cx="12" cy="12" r="4"/><path d="M12 2v2M12 20v2M4.93 4.93l1.42 1.42M17.65 17.65l1.42 1.42M2 12h2M20 12h2M4.93 19.07l1.42-1.42M17.65 6.35l1.42-1.42"/>',
    emerald: '<path d="M12 3c3.8 1.2 6 3.8 6 7.3 0 4.1-2.8 7.1-6 10.7-3.2-3.6-6-6.6-6-10.7C6 6.8 8.2 4.2 12 3Z"/>',
    violet: '<path d="m12 3 1.8 5.2L19 10l-5.2 1.8L12 17l-1.8-5.2L5 10l5.2-1.8L12 3Z"/>'
  };

  function getStoredTheme() {
    try {
      const authTheme = window.Auth && typeof Auth.getSettings === 'function' ? Auth.getSettings().theme : null;
      const local = localStorage.getItem('survillence_theme');
      const candidate = authTheme || local;
      return THEMES.includes(candidate) ? candidate : 'dark';
    } catch (_) { return 'dark'; }
  }

  function syncThemeUI(theme) {
    document.querySelectorAll('[data-theme-cycle]').forEach(btn => {
      btn.setAttribute('aria-label', `Switch theme. Current: ${THEME_LABELS[theme]}`);
      btn.title = `Theme: ${THEME_LABELS[theme]}`;
      const svg = btn.querySelector('svg');
      if (svg) svg.innerHTML = THEME_ICONS[theme];
      btn.dataset.currentTheme = theme;
    });
    document.querySelectorAll('[data-theme-label]').forEach(el => { el.textContent = THEME_LABELS[theme]; });
    document.querySelectorAll('.theme-btn').forEach(btn => {
      const active = btn.dataset.theme === theme;
      btn.classList.toggle('btn-primary', active);
      btn.classList.toggle('btn-ghost', !active);
      btn.setAttribute('aria-pressed', String(active));
    });
    document.querySelectorAll('meta[name="theme-color"]').forEach(meta => {
      const colors = { dark: '#060a10', light: '#f4f7fb', emerald: '#07110e', violet: '#0b0911' };
      meta.content = colors[theme] || colors.dark;
    });
  }

  function applyTheme(theme, persist = true) {
    if (!THEMES.includes(theme)) theme = 'dark';
    const root = document.documentElement;
    const commit = () => {
      root.setAttribute('data-theme', theme);
      syncThemeUI(theme);
      try { localStorage.setItem('survillence_theme', theme); } catch (_) {}
      if (persist && window.Auth && typeof Auth.updateSettings === 'function') {
        try { Auth.updateSettings({ theme }); } catch (_) {}
      }
      window.dispatchEvent(new CustomEvent('survillence:themechange', { detail: { theme } }));
    };

    if (root.getAttribute('data-theme') === theme) {
      syncThemeUI(theme);
      return;
    }

    if (root.animate && document.startViewTransition) {
      try {
        document.startViewTransition(commit);
      } catch (_) { commit(); }
    } else {
      commit();
    }
  }

  function nextTheme() {
    const current = rootTheme();
    const next = THEMES[(THEMES.indexOf(current) + 1) % THEMES.length];
    document.querySelectorAll('[data-theme-cycle]').forEach(btn => {
      btn.classList.add('theme-rotating');
      window.setTimeout(() => btn.classList.remove('theme-rotating'), 260);
    });
    applyTheme(next, true);
    if (window.Toast && typeof Toast.success === 'function') {
      try { Toast.success('Theme changed', `Switched to ${THEME_LABELS[next]} mode.`); } catch (_) {}
    }
  }

  function rootTheme() {
    const current = document.documentElement.getAttribute('data-theme');
    return THEMES.includes(current) ? current : getStoredTheme();
  }

  function injectThemeMeta() {
    let meta = document.querySelector('meta[name="theme-color"]');
    if (!meta) {
      meta = document.createElement('meta');
      meta.name = 'theme-color';
      document.head.appendChild(meta);
    }
  }

  function bindThemeControls() {
    document.querySelectorAll('[data-theme-cycle]').forEach(btn => {
      if (btn.dataset.bound === 'true') return;
      btn.dataset.bound = 'true';
      btn.addEventListener('click', nextTheme);
    });
  }

  function setupReveal() {
    const targets = document.querySelectorAll('.page-main > *, .features-section, .how-it-works, .metrics-strip, .cta-section, .landing-footer, .auth-trust-row, .auth-visual-footer');
    if (!targets.length) return;
    document.documentElement.classList.add('reveal-ready');
    targets.forEach((el, i) => {
      if (el.dataset.revealBound === 'true') return;
      el.dataset.revealBound = 'true';
      el.classList.add('reveal-on-scroll');
      el.dataset.revealDelay = String(Math.min((i % 5) + 1, 4));
    });
    if (!('IntersectionObserver' in window)) {
      targets.forEach(el => el.classList.add('is-visible'));
      return;
    }
    const observer = new IntersectionObserver(entries => {
      entries.forEach(entry => {
        if (entry.isIntersecting) {
          entry.target.classList.add('is-visible');
          observer.unobserve(entry.target);
        }
      });
    }, { threshold: 0.10, rootMargin: '0px 0px -6% 0px' });
    targets.forEach(el => observer.observe(el));
  }

  function setupTilt() {
    if (!window.matchMedia || !window.matchMedia('(hover: hover) and (pointer: fine)').matches) return;
    if (window.matchMedia('(prefers-reduced-motion: reduce)').matches) return;
    const cards = document.querySelectorAll('.hero-cockpit-frame, .feature-card, .class-showcase-box, .camera-cell');
    cards.forEach(card => {
      if (card.dataset.tiltBound === 'true') return;
      card.dataset.tiltBound = 'true';
      let raf = 0;
      let active = false;
      const reset = () => {
        cancelAnimationFrame(raf);
        card.classList.remove('is-tilting');
        card.style.transform = '';
      };
      card.addEventListener('pointermove', e => {
        const r = card.getBoundingClientRect();
        if (!r.width || !r.height) return;
        const px = (e.clientX - r.left) / r.width;
        const py = (e.clientY - r.top) / r.height;
        const rx = (0.5 - py) * 1.8;
        const ry = (px - 0.5) * 2.2;
        cancelAnimationFrame(raf);
        raf = requestAnimationFrame(() => {
          active = true;
          card.classList.add('is-tilting');
          card.style.transform = `perspective(1200px) rotateX(${rx.toFixed(2)}deg) rotateY(${ry.toFixed(2)}deg) translateY(-2px)`;
        });
      }, { passive: true });
      card.addEventListener('pointerleave', () => { if (active) reset(); });
      card.addEventListener('pointercancel', reset);
    });
  }

  function setupVideoFramePolish() {
    document.querySelectorAll('video').forEach(video => {
      const wrap = video.closest('.video-player-wrap');
      if (!wrap) return;
      video.addEventListener('play', () => wrap.classList.add('is-live'));
      video.addEventListener('pause', () => wrap.classList.remove('is-live'));
      video.addEventListener('ended', () => wrap.classList.remove('is-live'));
    });
  }

  function setupPageEnter() {
    document.querySelectorAll('.page-main').forEach(main => {
      requestAnimationFrame(() => main.classList.add('page-entered'));
    });
  }

  function boot() {
    injectThemeMeta();
    applyTheme(getStoredTheme(), false);
    bindThemeControls();
    setupReveal();
    setupTilt();
    setupVideoFramePolish();
    setupPageEnter();
  }

  if (document.readyState === 'loading') document.addEventListener('DOMContentLoaded', boot, { once: true });
  else boot();

  window.VisualEnhancements = { setTheme: applyTheme, nextTheme, THEMES };
})();
