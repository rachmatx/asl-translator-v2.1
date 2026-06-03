document.addEventListener('DOMContentLoaded', () => {
  // Check if we are on the main detector page
  const isIndex = window.location.pathname.endsWith('index.html') || window.location.pathname === '/' || window.location.pathname.endsWith('/');
  


  // Inject Theme Switcher UI into the DOM
  let themeSwitcherHTML = `<div class="theme-switcher">`;
  if (isIndex) {
    themeSwitcherHTML += `<button class="theme-btn" id="devToggleBtn" style="border-right: 2px solid var(--border); margin-right: 4px; padding-right: 12px; border-radius: 0; display: flex; align-items: center; gap: 4px;"><span class="material-symbols-outlined" style="font-size: 16px;">build</span> Dev Mode</button>`;
  }
  themeSwitcherHTML += `
      <button class="theme-btn" data-theme="default">Dark</button>
      <button class="theme-btn" data-theme="theme-clean">Clean</button>
      <button class="theme-btn" data-theme="theme-neobrutalism">Brutal</button>
    </div>
  `;
  
  // Create wrapper to easily append
  const tempDiv = document.createElement('div');
  tempDiv.innerHTML = themeSwitcherHTML.trim();
  const themeNode = tempDiv.firstChild;
  
  const footer = document.querySelector('footer');
  if (footer && footer.parentNode) {
    footer.parentNode.insertBefore(themeNode, footer);
  } else {
    document.body.appendChild(themeNode);
  }

  const buttons = document.querySelectorAll('.theme-btn:not(#devToggleBtn)');
  const savedTheme = localStorage.getItem('asl-theme') || 'default';

  // Apply initial theme
  applyTheme(savedTheme);

  buttons.forEach(btn => {
    btn.addEventListener('click', () => {
      const theme = btn.getAttribute('data-theme');
      applyTheme(theme);
      localStorage.setItem('asl-theme', theme);
    });
  });

  function applyTheme(theme) {
    // Remove all theme classes first
    document.body.classList.remove('theme-clean', 'theme-neobrutalism');
    
    // Add the new theme class if it's not the default
    if (theme !== 'default') {
      document.body.classList.add(theme);
    }
    
    // Update active state on theme buttons
    buttons.forEach(btn => {
      if (btn.getAttribute('data-theme') === theme) {
        btn.classList.add('active');
      } else {
        btn.classList.remove('active');
      }
    });

    // Update guide image based on theme
    const guideImage = document.getElementById('aslGuideImage');
    if (guideImage) {
      if (theme === 'theme-clean') {
        guideImage.src = '/assets/img/panduan-asl.jpg';
      } else if (theme === 'theme-neobrutalism') {
        guideImage.src = '/assets/img/panduan-asl neobrutalism.jpg';
      } else {
        // default or dark
        guideImage.src = '/assets/img/panduan-asl dark modern.jpg';
      }
    }
  }

  // --- Developer Mode Toggle Logic (Only for index.html) ---
  if (isIndex) {
    const devToggleBtn = document.getElementById('devToggleBtn');
    
    function setDevMode(isDev) {
      const statsCard = document.getElementById('statsCard');
      const logCard = document.getElementById('logCard');
      const aslGuideCard = document.getElementById('aslGuideCard');
      
      if (isDev) {
        document.body.classList.add('dev-mode');
        if (statsCard) statsCard.style.display = 'block';
        if (logCard) logCard.style.display = 'block';
        if (aslGuideCard) aslGuideCard.style.display = 'none';
        devToggleBtn.innerHTML = '<span class="material-symbols-outlined" style="font-size: 16px;">person</span> User Mode';
      } else {
        document.body.classList.remove('dev-mode');
        if (statsCard) statsCard.style.display = 'none';
        if (logCard) logCard.style.display = 'none';
        if (aslGuideCard) aslGuideCard.style.display = 'block';
        devToggleBtn.innerHTML = '<span class="material-symbols-outlined" style="font-size: 16px;">build</span> Dev Mode';
      }
    }
    
    // Default is false (User Mode)
    const devModeSaved = localStorage.getItem('asl-dev-mode') === 'true';
    let isDevMode = devModeSaved;
    setDevMode(isDevMode);
    
    devToggleBtn.addEventListener('click', () => {
      isDevMode = !isDevMode;
      setDevMode(isDevMode);
      localStorage.setItem('asl-dev-mode', isDevMode);
    });
  }
});
