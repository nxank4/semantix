// Logo theme switching script
// This script updates logo src when theme changes

function updateLogos() {
	const theme = document.documentElement.getAttribute('data-theme') || 
		(window.matchMedia('(prefers-color-scheme: dark)').matches ? 'dark' : 'light');
	
	// Update sidebar logo
	const sidebarLogos = document.querySelectorAll('.starlight-logo img');
	sidebarLogos.forEach((img) => {
		const currentSrc = img.getAttribute('src') || img.src;
		if (theme === 'dark') {
			// Switch to dark logo (white) for dark theme
			if (currentSrc.includes('for-light')) {
				const newSrc = currentSrc.replace('for-light', 'for-dark');
				img.setAttribute('src', newSrc);
				img.src = newSrc;
			}
		} else {
			// Switch to light logo (black) for light theme
			if (currentSrc.includes('for-dark')) {
				const newSrc = currentSrc.replace('for-dark', 'for-light');
				img.setAttribute('src', newSrc);
				img.src = newSrc;
			}
		}
	});
	
	// Update hero image
	const heroImages = document.querySelectorAll('.hero-image img, [class*="hero"] img');
	heroImages.forEach((img) => {
		const currentSrc = img.getAttribute('src') || img.src;
		if (theme === 'dark') {
			if (currentSrc.includes('for-light')) {
				const newSrc = currentSrc.replace('for-light', 'for-dark');
				img.setAttribute('src', newSrc);
				img.src = newSrc;
			}
		} else {
			if (currentSrc.includes('for-dark')) {
				const newSrc = currentSrc.replace('for-dark', 'for-light');
				img.setAttribute('src', newSrc);
				img.src = newSrc;
			}
		}
	});
}

// Watch for theme changes
const observer = new MutationObserver(updateLogos);
observer.observe(document.documentElement, {
	attributes: true,
	attributeFilter: ['data-theme']
});

// Initial update
if (document.readyState === 'loading') {
	document.addEventListener('DOMContentLoaded', updateLogos);
} else {
	updateLogos();
}

// Also listen for theme toggle clicks
document.addEventListener('click', (e) => {
	if (e.target.closest('[data-theme-toggle]') || e.target.closest('button[aria-label*="theme"]')) {
		setTimeout(updateLogos, 100);
	}
});
