// Logo theme switching script
// This script updates logo src when theme changes

function updateLogos() {
	const theme = document.documentElement.getAttribute('data-theme') || 
		(window.matchMedia('(prefers-color-scheme: dark)').matches ? 'dark' : 'light');
	
	const logoImages = document.querySelectorAll<HTMLImageElement>('.starlight-logo img, .hero-image img');
	
	logoImages.forEach((img) => {
		const currentSrc = img.src;
		if (theme === 'dark') {
			// Switch to dark logo (white)
			if (currentSrc.includes('for-light')) {
				img.src = currentSrc.replace('for-light', 'for-dark');
			}
		} else {
			// Switch to light logo (black)
			if (currentSrc.includes('for-dark')) {
				img.src = currentSrc.replace('for-dark', 'for-light');
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
