// @ts-check
import starlight from '@astrojs/starlight';
import { defineConfig } from 'astro/config';

// https://astro.build/config
export default defineConfig({
	// QUAN TRỌNG: Cấu hình cho GitHub Pages
	site: 'https://nxank4.github.io',
	base: '/loclean',
	integrations: [
		starlight({
			title: 'Loclean',
			logo: {
				src: './src/assets/loclean-logo-for-light.svg',
				alt: 'Loclean logo',
				replacesTitle: true,
			},
			social: [
				{
					icon: 'github',
					label: 'GitHub',
					href: 'https://github.com/nxank4/loclean',
				},
			],
			editLink: {
				baseUrl: 'https://github.com/nxank4/loclean/edit/main/website/src/content/docs',
			},
			lastUpdated: true,
			pagination: true,
			sidebar: [
				{
					label: '🚀 Getting Started',
					autogenerate: { directory: 'getting-started' },
				},
				{
					label: '📘 User Guide',
					autogenerate: { directory: 'guides' },
				},
				{
					label: '🧠 Concepts',
					autogenerate: { directory: 'concepts' },
				},
				{
					label: '🔬 API Reference',
					autogenerate: { directory: 'reference' },
				},
			],
			customCss: ['./src/styles/custom.css'],
			head: [
				{
					tag: 'script',
					content: `
						function updateLogos() {
							const theme = document.documentElement.getAttribute('data-theme') || 
								(window.matchMedia('(prefers-color-scheme: dark)').matches ? 'dark' : 'light');
							
							// Update all logo images - use more specific selectors
							const logoSelectors = [
								'.starlight-logo img',
								'.hero-image img',
								'header img[alt*="logo" i]',
								'header img[alt*="Loclean" i]',
								'a[href="/loclean/"] img',
								'.title-wrapper img',
								'img[src*="loclean-logo"]'
							];
							
							logoSelectors.forEach((selector) => {
								const logos = document.querySelectorAll(selector);
								logos.forEach((img) => {
									let src = img.getAttribute('src') || img.src;
									// Handle Astro's asset processing with query params
									src = src.split('?')[0];
									
									if (theme === 'dark' && src.includes('for-light')) {
										const newSrc = src.replace('for-light', 'for-dark');
										// Preserve query params if any
										const query = (img.getAttribute('src') || img.src).split('?')[1];
										img.src = query ? newSrc + '?' + query : newSrc;
										img.setAttribute('src', img.src);
									} else if (theme === 'light' && src.includes('for-dark')) {
										const newSrc = src.replace('for-dark', 'for-light');
										const query = (img.getAttribute('src') || img.src).split('?')[1];
										img.src = query ? newSrc + '?' + query : newSrc;
										img.setAttribute('src', img.src);
									}
								});
							});
						}
						
						function markOutputBlocks() {
							// Find all paragraphs containing "Output:" text
							const paragraphs = Array.from(document.querySelectorAll('.sl-markdown-content p'));
							
							paragraphs.forEach((p) => {
								const text = p.textContent || '';
								if (text.includes('Output:') || text.includes('**Output:**')) {
									// Find the next sibling that is a code block (Expressive Code or regular pre)
									let nextSibling = p.nextElementSibling;
									
									// Check for Expressive Code blocks (figure.expressive-code)
									if (nextSibling && nextSibling.classList.contains('expressive-code')) {
										nextSibling.setAttribute('data-output', 'true');
										nextSibling.classList.add('output-block');
									}
									// Check for regular pre blocks
									else if (nextSibling && nextSibling.tagName === 'PRE') {
										nextSibling.setAttribute('data-output', 'true');
										nextSibling.classList.add('output-block');
									}
								}
							});
						}
						
						// Run immediately
						updateLogos();
						markOutputBlocks();
						
						// Watch for theme changes
						const observer = new MutationObserver(() => {
							setTimeout(updateLogos, 50);
						});
						observer.observe(document.documentElement, { 
							attributes: true, 
							attributeFilter: ['data-theme'] 
						});
						
						// Also watch for theme toggle clicks
						document.addEventListener('click', (e) => {
							if (e.target.closest('[data-theme-toggle]') || 
								e.target.closest('button[aria-label*="theme" i]') ||
								e.target.closest('[aria-label*="theme" i]')) {
								setTimeout(updateLogos, 150);
							}
						});
						
						// Run on DOM ready
						if (document.readyState === 'loading') {
							document.addEventListener('DOMContentLoaded', () => {
								updateLogos();
								markOutputBlocks();
							});
						}
						
						// Run after a short delay to catch dynamically loaded content
						setTimeout(() => {
							updateLogos();
							markOutputBlocks();
						}, 500);
						
						// Watch for new content being added (for SPA navigation)
						const contentObserver = new MutationObserver(() => {
							markOutputBlocks();
						});
						contentObserver.observe(document.body, {
							childList: true,
							subtree: true
						});
					`,
				},
			],
		}),
	],
});
