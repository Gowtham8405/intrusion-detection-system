/* ============================================================
   AI-Based Intrusion Detection System — JavaScript
   Handles navigation toggle and basic UI interactions
   ============================================================ */

document.addEventListener('DOMContentLoaded', function () {

    // ── Mobile Navigation Toggle ──
    const navToggle = document.querySelector('.nav-toggle');
    const navLinks = document.querySelector('.nav-links');

    if (navToggle && navLinks) {
        navToggle.addEventListener('click', function () {
            navLinks.classList.toggle('open');
            // Toggle hamburger icon
            this.textContent = navLinks.classList.contains('open') ? '✕' : '☰';
        });

        // Close nav when a link is clicked (mobile)
        navLinks.querySelectorAll('a').forEach(function (link) {
            link.addEventListener('click', function () {
                navLinks.classList.remove('open');
                navToggle.textContent = '☰';
            });
        });
    }

    // ── Active Navigation Link Highlighting ──
    const currentPath = window.location.pathname;
    document.querySelectorAll('.nav-links a').forEach(function (link) {
        if (link.getAttribute('href') === currentPath) {
            link.classList.add('active');
        }
    });

    // ── Intersection Observer for Fade-In Animations ──
    const animatedElements = document.querySelectorAll('.animate-on-scroll');
    if (animatedElements.length > 0 && 'IntersectionObserver' in window) {
        const observer = new IntersectionObserver(function (entries) {
            entries.forEach(function (entry) {
                if (entry.isIntersecting) {
                    entry.target.classList.add('is-visible');
                    observer.unobserve(entry.target);
                }
            });
        }, { threshold: 0.1 });

        animatedElements.forEach(function (el) {
            observer.observe(el);
        });
    }
});
