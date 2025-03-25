// BugXpert main JavaScript file

document.addEventListener('DOMContentLoaded', function() {
    // Clear any loading indicators that might be visible from previous sessions
    document.querySelectorAll('[id*="loading"]').forEach(el => {
        el.classList.add('d-none');
    });
    
    // Hide result sections that might be visible
    document.querySelectorAll('#prediction-result, #criticality-result, #resultsSection').forEach(el => {
        if (el.id === 'resultsSection') {
            el.style.display = 'none';
        } else {
            el.classList.add('d-none');
        }
    });
    
    // Get current page URL
    const currentLocation = window.location.pathname;
    
    // Get all nav links
    const navLinks = document.querySelectorAll('.navbar-nav .nav-link');
    
    // Loop through all nav links
    navLinks.forEach(link => {
        // Get href attribute
        const href = link.getAttribute('href');
        
        // Check if the href matches the current location
        if (href === currentLocation || 
            (href === '/' && currentLocation === '/') ||
            (href !== '/' && currentLocation.includes(href))) {
            // Add active class
            link.classList.add('active');
        }
    });
    
    // Enhanced Smooth scrolling for anchor links
    document.querySelectorAll('a[href^="#"]').forEach(anchor => {
        anchor.addEventListener('click', function(e) {
            e.preventDefault();
            
            const targetId = this.getAttribute('href');
            const targetElement = document.querySelector(targetId);
            
            if (targetElement) {
                // Add highlight effect to the target element
                targetElement.classList.add('highlight-target');
                
                // Smooth scroll to target
                window.scrollTo({
                    top: targetElement.offsetTop - 80,
                    behavior: 'smooth'
                });
                
                // Remove highlight after scroll completes
                setTimeout(() => {
                    targetElement.classList.remove('highlight-target');
                }, 2000);
            }
        });
    });
    
    // Fixed input validation for numeric inputs
    const validateNumericInputs = () => {
        const numericInputs = document.querySelectorAll('input[type="text"][inputmode="numeric"]');
        
        if (numericInputs.length > 0) {
            numericInputs.forEach(input => {
                // Only validate on blur to allow normal typing
                input.addEventListener('blur', function() {
                    const value = parseInt(this.value);
                    const min = parseInt(this.getAttribute('min') || input.closest('.form-group, .mb-3').querySelector('small')?.textContent?.match(/\d+/g)?.[0] || 0);
                    const max = parseInt(this.getAttribute('max') || input.closest('.form-group, .mb-3').querySelector('small')?.textContent?.match(/\d+/g)?.[1] || 9999);
                    
                    if (isNaN(value) || value < min || value > max) {
                        this.classList.add('is-invalid');
                        const feedbackElement = this.nextElementSibling;
                        if (!feedbackElement || !feedbackElement.classList.contains('invalid-feedback')) {
                            const feedback = document.createElement('div');
                            feedback.classList.add('invalid-feedback');
                            feedback.textContent = `Please enter a value between ${min} and ${max}`;
                            this.parentNode.insertBefore(feedback, this.nextSibling);
                        }
                    } else {
                        this.classList.remove('is-invalid');
                        const feedbackElement = this.nextElementSibling;
                        if (feedbackElement && feedbackElement.classList.contains('invalid-feedback')) {
                            feedbackElement.remove();
                        }
                    }
                });
            });
        }
    };
    
    validateNumericInputs();
    
    // Enhanced animation on scroll
    const animateElements = document.querySelectorAll('.animate-on-scroll');
    
    function checkIfInView() {
        const windowHeight = window.innerHeight;
        const windowTopPosition = window.scrollY;
        const windowBottomPosition = windowTopPosition + windowHeight;
        
        animateElements.forEach(element => {
            const elementHeight = element.offsetHeight;
            const elementTopPosition = element.offsetTop;
            const elementBottomPosition = elementTopPosition + elementHeight;
            
            // Calculate the delay based on data attribute
            const delay = element.getAttribute('data-delay') || 0;
            
            // Check if element is in view
            if (elementBottomPosition >= windowTopPosition && elementTopPosition <= windowBottomPosition) {
                setTimeout(() => {
                    element.classList.add('visible');
                    
                    // Add additional animations based on data attributes
                    if (element.hasAttribute('data-animation')) {
                        const animation = element.getAttribute('data-animation');
                        element.classList.add(animation);
                    } else {
                        // Default animation
                        element.style.opacity = 1;
                        element.style.transform = 'translateY(0)';
                    }
                }, delay);
            }
        });
    }
    
    // Apply initial styles to animate-on-scroll elements
    animateElements.forEach(element => {
        // Skip elements that already have custom animations defined
        if (!element.hasAttribute('data-animation')) {
            element.style.opacity = 0;
            element.style.transform = 'translateY(20px)';
            element.style.transition = 'opacity 0.8s ease-out, transform 0.8s ease-out';
        }
    });
    
    // Add particle effects to hero section
    const heroSection = document.querySelector('.hero-section');
    if (heroSection && !document.querySelector('.particles-container')) {
        const particles = document.createElement('div');
        particles.classList.add('particles-container');
        
        // Create random particles
        for (let i = 0; i < 5; i++) {
            const particle = document.createElement('div');
            particle.classList.add('particle');
            particle.style.width = `${Math.random() * 20 + 5}px`;
            particle.style.height = particle.style.width;
            particle.style.top = `${Math.random() * 100}%`;
            particle.style.left = `${Math.random() * 100}%`;
            particle.style.animationDelay = `${Math.random() * 2}s`;
            particle.style.animationDuration = `${Math.random() * 10 + 10}s`;
            particles.appendChild(particle);
        }
        
        heroSection.prepend(particles);
    }
    
    // Handle feature card clicks
    const featureCards = document.querySelectorAll('.feature-card');
    if (featureCards) {
        featureCards.forEach(card => {
            card.addEventListener('click', function() {
                const link = this.getAttribute('data-href') || this.querySelector('a')?.getAttribute('href');
                if (link) {
                    window.location.href = link;
                }
            });
        });
    }
    
    // Add animation delay to cards
    const animatedCards = document.querySelectorAll('.card, .step-card');
    if (animatedCards) {
        animatedCards.forEach((card, index) => {
            const delay = index % 4;
            card.classList.add(`delay-${delay + 1}`);
            // Add a subtle hover effect
            card.addEventListener('mouseenter', function() {
                this.style.transform = 'translateY(-10px)';
                this.style.boxShadow = '0 15px 30px rgba(0,0,0,0.1)';
            });
            card.addEventListener('mouseleave', function() {
                this.style.transform = '';
                this.style.boxShadow = '';
            });
        });
    }
    
    // Auto-hide alerts after 5 seconds
    const alerts = document.querySelectorAll('.alert:not(.alert-permanent)');
    if (alerts) {
        alerts.forEach(alert => {
            setTimeout(() => {
                alert.classList.add('fade-out');
                setTimeout(() => {
                    alert.remove();
                }, 500);
            }, 5000);
        });
    }
    
    // Enhance form submissions with loading indicators
    const forms = document.querySelectorAll('form');
    forms.forEach(form => {
        form.addEventListener('submit', function() {
            // Removed button loading animation code
            // This prevents the submit button from showing a loading state
        });
    });
    
    // Code Quality Classification AJAX
    const codeQualityForm = document.getElementById('codeQualityForm');
    
    if (codeQualityForm) {
        codeQualityForm.addEventListener('submit', function(e) {
            e.preventDefault();
            
            // Get input values
            const loc = document.getElementById('lines_of_code').value;
            const complexity = document.getElementById('complexity_score').value;
            const execTime = document.getElementById('execution_time').value;
            const defectDensity = document.getElementById('defect_density').value;
            
            // Validate inputs
            if (!loc || !complexity || !execTime || !defectDensity) {
                alert('Please fill in all fields');
                return;
            }
            
            // Show loading indicator
            document.getElementById('loadingSection').style.display = 'block';
            document.getElementById('resultsSection').style.display = 'none';
            
            // Create form data
            const formData = new FormData();
            formData.append('lines_of_code', loc);
            formData.append('complexity_score', complexity);
            formData.append('execution_time', execTime);
            formData.append('defect_density', defectDensity);
            
            // Send AJAX request
            fetch('/predict_code_quality', {
                method: 'POST',
                body: formData
            })
            .then(response => response.json())
            .then(data => {
                if (data.success) {
                    // Process the results
                    const qualityLabel = data.quality_label;
                    const qualityCode = data.quality_code;
                    const probabilities = data.probabilities;
                    
                    // Update UI based on quality
                    let iconClass, description, defectRisk, performance, maintainability, suggestions;
                    
                    if (qualityLabel === 'High') {
                        iconClass = 'fa-award text-success';
                        description = 'Excellent code with minimal defects and good performance.';
                        defectRisk = 'Low';
                        performance = 'Good';
                        maintainability = 'High';
                        suggestions = [
                            'Continue with current development practices',
                            'Consider peer reviews to maintain quality',
                            'Document your successful practices for team knowledge sharing'
                        ];
                    } else if (qualityLabel === 'Medium') {
                        iconClass = 'fa-star text-warning';
                        description = 'Average code quality with moderate defect density and acceptable performance.';
                        defectRisk = 'Moderate';
                        performance = 'Average';
                        maintainability = 'Medium';
                        suggestions = [
                            'Focus on reducing complexity in key functions',
                            'Add more test cases to improve coverage',
                            'Review code documentation for clarity',
                            'Consider performance optimizations in critical paths'
                        ];
                    } else {
                        iconClass = 'fa-exclamation-triangle text-danger';
                        description = 'Poor code quality with high defect density and performance issues.';
                        defectRisk = 'High';
                        performance = 'Poor';
                        maintainability = 'Low';
                        suggestions = [
                            'Refactor complex parts of the code',
                            'Implement more unit tests to catch bugs early',
                            'Review algorithm efficiency to improve execution time',
                            'Consider code review sessions to identify improvement areas'
                        ];
                    }
                    
                    // Update the UI with animation
                    const qualityIcon = document.getElementById('qualityIcon');
                    qualityIcon.className = 'fas ' + iconClass + ' fa-4x';
                    
                    // Add entrance animation
                    qualityIcon.style.animation = 'bounceIn 0.8s';
                    
                    document.getElementById('qualityResult').textContent = qualityLabel + ' Quality';
                    document.getElementById('qualityDescription').textContent = description;
                    document.getElementById('defectRisk').textContent = defectRisk;
                    document.getElementById('performanceLevel').textContent = performance;
                    document.getElementById('maintainabilityLevel').textContent = maintainability;
                    
                    // Update suggestions list with staggered animation
                    const suggestionsList = document.getElementById('suggestionsList');
                    suggestionsList.innerHTML = '';
                    
                    suggestions.forEach((suggestion, index) => {
                        const li = document.createElement('li');
                        li.className = 'list-group-item';
                        li.style.opacity = '0';
                        li.style.transform = 'translateY(20px)';
                        li.style.transition = 'opacity 0.5s ease, transform 0.5s ease';
                        li.style.transitionDelay = `${index * 0.1}s`;
                        li.innerHTML = '<i class="fas fa-check-circle text-success me-2"></i>' + suggestion;
                        suggestionsList.appendChild(li);
                        
                        // Trigger animation
                        setTimeout(() => {
                            li.style.opacity = '1';
                            li.style.transform = 'translateY(0)';
                        }, 100);
                    });
                    
                    // Hide loading and show results with fade-in animation
                    document.getElementById('loadingSection').style.display = 'none';
                    
                    const resultsSection = document.getElementById('resultsSection');
                    resultsSection.style.display = 'block';
                    resultsSection.style.opacity = '0';
                    
                    setTimeout(() => {
                        resultsSection.style.transition = 'opacity 0.5s ease';
                        resultsSection.style.opacity = '1';
                    }, 50);
                    
                    // Scroll to results
                    document.getElementById('resultsSection').scrollIntoView({ behavior: 'smooth' });
                } else {
                    // Handle error
                    alert('Error: ' + (data.error || 'Failed to classify code quality'));
                    document.getElementById('loadingSection').style.display = 'none';
                }
            })
            .catch(error => {
                console.error('Error:', error);
                alert('An error occurred. Please try again.');
                document.getElementById('loadingSection').style.display = 'none';
            });
        });
    }
    
    // Run once to initialize animations
    checkIfInView();
    
    // Run on scroll
    window.addEventListener('scroll', checkIfInView);
    
    // Typed.js implementation for hero section if available
    if (typeof Typed !== 'undefined' && document.querySelector('.typed-text')) {
        new Typed('.typed-text', {
            strings: ['Predict Bugs Before They Happen', 
                      'Classify Critical Issues', 
                      'Assess Code Quality with KNN',
                      'Improve Software Quality'],
            typeSpeed: 50,
            backSpeed: 30,
            backDelay: 2000,
            loop: true,
            cursorChar: '|',
            fadeOut: true
        });
    }

    // Global loading state management
    window.handleLoadingState = function(isLoading, loadingElement, resultElement) {
        if (isLoading) {
            // Show loading indicator, hide result
            if (loadingElement) {
                if (loadingElement.id === 'loadingSection') {
                    loadingElement.style.display = 'block';
                } else {
                    loadingElement.classList.remove('d-none');
                }
            }
            
            if (resultElement) {
                if (resultElement.id === 'resultsSection') {
                    resultElement.style.display = 'none';
                } else {
                    resultElement.classList.add('d-none');
                }
            }
        } else {
            // Hide loading indicator
            if (loadingElement) {
                if (loadingElement.id === 'loadingSection') {
                    loadingElement.style.display = 'none';
                } else {
                    loadingElement.classList.add('d-none');
                }
            }
            
            // Note: We don't show the result element here because that should be handled
            // after processing the response data
        }
    };

    // Override fetch to ensure loading states are managed properly
    const originalFetch = window.fetch;
    window.fetch = function() {
        const fetchPromise = originalFetch.apply(this, arguments);
        
        // Make sure all fetch requests have proper error handling
        return fetchPromise.catch(error => {
            console.error('Fetch error:', error);
            
            // Hide any visible loading indicators
            document.querySelectorAll('[id*="loading"]').forEach(el => {
                if (el.id === 'loadingSection' && getComputedStyle(el).display !== 'none') {
                    el.style.display = 'none';
                } else if (!el.classList.contains('d-none') && getComputedStyle(el).display !== 'none') {
                    el.classList.add('d-none');
                }
            });
            
            // Display a general error alert
            setTimeout(() => alert('An error occurred. Please try again.'), 100);
            
            throw error;
        });
    };
}); 