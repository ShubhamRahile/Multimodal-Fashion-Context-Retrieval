// Front-end Interactions for Glance Fashion Retrieval System

document.addEventListener("DOMContentLoaded", function() {
    const searchForm = document.getElementById("search-form");
    const queryInput = document.getElementById("query-input");
    const searchBtn = document.getElementById("search-btn");
    const searchSpinner = document.getElementById("search-spinner");
    const loadingSection = document.getElementById("loading-section");
    const resultsSection = document.getElementById("results-section");

    if (searchForm) {
        searchForm.addEventListener("submit", function(e) {
            // Disable search button & show spinner
            if (searchBtn) {
                searchBtn.disabled = true;
            }
            if (searchSpinner) {
                searchSpinner.classList.remove("d-none");
            }

            // Hide results if they are currently displayed
            if (resultsSection) {
                resultsSection.classList.add("d-none");
            }

            // Show loading section
            if (loadingSection) {
                loadingSection.classList.remove("d-none");
                // Scroll to loading section smoothly
                loadingSection.scrollIntoView({ behavior: 'smooth', block: 'center' });
            }
        });
    }

    // Trigger loading state for suggestion tags
    const suggestionTags = document.querySelectorAll(".suggestion-tag");
    suggestionTags.forEach(tag => {
        tag.addEventListener("click", function() {
            // Populate search bar
            const query = this.textContent.trim().replace(/^"|"$/g, '');
            if (queryInput) {
                queryInput.value = query;
            }
            // Show loading section
            if (loadingSection) {
                loadingSection.classList.remove("d-none");
            }
            if (resultsSection) {
                resultsSection.classList.add("d-none");
            }
        });
    });
});
