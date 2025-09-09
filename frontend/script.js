class ClothingAnalyzer {
    constructor() {
        this.imageInput = document.getElementById('imageInput');
        this.uploadArea = document.getElementById('uploadArea');
        this.analyzeBtn = document.getElementById('analyzeBtn');
        this.previewSection = document.getElementById('previewSection');
        this.imagePreview = document.getElementById('imagePreview');
        this.loading = document.getElementById('loading');
        this.results = document.getElementById('results');
        
        this.initializeEventListeners();
    }
    
    initializeEventListeners() {
        // Upload area click
        this.uploadArea.addEventListener('click', () => {
            this.imageInput.click();
        });
        
        // File input change
        this.imageInput.addEventListener('change', (e) => {
            this.handleFileSelect(e.target.files[0]);
        });
        
        // Drag and drop
        this.uploadArea.addEventListener('dragover', (e) => {
            e.preventDefault();
            this.uploadArea.classList.add('dragover');
        });
        
        this.uploadArea.addEventListener('dragleave', () => {
            this.uploadArea.classList.remove('dragover');
        });
        
        this.uploadArea.addEventListener('drop', (e) => {
            e.preventDefault();
            this.uploadArea.classList.remove('dragover');
            this.handleFileSelect(e.dataTransfer.files[0]);
        });
        
        // Analyze button
        this.analyzeBtn.addEventListener('click', () => {
            this.analyzeImage();
        });
    }
    
    handleFileSelect(file) {
        if (!file) return;
        
        // Validate file type
        if (!file.type.startsWith('image/')) {
            alert('Please select a valid image file');
            return;
        }
        
        // Show preview
        const reader = new FileReader();
        reader.onload = (e) => {
            this.imagePreview.src = e.target.result;
            this.previewSection.style.display = 'block';
            this.analyzeBtn.disabled = false;
            this.results.style.display = 'none';
        };
        reader.readAsDataURL(file);
        
        // Store file for analysis
        this.selectedFile = file;
    }
    
    async analyzeImage() {
        if (!this.selectedFile) return;
        
        // Show loading
        this.loading.style.display = 'block';
        this.results.style.display = 'none';
        this.analyzeBtn.disabled = true;
        
        try {
            // Create form data
            const formData = new FormData();
            formData.append('file', this.selectedFile);
            
            // Send request
            const response = await fetch('/analyze', {
                method: 'POST',
                body: formData
            });
            
            if (!response.ok) {
                throw new Error(`HTTP error! status: ${response.status}`);
            }
            
            const data = await response.json();
            this.displayResults(data);
            
        } catch (error) {
            console.error('Analysis failed:', error);
            alert('Analysis failed. Please try again.');
        } finally {
            this.loading.style.display = 'none';
            this.analyzeBtn.disabled = false;
        }
    }
    
    displayResults(data) {
        // Display basic attributes
        document.getElementById('styleResult').textContent = data.style_classification;
        document.getElementById('formalityResult').textContent = data.formality;
        document.getElementById('textureResult').textContent = data.texture;
        
        // Display detected items
        const itemsHtml = data.clothing_items
            .map(item => `<div class="item-tag">${item.item_type} (${Math.round(item.confidence * 100)}%)</div>`)
            .join('');
        document.getElementById('itemsResult').innerHTML = itemsHtml || 'No items detected';
        
        // Display color palette
        this.displayColorPalette(data.dominant_colors);
        
        // Display confidence scores
        this.displayConfidenceScores(data.confidence_scores);
        
        // Show results
        this.results.style.display = 'block';
    }
    
    displayColorPalette(colors) {
        const colorPalette = document.getElementById('colorPalette');
        
        const colorHtml = colors.map(color => `
            <div class="color-item">
                <div class="color-swatch" style="background-color: ${color.hex}"></div>
                <div>
                    <div style="font-weight: 600">${color.color_name}</div>
                    <div style="font-size: 0.9rem; color: #64748b">${color.percentage}%</div>
                </div>
            </div>
        `).join('');
        
        colorPalette.innerHTML = colorHtml;
    }
    
    displayConfidenceScores(scores) {
        const confidenceContainer = document.getElementById('confidenceScores');
        
        const scoresHtml = Object.entries(scores).map(([key, value]) => `
            <div style="margin-bottom: 15px">
                <div style="display: flex; justify-content: space-between; margin-bottom: 5px">
                    <span style="text-transform: capitalize">${key}</span>
                    <span>${Math.round(value * 100)}%</span>
                </div>
                <div class="confidence-bar">
                    <div class="confidence-fill" style="width: ${value * 100}%"></div>
                </div>
            </div>
        `).join('');
        
        confidenceContainer.innerHTML = scoresHtml;
    }
}

// Initialize the application
document.addEventListener('DOMContentLoaded', () => {
    new ClothingAnalyzer();
});
