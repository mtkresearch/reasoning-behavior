"""
HTML/CSS/JavaScript Templates for Attention Visualization

This module contains all the HTML, CSS, and JavaScript templates used by
the attention visualization tool, separated from the main logic for better
maintainability and reusability.
"""

# =============================================================================
# CSS Template
# =============================================================================

CSS_TEMPLATE = """
body {
    font-family: Arial, sans-serif;
    margin: 20px;
    background-color: #f5f5f5;
}

h1 {
    color: #333;
}

.controls {
    margin: 20px 0;
    padding: 15px;
    padding-bottom: 35px;
    background-color: white;
    border-radius: 5px;
    box-shadow: 0 2px 4px rgba(0,0,0,0.1);
}

.controls label {
    margin-right: 10px;
    font-weight: bold;
}

.controls select {
    margin-right: 20px;
    padding: 5px 10px;
    font-size: 14px;
    border: 1px solid #ddd;
    border-radius: 3px;
}

.layer-control {
    display: inline-block;
    margin-left: 20px;
}

.slider-container {
    position: relative;
    width: 900px;
    margin-top: 10px;
    height: 45px;
}

#layer-slider {
    width: 100%;
    margin: 0;
    position: relative;
    z-index: 2;
}

#layer-value {
    font-weight: bold;
    color: #333;
    margin-left: 10px;
    font-size: 16px;
}

#slider-ticks {
    position: absolute;
    top: 25px;
    left: 0;
    width: 100%;
    height: 30px;
    display: flex;
    justify-content: space-between;
    pointer-events: none;
}

.tick {
    position: relative;
    width: 2px;
    height: 8px;
    background-color: #666;
}

.tick-label {
    position: absolute;
    top: 10px;
    left: 50%;
    transform: translateX(-50%);
    font-size: 11px;
    color: #666;
    white-space: nowrap;
}

#metadata {
    margin: 20px 0;
    padding: 15px;
    background-color: white;
    border-radius: 5px;
    box-shadow: 0 2px 4px rgba(0,0,0,0.1);
}

#metadata p {
    margin: 8px 0;
}

#tokens-container {
    margin: 20px 0;
    padding: 15px;
    background-color: white;
    border-radius: 5px;
    box-shadow: 0 2px 4px rgba(0,0,0,0.1);
    line-height: 2;
}

.token {
    display: inline-block;
    padding: 2px 4px;
    margin: 1px;
    border-radius: 2px;
    font-family: monospace;
    font-size: 7px;
    white-space: pre-wrap;
    word-break: break-word;
}

.incorrect-message {
    color: #d32f2f;
    font-size: 18px;
    font-weight: bold;
    padding: 20px;
    text-align: center;
}
"""


# =============================================================================
# JavaScript Template
# =============================================================================

JAVASCRIPT_TEMPLATE = """
// Populate instance select options
function populateInstanceSelect() {{
    const select = document.getElementById('instance-select');
    select.innerHTML = '';
    instances.forEach((instance, i) => {{
        const status = instance.is_correct ? 'CORRECT' : 'INCORRECT';
        const option = document.createElement('option');
        option.value = i;
        option.textContent = `Instance ${{i}} (${{status}})`;
        select.appendChild(option);
    }});
}}

function updateVisualization() {{
    const instanceIdx = parseInt(document.getElementById('instance-select').value);
    const instance = instances[instanceIdx];

    // Update metadata
    document.getElementById('question').textContent = instance.question;
    document.getElementById('ground-truth').textContent = instance.ground_truth;
    document.getElementById('is-correct').textContent = instance.is_correct ? 'Yes' : 'No';

    // Update layer slider if needed
    const layerSlider = document.getElementById('layer-slider');
    if (instance.is_correct && instance.attention_maps.length > 0) {{
        const numLayers = instance.attention_maps.length;
        layerSlider.max = numLayers - 1;
        layerSlider.value = Math.min(parseInt(layerSlider.value), numLayers - 1);
        document.getElementById('layer-value').textContent = `Layer ${{layerSlider.value}}`;

        // Update slider ticks
        updateSliderTicks(numLayers);
    }}

    // Check if correct
    if (!instance.is_correct) {{
        document.getElementById('tokens-container').innerHTML =
            '<div class="incorrect-message">INCORRECT - No visualization available</div>';
        return;
    }}

    // Get selected layer
    const layerIdx = parseInt(layerSlider.value);
    const attentionWeights = instance.attention_maps[layerIdx];
    const tokens = instance.tokens;

    // Find min and max for normalization
    const minWeight = Math.min(...attentionWeights);
    const maxWeight = Math.max(...attentionWeights);

    // Generate token HTML
    const tokensHtml = tokens.map((token, i) => {{
        const weight = attentionWeights[i];
        const normalized = (weight - minWeight) / (maxWeight - minWeight);

        // Use viridis colormap (simplified)
        const color = getColor(normalized);

        return `<span class="token" style="background-color: ${{color}}" title="Weight: ${{weight.toFixed(4)}}\">${{escapeHtml(token)}}</span>`;
    }}).join('');

    document.getElementById('tokens-container').innerHTML = tokensHtml;
}}

function updateSliderTicks(numLayers) {{
    const ticksContainer = document.getElementById('slider-ticks');
    ticksContainer.innerHTML = '';

    // Create tick marks for each layer
    for (let i = 0; i < numLayers; i++) {{
        const tick = document.createElement('div');
        tick.className = 'tick';

        const label = document.createElement('div');
        label.className = 'tick-label';
        label.textContent = i;

        tick.appendChild(label);
        ticksContainer.appendChild(tick);
    }}
}}

function getColor(value) {{
    // White to Red colormap
    // value in [0, 1], 0 = white (255,255,255), 1 = red (255,0,0)
    const r = 255;
    const g = Math.floor(255 * (1 - value));
    const b = Math.floor(255 * (1 - value));
    return `rgb(${{r}}, ${{g}}, ${{b}})`;
}}

function escapeHtml(text) {{
    const div = document.createElement('div');
    div.textContent = text;
    return div.innerHTML;
}}

// Event listeners
document.getElementById('instance-select').addEventListener('change', () => {{
    // Reset layer slider when instance changes
    const layerSlider = document.getElementById('layer-slider');
    layerSlider.value = 0;
    updateVisualization();
}});

document.getElementById('layer-slider').addEventListener('input', () => {{
    const layerIdx = document.getElementById('layer-slider').value;
    document.getElementById('layer-value').textContent = `Layer ${{layerIdx}}`;
    updateVisualization();
}});

// Keyboard navigation
document.addEventListener('keydown', (event) => {{
    const instanceSelect = document.getElementById('instance-select');
    const layerSlider = document.getElementById('layer-slider');

    switch(event.key) {{
        case 'ArrowLeft':
            // Decrease layer
            event.preventDefault();
            if (parseInt(layerSlider.value) > parseInt(layerSlider.min)) {{
                layerSlider.value = parseInt(layerSlider.value) - 1;
                document.getElementById('layer-value').textContent = `Layer ${{layerSlider.value}}`;
                updateVisualization();
            }}
            break;

        case 'ArrowRight':
            // Increase layer
            event.preventDefault();
            if (parseInt(layerSlider.value) < parseInt(layerSlider.max)) {{
                layerSlider.value = parseInt(layerSlider.value) + 1;
                document.getElementById('layer-value').textContent = `Layer ${{layerSlider.value}}`;
                updateVisualization();
            }}
            break;

        case 'ArrowUp':
            // Previous instance
            event.preventDefault();
            if (instanceSelect.selectedIndex > 0) {{
                instanceSelect.selectedIndex--;
                layerSlider.value = 0;
                updateVisualization();
            }}
            break;

        case 'ArrowDown':
            // Next instance
            event.preventDefault();
            if (instanceSelect.selectedIndex < instanceSelect.options.length - 1) {{
                instanceSelect.selectedIndex++;
                layerSlider.value = 0;
                updateVisualization();
            }}
            break;
    }}
}});

// Initial render
populateInstanceSelect();
updateVisualization();
"""


# =============================================================================
# HTML Template Components
# =============================================================================

HTML_HEADER = """<!DOCTYPE html>
<html>
<head>
    <meta charset="UTF-8">
    <title>Attention Visualization</title>
    <style>
        {css}
    </style>
</head>
<body>
    <h1>Attention Visualization</h1>

    <div class="controls">
        <label>Instance:</label>
        <select id="instance-select">
            <!-- Options will be generated by JavaScript -->
        </select>

        <div class="layer-control">
            <label>Layer:</label>
            <span id="layer-value">Layer 0</span>
            <div class="slider-container">
                <input type="range" id="layer-slider" min="0" max="0" value="0" step="1" />
                <div id="slider-ticks"></div>
            </div>
        </div>
    </div>

    <div id="metadata">
        <p><strong>Question:</strong> <span id="question"></span></p>
        <p><strong>Ground Truth:</strong> <span id="ground-truth"></span></p>
        <p><strong>Correct:</strong> <span id="is-correct"></span></p>
    </div>

    <div id="tokens-container">
        <!-- Tokens will be populated by JavaScript -->
    </div>

    <script>
"""

HTML_FOOTER = """
    </script>
</body>
</html>"""


# =============================================================================
# HTML Template for Legacy Batch Mode
# =============================================================================

HTML_TEMPLATE_LEGACY = """<!DOCTYPE html>
<html>
<head>
    <meta charset="UTF-8">
    <title>Attention Visualization</title>
    <style>
        {css}
    </style>
</head>
<body>
    <h1>Attention Visualization</h1>

    <div class="controls">
        <label>Instance:</label>
        <select id="instance-select">
            {instance_options}
        </select>

        <div class="layer-control">
            <label>Layer:</label>
            <span id="layer-value">Layer 0</span>
            <div class="slider-container">
                <input type="range" id="layer-slider" min="0" max="0" value="0" step="1" />
                <div id="slider-ticks"></div>
            </div>
        </div>
    </div>

    <div id="metadata">
        <p><strong>Question:</strong> <span id="question"></span></p>
        <p><strong>Ground Truth:</strong> <span id="ground-truth"></span></p>
        <p><strong>Correct:</strong> <span id="is-correct"></span></p>
    </div>

    <div id="tokens-container">
        <!-- Tokens will be populated by JavaScript -->
    </div>

    <script>
        const instances = {instances_json};

        {javascript}
    </script>
</body>
</html>"""


# =============================================================================
# Helper Functions
# =============================================================================

def get_html_header() -> str:
    """Get HTML header with embedded CSS"""
    return HTML_HEADER.format(css=CSS_TEMPLATE)


def get_html_footer() -> str:
    """Get HTML footer"""
    return HTML_FOOTER


def get_javascript_code() -> str:
    """Get JavaScript code with functions and event listeners"""
    return JAVASCRIPT_TEMPLATE


def build_complete_html(instances_json: str, instance_options: str) -> str:
    """
    Build complete HTML for legacy batch mode

    Args:
        instances_json: JSON string of all instances
        instance_options: HTML string of instance select options

    Returns:
        Complete HTML string
    """
    return HTML_TEMPLATE_LEGACY.format(
        css=CSS_TEMPLATE,
        instance_options=instance_options,
        instances_json=instances_json,
        javascript=JAVASCRIPT_TEMPLATE
    )
