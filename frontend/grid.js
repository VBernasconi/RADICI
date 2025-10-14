
const sidebar = document.getElementById('sidebar_metamotor');
const mainContent = document.getElementById('mainContent');

function parseDateRange(dateStr) {
    if (!dateStr) return null;
  
    // Remove whitespace
    dateStr = dateStr.trim();
  
    // Handle different formats
    if (/^\d{4}$/.test(dateStr)) {
      // Format: "YYYY"
      const year = parseInt(dateStr, 10);
      return { start: year, end: year };
    } else if (/^\d{4}[-/]\d{4}$/.test(dateStr)) {
      // Format: "YYYY-YYYY" or "YYYY/YYYY"
      const parts = dateStr.split(/[-/]/);
      const startYear = parseInt(parts[0], 10);
      const endYear = parseInt(parts[1], 10);
      return { start: startYear, end: endYear };
    } else {
      // Unrecognized format
      return null;
    }
}

function displayFeatures(features) {
    // Clear existing items
    document.querySelectorAll('.feature-item').forEach(item => item.remove());
  
    const columns = [
      document.getElementById('col-1'),
      document.getElementById('col-2'),
      document.getElementById('col-3'),
      document.getElementById('col-4')
    ];
  
    features.forEach((feature, index) => {
      const colIndex = index % columns.length;
      const col = columns[colIndex];
      const img_path = feature.properties.img_path;
      const parts = img_path.split("/");
      const img_file = parts[parts.length - 1];
  
      const itemDiv = document.createElement('div');
      itemDiv.className = 'feature-item';
  
      itemDiv.innerHTML = `
        <p class="grid-item-title">${feature.properties.title}</p>
        <img src="${serverURL}/images/${img_file}" alt="${feature.properties.title}" style="cursor:pointer;" onclick="triggerActions('${feature.properties.id}')"> 
        <div class="grid-item-description">
          <p>${feature.properties.description}</p>
        </div>
      `;
  
      col.appendChild(itemDiv);
    });
  }

//display sidebar on image click
function triggerActions(id) {
    document.getElementById('sidebar_metamotor').style.right = '0';
    document.querySelector('.mainContent').classList.add('blur-background');
    res = getObjectById(id);
    //displayResults(res);
    
    searchSimilarObjects(id);
}

// Function to get 10 random features
function getRandomFeatures(features, count = 10) {
    const shuffled = features.slice().sort(() => 0.5 - Math.random());
    return shuffled.slice(0, count);
}
    
/*function updateFeaturesByFilter(filterType, filterValue) {
    if (!geojsonData) return;
  
    let filteredFeatures = geojsonData.features;
  
    // Apply filters based on filterType
    if (filterType === 'type') {
      filteredFeatures = filteredFeatures.filter(f => f.properties.type === filterValue);
    } else if (filterType === 'date') {
      filteredFeatures = filteredFeatures.filter(f => {
        const dateRangeStr = f.properties.date;
        const dateRange = parseDateRange(dateRangeStr);
        if (!dateRange) return false;
        return (
        dateRange.start >= filterValue.start &&
        dateRange.end <= filterValue.end
        );
      });
    } else if (filterType === 'fondo') {
      filteredFeatures = filteredFeatures.filter(f => f.properties.fondo === filterValue);
    }
  
    // Update display
    displayFeatures(filteredFeatures);
}

function applyFilter(button) {
    const filterType = button.dataset.filterType;
    let filterValue = button.dataset.filterValue;
  
    // Parse JSON for date filter
    if (filterType === 'date') {
      filterValue = JSON.parse(filterValue);
      console.log(filterValue);
    }

    updateFeaturesByFilter(filterType, filterValue);
}
*/
fetch('redis_export.geojson')
  .then(response => response.json())
  .then(geojsonData => {
        // Select 10 random features
        const selectedFeatures = getRandomFeatures(geojsonData.features, 50);

        // Distribute features evenly across 4 columns
        const columns = [
            document.getElementById('col-1'),
            document.getElementById('col-2'),
            document.getElementById('col-3'),
            document.getElementById('col-4')
        ];

        selectedFeatures.forEach((feature, index) => {
            const colIndex = index % columns.length;
            const col = columns[colIndex];
            const img_path = feature.properties.img_path;
            // Split by "/" and get the last part
            const parts = img_path.split("/");
            const img_file = parts[parts.length - 1]; 
            // Create an element for each feature
            const itemDiv = document.createElement('div');
            itemDiv.className = 'feature-item';

            // Example content: an image and some info
            itemDiv.innerHTML = `
                <p class="grid-item-title">${feature.properties.title}</p>
                <img src="${serverURL}/images/${img_file}" alt="${feature.properties.title}" style="cursor:pointer;" onclick="triggerActions('${feature.properties.id}')"> 
                <div class="grid-item-description">
                    <p>${feature.properties.description}</p>
                </div>
            `;

            col.appendChild(itemDiv);
        });
    })
    .catch(error => console.error('Error loading GeoJSON:', error));