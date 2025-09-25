// List of attributes and years prepared as objects from assets
var assets = {"1994": {"AET (Annual Evapotranspiration)": AET_1994, "Annual Precipitation": Precip_1994, "ANPP (Aboveground Net Primary Productivity)": ANPP_1994,
              "ANPP StdErr": ANPP_SD_1994, "BNPP (Belowground Net Primary Productivity)": BNPP_1994, "BNPP StdErr": BNPP_SD_1994, "NEP (Net Ecoystem Productivity)": NEP_1994, 
              "NEP StdErr": NEP_SD_1994, "Rh (Respiration)": Rh_1994, "Rh StdErr": Rh_SD_1994},
              "2019": {"AET (Annual Evapotranspiration)": AET_2019, "Annual Precipitation": Precip_2019, "ANPP (Aboveground Net Primary Productivity)": ANPP_2019,
              "ANPP StdErr": ANPP_SD_2019, "BNPP (Belowground Net Primary Productivity)": BNPP_2019, "BNPP StdErr": BNPP_SD_2019, "NEP (Net Ecoystem Productivity)": NEP_2019, 
              "NEP StdErr": NEP_SD_2019, "Rh (Respiration)": Rh_2019, "Rh StdErr": Rh_SD_2019},
              "2019_new": {"AET (Annual Evapotranspiration)": AET_2019_, "Annual Precipitation": Precip_2019_, "ANPP (Aboveground Net Primary Productivity)": ANPP_2019_,
              "ANPP StdErr": ANPP_SD_2019_, "BNPP (Belowground Net Primary Productivity)": BNPP_2019_, "BNPP StdErr": BNPP_SD_2019_, "NEP (Net Ecoystem Productivity)": NEP_2019_, 
              "NEP StdErr": NEP_SD_2019_, "Rh (Respiration)": Rh_2019_, "Rh StdErr": Rh_SD_2019_, "Root Biomass": Root_2019_, "Active Growth Days": AGD_2019_,
              "Root Exudates": Exudates_2019_, "Root Turnover": Turnover_2019_, "Snow Rh Flux": Snow_Flux_2019_, "Aboveground Biomass": Herb_2019_},
              "2021": {"AET (Annual Evapotranspiration)": AET_2021, "Annual Precipitation": Precip_2021, "ANPP (Aboveground Net Primary Productivity)": ANPP_2021,
              "ANPP StdErr": ANPP_SD_2021, "BNPP (Belowground Net Primary Productivity)": BNPP_2021, "BNPP StdErr": BNPP_SD_2021, "NEP (Net Ecoystem Productivity)": NEP_2021, 
              "NEP StdErr": NEP_SD_2021, "Rh (Respiration)": Rh_2021, "Rh StdErr": Rh_SD_2021},
              "2021_new": {"AET (Annual Evapotranspiration)": AET_2021_, "Annual Precipitation": Precip_2021_, "ANPP (Aboveground Net Primary Productivity)": ANPP_2021_,
              "ANPP StdErr": ANPP_SD_2021_, "BNPP (Belowground Net Primary Productivity)": BNPP_2021_, "BNPP StdErr": BNPP_2021_, "NEP (Net Ecoystem Productivity)": NEP_2021_, 
              "NEP StdErr": NEP_SD_2021_, "Rh (Respiration)": Rh_2021_, "Rh StdErr": Rh_SD_2021_, "Root Biomass": Root_2021_, "Active Growth Days": AGD_2021_,
              "Root Exudates": Exudates_2021_, "Root Turnover": Turnover_2021_, "Snow Rh Flux": Snow_Flux_2021_, "Aboveground Biomass": Herb_2021_},
              "2023": {"AET (Annual Evapotranspiration)": AET_2023_, "Annual Precipitation": Precip_2023_, "ANPP (Aboveground Net Primary Productivity)": ANPP_2023_,
              "ANPP StdErr": ANPP_SD_2023_, "BNPP (Belowground Net Primary Productivity)": BNPP_2023_, "BNPP StdErr": BNPP_2023_, "NEP (Net Ecoystem Productivity)": NEP_2023_, 
              "NEP StdErr": NEP_SD_2023_, "Rh (Respiration)": Rh_2023_, "Rh StdErr": Rh_SD_2023_, "Root Biomass": Root_2023_, "Active Growth Days": AGD_2023_,
              "Root Exudates": Exudates_2023_, "Root Turnover": Turnover_2023_, "Snow Rh Flux": Snow_Flux_2023_, "Aboveground Biomass": Herb_2023_}
};

// extract years and create the selection panel
var years = Object.keys(assets);
var yearSelect = ui.Select({items: years, value: years[0], style: {margin: '0 0 6px 0', width: '100px', height: '40px'} });
var selectPanel = ui.Panel({style: {position: 'top-left', padding: '8px'}});
selectPanel.add(ui.Label('Select Year/Attribute to Display', {fontWeight: 'bold'}));
selectPanel.add(yearSelect);

// initialize checkbox and add it to the selection panel for attributes
var checkboxPanel = ui.Panel();
selectPanel.add(checkboxPanel);
Map.add(selectPanel);

// initialize some variables for Map display
var legendPanel, currentImage, attri, yr;
var pixelInfo = ui.Label('');
var checkboxes;

// create selection options for years and attributes
function makeAttributeCheckboxes(year) {
  checkboxPanel.clear();
  checkboxes = [];
  yr = year;

  var attributes = Object.keys(assets[year]);
  attributes.forEach(function(attr) {   // loop through attributes
    var checkbox = ui.Checkbox({
      label: attr,
      value: attr === 'NEP (Net Ecoystem Productivity)',  // Default checked if NEP exists
      style: {fontSize: '14px', margin: '0 0 2px 0'},
      onChange: function(state) {
        if (state) {
          // Uncheck all others
          checkboxes.forEach(function(cb) {
            if (cb !== checkbox) cb.setValue(false, false);
          });
          updateMap(year, attr);
        } else {
          Map.layers().reset();
        }
      }
    });
    checkboxes.push(checkbox);
    checkboxPanel.add(checkbox);
  });
  // make the default checkbox selection NEP or the first attribute if NEP is missing
  if (checkboxes.length > 0) {
    var defaultCb = null;
    for (var i = 0; i < checkboxes.length; i++) {
      if (checkboxes[i].getLabel() === 'NEP (Net Ecoystem Productivity)') {
        defaultCb = checkboxes[i];
        break;
      }
    }
    if (!defaultCb && checkboxes.length > 0) {
      defaultCb = checkboxes[0];
    }
    if (defaultCb) {
      defaultCb.setValue(true, true);
      updateMap(year, defaultCb.getLabel());
    }
  }
}

// function to update entire map layer when a year/attribute is changed
function updateMap(year, attribute) {
  Map.layers().reset();
  var img = ee.Image(assets[year][attribute]);
  
  // Calculate min/max of attribute and round to the nearest 10
  var stats = img.reduceRegion({reducer: ee.Reducer.percentile([1, 99]), geometry: img.geometry(), scale: 30, maxPixels: 1e13});
  var minVal = ee.Number(stats.get('b1_p1')).round().divide(10).floor().multiply(10);
  var maxVal = ee.Number(stats.get('b1_p99')).round().divide(10).ceil().multiply(10);
  // Style points by selected attribute
  var visParams = {palette: ['FF0000', 'FFFF00', '00FFFF', '0000FF']};
  
  //re-draw map with new layer after selection
  minVal.evaluate(function(mini) {
    maxVal.evaluate(function(maxi) {
      Map.addLayer(img, {min: mini, max: maxi, palette: visParams.palette}, attribute);
    });
  });
  // remove and redraw legend each time a new selection is made
  if (legendPanel) Map.remove(legendPanel);
  legendPanel = makeLegend(attribute, visParams, minVal, maxVal);
  Map.add(legendPanel);
  currentImage = img;
  attri = attribute;
}

// if year is changed, regenerate the checkboxes (and updatemap)
yearSelect.onChange(function(year) {
  makeAttributeCheckboxes(year);
});

// set default year to first in order (1994)
makeAttributeCheckboxes(years[0]);

// function to extract attribute value for a particular year when pixel on map is clicked
Map.onClick(function(coords) {
  if (!currentImage) return;
  // extract coordinates and pixel value when clicked
  var point = ee.Geometry.Point(coords.lon, coords.lat);
  var value = currentImage.sample(point, 30).first().get('b1');
  
  // display value at the top
  value.evaluate(function(v) {
    if (pixelInfo) Map.remove(pixelInfo);
    var msg = attri + ' value at (' + coords.lon.toFixed(4) + ', ' + coords.lat.toFixed(4) + ') for ' + yr + ' = ' + v;
    pixelInfo.setValue(msg);
    Map.add(pixelInfo);
  });
});

// create legend with range of values passed in
function makeLegend(attribute, visParams, minVal, maxVal) {
  var legend = ui.Panel({
    style: {position: 'bottom-left', padding: '8px', backgroundColor: 'white'}
  });
  legend.add(ui.Label(attribute + ' Value Range', {fontWeight: 'bold', textAlign: 'center', stretch: 'horizontal'}));
  var colorBar = ui.Thumbnail({
    image: ee.Image.pixelLonLat().select(0),
    params: {bbox: [0, 0, 1, 0.1], dimensions: '300x20', format: 'png', min: 0, max: 1, palette: visParams.palette},
    style: {stretch: 'horizontal', margin: '8px 0'}
  });
  legend.add(colorBar);

  // include min and max labels of legend
  minVal.evaluate(function(min) {
    maxVal.evaluate(function(max) {
      legend.add(ui.Panel([ui.Label(min, {margin: '4px 8px', width: '60px'}), ui.Label('', {stretch: 'horizontal'}),
        ui.Label(max, {margin: '4px 8px', width: '60px', textAlign: 'right'})], ui.Panel.Layout.flow('horizontal')));
    });
  });

  return legend;
}

// display map to appropriate zoom level (7) of entire region
Map.centerObject(currentImage, 8);