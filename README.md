# Small-LLM
Tough , but interesting .
Main issue - Dataset avail , but all scattered . Need to combine all of that dataset.

| Dataset Type                     | Use In Tool                                     | Sources                                                                                                                                                        |
| -------------------------------- | ----------------------------------------------- | -------------------------------------------------------------------------------------------------------------------------------------------------------------- |
| Crop recommendation data         | `crop_selector`, `soil_rule_checker`            | [Kaggle Crop Recommendation Dataset](https://www.kaggle.com/datasets/atharvaingle/crop-recommendation-dataset)                                                 |
| Crop disease images              | `disease_symptom_matcher` / vision add-on later | [PlantVillage GitHub](https://github.com/spMohanty/plantvillage-dataset), [TensorFlow PlantVillage](https://www.tensorflow.org/datasets/catalog/plant_village) |
| Indian agriculture district data | yield, rainfall, irrigation, crop patterns      | [ICRISAT District Level Data](https://data.icrisat.org/dld/)                                                                                                   |
| Crop yield + weather             | `weather_risk_tool`, `yield_risk_tool`          | [Indian Historical Crop Yield and Weather Data](https://www.kaggle.com/datasets/zoya77/indian-historical-crop-yield-and-weather-data)                          |
| Pest/disease field data          | more realistic disease/pest examples            | [CCMT crop pest and disease dataset paper](https://pmc.ncbi.nlm.nih.gov/articles/PMC10285554/)                                                                 |
We can build for some specific types of crops initially . Eg- Wheat ,Rice,Tomato . Why these ? These crops cover very different behaviour .

| Wheat | Staple crops and needs stage based irrigation and fert.
| Rice | heavily dependent on water / rain
| Tomato | disease/pest-heavy crop, good for symptom matching

Our model does not need to know much about agriculture but it should just know tool calling , tool combining , safe answering etc.

TOOLS = {
    "extract_farm_context": "Extract crop, age, location, symptoms, soil, weather",
    "crop_stage_tool": "Find crop growth stage from days after sowing",
    "weather_risk_tool": "Detect drought, heavy rain, heat, humidity risk",
    "soil_suitability_tool": "Check NPK, pH, moisture suitability",
    "symptom_matcher": "Match symptoms to likely disease/nutrient/stress issue",
    "safe_action_checker": "Prevent unsafe pesticide/fertilizer advice"
}


