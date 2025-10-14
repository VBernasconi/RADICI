![RADICI LOGO.](RADICI_LOGO.png)
# RADICI
**R**ealizzazione di una infrastruttura di **A**ggregazione e **D**igitalizzazione di patrimoni per favorire l’**I**nterazione **c**on il settore delle **I**ndustrie culturali e creative.

Informazioni generali sul progetto [qui](https://www.unibo.it/it/ricerca/progetti-e-iniziative/pr-fesr-emilia-romagna-2021-2027/1223/20430/20509).


## Project structure

```
.                  
├── backend
│   ├── app.py        
│   ├── image_search.py    
│   ├── requirements.txt
│   ├── redis_export.csv
│   └── downloaded_images
│       ├── img_00        
│       └── img_01    
│       └── ...                   
├── frontend
│   ├── index.html
│   ├── map.js
│   ├── grid.js     
│   ├── redis_export.geojson
│   └── img
│       └── icons        
│           └── documents-icon.png    
│           └── ...                      
└── README.md
```

## Backend
A Flask application running on a server holding the images of the different archives and corresponding .csv file holding all the information
Create a virtual environment and run 
```
pip install requirements
```
Then run the following command
```
python app.py
```
The first time, it will create a Fess index and store it locally as index.fess.

## Frontend
In index.html, adapt the url to corresponding server.
Setup a local server and run the following command
```
python3 -m http.server 8000
```
Open a browser and type `http://localhost:8000/`
## Sources
The project is based on three archives
- [Lodovico](https://lodovico.medialibrary.it/)
- [Classense](https://www.cdc.classense.ra.it/s/Classense/page/home)
- [Il corago](http://www.ilcorago.org/benedetti/benedetti.asp)
