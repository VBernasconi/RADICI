## Docker folder to prepare the database stored in Redis and export .csv and .geojson files

- Run the following commands to build and run the docker
  
```
docker build -t docker_preparedb -f docker_preparedb .
```
```
docker run -v /home/valentine/RADICI:/app/data docker_preparedb
```
```
docker run --network host -v /home/valentine/RADICI:/app/data docker_preparedb
```
