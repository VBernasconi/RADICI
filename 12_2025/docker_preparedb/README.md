docker build -t docker_preparedb -f docker_preparedb .
docker run -v /home/valentine.bernasconi/RADICI:/app/data docker_preparedb
docker run --network host -v /home/valentine.bernasconi/RADICI:/app/data docker_preparedb



docker run -it \
  --network host \
  -v /home/valentine.bernasconi/RADICI:/app/data \
  -v /home/valentine.bernasconi/RADICI/downloaded_images:/app/data/downloaded_images \
  --name docker_preparedb \
  docker_preparedb:latest

