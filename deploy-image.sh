HOST=marcelci

docker build -t urban-mobility-project .
docker tag urban-mobility-project $HOST/urban-mobility-project:latest
docker push $HOST/urban-mobility-project:latest