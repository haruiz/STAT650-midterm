### Deployment Commands


```
gcloud init # if you haven't already
gcloud login # login to your account
gcloud projects list # list your projects
gcloud config set project [PROJECT_ID] # set your default project
gcloud config set compute/zone [ZONE] # set your default zone
gcloud auth configure-docker # configure docker to use gcloud

gcloud builds list # list your builds
gcloud builds submit --tag gcr.io/stat650-midterm-project/stat650-dashboard # build the docker image and push it to gcr.io
gcloud run deploy stat650-dashboard --image gcr.io/stat650-midterm-project/stat650-dashboard # deploy the image to Cloud Run
```