# Deployment in Azure

1 - Asegurate de tener instalado Azure CLI y haber iniciado sesión:

```bash
APP_ID=""
PASSWORD=""
TENANT_ID=""
SUBSCRIPTION_ID=""

az login --service-principal -u <APP_ID> -p <PASSWORD> --tenant <TENANT_ID>
az account list --output table
az account set --subscription "<SUBSCRIPTION_ID>"
```

2 - Define las siguientes variables en bash y valida si existen los recursos en Azure con los nombres asignados:

```bash
RG_NAME=""
ACR_NAME=""
ACR_IMAGE="mlops-project-equipo63"
ACR_IMAGE_TAG="v1"
APIM_NAME="mlops-project-equipo63"
WEBAPP_NAME="mlops-project-equipo63"
SERVICE_PLAN_NAME="mlops-project-equipo63"

az group show -n ${RG_NAME} -o table
az acr show -n ${ACR_NAME} -g ${RG_NAME} -o table
az acr repository show -n ${ACR_NAME} --repository ${ACR_IMAGE} -o table
az apim show -n ${APIM_NAME} -g ${RG_NAME} -o table
az webapp show -n ${WEBAPP_NAME} --resource-group ${RG_NAME} -o table
az appservice plan show -n ${SERVICE_PLAN_NAME} --resource-group ${RG_NAME} -o table
```

3 - Crea la infrastructura en Azure (ACR, PLAN, WEB APP) utilizando Terraform (30 mins):

```bash
cd terraform
terraform init
terraform plan
terraform apply -auto-approve
terraform output
```

4 - Crea la image con Docker de manera local (Dockerfile.model) y ejecuta un contenedor para validarla:

```bash
az acr login -n ${ACR_NAME}
docker buildx build -f Dockerfile.model --platform linux/amd64 -t ${ACR_IMAGE}:${ACR_IMAGE_TAG} .
docker run -p 8000:8000 ${ACR_IMAGE}:${ACR_IMAGE_TAG}
```

5 - Empuja la image al Azure Container Registry (ACR):

```bash
docker tag ${ACR_IMAGE}:${ACR_IMAGE_TAG} ${ACR_NAME}.azurecr.io/${ACR_IMAGE}:${ACR_IMAGE_TAG}
docker push ${ACR_NAME}.azurecr.io/${ACR_IMAGE}:${ACR_IMAGE_TAG}
```

6 - Deploya la imagen en el Azure Web App:

```bash
az webapp config container set \
  -n "${WEBAPP_NAME}" \
  -g "${RG_NAME}" \
  --container-image-name "${ACR_NAME}.azurecr.io/${ACR_IMAGE}:${ACR_IMAGE_TAG}" \
  --container-registry-url "https://${ACR_NAME}.azurecr.io"
```

7 - Prueba la API deployada en Azure utilizando los payloads de la parte de reproducibilidad. Se han cargado unos screenshoot en *docs/ss/*. Final URL: https://mlops-project-equipo63.azurewebsites.net (disabled).
